"""Read the fixed test, generate responses, verify them, and summarize."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from data import plain_nested
from .verifier import VERIFICATION_FAILURE_REASONS, verify_answer


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_items(path) -> list[dict]:
    """Read evaluator (`messages`) or verl (`prompt`) Parquet rows as generation items."""
    frame = pd.read_parquet(path)
    column = "messages" if "messages" in frame else "prompt"
    return [
        {
            "example_id": str(row["example_id"]),
            "task": str(row["data_source"]),
            "messages": plain_nested(row[column]),
            "chat_template_kwargs": plain_nested(row["chat_template_kwargs"]),
            "ground_truth": plain_nested(row["reward_model"])["ground_truth"],
        }
        for _, row in frame.iterrows()
    ]


def read_predictions(path: Path) -> dict[str, dict]:
    """Read completed groups; a partial last line from an interrupted run is dropped."""
    if not path.exists():
        return {}
    groups = {}
    with path.open("rb+") as handle:
        while line := handle.readline():
            if not line.endswith(b"\n"):
                handle.seek(-len(line), os.SEEK_CUR)
                handle.truncate()
                break
            group = json.loads(line)
            example_id = str(group["example_id"])
            if example_id in groups:
                raise ValueError(f"predictions contain duplicate example_id: {example_id}")
            groups[example_id] = group
    return groups


def validate_completed_predictions(
    items: list[dict], completed: dict[str, dict], n_samples: int
) -> None:
    """Reject prediction records that do not exactly describe the active inputs."""

    items_by_id = {item["example_id"]: item for item in items}
    if len(items_by_id) != len(items):
        raise ValueError("inputs contain duplicate example IDs")
    unexpected = sorted(set(completed) - set(items_by_id))
    if unexpected:
        raise ValueError(f"predictions contain unexpected example IDs: {unexpected[:3]}")
    for example_id, group in completed.items():
        item = items_by_id[example_id]
        for field in ("example_id", "task", "ground_truth"):
            if group.get(field) != item[field]:
                raise ValueError(f"prediction {example_id} has inconsistent {field}")
        sample_ids = [int(sample["sample_id"]) for sample in group["samples"]]
        if sorted(sample_ids) != list(range(n_samples)):
            raise ValueError(f"prediction {example_id} has invalid sample IDs")


def append_predictions(path: Path, groups: list[dict]) -> None:
    if not groups:
        return
    with path.open("a", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resume_directory(run_dir: Path, config: dict) -> Path:
    """Return the predictions file, refusing to resume under a different configuration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    config_path = run_dir / "config.json"
    has_predictions = predictions_path.exists() and predictions_path.stat().st_size
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config and has_predictions:
            raise ValueError(f"{config_path} differs from the active configuration")
    elif has_predictions:
        raise ValueError("predictions exist without a run config")
    write_json(config_path, config)
    return predictions_path


def rollout(config: dict, items: list[dict], completed: dict[str, dict], predictions_path: Path) -> None:
    """Generate with the native verl standalone vLLM rollout service."""
    asyncio.run(_rollout(config, items, completed, predictions_path))


async def _rollout(config, items, completed, predictions_path):
    import aiohttp
    import ray
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    model = config["model"]
    generation = config["generation"]
    n_samples = int(generation["n_samples"])
    batch_size = int(generation["batch_size"])
    model_config = HFModelConfig(path=model["path"])
    tokenizer = model_config.tokenizer
    max_prompt = int(generation["max_prompt_tokens"])
    max_response = int(generation["max_new_tokens"])
    tensor_parallel = int(model["tensor_parallel_size"])
    rollout_config = RolloutConfig(
        name="vllm", mode="async", load_format="auto",
        tensor_model_parallel_size=tensor_parallel,
        dtype=model.get("dtype", "bfloat16"),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
        prompt_length=max_prompt, response_length=max_response,
        max_model_len=max_prompt + max_response,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]), top_k=int(generation["top_k"]),
        seed=int(config["seed"]),
        enable_prefix_caching=False, enable_sleep_mode=False,
        # The native server omits false CLI flags; vLLM otherwise enables this by default.
        engine_kwargs={"vllm": {"no_enable_prefix_caching": True}},
    )
    cpus = len(os.sched_getaffinity(0))
    # Native standalone pools reserve three CPUs per GPU plus one for the server.
    if cpus < 3 * tensor_parallel + 1:
        raise ValueError("verl standalone evaluation needs at least 3 * TP + 1 CPUs")
    ray.init(
        address="local", include_dashboard=False, num_cpus=cpus, num_gpus=tensor_parallel,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(sys.path)}},
    )
    try:
        replica = vLLMReplica(0, rollout_config, model_config, gpus_per_node=tensor_parallel)
        await replica.init_standalone()
        # Executable verification can block this event loop beyond the server's
        # keep-alive timeout, leaving a stale pooled connection for the next batch.
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True),
            timeout=aiohttp.ClientTimeout(total=None),
        ) as session:
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                if all(item["example_id"] in completed for item in batch):
                    continue
                prompts = [tokenizer.apply_chat_template(
                    item["messages"], tokenize=True, add_generation_prompt=True,
                    **{**model.get("chat_template_kwargs", {}), **item["chat_template_kwargs"]},
                ) for item in batch]
                for item, prompt in zip(batch, prompts, strict=True):
                    if len(prompt) > max_prompt:
                        raise ValueError(f"Prompt exceeds evaluation budget: {item['example_id']}")
                # Token prompts avoid a second chat-template pass. Native completions
                # retain n samples, actual token IDs, and stop/length separately.
                request = {
                    "model": model_config.local_path, "prompt": prompts,
                    "add_special_tokens": False, "return_token_ids": True,
                    "max_tokens": max_response, "n": n_samples,
                    "temperature": float(generation["temperature"]),
                    "top_p": float(generation["top_p"]), "top_k": int(generation["top_k"]),
                    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "seed": int(config["seed"]) + start,
                    "skip_special_tokens": True, "ignore_eos": False,
                }
                async with session.post(
                    f"http://{replica.server_address}/v1/completions", json=request,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                choices = sorted(result["choices"], key=lambda choice: int(choice["index"]))
                if [int(c["index"]) for c in choices] != list(range(len(batch) * n_samples)):
                    raise ValueError("Rollout returned incomplete or duplicate sample indices")
                generated = []
                for index, (item, prompt) in enumerate(zip(batch, prompts, strict=True)):
                    if item["example_id"] in completed:
                        continue
                    samples = []
                    for sample_id, choice in enumerate(choices[index*n_samples:(index+1)*n_samples]):
                        if choice["prompt_token_ids"] != prompt:
                            raise ValueError("Rollout changed the rendered prompt tokens")
                        if choice["finish_reason"] not in ("stop", "length"):
                            raise ValueError(f"Incomplete rollout: {choice['finish_reason']}")
                        tokens = choice["token_ids"]
                        text = choice["text"]
                        samples.append({
                            "sample_id": sample_id, "response": text,
                            "token_ids": tokens, "generated_tokens": len(tokens),
                            "finish_reason": choice["finish_reason"],
                            "score": verify_answer(item["task"], text, item["ground_truth"]),
                        })
                    group = {
                        "example_id": item["example_id"], "task": item["task"],
                        "ground_truth": item["ground_truth"],
                        "prompt_tokens": len(prompt), "prompt_token_ids": prompt, "samples": samples,
                    }
                    generated.append(group)
                    completed[item["example_id"]] = group
                append_predictions(predictions_path, generated)
    finally:
        ray.shutdown()


def prompt_metrics(group: dict, n_samples: int) -> dict[str, float]:
    samples = group["samples"]
    if len(samples) != n_samples:
        raise ValueError(f"prompt group contains {len(samples)} samples; expected {n_samples}")
    scores = [float(sample["score"]["score"]) for sample in samples]
    failure_counts = Counter(sample["score"]["failure_reason"] for sample in samples)
    unexpected_failures = set(failure_counts) - {None, *VERIFICATION_FAILURE_REASONS}
    if unexpected_failures:
        raise ValueError(f"unexpected verifier failure reasons: {sorted(unexpected_failures)}")
    if sum(scores) != failure_counts[None]:
        raise ValueError("verifier correctness and failure reasons disagree")
    votes = Counter(
        sample["score"]["normalized_answer"]
        for sample in samples
        if sample["score"]["normalized_answer"] is not None
    )
    winner, count = votes.most_common(1)[0] if votes else (None, 0)
    majority_score = 0.0
    if count > n_samples / 2:
        majority_score = next(
            float(sample["score"]["score"])
            for sample in samples
            if sample["score"]["normalized_answer"] == winner
        )
    metrics = {
        f"avg@{n_samples}": sum(scores) / n_samples,
        f"any@{n_samples}": float(any(scores)),
        "length_rate": sum(sample["finish_reason"] == "length" for sample in samples) / n_samples,
    }
    if group["task"] not in ("mbpp", "ifeval"):
        metrics[f"maj@{n_samples}"] = majority_score
    metrics.update({
        f"{reason}_rate": failure_counts[reason] / n_samples
        for reason in VERIFICATION_FAILURE_REASONS
    })
    return metrics


def aggregate(groups: list[dict], n_samples: int) -> dict[str, float | int]:
    points = [prompt_metrics(group, n_samples) for group in groups]
    result: dict[str, float | int] = {
        key: sum(point[key] for point in points) / len(points)
        for key in points[0] if all(key in point for point in points)
    }
    result["prompts"] = len(points)
    if all("ifeval" in sample["score"] for group in groups for sample in group["samples"]):
        official = [sample["score"]["ifeval"] for group in groups for sample in group["samples"]]
        for mode in ("strict", "loose"):
            result[f"prompt_{mode}_avg@{n_samples}"] = sum(
                item[f"prompt_{mode}"] for item in official) / len(official)
            instructions = [passed for item in official for passed in item[f"instruction_{mode}"]]
            result[f"instruction_{mode}_avg@{n_samples}"] = sum(instructions) / len(instructions)
    return result


def summarize(groups: list[dict], n_samples: int) -> dict:
    by_task = {
        task: aggregate([group for group in groups if group["task"] == task], n_samples)
        for task in sorted({group["task"] for group in groups})
    }
    overall = aggregate(groups, n_samples)
    average_key = f"avg@{n_samples}"
    overall[f"macro_task_avg@{n_samples}"] = sum(
        float(stats[average_key]) for stats in by_task.values()) / len(by_task)
    overall[f"worst_task_avg@{n_samples}"] = min(
        float(stats[average_key]) for stats in by_task.values())
    return {"n_samples": n_samples, "prompt_groups": len(groups),
            "metrics": {"overall": overall, "by_task": by_task}}


def evaluate(config: dict, run_dir: Path) -> dict:
    items = load_items(config["test_path"])
    n_samples = int(config["generation"]["n_samples"])
    predictions_path = resume_directory(run_dir, config)
    completed = read_predictions(predictions_path)
    validate_completed_predictions(items, completed, n_samples)
    if any(item["example_id"] not in completed for item in items):
        rollout(config, items, completed, predictions_path)
    summary = summarize([completed[item["example_id"]] for item in items], n_samples)
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", help="override model.path, e.g. a trained checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.model_path:
        config["model"]["path"] = args.model_path
    print(json.dumps(evaluate(config, args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
