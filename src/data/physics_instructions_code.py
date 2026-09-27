"""Physics L1, instruction following and MBPP code generation.

- Physics reuses the biology-chemistry-physics partition.
- Instruction training uses RLVR-IFeval prompts from eight constraint types; the test is
  all 541 official IFEval prompts.
- MBPP keeps the official split assignments (train 601-974, validation 511-600, test
  11-510); descriptions duplicated across splits are removed, the first is kept within a
  split, and every problem's canonical solution must pass its assertions.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re

import pandas as pd

from data import plain_nested
from data.sciknoweval import literqa_splits, prompt_tokens, question_key
from data.stream import TASK_SEED_OFFSETS, visit_order
from evaluation.verifier import if_check, verify_answer


CODE_INSTRUCTION = (
    "Write a self-contained Python solution. Use the function names and interfaces "
    "shown in the example, including any required helper classes. Return the solution in one Python code block, "
    "without explanations or test calls."
)


def mbpp_splits(config, tokenizer):
    """Eligible MBPP rows by official split, in file order."""
    with open(config["mbpp"]) as handle:
        raw = [json.loads(line) for line in handle]
    groups = defaultdict(list)
    for item in raw:
        task_id = item["task_id"]
        split = ("train" if 601 <= task_id <= 974 else "development" if 511 <= task_id <= 600
                 else "test" if 11 <= task_id <= 510 else None)
        if split is not None:
            groups[question_key(item["text"])].append((split, item))
    candidates = []
    for group in groups.values():
        if len({split for split, _ in group}) != 1:
            continue
        split, item = group[0]
        # The setup and first assertion show the interface; all assertions score the answer.
        messages = [{"role": "user", "content": (
            item["text"].strip() + "\n\nExample:\n" + item["test_setup_code"] + "\n"
            + item["test_list"][0] + "\n\n" + CODE_INSTRUCTION)}]
        length = prompt_tokens(tokenizer, messages)
        if length > config["max_prompt_tokens"]:
            continue
        candidates.append((split, item["code"], {
            "example_id": f"mbpp:{item['task_id']}", "data_source": "mbpp", "ability": "code",
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False},
            "reward_model": {"style": "rule", "ground_truth": {
                "task": "mbpp", "target": str(item["task_id"]), "test_list": item["test_list"],
                "test_setup_code": item["test_setup_code"]}},
            "extra_info": {"source_split": split, "protocol_split": split,
                           "source_task_id": item["task_id"], "prompt_tokens": length},
        }))

    def canonical_passes(candidate):
        _, code, row = candidate
        return verify_answer("mbpp", code, row["reward_model"]["ground_truth"])["score"] == 1

    splits = defaultdict(list)
    with ThreadPoolExecutor(max_workers=config["mbpp_verification_workers"]) as executor:
        for candidate, passed in zip(candidates, executor.map(canonical_passes, candidates)):
            if passed:
                splits[candidate[0]].append(candidate[2])
    return splits


def if_base_prompt(raw, parameters):
    """Recover the RLVR-IFeval question without its appended constraint sentence."""
    values = {key.replace("_", " "): value for key, value in parameters.items()}
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = ", ".join(value)
    for index, word in enumerate(parameters.get("keyword_list", []), 1):
        values[f"keyword{index}"] = word
    template = raw["constraint"].replace("at least / around / at most", parameters.get("quantifier", ""))
    instruction = template.format_map(values)
    prompt = raw["messages"][0]["content"].strip()
    if prompt.startswith(instruction):
        return prompt[len(instruction):].strip()
    if prompt.endswith(instruction):
        return prompt[:-len(instruction)].strip()
    raise ValueError("Cannot recover the released IF base prompt")


def if_candidates(config):
    """Yield (source index, row, parameters, base question) for the selected constraint types."""
    frame = pd.read_parquet(config["rlvr_ifeval"])
    for index in random.Random(config["seed"]).sample(range(len(frame)), len(frame)):
        raw = plain_nested(frame.iloc[index].to_dict())
        parameters = {key: value for key, value in json.loads(raw["ground_truth"]).items()
                      if value is not None}
        name = parameters["func_name"]
        if name not in config["if_constraints"]:
            continue
        bounds = config["if_constraints"][name]
        if bounds and not bounds[0] <= parameters["N"] <= bounds[1]:
            continue
        yield index, raw, parameters, if_base_prompt(raw, parameters)


def if_row(index, raw, parameters, base, split, length):
    target = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
    if_check(target)
    return {
        "example_id": f"ifeval:train:{index}", "data_source": "ifeval", "ability": "instruction_following",
        "messages": raw["messages"], "chat_template_kwargs": {"enable_thinking": False},
        "reward_model": {"style": "rule", "ground_truth": {"task": "ifeval", "target": target}},
        "extra_info": {"source_split": "train", "protocol_split": split, "source_index": index,
                       "constraint_group": parameters["func_name"], "base_question": base,
                       "prompt_tokens": length},
    }


def if_splits(config, tokenizer):
    """Development questions, then training pools whose bases avoid development and held-out bases."""
    choice = re.compile(config["choice_pattern"], re.I | re.M)
    per_constraint = config["if_development_per_constraint"]
    development, development_keys, counts = [], set(), Counter()
    for index, raw, parameters, base in if_candidates(config):
        name = parameters["func_name"]
        key = question_key(base)
        if (counts[name] == per_constraint or choice.search(base)
                or not re.search(config["open_generation_pattern"], base, re.I) or key in development_keys):
            continue
        length = prompt_tokens(tokenizer, raw["messages"])
        if length > config["max_prompt_tokens"]:
            continue
        development.append(if_row(index, raw, parameters, base, "development", length))
        development_keys.add(key)
        counts[name] += 1
        if all(counts[name] == per_constraint for name in config["if_constraints"]):
            break

    pools, seen = defaultdict(list), defaultdict(set)
    label = re.compile(config["label_pattern"], re.I | re.M)
    for index, raw, parameters, base in if_candidates(config):
        name = parameters["func_name"]
        key = question_key(base)
        if choice.search(base) or label.search(base) or key in development_keys or key in seen[name]:
            continue
        length = prompt_tokens(tokenizer, raw["messages"])
        if length > config["max_prompt_tokens"]:
            continue
        pools[name].append(if_row(index, raw, parameters, base, "train", length))
        seen[name].add(key)

    # Hold out distinct bases, rarest constraint first; every constraint variant of a
    # held-out base is excluded from training.
    heldout_keys = set()
    for name in sorted(config["if_constraints"], key=lambda name: len(pools[name])):
        taken = 0
        for row in pools[name]:
            key = question_key(row["extra_info"]["base_question"])
            if key not in heldout_keys and taken < config["if_heldout_per_constraint"]:
                heldout_keys.add(key)
                taken += 1
        if taken != config["if_heldout_per_constraint"]:
            raise ValueError(f"insufficient held-out {name} questions")
    train = {name: [row for row in pools[name]
                    if question_key(row["extra_info"]["base_question"]) not in heldout_keys
                    ][:config["if_train_per_constraint"]]
             for name in config["if_constraints"]}
    return development, train


def if_stream(pools, per_task, updates, seed):
    """Per update, the same number of questions from each constraint; small pools cycle."""
    per_constraint = per_task // len(pools)
    updates_rows = [[] for _ in range(updates)]
    for group_index, pool in enumerate(pools.values()):
        rng = random.Random(seed + group_index)
        indices, remaining = list(range(len(pool))), []
        for rows in updates_rows:
            chosen = []
            for _ in range(per_constraint):
                if not remaining:
                    remaining = indices.copy()
                    rng.shuffle(remaining)
                    remaining.sort(key=lambda index: index in chosen)
                chosen.append(remaining.pop(0))
            rows.extend(pool[index] for index in chosen)
    update_rng = random.Random(seed)
    for rows in updates_rows:
        update_rng.shuffle(rows)
    return [row for rows in updates_rows for row in rows]


def official_ifeval(config, tokenizer, training_rows):
    with open(config["ifeval"]) as handle:
        official = [json.loads(line) for line in handle]
    prompts = {question_key(item["prompt"]) for item in official}
    for row in training_rows:
        base = question_key(row["extra_info"]["base_question"])
        if base in prompts or question_key(row["messages"][0]["content"]) in prompts or (
                len(base) >= 64 and any(base in prompt for prompt in prompts)):
            raise ValueError(f"training question {row['example_id']} overlaps official IFEval")
    rows = []
    for item in official:
        messages = [{"role": "user", "content": item["prompt"]}]
        length = prompt_tokens(tokenizer, messages)
        if length > config["max_prompt_tokens"]:
            raise ValueError(f"IFEval prompt {item['key']} exceeds the prompt budget")
        rows.append({
            "example_id": f"ifeval:official:{item['key']}", "data_source": "ifeval",
            "ability": "instruction_following", "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "reward_model": {"style": "rule", "ground_truth": {
                "task": "ifeval", "target": json.dumps(item, ensure_ascii=False, sort_keys=True)}},
            "extra_info": {"source_split": "official_evaluation", "protocol_split": "test",
                           "source_index": item["key"], "prompt_tokens": length},
        })
    return rows


def build(config, tokenizer):
    physics_config = json.loads(open(config["physics"]).read())
    physics_train, physics_development, physics_test = literqa_splits(physics_config, tokenizer)
    physics = physics_train["physics_literqa"]
    physics_development = [row for row in physics_development if row["data_source"] == "physics_literqa"]
    physics_test = [row for row in physics_test if row["data_source"] == "physics_literqa"]

    mbpp = mbpp_splits(config, tokenizer)
    for index, split in enumerate(("train", "development")):
        random.Random(config["seed"] + index).shuffle(mbpp[split])
    if_development, if_pools = if_splits(config, tokenizer)

    per_task, updates, seed = config["questions_per_task_per_update"], config["updates"], config["seed"]
    sources = {}
    for task, pool in (("mbpp", mbpp["train"]), ("physics_literqa", physics)):
        visits = visit_order(len(pool), per_task, updates, random.Random(seed + TASK_SEED_OFFSETS[task]))
        sources[task] = iter([pool[index] for chosen in visits for index in chosen])
    instructions = if_stream(if_pools, per_task, updates, seed)
    sources["ifeval"] = iter(instructions)
    schedule_rng = random.Random(seed)
    stream = []
    for _ in range(updates):
        schedule = [task for task in ("ifeval", "mbpp", "physics_literqa") for _ in range(per_task)]
        schedule_rng.shuffle(schedule)
        stream.extend(next(sources[task]) for task in schedule)

    development = physics_development + mbpp["development"] + if_development
    test = (physics_test + official_ifeval(config, tokenizer, instructions)
            + sorted(mbpp["test"], key=lambda row: row["extra_info"]["source_task_id"]))
    return stream, development, test
