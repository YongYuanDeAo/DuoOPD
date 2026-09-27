"""Generate one independently verified teacher answer for each training question."""

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.evaluate import (
    load_items, read_predictions, resume_directory, rollout,
    validate_completed_predictions, write_json, write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["generation"]["n_samples"] != 1:
        raise ValueError("Reference preparation requires exactly one teacher draw per question")
    items = load_items(config["train_file"])
    output = Path(config["output"])
    # Interrupted generation resumes from the completed groups in predictions.jsonl.
    predictions_path = resume_directory(output.parent, config)
    completed = read_predictions(predictions_path)
    validate_completed_predictions(items, completed, 1)
    if len(completed) != len(items):
        rollout(config, items, completed, predictions_path)

    # Check actual reference lengths before training; no silent reference clipping.
    from transformers import AutoTokenizer
    from methods.duoopd.teacher import reference_prompt

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"])
    counts, correct, max_context = Counter(), Counter(), 0
    for item in items:
        sample = completed[item["example_id"]]["samples"][0]
        success = sample["score"]["score"] == 1
        template_kwargs = {**config["model"].get("chat_template_kwargs", {}),
                           **item["chat_template_kwargs"]}
        if success:
            context = reference_prompt(tokenizer, item["messages"], sample["response"], template_kwargs)
        else:
            context = tokenizer.apply_chat_template(
                item["messages"], tokenize=True, add_generation_prompt=True, **template_kwargs,
            )
        if len(context) > config["teacher_max_prompt_length"]:
            raise ValueError(f"Reference exceeds teacher prompt budget: {item['example_id']}")
        counts[item["task"]] += 1
        correct[item["task"]] += int(success)
        max_context = max(max_context, len(context))
    write_jsonl(output, [completed[item["example_id"]] for item in items])
    summary = {"questions": len(items), "correct": sum(correct.values()),
               "max_teacher_prompt_tokens": max_context,
               "by_task": {task: {"questions": count, "correct": correct[task]}
                           for task, count in counts.items()}}
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
