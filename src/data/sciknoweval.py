"""Custom partitions of the SciKnowEval V2 multiple-choice tasks.

SciKnowEval V2 publishes one collection (named test); every split here is drawn from it.
Questions are grouped by normalized text; groups with conflicting answers or tasks are
removed and one question is kept from each remaining group.
"""

from collections import defaultdict
import json
import random
import unicodedata

from data.stream import task_stream
from evaluation.verifier import verify_answer


INSTRUCTION = (
    "Explain the key reasoning briefly, then give only the final option letter "
    r"(A, B, C, or D) in \boxed{...}."
)
# Biology / chemistry / physics: L1 literature questions (source domain, source task).
LITERQA = {
    "biology_literqa": ("Biology", "literature_multi_choice_question"),
    "chemistry_literqa": ("Chemistry", "literature_multi_choice_question"),
    "physics_literqa": ("Physics", "physics_literature_QA"),
}
# Materials knowledge / chemistry reading comprehension / physics calculation
# (source domain, level, subtask).
SCIENCE = {
    "material_literqa": ("Material", "L1", "material_literature_QA", "scientific_knowledge"),
    "chemistry_understanding": ("Chemistry", "L2", "detailed_understanding", "scientific_comprehension"),
    "physics_calculation": ("Physics", "L3", "general_physics_calculation", "scientific_reasoning"),
}


def question_key(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def prompt_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))


def choice_row(task, ability, raw, line_number, tokenizer, max_prompt_tokens):
    """Rewrite one source question with the shared answer instruction; None if too long."""
    choices = raw["choices"]
    options = "\n".join(f"{label}. {text.strip()}"
                        for label, text in zip(choices["label"], choices["text"], strict=True))
    messages = [{"role": "user", "content": f"{raw['question'].strip()}\n\n{options}\n\n{INSTRUCTION}"}]
    length = prompt_tokens(tokenizer, messages)
    if length > max_prompt_tokens:
        return None
    details = raw["details"]
    return {
        "example_id": f"{task}:v2:{line_number}",
        "data_source": task,
        "ability": ability,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "reward_model": {"style": "rule", "ground_truth": {"task": task, "target": raw["answerKey"]}},
        "extra_info": {
            "source_split": "test", "source_line": line_number,
            "source_domain": raw["domain"], "source_level": details["level"],
            "source_task": details["task"], "source_subtask": details["subtask"],
            "source": details.get("source"), "prompt_tokens": length,
        },
    }


def deduplicate(candidates):
    """Keep the first question of each normalized-text group with one answer and task."""
    groups = defaultdict(list)
    for raw, row in candidates:
        choices = raw["choices"]
        answer = question_key(choices["text"][choices["label"].index(raw["answerKey"])])
        groups[question_key(raw["question"].strip())].append((answer, row))
    pools = defaultdict(list)
    for entries in groups.values():
        if len({(answer, row["data_source"]) for answer, row in entries}) == 1:
            pools[entries[0][1]["data_source"]].append(entries[0][1])
    return pools


def read_source(path):
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            yield line_number, json.loads(line)


def literqa_splits(config, tokenizer):
    """Biology, chemistry and physics L1: per-task training pools, development and test rows."""
    candidates = []
    for line_number, raw in read_source(config["source"]):
        task = next((task for task, (domain, source_task) in LITERQA.items()
                     if raw["domain"] == domain and raw["details"]["level"] == "L1"
                     and raw["details"]["task"] == source_task), None)
        if task is None:
            continue
        if raw["type"] != "mcq-4-choices" or raw["choices"]["label"] != list("ABCD"):
            raise ValueError(f"invalid multiple-choice question at source line {line_number}")
        row = choice_row(task, "scientific_knowledge", raw, line_number, tokenizer,
                         config["max_prompt_tokens"])
        if row is not None:
            candidates.append((raw, row))
    pools = deduplicate(candidates)

    train, development, test = {}, [], []
    for index, task in enumerate(LITERQA):
        rows = pools[task]
        random.Random(config["seed"] + index).shuffle(rows)
        sizes = {split: config[f"{split}_per_task"] for split in ("development", "test", "train")}
        if len(rows) < sum(sizes.values()):
            raise ValueError(f"{task}: {len(rows)} eligible questions, need {sum(sizes.values())}")
        offset, selected = 0, {}
        for split, size in sizes.items():
            selected[split] = rows[offset:offset + size]
            offset += size
            for row in selected[split]:
                row["extra_info"]["protocol_split"] = split
        train[task] = selected["train"]
        development.extend(selected["development"])
        test.extend(selected["test"])
    return train, development, test


def science_splits(config, tokenizer):
    """Materials knowledge, chemistry reading and physics calculation.

    Reading questions sharing a passage stay in one split. V2 has no passage IDs, so the
    normalized first 150 characters of the question define the passage group.
    """
    candidates = []
    for line_number, raw in read_source(config["source"]):
        details = raw["details"]
        task = next((task for task, (domain, level, subtask, _) in SCIENCE.items()
                     if (raw["domain"], details["level"], details["subtask"]) == (domain, level, subtask)),
                    None)
        if task is None:
            continue
        choices = raw["choices"]
        if raw["type"] != "mcq-4-choices" or choices["label"] != list("ABCD"):
            raise ValueError(f"invalid multiple-choice question at source line {line_number}")
        if len(choices["text"]) != 4 or any(not text.strip() for text in choices["text"]):
            continue
        row = choice_row(task, SCIENCE[task][3], raw, line_number, tokenizer, config["max_prompt_tokens"])
        if row is None:
            continue
        ground_truth = row["reward_model"]["ground_truth"]
        if verify_answer(task, rf"\boxed{{{ground_truth['target']}}}", ground_truth)["score"] != 1:
            raise ValueError(f"invalid gold option at source line {line_number}")
        key = question_key(raw["question"].strip())
        row["extra_info"]["source_group"] = key[:150] if task == "chemistry_understanding" else key
        candidates.append((raw, row))
    pools = deduplicate(candidates)

    train, development, test = {}, [], []
    for index, task in enumerate(SCIENCE):
        groups = defaultdict(list)
        for row in pools[task]:
            groups[row["extra_info"]["source_group"]].append(row)
        groups = list(groups.values())
        random.Random(config["seed"] + index).shuffle(groups)
        splits = {"development": [], "test": [], "train": []}
        for group in groups:
            split = next((split for split in ("development", "test")
                          if len(splits[split]) + len(group) <= config[f"{split}_per_task"]), "train")
            for row in group:
                row["extra_info"]["protocol_split"] = split
            splits[split].extend(group)
        for split in ("development", "test"):
            if len(splits[split]) != config[f"{split}_per_task"]:
                raise ValueError(f"cannot form {task}/{split} without splitting a passage group")
        train[task] = splits["train"]
        development.extend(splits["development"])
        test.extend(splits["test"])
    return train, development, test


def build(config, tokenizer):
    splits = literqa_splits if config["mixture"] == "biology-chemistry-physics" else science_splits
    train, development, test = splits(config, tokenizer)
    stream = task_stream(train, config["questions_per_task_per_update"], config["updates"], config["seed"])
    return stream, development, test
