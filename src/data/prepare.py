"""Build one task mixture's training stream, development and test Parquet files."""

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from data import physics_instructions_code, sciknoweval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"refusing to replace existing inputs: {output}")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    builder = physics_instructions_code if config["mixture"] == "physics-instructions-code" else sciknoweval
    stream, development, test = builder.build(config, tokenizer)

    train = pd.DataFrame(stream).rename(columns={"messages": "prompt"})
    files = {
        # Ordered training visits; smaller pools repeat questions across updates.
        "train_verl.parquet": train,
        # One row per distinct training question, for teacher-cache generation.
        "reference_train_verl.parquet": train.drop_duplicates("example_id"),
        "development_verl.parquet": pd.DataFrame(development).rename(columns={"messages": "prompt"}),
        "test.parquet": pd.DataFrame(test),
    }
    output.mkdir(parents=True)
    for name, frame in files.items():
        frame.to_parquet(output / name, index=False)
        counts = Counter(frame["data_source"])
        print(f"{name}: " + ", ".join(f"{task}: {count}" for task, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
