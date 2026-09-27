# Experimental inputs

Parquet rows preserve the exact IDs, prompts, targets, and order used by training
or evaluation.

- `train_verl.parquet`: 60 updates of 48 questions, 16 per task. Repeated IDs are
  intentional visits to smaller training pools.
- `reference_train_verl.parquet`: one row per distinct training question, used for
  teacher-cache generation.
- `development_verl.parquet`: development questions in the training loader format.
- `test.parquet`: test questions in the evaluator format.
- `teachers/*.jsonl`: one fixed teacher answer and verifier score per training
  question. Each training run selects its cache in `reward_model.reference.path`.

| File | Rows by task |
| --- | --- |
| `biology-chemistry-physics/train_verl.parquet` | biology_literqa: 960, chemistry_literqa: 960, physics_literqa: 960 |
| `biology-chemistry-physics/reference_train_verl.parquet` | biology_literqa: 960, chemistry_literqa: 960, physics_literqa: 960 |
| `biology-chemistry-physics/development_verl.parquet` | biology_literqa: 100, chemistry_literqa: 100, physics_literqa: 100 |
| `biology-chemistry-physics/test.parquet` | biology_literqa: 200, chemistry_literqa: 200, physics_literqa: 200 |
| `materials-chemistry-physics/train_verl.parquet` | chemistry_understanding: 960, material_literqa: 960, physics_calculation: 960 |
| `materials-chemistry-physics/reference_train_verl.parquet` | chemistry_understanding: 326, material_literqa: 960, physics_calculation: 474 |
| `materials-chemistry-physics/development_verl.parquet` | chemistry_understanding: 100, material_literqa: 100, physics_calculation: 100 |
| `materials-chemistry-physics/test.parquet` | chemistry_understanding: 200, material_literqa: 200, physics_calculation: 200 |
| `physics-instructions-code/train_verl.parquet` | ifeval: 960, mbpp: 960, physics_literqa: 960 |
| `physics-instructions-code/reference_train_verl.parquet` | ifeval: 750, mbpp: 371, physics_literqa: 960 |
| `physics-instructions-code/development_verl.parquet` | ifeval: 64, mbpp: 90, physics_literqa: 100 |
| `physics-instructions-code/test.parquet` | ifeval: 541, mbpp: 496, physics_literqa: 200 |

## Sources and partitions

- **Biology / chemistry / physics** and **materials / chemistry / physics** are custom
  partitions of [SciKnowEval V2](https://huggingface.co/datasets/hicai-zju/SciKnowEval/tree/92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2),
  whose only public collection is named test. Questions are grouped by normalized
  text; groups with conflicting answers are removed and one question per group is
  kept. Chemistry reading questions that share a passage stay in one split. Source
  line numbers are kept in `extra_info`.
- **Physics / instructions / code** reuses the physics partition above.
  Instruction training uses [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval/tree/47c03c73621c4aab2b824b7818681117d662770e)
  prompts from eight constraint types; evaluation uses all 541 prompts of
  [official IFEval](https://github.com/google-research/google-research/tree/e1a63cd0666f970fc5df3753e17456c9f011d2a1/instruction_following_eval),
  and no training question overlaps them.
  [MBPP](https://github.com/google-research/google-research/blob/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/mbpp.jsonl)
  keeps its original split assignments after duplicate-description removal, and
  every retained problem's reference solution passes its tests.

The source datasets retain their original terms and attribution; see
[THIRD_PARTY.md](../THIRD_PARTY.md) and the [source cards](sources/).

## Rebuilding from the sources

`src/data/prepare.py` rebuilds each mixture from the original sources with the
settings in `configs/data/`. The two SciKnowEval mixtures are reproduced row for
row. For physics / instructions / code, development and test files are identical,
and every update contains the same questions with the same task at each position;
the order of physics and MBPP questions within an update differs from the released
file, which keeps the exact order used in the paper. Download the pinned sources
into `raw/` and place the `Qwen3-0.6B` tokenizer under `models/`:

```bash
huggingface-cli download hicai-zju/SciKnowEval data/v2/sciknoweval_test_v2.jsonl --repo-type dataset \
    --revision 92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2 --local-dir raw/SciKnowEval
huggingface-cli download allenai/RLVR-IFeval data/train-00000-of-00001.parquet --repo-type dataset \
    --revision 47c03c73621c4aab2b824b7818681117d662770e --local-dir raw/RLVR-IFeval
curl -L -o raw/mbpp.jsonl https://raw.githubusercontent.com/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/mbpp.jsonl
curl -L -o raw/ifeval_input_data.jsonl https://raw.githubusercontent.com/google-research/google-research/e1a63cd0666f970fc5df3753e17456c9f011d2a1/instruction_following_eval/data/input_data.jsonl

source env.sh
python -m data.prepare --config configs/data/biology-chemistry-physics.json
python -m data.prepare --config configs/data/materials-chemistry-physics.json
python -m data.prepare --config configs/data/physics-instructions-code.json
```

Outputs are written to `outputs/data/<mixture>/`. The MBPP reference-solution check
uses the verifier's 3-second CPU limit. The reference solution of MBPP task 123 needs
close to that limit, so a slow or heavily loaded CPU can exclude it; the released
test set includes it.
