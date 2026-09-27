# DuoOPD

**Learning from Joint Teacher–Student Outcomes for Multi-Task On-Policy Distillation**

DuoOPD trains a student on its own responses using feedback from a frozen teacher.
The student's verified outcome sets the feedback direction; the joint teacher–student
outcome selects the teacher's scoring context and how feedback weights are shared.
One rule covers all four outcomes across tasks.

![DuoOPD method](assets/method.png)

[Method implementation](src/methods/duoopd/objective.py) ·
[Fixed data](data/README.md) · [Third-party sources](THIRD_PARTY.md)

This repository provides DuoOPD, sampled-token OPD, the method's ablation switches,
and five representative training/evaluation configurations. The paper also compares
EOPD, OPDVR, ExOPD, and FiRe-OPD; their separate training implementations are not
included here. Experiments in the paper report three training runs per condition;
the supplied configurations start with seed 42 and the repeat instructions below
cover seeds 43 and 44.

Model weights and trained checkpoints are obtained or generated separately.


## Setup

Use Linux, Python 3.10, CUDA 12.8, and NVIDIA GPUs. Install Bubblewrap (`bwrap`)
for MBPP verification; user namespaces must be enabled.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install --no-deps --no-build-isolation -e external/verl
python -m nltk.downloader -d nltk_data punkt_tab
```

The bundled verl dependency metadata matches the pinned NumPy version used by
the experiments.

Place the Hugging Face model directories (or symlinks to them) under `models/`:

| Directory | Model |
| --- | --- |
| `Qwen3-0.6B` | `Qwen/Qwen3-0.6B` |
| `Qwen3-4B` | `Qwen/Qwen3-4B` |
| `Llama-3.2-3B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| `Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` |

Only the two models for the selected family are needed. Model access and weights
come from their original providers.

## Train and Evaluate

From the repository root, for example:

```bash
bash configs/qwen/duoopd/train.sh
CUDA_VISIBLE_DEVICES=0 bash configs/qwen/duoopd/evaluate.sh
```

These five representative configurations use seed 42 and four GPUs. Allocate
the GPUs before running. Evaluation uses one GPU and the fixed update-60 checkpoint.
Outputs go to `outputs/<setting>/<method>/`, with evaluation results in its
`evaluation/` directory; a completed checkpoint can be evaluated again without
training. The checkpoint settings retain one resumable checkpoint. Evaluation reports avg@8 (mean success across eight answers), with
Macro averaging tasks equally. IFEval uses prompt-level strict accuracy.

| Configuration directory | Paper experiment |
| --- | --- |
| `configs/qwen/duoopd/` | Biology / chemistry / physics, Qwen3 pair |
| `configs/qwen/opd/` | Sampled-token OPD under the same Qwen3 protocol |
| `configs/llama/duoopd/` | Biology / chemistry / physics, Llama pair |
| `configs/science/duoopd/` | Materials knowledge / chemistry understanding / physics calculation |
| `configs/heterogeneous/duoopd/` | Physics / instruction following / code generation |

## Ablations and Repeats

Append these Hydra overrides to the Qwen DuoOPD training command:

| Variant | Overrides |
| --- | --- |
| No teacher reference | `+algorithm.duoopd.teacher_reference=false` |
| Raw reference log-ratio | `+algorithm.duoopd.raw_reference_ratio=true` |
| No reference or sharing | `+algorithm.duoopd.teacher_reference=false +algorithm.duoopd.online_mean=false ~algorithm.duoopd.online_mean_scope` |
| No weight sharing | `+algorithm.duoopd.online_mean=false ~algorithm.duoopd.online_mean_scope` |
| Within-response sharing | `+algorithm.duoopd.online_mean=false +algorithm.duoopd.response_mean=true ~algorithm.duoopd.online_mean_scope` |
| Across-task sharing | `algorithm.duoopd.online_mean_scope=batch` |
| One outcome restored to OPD | `+algorithm.duoopd.opd_cell=teacher_only` (or `student_only`, `shared_success`, `shared_failure`) |

Use a separate `RUN_DIR` for each variant, for example:

```bash
RUN_DIR="$PWD/outputs/qwen/no-reference" bash configs/qwen/duoopd/train.sh +algorithm.duoopd.teacher_reference=false
RUN_DIR="$PWD/outputs/qwen/no-reference" CUDA_VISIBLE_DEVICES=0 bash configs/qwen/duoopd/evaluate.sh
```

For repeats, set `data.seed`, `actor_rollout_ref.actor.data_loader_seed`, and
`actor_rollout_ref.rollout.seed` together to 43 or 44, with a separate `RUN_DIR`.

## Code and Inputs

- `src/methods/duoopd/objective.py`: four-outcome feedback and within-task sharing.
- `src/methods/duoopd/teacher.py`: verified-reference routing and aligned token scoring.
- `src/methods/duoopd/trainer.py`: complete-batch feedback followed by the verl update.
- `src/evaluation/verifier.py`: binary verifiers for all tasks, shared by training and evaluation.
- `src/evaluation/evaluate.py`: generation, verification, and avg@8 aggregation.
- `src/data/prepare.py`: rebuild a task mixture from the original datasets
  (`sciknoweval.py`, `physics_instructions_code.py`, and the ordered stream in `stream.py`).
- `src/data/prepare_teacher_references.py`: generate a teacher cache from the training inputs.
- `external/`: the required framework runtime sources and third-party checkers.
- `data/`: exact processed partitions, ordered training streams, and teacher caches.

The Parquet files preserve question IDs, prompts, targets, and ordering, and are
ready to use. `data/README.md` describes their sources and partitions, and how
`src/data/prepare.py` rebuilds them from the original datasets. Each cache contains
one teacher response and its verifier result per training question. DuoOPD uses a
successful teacher response as reference context when the student fails.

To generate a new teacher cache, choose a JSON in `configs/teachers/` and run the
command below from the repository root on one GPU. New caches are written to
`outputs/teacher-generation/`, leaving the supplied caches unchanged; an interrupted
run resumes from the answers already generated.

```bash
source env.sh
python -m data.prepare_teacher_references --config configs/teachers/qwen.json
```

To train with the new cache, override `reward_model.reference.path` with the
generated `teacher.jsonl` path.

The full reference prompt is defined in `reference_prompt()` in
`src/methods/duoopd/teacher.py`.

## Acknowledgements

DuoOPD builds on [verl](https://github.com/verl-project/verl) for training and
rollout. Evaluation uses the official
[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
checkers and the RLVR-IFeval checks from AllenAI's
[Open Instruct](https://github.com/allenai/open-instruct) (Tülu 3). Tasks come from
[SciKnowEval](https://github.com/HICAI-ZJU/sciknoweval),
[RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), IFEval, and
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp).
The experiments use [Qwen3](https://huggingface.co/Qwen) and
[Llama 3](https://huggingface.co/meta-llama) models. We thank their authors for
releasing them.

## License

The original DuoOPD code is released under the [Apache License 2.0](LICENSE).
Third-party code, data, and models keep their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md). Their licenses and notices are retained in their directories.
