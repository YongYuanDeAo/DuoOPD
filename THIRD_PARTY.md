# Sources and licenses

The original DuoOPD code is licensed under Apache-2.0 ([LICENSE](LICENSE)).
Licenses of third-party code, datasets, and models apply independently.

## Code

- **verl**: [verl-project/verl v0.7.0](https://github.com/verl-project/verl/tree/v0.7.0),
  upstream commit `f9c855f7cf04d603c9546bc01776c74806a879c1`, Apache-2.0.
  The complete framework is bundled under `external/verl/`, including its
  `LICENSE` and `Notice.txt`. Local adaptations expose the complete-batch feedback
  hook and trainer class, preserve rollout token limits and seeds, configure teacher
  offloading, retain a resumable checkpoint, and maintain the rollout server's
  lifecycle. The package pins its NumPy dependency to the experimental environment.
- **Google Research IFEval checkers**: [google-research](https://github.com/google-research/google-research/tree/e1a63cd0666f970fc5df3753e17456c9f011d2a1/instruction_following_eval),
  Apache-2.0; files and license are under `external/google-research/`.
- **Open Instruct checks**: [allenai/open-instruct](https://github.com/allenai/open-instruct),
  Apache-2.0; `if_functions.py` and its source license are under `external/open-instruct/`.

## Data

- **SciKnowEval V2**, HICAI-ZJU, revision
  `92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2`: the source card declares MIT.
  See [`data/sources/SciKnowEval.md`](data/sources/SciKnowEval.md) and the
  [original dataset](https://huggingface.co/datasets/hicai-zju/SciKnowEval/tree/92ef969ad0a8bd6e195e0ac18af2c46e307e0cc2).
  This release contains the paper's selected, deduplicated, custom partitions.
- **RLVR-IFeval**, Allen Institute for AI, revision
  `47c03c73621c4aab2b824b7818681117d662770e`: the source card declares
  [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/).
  See [`data/sources/RLVR-IFeval.md`](data/sources/RLVR-IFeval.md) and the
  [original dataset](https://huggingface.co/datasets/allenai/RLVR-IFeval/tree/47c03c73621c4aab2b824b7818681117d662770e).
  Included prompts are selected and formatted for the paper's instruction-following task.
- **IFEval evaluation prompts** and **MBPP**, Google Research: original repository
  license is preserved in `data/sources/google-research-LICENSE`; the IFEval source
  description is in the same directory. IFEval retains all 541 evaluation prompts. MBPP retains
  the original split assignments after duplicate-description removal, and includes
  task descriptions, reference code, and tests.
  The pinned [MBPP source](https://github.com/google-research/google-research/blob/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/mbpp.jsonl)
  and IFEval source linked above identify the input versions.
- **Teacher caches** in `data/teachers/` are generated responses and verifier labels
  from Qwen3-4B or Llama-3.1-8B-Instruct. They are derived experimental inputs;
  original prompt attribution remains applicable. Generation settings are in
  `configs/teachers/`.

## Models

Model weights are not distributed. Obtain Qwen3 models from
[Qwen](https://huggingface.co/Qwen) and Llama models from
[Meta](https://huggingface.co/meta-llama), following each model's access and license
requirements. The code's license does not grant rights to those weights.
