#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../../.."
source env.sh
export RUN_DIR="${RUN_DIR:-$PROJECT_SOURCE/outputs/science/duoopd}"
python -m evaluation.evaluate --config configs/science/duoopd/evaluate.json \
    --model-path "$RUN_DIR/train/global_step_60/actor/huggingface" --run-dir "$RUN_DIR/evaluation"
