#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../../.."
source env.sh
export RUN_DIR="${RUN_DIR:-$PROJECT_SOURCE/outputs/llama/duoopd}"
export EXPERIMENT_ID="llama-duoopd"
mkdir -p "$RUN_DIR"
python -m methods.duoopd.main --config-path "$PROJECT_SOURCE/configs/llama/duoopd" --config-name train "$@"
