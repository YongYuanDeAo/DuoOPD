#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../../.."
source env.sh
export RUN_DIR="${RUN_DIR:-$PROJECT_SOURCE/outputs/qwen/opd}"
export EXPERIMENT_ID="qwen-opd"
mkdir -p "$RUN_DIR"
python -m methods.duoopd.main --config-path "$PROJECT_SOURCE/configs/qwen/opd" --config-name train "$@"
