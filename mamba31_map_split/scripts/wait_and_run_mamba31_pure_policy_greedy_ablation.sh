#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"

if [[ ! "$CHECK_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHECK_INTERVAL_SECONDS must be a positive integer" >&2
    exit 1
fi

while ! "$PYTHON_BIN" -c '
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1)
'; do
    echo "$(date --iso-8601=seconds) waiting for at least two healthy CUDA GPUs"
    sleep "$CHECK_INTERVAL_SECONDS"
done

echo "$(date --iso-8601=seconds) CUDA recovered; starting pure-policy greedy ablation"
exec /usr/bin/bash "$ROOT_DIR/scripts/run_mamba31_pure_policy_greedy_ablation.sh"

