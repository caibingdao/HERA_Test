#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/users/cbd/code/ARiADNE_MAMBA_Test"
PYTHON_BIN="/users/cbd/miniconda3/envs/ariadne_mamba/bin/python"
OUTPUT_DIR="experiment/checkpoints/ariadne_no_mamba_seed_0"
SPLIT_FILE="experiment/map_splits/map_split_seed_42.json"
MIN_FREE_MEMORY_MIB=16000
MAX_GPU_UTILIZATION=20
REQUIRED_IDLE_CHECKS=3
CHECK_INTERVAL_SECONDS=60

cd "$PROJECT_DIR"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "ERROR: fresh output target already exists: $OUTPUT_DIR" >&2
  exit 1
fi

idle_checks=0
while true; do
  IFS=, read -r free_memory utilization < <(
    nvidia-smi \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      | head -n 1
  )
  free_memory="${free_memory//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  timestamp="$(date --iso-8601=seconds)"

  if ((
    free_memory >= MIN_FREE_MEMORY_MIB &&
    utilization <= MAX_GPU_UTILIZATION
  )); then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi

  echo "$timestamp GPU free=${free_memory}MiB util=${utilization}% idle_checks=${idle_checks}/${REQUIRED_IDLE_CHECKS}"
  if (( idle_checks >= REQUIRED_IDLE_CHECKS )); then
    break
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "GPU is available; starting fresh ARIADNE training."
exec "$PYTHON_BIN" -u scripts/train_on_split.py \
  --split-file "$SPLIT_FILE" \
  --split-name train \
  --training-seed 0 \
  --total-env-steps 1000000 \
  --output-dir "$OUTPUT_DIR"
