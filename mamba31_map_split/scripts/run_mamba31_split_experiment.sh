#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
SPLIT_FILE="${SPLIT_FILE:-$ROOT_DIR/experiment/map_splits/map_split_seed_42.json}"
LEGACY_CHECKPOINT="${LEGACY_CHECKPOINT:-/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/ARiADNE_MAMBA_3.1/model/parallel_ariadne_mamba_3_1/checkpoint.pth}"
TRAINING_SEED="${TRAINING_SEED:-0}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-1000000}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/experiment/checkpoints/mamba31_seed_${TRAINING_SEED}}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/experiment/results/mamba31_seed_${TRAINING_SEED}}"
RESUME_FROM_LEGACY="${RESUME_FROM_LEGACY:-0}"
EVAL_SEEDS=(100 101 102 103 104)
EPISODES_PER_MAP="${EPISODES_PER_MAP:-5}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.99}"
MAX_STEPS="${MAX_STEPS:-128}"
EVAL_WORKERS="${EVAL_WORKERS:-5}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"

cd "$ROOT_DIR"

if [[ "$RESUME_FROM_LEGACY" != "0" && "$RESUME_FROM_LEGACY" != "1" ]]; then
  echo "RESUME_FROM_LEGACY must be 0 (fresh) or 1 (resume)" >&2
  exit 2
fi

if [[ "${#EVAL_SEEDS[@]}" -ne "$EPISODES_PER_MAP" ]]; then
  echo "EPISODES_PER_MAP must equal the number of evaluation seeds" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/split_maps.py \
  --map-root maps \
  --train-ratio 0.8 \
  --seed 42 \
  --output "$SPLIT_FILE" \
  --group-by-base-map \
  --stratify

if [[ "$RESUME_FROM_LEGACY" == "0" ]]; then
  if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Fresh training refuses a non-empty run directory: $RUN_DIR" >&2
    echo "Choose a new RUN_DIR or remove/archive that directory explicitly." >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR"
  TRAIN_ARGS=()
else
  if [[ ! -f "$LEGACY_CHECKPOINT" ]]; then
    echo "Legacy checkpoint not found: $LEGACY_CHECKPOINT" >&2
    exit 2
  fi
  if [[ ! -f "$RUN_DIR/checkpoint.pth" ]]; then
    "$PYTHON_BIN" scripts/prepare_legacy_resume.py \
      --checkpoint "$LEGACY_CHECKPOINT" \
      --split-file "$SPLIT_FILE" \
      --split-name train \
      --training-seed "$TRAINING_SEED" \
      --output-dir "$RUN_DIR"
  fi
  TRAIN_ARGS=(--resume-checkpoint "$RUN_DIR/checkpoint.pth")
fi

"$PYTHON_BIN" scripts/train_on_split.py \
  --split-file "$SPLIT_FILE" \
  --split-name train \
  --training-seed "$TRAINING_SEED" \
  --total-env-steps "$TOTAL_ENV_STEPS" \
  --output-dir "$RUN_DIR" \
  "${TRAIN_ARGS[@]}"

"$PYTHON_BIN" scripts/evaluate_parallel.py \
  --checkpoint "$RUN_DIR/final_checkpoint.pth" \
  --split-file "$SPLIT_FILE" \
  --split-name test \
  --eval-seeds "${EVAL_SEEDS[@]}" \
  --episodes-per-map "$EPISODES_PER_MAP" \
  --target-coverage "$TARGET_COVERAGE" \
  --max-steps "$MAX_STEPS" \
  --workers "$EVAL_WORKERS" \
  --threads-per-worker "$THREADS_PER_WORKER" \
  --output-dir "$RESULT_DIR" \
  --deterministic \
  --adapter evaluation_adapters.ariadne:create_adapter \
  --adapter-arg model_name='"ARiADNE-Mamba-3.1"' \
  --adapter-arg model_seed="$TRAINING_SEED" \
  --fail-fast

echo "Mamba 3.1 split training and evaluation complete."
echo "Checkpoint: $RUN_DIR/final_checkpoint.pth"
echo "Results: $RESULT_DIR"
