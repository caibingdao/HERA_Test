#!/usr/bin/env bash
set -euo pipefail

# ---------------------------- experiment settings ---------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
MAP_ROOT="${MAP_ROOT:-maps}"
SPLIT_FILE="${SPLIT_FILE:-experiment/map_splits/map_split_seed_42.json}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_SEEDS=(0 1 2)
EVAL_SEEDS=(100 101 102 103 104)
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-1000000}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.99}"
EPISODES_PER_MAP="${EPISODES_PER_MAP:-5}"
MAX_STEPS="${MAX_STEPS:-128}"
MAX_PATH_LENGTH="${MAX_PATH_LENGTH:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-experiment/checkpoints}"
RESULT_ROOT="${RESULT_ROOT:-experiment/results}"
SUMMARY_DIR="${SUMMARY_DIR:-experiment/summary}"
EVAL_ADAPTER="${EVAL_ADAPTER:-evaluation_adapters.ariadne:create_adapter}"
# Repeat model-specific KEY=JSON_VALUE entries here when needed.
EVAL_ADAPTER_ARGS=()
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
# -----------------------------------------------------------------------------

CURRENT_STAGE="initialization"
trap 'echo "ERROR: stage \"$CURRENT_STAGE\" failed (line $LINENO)." >&2' ERR

if [[ "${#EVAL_SEEDS[@]}" -ne "$EPISODES_PER_MAP" ]]; then
  echo "ERROR: EPISODES_PER_MAP must equal the number of EVAL_SEEDS." >&2
  exit 1
fi

CURRENT_STAGE="map split validation/generation"
"$PYTHON_BIN" scripts/split_maps.py \
  --map-root "$MAP_ROOT" \
  --train-ratio "$TRAIN_RATIO" \
  --seed "$SPLIT_SEED" \
  --output "$SPLIT_FILE" \
  --group-by-base-map \
  --stratify

RESULT_INPUTS=()
for TRAIN_SEED in "${TRAIN_SEEDS[@]}"; do
  MODEL_DIR="$CHECKPOINT_ROOT/seed_${TRAIN_SEED}"
  CHECKPOINT="$MODEL_DIR/final_checkpoint.pth"
  RESULT_DIR="$RESULT_ROOT/seed_${TRAIN_SEED}"

  CURRENT_STAGE="training seed ${TRAIN_SEED}"
  "$PYTHON_BIN" scripts/train_on_split.py \
    --split-file "$SPLIT_FILE" \
    --split-name train \
    --training-seed "$TRAIN_SEED" \
    --total-env-steps "$TOTAL_ENV_STEPS" \
    --output-dir "$MODEL_DIR"

  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: final checkpoint not found after seed $TRAIN_SEED: $CHECKPOINT" >&2
    exit 1
  fi

  CURRENT_STAGE="evaluation seed ${TRAIN_SEED}"
  EVAL_COMMAND=(
    "$PYTHON_BIN" scripts/evaluate_on_split.py
    --checkpoint "$CHECKPOINT"
    --split-file "$SPLIT_FILE"
    --split-name test
    --eval-seeds "${EVAL_SEEDS[@]}"
    --episodes-per-map "$EPISODES_PER_MAP"
    --target-coverage "$TARGET_COVERAGE"
    --max-steps "$MAX_STEPS"
    --output-dir "$RESULT_DIR"
    --deterministic
    --adapter "$EVAL_ADAPTER"
    --fail-fast
  )
  if [[ -n "$MAX_PATH_LENGTH" ]]; then
    EVAL_COMMAND+=(--max-path-length "$MAX_PATH_LENGTH")
  fi
  for ADAPTER_ARG in "${EVAL_ADAPTER_ARGS[@]}"; do
    EVAL_COMMAND+=(--adapter-arg "$ADAPTER_ARG")
  done
  "${EVAL_COMMAND[@]}"
  RESULT_INPUTS+=("$RESULT_DIR")
done

CURRENT_STAGE="aggregation and plotting"
"$PYTHON_BIN" scripts/aggregate_results.py \
  "${RESULT_INPUTS[@]}" \
  --output-dir "$SUMMARY_DIR" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"

CURRENT_STAGE="complete"
echo "Experiment complete. Summary: $SUMMARY_DIR"
