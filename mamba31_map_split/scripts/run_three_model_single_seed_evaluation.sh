#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
GLOBAL_SEED="${GLOBAL_SEED:-4777}"
RUN_NAME="${RUN_NAME:-three_models_single_seed_${GLOBAL_SEED}}"

SPLIT_FILE="$ROOT_DIR/experiment/map_splits/map_split_seed_42.json"
RESULT_ROOT="$ROOT_DIR/experiment/results/$RUN_NAME"
VISUAL_ROOT="$ROOT_DIR/experiment/visualization/$RUN_NAME"

ORIGINAL_CHECKPOINT="/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/ARiADNE_ORIGINAL_PARALLEL/model/parallel_ariadne_original/checkpoint.pth"
MAMBA_CHECKPOINT="/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/ARiADNE_MAMBA_Test/model/parallel_ariadne_mamba_optimized_mamba/checkpoint.pth"
MAMBA31_CHECKPOINT="$ROOT_DIR/experiment/checkpoints/mamba31_fresh_seed_0_v5/checkpoint.pth"

for checkpoint in \
    "$ORIGINAL_CHECKPOINT" \
    "$MAMBA_CHECKPOINT" \
    "$MAMBA31_CHECKPOINT"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "Checkpoint does not exist: $checkpoint" >&2
        exit 1
    fi
done

for target in "$RESULT_ROOT" "$VISUAL_ROOT"; do
    if [[ -e "$target" ]]; then
        echo "Fresh evaluation refuses an existing target: $target" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT" "$VISUAL_ROOT"
cd "$ROOT_DIR"

run_model() {
    local model_key="$1"
    local model_name="$2"
    local checkpoint="$3"

    echo "Starting $model_name with global seed $GLOBAL_SEED"
    "$PYTHON_BIN" scripts/evaluate_single_seed_stochastic.py \
        --checkpoint "$checkpoint" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --global-seed "$GLOBAL_SEED" \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$RESULT_ROOT/$model_key" \
        --visual-output-dir "$VISUAL_ROOT/$model_key" \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$model_name\"" \
        --adapter-arg model_seed=0
    echo "Completed $model_name"
}

# These calls are deliberately sequential.  Each model owns one continuous
# stochastic action stream initialized once from the same global seed.
run_model \
    "ariadne_original" \
    "ARiADNE-original" \
    "$ORIGINAL_CHECKPOINT"
run_model \
    "ariadne_mamba" \
    "ARiADNE-Mamba" \
    "$MAMBA_CHECKPOINT"
run_model \
    "ariadne_mamba_3_1" \
    "ARiADNE-Mamba-3.1" \
    "$MAMBA31_CHECKPOINT"

echo "Creating combined three-model comparison"
"$PYTHON_BIN" scripts/aggregate_results.py \
    "$RESULT_ROOT/ariadne_original" \
    "$RESULT_ROOT/ariadne_mamba" \
    "$RESULT_ROOT/ariadne_mamba_3_1" \
    --output-dir "$VISUAL_ROOT/comparison" \
    --bootstrap-seed 42 \
    --bootstrap-samples 10000

echo "Three-model single-seed evaluation complete."
echo "Results: $RESULT_ROOT"
echo "Visualizations: $VISUAL_ROOT"
