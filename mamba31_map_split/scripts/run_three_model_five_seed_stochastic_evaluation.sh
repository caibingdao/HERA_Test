#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
RUN_NAME="${RUN_NAME:-three_models_five_seed_stochastic_100_104}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2}"
read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"

if [[ "${#GPU_IDS_ARRAY[@]}" -ne 3 ]]; then
    echo "GPU_IDS must contain exactly three GPU IDs, for example: GPU_IDS='0 1 2'" >&2
    exit 1
fi
if [[ "${GPU_IDS_ARRAY[0]}" == "${GPU_IDS_ARRAY[1]}" || \
      "${GPU_IDS_ARRAY[0]}" == "${GPU_IDS_ARRAY[2]}" || \
      "${GPU_IDS_ARRAY[1]}" == "${GPU_IDS_ARRAY[2]}" ]]; then
    echo "GPU_IDS must contain three distinct GPU IDs" >&2
    exit 1
fi

SPLIT_FILE="$ROOT_DIR/experiment/map_splits/map_split_seed_42.json"
RESULT_ROOT="$ROOT_DIR/experiment/results/$RUN_NAME"
VISUAL_ROOT="$ROOT_DIR/experiment/visualization/$RUN_NAME"
MODEL_LOG_ROOT="$ROOT_DIR/experiment/logs/$RUN_NAME"
MPL_CONFIG_DIR="$ROOT_DIR/experiment/.matplotlib"

ORIGINAL_CHECKPOINT="/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/ARiADNE_ORIGINAL_PARALLEL/model/parallel_ariadne_original/checkpoint.pth"
MAMBA_CHECKPOINT="/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/ARiADNE_MAMBA_Test/model/parallel_ariadne_mamba_optimized_mamba/checkpoint.pth"
MAMBA31_CHECKPOINT="$ROOT_DIR/experiment/checkpoints/mamba31_fresh_seed_0_v5/checkpoint.pth"

for required_file in \
    "$PYTHON_BIN" \
    "$SPLIT_FILE" \
    "$ORIGINAL_CHECKPOINT" \
    "$MAMBA_CHECKPOINT" \
    "$MAMBA31_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file does not exist: $required_file" >&2
        exit 1
    fi
done

for target in "$RESULT_ROOT" "$VISUAL_ROOT" "$MODEL_LOG_ROOT"; do
    if [[ -e "$target" ]]; then
        echo "Fresh evaluation refuses an existing target: $target" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT" "$VISUAL_ROOT" "$MODEL_LOG_ROOT" "$MPL_CONFIG_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"
cd "$ROOT_DIR"

run_model() {
    local model_key="$1"
    local model_name="$2"
    local checkpoint="$3"
    local gpu_id="$4"

    echo "Starting $model_name on physical GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
        scripts/evaluate_parallel_stochastic.py \
        --checkpoint "$checkpoint" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --eval-seeds 100 101 102 103 104 \
        --episodes-per-map 5 \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$RESULT_ROOT/$model_key" \
        --workers 5 \
        --threads-per-worker 4 \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$model_name\"" \
        --adapter-arg model_seed=0

    "$PYTHON_BIN" scripts/create_visual_report.py \
        --results-dir "$RESULT_ROOT/$model_key" \
        --output-dir "$VISUAL_ROOT/$model_key"
    echo "Completed $model_name on physical GPU $gpu_id"
}

model_keys=("ariadne_original" "ariadne_mamba" "ariadne_mamba_3_1")
model_names=("ARiADNE-original" "ARiADNE-Mamba" "ARiADNE-Mamba-3.1")
checkpoints=("$ORIGINAL_CHECKPOINT" "$MAMBA_CHECKPOINT" "$MAMBA31_CHECKPOINT")
pids=()

for index in 0 1 2; do
    run_model \
        "${model_keys[$index]}" \
        "${model_names[$index]}" \
        "${checkpoints[$index]}" \
        "${GPU_IDS_ARRAY[$index]}" \
        >"$MODEL_LOG_ROOT/${model_keys[$index]}.log" 2>&1 &
    pids+=("$!")
    echo "Launched ${model_names[$index]}: pid=${pids[$index]}, GPU=${GPU_IDS_ARRAY[$index]}, log=$MODEL_LOG_ROOT/${model_keys[$index]}.log"
done

failed=0
for index in 0 1 2; do
    if wait "${pids[$index]}"; then
        echo "Model job succeeded: ${model_names[$index]}"
    else
        echo "Model job failed: ${model_names[$index]} (see $MODEL_LOG_ROOT/${model_keys[$index]}.log)" >&2
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "At least one model failed; completed partial outputs were preserved for diagnosis/resume." >&2
    exit 1
fi

echo "Creating combined three-model comparison"
"$PYTHON_BIN" scripts/aggregate_results.py \
    "$RESULT_ROOT/ariadne_original" \
    "$RESULT_ROOT/ariadne_mamba" \
    "$RESULT_ROOT/ariadne_mamba_3_1" \
    --output-dir "$VISUAL_ROOT/comparison" \
    --bootstrap-seed 42 \
    --bootstrap-samples 10000

echo "Three-model five-seed stochastic evaluation complete."
echo "Results: $RESULT_ROOT"
echo "Visualizations: $VISUAL_ROOT"
