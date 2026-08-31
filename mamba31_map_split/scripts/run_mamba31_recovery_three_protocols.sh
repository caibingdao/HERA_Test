#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
RUN_NAME="${RUN_NAME:-mamba31_recovery_v4_three_protocols}"
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

CHECKPOINT="$ROOT_DIR/experiment/checkpoints/mamba31_fresh_seed_0_v5/checkpoint.pth"
SPLIT_FILE="$ROOT_DIR/experiment/map_splits/map_split_seed_42.json"
RESULT_ROOT="$ROOT_DIR/experiment/results/$RUN_NAME"
VISUAL_ROOT="$ROOT_DIR/experiment/visualization/$RUN_NAME"
PROTOCOL_LOG_ROOT="$ROOT_DIR/experiment/logs/$RUN_NAME"
MPL_CONFIG_DIR="$ROOT_DIR/experiment/.matplotlib"
MODEL_NAME="ARiADNE-Mamba-3.1+Recovery-V4"

for required_file in "$PYTHON_BIN" "$CHECKPOINT" "$SPLIT_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file does not exist: $required_file" >&2
        exit 1
    fi
done

for target in "$RESULT_ROOT" "$VISUAL_ROOT" "$PROTOCOL_LOG_ROOT"; do
    if [[ -e "$target" ]]; then
        echo "Fresh evaluation refuses an existing target: $target" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT" "$VISUAL_ROOT" "$PROTOCOL_LOG_ROOT" "$MPL_CONFIG_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"
cd "$ROOT_DIR"

run_five_seed_greedy() {
    local gpu_id="$1"
    local result_dir="$RESULT_ROOT/five_seed_greedy"
    local visual_dir="$VISUAL_ROOT/five_seed_greedy"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u scripts/evaluate_parallel.py \
        --checkpoint "$CHECKPOINT" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --eval-seeds 100 101 102 103 104 \
        --episodes-per-map 5 \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$result_dir" \
        --workers 5 \
        --threads-per-worker 4 \
        --deterministic \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$MODEL_NAME\"" \
        --adapter-arg model_seed=0 \
        --adapter-arg enable_recovery=true

    "$PYTHON_BIN" scripts/create_visual_report.py \
        --results-dir "$result_dir" \
        --output-dir "$visual_dir"
}

run_single_seed_stochastic() {
    local gpu_id="$1"
    local result_dir="$RESULT_ROOT/single_seed_stochastic_4777"
    local visual_dir="$VISUAL_ROOT/single_seed_stochastic_4777"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
        scripts/evaluate_single_seed_stochastic.py \
        --checkpoint "$CHECKPOINT" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --global-seed 4777 \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$result_dir" \
        --visual-output-dir "$visual_dir" \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$MODEL_NAME\"" \
        --adapter-arg model_seed=0 \
        --adapter-arg enable_recovery=true
}

run_five_seed_stochastic() {
    local gpu_id="$1"
    local result_dir="$RESULT_ROOT/five_seed_stochastic"
    local visual_dir="$VISUAL_ROOT/five_seed_stochastic"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
        scripts/evaluate_parallel_stochastic.py \
        --checkpoint "$CHECKPOINT" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --eval-seeds 100 101 102 103 104 \
        --episodes-per-map 5 \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$result_dir" \
        --workers 5 \
        --threads-per-worker 4 \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$MODEL_NAME\"" \
        --adapter-arg model_seed=0 \
        --adapter-arg enable_recovery=true

    "$PYTHON_BIN" scripts/create_visual_report.py \
        --results-dir "$result_dir" \
        --output-dir "$visual_dir"
}

protocol_names=(
    "five_seed_greedy"
    "single_seed_stochastic_4777"
    "five_seed_stochastic"
)
pids=()

run_five_seed_greedy "${GPU_IDS_ARRAY[0]}" \
    >"$PROTOCOL_LOG_ROOT/${protocol_names[0]}.log" 2>&1 &
pids+=("$!")
echo "Launched ${protocol_names[0]}: pid=${pids[0]}, GPU=${GPU_IDS_ARRAY[0]}"

run_single_seed_stochastic "${GPU_IDS_ARRAY[1]}" \
    >"$PROTOCOL_LOG_ROOT/${protocol_names[1]}.log" 2>&1 &
pids+=("$!")
echo "Launched ${protocol_names[1]}: pid=${pids[1]}, GPU=${GPU_IDS_ARRAY[1]}"

run_five_seed_stochastic "${GPU_IDS_ARRAY[2]}" \
    >"$PROTOCOL_LOG_ROOT/${protocol_names[2]}.log" 2>&1 &
pids+=("$!")
echo "Launched ${protocol_names[2]}: pid=${pids[2]}, GPU=${GPU_IDS_ARRAY[2]}"

failed=0
for index in 0 1 2; do
    if wait "${pids[$index]}"; then
        echo "Protocol succeeded: ${protocol_names[$index]}"
    else
        echo "Protocol failed: ${protocol_names[$index]} (see $PROTOCOL_LOG_ROOT/${protocol_names[$index]}.log)" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "At least one protocol failed; partial outputs were preserved for diagnosis." >&2
    exit 1
fi

echo "All three Mamba 3.1 + Recovery V4 evaluations completed."
echo "Results: $RESULT_ROOT"
echo "Visualizations: $VISUAL_ROOT"

