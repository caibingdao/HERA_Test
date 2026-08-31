#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
RUN_NAME="${RUN_NAME:-three_models_five_seed_greedy_100_104_recovery_v4}"
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
FINAL_VISUAL_ROOT="$ROOT_DIR/experiment/visualization/greedy_5_seeds_100_104"
STAGING_VISUAL_ROOT="$ROOT_DIR/experiment/visualization/.greedy_5_seeds_100_104_rebuild"
ARCHIVE_ROOT="$ROOT_DIR/experiment/visualization_archive"
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
if [[ ! -d "$FINAL_VISUAL_ROOT" ]]; then
    echo "Existing visualization directory is missing: $FINAL_VISUAL_ROOT" >&2
    exit 1
fi
for fresh_target in "$RESULT_ROOT" "$STAGING_VISUAL_ROOT" "$MODEL_LOG_ROOT"; do
    if [[ -e "$fresh_target" ]]; then
        echo "Fresh rerun refuses an existing target: $fresh_target" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT" "$STAGING_VISUAL_ROOT" "$MODEL_LOG_ROOT" "$MPL_CONFIG_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"
cd "$ROOT_DIR"

run_model() {
    local model_key="$1"
    local model_name="$2"
    local checkpoint="$3"
    local enable_recovery="$4"
    local gpu_id="$5"
    local visual_dir="$6"

    echo "Starting $model_name on physical GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u \
        scripts/evaluate_parallel.py \
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
        --deterministic \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg "model_name=\"$model_name\"" \
        --adapter-arg model_seed=0 \
        --adapter-arg "enable_recovery=$enable_recovery"

    "$PYTHON_BIN" scripts/create_visual_report.py \
        --results-dir "$RESULT_ROOT/$model_key" \
        --output-dir "$visual_dir"
    echo "Completed $model_name"
}

model_keys=("ariadne_original" "ariadne_mamba" "ariadne_mamba_3_1_recovery_v4")
model_names=("ARiADNE-original" "ARiADNE-Mamba" "ARiADNE-Mamba-3.1+Recovery-V4")
checkpoints=("$ORIGINAL_CHECKPOINT" "$MAMBA_CHECKPOINT" "$MAMBA31_CHECKPOINT")
recovery_flags=("false" "false" "true")
visual_dirs=(
    "$STAGING_VISUAL_ROOT/ariadne_original"
    "$STAGING_VISUAL_ROOT/ariadne_mamba"
    "$STAGING_VISUAL_ROOT/ariadne_mamba_3_1/recovery_v4"
)
pids=()

for index in 0 1 2; do
    run_model \
        "${model_keys[$index]}" \
        "${model_names[$index]}" \
        "${checkpoints[$index]}" \
        "${recovery_flags[$index]}" \
        "${GPU_IDS_ARRAY[$index]}" \
        "${visual_dirs[$index]}" \
        >"$MODEL_LOG_ROOT/${model_keys[$index]}.log" 2>&1 &
    pids+=("$!")
    echo "Launched ${model_names[$index]}: pid=${pids[$index]}, GPU=${GPU_IDS_ARRAY[$index]}"
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
    echo "At least one model failed; the current visualizations were not changed." >&2
    exit 1
fi

echo "Creating combined three-model comparison"
"$PYTHON_BIN" scripts/aggregate_results.py \
    "$RESULT_ROOT/ariadne_original" \
    "$RESULT_ROOT/ariadne_mamba" \
    "$RESULT_ROOT/ariadne_mamba_3_1_recovery_v4" \
    --output-dir "$STAGING_VISUAL_ROOT/cross_model_comparison" \
    --bootstrap-seed 42 \
    --bootstrap-samples 10000

expected_episodes=5665
for model_key in "${model_keys[@]}"; do
    episode_file="$RESULT_ROOT/$model_key/episodes.csv"
    metadata_file="$RESULT_ROOT/$model_key/evaluation_metadata.json"
    if [[ ! -f "$episode_file" || ! -f "$metadata_file" ]]; then
        echo "Missing completed result files for $model_key" >&2
        exit 1
    fi
    episode_count=$(( $(wc -l < "$episode_file") - 1 ))
    if [[ "$episode_count" -ne "$expected_episodes" ]]; then
        echo "Expected $expected_episodes episodes for $model_key, found $episode_count" >&2
        exit 1
    fi
done

for index in 0 1 2; do
    "$PYTHON_BIN" -c '
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected_recovery = sys.argv[2] == "true"
if metadata.get("deterministic") is not True:
    raise SystemExit("metadata does not record deterministic greedy evaluation")
if int(metadata.get("episodes_per_map", -1)) != 5:
    raise SystemExit("metadata does not record five episodes per map")
if metadata.get("eval_seeds") != [100, 101, 102, 103, 104]:
    raise SystemExit("metadata records unexpected evaluation seeds")
if metadata.get("adapter_args", {}).get("enable_recovery") is not expected_recovery:
    raise SystemExit("adapter_args records an unexpected recovery setting")
if metadata.get("adapter_metadata", {}).get("recovery_enabled") is not expected_recovery:
    raise SystemExit("adapter metadata records an unexpected recovery setting")
' "$RESULT_ROOT/${model_keys[$index]}/evaluation_metadata.json" "${recovery_flags[$index]}"
done

for visual_dir in "${visual_dirs[@]}"; do
    chart_count=$(find "$visual_dir" -maxdepth 1 -type f -name '*.png' | wc -l)
    if [[ "$chart_count" -ne 6 ]]; then
        echo "Expected six standard charts, found $chart_count: $visual_dir" >&2
        exit 1
    fi
done
if [[ ! -f "$STAGING_VISUAL_ROOT/cross_model_comparison/overall_summary.json" ]]; then
    echo "Cross-model comparison is incomplete; current visualizations were not changed." >&2
    exit 1
fi

mkdir -p "$ARCHIVE_ROOT"
archive_stamp="$(date +%Y%m%d_%H%M%S)"
archive_dir="$ARCHIVE_ROOT/greedy_5_seeds_100_104_before_${archive_stamp}"
mv "$FINAL_VISUAL_ROOT" "$archive_dir"
if ! mv "$STAGING_VISUAL_ROOT" "$FINAL_VISUAL_ROOT"; then
    mv "$archive_dir" "$FINAL_VISUAL_ROOT"
    echo "Visualization replacement failed; the previous directory was restored." >&2
    exit 1
fi

echo "Three-model five-seed greedy rerun complete."
echo "Results: $RESULT_ROOT"
echo "Visualizations replaced at: $FINAL_VISUAL_ROOT"
echo "Previous visualizations archived at: $archive_dir"
