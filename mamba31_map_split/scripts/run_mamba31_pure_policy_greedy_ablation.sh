#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
RUN_NAME="${RUN_NAME:-mamba31_pure_policy_greedy_ablation}"
SINGLE_SEED="${SINGLE_SEED:-4777}"
GPU_IDS_TEXT="${GPU_IDS:-0 1}"
read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_TEXT"

if [[ "${#GPU_IDS_ARRAY[@]}" -ne 2 ]]; then
    echo "GPU_IDS must contain exactly two GPU IDs, for example: GPU_IDS='0 1'" >&2
    exit 1
fi
if [[ "${GPU_IDS_ARRAY[0]}" == "${GPU_IDS_ARRAY[1]}" ]]; then
    echo "GPU_IDS must contain two distinct GPU IDs" >&2
    exit 1
fi

CHECKPOINT="$ROOT_DIR/experiment/checkpoints/mamba31_fresh_seed_0_v5/checkpoint.pth"
SPLIT_FILE="$ROOT_DIR/experiment/map_splits/map_split_seed_42.json"
RESULT_ROOT="$ROOT_DIR/experiment/results/$RUN_NAME"
MODEL_LOG_ROOT="$ROOT_DIR/experiment/logs/$RUN_NAME"
STAGING_VISUAL_ROOT="$ROOT_DIR/experiment/visualization/.mamba31_pure_policy_greedy_ablation"
SINGLE_VISUAL_PARENT="$ROOT_DIR/experiment/visualization/greedy_1_seed_${SINGLE_SEED}/ariadne_mamba_3_1"
FIVE_VISUAL_PARENT="$ROOT_DIR/experiment/visualization/greedy_5_seeds_100_104/ariadne_mamba_3_1"
SINGLE_FINAL_VISUAL="$SINGLE_VISUAL_PARENT/pure_policy"
FIVE_FINAL_VISUAL="$FIVE_VISUAL_PARENT/pure_policy"
SINGLE_FINAL_COMPARISON="$SINGLE_VISUAL_PARENT/comparison_pure_vs_recovery_v4"
FIVE_FINAL_COMPARISON="$FIVE_VISUAL_PARENT/comparison_pure_vs_recovery_v4"
SINGLE_RECOVERY_RESULTS="$ROOT_DIR/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba_3_1_recovery_v4"
FIVE_RECOVERY_RESULTS="$ROOT_DIR/experiment/results/three_models_five_seed_greedy_100_104_recovery_v4/ariadne_mamba_3_1_recovery_v4"
MPL_CONFIG_DIR="$ROOT_DIR/experiment/.matplotlib"

for required_file in "$PYTHON_BIN" "$CHECKPOINT" "$SPLIT_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file does not exist: $required_file" >&2
        exit 1
    fi
done
for recovery_dir in "$SINGLE_RECOVERY_RESULTS" "$FIVE_RECOVERY_RESULTS"; do
    for required_name in episodes.csv coverage_curve.csv evaluation_metadata.json; do
        if [[ ! -f "$recovery_dir/$required_name" ]]; then
            echo "Recovery baseline is incomplete: $recovery_dir/$required_name" >&2
            exit 1
        fi
    done
done
for fresh_target in \
    "$RESULT_ROOT" \
    "$MODEL_LOG_ROOT" \
    "$STAGING_VISUAL_ROOT" \
    "$SINGLE_FINAL_VISUAL" \
    "$FIVE_FINAL_VISUAL" \
    "$SINGLE_FINAL_COMPARISON" \
    "$FIVE_FINAL_COMPARISON"; do
    if [[ -e "$fresh_target" ]]; then
        echo "Pure-policy ablation refuses an existing target: $fresh_target" >&2
        exit 1
    fi
done

mkdir -p "$RESULT_ROOT" "$MODEL_LOG_ROOT" "$STAGING_VISUAL_ROOT" "$MPL_CONFIG_DIR"
export MPLCONFIGDIR="$MPL_CONFIG_DIR"
cd "$ROOT_DIR"

run_single_seed() {
    local result_dir="$RESULT_ROOT/single_seed_greedy_${SINGLE_SEED}"
    local visual_dir="$STAGING_VISUAL_ROOT/single_seed/pure_policy"

    CUDA_VISIBLE_DEVICES="${GPU_IDS_ARRAY[0]}" "$PYTHON_BIN" -u \
        scripts/evaluate_single_seed_greedy.py \
        --checkpoint "$CHECKPOINT" \
        --split-file "$SPLIT_FILE" \
        --split-name test \
        --global-seed "$SINGLE_SEED" \
        --target-coverage 0.99 \
        --max-steps 128 \
        --output-dir "$result_dir" \
        --visual-output-dir "$visual_dir" \
        --adapter evaluation_adapters.ariadne:create_adapter \
        --adapter-arg 'model_name="ARiADNE-Mamba-3.1-Pure-Policy"' \
        --adapter-arg model_seed=0 \
        --adapter-arg enable_recovery=false
}

run_five_seed() {
    local result_dir="$RESULT_ROOT/five_seed_greedy_100_104"
    local visual_dir="$STAGING_VISUAL_ROOT/five_seed/pure_policy"

    CUDA_VISIBLE_DEVICES="${GPU_IDS_ARRAY[1]}" "$PYTHON_BIN" -u \
        scripts/evaluate_parallel.py \
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
        --adapter-arg 'model_name="ARiADNE-Mamba-3.1-Pure-Policy"' \
        --adapter-arg model_seed=0 \
        --adapter-arg enable_recovery=false

    "$PYTHON_BIN" scripts/create_visual_report.py \
        --results-dir "$result_dir" \
        --output-dir "$visual_dir"
}

run_single_seed >"$MODEL_LOG_ROOT/single_seed_greedy_${SINGLE_SEED}.log" 2>&1 &
single_pid="$!"
echo "Launched single-seed pure-policy evaluation: pid=$single_pid, GPU=${GPU_IDS_ARRAY[0]}"

run_five_seed >"$MODEL_LOG_ROOT/five_seed_greedy_100_104.log" 2>&1 &
five_pid="$!"
echo "Launched five-seed pure-policy evaluation: pid=$five_pid, GPU=${GPU_IDS_ARRAY[1]}"

failed=0
if wait "$single_pid"; then
    echo "Single-seed pure-policy evaluation succeeded"
else
    echo "Single-seed pure-policy evaluation failed" >&2
    failed=1
fi
if wait "$five_pid"; then
    echo "Five-seed pure-policy evaluation succeeded"
else
    echo "Five-seed pure-policy evaluation failed" >&2
    failed=1
fi
if [[ "$failed" -ne 0 ]]; then
    echo "At least one ablation failed; existing visualizations were not changed." >&2
    exit 1
fi

single_results="$RESULT_ROOT/single_seed_greedy_${SINGLE_SEED}"
five_results="$RESULT_ROOT/five_seed_greedy_100_104"

"$PYTHON_BIN" scripts/aggregate_results.py \
    "$single_results" \
    "$SINGLE_RECOVERY_RESULTS" \
    --output-dir "$STAGING_VISUAL_ROOT/single_seed/comparison_pure_vs_recovery_v4" \
    --bootstrap-seed 42 \
    --bootstrap-samples 10000

"$PYTHON_BIN" scripts/aggregate_results.py \
    "$five_results" \
    "$FIVE_RECOVERY_RESULTS" \
    --output-dir "$STAGING_VISUAL_ROOT/five_seed/comparison_pure_vs_recovery_v4" \
    --bootstrap-seed 42 \
    --bootstrap-samples 10000

"$PYTHON_BIN" -c '
import csv
import json
import sys
from pathlib import Path

checks = [
    (Path(sys.argv[1]), 1133, 1, [4777]),
    (Path(sys.argv[2]), 5665, 5, [100, 101, 102, 103, 104]),
]
for result_dir, expected_rows, episodes_per_map, eval_seeds in checks:
    with (result_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows:
        raise SystemExit(
            f"{result_dir}: expected {expected_rows} episodes, found {len(rows)}"
        )
    metadata = json.loads((result_dir / "evaluation_metadata.json").read_text())
    if metadata.get("deterministic") is not True:
        raise SystemExit(f"{result_dir}: deterministic is not true")
    if int(metadata.get("episodes_per_map", -1)) != episodes_per_map:
        raise SystemExit(f"{result_dir}: unexpected episodes_per_map")
    if metadata.get("eval_seeds") != eval_seeds:
        raise SystemExit(f"{result_dir}: unexpected eval_seeds")
    if metadata.get("adapter_args", {}).get("enable_recovery") is not False:
        raise SystemExit(f"{result_dir}: recovery is not disabled in adapter_args")
    if metadata.get("adapter_metadata", {}).get("recovery_enabled") is not False:
        raise SystemExit(f"{result_dir}: recovery is not disabled in adapter metadata")
    if any(row.get("recovery_enabled") != "False" for row in rows):
        raise SystemExit(f"{result_dir}: an episode records recovery enabled")
    if any(int(row.get("recovery_trigger_count", "0")) != 0 for row in rows):
        raise SystemExit(f"{result_dir}: an episode records a recovery trigger")
' "$single_results" "$five_results"

for chart_dir in \
    "$STAGING_VISUAL_ROOT/single_seed/pure_policy" \
    "$STAGING_VISUAL_ROOT/five_seed/pure_policy"; do
    chart_count=$(find "$chart_dir" -maxdepth 1 -type f -name '*.png' | wc -l)
    if [[ "$chart_count" -ne 6 ]]; then
        echo "Expected six standard charts, found $chart_count: $chart_dir" >&2
        exit 1
    fi
done
for comparison_dir in \
    "$STAGING_VISUAL_ROOT/single_seed/comparison_pure_vs_recovery_v4" \
    "$STAGING_VISUAL_ROOT/five_seed/comparison_pure_vs_recovery_v4"; do
    if [[ ! -f "$comparison_dir/overall_summary.json" ]]; then
        echo "Comparison output is incomplete: $comparison_dir" >&2
        exit 1
    fi
done

mkdir -p "$SINGLE_VISUAL_PARENT" "$FIVE_VISUAL_PARENT"
mv "$STAGING_VISUAL_ROOT/single_seed/pure_policy" "$SINGLE_FINAL_VISUAL"
mv "$STAGING_VISUAL_ROOT/single_seed/comparison_pure_vs_recovery_v4" "$SINGLE_FINAL_COMPARISON"
mv "$STAGING_VISUAL_ROOT/five_seed/pure_policy" "$FIVE_FINAL_VISUAL"
mv "$STAGING_VISUAL_ROOT/five_seed/comparison_pure_vs_recovery_v4" "$FIVE_FINAL_COMPARISON"
rmdir "$STAGING_VISUAL_ROOT/single_seed" "$STAGING_VISUAL_ROOT/five_seed" "$STAGING_VISUAL_ROOT"

echo "Mamba 3.1 pure-policy greedy ablation complete."
echo "Results: $RESULT_ROOT"
echo "Single-seed visuals: $SINGLE_FINAL_VISUAL"
echo "Five-seed visuals: $FIVE_FINAL_VISUAL"

