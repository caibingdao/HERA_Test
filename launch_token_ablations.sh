#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

python_bin="${ARIADNE_PYTHON:-/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python}"
seed="${ARIADNE_ABLATION_SEED:-4777}"
split_file="$(realpath "$script_dir/../ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json")"
log_dir="$script_dir/ablation_logs"
mkdir -p "$log_dir"

launch_ablation() {
    local gpu="$1"
    local run_name="$2"
    local features="$3"
    local behavior_reward="$4"
    local map_delta="$5"
    local log_file="$log_dir/$run_name.log"
    local unit_name="ariadne-${run_name//_/-}"

    if systemctl --user is-active --quiet "$unit_name.service"; then
        echo "$run_name is already running in $unit_name.service" >&2
        return 0
    fi

    systemd-run --user --collect \
        --unit="$unit_name" \
        --property="WorkingDirectory=$script_dir" \
        --property=KillMode=mixed \
        --property="StandardOutput=append:$log_file" \
        --property="StandardError=append:$log_file" \
        --setenv="CUDA_VISIBLE_DEVICES=$gpu" \
        --setenv="RAY_TMPDIR=/tmp/ariadne_ray_$gpu" \
        --setenv="MPLCONFIGDIR=/tmp/matplotlib_$run_name" \
        --setenv="ARIADNE_RUN_NAME=$run_name" \
        --setenv=ARIADNE_LOAD_MODEL=0 \
        --setenv=ARIADNE_LEARNER_GPU_INDEX=0 \
        --setenv="ARIADNE_RANDOM_SEED=$seed" \
        --setenv="ARIADNE_MAP_SPLIT_FILE=$split_file" \
        --setenv=ARIADNE_MAP_SPLIT_NAME=train \
        --setenv="ARIADNE_TOKEN_FEATURES=$features" \
        --setenv="ARIADNE_TOKEN_BEHAVIOR_REWARD=$behavior_reward" \
        --setenv="ARIADNE_TOKEN_MAP_DELTA=$map_delta" \
        --setenv=ARIADNE_USE_RECOVERY=0 \
        "$python_bin" -u driver.py
    echo "$run_name gpu=$gpu unit=$unit_name.service log=$log_file"
}

# One semantic token group is disabled in each run. Physical GPU 3 remains
# unused so the three ablations do not consume all four server GPUs.
launch_ablation 0 mamba31_ablation_no_features 0 1 1
launch_ablation 1 mamba31_ablation_no_behavior_reward 1 0 1
launch_ablation 2 mamba31_ablation_no_map_delta 1 1 0
