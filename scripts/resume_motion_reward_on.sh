#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/experiments/motion_reward_20260912"
trap 'echo FAILED > "$OUT/recovery_on.status"' ERR
echo RUNNING > "$OUT/recovery_on.status"
export CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python -u \
    "$ROOT/scripts/train_pose_original_parallel.py" \
    --recovery on --features off --motion-delta on --map-delta off --reward on \
    --seed 0 --resume --until-stopped --output-dir "$OUT/recovery_on" \
    >> "$OUT/recovery_on.log" 2>&1
echo STOPPED > "$OUT/recovery_on.status"
