#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python
OUTPUT="$ROOT/experiments/motion_only_20260911"
EVAL_STATUS="$ROOT/../ARiADNE_MAMBA_Test/mamba31_map_split/experiment/logs/no_motion_delta_five_seed_stochastic_100_104_retry/status"
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing existing experiment: $OUTPUT" >&2
    exit 1
fi
mkdir -p "$OUTPUT"
echo $$ > "$OUTPUT/launcher.pid"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
run_model() {
    local recovery="$1" gpus="$2"
    local status="$OUTPUT/recovery_${recovery}.status"
    trap 'echo FAILED > "$status"' ERR
    if [[ "$recovery" == off ]]; then
        echo WAITING_FOR_STOCHASTIC_EVALUATION > "$status"
        while [[ "$(cat "$EVAL_STATUS")" != COMPLETED ]]; do
            if [[ "$(cat "$EVAL_STATUS")" == FAILED ]]; then
                echo BLOCKED_EVALUATION_FAILED > "$status"
                return 1
            fi
            sleep 30
        done
    fi
    echo RUNNING > "$status"
    CUDA_VISIBLE_DEVICES="$gpus" "$PY" -u "$ROOT/scripts/train_pose_original_parallel.py" \
        --recovery "$recovery" --features off --motion-delta on \
        --map-delta off --reward off --seed 0 \
        --total-episodes 10000 --total-env-steps 0 \
        --output-dir "$OUTPUT/recovery_${recovery}"
    "$PY" - "$OUTPUT/recovery_${recovery}/model/checkpoint.pth" <<'PY'
import sys
import torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert c['completed_episodes'] == 10000, c['completed_episodes']
assert c['is_final'] is True
assert c['temporal_token_config'] == dict(features=False, behavior_reward=True,
    motion_delta=True, map_delta=False, reward=False)
PY
    echo COMPLETED > "$status"
}
run_model on 1,3 > "$OUTPUT/recovery_on.log" 2>&1 &
on_pid=$!
run_model off 0,2 > "$OUTPUT/recovery_off.log" 2>&1 &
off_pid=$!
echo "Recovery on PID=$on_pid; recovery off scheduler PID=$off_pid"
failed=0
wait "$on_pid" || failed=1
wait "$off_pid" || failed=1
exit "$failed"
