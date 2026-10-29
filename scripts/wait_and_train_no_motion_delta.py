#!/usr/bin/env python3
"""Wait for single-seed evaluators to release GPUs 0/2, then exec training."""
import os
from pathlib import Path
import sys
import time

EVALUATORS = (1018650, 1018652)
ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777'
OUTPUT = ROOT/'experiments/pose_ablation_no_motion_delta_20260908/recovery_off'
GIFS = ROOT.parent/'ARiADNE_MAMBA_Test/mamba31_map_split/gifs/ariadne_mamba_new_token/no_motion_delta_recovery_off'
LOG = ROOT/'experiments/pose_ablation_no_motion_delta_20260908.launch.log'


def alive(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
    except FileNotFoundError:
        return False
    return b'evaluate_single_seed_greedy.py' in command


while any(alive(pid) for pid in EVALUATORS):
    time.sleep(30)
for model in ('recovery_off', 'recovery_on'):
    if not (TEST_ROOT/model/'evaluation_metadata.json').is_file():
        raise RuntimeError(f'single-seed evaluation did not complete: {model}')
if OUTPUT.exists() or GIFS.exists():
    raise FileExistsError(f'fresh targets must not exist: {OUTPUT}, {GIFS}')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,2', OMP_NUM_THREADS='4',
           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
command = [sys.executable, '-u', str(ROOT/'scripts/train_pose_original_parallel.py'),
           '--recovery', 'off', '--motion-delta', 'off', '--seed', '0',
           '--total-env-steps', '1000000', '--output-dir', str(OUTPUT),
           '--gifs-dir', str(GIFS)]
with LOG.open('x') as stream:
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)
