#!/usr/bin/env python3
"""Fresh delta-local experiments using the original five-worker Ray SAC loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trigger-steps', type=int, default=2)
    p.add_argument('--point-repeat-trigger-steps', type=int, default=None)
    p.add_argument('--split-file', type=Path, default=ROOT.parent/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json')
    p.add_argument('--recovery', choices=('off', 'on'), required=True)
    p.add_argument('--action-features', choices=('on', 'off'), default=None,
                   help='Target-node feature override; defaults to --features')
    p.add_argument('--features', choices=('on', 'off'), default='on')
    p.add_argument('--motion-delta', choices=('on', 'off'), default='on')
    p.add_argument('--frontier-delta', choices=('on', 'off'), default='on')
    p.add_argument('--map-delta', choices=('on', 'off'), default='on')
    p.add_argument('--reward', choices=('on', 'off'), default='on')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gifs-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--until-stopped', action='store_true', help='Disable automatic episode and environment-step stopping')
    p.add_argument('--total-episodes', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-env-steps', type=int, default=1000000)
    a = p.parse_args()
    if a.point_repeat_trigger_steps is None:
        a.point_repeat_trigger_steps = a.trigger_steps
    if min(a.trigger_steps, a.point_repeat_trigger_steps) < 1:
        raise ValueError('Trigger streaks must be positive')
    if a.action_features is None:
        a.action_features = a.features
    if a.until_stopped:
        a.total_episodes = 0
        a.total_env_steps = 0
    if a.seed < 0 or a.total_episodes < 0 or (not a.until_stopped and a.total_episodes == 0 and a.total_env_steps <= 0):
        raise ValueError('invalid seed/budget')
    output = a.output_dir.resolve()
    if output.exists() and not a.resume:
        raise FileExistsError(f'fresh training refuses existing output: {output}')
    gifs = (
        a.gifs_dir.resolve()
        if a.gifs_dir
        else ROOT/'gifs'/f'recovery_{a.recovery}'/output.name
    )
    if gifs.exists() and not a.resume:
        raise FileExistsError(f'fresh training refuses existing GIF output: {gifs}')
    split = a.split_file.resolve()
    manifest = json.loads(split.read_text())
    if not manifest['train_maps'] or set(manifest['train_maps']) & set(manifest['test_maps']):
        raise ValueError('invalid training split')
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expose exactly two GPUs: first rollout, second learner')
    if a.resume:
        previous = json.loads((output/'run_config.json').read_text())
        if not (output/'model/checkpoint.pth').is_file():
            raise FileNotFoundError('Resume checkpoint missing')
        if previous['split_sha256'] != hashlib.sha256(split.read_bytes()).hexdigest():
            raise ValueError('Training split changed')
        if (previous['hyperparameters']['LOCAL_REVISIT_TRIGGER_STEPS'] != a.trigger_steps or
                previous['hyperparameters']['POINT_REPEAT_TRIGGER_STEPS'] != a.point_repeat_trigger_steps):
            raise ValueError('Resume recovery trigger streak mismatch')
        for key, value in dict(training_seed=a.seed, recovery=a.recovery, pose_encoding='delta_local',
                               features=a.features, action_features=a.action_features, motion_delta=a.motion_delta,
                               frontier_delta=a.frontier_delta, map_delta=a.map_delta, reward=a.reward).items():
            default = previous.get('features', 'on') if key == 'action_features' else ('on' if key in ('features', 'frontier_delta') else None)
            if previous.get(key, default) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
    output.mkdir(parents=True, exist_ok=a.resume)
    ray_temp = tempfile.mkdtemp(prefix=f'ray-pose-{a.recovery}-', dir='/tmp')
    os.environ.update(
        ARIADNE_RECOVERY_TRIGGER_STEPS=str(a.trigger_steps),
        ARIADNE_POINT_REPEAT_TRIGGER_STEPS=str(a.point_repeat_trigger_steps),
        ARIADNE_OUTPUT_ROOT=str(output), ARIADNE_RUN_NAME=f'delta_local_recovery_{a.recovery}',
        ARIADNE_RANDOM_SEED=str(a.seed), ARIADNE_LOAD_MODEL=str(int(a.resume)),
        ARIADNE_LEARNER_GPU_INDEX='1', ARIADNE_USE_RECOVERY=str(int(a.recovery == 'on')),
        ARIADNE_MAP_SPLIT_FILE=str(split), ARIADNE_MAP_SPLIT_NAME='train',
        ARIADNE_TEMPORAL_POSE_ENCODING='delta_local', ARIADNE_TEMPORAL_DELTA_POSITION_SCALE='4',
        ARIADNE_ALLOW_TEMPORAL_POSE_OVERRIDE='0', ARIADNE_TOKEN_FEATURES=str(int(a.features == 'on')),
        ARIADNE_TOKEN_BEHAVIOR_REWARD='1', ARIADNE_TOKEN_MAP_DELTA=str(int(a.map_delta == 'on')),
        ARIADNE_TOKEN_ACTION_FEATURES=str(int(a.action_features == 'on')),
        ARIADNE_TOKEN_FRONTIER_DELTA=str(int(a.frontier_delta == 'on')),
        ARIADNE_TOKEN_REWARD=str(int(a.reward == 'on')),
        ARIADNE_TOKEN_MOTION_DELTA=str(int(a.motion_delta == 'on')),
        ARIADNE_RAY_TEMP_DIR=ray_temp, ARIADNE_TOTAL_ENV_STEPS=str(0 if a.total_episodes else a.total_env_steps),
        ARIADNE_TOTAL_EPISODES=str(a.total_episodes),
        ARIADNE_MEASURE_DECISIONS='1', MPLCONFIGDIR=str(output/'matplotlib'),
        ARIADNE_GIFS_PATH=str(gifs),
    )
    os.chdir(ROOT)
    torch.set_num_threads(4)
    import parameter as config
    import driver
    assert (config.NUM_META_AGENT, config.RUNNER_GPU_FRACTION,
            config.RAY_ROLLOUT_GPUS, config.LEARNER_GPU_INDEX,
            config.LEARNER_UPDATES_PER_TRAIN) == (5, .2, 1, 1, 8)
    assert config.USE_BF16 and not config.CHECK_TRAIN_FINITE
    saved = dict(regional_trigger_steps=a.trigger_steps, point_repeat_trigger_steps=a.point_repeat_trigger_steps, exploration_progress_signal='new_free_cell_count_v1', training_seed=a.seed, pose_encoding='delta_local', recovery=a.recovery,
        features=a.features, action_features=a.action_features, motion_delta=a.motion_delta, frontier_delta=a.frontier_delta, map_delta=a.map_delta, reward=a.reward,
        gifs_dir=str(gifs),
        initialization='resume_from_checkpoint' if a.resume else 'from_scratch', collector='original_driver_five_ray_workers',
        total_env_steps=0 if a.total_episodes else a.total_env_steps, total_episodes=a.total_episodes,
        resumed=a.resume, until_stopped=a.until_stopped,
        budget_boundary='manual stop only' if a.until_stopped else ('exact completed episodes' if a.total_episodes else 'completed_episode; overshoot at most 127'),
        train_map_count=len(manifest['train_maps']), split_file=str(split),
        split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), ray_temp_dir=ray_temp,
        timing='CUDA-synchronized action selection including recovery detection/planning; excludes Env.step and graph update',
        hyperparameters={k:v for k,v in vars(config).items()
                         if k.isupper() and isinstance(v,(int,float,bool,str))},
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in list(ROOT.glob('*.py'))+list((ROOT/'models').glob('*.py'))})
    if a.resume:
        import time
        (output/f'run_config.before_resume_{time.time_ns()}.json').write_text(json.dumps(previous,indent=2))
    (output/'run_config.json').write_text(json.dumps(saved,indent=2))
    print(json.dumps(saved,indent=2),flush=True)
    driver.main()


if __name__ == '__main__':
    main()
