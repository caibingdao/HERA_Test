"""Evaluate trained on/off policies on GPUs 0/1; replace fixed-start results after validation."""
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
BASE = REPO / 'results'
STAGE = BASE / 'cache' / 'seed4777_seeded_pair_20261017'
SPLIT = ROOT / 'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json'
REFERENCE = ROOT / 'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_original/episodes.csv'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    STAGE.mkdir(parents=True, exist_ok=False)
    logs = BASE / 'logs/full_test'
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gpu, recovery in enumerate(('on', 'off')):
        run = 'token21_local_motion_10k_seed4777' + ('_recovery_off' if recovery == 'off' else '') + '_20260928'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        log = (logs / f'recovery_{recovery}_single_seed_4777_seeded.log').open('w')
        command = [sys.executable, '-u', str(REPO/'scripts/run_full_test_eval.py'),
                   '--checkpoint', str(ROOT/'training_runs'/run/'model/checkpoint.pth'),
                   '--split-file', str(SPLIT), '--recovery', recovery,
                   '--global-seed', '4777', '--output-dir', str(STAGE/f'recovery_{recovery}'),
                   '--visual-output-dir', str(STAGE/f'visuals_{recovery}')]
        jobs.append((recovery, subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f'Started recovery {recovery} on physical GPU {gpu}', flush=True)
    codes = []
    for recovery, process, log in jobs:
        code = process.wait()
        log.close()
        codes.append(code)
        print(f'recovery {recovery} exited {code}', flush=True)
    if any(codes):
        raise RuntimeError('Evaluation failed; existing fixed-start results were not replaced')
    expected = json.loads(SPLIT.read_text())['test_maps']
    reference = {r['map_id']: r for r in read(REFERENCE)}
    for recovery in ('on', 'off'):
        directory = STAGE/f'recovery_{recovery}'
        rows = read(directory/'episodes.csv')
        assert len(rows) == len(expected) == 1133
        assert [r['map_id'] for r in rows] == expected
        for row in rows:
            assert row['eval_seed'] == row['start_position_seed'] == '4777'
            assert not row.get('error', '').strip(), row
            old = reference[row['map_id']]
            assert all(row[k] == old[k] for k in ('start_cell_x', 'start_cell_y', 'start_resample_count')), row['map_id']
        meta_path = directory/'evaluation_metadata.json'
        meta = json.loads(meta_path.read_text())
        assert meta['deterministic'] and meta['action_selection'] == 'greedy_argmax'
        meta.update(start_position_mode='seeded_map_marker_selection', start_seed_applied=True,
                    start_validation_reference=str(REFERENCE), validated_start_count=len(rows), physical_gpu=0 if recovery=='on' else 1)
        meta_path.write_text(json.dumps(meta, indent=2)+'\n')
        success = [r for r in rows if r['success'].lower() == 'true']
        summary = dict(model_version='full_test', recovery=recovery, seed=4777,
                       episodes=len(rows), successes=len(success), success_rate=len(success)/len(rows),
                       mean_path_all_m=statistics.mean(float(r['path_length']) for r in rows),
                       mean_path_success_m=statistics.mean(float(r['path_length']) for r in success) if success else None,
                       mean_steps=statistics.mean(float(r['steps']) for r in rows))
        (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        with (directory/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    # Generate the new comparison before touching any existing published results.
    subprocess.run([sys.executable, str(REPO/'scripts/compare_full_test.py'), '--results-root', str(STAGE)], check=True)
    replacements = []
    for recovery in ('on', 'off'):
        replacements.extend([
            (STAGE/f'recovery_{recovery}', BASE/f'data/full_test/recovery_{recovery}/single_seed_4777'),
            (STAGE/f'visuals_{recovery}', BASE/f'visualizations/full_test/recovery_{recovery}/single_seed_4777')])
    comparison_data = STAGE/'comparison_data'
    comparison_visual = STAGE/'comparison_visual'
    comparison_data.mkdir()
    comparison_visual.mkdir()
    for name in ('comparison_summary.csv', 'comparison_summary.json', 'paired_map_comparison.csv'):
        shutil.move(str(STAGE/name), comparison_data/name)
    shutil.move(str(STAGE/'recovery_on_off_comparison.png'), comparison_visual/'recovery_on_off_comparison.png')
    replacements.extend([(comparison_data, BASE/'data/full_test/comparison_single_seed_4777'),
                         (comparison_visual, BASE/'visualizations/full_test/comparison_single_seed_4777')])
    backups = []
    try:
        for source, destination in replacements:
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = STAGE/f'previous_{len(backups)}'
            if destination.exists(): destination.rename(backup)
            backups.append((destination, backup))
            source.rename(destination)
    except Exception:
        for destination, backup in reversed(backups):
            if destination.exists(): shutil.rmtree(destination)
            if backup.exists(): backup.rename(destination)
        raise
    for _, backup in backups:
        if backup.exists(): shutil.rmtree(backup)
    print('Both 1133-map seeded greedy evaluations validated; fixed-start results and comparisons replaced.', flush=True)


if __name__ == '__main__':
    main()
