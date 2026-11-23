"""Read-only training snapshot: never signal or pause either trainer."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


def summarize(rows):
    if not rows:
        return {"episodes": 0}
    counts = sum(int(r['decision_count']) for r in rows)
    warm = sum(int(r['warm_decision_count']) for r in rows)
    success = [r for r in rows if r['success'] in ('True', '1')]
    return dict(episodes=len(rows), env_steps=int(rows[-1]['env_steps']),
        learner_updates=int(rows[-1]['learner_updates']),
        mean_decision_ms=sum(float(r['decision_total_ms']) for r in rows)/counts,
        mean_decision_ms_excluding_episode_first=(
            sum(float(r['warm_decision_total_ms']) for r in rows)/warm if warm else None),
        mean_episode_steps=counts/len(rows), completion_rate=len(success)/len(rows),
        mean_successful_episode_steps=(sum(int(r['episode_steps']) for r in success)/len(success)
                                       if success else None),
        mean_coverage=sum(float(r['coverage']) for r in rows)/len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    runs = {}
    keys = ('delta_local_pose', 'delta_local_pose_recovery_on')
    for key in keys:
        path = args.directory/key/'episodes.csv'
        with path.open() as stream:
            # Ignore an incomplete final line if read during a concurrent write.
            lines = stream.read().splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'):
            lines.pop()
        runs[key] = list(csv.DictReader(lines))
    n = min(map(len, runs.values()))
    matched = {key: summarize(rows[:n]) for key, rows in runs.items()}
    report = dict(timestamp=datetime.now().astimezone().isoformat(),
        scope='ongoing stochastic training rollouts; not fixed-checkpoint evaluation',
        timing='CUDA-synchronized cached policy action selection, excluding Env.step, graph update, token finalization, SAC and GIF',
        matching='first N complete episodes per model; map/seed checked; training policies evolve',
        latest={key:summarize(rows) for key,rows in runs.items()}, matched=matched)
    report['model_order'] = keys
    for a,b in zip(runs[keys[0]][:n],runs[keys[1]][:n]):
        if (a['map_path'],a['seed']) != (b['map_path'],b['seed']):
            raise ValueError('episode map/seed mismatch')
    if n:
        a,b = matched.values()
        report['recovery_on_minus_off'] = {
            k: dict(difference=b[k]-a[k], percent=(b[k]/a[k]-1)*100 if a[k] else None)
            for k in ('mean_decision_ms','mean_episode_steps','completion_rate')}
    out = args.directory/'comparison'
    out.mkdir(exist_ok=True)
    filename = out/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    with filename.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
    print('Snapshot:', filename)


if __name__ == '__main__':
    main()
