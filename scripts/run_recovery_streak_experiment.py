"""Fresh 10k HERA experiment with isolated trigger streaks and full artifacts."""
import argparse,csv,hashlib,json,os,subprocess,sys,time,shutil,statistics
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];WORKSPACE=REPO.parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--streak',type=int,choices=[2,3],required=True);parser.add_argument('--gpus',required=True)
    args=parser.parse_args();name=f'recovery_streak{args.streak}_seed4777_10k_20261024'
    run=REPO/'experiments'/name;control=run.with_name(name+'_control');gifs=REPO/'gifs'/name
    if run.exists() or gifs.exists():raise FileExistsError('Fresh-only experiment refuses existing run/artifacts')
    control.mkdir(exist_ok=True)
    split=WORKSPACE/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json'
    manifest=json.loads(split.read_text());assert len(manifest['train_maps'])==4530
    if len(args.gpus.split(','))!=2:raise ValueError('Need rollout GPU then learner GPU')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=args.gpus,
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        ARIADNE_MAP_DIR=str(WORKSPACE/'ARiADNE_MAMBA_3.1_TOKEN21/maps'),
        ARIADNE_DISABLE_GIFS='0',ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS='1',
        ARIADNE_WRITE_RECOVERY_DIAGNOSTICS='1')
    launch=dict(streak=args.streak,seed=4777,episodes=10000,initialization='from_scratch',recovery='on',
        physical_gpu_uuids=args.gpus.split(','),split=str(split),split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        training_maps=4530,run=str(run),gifs=str(gifs),started_unix=time.time())
    (control/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch,indent=2),flush=True)
    command=[sys.executable,'-u',str(REPO/'scripts/train_pose_original_parallel.py'),
        '--recovery','on','--trigger-steps',str(args.streak),'--point-repeat-trigger-steps',str(args.streak),
        '--split-file',str(split),'--seed','4777','--total-episodes','10000','--total-env-steps','0',
        '--features','on','--action-features','on','--motion-delta','on','--map-delta','on','--reward','on',
        '--output-dir',str(run),'--gifs-dir',str(gifs)]
    with (control/'train.log').open('w') as stream:
        process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while not (run/'episodes.csv').exists() and process.poll() is None:time.sleep(2)
        plot=None
        if process.poll() is None:
            with (control/'plots.log').open('w') as plotlog:
                plot=subprocess.Popen([sys.executable,'-u',str(REPO/'scripts/plot_training_progress.py'),
                    '--run-dir',str(run),'--total-episodes','10000','--streak',str(args.streak),'--watch'],
                    env=env,stdout=plotlog,stderr=subprocess.STDOUT)
        code=process.wait()
    (control/'exit_status.json').write_text(json.dumps({'returncode':code,'finished_unix':time.time()},indent=2))
    if plot is not None:
        try:plot.wait(timeout=60)
        except subprocess.TimeoutExpired:plot.terminate()
    if run.exists():
        shutil.copy2(control/'launch.json',run/'launch.json');shutil.copy2(control/'train.log',run/'train.log')
    if code:raise RuntimeError(f'Training exited {code}; see {control}/train.log')
    import torch
    cp=torch.load(run/'model/checkpoint.pth',map_location='cpu',weights_only=False)
    assert cp.get('is_final') and cp.get('completed_episodes')==10000
    assert cp['recovery_trigger_config']['regional_trigger_steps']==args.streak
    assert cp['recovery_trigger_config']['point_repeat_trigger_steps']==args.streak
    rows=list(csv.DictReader((run/'episodes.csv').open()));assert len(rows)==10000
    summary=dict(completed_episodes=len(rows),streak=args.streak,successes=sum(r['success']=='True' for r in rows),
        coverage_ge99=sum(float(r['coverage'])>=.99 for r in rows),mean_coverage=statistics.mean(float(r['coverage']) for r in rows),
        recovery_triggers=sum(int(r['recovery_triggers']) for r in rows),recovery_steps=sum(int(r['recovery_steps']) for r in rows),
        total_steps=sum(int(r['episode_steps']) for r in rows),finished_unix=time.time())
    (run/'complete.json').write_text(json.dumps(summary,indent=2));print('TRAINING_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
