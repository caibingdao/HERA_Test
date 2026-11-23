"""Branch from final 10k checkpoints, add exactly 2k episodes, then full test."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--streak',type=int,choices=[2,3],required=True);p.add_argument('--gpus',required=True);a=p.parse_args()
    import torch
    parent=REPO/'experiments'/f'recovery_streak{a.streak}_seed4777_10k_20261024'
    name=f'recovery_streak{a.streak}_seed4777_12k_20261025'
    run=REPO/'experiments'/name;control=run.with_name(name+'_control');gifs=REPO/'gifs'/name
    if run.exists() or gifs.exists():raise FileExistsError('Refusing to overwrite continuation output')
    control.mkdir(exist_ok=True)
    source=parent/'model/checkpoint.pth';identity=sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=False)
    assert saved.get('is_final') and saved.get('completed_episodes')==10000
    assert saved['recovery_trigger_config']['regional_trigger_steps']==a.streak
    assert saved['recovery_trigger_config']['point_repeat_trigger_steps']==a.streak
    for key in ['policy_optimizer','q_net1_optimizer','q_net2_optimizer','log_alpha_optimizer','target_q_net1_model','target_q_net2_model']:
        assert key in saved,key
    sys.path.insert(0,str(REPO))
    from training_budget import restore_progress
    complete,steps,updates,elapsed=restore_progress(saved,parent/'episodes.csv')
    assert complete==10000
    del saved
    run.mkdir();(run/'model').mkdir()
    shutil.copy2(source,run/'model/checkpoint.pth')
    for filename in ['run_config.json','episodes.csv']:shutil.copy2(parent/filename,run/filename)
    shutil.copytree(parent/'train',run/'train')
    assert sha(run/'model/checkpoint.pth')==identity
    lineage=dict(streak=a.streak,parent_run=str(parent),parent_checkpoint_sha256=identity,
        resumed_completed_episodes=10000,target_completed_episodes=12000,additional_episodes=2000,
        resumed_env_steps=steps,resumed_learner_updates=updates,gpus=a.gpus.split(','),gifs=str(gifs),started_unix=time.time(),
        restored='policy, critics, target critics, optimizers, entropy coefficient and counters',
        replay_note='Existing resume implementation rebuilds replay; replay contents and in-flight workers are not stored in checkpoint.')
    (run/'continuation.json').write_text(json.dumps(lineage,indent=2));print(json.dumps(lineage,indent=2),flush=True)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpus,
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        ARIADNE_MAP_DIR=str(ROOT/'ARiADNE_MAMBA_3.1_TOKEN21/maps'),ARIADNE_DISABLE_GIFS='0',
        ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS='1',ARIADNE_WRITE_RECOVERY_DIAGNOSTICS='1')
    command=[sys.executable,'-u',str(REPO/'scripts/train_pose_original_parallel.py'),
        '--resume','--recovery','on','--trigger-steps',str(a.streak),'--point-repeat-trigger-steps',str(a.streak),
        '--seed','4777','--total-episodes','12000','--total-env-steps','0',
        '--features','on','--action-features','on','--motion-delta','on','--map-delta','on','--reward','on',
        '--output-dir',str(run),'--gifs-dir',str(gifs)]
    with (control/'train.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        with (control/'plots.log').open('w') as plotlog:
            plot=subprocess.Popen([sys.executable,'-u',str(REPO/'scripts/plot_training_progress.py'),
                '--run-dir',str(run),'--total-episodes','12000','--streak',str(a.streak),'--watch'],
                env=env,stdout=plotlog,stderr=subprocess.STDOUT)
        code=process.wait()
    (control/'exit_status.json').write_text(json.dumps({'returncode':code,'finished_unix':time.time()},indent=2))
    try:plot.wait(timeout=60)
    except subprocess.TimeoutExpired:plot.terminate()
    shutil.copy2(control/'train.log',run/'train.log')
    if code:raise RuntimeError(f'Continuation training exited {code}')
    assert sha(source)==identity,'Parent 10k checkpoint changed'
    final=run/'model/checkpoint.pth';checkpoint=torch.load(final,map_location='cpu',weights_only=False)
    assert checkpoint.get('is_final') and checkpoint.get('completed_episodes')==12000
    assert checkpoint['recovery_trigger_config']['regional_trigger_steps']==checkpoint['recovery_trigger_config']['point_repeat_trigger_steps']==a.streak
    del checkpoint
    rows=list(csv.DictReader((run/'episodes.csv').open()));assert len(rows)==12000
    assert int(rows[-1]['completed_episodes'])==12000
    (run/'complete.json').write_text(json.dumps({'completed_episodes':12000,'additional_episodes':2000,'checkpoint_sha256':sha(final),'finished_unix':time.time()},indent=2))
    print('CONTINUATION_COMPLETE; starting 1133-map evaluation',flush=True)
    evaluation=[sys.executable,'-u',str(REPO/'scripts/evaluate_recovery_streak.py'),
        '--streak',str(a.streak),'--gpu',a.gpus.split(',')[1],'--checkpoint',str(final),
        '--training-episodes','12000','--result-suffix','_12k']
    with (REPO/'results/logs'/f'recovery_streak{a.streak}_12k_seed4777_20261025.log').open('w') as log:
        result=subprocess.run(evaluation,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Automatic evaluation exited {result.returncode}')
    (run/'training_and_test_complete.json').write_text(json.dumps({'completed_episodes':12000,'validated_test_maps':1133,'finished_unix':time.time()},indent=2))
    print('CONTINUATION_AND_TEST_COMPLETE',flush=True)

if __name__=='__main__':main()
