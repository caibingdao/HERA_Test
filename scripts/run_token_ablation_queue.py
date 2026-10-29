"""Sequential fresh 7500-episode token ablations with automatic full evaluation."""
import argparse,csv,hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
QUEUE=REPO/'experiments/token_ablation_queue_7500_seed4777_20261026'
SPLIT=ROOT/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json'
BASE=REPO/'experiments/recovery_streak2_seed4777_10k_20261024'
SPECS=[dict(name='no_target',action_features='off',motion_delta='on',map_delta='on',reward='on'),
       dict(name='no_feedback',action_features='on',motion_delta='on',map_delta='off',reward='off'),
       dict(name='no_motion',action_features='on',motion_delta='off',map_delta='on',reward='on'),
       dict(name='no_recovery',action_features='on',motion_delta='on',map_delta='on',reward='on',recovery='off',episodes=10000)]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def atomic_json(p,obj):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2));q.replace(p)

def select_gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    devices=[]
    for row in csv.reader(text.splitlines()):
        idx,uuid,used,total,util=[x.strip() for x in row]
        devices.append(dict(index=int(idx),uuid=uuid,memory_used=int(used),memory_total=int(total),utilization=int(util)))
    fixed=os.environ.get('HERA_FIXED_GPU_INDICES')
    selected=([next(d for d in devices if d['index']==int(i)) for i in fixed.split(',')]
              if fixed else sorted(devices,key=lambda d:(d['memory_used'],d['utilization']))[:2])
    if len(selected)!=2:raise RuntimeError('Two GPUs required')
    return devices,selected

def validate_checkpoint(cp,spec):
    import torch
    d=torch.load(cp,map_location='cpu',weights_only=False)
    assert d.get('is_final') and d.get('completed_episodes')==spec.get('episodes',7500)
    assert d['exploration_progress_signal']=='new_free_cell_count_v1'
    assert d['use_mamba'] is True
    assert d['use_local_revisit_recovery'] is (spec.get('recovery','on')=='on')
    assert d['recovery_trigger_config']['regional_trigger_steps']==d['recovery_trigger_config']['point_repeat_trigger_steps']==2
    t=d['temporal_token_config']
    expected=dict(features=True,behavior_reward=True,action_features=spec['action_features']=='on',motion_delta=spec['motion_delta']=='on',map_delta=spec['map_delta']=='on',reward=spec['reward']=='on')
    for key,value in expected.items():
        actual=t.get(key,t.get('features') if key=='action_features' else None)
        assert actual is value,(key,actual,value)
    assert t['raw_input_dim']==21

def run_trial(spec, resume=False):
    episodes=spec.get('episodes',7500)
    recovery=spec.get('recovery','on')
    plan=json.loads((QUEUE/'plan.json').read_text())
    for relative,digest in plan['source_sha256'].items():
        assert sha(REPO/relative)==digest, f'Queued training source changed: {relative}'
    name=f"ablation_{spec['name']}_{episodes}_seed4777_20261026"
    run=REPO/'experiments'/name;control=run.with_name(name+'_control');gifs=REPO/'gifs'/name
    if not resume and (run.exists() or gifs.exists()):raise FileExistsError(f'Refusing existing experiment {name}')
    control.mkdir(exist_ok=True)
    inventory,gpus=select_gpus()
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        ARIADNE_MAP_DIR=str(ROOT/'ARiADNE_MAMBA_3.1_TOKEN21/maps'),ARIADNE_DISABLE_GIFS='0',
        ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS='1',ARIADNE_WRITE_RECOVERY_DIAGNOSTICS='1')
    launch=dict(spec=spec,episodes=episodes,seed=4777,recovery=(recovery=='on'),regional_trigger_steps=2,point_repeat_trigger_steps=2,
        gpu_inventory=inventory,selected_gpus=gpus,split_sha256=sha(SPLIT),baseline_checkpoint_sha256=sha(BASE/'model/checkpoint.pth'),
        baseline_checkpoint_used_for_initialization=False,started_unix=time.time(),run=str(run),gifs=str(gifs))
    atomic_json(control/'launch.json',launch)
    atomic_json(QUEUE/'status.json',dict(state='training',current=spec['name'],launch=launch,updated_unix=time.time()))
    print('START_TRAINING',spec['name'],json.dumps(launch),flush=True)
    command=[sys.executable,'-u',str(REPO/'scripts/train_pose_original_parallel.py'),
        '--split-file',str(SPLIT),'--recovery',recovery,'--trigger-steps','2','--point-repeat-trigger-steps','2',
        '--seed','4777','--total-episodes',str(episodes),'--total-env-steps','0',
        '--features','on','--action-features',spec['action_features'],'--motion-delta',spec['motion_delta'],
        '--map-delta',spec['map_delta'],'--reward',spec['reward'],'--output-dir',str(run),'--gifs-dir',str(gifs)]
    if resume:
        command.append('--resume')
        for old in ['train.log','launch.json','exit_status.json']:
            path=control/old
            if path.exists():shutil.copy2(path,control/(old+f'.before_restart_{time.time_ns()}'))
        (control/'exit_status.json').unlink(missing_ok=True)
    with (control/'train.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        while not (run/'episodes.csv').exists() and process.poll() is None:time.sleep(2)
        plot=None
        if process.poll() is None:
            with (control/'plots.log').open('w') as plotlog:
                plot=subprocess.Popen([sys.executable,'-u',str(REPO/'scripts/plot_training_progress.py'),
                    '--run-dir',str(run),'--total-episodes',str(episodes),'--streak','2','--watch'],env=env,stdout=plotlog,stderr=subprocess.STDOUT)
        code=process.wait()
    atomic_json(control/'exit_status.json',dict(returncode=code,finished_unix=time.time()))
    if plot:
        try:plot.wait(timeout=60)
        except subprocess.TimeoutExpired:plot.terminate()
    if run.exists():
        shutil.copy2(control/'train.log',run/'train.log');shutil.copy2(control/'launch.json',run/'launch.json')
    if code:raise RuntimeError(f'{name} training failed: {code}')
    checkpoint=run/'model/checkpoint.pth';validate_checkpoint(checkpoint,spec)
    rows=list(csv.DictReader((run/'episodes.csv').open()));assert len(rows)==episodes
    atomic_json(run/'complete.json',dict(completed_episodes=episodes,checkpoint_sha256=sha(checkpoint),finished_unix=time.time()))
    result_name=f"ablation_{spec['name']}_{episodes}"
    atomic_json(QUEUE/'status.json',dict(state='testing',current=spec['name'],result_name=result_name,updated_unix=time.time()))
    print('START_TESTING',spec['name'],flush=True)
    command=[sys.executable,'-u',str(REPO/'scripts/evaluate_recovery_streak.py'),
        '--streak','2','--gpu',gpus[1]['uuid'],'--checkpoint',str(checkpoint),
        '--training-episodes',str(episodes),'--result-name',result_name,'--recovery',recovery]
    with (REPO/'results/logs'/f'{result_name}_seed4777_20261026.log').open('w') as log:
        code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if code:raise RuntimeError(f'{name} evaluation failed: {code}')
    data=REPO/'results/data'/result_name/f'recovery_{recovery}'/'single_seed_4777'
    assert (data/'publication_complete.json').is_file()
    metadata=json.loads((data/'evaluation_metadata.json').read_text())
    expected=dict(features=True,behavior_reward=True,action_features=spec['action_features']=='on',motion_delta=spec['motion_delta']=='on',map_delta=spec['map_delta']=='on',reward=spec['reward']=='on')
    actual=metadata['adapter_metadata']['temporal_token_config']
    for k,v in expected.items():assert actual.get(k,actual.get('features') if k=='action_features' else None) is v,(k,actual)
    atomic_json(run/'training_and_test_complete.json',dict(completed_episodes=episodes,validated_test_maps=1133,finished_unix=time.time()))
    print('TRIAL_COMPLETE',spec['name'],flush=True)
    return dict(name=spec['name'],run=str(run),results=str(data),summary=json.loads((data/'hera_summary.json').read_text()))

def summarize(completed):
    baseline=REPO/'results/data/recovery_streak2/recovery_on/single_seed_4777'
    entries=[dict(name='baseline_streak2_10k',summary=json.loads((baseline/'hera_summary.json').read_text()))]+completed
    atomic_json(QUEUE/'comparison.json',entries)
    lines=['# HERA Token Ablation','', '基线：2步门槛、10000训练回合；三组Token消融各7500回合，完整Mamba不开recovery组为10000回合。不同训练预算需披露，不能将差异全部归因于token定义。', '',
        '|模型|训练回合|成功数/1133|平均路径m|平均决策步数|','|---|---:|---:|---:|---:|']
    for e in entries:
        s=e['summary'];lines.append(f"|{e['name']}|{10000 if e['name'].startswith('baseline') or e['name']=='no_recovery' else 7500}|{s['successes']}|{s['mean_path_all']:.3f}|{s['mean_steps']:.3f}|")
    (QUEUE/'comparison.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate-plan',action='store_true');a=p.parse_args()
    base=json.loads((BASE/'run_config.json').read_text())
    assert base['total_episodes']==10000 and base['regional_trigger_steps']==base['point_repeat_trigger_steps']==2
    assert len(json.loads(SPLIT.read_text())['train_maps'])==4530
    if a.validate_plan:
        for s in SPECS:
            print(json.dumps(dict(spec=s,episodes=s.get('episodes',7500),recovery=s.get('recovery','on')=='on',streak=2,seed=4777,initialization='from_scratch')))
        return
    if (QUEUE/'status.json').exists():raise FileExistsError('Inspect existing queue status before restarting')
    QUEUE.mkdir(exist_ok=True)
    baseline=dict(selection_criterion='mean_path_all',checkpoint=str(BASE/'model/checkpoint.pth'),checkpoint_sha256=sha(BASE/'model/checkpoint.pth'),
        training_episodes=10000,recovery=True,regional_trigger_steps=2,point_repeat_trigger_steps=2,seed=4777,
        source_test_summary=str(REPO/'results/data/recovery_streak2/recovery_on/single_seed_4777/hera_summary.json'))
    atomic_json(REPO/'baseline.json',baseline)
    atomic_json(QUEUE/'plan.json',dict(baseline=baseline,trial_order=SPECS,episodes_per_trial=7500,
        source_sha256={str(p.relative_to(REPO)):sha(p) for p in list(REPO.glob('*.py'))+list((REPO/'models').glob('*.py'))+[REPO/'scripts/train_pose_original_parallel.py']},
        scheduling='one trial trains and passes full test before next; reselect lowest-memory GPU pair for each trial',
        test=dict(maps=1133,seed=4777,greedy=True,target_coverage=.99,max_steps=128,recovery=True),
        note='no_feedback masks token reward/new-free-cell input only; recovery observes the same raw counts as baseline'))
    completed=[]
    try:
        for spec in SPECS:
            completed.append(run_trial(spec))
            summarize(completed)
            atomic_json(QUEUE/'completed_trials.json',completed)
        atomic_json(QUEUE/'status.json',dict(state='complete',completed=[e['name'] for e in completed],finished_unix=time.time()))
    except BaseException as error:
        old=json.loads((QUEUE/'status.json').read_text()) if (QUEUE/'status.json').exists() else {}
        atomic_json(QUEUE/'status.json',dict(state='failed',previous_state=old,error=str(error),failed_unix=time.time()))
        raise

if __name__=='__main__':main()
