"""Read-only full 1133-map, seed4777 greedy evaluation of final HERA checkpoints."""
import argparse,csv,hashlib,json,os,subprocess,sys,time,statistics
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
SPLIT=ROOT/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json'
REFERENCE=ROOT/'ARiADNE_MAMBA_3.1_TOKEN21/results/data/ariadne/single_seed_greedy_4777/episodes.csv'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def paths(streak, suffix='', result_name=None, recovery='on'):
    name=result_name or f'recovery_streak{streak}{suffix}'
    return (REPO/'results/data'/name/f'recovery_{recovery}'/'single_seed_4777',REPO/'results/visualizations'/name/f'recovery_{recovery}'/'single_seed_4777')

def compare(suffix='', training_episodes=10000):
    dirs={s:paths(s,suffix)[0] for s in [3,2]}
    if not all((d/'publication_complete.json').exists() for d in dirs.values()):return
    rows={s:list(csv.DictReader((d/'episodes.csv').open())) for s,d in dirs.items()}
    by={s:{r['map_id']:r for r in rs} for s,rs in rows.items()};pair=[]
    for name in json.loads(SPLIT.read_text())['test_maps']:
        a,b=by[3][name],by[2][name]
        assert all(a[k]==b[k] for k in ['start_cell_x','start_cell_y','eval_seed'])
        pair.append(dict(map_id=name,streak3_success=a['success'],streak2_success=b['success'],
            streak3_path_length=float(a['path_length']),streak2_path_length=float(b['path_length']),
            path_delta_2_minus_3=float(b['path_length'])-float(a['path_length']),
            streak3_coverage=float(a['final_coverage']),streak2_coverage=float(b['final_coverage'])))
    dest=REPO/'results/comparison'/f'recovery_streak3_vs_streak2{suffix}_seed4777';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'paired_maps.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pair[0]));w.writeheader();w.writerows(pair)
    summaries=[json.loads((dirs[s]/'hera_summary.json').read_text()) for s in [3,2]]
    (dest/'summary.json').write_text(json.dumps(summaries,indent=2))
    (dest/'README.md').write_text(f'# HERA 连续门槛测试对照\n\n两个独立训练模型，各{training_episodes}训练回合；均开启recovery。1133测试地图，seed4777，greedy，相同起点，目标覆盖率99%，最多128步。paired_maps.csv为逐图配对，summary.json为重算汇总。差异包含训练轨迹及权重变化，不是相同checkpoint仅改变测试门槛的纯消融。\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--streak',type=int,choices=[2,3],required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--training-episodes',type=int,default=10000)
    p.add_argument('--result-suffix',default='')
    p.add_argument('--result-name')
    p.add_argument('--recovery',choices=['on','off'],default='on')
    a=p.parse_args()
    if a.result_name and any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.result_name):
        raise ValueError('Invalid result name')
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.result_suffix):
        raise ValueError('Invalid result suffix')
    import torch
    checkpoint=a.checkpoint or REPO/'experiments'/f'recovery_streak{a.streak}_seed4777_10k_20261024/model/checkpoint.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved.get('is_final') and saved.get('completed_episodes')==a.training_episodes
    assert saved.get('exploration_progress_signal')=='new_free_cell_count_v1'
    cfg=saved['recovery_trigger_config'];assert cfg['regional_trigger_steps']==cfg['point_repeat_trigger_steps']==a.streak
    del saved
    # Refresh GPU choice after training releases its workers and learner.
    gpu_inventory = None
    if os.environ.get('HERA_FIXED_TEST_GPU'):
        a.gpu = os.environ['HERA_FIXED_TEST_GPU']
    elif a.result_name and a.result_name.startswith('ablation_'):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
        gpu_inventory = []
        for line in output.strip().splitlines():
            index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
            gpu_inventory.append(dict(index=int(index), uuid=uuid, memory_used=int(memory), utilization=int(utilization)))
        selected = min(gpu_inventory, key=lambda item: (item['memory_used'], item['utilization'], item['index']))
        a.gpu = selected['uuid']
        print('AUTO_SELECTED_TEST_GPU', json.dumps(selected), flush=True)
    data,visual=paths(a.streak,a.result_suffix,a.result_name,a.recovery);identity=sha(checkpoint)
    result_name=a.result_name or f'recovery_streak{a.streak}{a.result_suffix}'
    record=REPO/'results/runs'/f'{result_name}_seed4777.json'
    launch=dict(streak=a.streak,training_episodes=a.training_episodes,checkpoint=str(checkpoint),checkpoint_sha256=identity,gpu=a.gpu,gpu_inventory=gpu_inventory,
        split=str(SPLIT),split_sha256=sha(SPLIT),maps=1133,global_seed=4777,action_selection='greedy_argmax',
        target_coverage=.99,max_steps=128,recovery=a.recovery,recovery_config=cfg,output=str(data),visual=str(visual),started_unix=time.time())
    if data.exists() or visual.exists():raise FileExistsError('Refusing to overwrite evaluation output')
    record.write_text(json.dumps(launch,indent=2));print(json.dumps(launch,indent=2),flush=True)
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpu,PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        ARIADNE_RECOVERY_TRIGGER_STEPS=str(a.streak),ARIADNE_POINT_REPEAT_TRIGGER_STEPS=str(a.streak))
    command=[sys.executable,'-u',str(REPO/'scripts/run_full_test_eval.py'),'--checkpoint',str(checkpoint),
        '--split-file',str(SPLIT),'--output-dir',str(data),'--visual-output-dir',str(visual),
        '--recovery',a.recovery,'--model-version',result_name,
        '--global-seed','4777','--target-coverage','0.99','--max-steps','128']
    result=subprocess.run(command,env=env)
    if result.returncode:raise RuntimeError(f'Evaluation failed: {result.returncode}')
    assert sha(checkpoint)==identity,'checkpoint changed during evaluation'
    m=json.loads((data/'evaluation_metadata.json').read_text());rows=list(csv.DictReader((data/'episodes.csv').open()))
    manifest=json.loads(SPLIT.read_text());reference={r['map_id']:r for r in csv.DictReader(REFERENCE.open())}
    assert len(rows)==len({r['map_id'] for r in rows})==1133
    assert {r['map_id'] for r in rows}==set(manifest['test_maps'])
    assert m['eval_seeds']==[4777] and m['evaluation_mode']=='single_global_seed_greedy'
    assert m['target_coverage']==.99 and m['max_steps']==128 and m['split_manifest_sha256']==sha(SPLIT)
    actual=m['adapter_metadata']['recovery_configuration']
    assert m['adapter_metadata']['recovery_enabled'] is (a.recovery=='on')
    if a.recovery=='on':
        assert actual['trigger_steps']==actual['point_repeat_trigger_steps']==a.streak
    else:
        assert actual is None
        assert all(int(r.get('recovery_trigger_count',0))==int(r.get('recovery_steps',0))==0 for r in rows)
    for r in rows:
        assert int(r['eval_seed'])==4777
        assert all(r[k]==reference[r['map_id']][k] for k in ['start_cell_x','start_cell_y'])
    assert not any(r.get('error') for r in rows),'One or more maps failed with an error'
    successful=[r for r in rows if r['success'].lower()=='true']
    summary=dict(streak=a.streak,recovery=a.recovery,training_episodes=a.training_episodes,maps=1133,successes=len(successful),success_rate=len(successful)/1133,
        mean_coverage=statistics.mean(float(r['final_coverage']) for r in rows),
        mean_path_all=statistics.mean(float(r['path_length']) for r in rows),
        mean_path_success=statistics.mean(float(r['path_length']) for r in successful) if successful else None,
        mean_steps=statistics.mean(float(r['steps']) for r in rows),
        recovery_triggers=sum(int(r.get('recovery_trigger_count',0)) for r in rows),
        checkpoint_sha256=identity,paired_starts_validated=1133)
    (data/'hera_summary.json').write_text(json.dumps(summary,indent=2))
    (data/'publication_complete.json').write_text(json.dumps(dict(finished_unix=time.time(),validated_maps=1133),indent=2))
    if not a.result_name and a.recovery=='on':compare(a.result_suffix,a.training_episodes)
    print('HERA_EVALUATION_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
