"""GPU-aware 21D no_target ablation training and full-test supervisor, with CUDA OOM recovery."""
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
NAME='ablation_no_target_7000_seed4777_20261101'
RUN=REPO/'experiments'/NAME
CONTROL=REPO/'experiments'/(NAME+'_control')
SPLIT=ROOT/'ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json'
RESULT_NAME='ablation_no_target_7000'
RESULT=REPO/'results/data'/RESULT_NAME/'recovery_on/single_seed_4777'
GIFS=REPO/'gifs'/NAME

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def status(state,**kw):
    data=dict(state=state,updated_unix=time.time(),**kw)
    atomic(CONTROL/'status.json',data);print(json.dumps(data),flush=True)

def main():
    import torch
    CONTROL.mkdir(parents=True,exist_ok=True)
    previous=REPO/'results/data/ablation_no_recovery_10000/recovery_off/single_seed_4777/publication_complete.json'
    previous_run=REPO/'experiments/ablation_no_recovery_10000_seed4777_20261026/training_and_test_complete.json'
    while not (previous.exists() and previous_run.exists()):
        status('waiting_for_previous_test',previous_test=str(previous),previous_completion=str(previous_run))
        time.sleep(30)
    attempt=0
    while not (RESULT/'publication_complete.json').exists():
        cp=RUN/'model/checkpoint.pth'
        saved=torch.load(cp,map_location='cpu',weights_only=False) if cp.exists() else {}
        completed=saved.get('completed_episodes',0)
        final=saved.get('is_final',False) and completed==7000
        if saved:
            assert saved['temporal_token_config'].get('action_features') is False
            assert saved['temporal_token_config']['raw_input_dim']==21
            assert saved['use_local_revisit_recovery'] is True
        del saved
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        devices=sorted([dict(index=int(i),uuid=u.strip(),free_mib=int(f)) for i,u,f in csv.reader(out.splitlines())],key=lambda x:-x['free_mib'])
        learner,sampler=devices[:2]
        if learner['free_mib']<(4096 if final else 14336) or (not final and sampler['free_mib']<5120):
            status('waiting_for_gpu',completed_episodes=completed,gpus=devices,learner_required_mib=4096 if final else 14336,sampler_required_mib=0 if final else 5120)
            time.sleep(60);continue
        attempt+=1
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',ARIADNE_MAP_DIR=str(ROOT/'ARiADNE_MAMBA_3.1_TOKEN21/maps'),ARIADNE_DISABLE_GIFS='0',ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS='1',ARIADNE_WRITE_RECOVERY_DIAGNOSTICS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        env.pop('HERA_FIXED_GPU_INDICES',None);env.pop('HERA_FIXED_TEST_GPU',None)
        log=CONTROL/f'attempt_{attempt}_{time.time_ns()}.log'
        if final:
            for parent in [REPO/'results/data',REPO/'results/visualizations']:
                p=parent/RESULT_NAME/'recovery_on/single_seed_4777'
                for candidate in [p,p.parent/('.'+p.name+'.staging')]:
                    if candidate.exists():candidate.rename(candidate.with_name(candidate.name+f'.failed_{time.time_ns()}'))
            env['CUDA_VISIBLE_DEVICES']=learner['uuid']
            (REPO/'results/runs').mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u',str(REPO/'scripts/evaluate_recovery_streak.py'),'--streak','2','--gpu',learner['uuid'],'--checkpoint',str(cp),'--training-episodes','7000','--result-name',RESULT_NAME,'--recovery','on']
            phase='testing'
        else:
            if not cp.exists():
                for p in [RUN,GIFS]:
                    if p.exists():p.rename(p.with_name(p.name+f'.before_first_checkpoint_{time.time_ns()}'))
            env['CUDA_VISIBLE_DEVICES']=sampler['uuid']+','+learner['uuid']
            command=[sys.executable,'-u',str(REPO/'scripts/train_pose_original_parallel.py'),'--split-file',str(SPLIT),'--recovery','on','--trigger-steps','2','--point-repeat-trigger-steps','2','--seed','4777','--total-episodes','7000','--total-env-steps','0','--features','on','--action-features','off','--motion-delta','on','--map-delta','on','--reward','on','--output-dir',str(RUN),'--gifs-dir',str(GIFS)]
            if cp.exists():command.append('--resume')
            phase='training'
        status(phase,attempt=attempt,learner=learner,sampler=sampler,completed_episodes=completed,log=str(log))
        with log.open('w') as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
            plot=None
            if phase=='training':
                with (CONTROL/'plots.log').open('a') as plots:
                    plot=subprocess.Popen([sys.executable,'-u',str(REPO/'scripts/plot_training_progress.py'),'--run-dir',str(RUN),'--total-episodes','7000','--streak','2','--watch'],env=env,stdout=plots,stderr=subprocess.STDOUT)
            code=child.wait()
            if plot:
                plot.terminate();plot.wait()
        if code:
            tail=log.read_text(errors='replace')[-24000:]
            if not any(x in tail.lower() for x in ['out of memory','cuda error','rayactorerror','actor died','worker unexpectedly exits']):
                status('failed',log=str(log),error=tail[-2500:]);raise RuntimeError(f'Non-CUDA failure: {log}')
            status('retry_wait',log=str(log),error=tail[-1800:]);time.sleep(60)
    atomic(RUN/'training_and_test_complete.json',dict(completed_episodes=7000,validated_test_maps=1133,finished_unix=time.time()))
    status('complete',completed_episodes=7000,test_maps=1133,results=str(RESULT))

if __name__=='__main__':main()
