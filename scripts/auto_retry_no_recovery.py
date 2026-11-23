"""Wait for GPU capacity, resume training, and retry CUDA OOM without changing the experiment."""
import csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
from run_token_ablation_queue import REPO,ROOT,QUEUE,SPLIT,SPECS,atomic_json,validate_checkpoint,summarize,sha
RUN=REPO/'experiments/ablation_no_recovery_10000_seed4777_20261026'
CONTROL=Path(str(RUN)+'_control')
STATUS=QUEUE/'auto_retry_status.json'
SPEC=next(s for s in SPECS if s['name']=='no_recovery')
RESULT=REPO/'results/data/ablation_no_recovery_10000/recovery_off/single_seed_4777'

def inventory():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    return sorted([dict(index=int(i),uuid=u.strip(),free_mib=int(f)) for i,u,f in csv.reader(out.splitlines())],key=lambda d:-d['free_mib'])

def state(name,**kw):
    d=dict(state=name,current='no_recovery',updated_unix=time.time(),**kw)
    atomic_json(STATUS,d);atomic_json(QUEUE/'status.json',d)
    print(json.dumps(d),flush=True)

def retryable(text):
    return any(s in text.lower() for s in ['out of memory','cuda error','rayactorerror','actor died','worker unexpectedly exits'])

def main():
    import torch
    attempt=0
    while True:
        checkpoint=torch.load(RUN/'model/checkpoint.pth',map_location='cpu',weights_only=False)
        final=checkpoint.get('is_final',False) and checkpoint.get('completed_episodes')==10000
        completed=checkpoint.get('completed_episodes');del checkpoint
        if (RESULT/'publication_complete.json').exists():break
        devices=inventory()
        learner=devices[0]
        sampler=devices[1] if len(devices)>1 else None
        if learner['free_mib']<(4096 if final else 14336) or (not final and (not sampler or sampler['free_mib']<5120)):
            state('waiting_for_gpu',completed_episodes=completed,gpus=devices,learner_required_mib=4096 if final else 14336,sampler_required_mib=0 if final else 5120)
            time.sleep(60);continue
        attempt+=1
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',ARIADNE_MAP_DIR=str(ROOT/'ARiADNE_MAMBA_3.1_TOKEN21/maps'),ARIADNE_DISABLE_GIFS='0',ARIADNE_SAVE_TRIGGERED_RECOVERY_ARTIFACTS='1',ARIADNE_WRITE_RECOVERY_DIAGNOSTICS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        env.pop('HERA_FIXED_GPU_INDICES',None);env.pop('HERA_FIXED_TEST_GPU',None)
        log=CONTROL/f'auto_retry_{attempt}_{time.time_ns()}.log'
        if final:
            validate_checkpoint(RUN/'model/checkpoint.pth',SPEC)
            # Preserve failed partial outputs before a deterministic full rerun.
            for p in [RESULT,REPO/'results/visualizations/ablation_no_recovery_10000/recovery_off/single_seed_4777']:
                for candidate in [p,p.parent/('.'+p.name+'.staging')]:
                    if candidate.exists():candidate.rename(candidate.with_name(candidate.name+f'.failed_{time.time_ns()}'))
            state('testing',attempt=attempt,gpu=learner,completed_episodes=completed,log=str(log))
            env['HERA_FIXED_TEST_GPU']=learner['uuid']
            command=[sys.executable,'-u',str(REPO/'scripts/evaluate_recovery_streak.py'),'--streak','2','--gpu',learner['uuid'],'--checkpoint',str(RUN/'model/checkpoint.pth'),'--training-episodes','10000','--result-name','ablation_no_recovery_10000','--recovery','off']
        else:
            state('training',attempt=attempt,learner=learner,sampler=sampler,completed_episodes=completed,log=str(log))
            env['CUDA_VISIBLE_DEVICES']=sampler['uuid']+','+learner['uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            command=[sys.executable,'-u',str(REPO/'scripts/train_pose_original_parallel.py'),'--resume','--split-file',str(SPLIT),'--recovery','off','--trigger-steps','2','--point-repeat-trigger-steps','2','--seed','4777','--total-episodes','10000','--total-env-steps','0','--features','on','--action-features','on','--motion-delta','on','--map-delta','on','--reward','on','--output-dir',str(RUN),'--gifs-dir',str(REPO/'gifs'/RUN.name)]
        with log.open('w') as f:code=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:
            tail=log.read_text(errors='replace')[-20000:]
            if not retryable(tail):
                state('failed',error=f'Non-CUDA failure {code}; inspect {log}');raise RuntimeError(tail[-2000:])
            state('retry_wait',attempt=attempt,error=tail[-1800:],log=str(log));time.sleep(60)
    completed=json.loads((QUEUE/'completed_trials.json').read_text())
    if not any(x['name']=='no_recovery' for x in completed):
        completed.append(dict(name='no_recovery',run=str(RUN),results=str(RESULT),summary=json.loads((RESULT/'hera_summary.json').read_text())))
    atomic_json(QUEUE/'completed_trials.json',completed);summarize(completed)
    atomic_json(RUN/'training_and_test_complete.json',dict(completed_episodes=10000,validated_test_maps=1133,finished_unix=time.time()))
    atomic_json(QUEUE/'no_recovery_followup_status.json',dict(state='complete',finished_unix=time.time()))
    state('complete',completed=[x['name'] for x in completed])

if __name__=='__main__':main()
