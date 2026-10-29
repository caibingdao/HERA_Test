"""Resume failed no_motion, then train full Mamba recovery-off."""
import json,time
from run_token_ablation_queue import QUEUE,SPECS,atomic_json,run_trial,summarize

def main():
    completed=json.loads((QUEUE/'completed_trials.json').read_text())
    assert {x['name'] for x in completed}=={'no_target','no_feedback'}
    atomic_json(QUEUE/'no_recovery_followup_status.json',dict(state='waiting',managed_by='restart_remaining_ablations',episodes=10000))
    try:
        for name in ['no_motion','no_recovery']:
            spec=next(s for s in SPECS if s['name']==name)
            completed.append(run_trial(spec,resume=name=='no_motion'))
            atomic_json(QUEUE/'completed_trials.json',completed)
            summarize(completed)
        atomic_json(QUEUE/'status.json',dict(state='complete',completed=[x['name'] for x in completed],finished_unix=time.time()))
        atomic_json(QUEUE/'no_recovery_followup_status.json',dict(state='complete',finished_unix=time.time()))
    except BaseException as error:
        atomic_json(QUEUE/'status.json',dict(state='failed',error=str(error),failed_unix=time.time()))
        raise

if __name__=='__main__':main()
