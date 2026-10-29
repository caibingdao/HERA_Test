"""Resume final recovery-off experiment and automatically evaluate."""
import json,time
from run_token_ablation_queue import QUEUE,SPECS,atomic_json,run_trial,summarize

def main():
    completed=json.loads((QUEUE/'completed_trials.json').read_text())
    assert {x['name'] for x in completed}=={'no_target','no_feedback','no_motion'}
    atomic_json(QUEUE/'no_recovery_followup_status.json',dict(state='resuming',episodes=10000,updated_unix=time.time()))
    try:
        completed.append(run_trial(next(s for s in SPECS if s['name']=='no_recovery'),resume=True))
        atomic_json(QUEUE/'completed_trials.json',completed)
        summarize(completed)
        atomic_json(QUEUE/'status.json',dict(state='complete',completed=[x['name'] for x in completed],finished_unix=time.time()))
        atomic_json(QUEUE/'no_recovery_followup_status.json',dict(state='complete',finished_unix=time.time()))
    except BaseException as error:
        atomic_json(QUEUE/'status.json',dict(state='failed',error=str(error),failed_unix=time.time()))
        raise

if __name__=='__main__':main()
