"""Append full-Mamba recovery-off after the already running three-trial queue."""
import json,time
from pathlib import Path
from run_token_ablation_queue import QUEUE,SPECS,atomic_json,run_trial,summarize


def main():
    status_path=QUEUE/'no_recovery_followup_status.json'
    atomic_json(status_path,dict(state='waiting',after=['no_target','no_feedback','no_motion'],episodes=10000,recovery=False,use_mamba=True,started_unix=time.time()))
    while True:
        status=json.loads((QUEUE/'status.json').read_text())
        if status['state']=='failed':
            atomic_json(status_path,dict(state='blocked_by_previous_trial_failure',previous=status,updated_unix=time.time()))
            raise RuntimeError('Previous ablation queue failed; refusing overlapping trial')
        if status['state']=='complete':break
        time.sleep(20)
    completed=json.loads((QUEUE/'completed_trials.json').read_text())
    names={entry['name'] for entry in completed}
    assert {'no_target','no_feedback','no_motion'}<=names
    assert 'no_recovery' not in names,'Already completed recovery-off experiment'
    atomic_json(status_path,dict(state='starting',updated_unix=time.time()))
    try:
        result=run_trial(next(s for s in SPECS if s['name']=='no_recovery'))
        completed.append(result);summarize(completed)
        atomic_json(QUEUE/'completed_trials.json',completed)
        atomic_json(QUEUE/'status.json',dict(state='complete',completed=[e['name'] for e in completed],finished_unix=time.time()))
        atomic_json(status_path,dict(state='complete',finished_unix=time.time()))
    except BaseException as error:
        atomic_json(status_path,dict(state='failed',error=str(error),failed_unix=time.time()))
        old=json.loads((QUEUE/'status.json').read_text())
        atomic_json(QUEUE/'status.json',dict(state='failed',previous_state=old,error=str(error),failed_unix=time.time()))
        raise

if __name__=='__main__':main()
