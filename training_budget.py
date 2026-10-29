"""Completed-episode budgets and checkpoint-aligned metric recovery."""
import csv
from pathlib import Path
import shutil
import time


def budget_reached(episodes, steps, episode_budget, step_budget):
    if episode_budget > 0:
        return episodes >= episode_budget
    return step_budget > 0 and steps >= step_budget


def can_launch(episodes, pending, episode_budget):
    return episode_budget <= 0 or episodes + pending < episode_budget


def restore_progress(checkpoint, csv_path):
    """Resolve legacy counters from the exact checkpoint row, archive later rows."""
    path = Path(csv_path)
    steps = int(checkpoint.get('env_steps', 0))
    updates = int(checkpoint.get('learner_updates', 0))
    completed = checkpoint.get('completed_episodes')
    elapsed = 0.0
    if path.exists():
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        matches = [i for i, row in enumerate(rows)
                   if int(row['env_steps']) == steps and int(row['learner_updates']) == updates]
        if not matches:
            raise ValueError('No metrics row matches the resume checkpoint')
        index = matches[-1]
        row = rows[index]
        if completed is not None and int(completed) != int(row['completed_episodes']):
            raise ValueError('Checkpoint and CSV episode counters disagree')
        completed = int(row['completed_episodes'])
        elapsed = float(row['elapsed_seconds'])
        if index + 1 < len(rows):
            shutil.copy2(path, path.with_name(f'{path.name}.before_resume_{time.time_ns()}'))
            temporary = path.with_suffix('.resume.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[:index + 1])
            temporary.replace(path)
    if completed is None:
        raise ValueError('Resume requires completed_episodes or matching training CSV')
    return int(completed), steps, updates, elapsed
