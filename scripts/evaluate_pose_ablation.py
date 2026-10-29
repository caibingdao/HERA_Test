#!/usr/bin/env python3
"""Validate two final fresh checkpoints, evaluate paired starts, and report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT.parent / "ARiADNE_MAMBA_Test/mamba31_map_split"


def validate_pair(checkpoints):
    from temporal_pose import validate_checkpoint_pose
    configs = []
    for c, mode in zip(checkpoints, ("absolute", "delta_local")):
        config = c["pose_ablation_config"]
        if not c.get("is_final") or config["initialization"] != "from_scratch":
            raise ValueError("paired evaluation requires final, from-scratch checkpoints")
        if c["env_steps"] != config["total_env_steps"]:
            raise ValueError("checkpoint has not reached its exact training budget")
        if c["learner_updates"] != config["expected_learner_updates"]:
            raise ValueError("checkpoint learner update budget mismatch")
        if c["use_local_revisit_recovery"] is not False or c.get("temporal_pose_override_allowed"):
            raise ValueError("recovery or semantic override invalidates experiment")
        if c["temporal_token_config"] != dict(features=True, behavior_reward=True, map_delta=True):
            raise ValueError("pose comparison requires all three semantic groups enabled")
        validate_checkpoint_pose(c, mode, config["delta_position_scale"])
        configs.append(config)
    for key in ("seed", "total_env_steps", "updates_per_interval", "update_interval_env_steps",
                "checkpoint_selection", "split_sha256", "source_sha256", "collector",
                "expected_learner_updates", "delta_position_scale"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"training configuration mismatch: {key}")
    ignored = {"FOLDER_NAME", "RUN_FOLDER_NAME", "TEMPORAL_POSE_ENCODING", "TEMPORAL_TOKEN_VERSION",
               "RECOVERY_EVENT_LOG_PATH", "TRIGGERED_RECOVERY_ARTIFACT_PATH"}
    hyper = [{k: v for k, v in c["hyperparameters"].items() if k not in ignored} for c in configs]
    if hyper[0] != hyper[1]:
        raise ValueError("training hyperparameters differ beyond pose semantics")
    return configs[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--absolute-run", type=Path, required=True)
    p.add_argument("--delta-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[100,101,102,103,104])
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    import torch
    paths = [(run / "model/final_checkpoint.pth").resolve() for run in (a.absolute_run, a.delta_run)]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    config = validate_pair(checkpoints)
    del checkpoints
    import hashlib
    split = Path(config["split_file"])
    if hashlib.sha256(split.read_bytes()).hexdigest() != config["split_sha256"]:
        raise ValueError("training split manifest changed")
    output = a.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for path, key, mode in zip(paths, ("absolute_pose", "delta_local_pose"), ("absolute", "delta_local")):
        subprocess.run([sys.executable, "-u", str(EVAL_ROOT / "scripts/evaluate_parallel.py"),
            "--checkpoint", str(path), "--split-file", str(split),
            "--eval-seeds", *map(str, a.eval_seeds), "--episodes-per-map", str(len(a.eval_seeds)),
            "--workers", str(a.workers), "--threads-per-worker", "4", "--deterministic", "--fail-fast",
            "--target-coverage", "0.99", "--max-steps", "128",
            "--output-dir", str(output / key), "--adapter-arg", "enable_recovery=false",
            "--adapter-arg", f"model_seed={config['seed']}",
            "--adapter-arg", f"model_name=Mamba31-{key}-RecoveryOff",
            "--adapter-arg", f"temporal_pose_encoding={mode}",
            "--adapter-arg", f"temporal_delta_position_scale={config['delta_position_scale']}",
            "--adapter-arg", "allow_temporal_pose_override=false"],
            cwd=EVAL_ROOT, check=True)
        subprocess.run([sys.executable, str(EVAL_ROOT / "scripts/create_visual_report.py"),
            "--results-dir", str(output / key), "--output-dir", str(output / "visualization" / key)], check=True)
    from scripts.compare_pose_ablation import compare
    compare(output / "absolute_pose", output / "delta_local_pose", output / "paired_comparison")
    print(f"Completed paired evaluation: {output}")


if __name__ == "__main__":
    main()
