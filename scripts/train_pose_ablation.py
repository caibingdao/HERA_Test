#!/usr/bin/env python3
"""Fresh, synchronous, exactly budgeted paired pose ablations using existing SAC."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SPLIT = ROOT.parent / "ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json"


def updates_due(env_steps, warmup, interval, updates_per_interval):
    return max(0, (env_steps - warmup) // interval) * updates_per_interval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    for name in ("seed", "total_env_steps"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if config["seed"] < 0 or any(config[k] <= 0 for k in (
        "total_env_steps", "updates_per_interval", "update_interval_env_steps",
        "checkpoint_interval_env_steps",
    )):
        raise ValueError("invalid seed or training budget")
    if not isinstance(config["recovery"], bool):
        raise ValueError("recovery must be a boolean")
    if config["checkpoint_selection"] != "final_at_exact_env_step_budget":
        raise ValueError("paired experiments select the final budget checkpoint")
    from temporal_pose import pose_config
    semantic = pose_config(config["pose_encoding"], config["delta_position_scale"])
    split = args.split_file.resolve()
    manifest = json.loads(split.read_text())
    train_maps, test_maps = manifest["train_maps"], manifest["test_maps"]
    if not train_maps or set(train_maps) & set(test_maps):
        raise ValueError("empty or overlapping map split")
    if len(set(train_maps)) != len(train_maps):
        raise ValueError("duplicate training maps")
    key = "absolute_pose" if config["pose_encoding"] == "absolute" else "delta_local_pose"
    output = (args.output_dir or ROOT / "experiments/pose_ablation" / key /
              ("recovery_on" if config["recovery"] else "recovery_off") / f"seed_{config['seed']}").resolve()
    if output.exists():
        raise FileExistsError(f"fresh-only training refuses existing output: {output}")
    config.update(semantic)
    config.update(split_file=str(split), output_dir=str(output),
                  split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                  train_map_count=len(train_maps), initialization="from_scratch",
                  collector="synchronous_single_worker", resume_supported=False)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0

    os.environ.update(
        ARIADNE_OUTPUT_ROOT=str(output),
        ARIADNE_RUN_NAME=f"pose_ablation_{key}_seed_{config['seed']}",
        ARIADNE_RANDOM_SEED=str(config["seed"]), ARIADNE_LOAD_MODEL="0",
        ARIADNE_LEARNER_GPU_INDEX="0", ARIADNE_USE_RECOVERY=str(int(config["recovery"])),
        ARIADNE_MAP_SPLIT_FILE=str(split), ARIADNE_MAP_SPLIT_NAME="train",
        ARIADNE_TEMPORAL_POSE_ENCODING=config["pose_encoding"],
        ARIADNE_TEMPORAL_DELTA_POSITION_SCALE=str(config["delta_position_scale"]),
        ARIADNE_ALLOW_TEMPORAL_POSE_OVERRIDE="0",
        ARIADNE_TOKEN_FEATURES="1", ARIADNE_TOKEN_BEHAVIOR_REWARD="1",
        ARIADNE_TOKEN_MAP_DELTA="1",
    )
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("official Mamba training requires CUDA")
    # Reserve an independent run before any imports with output side effects.
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    import parameter as p
    import driver
    from worker import Worker
    from models.rollout_history_cache import RolloutHistoryCache
    from replay_buffer import TrajectoryReplayBuffer
    from torch.utils.tensorboard import SummaryWriter

    os.chdir(ROOT)
    torch.set_num_threads(4)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    policy = driver.build_policy_network(device)
    q1, q2 = driver.build_q_network(device), driver.build_q_network(device)
    tq1, tq2 = driver.build_q_network(device), driver.build_q_network(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    tq1.eval()
    tq2.eval()
    optimizers = [torch.optim.Adam(net.parameters(), lr=p.LR) for net in (policy, q1, q2)]
    log_alpha = torch.tensor([-2.0], device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
    replay = TrajectoryReplayBuffer(p.REPLAY_SIZE, p.USE_UNION_HISTORY_COLLATE)
    cache = RolloutHistoryCache(p.MAMBA_SEQ_LEN, p.EMBEDDING_DIM, p.MAMBA_D_MODEL)
    config["hyperparameters"] = {k: v for k, v in vars(p).items()
                                if k.isupper() and isinstance(v, (int, float, bool, str))}
    config["expected_learner_updates"] = updates_due(
        config["total_env_steps"], p.MINIMUM_BUFFER_SIZE,
        config["update_interval_env_steps"], config["updates_per_interval"])
    config["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.glob("*.py")) + sorted((ROOT / "models").glob("*.py")) +
        [Path(__file__).resolve()]
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    checkpoint_dir = output / "model"
    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(output / "train"))
    steps, updates, episode, target_counter = 0, 0, 0, 1
    next_save = config["checkpoint_interval_env_steps"]

    def save(name, final=False):
        checkpoint = driver.build_checkpoint(
            policy, q1, q2, tq1, tq2, log_alpha,
            *optimizers, alpha_optimizer, episode, target_counter)
        checkpoint.update(env_steps=steps, learner_updates=updates,
                          model_seed=config["seed"], training_seed=config["seed"],
                          pose_ablation_config=config, is_final=final,
                          checkpoint_selection=config["checkpoint_selection"])
        driver.save_checkpoint_atomic(checkpoint, str(checkpoint_dir / name))

    try:
        with (output / "episodes.csv").open("x", newline="") as stream:
            fields = ["episode", "map_path", "seed", "env_steps", "episode_steps",
                      "learner_updates", "success", "path_length", "coverage",
                      "decision_count", "decision_total_ms", "mean_decision_ms",
                      "warm_decision_count", "warm_decision_total_ms",
                      "recovery_steps", "recovery_triggers"]
            rows = csv.DictWriter(stream, fieldnames=fields)
            rows.writeheader()
            while steps < config["total_env_steps"]:
                episode += 1
                # Per-episode RNG is independent of variable learner consumption.
                rollout_seed = config["seed"] + episode
                np.random.seed(rollout_seed)
                torch.manual_seed(rollout_seed)
                torch.cuda.manual_seed_all(rollout_seed)
                policy.eval()
                worker = Worker(0, policy, episode, device=device,
                                save_image=(episode % p.SAVE_IMG_GAP == 0),
                                rollout_history_cache=cache, policy_version=updates,
                                max_episode_steps=min(p.MAX_EPISODE_STEP,
                                    config["total_env_steps"] - steps))
                decision_times = []
                select = worker.robot.select_next_waypoint_cached
                def timed_select(*args, **kwargs):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    result = select(*args, **kwargs)
                    torch.cuda.synchronize(device)
                    decision_times.append((time.perf_counter() - started) * 1000)
                    return result
                worker.robot.select_next_waypoint_cached = timed_select
                recovery_context = worker.robot.get_rollout_context_cached
                def timed_recovery_context(*args, **kwargs):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    result = recovery_context(*args, **kwargs)
                    torch.cuda.synchronize(device)
                    decision_times.append((time.perf_counter() - started) * 1000)
                    return result
                worker.robot.get_rollout_context_cached = timed_recovery_context
                worker.run_episode()
                count = len(worker.trajectory)
                if count == 0:
                    raise RuntimeError("empty rollout cannot advance the training budget")
                if len(decision_times) != count:
                    raise RuntimeError("decision timing must cover every executed action")
                steps += count
                replay.add_episode(worker.trajectory)
                due = updates_due(steps, p.MINIMUM_BUFFER_SIZE,
                                  config["update_interval_env_steps"],
                                  config["updates_per_interval"])
                policy.train()
                while updates < due:
                    if len(replay) < p.MINIMUM_BUFFER_SIZE:
                        raise RuntimeError("replay capacity/warmup incompatible with update schedule")
                    batch = replay.sample_aligned_transitions(
                        p.BATCH_SIZE, p.MAMBA_SEQ_LEN,
                        seed=config["seed"] + updates).to(device)
                    result = driver.sac_update(
                        policy, q1, q2, tq1, tq2, *optimizers,
                        log_alpha, alpha_optimizer, 0.05 * np.log(p.K_SIZE),
                        list(batch.observations), batch.actions, batch.rewards,
                        batch.dones, list(batch.next_observations),
                        batch.current_history, batch.next_history,
                        check_finite=True)
                    updates += 1
                    target_counter += 1
                    if target_counter > 64:
                        tq1.load_state_dict(q1.state_dict())
                        tq2.load_state_dict(q2.state_dict())
                        target_counter = 1
                metrics = worker.perf_metrics
                rows.writerow(dict(episode=episode, map_path=worker.env.map_path,
                    seed=rollout_seed, env_steps=steps, episode_steps=count,
                    learner_updates=updates, success=metrics["success_rate"],
                    path_length=metrics["travel_dist"], coverage=metrics["explored_rate"],
                    decision_count=count, decision_total_ms=sum(decision_times),
                    mean_decision_ms=float(np.mean(decision_times)),
                    warm_decision_count=max(0, count-1),
                    warm_decision_total_ms=sum(decision_times[1:]),
                    recovery_steps=metrics['total_recovery_steps'],
                    recovery_triggers=metrics['total_recovery_triggers']))
                stream.flush()
                writer.add_scalar("Perf/Completion", float(metrics["success_rate"]), steps)
                writer.add_scalar("Perf/EpisodeDecisionSteps", count, steps)
                writer.add_scalar("Timing/MeanDecisionMs", float(np.mean(decision_times)), steps)
                if metrics["success_rate"]:
                    writer.add_scalar("Perf/SuccessfulPathLength", metrics["travel_dist"], steps)
                print(f"env_steps={steps}/{config['total_env_steps']} updates={updates} "
                      f"episode={episode} completion={metrics['success_rate']}", flush=True)
                if steps >= next_save:
                    save("checkpoint.pth")
                    next_save = (steps // config["checkpoint_interval_env_steps"] + 1) * config["checkpoint_interval_env_steps"]
            assert steps == config["total_env_steps"]
            assert updates == config["expected_learner_updates"]
            save("final_checkpoint.pth", final=True)
    finally:
        writer.close()
    print(f"Final checkpoint: {checkpoint_dir / 'final_checkpoint.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
