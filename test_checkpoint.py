"""Evaluate the policy checkpoint written by driver.py."""

import argparse
from pathlib import Path

import numpy as np
import torch

from driver import (
    build_policy_network,
    load_model_state_dict,
    validate_checkpoint_mode,
)
from parameter import model_path
from runner import Runner
from worker import Worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(model_path) / "checkpoint.pth",
        help="checkpoint produced by driver.py (default: %(default)s)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4777)
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="write evaluation GIFs under the configured gifs_path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train until driver.py "
            "prints 'Saved model' (the model is saved every 32 episodes)."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ARiADNE Mamba evaluation requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    validate_checkpoint_mode(checkpoint)
    policy = build_policy_network(device)
    load_model_state_dict(policy, checkpoint["policy_model"], "policy")
    policy.eval()

    runner = Runner(meta_agent_id=0)
    runner.local_network = policy
    runner.set_policy_net_weights(checkpoint["policy_model"])
    print(
        f"Loaded {args.checkpoint} from training episode "
        f"{checkpoint.get('episode', 'unknown')}"
    )

    results = []
    with torch.inference_mode():
        for index in range(args.episodes):
            episode_id = args.seed + index
            worker = Worker(
                meta_agent_id=0,
                policy_net=runner.local_network,
                global_step=episode_id,
                device=device,
                save_image=args.save_images,
                rollout_history_cache=runner.rollout_history_cache,
                policy_version=runner.policy_version,
            )
            worker.run_episode()
            results.append(worker.perf_metrics)
            print(f"episode {index + 1}/{args.episodes}: {worker.perf_metrics}")

    for metric in ("travel_dist", "explored_rate", "success_rate"):
        values = [float(result[metric]) for result in results]
        print(f"mean {metric}: {np.mean(values):.6f}")


if __name__ == "__main__":
    main()
