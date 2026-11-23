#!/usr/bin/env python3
"""Run the held-out split with the token21 policy and its native Env.

This small launcher adapts token21's index-based Env and stochastic Agent API
to the repository's read-only, deterministic full-split evaluation harness.
It does not change training behavior or write model state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


TOKEN21_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TOKEN21_ROOT.parent
EVALUATION_PROJECT = WORKSPACE_ROOT / "ARiADNE_MAMBA_Test" / "mamba31_map_split"
EVALUATION_SCRIPT = EVALUATION_PROJECT / "scripts" / "evaluate_single_seed_greedy.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--visual-output-dir", required=True, type=Path)
    parser.add_argument("--recovery", choices=("on", "off"), required=True)
    parser.add_argument("--model-version", default="full_test")
    parser.add_argument("--global-seed", type=int, default=4777)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--limit-maps", type=int, default=None)
    return parser


def _configure_recovery_from_checkpoint(checkpoint_path: Path) -> dict:
    """Restore controller semantics before importing parameter/worker modules."""
    import torch
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("exploration_progress_signal") != "new_free_cell_count_v1":
        raise ValueError("HERA evaluation requires raw new-free-cell feedback")
    config = saved.get("recovery_trigger_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing recovery_trigger_config")
    for key in ("regional_trigger_steps", "point_repeat_trigger_steps"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"invalid recovery trigger configuration: {key}")
    if config.get("new_free_cell_threshold") != 1.0 or config.get("progress_window") != 8:
        raise ValueError("checkpoint progress settings differ from the supported HERA evaluator")
    os.environ["ARIADNE_RECOVERY_TRIGGER_STEPS"] = str(config["regional_trigger_steps"])
    os.environ["ARIADNE_POINT_REPEAT_TRIGGER_STEPS"] = str(config["point_repeat_trigger_steps"])
    return dict(config)


def _write_token21_manifest(split_file: Path) -> Path:
    with split_file.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    dataset_root = (split_file.parent / manifest["dataset_root"]).resolve()
    test_maps = manifest.get("test_maps")
    if not isinstance(test_maps, list) or not test_maps:
        raise ValueError("split manifest has no test_maps")
    absolute_maps = []
    for name in test_maps:
        path = (dataset_root / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"test map does not exist: {path}")
        absolute_maps.append(str(path))
    if len(set(absolute_maps)) != len(absolute_maps):
        raise ValueError("test_maps contains duplicate map paths")

    output_root = TOKEN21_ROOT / "results" / "cache"
    output_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(split_file.resolve()).encode()).hexdigest()[:16]
    compatible_manifest = output_root / f"token21_test_maps_{identity}_{os.getpid()}.json"
    payload = {
        "source_split_manifest": str(split_file),
        "source_split_name": "test",
        "source_test_map_count": len(absolute_maps),
        "train_maps": absolute_maps,
    }
    temporary = compatible_manifest.with_name(
        f".{compatible_manifest.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(compatible_manifest)
    return compatible_manifest


def _install_token21_compatibility() -> None:
    if str(TOKEN21_ROOT) not in sys.path:
        sys.path.insert(0, str(TOKEN21_ROOT))
    import numpy as np
    import torch

    import agent as agent_module
    import env as env_module
    # The shared evaluator lives beside the older 150D checkout. Preload
    # token21 modules so its imports resolve to this repository's 21D model.
    import model  # noqa: F401
    import replay_buffer  # noqa: F401
    import local_revisit_recovery  # noqa: F401
    import temporal_pose  # noqa: F401
    import utils  # noqa: F401

    class NoValidActionError(RuntimeError):
        """Raised when a policy state has no finite action log probability."""

    agent_module.NoValidActionError = NoValidActionError
    original_agent = agent_module.Agent

    def select_next_waypoint(
        self,
        observation,
        history_batch=None,
        deterministic=False,
    ):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(
                *observation,
                history_batch=history_batch,
            )
        if deterministic:
            if not torch.isfinite(logp).any():
                raise NoValidActionError("policy has no valid action")
            action_index = torch.argmax(logp, dim=1)
        else:
            probabilities = logp.exp()
            if not torch.isfinite(probabilities).all():
                probabilities = torch.nan_to_num(probabilities, nan=0.0)
            if float(probabilities.sum().item()) <= 0:
                raise NoValidActionError("policy has no valid action")
            action_index = torch.multinomial(
                probabilities,
                1,
            ).long().squeeze(1)
        next_node_index = current_edge[0, action_index.item(), 0].item()
        return self.node_coords[next_node_index], action_index

    original_agent.select_next_waypoint = select_next_waypoint

    original_env = env_module.Env
    from env import configured_training_map_paths

    configured_paths = tuple(
        Path(path).expanduser().resolve()
        for path in configured_training_map_paths()
    )

    class EvaluationEnv(original_env):
        def __init__(self, episode_index, plot=False, map_path=None, start_seed=None):
            self.start_seed = start_seed
            self.start_resample_count = 0
            if map_path is None:
                selected_index = int(episode_index) % len(configured_paths)
            else:
                requested_path = Path(map_path).expanduser().resolve()
                try:
                    selected_index = configured_paths.index(requested_path)
                except ValueError as exc:
                    raise ValueError(
                        f"requested map is not in token21 test manifest: {requested_path}"
                    ) from exc
            super().__init__(selected_index, plot=plot)
            self.start_cell = np.asarray(self.robot_cell).copy()
            self.robot_trajectory = [self.robot_location.copy()]

        def import_ground_truth(self, episode_index):
            # Choose the legacy seeded start before initializing belief maps.
            path = configured_paths[int(episode_index) % len(configured_paths)]
            self.map_path = str(path)
            ground_truth = (env_module.io.imread(str(path), as_gray=True) * 255).astype(int)
            ground_truth = env_module.block_reduce(ground_truth, 2, np.min)
            markers = np.argwhere(ground_truth == 208)
            ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
            ground_truth = ground_truth * 254 + 1
            if self.start_seed is None:
                order = list(range(len(markers)))
                if len(markers) <= 10:
                    raise ValueError(f"map has fewer than 11 start-marker cells: {path}")
                order.remove(10)
                order.insert(0, 10)
            else:
                order = np.random.default_rng(int(self.start_seed)).permutation(len(markers)).tolist()
            for attempt, index in enumerate(order):
                row, column = markers[index]
                if ground_truth[row, column] == env_module.FREE:
                    self.start_resample_count = attempt
                    return ground_truth, np.array([column, row], dtype=int)
            raise ValueError(f"map contains no valid start marker: {path}")

        def step(self, next_waypoint):
            reward = super().step(next_waypoint)
            self.robot_trajectory.append(self.robot_location.copy())
            return reward

    env_module.Env = EvaluationEnv


def main() -> int:
    args = _parser().parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    visual_output_dir = args.visual_output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not split_file.is_file():
        raise FileNotFoundError(split_file)
    if not EVALUATION_SCRIPT.is_file():
        raise FileNotFoundError(EVALUATION_SCRIPT)
    if output_dir.exists() or visual_output_dir.exists():
        raise FileExistsError(
            "evaluation or visual output directory already exists; inspect it "
            "or choose a fresh output path"
        )

    recovery_config = _configure_recovery_from_checkpoint(checkpoint)
    compatible_manifest = _write_token21_manifest(split_file)
    os.environ["ARIADNE_MAP_SPLIT_FILE"] = str(compatible_manifest)
    os.environ["ARIADNE_MAP_SPLIT_NAME"] = "train"
    os.environ["ARIADNE_USE_RECOVERY"] = "1" if args.recovery == "on" else "0"
    os.environ["MPLCONFIGDIR"] = str(
        output_dir.parent / f".{output_dir.name}.matplotlib"
    )
    _install_token21_compatibility()

    adapter_args = {
        "enable_recovery": args.recovery == "on",
        "model_name": f"HERA {args.model_version} recovery {args.recovery}",
    }
    sys.argv = [
        str(EVALUATION_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--split-file",
        str(split_file),
        "--split-name",
        "test",
        "--global-seed",
        str(args.global_seed),
        "--target-coverage",
        str(args.target_coverage),
        "--max-steps",
        str(args.max_steps),
        "--output-dir",
        str(output_dir),
        "--visual-output-dir",
        str(visual_output_dir),
        "--adapter-arg",
        f"enable_recovery={json.dumps(adapter_args['enable_recovery'])}",
        "--adapter-arg",
        f"model_name={json.dumps(adapter_args['model_name'])}",
    ]
    if args.limit_maps is not None:
        sys.argv.extend(["--limit-maps", str(args.limit_maps)])
    try:
        runpy.run_path(str(EVALUATION_SCRIPT), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    metadata_path = output_dir / "evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if args.recovery == "on":
        actual_recovery = metadata["adapter_metadata"]["recovery_configuration"]
        if (actual_recovery["trigger_steps"] != recovery_config["regional_trigger_steps"] or
                actual_recovery["point_repeat_trigger_steps"] != recovery_config["point_repeat_trigger_steps"]):
            raise RuntimeError("Evaluated recovery streaks do not match the checkpoint")
    metadata.update(
        recovery_trigger_config=recovery_config,
        model_version=args.model_version,
        start_position_mode="seeded_map_marker_selection",
        start_seed_applied=True,
        action_selection="greedy_argmax",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
