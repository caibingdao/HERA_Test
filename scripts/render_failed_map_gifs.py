#!/usr/bin/env python3
"""Deterministically replay failed ARIADNE episodes and render GIFs.

The output directory contains GIF files only.  Each replay is checked against
the saved evaluation row before its animation is committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Agent, NoValidActionError  # noqa: E402
from env import Env  # noqa: E402
from evaluation_adapters.ariadne import ARIADNEAdapter  # noqa: E402
from experiment_utils import coverage_success, seed_everything  # noqa: E402
from replay_buffer import (  # noqa: E402
    EpisodeTrajectory,
    TrajectoryStep,
    build_history_sequence,
    detach_observation,
)
from utils import get_cell_position_from_coords, get_frontier_in_map  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _failed_rows(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "episodes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    failed = [
        row
        for row in rows
        if row.get("success", "").strip().lower() != "true"
    ]
    if not failed:
        raise ValueError("evaluation contains no failed episodes")
    unexpected = sorted(
        {
            row.get("termination_reason", "")
            for row in failed
            if row.get("termination_reason") != "max_steps"
        }
    )
    if unexpected:
        raise ValueError(
            "only max_steps failures can be replayed by this command: "
            f"{unexpected}"
        )
    failed.sort(
        key=lambda row: (
            str(row["map_id"]),
            int(row["episode_id"]),
            int(row["eval_seed"]),
        )
    )
    return failed


def _safe_stem(map_id: str) -> str:
    stem = Path(map_id).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "map"


class EpisodeRenderer:
    """Render the same three-panel layout as the project's original GIFs."""

    def __init__(self, *, map_id: str, eval_seed: int, figure_dpi: int):
        self.map_id = map_id
        self.eval_seed = eval_seed
        self.figure_dpi = figure_dpi
        self.frames: list[Image.Image] = []
        self.last_step = -1

    @staticmethod
    def _trajectory_cells(environment: Env) -> np.ndarray:
        trajectory = np.asarray(environment.robot_trajectory, dtype=np.float64)
        cells = np.empty_like(trajectory)
        cells[:, 0] = (
            trajectory[:, 0] - environment.belief_origin_x
        ) / environment.cell_size
        cells[:, 1] = (
            trajectory[:, 1] - environment.belief_origin_y
        ) / environment.cell_size
        return cells

    def add(self, environment: Env, robot: Agent, *, step: int) -> None:
        figure, axes = plt.subplots(1, 3, figsize=(18, 5))

        trajectory = self._trajectory_cells(environment)
        robot_cell = trajectory[-1]
        axes[0].imshow(environment.robot_belief, cmap="gray")
        axes[0].axis("off")
        axes[0].plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "b",
            linewidth=2,
            zorder=1,
        )
        axes[0].plot(
            robot_cell[0],
            robot_cell[1],
            "mo",
            markersize=4,
            zorder=5,
        )

        nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
        robot_position = get_cell_position_from_coords(
            robot.location,
            robot.map_info,
        )
        axes[1].imshow(robot.map_info.map, cmap="gray")
        axes[1].axis("off")
        if len(robot.frontier) > 0:
            frontiers = get_cell_position_from_coords(
                np.asarray(list(robot.frontier)),
                robot.map_info,
            ).reshape(-1, 2)
            axes[1].scatter(frontiers[:, 0], frontiers[:, 1], c="r", s=2)
        axes[1].scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)
        for node, utility in zip(nodes, robot.utility):
            axes[1].text(node[0], node[1], str(utility), zorder=3)
        axes[1].plot(
            robot_position[0],
            robot_position[1],
            "mo",
            markersize=16,
            zorder=5,
        )
        for coordinates in robot.node_coords:
            node = robot.node_manager.nodes_dict.find(
                coordinates.tolist()
            ).data
            for neighbor_coordinates in node.neighbor_set:
                end = (
                    np.asarray(neighbor_coordinates) - coordinates
                ) / 2 + coordinates
                axes[1].plot(
                    (
                        np.asarray([coordinates[0], end[0]])
                        - robot.map_info.map_origin_x
                    )
                    / robot.cell_size,
                    (
                        np.asarray([coordinates[1], end[1]])
                        - robot.map_info.map_origin_y
                    )
                    / robot.cell_size,
                    "tan",
                    zorder=1,
                )

        axes[2].imshow(robot.map_info.map, cmap="gray")
        axes[2].axis("off")
        axes[2].scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=robot.guidepost,
            zorder=2,
        )
        axes[2].plot(
            robot_position[0],
            robot_position[1],
            "mo",
            markersize=16,
            zorder=5,
        )
        figure.suptitle(
            "Explored ratio: {:.4g}  Travel distance: {:.4g}".format(
                environment.explored_rate,
                environment.travel_dist,
            )
        )
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=self.figure_dpi,
        )
        plt.close(figure)
        buffer.seek(0)
        with Image.open(buffer) as image:
            self.frames.append(image.convert("RGB").copy())
        self.last_step = step

    def save(self, output: Path, *, duration_ms: int) -> None:
        if not self.frames:
            raise RuntimeError("cannot save an empty animation")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".gif.tmp")
        frames = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
            for frame in self.frames
        ]
        frames[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=frames[1:] + [frames[-1]] * 5,
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,
        )
        temporary.replace(output)


@torch.no_grad()
def _replay(
    adapter: ARIADNEAdapter,
    row: Mapping[str, str],
    *,
    map_path: Path,
    frame_stride: int,
    figure_dpi: int,
) -> EpisodeRenderer:
    eval_seed = int(row["eval_seed"])
    episode_id = int(row["episode_id"])
    start_seed = int(row["start_position_seed"])
    max_steps = int(row["steps"])
    target_coverage = float(row["target_coverage"])

    seed_everything(eval_seed, deterministic=True)
    adapter.policy_net.eval()
    environment = Env(
        episode_id,
        plot=False,
        map_path=map_path,
        start_seed=start_seed,
    )
    robot = Agent(adapter.policy_net, adapter.device, plot=False)
    robot.update_planning_state(
        environment.belief_info,
        environment.robot_location,
    )
    observation = robot.get_observation()
    trajectory = EpisodeTrajectory(episode_id=episode_id)
    robot_heading = 0.0
    completed_steps = 0
    termination_reason = None
    renderer = EpisodeRenderer(
        map_id=row["map_id"],
        eval_seed=eval_seed,
        figure_dpi=figure_dpi,
    )
    renderer.add(environment, robot, step=0)

    while termination_reason is None and completed_steps < max_steps:
        current_observation = observation
        coverage_before = float(environment.get_exploration_rate())
        frontier_before = len(get_frontier_in_map(environment.belief_info))
        robot_pose = torch.tensor(
            [
                environment.robot_location[0],
                environment.robot_location[1],
                robot_heading,
            ],
            dtype=torch.float32,
        )
        history_batch = build_history_sequence(
            trajectory,
            sequence_length=adapter.policy_net.mamba_seq_len,
        )
        if history_batch is not None:
            history_batch = history_batch.to(adapter.device)
        try:
            next_location, action_index = adapter._select_next_waypoint(
                robot,
                observation,
                history_batch,
                deterministic=True,
            )
        except NoValidActionError as exc:
            raise RuntimeError(
                f"replay unexpectedly had no valid action at step "
                f"{completed_steps}: {row['map_id']}"
            ) from exc

        movement = next_location - environment.robot_location
        next_heading = float(np.arctan2(movement[1], movement[0]))
        reward = environment.step(next_location)
        completed_steps += 1
        robot.update_planning_state(
            environment.belief_info,
            environment.robot_location,
        )
        environment_success = bool(robot.utility.sum() == 0)
        if environment_success:
            reward += 20
        observation = robot.get_observation()
        coverage_after = float(environment.explored_rate)
        frontier_after = len(get_frontier_in_map(environment.belief_info))
        trajectory.append(
            TrajectoryStep(
                observation=detach_observation(current_observation),
                action=int(action_index.item()),
                reward=float(reward),
                next_observation=detach_observation(observation),
                robot_pose=robot_pose,
                delta_coverage=coverage_after - coverage_before,
                delta_frontier=frontier_after - frontier_before,
                done=environment_success,
                step_index=completed_steps - 1,
            )
        )
        robot_heading = next_heading
        reached_target = coverage_success(coverage_after, target_coverage)
        if reached_target:
            termination_reason = "target_reached"
        elif environment_success:
            termination_reason = "environment_complete"
        elif completed_steps >= max_steps:
            termination_reason = "max_steps"

        if completed_steps % frame_stride == 0 or termination_reason is not None:
            renderer.add(environment, robot, step=completed_steps)

    expected_step = int(row["steps"])
    expected_coverage = float(row["final_coverage"])
    expected_distance = float(row["path_length"])
    actual_coverage = float(environment.get_exploration_rate())
    if completed_steps != expected_step:
        raise RuntimeError(
            f"step mismatch for {row['map_id']}: "
            f"replay={completed_steps}, recorded={expected_step}"
        )
    if termination_reason != row["termination_reason"]:
        raise RuntimeError(
            f"termination mismatch for {row['map_id']}: "
            f"replay={termination_reason}, recorded={row['termination_reason']}"
        )
    if not math.isclose(actual_coverage, expected_coverage, abs_tol=1e-12):
        raise RuntimeError(
            f"coverage mismatch for {row['map_id']}: "
            f"replay={actual_coverage}, recorded={expected_coverage}"
        )
    if not math.isclose(
        float(environment.travel_dist), expected_distance, abs_tol=1e-8
    ):
        raise RuntimeError(
            f"distance mismatch for {row['map_id']}: "
            f"replay={environment.travel_dist}, recorded={expected_distance}"
        )
    if renderer.last_step != completed_steps:
        renderer.add(environment, robot, step=completed_steps)
    return renderer


def _render_worker(args: argparse.Namespace) -> int:
    results_dir = args.results_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    maps_dir = args.maps_dir.resolve()
    rows = _failed_rows(results_dir)
    if args.limit is not None:
        rows = rows[: args.limit]
    selected = list(enumerate(rows))[args.worker_index :: args.workers]
    if not selected:
        return 0

    adapter = ARIADNEAdapter(
        checkpoint_path=checkpoint,
        deterministic=True,
        adapter_args={},
    )
    if not adapter.use_mamba:
        raise RuntimeError("checkpoint is not a Mamba model")

    for ordinal, row in selected:
        coverage_percent = 100.0 * float(row["final_coverage"])
        filename = (
            f"{ordinal + 1:03d}_{_safe_stem(row['map_id'])}"
            f"_episode_{int(row['episode_id'])}"
            f"_seed_{int(row['eval_seed'])}"
            f"_coverage_{coverage_percent:.2f}.gif"
        )
        output_path = args.output_dir / filename
        if output_path.is_file():
            print(
                f"[{ordinal + 1}/{len(rows)}] reusing {filename}",
                flush=True,
            )
            continue
        map_path = maps_dir / row["map_id"]
        if not map_path.is_file():
            raise FileNotFoundError(f"map does not exist: {map_path}")
        renderer = _replay(
            adapter,
            row,
            map_path=map_path,
            frame_stride=args.frame_stride,
            figure_dpi=args.figure_dpi,
        )
        renderer.save(output_path, duration_ms=args.duration_ms)
        print(
            f"[{ordinal + 1}/{len(rows)}] {filename}",
            flush=True,
        )
    return 0


def _validate_inputs(args: argparse.Namespace) -> int:
    results_dir = args.results_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    metadata = _read_json(results_dir / "evaluation_metadata.json")
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError(
            "checkpoint hash does not match the evaluation metadata"
        )
    adapter_metadata = metadata.get("adapter_metadata", {})
    if not isinstance(adapter_metadata, Mapping) or not adapter_metadata.get(
        "use_mamba"
    ):
        raise RuntimeError("evaluation metadata is not for a Mamba model")
    rows = _failed_rows(results_dir)
    if args.limit is not None:
        rows = rows[: args.limit]
    return len(rows)


def _parent(args: argparse.Namespace) -> int:
    expected = _validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    non_gif_existing = [
        path
        for path in args.output_dir.iterdir()
        if not path.is_file() or path.suffix.lower() != ".gif"
    ]
    if non_gif_existing:
        raise ValueError(
            "output directory contains non-GIF entries: "
            + ", ".join(str(path) for path in non_gif_existing)
        )

    processes = []
    for worker_index in range(args.workers):
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--results-dir",
            str(args.results_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--maps-dir",
            str(args.maps_dir),
            "--output-dir",
            str(args.output_dir),
            "--workers",
            str(args.workers),
            "--worker-index",
            str(worker_index),
            "--frame-stride",
            str(args.frame_stride),
            "--figure-dpi",
            str(args.figure_dpi),
            "--duration-ms",
            str(args.duration_ms),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT))

    failures = []
    for worker_index, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failures.append((worker_index, return_code))
    if failures:
        raise RuntimeError(f"GIF rendering workers failed: {failures}")

    gifs = sorted(args.output_dir.glob("*.gif"))
    if len(gifs) != expected:
        raise RuntimeError(
            f"expected {expected} GIFs, found {len(gifs)} in "
            f"{args.output_dir.resolve()}"
        )
    non_gifs = [path for path in args.output_dir.iterdir() if path.suffix != ".gif"]
    if non_gifs:
        raise RuntimeError(f"non-GIF outputs were created: {non_gifs}")
    print(
        f"Created and validated {len(gifs)} failed-episode GIFs in "
        f"{args.output_dir.resolve()}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--maps-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--worker-index",
        type=_non_negative_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frame-stride", type=_positive_int, default=2)
    parser.add_argument("--figure-dpi", type=_positive_int, default=90)
    parser.add_argument("--duration-ms", type=_positive_int, default=120)
    parser.add_argument("--limit", type=_positive_int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_index is None:
        return _parent(args)
    if args.worker_index >= args.workers:
        raise ValueError("worker-index must be smaller than workers")
    return _render_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
