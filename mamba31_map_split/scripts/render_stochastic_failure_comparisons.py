#!/usr/bin/env python3
"""Replay Mamba 3.1 stochastic failures and render paired trajectory pages.

Each output page compares the corresponding ARiADNE-Mamba episode on the left
with ARiADNE-Mamba-3.1 on the right.  Replays use the saved evaluation seed,
episode ID and start-position seed, force stochastic ``torch.multinomial``
action selection, and are rejected unless steps, termination, final coverage
and path length reproduce the saved CSV row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


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
from utils import get_frontier_in_map  # noqa: E402


MODEL_FIELDS = (
    "success",
    "termination_reason",
    "steps",
    "final_coverage",
    "path_length",
)
IDENTITY_FIELDS = (
    "map_id",
    "eval_seed",
    "episode_id",
    "start_position_seed",
)


@dataclass
class ReplayResult:
    row: Mapping[str, str]
    trajectory_cells: np.ndarray
    final_belief: np.ndarray
    coverage_curve: np.ndarray


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
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


def _read_episodes(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / "episodes.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _episode_key(row: Mapping[str, str]) -> tuple[str, int, int]:
    return (
        row["map_id"],
        int(row["eval_seed"]),
        int(row["episode_id"]),
    )


def _select_pairs(
    mamba_results: Path,
    mamba31_results: Path,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    mamba_rows = _read_episodes(mamba_results)
    mamba31_rows = _read_episodes(mamba31_results)
    mamba_by_key = {_episode_key(row): row for row in mamba_rows}
    failures = [
        row
        for row in mamba31_rows
        if row.get("coverage_success", "").strip().lower() != "true"
    ]
    failures.sort(key=lambda row: (row["map_id"], int(row["episode_id"])))
    pairs = []
    for failed in failures:
        key = _episode_key(failed)
        if key not in mamba_by_key:
            raise KeyError(f"ordinary Mamba result is missing episode {key}")
        ordinary = mamba_by_key[key]
        for field in IDENTITY_FIELDS:
            if ordinary[field] != failed[field]:
                raise ValueError(
                    f"paired episode identity mismatch for {key}: {field}"
                )
        pairs.append((ordinary, failed))
    if not pairs:
        raise ValueError("Mamba 3.1 evaluation contains no failed episodes")
    return pairs


def _validate_metadata(
    results_dir: Path,
    checkpoint: Path,
    *,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _read_json(results_dir / "evaluation_metadata.json")
    if metadata.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError(
            f"checkpoint hash does not match metadata: {checkpoint}"
        )
    if metadata.get("deterministic") is not False:
        raise ValueError(f"evaluation is not stochastic: {results_dir}")
    if metadata.get("eval_seeds") != [100, 101, 102, 103, 104]:
        raise ValueError(f"unexpected evaluation seeds: {results_dir}")
    if metadata.get("episodes_per_map") != 5:
        raise ValueError(f"unexpected episodes_per_map: {results_dir}")
    if metadata.get("max_steps") != 128:
        raise ValueError(f"unexpected max_steps: {results_dir}")
    if not math.isclose(
        float(metadata.get("target_coverage", math.nan)),
        0.99,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"unexpected target coverage: {results_dir}")
    if reference is not None:
        for field in (
            "split_manifest_sha256",
            "split_name",
            "num_maps",
            "eval_seeds",
            "episodes_per_map",
            "target_coverage",
            "max_steps",
            "max_path_length",
            "deterministic",
        ):
            if metadata.get(field) != reference.get(field):
                raise ValueError(
                    f"evaluation metadata mismatch for {field}: {results_dir}"
                )
    return metadata


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


@torch.no_grad()
def _replay_episode(
    adapter: ARIADNEAdapter,
    row: Mapping[str, str],
    *,
    map_path: Path,
) -> ReplayResult:
    eval_seed = int(row["eval_seed"])
    episode_id = int(row["episode_id"])
    start_seed = int(row["start_position_seed"])
    max_steps = 128
    target_coverage = float(row["target_coverage"])

    # This matches evaluate_on_split.py for the stochastic five-seed run: the
    # RNG is reset at the beginning of every map episode.
    seed_everything(eval_seed, deterministic=False)
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
    initial_coverage = float(environment.get_exploration_rate())
    curve = [[0.0, initial_coverage]]
    environment_success = bool(robot.utility.sum() == 0)

    if coverage_success(initial_coverage, target_coverage):
        termination_reason = "target_reached"
    elif environment_success:
        termination_reason = "environment_complete"

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
                deterministic=False,
            )
        except NoValidActionError:
            termination_reason = "no_valid_action"
            break

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
        curve.append([float(completed_steps), coverage_after])

        if coverage_success(coverage_after, target_coverage):
            termination_reason = "target_reached"
        elif environment_success:
            termination_reason = "environment_complete"
        elif completed_steps >= max_steps:
            termination_reason = "max_steps"

    if termination_reason is None:
        termination_reason = "max_steps"

    actual_coverage = float(environment.get_exploration_rate())
    expected_steps = int(row["steps"])
    expected_coverage = float(row["final_coverage"])
    expected_distance = float(row["path_length"])
    mismatches = []
    if completed_steps != expected_steps:
        mismatches.append(f"steps {completed_steps} != {expected_steps}")
    if termination_reason != row["termination_reason"]:
        mismatches.append(
            f"termination {termination_reason} != {row['termination_reason']}"
        )
    if not math.isclose(
        actual_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12
    ):
        mismatches.append(
            f"coverage {actual_coverage} != {expected_coverage}"
        )
    if not math.isclose(
        float(environment.travel_dist),
        expected_distance,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        mismatches.append(
            f"distance {environment.travel_dist} != {expected_distance}"
        )
    if mismatches:
        raise RuntimeError(
            f"stochastic replay mismatch for {_episode_key(row)}: "
            + "; ".join(mismatches)
        )

    return ReplayResult(
        row=row,
        trajectory_cells=_trajectory_cells(environment),
        final_belief=np.asarray(environment.robot_belief).copy(),
        coverage_curve=np.asarray(curve, dtype=np.float64),
    )


def _plot_route(
    ax: Any,
    replay: ReplayResult,
    *,
    model_label: str,
    cmap: str,
) -> LineCollection:
    row = replay.row
    belief = replay.final_belief
    trajectory = replay.trajectory_cells
    ax.imshow(belief, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    points = trajectory.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=0, vmax=max(1, len(segments)))
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=2.2,
        alpha=0.92,
        zorder=3,
    )
    collection.set_array(np.arange(len(segments), dtype=np.float64))
    ax.add_collection(collection)
    ax.scatter(
        trajectory[:, 0],
        trajectory[:, 1],
        s=7,
        c=np.arange(len(trajectory)),
        cmap=cmap,
        norm=norm,
        alpha=0.55,
        linewidths=0,
        zorder=4,
    )
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        marker="*",
        s=180,
        color="#22C55E",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        marker="X",
        s=110,
        color="#EF4444",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )
    label_stride = 16
    for step in range(label_stride, len(trajectory), label_stride):
        ax.text(
            trajectory[step, 0],
            trajectory[step, 1],
            str(step),
            fontsize=7,
            weight="bold",
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#334155",
                "alpha": 0.82,
            },
            zorder=7,
        )

    success = row["coverage_success"].strip().lower() == "true"
    status = "SUCCESS" if success else "FAILURE"
    status_color = "#15803D" if success else "#B91C1C"
    coverage_gains = np.diff(replay.coverage_curve[:, 1])
    final_zero_gain_steps = 0
    for gain in coverage_gains[::-1]:
        if gain > 1e-12:
            break
        final_zero_gain_steps += 1
    ax.set_title(
        f"{model_label} — {status}\n"
        f"coverage={100 * float(row['final_coverage']):.2f}%  "
        f"steps={int(row['steps'])}  distance={float(row['path_length']):.1f} m\n"
        f"termination={row['termination_reason']}",
        fontsize=10.5,
        color=status_color,
        weight="bold",
        pad=7,
    )
    ax.text(
        0.012,
        0.982,
        f"Final consecutive zero-coverage steps: {final_zero_gain_steps}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        weight="bold",
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": status_color,
            "alpha": 0.88,
        },
        zorder=9,
    )
    ax.set_xlim(-0.5, belief.shape[1] - 0.5)
    ax.set_ylim(belief.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return collection


def _render_page(
    ordinary: ReplayResult,
    mamba31: ReplayResult,
    *,
    page_number: int,
    page_count: int,
) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(17, 9.2))
    left = _plot_route(
        axes[0], ordinary, model_label="ARiADNE-Mamba", cmap="viridis"
    )
    right = _plot_route(
        axes[1], mamba31, model_label="ARiADNE-Mamba-3.1", cmap="plasma"
    )
    for ax, collection in zip(axes, (left, right)):
        colorbar = figure.colorbar(
            collection, ax=ax, orientation="horizontal", pad=0.035, fraction=0.035
        )
        colorbar.set_label("Decision step (trajectory direction)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    row = mamba31.row
    figure.suptitle(
        f"Stochastic trajectory comparison {page_number}/{page_count} — "
        f"map={row['map_id']}, eval_seed={row['eval_seed']}, "
        f"episode={row['episode_id']}",
        fontsize=15,
        weight="bold",
        y=0.987,
    )
    figure.text(
        0.5,
        0.945,
        f"Identical start_position_seed={row['start_position_seed']}",
        ha="center",
        fontsize=11,
        color="#334155",
        weight="bold",
    )
    legend = [
        Patch(facecolor="black", label="Obstacle"),
        Patch(facecolor="white", edgecolor="#64748B", label="Explored free"),
        Patch(facecolor="#7F7F7F", label="Unknown at termination"),
        Line2D(
            [0], [0], marker="*", color="none", markerfacecolor="#22C55E",
            markeredgecolor="black", markersize=12, label="Start"
        ),
        Line2D(
            [0], [0], marker="X", color="none", markerfacecolor="#EF4444",
            markeredgecolor="black", markersize=9, label="End"
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=len(legend),
        frameon=True,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.text(
        0.5,
        0.065,
        "Numbered markers show every 16th decision; overlapping segments reveal revisits/loops.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.82,
        bottom=0.11,
        wspace=0.06,
    )
    return figure


def _safe_stem(map_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(map_id).stem)


def _write_summary(
    output: Path,
    pairs: Sequence[tuple[Mapping[str, str], Mapping[str, str]]],
) -> None:
    fields = [
        "map_id",
        "eval_seed",
        "episode_id",
        "start_position_seed",
        *[f"mamba_{field}" for field in MODEL_FIELDS],
        *[f"mamba31_{field}" for field in MODEL_FIELDS],
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ordinary, mamba31 in pairs:
            row: dict[str, Any] = {
                field: mamba31[field] for field in IDENTITY_FIELDS
            }
            row.update({f"mamba_{field}": ordinary[field] for field in MODEL_FIELDS})
            row.update({f"mamba31_{field}": mamba31[field] for field in MODEL_FIELDS})
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba-results", required=True, type=Path)
    parser.add_argument("--mamba31-results", required=True, type=Path)
    parser.add_argument("--mamba-checkpoint", required=True, type=Path)
    parser.add_argument("--mamba31-checkpoint", required=True, type=Path)
    parser.add_argument("--maps-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=_positive_int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [
        args.mamba_results,
        args.mamba31_results,
        args.mamba_checkpoint,
        args.mamba31_checkpoint,
        args.maps_dir,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(
            f"fresh comparison refuses existing output: {args.output_dir}"
        )

    mamba_metadata = _validate_metadata(
        args.mamba_results, args.mamba_checkpoint
    )
    _validate_metadata(
        args.mamba31_results,
        args.mamba31_checkpoint,
        reference=mamba_metadata,
    )
    pairs = _select_pairs(args.mamba_results, args.mamba31_results)
    args.output_dir.mkdir(parents=True)

    mamba_adapter = ARIADNEAdapter(
        checkpoint_path=args.mamba_checkpoint,
        deterministic=False,
        adapter_args={"model_name": "ARiADNE-Mamba", "model_seed": 0},
    )
    mamba31_adapter = ARIADNEAdapter(
        checkpoint_path=args.mamba31_checkpoint,
        deterministic=False,
        adapter_args={"model_name": "ARiADNE-Mamba-3.1", "model_seed": 0},
    )
    if not mamba_adapter.use_mamba or not mamba31_adapter.use_mamba:
        raise RuntimeError("both comparison checkpoints must use Mamba")

    pdf_path = args.output_dir / "all_5_failure_trajectory_comparisons.pdf"
    try:
        with PdfPages(pdf_path) as pdf:
            for index, (ordinary_row, mamba31_row) in enumerate(pairs, 1):
                map_path = args.maps_dir / mamba31_row["map_id"]
                if not map_path.is_file():
                    raise FileNotFoundError(map_path)
                ordinary = _replay_episode(
                    mamba_adapter, ordinary_row, map_path=map_path
                )
                mamba31 = _replay_episode(
                    mamba31_adapter, mamba31_row, map_path=map_path
                )
                figure = _render_page(
                    ordinary,
                    mamba31,
                    page_number=index,
                    page_count=len(pairs),
                )
                png_path = args.output_dir / (
                    f"{index:02d}_{_safe_stem(mamba31_row['map_id'])}"
                    f"_seed_{mamba31_row['eval_seed']}"
                    f"_episode_{mamba31_row['episode_id']}.png"
                )
                figure.savefig(png_path, dpi=args.dpi, facecolor="white")
                pdf.savefig(figure, facecolor="white")
                plt.close(figure)
                print(
                    f"[{index}/{len(pairs)}] replay validated and rendered: "
                    f"{png_path.name}",
                    flush=True,
                )
    finally:
        for adapter in (mamba_adapter, mamba31_adapter):
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    _write_summary(args.output_dir / "episode_comparison.csv", pairs)
    print(f"Created {len(pairs)} paired pages: {args.output_dir.resolve()}")
    print(f"Multi-page PDF: {pdf_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
