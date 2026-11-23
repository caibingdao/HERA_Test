#!/usr/bin/env python3
"""Export high-resolution paper figures from ARiADNE map arrays.

The exporter deliberately uses the same ``Env`` map preprocessing and the
same world-to-grid conversion as the existing GIF visualization:

    grid_x = (world_x - belief_origin_x) / CELL_SIZE
    grid_y = (world_y - belief_origin_y) / CELL_SIZE

``imshow(origin="upper")`` matches the current GIF orientation: grid x is the
image column, while grid y is the image row increasing downwards. No image or
GIF screenshots are used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "model"
    / "parallel_ariadne_mamba_3_1"
    / "checkpoint.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
DEFAULT_MAP_DIR = PROJECT_ROOT / "maps"

FREE_COLOR = "#FFFFFF"
NON_FREE_COLOR = "#8A8A8A"
TRAJECTORY_COLOR = "#0000FF"
START_COLOR = "#C000C0"
END_COLOR = "#00BFC4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Render publication-quality occupancy-map figures directly "
            "from ARiADNE simulation arrays."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--map-path",
        type=Path,
        help="exact map image under this project's maps/ directory",
    )
    selection.add_argument(
        "--split-file",
        type=Path,
        help="map split manifest containing train_maps/test_maps",
    )
    parser.add_argument(
        "--split-name",
        choices=("train", "test"),
        default="test",
        help="manifest side used with --split-file (default: %(default)s)",
    )
    parser.add_argument(
        "--map-index",
        type=int,
        default=0,
        help="zero-based index in the selected source (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--with-trajectory", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--use-recovery",
        action="store_true",
        help="enable heuristic recovery during trajectory inference",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="fractional crop margin around free space (default: 0.03)",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="also export SVG files",
    )
    args = parser.parse_args(argv)
    if args.map_index < 0:
        parser.error("--map-index must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.dpi < 600:
        parser.error("--dpi must be at least 600")
    if not 0 <= args.margin <= 0.25:
        parser.error("--margin must be between 0 and 0.25")
    return args


def _manifest_entries(split_file: Path, split_name: str):
    split_file = split_file.expanduser().resolve()
    with split_file.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    key = f"{split_name}_maps"
    entries = manifest.get(key)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest has no non-empty {key}: {split_file}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"manifest {key} contains duplicate entries")
    return entries


def select_map_path(args) -> Path:
    if args.map_path is not None:
        selected = args.map_path.expanduser().resolve()
    elif args.split_file is not None:
        entries = _manifest_entries(args.split_file, args.split_name)
        if args.map_index >= len(entries):
            raise IndexError(
                f"map index {args.map_index} is outside the "
                f"{len(entries)}-map {args.split_name} split"
            )
        entry = Path(entries[args.map_index])
        selected = (
            entry.resolve()
            if entry.is_absolute()
            else (DEFAULT_MAP_DIR / entry).resolve()
        )
    else:
        entries = sorted(
            path.resolve()
            for path in DEFAULT_MAP_DIR.iterdir()
            if path.is_file()
        )
        if args.map_index >= len(entries):
            raise IndexError(
                f"map index {args.map_index} is outside the "
                f"{len(entries)}-map maps/ directory"
            )
        selected = entries[args.map_index]

    if not selected.is_file():
        raise FileNotFoundError(f"selected map does not exist: {selected}")
    return selected


def legacy_episode_index(selected_map: Path) -> int:
    """Return the episode id that makes the unmodified Env select this map."""
    selected_map = selected_map.resolve()
    legacy_entries = os.listdir(DEFAULT_MAP_DIR)
    for index, entry in enumerate(legacy_entries):
        if (DEFAULT_MAP_DIR / entry).resolve() == selected_map:
            return index
    raise ValueError(
        "trajectory export requires a map from this project's maps/ "
        f"directory: {selected_map}"
    )


def configure_project_runtime(args, checkpoint=None):
    """Set project flags before importing parameter-dependent modules."""
    os.environ.pop("ARIADNE_MAP_SPLIT_FILE", None)
    os.environ.pop("ARIADNE_MAP_SPLIT_NAME", None)
    os.environ["ARIADNE_LEARNER_GPU_INDEX"] = "0"
    os.environ["ARIADNE_RANDOM_SEED"] = str(args.seed)
    os.environ["ARIADNE_USE_RECOVERY"] = (
        "1" if args.use_recovery else "0"
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from temporal_pose import checkpoint_pose_config
    pose = checkpoint_pose_config(checkpoint or {})
    os.environ["ARIADNE_TEMPORAL_POSE_ENCODING"] = pose["temporal_pose_encoding"]
    os.environ["ARIADNE_TEMPORAL_DELTA_POSITION_SCALE"] = str(pose["temporal_delta_position_scale"])

    token_config = (
        None if checkpoint is None else checkpoint.get("temporal_token_config")
    )
    if token_config is None:
        token_config = {
            "features": True,
            "behavior_reward": True,
            "map_delta": True,
        }
    environment_names = {
        "features": "ARIADNE_TOKEN_FEATURES",
        "behavior_reward": "ARIADNE_TOKEN_BEHAVIOR_REWARD",
        "map_delta": "ARIADNE_TOKEN_MAP_DELTA",
    }
    for name, environment_name in environment_names.items():
        value = token_config.get(name)
        if not isinstance(value, bool):
            raise ValueError(
                f"checkpoint has invalid temporal token field {name}: "
                f"{value!r}"
            )
        os.environ[environment_name] = "1" if value else "0"


def world_to_map_cells(
    trajectory,
    belief_origin_x,
    belief_origin_y,
    cell_size,
):
    """Apply the exact conversion used by ``Env.plot_env``."""
    points = np.asarray(trajectory, dtype=np.float64).reshape(-1, 2)
    origins = np.asarray(
        [belief_origin_x, belief_origin_y],
        dtype=np.float64,
    )
    return (points - origins) / float(cell_size)


def occupancy_rgb(occupancy_map, free_value):
    """Render free as white; obstacle and unknown as one medium gray."""
    occupancy_map = np.asarray(occupancy_map)
    if occupancy_map.ndim != 2:
        raise ValueError("occupancy map must be a 2D array")
    rgb = np.empty(occupancy_map.shape + (3,), dtype=np.uint8)
    rgb[...] = (138, 138, 138)
    rgb[occupancy_map == free_value] = (255, 255, 255)
    return rgb


def crop_limits(occupancy_map, free_value, margin_fraction=0.03):
    rows, columns = np.nonzero(np.asarray(occupancy_map) == free_value)
    if rows.size == 0:
        raise ValueError("occupancy map contains no free-space cells")
    height, width = occupancy_map.shape
    min_x, max_x = int(columns.min()), int(columns.max())
    min_y, max_y = int(rows.min()), int(rows.max())
    margin_x = max(1.0, (max_x - min_x + 1) * margin_fraction)
    margin_y = max(1.0, (max_y - min_y + 1) * margin_fraction)
    left = max(-0.5, min_x - 0.5 - margin_x)
    right = min(width - 0.5, max_x + 0.5 + margin_x)
    top = max(-0.5, min_y - 0.5 - margin_y)
    bottom = min(height - 0.5, max_y + 0.5 + margin_y)
    return left, right, top, bottom


def _figure_size(limits, long_edge=7.0):
    left, right, top, bottom = limits
    width = right - left
    height = bottom - top
    if width >= height:
        return long_edge, max(2.0, long_edge * height / width)
    return max(2.0, long_edge * width / height), long_edge


def render_figure(
    occupancy_map,
    free_value,
    output_stem,
    dpi,
    margin,
    trajectory_cells=None,
    export_svg=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limits = crop_limits(occupancy_map, free_value, margin)
    fig, ax = plt.subplots(figsize=_figure_size(limits))
    ax.imshow(
        occupancy_rgb(occupancy_map, free_value),
        origin="upper",
        interpolation="nearest",
    )
    if trajectory_cells is not None:
        trajectory_cells = np.asarray(trajectory_cells, dtype=np.float64)
        ax.plot(
            trajectory_cells[:, 0],
            trajectory_cells[:, 1],
            color=TRAJECTORY_COLOR,
            linewidth=2.25,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        ax.plot(
            trajectory_cells[0, 0],
            trajectory_cells[0, 1],
            marker="o",
            color=START_COLOR,
            markersize=9,
            linestyle="none",
            zorder=4,
        )
        ax.plot(
            trajectory_cells[-1, 0],
            trajectory_cells[-1, 1],
            marker="o",
            color=END_COLOR,
            markersize=9,
            linestyle="none",
            zorder=5,
        )
    left, right, top, bottom = limits
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.tight_layout(pad=0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    suffixes = ("png", "pdf", "svg") if export_svg else ("png", "pdf")
    saved = []
    for suffix in suffixes:
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor=fig.get_facecolor(),
        )
        saved.append(output_path)
    plt.close(fig)
    return saved


def count_nonfree_trajectory_samples(
    occupancy_map,
    trajectory_cells,
    free_value,
    samples_per_cell=4,
):
    """Densely sample every route segment to catch wall crossings."""
    occupancy_map = np.asarray(occupancy_map)
    points = np.asarray(trajectory_cells, dtype=np.float64).reshape(-1, 2)
    invalid = set()
    height, width = occupancy_map.shape
    for start, end in zip(points[:-1], points[1:]):
        cell_distance = float(np.max(np.abs(end - start)))
        sample_count = max(2, int(math.ceil(cell_distance * samples_per_cell)) + 1)
        for fraction in np.linspace(0.0, 1.0, sample_count):
            point = start + fraction * (end - start)
            column = int(round(point[0]))
            row = int(round(point[1]))
            if not (0 <= row < height and 0 <= column < width):
                invalid.add((row, column))
            elif occupancy_map[row, column] != free_value:
                invalid.add((row, column))
    return len(invalid)


def run_policy_trajectory(args, selected_map, episode_index, checkpoint):
    import torch

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from driver import (
        build_policy_network,
        load_model_state_dict,
        validate_checkpoint_mode,
    )
    from runner import Runner
    from worker import Worker

    if not torch.cuda.is_available():
        raise RuntimeError("Mamba trajectory export requires a CUDA GPU")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    validate_checkpoint_mode(checkpoint)
    policy = build_policy_network(device)
    load_model_state_dict(policy, checkpoint["policy_model"], "policy")
    policy.eval()

    runner = Runner(meta_agent_id=0)
    runner.local_network = policy
    runner.set_policy_net_weights(checkpoint["policy_model"])
    worker = Worker(
        meta_agent_id=0,
        policy_net=runner.local_network,
        global_step=episode_index,
        device=device,
        save_image=False,
        rollout_history_cache=runner.rollout_history_cache,
        policy_version=runner.policy_version,
    )
    if Path(worker.env.map_path).resolve() != selected_map.resolve():
        raise RuntimeError(
            "environment selected a different map: "
            f"expected={selected_map}, actual={worker.env.map_path}"
        )

    trajectory = [np.asarray(worker.env.robot_location, dtype=float).copy()]
    update_robot_location = worker.env.update_robot_location

    def update_and_record(robot_location):
        update_robot_location(robot_location)
        trajectory.append(
            np.asarray(worker.env.robot_location, dtype=float).copy()
        )

    worker.env.update_robot_location = update_and_record
    with torch.inference_mode():
        worker.run_episode()
    return worker.env, np.asarray(trajectory), worker.perf_metrics


def verify_png(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if min(rgb.shape[:2]) < 100:
        raise RuntimeError(f"exported image is unexpectedly small: {rgb.shape}")
    if np.unique(rgb.reshape(-1, 3), axis=0).shape[0] < 2:
        raise RuntimeError(f"exported image has no map contrast: {path}")


def main(argv=None):
    args = parse_args(argv)
    selected_map = select_map_path(args)
    episode_index = legacy_episode_index(selected_map)

    checkpoint = None
    if args.with_trajectory:
        import torch

        checkpoint_path = args.checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    configure_project_runtime(args, checkpoint)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from env import Env
    from parameter import FREE

    output_dir = args.output_dir.expanduser().resolve()
    trajectory = None
    metrics = None
    if args.with_trajectory:
        environment, trajectory, metrics = run_policy_trajectory(
            args,
            selected_map,
            episode_index,
            checkpoint,
        )
    else:
        environment = Env(episode_index, plot=False)
        if Path(environment.map_path).resolve() != selected_map.resolve():
            raise RuntimeError("environment map selection did not match request")

    occupancy_map = np.asarray(environment.ground_truth)
    saved = render_figure(
        occupancy_map,
        FREE,
        output_dir / "paper_map_only",
        args.dpi,
        args.margin,
        export_svg=args.svg,
    )

    trajectory_cells = None
    if trajectory is not None:
        trajectory_cells = world_to_map_cells(
            trajectory,
            environment.belief_origin_x,
            environment.belief_origin_y,
            environment.cell_size,
        )
        invalid_cells = count_nonfree_trajectory_samples(
            occupancy_map,
            trajectory_cells,
            FREE,
        )
        if invalid_cells:
            raise RuntimeError(
                "trajectory alignment validation found "
                f"{invalid_cells} non-free sampled cells"
            )
        saved.extend(
            render_figure(
                occupancy_map,
                FREE,
                output_dir / "paper_map_trajectory",
                args.dpi,
                args.margin,
                trajectory_cells=trajectory_cells,
                export_svg=args.svg,
            )
        )

    for path in saved:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"export was not written correctly: {path}")
    verify_png(output_dir / "paper_map_only.png")
    if trajectory is not None:
        verify_png(output_dir / "paper_map_trajectory.png")

    print(f"Selected map: {selected_map}")
    print(f"Seed: {args.seed}")
    print(
        "Trajectory steps: "
        f"{0 if trajectory is None else len(trajectory) - 1}"
    )
    print(
        "Final coverage: "
        f"{'N/A' if metrics is None else float(metrics['explored_rate']):}"
    )
    print("Saved:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
