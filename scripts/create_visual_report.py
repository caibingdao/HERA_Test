#!/usr/bin/env python3
"""Create a PNG-only visual report from model-independent evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#16A34A",
    "amber": "#D97706",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#475569",
    "light": "#E2E8F0",
    "dark": "#0F172A",
}
THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    output = [_float(row.get(field)) for row in rows]
    return np.asarray([value for value in output if math.isfinite(value)])


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _variance(values: np.ndarray) -> float:
    return float(np.var(values, ddof=1)) if values.size >= 2 else math.nan


def _episode_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("map_id", "")),
        str(row.get("eval_seed", "")),
        str(row.get("episode_id", "")),
    )


def _coverage_trajectories(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[_episode_key(row)].append(row)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda row: _float(row.get("step")))
    return dict(grouped)


def _threshold_statistics(
    trajectories: Mapping[
        tuple[str, str, str], Sequence[Mapping[str, Any]]
    ],
    episode_count: int,
) -> list[dict[str, float]]:
    output = []
    for threshold in THRESHOLDS:
        first_steps = []
        for trajectory in trajectories.values():
            crossing = next(
                (
                    _float(row.get("step"))
                    for row in trajectory
                    if _float(row.get("coverage")) >= threshold
                ),
                math.nan,
            )
            if math.isfinite(crossing):
                first_steps.append(crossing)
        values = np.asarray(first_steps, dtype=np.float64)
        output.append(
            {
                "threshold": threshold,
                "count": float(values.size),
                "rate": float(values.size / episode_count),
                "mean": _mean(values),
                "variance": _variance(values),
            }
        )
    return output


def _configure_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: Any, output: Path, *, dpi: int) -> None:
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def _kpi_card(
    ax: Any,
    *,
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    ax.set_facecolor("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.78, title, transform=ax.transAxes, color=COLORS["slate"])
    ax.text(
        0.06,
        0.43,
        value,
        transform=ax.transAxes,
        fontsize=24,
        weight="bold",
        color=color,
        va="center",
    )
    ax.text(
        0.06,
        0.12,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#64748B",
    )


def _add_bar_labels(
    ax: Any,
    bars: Iterable[Any],
    *,
    formatter,
) -> None:
    for bar in bars:
        height = bar.get_height()
        if not math.isfinite(float(height)):
            continue
        ax.annotate(
            formatter(float(height)),
            (bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )


def plot_overview(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    threshold_stats: Sequence[Mapping[str, float]],
    output: Path,
    *,
    dpi: int,
) -> None:
    successes = np.asarray([_bool(row.get("success")) for row in episodes])
    final_coverage = _values(episodes, "final_coverage")
    steps = _values(episodes, "steps")
    distances = _values(episodes, "path_length")
    num_maps = len({str(row.get("map_id")) for row in episodes})
    success_count = int(np.sum(successes))
    failure_count = len(episodes) - success_count
    adapter_metadata = metadata.get("adapter_metadata", {})
    use_mamba = adapter_metadata.get("use_mamba")
    if use_mamba is True:
        model_title = "ARIADNE + Mamba"
    elif use_mamba is False:
        model_title = "ARIADNE (No Mamba)"
    else:
        model_title = str(adapter_metadata.get("model_name", "Model"))

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(3, 4, height_ratios=(0.85, 0.85, 2.2))
    fig.suptitle(
        f"{model_title} Held-out Map Evaluation — Visual Summary",
        fontsize=22,
        weight="bold",
        color=COLORS["dark"],
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"model={adapter_metadata.get('model_name', 'model')}  |  "
            f"target coverage={float(metadata.get('target_coverage', 0.99)):.0%}  |  "
            f"max steps={metadata.get('max_steps', 'N/A')}"
        ),
        ha="center",
        color=COLORS["slate"],
        fontsize=11,
    )

    cards = [
        ("Test maps", f"{num_maps:,}", "held-out maps", COLORS["blue"]),
        ("Episodes", f"{len(episodes):,}", "5 repeated starts per map", COLORS["cyan"]),
        (
            "Success rate",
            f"{success_count / len(episodes):.2%}",
            f"{success_count:,} reached 99% coverage",
            COLORS["green"],
        ),
        (
            "Mean final coverage",
            f"{_mean(final_coverage):.2%}",
            "all episodes",
            COLORS["purple"],
        ),
        ("Mean decision steps", f"{_mean(steps):.2f}", f"variance={_variance(steps):.2f}", COLORS["amber"]),
        (
            "Mean travel distance",
            f"{_mean(distances):.2f} m",
            f"variance={_variance(distances):,.2f} m²",
            COLORS["red"],
        ),
        (
            "Failed episodes",
            f"{failure_count:,}",
            "all ended at max_steps",
            COLORS["red"],
        ),
        (
            "Exceptions",
            f"{int(metadata.get('num_exception_episodes', 0))}",
            "evaluation runtime errors",
            COLORS["green"],
        ),
    ]
    for index, (title, value, subtitle, color) in enumerate(cards):
        _kpi_card(
            fig.add_subplot(grid[index // 4, index % 4]),
            title=title,
            value=value,
            subtitle=subtitle,
            color=color,
        )

    ax_donut = fig.add_subplot(grid[2, :2])
    ax_donut.pie(
        [success_count, failure_count],
        labels=["Reached 99%", "Max steps"],
        colors=[COLORS["green"], COLORS["red"]],
        autopct=lambda value: f"{value:.2f}%",
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    ax_donut.text(
        0,
        0.03,
        f"{len(episodes):,}",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color=COLORS["dark"],
    )
    ax_donut.text(0, -0.16, "episodes", ha="center", color=COLORS["slate"])
    ax_donut.set_title("Termination outcome")

    ax_rate = fig.add_subplot(grid[2, 2:])
    labels = [f"{item['threshold']:.0%}" for item in threshold_stats]
    rates = [100.0 * item["rate"] for item in threshold_stats]
    bars = ax_rate.bar(labels, rates, color=COLORS["blue"], width=0.68)
    _add_bar_labels(ax_rate, bars, formatter=lambda value: f"{value:.2f}%")
    ax_rate.set_ylim(min(95.0, min(rates) - 0.5), 100.3)
    ax_rate.set_ylabel("Episodes reaching threshold (%)")
    ax_rate.set_xlabel("Coverage threshold")
    ax_rate.set_title("Coverage-threshold reach rate")
    ax_rate.grid(axis="y", alpha=0.3)

    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.22)
    _save(fig, output, dpi=dpi)


def plot_thresholds(
    threshold_stats: Sequence[Mapping[str, float]],
    output: Path,
    *,
    dpi: int,
) -> None:
    labels = [f"{item['threshold']:.0%}" for item in threshold_stats]
    means = [item["mean"] for item in threshold_stats]
    variances = [item["variance"] for item in threshold_stats]
    rates = [100.0 * item["rate"] for item in threshold_stats]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    specs = (
        (means, COLORS["blue"], "Mean first-crossing steps", "Decision steps", lambda x: f"{x:.2f}"),
        (variances, COLORS["purple"], "First-crossing step variance", "Sample variance", lambda x: f"{x:.2f}"),
        (rates, COLORS["green"], "Episodes reaching threshold", "Reach rate (%)", lambda x: f"{x:.2f}%"),
    )
    for ax, (values, color, title, ylabel, formatter) in zip(axes, specs):
        bars = ax.bar(labels, values, color=color, width=0.68)
        _add_bar_labels(ax, bars, formatter=formatter)
        ax.set_title(title)
        ax.set_xlabel("Coverage threshold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.14)
    axes[2].set_ylim(min(95.0, min(rates) - 0.5), 100.4)
    fig.suptitle(
        "Coverage Threshold Efficiency",
        fontsize=20,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output, dpi=dpi)


def plot_distributions(
    episodes: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    dpi: int,
) -> None:
    successful = [row for row in episodes if _bool(row.get("success"))]
    failed = [row for row in episodes if not _bool(row.get("success"))]
    final_coverage = _values(episodes, "final_coverage")
    success_steps = _values(successful, "steps")
    failed_steps = _values(failed, "steps")
    success_distance = _values(successful, "path_length")
    failed_distance = _values(failed, "path_length")
    steps = _values(episodes, "steps")
    distances = _values(episodes, "path_length")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.hist(final_coverage, bins=35, color=COLORS["purple"], alpha=0.86)
    ax.axvline(0.99, color=COLORS["red"], linestyle="--", linewidth=2, label="99% target")
    ax.set_title("Final coverage distribution")
    ax.set_xlabel("Final coverage")
    ax.set_ylabel("Episodes")
    ax.legend()

    ax = axes[0, 1]
    bins = np.arange(0, max(steps) + 5, 5)
    ax.hist(success_steps, bins=bins, color=COLORS["green"], alpha=0.75, label="Reached 99%")
    ax.hist(failed_steps, bins=bins, color=COLORS["red"], alpha=0.78, label="Max steps")
    ax.set_title("Decision-step distribution")
    ax.set_xlabel("Decision steps")
    ax.set_ylabel("Episodes")
    ax.legend()

    ax = axes[1, 0]
    max_distance = float(max(distances))
    bins = np.linspace(0.0, max_distance, 45)
    ax.hist(success_distance, bins=bins, color=COLORS["blue"], alpha=0.72, label="Reached 99%")
    ax.hist(failed_distance, bins=bins, color=COLORS["red"], alpha=0.78, label="Max steps")
    ax.axvline(_mean(distances), color=COLORS["dark"], linestyle="--", label=f"Overall mean={_mean(distances):.1f} m")
    ax.set_title("Travel-distance distribution")
    ax.set_xlabel("Travel distance (m)")
    ax.set_ylabel("Episodes")
    ax.legend()

    ax = axes[1, 1]
    plot = ax.hexbin(
        steps,
        distances,
        gridsize=35,
        cmap="viridis",
        mincnt=1,
    )
    fig.colorbar(plot, ax=ax, label="Episodes per bin")
    ax.set_title("Decision steps vs travel distance")
    ax.set_xlabel("Decision steps")
    ax.set_ylabel("Travel distance (m)")

    fig.suptitle("Episode Efficiency Distributions", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, output, dpi=dpi)


def _unique_xy(
    trajectory: Sequence[Mapping[str, Any]],
    x_field: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    points: dict[float, float] = {}
    for row in trajectory:
        x_value = _float(row.get(x_field))
        coverage = _float(row.get("coverage"))
        if math.isfinite(x_value) and math.isfinite(coverage):
            points[x_value] = coverage
    if not points:
        return None
    x = np.asarray(sorted(points), dtype=np.float64)
    y = np.asarray([points[value] for value in x], dtype=np.float64)
    return x, y


def _mean_curve(
    trajectories: Mapping[
        tuple[str, str, str], Sequence[Mapping[str, Any]]
    ],
    x_field: str,
    *,
    distance_points: int = 151,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared = []
    maximum = 0.0
    for trajectory in trajectories.values():
        xy = _unique_xy(trajectory, x_field)
        if xy is None:
            continue
        prepared.append(xy)
        maximum = max(maximum, float(xy[0][-1]))
    if x_field == "step":
        grid = np.arange(0, int(math.ceil(maximum)) + 1, dtype=np.float64)
    else:
        grid = np.linspace(0.0, maximum, distance_points)
    interpolated = np.stack(
        [
            np.interp(grid, x, y, left=float(y[0]), right=float(y[-1]))
            for x, y in prepared
        ],
        axis=0,
    )
    return (
        grid,
        np.mean(interpolated, axis=0),
        np.std(interpolated, axis=0, ddof=1),
    )


def plot_progress(
    trajectories: Mapping[
        tuple[str, str, str], Sequence[Mapping[str, Any]]
    ],
    output: Path,
    *,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    specs = (
        ("step", "Decision step", "Coverage vs decision step"),
        ("path_length", "Travel distance (m)", "Coverage vs travel distance"),
    )
    for ax, (field, xlabel, title) in zip(axes, specs):
        grid, mean, std = _mean_curve(trajectories, field)
        ax.plot(grid, mean, color=COLORS["blue"], linewidth=2.7, label="Mean coverage")
        ax.fill_between(
            grid,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=COLORS["blue"],
            alpha=0.18,
            label="±1 sample SD",
        )
        for threshold in THRESHOLDS:
            ax.axhline(threshold, color="#94A3B8", linewidth=0.8, alpha=0.45)
        ax.axhline(0.99, color=COLORS["red"], linestyle="--", linewidth=1.8, label="99% target")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.legend(loc="lower right")
    fig.suptitle(
        f"Exploration Progress (mean ± sample SD, n={len(trajectories):,})",
        fontsize=20,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output, dpi=dpi)


def _map_statistics(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[str(row.get("map_id", ""))].append(row)
    output = []
    for map_id, rows in groups.items():
        success = np.asarray([_bool(row.get("success")) for row in rows])
        output.append(
            {
                "map_id": map_id,
                "success_rate": float(np.mean(success)),
                "mean_coverage": _mean(_values(rows, "final_coverage")),
                "mean_steps": _mean(_values(rows, "steps")),
                "failures": int(len(rows) - np.sum(success)),
            }
        )
    return output


def plot_map_reliability(
    map_stats: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    dpi: int,
) -> None:
    rates = np.asarray([float(row["success_rate"]) for row in map_stats])
    sorted_rows = sorted(map_stats, key=lambda row: (row["success_rate"], row["mean_coverage"]))
    hardest = sorted_rows[:20]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    bins = np.linspace(-0.05, 1.05, 12)
    ax.hist(rates, bins=bins, color=COLORS["green"], rwidth=0.88)
    ax.set_title("Per-map success-rate distribution")
    ax.set_xlabel("Success rate across 5 episodes")
    ax.set_ylabel("Maps")
    ax.set_xticks(np.linspace(0, 1, 6))

    ax = axes[0, 1]
    sorted_rates = np.sort(rates)
    ax.plot(np.arange(len(sorted_rates)), sorted_rates, color=COLORS["blue"], linewidth=2)
    ax.set_title("Sorted map reliability")
    ax.set_xlabel("Maps sorted from hardest to easiest")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1, 0]
    names = [str(row["map_id"]) for row in hardest][::-1]
    values = [100.0 * float(row["success_rate"]) for row in hardest][::-1]
    bars = ax.barh(names, values, color=COLORS["red"], alpha=0.82)
    for bar, value in zip(bars, values):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_title("20 hardest maps — success rate")
    ax.set_xlabel("Success rate (%)")
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[1, 1]
    coverage = [100.0 * float(row["mean_coverage"]) for row in hardest][::-1]
    bars = ax.barh(names, coverage, color=COLORS["purple"], alpha=0.82)
    for bar, value in zip(bars, coverage):
        ax.text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=8)
    ax.set_xlim(min(95.0, min(coverage) - 0.3), 100.1)
    ax.set_title("20 hardest maps — mean final coverage")
    ax.set_xlabel("Mean final coverage (%)")
    ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Held-out Map Reliability", fontsize=20, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, output, dpi=dpi)


def plot_failures(
    episodes: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    dpi: int,
) -> None:
    failed = [row for row in episodes if not _bool(row.get("success"))]
    failure_maps = Counter(str(row.get("map_id", "")) for row in failed)
    top = failure_maps.most_common(20)
    final_coverage = 100.0 * _values(failed, "final_coverage")
    distances = _values(failed, "path_length")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    ax = axes[0]
    ax.hist(final_coverage, bins=20, color=COLORS["red"], alpha=0.82)
    ax.axvline(99.0, color=COLORS["dark"], linestyle="--", label="99% target")
    ax.set_title("Failed episodes: final coverage")
    ax.set_xlabel("Final coverage (%)")
    ax.set_ylabel("Episodes")
    ax.legend()

    ax = axes[1]
    ax.hist(distances, bins=20, color=COLORS["amber"], alpha=0.82)
    ax.axvline(_mean(distances), color=COLORS["dark"], linestyle="--", label=f"Mean={_mean(distances):.1f} m")
    ax.set_title("Failed episodes: travel distance")
    ax.set_xlabel("Travel distance (m)")
    ax.set_ylabel("Episodes")
    ax.legend()

    ax = axes[2]
    names = [name for name, _ in top][::-1]
    counts = [count for _, count in top][::-1]
    bars = ax.barh(names, counts, color=COLORS["red"], alpha=0.82)
    for bar, count in zip(bars, counts):
        ax.text(count + 0.08, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)
    ax.set_title("Maps with most failed episodes")
    ax.set_xlabel("Failures out of 5")
    ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        f"Failure Analysis — {len(failed):,} episodes reached max_steps",
        fontsize=20,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output, dpi=dpi)


def create_report(results_dir: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    episodes_path = results_dir / "episodes.csv"
    curve_path = results_dir / "coverage_curve.csv"
    metadata_path = results_dir / "evaluation_metadata.json"
    for path in (episodes_path, curve_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file is missing: {path}")

    if output_dir.exists():
        non_png = [path for path in output_dir.iterdir() if path.suffix.lower() != ".png"]
        if non_png:
            raise ValueError(
                "PNG-only output directory contains non-PNG files: "
                + ", ".join(str(path) for path in non_png)
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _read_csv(episodes_path)
    coverage_rows = _read_csv(curve_path)
    metadata = _read_json(metadata_path)
    if len(episodes) != int(metadata.get("num_episodes", -1)):
        raise ValueError("Episode count does not match evaluation metadata")

    trajectories = _coverage_trajectories(coverage_rows)
    threshold_stats = _threshold_statistics(trajectories, len(episodes))
    map_stats = _map_statistics(episodes)
    outputs = [
        output_dir / "01_overview_dashboard.png",
        output_dir / "02_coverage_thresholds.png",
        output_dir / "03_efficiency_distributions.png",
        output_dir / "04_exploration_progress.png",
        output_dir / "05_map_reliability.png",
        output_dir / "06_failure_analysis.png",
    ]
    plot_overview(episodes, metadata, threshold_stats, outputs[0], dpi=dpi)
    plot_thresholds(threshold_stats, outputs[1], dpi=dpi)
    plot_distributions(episodes, outputs[2], dpi=dpi)
    plot_progress(trajectories, outputs[3], dpi=dpi)
    plot_map_reliability(map_stats, outputs[4], dpi=dpi)
    plot_failures(episodes, outputs[5], dpi=dpi)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    outputs = create_report(args.results_dir, args.output_dir, dpi=args.dpi)
    print(f"Created {len(outputs)} PNG charts in {args.output_dir.resolve()}")
    for path in outputs:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
