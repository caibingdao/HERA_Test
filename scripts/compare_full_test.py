#!/usr/bin/env python3
"""Create paired recovery-on/off summaries and charts for full_test results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _summary(name: str, rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    paths = np.asarray([_number(row, "path_length") for row in rows])
    coverages = np.asarray([_number(row, "final_coverage") for row in rows])
    steps = np.asarray([_number(row, "steps") for row in rows])
    success_mask = np.asarray([_truth(row.get("success", "")) for row in rows])
    success_count = int(success_mask.sum())
    successful_paths = paths[success_mask]
    return {
        "model": name,
        "map_count": len(rows),
        "success_count": success_count,
        "success_rate": success_count / len(rows) if rows else math.nan,
        "mean_final_coverage": float(np.nanmean(coverages)),
        "mean_endpoint_distance_all_runs": float(np.nanmean(paths)),
        "mean_success_path_length": float(np.nanmean(successful_paths)),
        "median_success_path_length": float(np.nanmedian(successful_paths)),
        "p90_success_path_length": float(np.nanpercentile(successful_paths, 90)),
        "mean_steps": float(np.nanmean(steps)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "full_test",
    )
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    on_rows = _read_csv(root / "recovery_on" / "episodes.csv")
    off_rows = _read_csv(root / "recovery_off" / "episodes.csv")
    on_by_map = {row["map_id"]: row for row in on_rows}
    off_by_map = {row["map_id"]: row for row in off_rows}
    if len(on_by_map) != len(on_rows) or len(off_by_map) != len(off_rows):
        raise ValueError("episode data contains duplicate map_id rows")
    common_maps = sorted(set(on_by_map) & set(off_by_map))
    if len(common_maps) != len(on_rows) or len(common_maps) != len(off_rows):
        raise ValueError("recovery on/off runs do not contain the same maps")

    summaries = [
        _summary("recovery_on", on_rows),
        _summary("recovery_off", off_rows),
    ]
    paired = []
    for map_id in common_maps:
        on = on_by_map[map_id]
        off = off_by_map[map_id]
        on_path = _number(on, "path_length")
        off_path = _number(off, "path_length")
        on_success = _truth(on.get("success", ""))
        off_success = _truth(off.get("success", ""))
        paired.append(
            {
                "map_id": map_id,
                "recovery_on_success": on_success,
                "recovery_off_success": off_success,
                "recovery_on_path_length": on_path,
                "recovery_off_path_length": off_path,
                "endpoint_delta_on_minus_off": on_path - off_path,
                "both_successful": on_success and off_success,
                "success_path_delta_on_minus_off": (
                    on_path - off_path
                    if on_success and off_success
                    else math.nan
                ),
                "coverage_delta_on_minus_off": (
                    _number(on, "final_coverage")
                    - _number(off, "final_coverage")
                ),
            }
        )
    endpoint_deltas = np.asarray(
        [row["endpoint_delta_on_minus_off"] for row in paired], dtype=float
    )
    target_path_deltas = np.asarray(
        [row["success_path_delta_on_minus_off"] for row in paired], dtype=float
    )
    comparison = {
        "map_count": len(common_maps),
        "both_models_successful_maps": int(np.isfinite(target_path_deltas).sum()),
        "paired_success_path_delta_mean_on_minus_off": float(np.nanmean(target_path_deltas)),
        "paired_success_path_delta_median_on_minus_off": float(np.nanmedian(target_path_deltas)),
        "recovery_on_shorter_maps_both_success": int(np.sum(target_path_deltas < 0)),
        "recovery_off_shorter_maps_both_success": int(np.sum(target_path_deltas > 0)),
        "equal_success_path_maps_both_success": int(np.sum(target_path_deltas == 0)),
        "mean_endpoint_distance_delta_all_runs": float(np.nanmean(endpoint_deltas)),
        "recovery_on_only_successes": sum(
            row["recovery_on_success"] and not row["recovery_off_success"]
            for row in paired
        ),
        "recovery_off_only_successes": sum(
            row["recovery_off_success"] and not row["recovery_on_success"]
            for row in paired
        ),
    }

    with (root / "comparison_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (root / "paired_map_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    (root / "comparison_summary.json").write_text(
        json.dumps(
            {"models": summaries, "paired_comparison": comparison},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    curves_by_model: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for name in ("recovery_on", "recovery_off"):
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in _read_csv(root / name / "coverage_curve.csv"):
            grouped[row["map_id"]].append(
                (int(float(row["step"])), float(row["coverage"]))
            )
        curves_by_model[name] = grouped
    step_axis = np.arange(129)
    mean_curves = {}
    for name, groups in curves_by_model.items():
        map_curves = []
        for entries in groups.values():
            entries.sort(key=lambda item: item[0])
            x = np.asarray([item[0] for item in entries], dtype=float)
            y = np.asarray([item[1] for item in entries], dtype=float)
            map_curves.append(np.interp(step_axis, x, y))
        mean_curves[name] = np.mean(np.stack(map_curves), axis=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"recovery_on": "#2563EB", "recovery_off": "#D97706"}
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = ["Recovery on", "Recovery off"]
    positions = np.arange(2)
    axes[0, 0].bar(
        positions,
        [item["success_rate"] for item in summaries],
        color=[colors["recovery_on"], colors["recovery_off"]],
    )
    axes[0, 0].set_xticks(positions, labels)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Success rate at 99% coverage")
    axes[0, 0].set_title("Success rate")
    axes[0, 1].boxplot(
        [
            [_number(row, "path_length") for row in on_rows if _truth(row.get("success", ""))],
            [_number(row, "path_length") for row in off_rows if _truth(row.get("success", ""))],
        ],
        tick_labels=labels,
        showfliers=False,
    )
    axes[0, 1].set_ylabel("Travel distance (m)")
    axes[0, 1].set_title("Travel distance among successful maps")
    axes[1, 0].plot(
        step_axis,
        mean_curves["recovery_on"],
        color=colors["recovery_on"],
        label="Recovery on",
    )
    axes[1, 0].plot(
        step_axis,
        mean_curves["recovery_off"],
        color=colors["recovery_off"],
        label="Recovery off",
    )
    axes[1, 0].axhline(0.99, color="#64748B", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="Decision step", ylabel="Mean explored coverage")
    axes[1, 0].set_title("Mean coverage progress across test maps")
    axes[1, 0].legend()
    axes[1, 1].hist(target_path_deltas[np.isfinite(target_path_deltas)], bins=35, color="#7C3AED", alpha=0.85)
    axes[1, 1].axvline(0, color="#0F172A", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Success path delta (on − off, m; both succeeded)")
    axes[1, 1].set_ylabel("Map count")
    axes[1, 1].set_title("Paired successful-map path difference")
    figure.suptitle("Token21 full held-out test comparison", fontsize=16, weight="bold")
    figure.tight_layout()
    figure.savefig(root / "recovery_on_off_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"models": summaries, "paired_comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
