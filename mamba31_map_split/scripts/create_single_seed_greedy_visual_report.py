#!/usr/bin/env python3
"""Create the standard six-chart report for one greedy rollout per map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matplotlib.axes import Axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import create_visual_report as standard_report  # noqa: E402


TEXT_REPLACEMENTS = {
    "5 repeated starts per map": "1 deterministic greedy rollout per map",
    "all ended at max_steps": "did not reach 99% coverage",
}
XLABEL_REPLACEMENTS = {
    "Success rate across 5 episodes": "Single-run success (0 or 1)",
    "Failures out of 5": "Failure indicator (0 or 1)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _validate_metadata(results_dir: Path) -> None:
    metadata_path = results_dir / "evaluation_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("evaluation_mode") != "single_global_seed_greedy":
        raise ValueError(
            "single-seed greedy report requires evaluation_mode="
            "'single_global_seed_greedy'"
        )
    if metadata.get("action_selection") != "greedy_argmax":
        raise ValueError("single-seed greedy report requires greedy_argmax")
    if metadata.get("deterministic") is not True:
        raise ValueError("single-seed greedy report requires deterministic=true")
    if int(metadata.get("episodes_per_map", -1)) != 1:
        raise ValueError("single-seed greedy report requires one episode per map")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    _validate_metadata(results_dir)

    original_text = Axes.text
    original_set_xlabel = Axes.set_xlabel

    def greedy_text(self, x, y, text, *positional, **keyword):
        return original_text(
            self,
            x,
            y,
            TEXT_REPLACEMENTS.get(text, text),
            *positional,
            **keyword,
        )

    def greedy_xlabel(self, xlabel, *positional, **keyword):
        return original_set_xlabel(
            self,
            XLABEL_REPLACEMENTS.get(xlabel, xlabel),
            *positional,
            **keyword,
        )

    Axes.text = greedy_text
    Axes.set_xlabel = greedy_xlabel
    try:
        outputs = standard_report.create_report(
            results_dir,
            output_dir,
            dpi=args.dpi,
        )
    finally:
        Axes.text = original_text
        Axes.set_xlabel = original_set_xlabel

    print(f"Created {len(outputs)} single-seed greedy PNG charts in {output_dir}")
    for path in outputs:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

