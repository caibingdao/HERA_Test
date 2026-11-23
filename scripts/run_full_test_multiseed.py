#!/usr/bin/env python3
"""Run five greedy, matched-seed rollouts per held-out map."""

from __future__ import annotations

import argparse
import csv
import json
import os
import runpy
import subprocess
import sys
import statistics
from pathlib import Path

from run_full_test_eval import (
    TOKEN21_ROOT,
    WORKSPACE_ROOT,
    _install_token21_compatibility,
    _write_token21_manifest,
)

EVALUATION_PROJECT = WORKSPACE_ROOT / "ARiADNE_MAMBA_Test" / "mamba31_map_split"
EVALUATION_SCRIPT = EVALUATION_PROJECT / "scripts" / "evaluate_on_split.py"
REPORT_SCRIPT = EVALUATION_PROJECT / "scripts" / "create_visual_report.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--visual-output-dir", required=True, type=Path)
    parser.add_argument("--recovery", choices=("on", "off"), required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--model-version", default="full_test")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    visual_dir = args.visual_output_dir.expanduser().resolve()
    if not checkpoint.is_file() or not split_file.is_file():
        raise FileNotFoundError("checkpoint or split manifest does not exist")
    if not EVALUATION_SCRIPT.is_file() or not REPORT_SCRIPT.is_file():
        raise FileNotFoundError("shared split evaluation scripts are missing")
    if len(args.eval_seeds) != 5 or len(set(args.eval_seeds)) != 5 or min(args.eval_seeds) < 0:
        raise ValueError("provide exactly five unique non-negative evaluation seeds")
    if output_dir.exists() or visual_dir.exists():
        raise FileExistsError("five-seed result or visual directory already exists")

    compatible_manifest = _write_token21_manifest(split_file)
    os.environ["ARIADNE_MAP_SPLIT_FILE"] = str(compatible_manifest)
    os.environ["ARIADNE_MAP_SPLIT_NAME"] = "train"
    os.environ["ARIADNE_USE_RECOVERY"] = "1" if args.recovery == "on" else "0"
    os.environ["MPLCONFIGDIR"] = str(output_dir.parent / f".{output_dir.name}.matplotlib")
    _install_token21_compatibility()

    sys.argv = [
        str(EVALUATION_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--split-file", str(split_file),
        "--split-name", "test",
        "--eval-seeds", *[str(seed) for seed in args.eval_seeds],
        "--episodes-per-map", "5",
        "--target-coverage", "0.99",
        "--max-steps", "128",
        "--output-dir", str(output_dir),
        "--deterministic",
        "--adapter-arg", "enable_recovery=" + json.dumps(args.recovery == "on"),
        "--adapter-arg", "model_name=" + json.dumps(f"ARIADNE Mamba token21 {args.model_version} recovery {args.recovery} five-seed greedy"),
        "--fail-fast",
    ]
    try:
        runpy.run_path(str(EVALUATION_SCRIPT), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    metadata_path = output_dir / "evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["start_position_mode"] = "seeded_map_marker_selection"
    metadata["start_seed_applied"] = True
    metadata["start_position_seed_note"] = (
        "Actions use argmax. The evaluator derives a start seed for each "
        "map and evaluation seed; marker cells are permuted using NumPy "
        "default_rng and the first valid free marker is selected, matching "
        "the legacy 150D evaluation protocol."
    )
    metadata["action_selection"] = "greedy_argmax"
    metadata["model_version"] = args.model_version
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summaries = []
    for seed in [None, *args.eval_seeds]:
        subset = rows if seed is None else [r for r in rows if int(r["eval_seed"]) == seed]
        successful = [r for r in subset if r["success"].lower() in ("true", "1")]
        summaries.append(dict(
            model_version=args.model_version, recovery=args.recovery,
            policy="greedy", eval_seed="all" if seed is None else seed,
            episodes=len(subset), successes=len(successful),
            failures=len(subset)-len(successful), success_rate=len(successful)/len(subset),
            mean_path_all_m=statistics.mean(float(r["path_length"]) for r in subset),
            mean_path_success_m=statistics.mean(float(r["path_length"]) for r in successful) if successful else None,
            mean_steps=statistics.mean(float(r["steps"]) for r in subset),
            mean_coverage=statistics.mean(float(r["final_coverage"]) for r in subset),
        ))
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--results-dir", str(output_dir),
            "--output-dir", str(visual_dir),
        ],
        check=True,
    )
    print(f"five-seed greedy report complete: {output_dir}")
    print(f"visual report: {visual_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
