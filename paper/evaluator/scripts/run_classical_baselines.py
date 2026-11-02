#!/usr/bin/env python3
"""Run untrained baselines with Mamba's single-seed, full-test evaluator.

Each model has its own process and one sequential worker. The JSON supplied
as --checkpoint is a versioned planner specification; no weights are loaded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from test_datasets import add_test_dataset_arguments, resolve_test_dataset
MODELS = {
    "utility_1": ("utility", "ARiADNE_UTILITY", 1),
    "utility_10": ("utility", "ARiADNE_UTILITY", 10),
    "utility_25": ("utility", "ARiADNE_UTILITY", 25),
    "nbvp": ("nbvp", "ARiADNE_NBVP", None),
    "tare_local": ("tare_local", "ARiADNE_TARE_LOCAL", None),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planner_specification(key):
    kind, repository, lambda_value = MODELS[key]
    root = WORKSPACE / repository
    payload = {
        "artifact_type": "untrained_planner_specification",
        "planner": kind,
        "model_name": f"Utility-lambda-{lambda_value}" if kind == "utility" else {"nbvp": "NBVP-Local", "tare_local": "TARE-Local"}[kind],
        "repository": repository,
        "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {path.name: digest(path) for path in sorted(root.glob("*.py"))},
    }
    if kind == "tare_local":
        official = root / "official_tare"
        payload["model_name"] = "TARE-Local official core (2D coverage and scheduling adapter)"
        payload["official_source"] = json.loads((official / "provenance.json").read_text())
        payload["adaptation"] = {
            "coverage": "belief-only optimistic unknown area at 0.8m sampling plus observed frontiers; not 3D surface quality",
            "gain_thresholds": "one 2D sample; positive gain only; 3D point counts are not copied",
            "global_path": "planar 16m cells, informative representatives, graph-distance OR-Tools TSP; continue on local completion",
            "viewpoints": "existing planar graph revalidated in both directions against current belief; local reachable 80m square",
            "lookahead": "8m along selected path; execute from robot index, not boundary index",
            "randomness": "upstream randomized greedy sampling; experiment-seeded mt19937",
            "solver": "OR-Tools 9.10.4067 PATH_CHEAPEST_ARC",
        }
        for source in sorted(official.rglob("*")):
            if source.is_file() and "python_deps" not in source.parts and "__pycache__" not in source.parts:
                payload["source_sha256"][str(source.relative_to(root))] = digest(source)
    if kind == "utility":
        payload["lambda_value"] = lambda_value
        payload["score_comparison"] = "log domain; exact monotonic equivalent of A * exp(-lambda * L)"
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    add_test_dataset_arguments(parser)
    parser.add_argument("--global-seed", type=int, default=4777)
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--jobs", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--limit-maps", type=int, default=None, help="Diagnostic prefix only; omit for full test")
    parser.add_argument("--reference-results", type=Path, help="Optional Mamba run to verify paired starts and include in plots")
    parser.add_argument("--visual-output-root", type=Path, help="Defaults to token21 results/visualizations/other_models (diagnostics stay with their results)")
    return parser


def audit_results(output, model_keys, reference=None):
    """Require identical maps/starts across models and an optional Mamba run."""
    model_rows = {}
    for key in model_keys:
        with (output / key / "episodes.csv").open() as stream:
            model_rows[key] = list(csv.DictReader(stream))
    if reference:
        with (reference / "episodes.csv").open() as stream:
            model_rows["mamba_reference"] = list(csv.DictReader(stream))
    first_key = model_keys[0]
    reference_rows = model_rows[first_key]
    reference_metadata = json.loads((output / first_key / "evaluation_metadata.json").read_text())
    identity_fields = ("map_id", "eval_seed", "start_position_seed", "start_cell_x", "start_cell_y", "target_coverage")
    paired = [tuple(row[field] for field in identity_fields) for row in reference_rows]
    summary = {}
    for key, rows in model_rows.items():
        directory = reference if key == "mamba_reference" else output / key
        metadata = json.loads((directory / "evaluation_metadata.json").read_text())
        for field in ("split_manifest_sha256", "split_name", "global_seed", "target_coverage", "max_steps", "max_path_length"):
            if metadata.get(field) != reference_metadata.get(field):
                raise ValueError(f"evaluation configuration {field} differs for {key}")
        if [tuple(row[field] for field in identity_fields) for row in rows] != paired:
            raise ValueError(f"map order, seed, start cells, or target coverage differ for {key}")
        if any(row.get("error") or row["termination_reason"] == "exception" for row in rows):
            raise ValueError(f"evaluation errors in {key}")
        succeeded = [row for row in rows if row["success"].lower() in ("true", "1")]
        summary[key] = {
            "maps": len(rows), "successes": len(succeeded),
            "success_rate": len(succeeded) / len(rows),
            "mean_path_length_all": sum(float(row["path_length"]) for row in rows) / len(rows),
            "mean_path_length_success": sum(float(row["path_length"]) for row in succeeded) / len(succeeded) if succeeded else None,
            "mean_steps": sum(int(row["steps"]) for row in rows) / len(rows),
            "mean_final_coverage": sum(float(row["final_coverage"]) for row in rows) / len(rows),
        }
    payload = {"paired_maps_and_starts_match": True, "summary": summary}
    (output / "comparison_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    args = build_parser().parse_args()
    resolve_test_dataset(args)
    if args.reference_results:
        reference = json.loads((args.reference_results / 'evaluation_metadata.json').read_text())
        if reference.get('split_manifest_sha256') != digest(args.split_file):
            raise ValueError('reference-results uses a different map manifest; run Mamba on the selected dataset first')
    output = args.output_root.resolve()
    if args.visual_output_root is None:
        args.visual_output_root = (
            output / "visualizations" if args.limit_maps is not None else
            (WORKSPACE / "ARiADNE_MAMBA_3.1_TOKEN21/results/visualizations/other_models" /
             "self-generated" / args.map_difficulty if args.dataset_name == "self-generated" else
             WORKSPACE / "ARiADNE_MAMBA_3.1_TOKEN21/results/visualizations/other_models")
        )
    output.mkdir(parents=True, exist_ok=False)
    specs = output / "planner_specs"
    logs = output / "logs"
    specs.mkdir()
    logs.mkdir()
    keys = list(dict.fromkeys(args.models))
    for key in keys:
        (specs / f"{key}.json").write_text(json.dumps(planner_specification(key), indent=2) + "\n")
    config = vars(args).copy()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({
        "python": sys.executable,
        "split_sha256": digest(args.split_file),
        "shared_environment_sha256": digest(PROJECT / "env.py"),
        "adapter_sha256": digest(PROJECT / "evaluation_adapters/classical.py"),
        "protocol": "Mamba single_global_seed_greedy harness; native seed-controlled planners; no training",
        "recovery_enabled": False,
    })
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    child_env = os.environ.copy()
    child_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(output / ".matplotlib")})

    def run_model(key):
        command = [sys.executable, "-u", str(PROJECT / "scripts/evaluate_single_seed_greedy.py"),
                   "--checkpoint", str(specs / f"{key}.json"),
                   "--split-file", str(args.split_file.resolve()), "--global-seed", str(args.global_seed),
                   "--target-coverage", str(args.target_coverage), "--max-steps", str(args.max_steps),
                   "--output-dir", str(output / key),
                   "--adapter", "evaluation_adapters.classical:create_adapter"]
        if args.limit_maps is not None:
            command += ["--limit-maps", str(args.limit_maps)]
        print(f"Starting {key}; log: {logs / (key + '.log')}", flush=True)
        with (logs / f"{key}.log").open("w") as stream:
            completed = subprocess.run(command, cwd=PROJECT, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"Finished {key}: exit={completed.returncode}", flush=True)
        return key, completed.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        statuses = dict(pool.map(run_model, keys))
    (output / "exit_status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    if any(statuses.values()):
        raise RuntimeError(f"evaluation failed; inspect logs: {statuses}")
    audit = audit_results(output, keys, args.reference_results)
    inputs = [str(output / key) for key in keys]
    if args.reference_results:
        inputs.append(str(args.reference_results.resolve()))
    subprocess.run([sys.executable, str(PROJECT / "scripts/aggregate_results.py"), *inputs,
                    "--output-dir", str(output / "comparison"), "--bootstrap-seed", "42",
                    "--bootstrap-samples", "10000"], cwd=PROJECT, env=child_env, check=True)
    subprocess.run([sys.executable, str(PROJECT / "scripts/visualize_classical_baselines.py"),
                    "--results-root", str(output), "--output-root", str(args.visual_output_root.resolve())],
                   cwd=PROJECT, env=child_env, check=True)
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
