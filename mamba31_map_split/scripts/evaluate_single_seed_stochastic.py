#!/usr/bin/env python3
"""Sequential single-seed stochastic evaluation on a held-out map split.

This is intentionally separate from ``evaluate_on_split.py`` and
``evaluate_parallel.py``.  It loads one checkpoint read-only, initializes one
global random stream exactly once, then evaluates every test map once in
manifest order.  Policy actions are sampled from the categorical policy
distribution; no optimizer, gradient, replay buffer, or checkpoint write is
created.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_adapters import ariadne as ariadne_adapter_module  # noqa: E402
from experiment_utils import (  # noqa: E402
    load_split_manifest,
    seed_everything,
    sha256_artifact,
    sha256_file,
    write_json,
)
from scripts.evaluate_on_split import (  # noqa: E402
    CURVE_FIELDS,
    EPISODE_FIELDS,
    _atomic_write_csv,
    _validate_curve,
    _validate_episode,
    load_adapter_factory,
    parse_adapter_args,
    positive_int,
    probability,
)


DEFAULT_GLOBAL_SEED = 4777
EVALUATION_MODE = "single_global_seed_stochastic"


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each held-out map once, sequentially, with one global "
            "random stream and stochastic policy action sampling."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--split-name", default="test", choices=("test",))
    parser.add_argument(
        "--global-seed",
        type=non_negative_int,
        default=DEFAULT_GLOBAL_SEED,
    )
    parser.add_argument("--target-coverage", type=probability, default=0.99)
    parser.add_argument("--max-steps", type=positive_int, default=128)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        default="evaluation_adapters.ariadne:create_adapter",
    )
    parser.add_argument("--adapter-arg", action="append", default=[])
    parser.add_argument(
        "--limit-maps",
        type=positive_int,
        default=None,
        help="Diagnostic-only prefix limit; omit for the official full run.",
    )
    parser.add_argument(
        "--visual-output-dir",
        type=Path,
        default=None,
        help="Generate the standard six-chart report here after evaluation.",
    )
    return parser


def _validate_result(
    result: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    target_coverage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(result, Mapping):
        raise TypeError("adapter evaluate_episode must return a mapping")
    episode = _validate_episode(
        result.get("episode"),
        identity=identity,
        target_coverage=target_coverage,
    )
    curve_identity = {
        key: identity[key]
        for key in (
            "model_name",
            "model_seed",
            "checkpoint_sha256",
            "map_id",
            "eval_seed",
            "episode_id",
        )
    }
    curve = _validate_curve(
        result.get("coverage_curve"),
        curve_identity,
    )
    final_curve = curve[-1]
    if final_curve["step"] != episode["steps"]:
        raise ValueError("final coverage-curve step does not match episode")
    if not math.isclose(
        final_curve["path_length"],
        episode["path_length"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "final coverage-curve path length does not match episode"
        )
    if not math.isclose(
        final_curve["coverage"],
        episode["final_coverage"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("final coverage-curve coverage does not match episode")
    return episode, curve


def _run_visual_report(results_dir: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "create_single_seed_visual_report.py"),
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    checkpoint = args.checkpoint.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {output_dir}"
        )

    manifest, map_paths = load_split_manifest(split_file, args.split_name)
    if args.limit_maps is not None:
        map_paths = map_paths[: args.limit_maps]
    dataset_root = Path(manifest["_dataset_root"])
    checkpoint_sha256 = sha256_artifact(checkpoint)
    split_manifest_sha256 = sha256_file(split_file)

    factory = load_adapter_factory(args.adapter)
    adapter_args = parse_adapter_args(args.adapter_arg)
    adapter = factory(
        checkpoint=checkpoint,
        deterministic=False,
        adapter_args=adapter_args,
    )
    metadata = dict(adapter.metadata())
    model_name = str(metadata.get("model_name", "")).strip()
    model_seed = metadata.get("model_seed")
    if not model_name:
        raise ValueError("adapter metadata must contain a non-empty model_name")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("adapter metadata must contain an integer model_seed")

    staging = output_dir.parent / f".{output_dir.name}.staging"
    if staging.exists():
        raise FileExistsError(
            "staging directory from an earlier run exists; inspect or move it "
            f"before retrying: {staging}"
        )
    staging.mkdir(parents=True)
    progress_path = staging / "progress.jsonl"
    episode_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ignored_episode_reseed_calls = 0

    # Seed once after model construction.  The adapter normally reseeds every
    # episode for the multi-seed protocol, so suppress only that behavior in
    # this isolated, single-process evaluator.
    seed_everything(args.global_seed, deterministic=False)
    original_seed_everything = ariadne_adapter_module.seed_everything

    def preserve_global_stream(*_args: Any, **_kwargs: Any) -> None:
        nonlocal ignored_episode_reseed_calls
        ignored_episode_reseed_calls += 1

    ariadne_adapter_module.seed_everything = preserve_global_stream
    started_at = time.time()
    try:
        with progress_path.open("a", encoding="utf-8", buffering=1) as progress:
            for index, map_path in enumerate(map_paths):
                map_id = map_path.relative_to(dataset_root).as_posix()
                identity = {
                    "model_name": model_name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "map_id": map_id,
                    "model_seed": model_seed,
                    "eval_seed": args.global_seed,
                    "start_position_seed": args.global_seed,
                    "episode_id": 0,
                }
                result = adapter.evaluate_episode(
                    map_path=map_path,
                    map_id=map_id,
                    eval_seed=args.global_seed,
                    start_seed=args.global_seed,
                    episode_id=0,
                    config={
                        "target_coverage": args.target_coverage,
                        "max_steps": args.max_steps,
                        "max_path_length": None,
                        "deterministic": False,
                    },
                )
                episode, curve = _validate_result(
                    result,
                    identity=identity,
                    target_coverage=args.target_coverage,
                )
                episode_rows.append(episode)
                curve_rows.extend(curve)
                progress.write(
                    json.dumps(
                        {
                            "completed": index + 1,
                            "map_id": map_id,
                            "success": episode["success"],
                            "termination_reason": episode["termination_reason"],
                            "steps": episode["steps"],
                            "final_coverage": episode["final_coverage"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                print(
                    f"[{index + 1}/{len(map_paths)}] {map_id} "
                    f"termination={episode['termination_reason']} "
                    f"coverage={episode['final_coverage']:.6f}",
                    flush=True,
                )
    finally:
        ariadne_adapter_module.seed_everything = original_seed_everything
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    _atomic_write_csv(staging / "episodes.csv", episode_rows, EPISODE_FIELDS)
    _atomic_write_csv(staging / "coverage_curve.csv", curve_rows, CURVE_FIELDS)
    write_json(
        staging / "evaluation_metadata.json",
        {
            "evaluation_mode": EVALUATION_MODE,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "split_manifest": str(split_file),
            "split_manifest_sha256": split_manifest_sha256,
            "split_name": args.split_name,
            "num_maps": len(map_paths),
            "num_episodes": len(episode_rows),
            "episodes_per_map": 1,
            "global_seed": args.global_seed,
            "eval_seeds": [args.global_seed],
            "start_position_seed": args.global_seed,
            "seed_initialized_once_after_model_load": True,
            "reseed_between_maps": False,
            "ignored_adapter_reseed_calls": ignored_episode_reseed_calls,
            "single_process": True,
            "single_worker": True,
            "map_order": "split_manifest_test_order",
            "action_selection": "stochastic_categorical_torch_multinomial",
            "deterministic": False,
            "deterministic_policy": False,
            "target_coverage": args.target_coverage,
            "max_steps": args.max_steps,
            "max_path_length": None,
            "adapter": args.adapter,
            "adapter_args": adapter_args,
            "adapter_metadata": metadata,
            "diagnostic_map_limit": args.limit_maps,
            "elapsed_time_seconds": time.time() - started_at,
        },
    )
    progress_path.unlink()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output_dir)

    if args.visual_output_dir is not None:
        visual_output = args.visual_output_dir.expanduser().resolve()
        if visual_output.exists():
            raise FileExistsError(
                "visual output directory already exists; choose a new path: "
                f"{visual_output}"
            )
        _run_visual_report(output_dir, visual_output)

    return {
        "episodes": output_dir / "episodes.csv",
        "coverage_curve": output_dir / "coverage_curve.csv",
        "metadata": output_dir / "evaluation_metadata.json",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run(args)
    print(f"Single-seed evaluation complete: {outputs['episodes'].parent}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if args.visual_output_dir is not None:
        print(f"visual_report: {args.visual_output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
