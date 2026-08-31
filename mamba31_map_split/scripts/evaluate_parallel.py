#!/usr/bin/env python3
"""Run reproducible held-out evaluation seeds in parallel and merge results.

Each evaluation seed is an independent worker process with its own model and
environment instance. Completed worker outputs remain in a sibling partial
directory and are reused after interruption. The final model-independent CSV
and JSON files are assembled only after every worker passes compatibility and
grid validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import (  # noqa: E402
    load_split_manifest,
    sha256_artifact,
    sha256_file,
    write_json,
)
from scripts.evaluate_on_split import (  # noqa: E402
    CURVE_FIELDS,
    EPISODE_FIELDS,
    _atomic_write_csv,
    parse_adapter_args,
    positive_int,
    probability,
)


class ParallelEvaluationError(RuntimeError):
    """Raised when a parallel worker or merge validation fails."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ParallelEvaluationError(f"JSON root must be an object: {path}")
    return payload


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _worker_is_complete(
    directory: Path,
    *,
    eval_seed: int,
    episode_id: int,
    expected_maps: int,
    checkpoint_sha256: str,
    split_manifest_sha256: str,
    args: argparse.Namespace,
) -> bool:
    episodes_path = directory / "episodes.csv"
    curve_path = directory / "coverage_curve.csv"
    metadata_path = directory / "evaluation_metadata.json"
    if not all(path.is_file() for path in (episodes_path, curve_path, metadata_path)):
        return False
    try:
        metadata = _read_json(metadata_path)
        compatible = (
            metadata.get("checkpoint_sha256") == checkpoint_sha256
            and metadata.get("split_manifest_sha256") == split_manifest_sha256
            and metadata.get("split_name") == "test"
            and metadata.get("eval_seeds") == [eval_seed]
            and metadata.get("episodes_per_map") == 1
            and metadata.get("episode_id_offset") == episode_id
            and metadata.get("num_maps") == expected_maps
            and metadata.get("num_episodes") == expected_maps
            and metadata.get("deterministic") is args.deterministic
            and metadata.get("max_steps") == args.max_steps
            and _same_number(metadata.get("target_coverage"), args.target_coverage)
            and _same_number(metadata.get("max_path_length"), args.max_path_length)
            and metadata.get("adapter") == args.adapter
            and metadata.get("adapter_args") == parse_adapter_args(args.adapter_arg)
        )
        if not compatible:
            return False
        return len(_read_csv(episodes_path)) == expected_maps
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _worker_command(
    args: argparse.Namespace,
    *,
    eval_seed: int,
    episode_id: int,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "evaluate_on_split.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--split-file",
        str(args.split_file),
        "--split-name",
        "test",
        "--eval-seeds",
        str(eval_seed),
        "--episodes-per-map",
        "1",
        "--episode-id-offset",
        str(episode_id),
        "--target-coverage",
        str(args.target_coverage),
        "--max-steps",
        str(args.max_steps),
        "--output-dir",
        str(output_dir),
        "--adapter",
        args.adapter,
        "--deterministic" if args.deterministic else "--no-deterministic",
    ]
    if args.max_path_length is not None:
        command.extend(["--max-path-length", str(args.max_path_length)])
    for adapter_arg in args.adapter_arg:
        command.extend(["--adapter-arg", adapter_arg])
    if args.fail_fast:
        command.append("--fail-fast")
    return command


def _run_workers(
    args: argparse.Namespace,
    *,
    partial_root: Path,
    expected_maps: int,
    checkpoint_sha256: str,
    split_manifest_sha256: str,
) -> list[Path]:
    partial_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for episode_id, eval_seed in enumerate(args.eval_seeds):
        output_dir = partial_root / f"episode_{episode_id}_seed_{eval_seed}"
        if _worker_is_complete(
            output_dir,
            eval_seed=eval_seed,
            episode_id=episode_id,
            expected_maps=expected_maps,
            checkpoint_sha256=checkpoint_sha256,
            split_manifest_sha256=split_manifest_sha256,
            args=args,
        ):
            print(
                f"Reusing completed worker: episode={episode_id}, "
                f"eval_seed={eval_seed}"
            )
            continue
        if output_dir.exists():
            raise ParallelEvaluationError(
                "Incomplete or incompatible worker output exists; move it "
                f"aside before retrying: {output_dir}"
            )
        jobs.append((episode_id, eval_seed, output_dir))

    pending = list(jobs)
    running: dict[subprocess.Popen[Any], tuple[int, int, Path, Any]] = {}
    failures = []
    while pending or running:
        while pending and len(running) < args.workers:
            episode_id, eval_seed, output_dir = pending.pop(0)
            log_path = partial_root / f"episode_{episode_id}_seed_{eval_seed}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(
                    args,
                    eval_seed=eval_seed,
                    episode_id=episode_id,
                    output_dir=output_dir,
                ),
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(args.threads_per_worker),
                    "MKL_NUM_THREADS": str(args.threads_per_worker),
                    "OPENBLAS_NUM_THREADS": str(args.threads_per_worker),
                    "NUMEXPR_NUM_THREADS": str(args.threads_per_worker),
                    "VECLIB_MAXIMUM_THREADS": str(args.threads_per_worker),
                },
            )
            running[process] = (episode_id, eval_seed, output_dir, log_handle)
            print(
                f"Started worker pid={process.pid}: episode={episode_id}, "
                f"eval_seed={eval_seed}, log={log_path}"
            )

        if not running:
            continue
        time.sleep(2.0)
        for process, job in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            episode_id, eval_seed, output_dir, log_handle = job
            log_handle.close()
            del running[process]
            if return_code == 0 and _worker_is_complete(
                output_dir,
                eval_seed=eval_seed,
                episode_id=episode_id,
                expected_maps=expected_maps,
                checkpoint_sha256=checkpoint_sha256,
                split_manifest_sha256=split_manifest_sha256,
                args=args,
            ):
                print(
                    f"Completed worker: episode={episode_id}, "
                    f"eval_seed={eval_seed}"
                )
            else:
                failures.append((episode_id, eval_seed, return_code))
                print(
                    f"Worker failed: episode={episode_id}, "
                    f"eval_seed={eval_seed}, exit={return_code}",
                    file=sys.stderr,
                )

    if failures:
        raise ParallelEvaluationError(
            "One or more workers failed; completed outputs were preserved: "
            + repr(failures)
        )
    return [
        partial_root / f"episode_{episode_id}_seed_{eval_seed}"
        for episode_id, eval_seed in enumerate(args.eval_seeds)
    ]


def _merge_workers(
    args: argparse.Namespace,
    worker_dirs: Sequence[Path],
    *,
    map_paths: Sequence[Path],
    dataset_root: Path,
) -> dict[str, Path]:
    if args.output_dir.exists():
        raise ParallelEvaluationError(
            f"Final output directory already exists: {args.output_dir}"
        )

    map_order = {
        path.relative_to(dataset_root).as_posix(): index
        for index, path in enumerate(map_paths)
    }
    episodes: list[dict[str, str]] = []
    curves: list[dict[str, str]] = []
    metadata_rows = []
    for worker_dir in worker_dirs:
        episodes.extend(_read_csv(worker_dir / "episodes.csv"))
        curves.extend(_read_csv(worker_dir / "coverage_curve.csv"))
        metadata_rows.append(_read_json(worker_dir / "evaluation_metadata.json"))

    expected_episode_count = len(map_paths) * len(args.eval_seeds)
    if len(episodes) != expected_episode_count:
        raise ParallelEvaluationError(
            f"Merged episode count {len(episodes)} != {expected_episode_count}"
        )
    episode_grid = {
        (row.get("map_id", ""), int(row.get("episode_id", -1)))
        for row in episodes
    }
    expected_grid = {
        (map_id, episode_id)
        for map_id in map_order
        for episode_id in range(len(args.eval_seeds))
    }
    if episode_grid != expected_grid:
        raise ParallelEvaluationError("Merged episode grid is incomplete or duplicated")

    episodes.sort(
        key=lambda row: (
            map_order[row["map_id"]],
            int(row["episode_id"]),
        )
    )
    curves.sort(
        key=lambda row: (
            map_order[row["map_id"]],
            int(row["episode_id"]),
            int(row["step"]),
        )
    )

    staging = args.output_dir.parent / f".{args.output_dir.name}.merge_tmp"
    if staging.exists():
        raise ParallelEvaluationError(
            "A previous merge staging directory exists; it was preserved "
            f"for inspection: {staging}"
        )
    staging.mkdir(parents=True)
    _atomic_write_csv(staging / "episodes.csv", episodes, EPISODE_FIELDS)
    _atomic_write_csv(staging / "coverage_curve.csv", curves, CURVE_FIELDS)

    metadata = dict(metadata_rows[0])
    metadata.update(
        {
            "eval_seeds": list(args.eval_seeds),
            "episodes_per_map": len(args.eval_seeds),
            "episode_id_offset": 0,
            "num_maps": len(map_paths),
            "num_episodes": len(episodes),
            "num_exception_episodes": sum(
                int(item.get("num_exception_episodes", 0))
                for item in metadata_rows
            ),
            "parallel_workers": min(args.workers, len(args.eval_seeds)),
            "parallel_unit": "evaluation_seed",
        }
    )
    write_json(staging / "evaluation_metadata.json", metadata)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(args.output_dir)
    return {
        "episodes": args.output_dir / "episodes.csv",
        "coverage_curve": args.output_dir / "coverage_curve.csv",
        "metadata": args.output_dir / "evaluation_metadata.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate independent seeds in parallel, preserve completed "
            "workers, and atomically merge model-independent outputs."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--split-name", default="test", choices=("test",))
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=[100, 101, 102, 103, 104],
    )
    parser.add_argument("--episodes-per-map", type=positive_int, default=5)
    parser.add_argument("--target-coverage", type=probability, default=0.99)
    parser.add_argument("--max-steps", type=positive_int, default=128)
    parser.add_argument("--max-path-length", type=float, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=positive_int, default=5)
    parser.add_argument(
        "--threads-per-worker",
        type=positive_int,
        default=4,
        help=(
            "CPU thread limit for OpenMP/MKL/OpenBLAS in each worker "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adapter",
        default="evaluation_adapters.ariadne:create_adapter",
    )
    parser.add_argument("--adapter-arg", action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes_per_map != len(args.eval_seeds):
        raise ParallelEvaluationError(
            "episodes_per_map must equal the number of eval_seeds"
        )
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ParallelEvaluationError("eval_seeds must not contain duplicates")
    if args.max_path_length is not None and args.max_path_length <= 0:
        raise ParallelEvaluationError("max_path_length must be positive")

    manifest, map_paths = load_split_manifest(args.split_file, "test")
    dataset_root = Path(manifest["_dataset_root"])
    checkpoint_sha256 = sha256_artifact(args.checkpoint)
    split_manifest_sha256 = sha256_file(args.split_file)
    partial_root = (
        args.output_dir.parent / f".{args.output_dir.name}.parallel_workers"
    )
    worker_dirs = _run_workers(
        args,
        partial_root=partial_root,
        expected_maps=len(map_paths),
        checkpoint_sha256=checkpoint_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    outputs = _merge_workers(
        args,
        worker_dirs,
        map_paths=map_paths,
        dataset_root=dataset_root,
    )
    print(f"Parallel evaluation complete: {len(map_paths)} maps")
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
