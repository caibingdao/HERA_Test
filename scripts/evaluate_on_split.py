#!/usr/bin/env python3
"""Model-agnostic evaluation driver for a fixed map split.

The driver owns experiment orchestration and the stable CSV schema. Model and
environment specifics live behind a small adapter interface:

``create_adapter(checkpoint, deterministic, adapter_args)`` returns an object
with ``metadata()`` and ``evaluate_episode(...)`` methods. This keeps the test
runner reusable for policies outside ARIADNE/Mamba.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import traceback
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import (  # noqa: E402
    coverage_success,
    load_split_manifest,
    sha256_artifact,
    sha256_file,
    stable_seed,
    write_json,
)


EPISODE_FIELDS = [
    "model_name",
    "checkpoint",
    "checkpoint_sha256",
    "map_id",
    "model_seed",
    "eval_seed",
    "start_position_seed",
    "episode_id",
    "start_x",
    "start_y",
    "start_cell_x",
    "start_cell_y",
    "start_resample_count",
    "success",
    "coverage_success",
    "environment_success",
    "termination_reason",
    "path_length",
    "steps",
    "final_coverage",
    "target_coverage",
    "path_length_to_target",
    "steps_to_target",
    "elapsed_time_seconds",
    "mean_inference_time_ms",
    "max_inference_time_ms",
    "error",
]

CURVE_FIELDS = [
    "model_name",
    "model_seed",
    "checkpoint_sha256",
    "map_id",
    "eval_seed",
    "episode_id",
    "step",
    "path_length",
    "coverage",
]

TERMINATION_REASONS = {
    "target_reached",
    "max_steps",
    "max_path_length",
    "no_valid_action",
    "environment_complete",
    "environment_failure",
    "collision",
    "exception",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def parse_adapter_args(values: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"adapter argument must have KEY=VALUE form, got {item!r}"
            )
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError("adapter argument key cannot be empty")
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed[key] = raw_value
    return parsed


def load_adapter_factory(specification: str):
    if ":" not in specification:
        raise ValueError(
            "--adapter must use 'python.module:factory' notation"
        )
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError(
            "--adapter must use 'python.module:factory' notation"
        )
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise AttributeError(
            f"adapter factory {attribute_name!r} not found in {module_name!r}"
        ) from exc
    if not callable(factory):
        raise TypeError(f"adapter factory is not callable: {specification}")
    return factory


def _atomic_write_csv(
    path: Path,
    rows: List[Mapping[str, Any]],
    preferred_fields: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in preferred_fields
        }
    )
    fieldnames = preferred_fields + extras
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _failure_episode(
    *,
    identity: Mapping[str, Any],
    target_coverage: float,
    elapsed: float,
    exc: BaseException,
) -> Dict[str, Any]:
    row = {field: math.nan for field in EPISODE_FIELDS}
    row.update(identity)
    row.update(
        {
            "success": False,
            "coverage_success": False,
            "environment_success": False,
            "termination_reason": "exception",
            "target_coverage": target_coverage,
            "path_length": math.nan,
            "elapsed_time_seconds": elapsed,
            "mean_inference_time_ms": math.nan,
            "max_inference_time_ms": math.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    parsed = _finite_number(value, name, minimum=0)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(parsed)


def _optional_non_negative_timing(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric or NaN") from exc
    if math.isnan(parsed):
        return parsed
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be non-negative and finite or NaN")
    return parsed


def _validate_episode(
    episode: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    target_coverage: float,
) -> Dict[str, Any]:
    if not isinstance(episode, Mapping):
        raise TypeError("adapter episode result must be a mapping")
    row = dict(episode)
    row.update(identity)

    required_adapter_fields = {
        "start_x",
        "start_y",
        "start_resample_count",
        "environment_success",
        "termination_reason",
        "path_length",
        "steps",
        "final_coverage",
        "elapsed_time_seconds",
        "mean_inference_time_ms",
        "max_inference_time_ms",
    }
    missing = sorted(required_adapter_fields - row.keys())
    if missing:
        raise ValueError(
            f"adapter episode result is missing fields: {missing}"
        )

    row["start_x"] = _finite_number(row["start_x"], "start_x")
    row["start_y"] = _finite_number(row["start_y"], "start_y")
    row["start_resample_count"] = _non_negative_integer(
        row["start_resample_count"],
        "start_resample_count",
    )
    if not isinstance(row["environment_success"], bool):
        raise ValueError("environment_success must be a boolean")
    termination_reason = str(row["termination_reason"]).strip()
    if termination_reason not in TERMINATION_REASONS:
        raise ValueError(
            "termination_reason must be one of "
            f"{sorted(TERMINATION_REASONS)}, got {termination_reason!r}"
        )
    if termination_reason == "exception":
        raise ValueError(
            "termination_reason='exception' is reserved for errors caught by "
            "the generic evaluation driver; adapters should raise the error "
            "or return environment_failure"
        )
    row["termination_reason"] = termination_reason
    row["path_length"] = _finite_number(
        row["path_length"],
        "path_length",
        minimum=0,
    )
    row["steps"] = _non_negative_integer(row["steps"], "steps")
    final_coverage = _finite_number(
        row["final_coverage"],
        "final_coverage",
        minimum=0,
        maximum=1,
    )
    row["final_coverage"] = final_coverage
    row["elapsed_time_seconds"] = _finite_number(
        row["elapsed_time_seconds"],
        "elapsed_time_seconds",
        minimum=0,
    )
    row["mean_inference_time_ms"] = _optional_non_negative_timing(
        row["mean_inference_time_ms"],
        "mean_inference_time_ms",
    )
    row["max_inference_time_ms"] = _optional_non_negative_timing(
        row["max_inference_time_ms"],
        "max_inference_time_ms",
    )
    if row["steps"] > 0 and (
        math.isnan(row["mean_inference_time_ms"])
        or math.isnan(row["max_inference_time_ms"])
    ):
        raise ValueError(
            "inference timing must be finite when at least one step ran"
        )
    if (
        not math.isnan(row["mean_inference_time_ms"])
        and not math.isnan(row["max_inference_time_ms"])
        and row["max_inference_time_ms"] + 1e-12
        < row["mean_inference_time_ms"]
    ):
        raise ValueError(
            "max_inference_time_ms cannot be below "
            "mean_inference_time_ms"
        )

    primary_success = coverage_success(final_coverage, target_coverage)
    row["target_coverage"] = target_coverage
    row["coverage_success"] = primary_success
    row["success"] = primary_success
    row.setdefault("error", "")
    # These two primary metrics are derived from the saved coverage curve by
    # the generic driver, rather than trusted from a model-specific adapter.
    row["path_length_to_target"] = math.nan
    row["steps_to_target"] = math.nan
    return row


def _validate_curve(
    rows: Any,
    identity: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("adapter must return a non-empty coverage_curve list")
    validated = []
    previous_step = -1
    previous_distance = -math.inf
    previous_coverage = -math.inf
    for item in rows:
        if not isinstance(item, Mapping):
            raise TypeError("coverage curve rows must be mappings")
        row = dict(item)
        for field in ("step", "path_length", "coverage"):
            if field not in row:
                raise ValueError(f"coverage curve row is missing {field!r}")
        step = _non_negative_integer(row["step"], "curve step")
        distance = _finite_number(
            row["path_length"],
            "curve path_length",
            minimum=0,
        )
        coverage = _finite_number(
            row["coverage"],
            "curve coverage",
            minimum=0,
            maximum=1,
        )
        if step != previous_step + 1:
            raise ValueError(
                "coverage curve must contain one row for every consecutive "
                "decision step"
            )
        if (
            distance + 1e-9 < previous_distance
            or coverage + 1e-12 < previous_coverage
        ):
            raise ValueError(
                "coverage curve path_length and coverage must be "
                "non-decreasing"
            )
        previous_step = step
        previous_distance = distance
        previous_coverage = coverage
        row["step"] = step
        row["path_length"] = distance
        row["coverage"] = coverage
        row.update(identity)
        validated.append(row)
    if validated[0]["step"] != 0:
        raise ValueError("coverage curve must begin at decision step 0")
    if not math.isclose(
        validated[0]["path_length"],
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("coverage curve must begin at path_length 0")
    return validated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate any model adapter on the test side of a fixed map "
            "manifest and emit a model-independent result schema."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument(
        "--split-name",
        default="test",
        choices=("test",),
        help="Evaluation is deliberately restricted to held-out test maps.",
    )
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=[100, 101, 102, 103, 104],
    )
    parser.add_argument("--episodes-per-map", type=positive_int, default=5)
    parser.add_argument(
        "--episode-id-offset",
        type=non_negative_int,
        default=0,
        help=(
            "Offset applied to saved episode IDs and start-position seeds. "
            "Used by the parallel evaluation orchestrator."
        ),
    )
    parser.add_argument(
        "--target-coverage",
        type=probability,
        default=0.99,
    )
    parser.add_argument("--max-steps", type=positive_int, default=128)
    parser.add_argument(
        "--max-path-length",
        type=float,
        default=None,
        help="Optional positive distance budget in metres.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adapter",
        default="evaluation_adapters.ariadne:create_adapter",
        help="Model adapter factory in python.module:callable form.",
    )
    parser.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable model-specific option passed to the adapter factory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first episode exception instead of recording it.",
    )
    return parser


def run_evaluation(args: argparse.Namespace) -> Dict[str, Path]:
    if args.split_name != "test":
        raise ValueError(
            "independent evaluation may only use split_name='test'"
        )
    if (
        isinstance(args.episodes_per_map, bool)
        or not isinstance(args.episodes_per_map, Integral)
        or args.episodes_per_map <= 0
    ):
        raise ValueError("episodes_per_map must be a positive integer")
    episode_id_offset = getattr(args, "episode_id_offset", 0)
    if (
        isinstance(episode_id_offset, bool)
        or not isinstance(episode_id_offset, Integral)
        or episode_id_offset < 0
    ):
        raise ValueError("episode_id_offset must be a non-negative integer")
    args.episode_id_offset = int(episode_id_offset)
    if (
        isinstance(args.max_steps, bool)
        or not isinstance(args.max_steps, Integral)
        or args.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    if isinstance(args.target_coverage, bool):
        raise ValueError("target_coverage must be finite and in (0, 1]")
    try:
        target_coverage = float(args.target_coverage)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_coverage must be finite and in (0, 1]"
        ) from exc
    if (
        not math.isfinite(target_coverage)
        or not 0 < target_coverage <= 1
    ):
        raise ValueError("target_coverage must be finite and in (0, 1]")
    args.target_coverage = target_coverage
    if not isinstance(args.deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint does not exist: {args.checkpoint}"
        )
    if args.max_path_length is not None:
        if isinstance(args.max_path_length, bool):
            raise ValueError(
                "--max-path-length must be finite and positive when supplied"
            )
        try:
            args.max_path_length = float(args.max_path_length)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--max-path-length must be finite and positive when supplied"
            ) from exc
        if (
            not math.isfinite(args.max_path_length)
            or args.max_path_length <= 0
        ):
            raise ValueError(
                "--max-path-length must be finite and positive when supplied"
            )
    if not args.eval_seeds or any(
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or seed < 0
        for seed in args.eval_seeds
    ):
        raise ValueError("--eval-seeds must contain non-negative integers")
    args.eval_seeds = [int(seed) for seed in args.eval_seeds]
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("--eval-seeds must not contain duplicates")
    if len(args.eval_seeds) != args.episodes_per_map:
        raise ValueError(
            "--episodes-per-map must equal the number of --eval-seeds; "
            "each map runs exactly one episode for each evaluation seed"
        )

    manifest, map_paths = load_split_manifest(
        args.split_file,
        args.split_name,
    )
    dataset_root = Path(manifest["_dataset_root"])
    checkpoint_sha256 = sha256_artifact(args.checkpoint)
    factory = load_adapter_factory(args.adapter)
    parsed_adapter_args = parse_adapter_args(args.adapter_arg)
    adapter = factory(
        checkpoint=args.checkpoint.resolve(),
        deterministic=args.deterministic,
        adapter_args=parsed_adapter_args,
    )
    try:
        adapter_metadata = dict(adapter.metadata())
        for required in ("model_name", "model_seed"):
            if required not in adapter_metadata:
                raise ValueError(
                    f"adapter metadata is missing {required!r}"
                )
        raw_model_name = adapter_metadata["model_name"]
        if not isinstance(raw_model_name, str):
            raise ValueError("adapter model_name must be a string")
        model_name = raw_model_name.strip()
        if not model_name:
            raise ValueError("adapter model_name must be non-empty")
        model_seed = adapter_metadata["model_seed"]
        if (
            isinstance(model_seed, bool)
            or not isinstance(model_seed, Integral)
            or model_seed < 0
        ):
            raise ValueError(
                "adapter model_seed must be a non-negative integer"
            )
        adapter_metadata["model_name"] = model_name
        adapter_metadata["model_seed"] = int(model_seed)
    except BaseException:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        raise
    config = {
        "target_coverage": args.target_coverage,
        "max_steps": args.max_steps,
        "max_path_length": args.max_path_length,
        "deterministic": args.deterministic,
    }
    episode_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    exception_count = 0

    try:
        for map_path in map_paths:
            map_id = map_path.relative_to(dataset_root).as_posix()
            for local_episode_id in range(args.episodes_per_map):
                episode_id = args.episode_id_offset + local_episode_id
                eval_seed = args.eval_seeds[local_episode_id]
                start_seed = stable_seed(
                    "evaluation_start",
                    map_id,
                    eval_seed,
                    episode_id,
                )
                identity = {
                    "model_name": adapter_metadata["model_name"],
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "map_id": map_id,
                    "model_seed": adapter_metadata["model_seed"],
                    "eval_seed": eval_seed,
                    "start_position_seed": start_seed,
                    "episode_id": episode_id,
                }
                started = time.perf_counter()
                try:
                    result = adapter.evaluate_episode(
                        map_path=map_path,
                        map_id=map_id,
                        eval_seed=eval_seed,
                        start_seed=start_seed,
                        episode_id=episode_id,
                        config=dict(config),
                    )
                    if not isinstance(result, Mapping):
                        raise TypeError(
                            "adapter evaluate_episode must return a mapping"
                        )
                    episode = _validate_episode(
                        result.get("episode"),
                        identity=identity,
                        target_coverage=args.target_coverage,
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
                        raise ValueError(
                            "final coverage-curve step does not match episode "
                            "steps"
                        )
                    if not math.isclose(
                        final_curve["path_length"],
                        episode["path_length"],
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "final coverage-curve path_length does not match "
                            "the episode"
                        )
                    if not math.isclose(
                        final_curve["coverage"],
                        episode["final_coverage"],
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(
                            "final coverage-curve coverage does not match "
                            "the episode"
                        )
                    first_target = next(
                        (
                            item
                            for item in curve
                            if coverage_success(
                                item["coverage"],
                                args.target_coverage,
                            )
                        ),
                        None,
                    )
                    if episode["coverage_success"]:
                        if first_target is None:
                            raise ValueError(
                                "successful episode has no target crossing "
                                "in its coverage curve"
                            )
                        if first_target["step"] != final_curve["step"]:
                            raise ValueError(
                                "evaluation must stop at the first target "
                                "coverage crossing"
                            )
                        episode["path_length_to_target"] = first_target[
                            "path_length"
                        ]
                        episode["steps_to_target"] = first_target["step"]
                    elif first_target is not None:
                        raise ValueError(
                            "failed episode coverage curve reaches the "
                            "target threshold"
                        )
                    target_termination = (
                        episode["termination_reason"] == "target_reached"
                    )
                    if target_termination != episode["coverage_success"]:
                        raise ValueError(
                            "termination_reason='target_reached' must match "
                            "coverage-based success"
                        )
                    if episode["steps"] > args.max_steps:
                        raise ValueError(
                            "episode steps exceed the configured max_steps"
                        )
                    if (
                        episode["termination_reason"] == "max_steps"
                        and episode["steps"] != args.max_steps
                    ):
                        raise ValueError(
                            "termination_reason='max_steps' requires steps "
                            "to equal the configured max_steps"
                        )
                    environment_termination = (
                        episode["termination_reason"]
                        == "environment_complete"
                    )
                    if (
                        environment_termination
                        and not episode["environment_success"]
                    ):
                        raise ValueError(
                            "environment_complete termination requires "
                            "environment_success=True"
                        )
                    if (
                        episode["environment_success"]
                        and episode["termination_reason"]
                        not in {"target_reached", "environment_complete"}
                    ):
                        raise ValueError(
                            "environment_success=True requires a target or "
                            "environment_complete termination"
                        )
                    if args.max_path_length is None:
                        if (
                            episode["termination_reason"]
                            == "max_path_length"
                        ):
                            raise ValueError(
                                "max_path_length termination requires a "
                                "configured path-length budget"
                            )
                    else:
                        crossed_before_final = any(
                            item["path_length"] >= args.max_path_length
                            for item in curve[:-1]
                        )
                        if crossed_before_final:
                            raise ValueError(
                                "evaluation continued after reaching the "
                                "configured path-length budget"
                            )
                        reached_path_budget = (
                            final_curve["path_length"]
                            >= args.max_path_length
                        )
                        path_termination = (
                            episode["termination_reason"]
                            == "max_path_length"
                        )
                        if path_termination != reached_path_budget and (
                            path_termination
                            or episode["termination_reason"]
                            in {
                                "max_steps",
                                "no_valid_action",
                            }
                        ):
                            raise ValueError(
                                "max_path_length termination does not match "
                                "the configured path-length budget"
                            )
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    exception_count += 1
                    episode = _failure_episode(
                        identity=identity,
                        target_coverage=args.target_coverage,
                        elapsed=time.perf_counter() - started,
                        exc=exc,
                    )
                    curve = []
                    print(
                        f"Episode failed: map={map_id}, "
                        f"episode={episode_id}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                episode_rows.append(episode)
                curve_rows.extend(curve)
                print(
                    f"[{len(episode_rows)}/{len(map_paths) * args.episodes_per_map}] "
                    f"{map_id} episode={episode_id} "
                    f"termination={episode['termination_reason']}"
                )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    curve_path = output_dir / "coverage_curve.csv"
    _atomic_write_csv(episodes_path, episode_rows, EPISODE_FIELDS)
    _atomic_write_csv(curve_path, curve_rows, CURVE_FIELDS)
    metadata_path = output_dir / "evaluation_metadata.json"
    write_json(
        metadata_path,
        {
            "adapter": args.adapter,
            "adapter_args": parsed_adapter_args,
            "adapter_metadata": adapter_metadata,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "split_manifest": str(args.split_file.resolve()),
            "split_manifest_sha256": sha256_file(args.split_file),
            "split_name": args.split_name,
            "num_maps": len(map_paths),
            "num_episodes": len(episode_rows),
            "num_exception_episodes": exception_count,
            "eval_seeds": args.eval_seeds,
            "episodes_per_map": args.episodes_per_map,
            "episode_id_offset": args.episode_id_offset,
            **config,
        },
    )
    outputs = {
        "episodes": episodes_path,
        "coverage_curve": curve_path,
        "metadata": metadata_path,
    }
    if episode_rows and exception_count == len(episode_rows):
        raise RuntimeError(
            "all evaluation episodes failed; diagnostic CSV/JSON files were "
            f"written to {output_dir.resolve()}"
        )
    return outputs


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_evaluation(args)
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
