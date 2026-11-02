#!/usr/bin/env python3
"""Aggregate map-split evaluation CSV files and render reproducible plots.

The script intentionally uses only the Python standard library, NumPy, and
Matplotlib.  Each input may be either:

* an ``episodes.csv`` file, in which case ``coverage_curve.csv`` is read from
  the same directory; or
* an evaluation directory.  A direct ``episodes.csv`` is preferred, otherwise
  the directory is searched recursively for files named ``episodes.csv``.

Model runs, identified by ``(model_name, model_seed, checkpoint_sha256)``, are
the independent units for cross-model confidence intervals.  Legacy inputs
without metadata or checkpoint hashes fall back to checkpoint paths.
Episode-level confidence intervals are also reported, but are named explicitly
so the two uncertainty levels cannot be confused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPISODE_CANONICAL_FIELDS = [
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
]

COVERAGE_CANONICAL_FIELDS = [
    "model_name",
    "checkpoint",
    "checkpoint_sha256",
    "map_id",
    "model_seed",
    "eval_seed",
    "episode_id",
    "step",
    "path_length",
    "coverage",
]

SUMMARY_FIELDS = [
    "num_maps",
    "num_episodes",
    "num_successes",
    "success_rate",
    "success_rate_ci95_low",
    "success_rate_ci95_high",
    "mean_final_coverage",
    "variance_final_coverage",
    "std_final_coverage",
    "median_final_coverage",
    "mean_final_coverage_ci95_low",
    "mean_final_coverage_ci95_high",
    "mean_path_length_all",
    "variance_path_length_all",
    "std_path_length_all",
    "median_path_length_all",
    "mean_path_length_all_ci95_low",
    "mean_path_length_all_ci95_high",
    "mean_path_length_success",
    "variance_path_length_success",
    "std_path_length_success",
    "median_path_length_success",
    "mean_steps",
    "variance_steps",
    "std_steps",
    "median_steps",
    "mean_steps_success",
    "variance_steps_success",
    "std_steps_success",
    "median_steps_success",
    "mean_path_length_to_target",
    "mean_steps_to_target",
    "mean_elapsed_time_seconds",
    "mean_inference_time_ms",
    "mean_max_inference_time_ms",
    "coverage_success_rate",
    "environment_success_rate",
]

COVERAGE_STEP_THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)

COVERAGE_THRESHOLD_FIELDS = [
    "model_seed",
    "model_name",
    "checkpoint",
    "checkpoint_sha256",
    "coverage_threshold",
    "coverage_threshold_percent",
    "num_episodes",
    "num_reached",
    "reach_rate",
    "mean_steps_to_threshold",
    "variance_steps_to_threshold",
    "std_steps_to_threshold",
]

EVALUATION_COMPATIBILITY_FIELDS = [
    "split_manifest_sha256",
    "split_name",
    "num_maps",
    "num_episodes",
    "eval_seeds",
    "episodes_per_map",
    "target_coverage",
    "deterministic",
    "max_steps",
    "max_path_length",
]


class AggregationError(ValueError):
    """Raised when an input file cannot be aggregated safely."""


class VarianceTracker:
    """Collect concise diagnostics for sample variances with too few values."""

    def __init__(self) -> None:
        self.count = 0
        self.examples: list[str] = []

    def note(self, context: str) -> None:
        self.count += 1
        if len(self.examples) < 5:
            self.examples.append(context)

    def report(self) -> None:
        if not self.count:
            return
        example_text = "; ".join(self.examples)
        print(
            "Note: sample variance was written as NaN for "
            f"{self.count} group/metric combinations with fewer than two "
            f"valid samples. Examples: {example_text}",
            file=sys.stderr,
        )


def _nonempty(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _canonical_seed(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _canonical_sha256(value: Any, *, field: str, location: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise AggregationError(
            f"{location}: field {field!r} must be a 64-character SHA-256 "
            f"hex digest; got {value!r}"
        )
    return text


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _as_bool(value: Any, *, field: str, location: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise AggregationError(
        f"{location}: field {field!r} must be a boolean or 0/1; got {value!r}"
    )


def _natural_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        numeric = float(text)
    except ValueError:
        return (1, text)
    if math.isfinite(numeric):
        return (0, numeric)
    return (1, text)


def _model_run_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the stable identity of one independently trained model run."""

    model_name = str(row.get("model_name", "")).strip() or "model"
    model_seed = _canonical_seed(row.get("model_seed", ""))
    checkpoint_sha256 = str(row.get("checkpoint_sha256", "")).strip().lower()
    checkpoint = str(row.get("checkpoint", "")).strip()
    return model_name, model_seed, checkpoint_sha256 or checkpoint


def _model_run_sort_key(
    identity: tuple[str, str, str],
) -> tuple[tuple[int, Any], tuple[int, Any], str]:
    return (
        _natural_key(identity[0]),
        _natural_key(identity[1]),
        identity[2],
    )


def _model_run_fields(
    identity: tuple[str, str, str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    model_name, model_seed, _artifact_identity = identity
    return {
        "model_name": model_name,
        "model_seed": model_seed,
        "checkpoint": _joined_values(rows, "checkpoint"),
        "checkpoint_sha256": _joined_values(rows, "checkpoint_sha256"),
    }


def _model_run_label(identity: tuple[str, str, str]) -> str:
    model_name, model_seed, artifact_identity = identity
    label = f"{model_name} | seed {model_seed}"
    if artifact_identity:
        if (
            len(artifact_identity) == 64
            and all(
                character in "0123456789abcdef"
                for character in artifact_identity.lower()
            )
        ):
            return f"{label}\nsha256:{artifact_identity[:12]}"
        checkpoint_path = Path(artifact_identity)
        short_checkpoint = checkpoint_path.name
        if checkpoint_path.parent.name:
            short_checkpoint = (
                f"{checkpoint_path.parent.name}/{short_checkpoint}"
            )
        label = f"{label}\n{short_checkpoint}"
    return label


def _joined_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> str:
    values = sorted(
        {
            str(row.get(field, "")).strip()
            for row in rows
            if str(row.get(field, "")).strip()
        }
    )
    return "|".join(values)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise AggregationError(f"Cannot read CSV file {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AggregationError(f"CSV file has no header: {path}")
        fields = [str(field).strip() for field in reader.fieldnames]
        if any(not field for field in fields):
            raise AggregationError(f"CSV file has an empty header field: {path}")
        rows = []
        for row in reader:
            cleaned = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in row.items()
                if key is not None
            }
            if any(value != "" for value in cleaned.values()):
                rows.append(cleaned)
    return fields, rows


def _discover_episode_files(inputs: Sequence[str | Path]) -> list[Path]:
    discovered: dict[Path, Path] = {}
    for raw_input in inputs:
        path = Path(raw_input).expanduser()
        if not path.exists():
            raise AggregationError(f"Evaluation input does not exist: {path}")
        if path.is_file():
            if path.name != "episodes.csv":
                raise AggregationError(
                    "A CSV input must be named episodes.csv so that its paired "
                    f"coverage_curve.csv can be found; got {path}"
                )
            candidates = [path]
        elif path.is_dir():
            direct = path / "episodes.csv"
            candidates = [direct] if direct.is_file() else sorted(
                path.rglob("episodes.csv")
            )
            if not candidates:
                raise AggregationError(
                    f"No episodes.csv was found under evaluation directory {path}"
                )
        else:
            raise AggregationError(f"Unsupported evaluation input: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            discovered.setdefault(resolved, candidate)

    return [discovered[key] for key in sorted(discovered, key=str)]


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError(
            f"Cannot read evaluation metadata {path}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise AggregationError(
            f"{path}: evaluation metadata must be a JSON object"
        )
    return metadata


def _metadata_positive_int(value: Any, *, field: str, location: str) -> int:
    if isinstance(value, bool):
        raise AggregationError(
            f"{location}: metadata field {field!r} must be a positive integer"
        )
    numeric = _as_float(value)
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or numeric <= 0
    ):
        raise AggregationError(
            f"{location}: metadata field {field!r} must be a positive integer; "
            f"got {value!r}"
        )
    return int(numeric)


def _normalize_evaluation_metadata(
    metadata: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    location = str(path)
    required_fields = [
        *EVALUATION_COMPATIBILITY_FIELDS,
        "checkpoint_sha256",
    ]
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise AggregationError(
            f"{location}: missing required evaluation metadata fields: "
            f"{', '.join(missing)}"
        )

    split_manifest_sha256 = _canonical_sha256(
        metadata["split_manifest_sha256"],
        field="split_manifest_sha256",
        location=location,
    )
    if not split_manifest_sha256:
        raise AggregationError(
            f"{location}: metadata field 'split_manifest_sha256' cannot be empty"
        )

    checkpoint_sha256 = _canonical_sha256(
        metadata["checkpoint_sha256"],
        field="checkpoint_sha256",
        location=location,
    )
    if not checkpoint_sha256:
        raise AggregationError(
            f"{location}: metadata field 'checkpoint_sha256' cannot be empty"
        )

    split_name = str(metadata["split_name"]).strip()
    if split_name != "test":
        raise AggregationError(
            f"{location}: metadata field 'split_name' must be 'test'; "
            f"got {metadata['split_name']!r}"
        )

    raw_eval_seeds = metadata["eval_seeds"]
    if not isinstance(raw_eval_seeds, list) or not raw_eval_seeds:
        raise AggregationError(
            f"{location}: metadata field 'eval_seeds' must be a non-empty list"
        )
    normalized_eval_seeds = []
    for seed in raw_eval_seeds:
        if isinstance(seed, bool):
            numeric_seed = math.nan
        else:
            numeric_seed = _as_float(seed)
        if (
            not math.isfinite(numeric_seed)
            or not numeric_seed.is_integer()
            or numeric_seed < 0
        ):
            raise AggregationError(
                f"{location}: metadata field 'eval_seeds' must contain "
                f"non-negative integers; got {seed!r}"
            )
        normalized_eval_seeds.append(str(int(numeric_seed)))
    eval_seeds = tuple(normalized_eval_seeds)
    if len(set(eval_seeds)) != len(eval_seeds):
        raise AggregationError(
            f"{location}: metadata field 'eval_seeds' contains duplicates"
        )

    target_coverage = _as_float(metadata["target_coverage"])
    if (
        not math.isfinite(target_coverage)
        or target_coverage <= 0.0
        or target_coverage > 1.0
    ):
        raise AggregationError(
            f"{location}: metadata field 'target_coverage' must be in (0, 1]; "
            f"got {metadata['target_coverage']!r}"
        )

    deterministic = metadata["deterministic"]
    if not isinstance(deterministic, bool):
        raise AggregationError(
            f"{location}: metadata field 'deterministic' must be boolean; "
            f"got {deterministic!r}"
        )

    raw_max_path_length = metadata["max_path_length"]
    if raw_max_path_length is None:
        max_path_length = None
    else:
        max_path_length = _as_float(raw_max_path_length)
        if not math.isfinite(max_path_length) or max_path_length <= 0.0:
            raise AggregationError(
                f"{location}: metadata field 'max_path_length' must be null "
                f"or positive; got {raw_max_path_length!r}"
            )

    num_maps = _metadata_positive_int(
        metadata["num_maps"],
        field="num_maps",
        location=location,
    )
    num_episodes = _metadata_positive_int(
        metadata["num_episodes"],
        field="num_episodes",
        location=location,
    )
    episodes_per_map = _metadata_positive_int(
        metadata["episodes_per_map"],
        field="episodes_per_map",
        location=location,
    )
    if episodes_per_map != len(eval_seeds):
        raise AggregationError(
            f"{location}: metadata episodes_per_map={episodes_per_map} must "
            f"equal the number of eval_seeds={len(eval_seeds)}"
        )
    expected_num_episodes = num_maps * episodes_per_map
    if num_episodes != expected_num_episodes:
        raise AggregationError(
            f"{location}: metadata num_episodes={num_episodes} does not equal "
            f"num_maps * episodes_per_map = {expected_num_episodes}"
        )

    return {
        "split_manifest_sha256": split_manifest_sha256,
        "split_name": split_name,
        "num_maps": num_maps,
        "num_episodes": num_episodes,
        "eval_seeds": eval_seeds,
        "episodes_per_map": episodes_per_map,
        "target_coverage": target_coverage,
        "deterministic": deterministic,
        "max_steps": _metadata_positive_int(
            metadata["max_steps"],
            field="max_steps",
            location=location,
        ),
        "max_path_length": max_path_length,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _load_and_validate_evaluation_metadata(
    episode_files: Sequence[Path],
) -> dict[Path, dict[str, Any] | None]:
    metadata_by_file: dict[Path, dict[str, Any] | None] = {}
    missing_paths = []

    for episodes_path in episode_files:
        resolved = episodes_path.resolve()
        metadata_path = episodes_path.with_name("evaluation_metadata.json")
        if not metadata_path.exists():
            metadata_by_file[resolved] = None
            missing_paths.append(metadata_path)
            continue
        metadata_by_file[resolved] = _normalize_evaluation_metadata(
            _read_metadata(metadata_path),
            path=metadata_path,
        )

    present_count = len(episode_files) - len(missing_paths)
    if present_count == 0:
        print(
            "Warning: no evaluation_metadata.json was found for any input; "
            "using legacy compatibility mode without cross-directory metadata "
            "checks. Model identity falls back to checkpoint path whenever "
            "checkpoint_sha256 is unavailable.",
            file=sys.stderr,
        )
        return metadata_by_file

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise AggregationError(
            "Every evaluation input must include evaluation_metadata.json "
            "when metadata is present for any input. Missing: "
            f"{missing}"
        )

    reference_path = episode_files[0].resolve()
    reference = metadata_by_file[reference_path]
    assert reference is not None
    for episodes_path in episode_files[1:]:
        resolved = episodes_path.resolve()
        current = metadata_by_file[resolved]
        assert current is not None
        differences = [
            field
            for field in EVALUATION_COMPATIBILITY_FIELDS
            if current[field] != reference[field]
        ]
        if differences:
            details = "; ".join(
                f"{field}: {reference[field]!r} != {current[field]!r}"
                for field in differences
            )
            raise AggregationError(
                "Incompatible evaluation metadata between "
                f"{reference_path.with_name('evaluation_metadata.json')} and "
                f"{resolved.with_name('evaluation_metadata.json')}: {details}"
            )

    return metadata_by_file


def _canonicalize_episode(
    row: Mapping[str, str],
    *,
    source_id: str,
    location: str,
    default_checkpoint_sha256: str,
) -> dict[str, Any]:
    record: dict[str, Any] = dict(row)

    map_id = _nonempty(row, ("map_id", "map", "map_path"))
    if not map_id:
        raise AggregationError(f"{location}: missing required map_id")
    record["map_id"] = map_id

    model_seed = _canonical_seed(
        _nonempty(row, ("model_seed", "training_seed", "train_seed"))
    )
    if not model_seed:
        raise AggregationError(
            f"{location}: missing required model_seed (or training_seed)"
        )
    record["model_seed"] = model_seed

    success_name = next(
        (
            name
            for name in ("success", "coverage_success")
            if _nonempty(row, (name,))
        ),
        "",
    )
    if not success_name:
        raise AggregationError(
            f"{location}: missing required success (or coverage_success)"
        )
    success = _as_bool(
        row[success_name],
        field=success_name,
        location=location,
    )
    record["success"] = int(success)

    boolean_aliases = {
        "coverage_success": ("coverage_success",),
        "environment_success": ("environment_success",),
    }
    for canonical, aliases in boolean_aliases.items():
        value = _nonempty(row, aliases)
        if value:
            record[canonical] = int(
                _as_bool(value, field=canonical, location=location)
            )

    aliases = {
        "model_name": ("model_name", "model"),
        "checkpoint": ("checkpoint", "checkpoint_path"),
        "checkpoint_sha256": ("checkpoint_sha256",),
        "eval_seed": ("eval_seed", "evaluation_seed"),
        "start_position_seed": ("start_position_seed", "start_seed"),
        "episode_id": ("episode_id", "episode"),
        "path_length": ("path_length", "travel_dist", "travel_distance"),
        "steps": ("steps", "num_steps", "decision_steps"),
        "final_coverage": (
            "final_coverage",
            "explored_rate",
            "exploration_rate",
        ),
        "target_coverage": ("target_coverage",),
        "path_length_to_target": ("path_length_to_target",),
        "steps_to_target": ("steps_to_target",),
        "elapsed_time_seconds": ("elapsed_time_seconds", "elapsed_seconds"),
        "mean_inference_time_ms": (
            "mean_inference_time_ms",
            "inference_time_mean_ms",
        ),
        "max_inference_time_ms": (
            "max_inference_time_ms",
            "inference_time_max_ms",
        ),
    }
    for canonical, candidates in aliases.items():
        value = _nonempty(row, candidates)
        if value or canonical not in record:
            record[canonical] = value

    row_checkpoint_sha256 = _canonical_sha256(
        record.get("checkpoint_sha256", ""),
        field="checkpoint_sha256",
        location=location,
    )
    if (
        default_checkpoint_sha256
        and row_checkpoint_sha256
        and row_checkpoint_sha256 != default_checkpoint_sha256
    ):
        raise AggregationError(
            f"{location}: checkpoint_sha256 does not match "
            "evaluation_metadata.json"
        )
    record["checkpoint_sha256"] = (
        default_checkpoint_sha256 or row_checkpoint_sha256
    )

    for field in ("eval_seed", "episode_id"):
        record[field] = _canonical_seed(record.get(field, ""))
        if not record[field]:
            raise AggregationError(f"{location}: missing required {field}")
    record["start_position_seed"] = _canonical_seed(
        record.get("start_position_seed", "")
    )

    if not record.get("model_name"):
        checkpoint = str(record.get("checkpoint", "")).strip()
        record["model_name"] = Path(checkpoint).parent.name if checkpoint else "model"

    record["_source_id"] = source_id
    record["_source_location"] = location
    return record


def _canonicalize_curve_row(
    row: Mapping[str, str],
    *,
    source_id: str,
    location: str,
    default_model_seed: str,
    default_model_name: str,
    default_checkpoint: str,
    default_checkpoint_sha256: str,
    checkpoint_by_model: Mapping[tuple[str, str], Any],
    checkpoint_by_model_run: Mapping[tuple[str, str, str], Any],
    checkpoint_sha256_by_model: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    record: dict[str, Any] = dict(row)
    aliases = {
        "map_id": ("map_id", "map", "map_path"),
        "model_seed": ("model_seed", "training_seed", "train_seed"),
        "model_name": ("model_name", "model"),
        "checkpoint": ("checkpoint", "checkpoint_path"),
        "checkpoint_sha256": ("checkpoint_sha256",),
        "eval_seed": ("eval_seed", "evaluation_seed"),
        "episode_id": ("episode_id", "episode"),
        "step": ("step", "decision_step"),
        "path_length": ("path_length", "travel_dist", "travel_distance"),
        "coverage": ("coverage", "explored_rate", "exploration_rate"),
    }
    for canonical, candidates in aliases.items():
        value = _nonempty(row, candidates)
        if value or canonical not in record:
            record[canonical] = value

    record["model_seed"] = _canonical_seed(
        record.get("model_seed") or default_model_seed
    )
    record["model_name"] = (
        str(record.get("model_name", "")).strip() or default_model_name
    )
    record["eval_seed"] = _canonical_seed(record.get("eval_seed", ""))
    record["episode_id"] = _canonical_seed(record.get("episode_id", ""))

    model_key = (record["model_name"], record["model_seed"])
    row_checkpoint_sha256 = _canonical_sha256(
        record.get("checkpoint_sha256", ""),
        field="checkpoint_sha256",
        location=location,
    )
    if (
        default_checkpoint_sha256
        and row_checkpoint_sha256
        and row_checkpoint_sha256 != default_checkpoint_sha256
    ):
        raise AggregationError(
            f"{location}: checkpoint_sha256 does not match "
            "evaluation_metadata.json"
        )
    if not row_checkpoint_sha256 and model_key in checkpoint_sha256_by_model:
        inferred_sha256 = checkpoint_sha256_by_model[model_key]
        if inferred_sha256 is None:
            raise AggregationError(
                f"{location}: checkpoint_sha256 is required because the "
                "paired episodes.csv has multiple checkpoint hashes for "
                f"model {record['model_name']!r}, "
                f"seed {record['model_seed']!r}"
            )
        row_checkpoint_sha256 = str(inferred_sha256)
    record["checkpoint_sha256"] = (
        default_checkpoint_sha256 or row_checkpoint_sha256
    )

    checkpoint = str(record.get("checkpoint", "")).strip()
    model_run_key = (
        record["model_name"],
        record["model_seed"],
        record["checkpoint_sha256"],
    )
    if not checkpoint and record["checkpoint_sha256"]:
        if model_run_key in checkpoint_by_model_run:
            inferred_checkpoint = checkpoint_by_model_run[model_run_key]
            if inferred_checkpoint is None:
                raise AggregationError(
                    f"{location}: checkpoint is required because the paired "
                    "episodes.csv has multiple paths for model run "
                    f"{model_run_key!r}"
                )
            checkpoint = str(inferred_checkpoint)
    elif not checkpoint and model_key in checkpoint_by_model:
        inferred_checkpoint = checkpoint_by_model[model_key]
        if inferred_checkpoint is None:
            raise AggregationError(
                f"{location}: checkpoint is required because the paired "
                "episodes.csv has multiple checkpoints for model "
                f"{record['model_name']!r}, seed {record['model_seed']!r}"
            )
        checkpoint = str(inferred_checkpoint)
    record["checkpoint"] = checkpoint or default_checkpoint

    for required in ("map_id", "model_seed", "eval_seed", "episode_id"):
        if not str(record.get(required, "")).strip():
            raise AggregationError(f"{location}: missing required {required}")
    for required in ("step", "path_length", "coverage"):
        value = _as_float(record.get(required))
        if not math.isfinite(value):
            raise AggregationError(
                f"{location}: {required} must be a finite number; "
                f"got {record.get(required)!r}"
            )

    record["_source_id"] = source_id
    record["_source_location"] = location
    return record


def _episode_grid_identity(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    return (
        str(row.get("map_id", "")).strip(),
        _canonical_seed(row.get("eval_seed", "")),
        _canonical_seed(row.get("episode_id", "")),
    )


def _episode_grid_sort_key(
    identity: tuple[str, str, str],
) -> tuple[tuple[int, Any], tuple[int, Any], tuple[int, Any]]:
    return tuple(_natural_key(value) for value in identity)


def _short_grid_difference(
    identities: Iterable[tuple[str, str, str]],
) -> str:
    ordered = sorted(identities, key=_episode_grid_sort_key)
    preview = ", ".join(repr(identity) for identity in ordered[:5])
    if len(ordered) > 5:
        preview = f"{preview}, ... ({len(ordered)} total)"
    return preview or "none"


def _validate_episode_grids_and_curves(
    episodes: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    metadata_by_file: Mapping[Path, Mapping[str, Any] | None],
) -> None:
    grids: MutableMapping[
        tuple[str, str, str],
        dict[tuple[str, str, str], Mapping[str, Any]],
    ] = defaultdict(dict)
    source_episode_identities = set()

    for row in episodes:
        model_identity = _model_run_identity(row)
        if not model_identity[2]:
            raise AggregationError(
                f"{row.get('_source_location', 'episode row')}: model run "
                "requires checkpoint_sha256, or checkpoint path in legacy "
                "compatibility mode"
            )
        grid_identity = _episode_grid_identity(row)
        prior = grids[model_identity].get(grid_identity)
        if prior is not None:
            raise AggregationError(
                f"{row.get('_source_location', 'episode row')}: duplicate "
                f"episode identity {grid_identity!r} for model run "
                f"{model_identity!r}; first seen at "
                f"{prior.get('_source_location', 'episode row')}"
            )
        grids[model_identity][grid_identity] = row
        source_episode_identities.add(
            (
                str(row.get("_source_id", "")),
                model_identity,
                grid_identity,
            )
        )

    model_identities = sorted(grids, key=_model_run_sort_key)
    if model_identities:
        reference_identity = model_identities[0]
        reference_grid = set(grids[reference_identity])
        for model_identity in model_identities[1:]:
            current_grid = set(grids[model_identity])
            if current_grid == reference_grid:
                continue
            raise AggregationError(
                "Episode grids differ across model runs. "
                f"Reference {reference_identity!r}; current "
                f"{model_identity!r}; missing from current: "
                f"{_short_grid_difference(reference_grid - current_grid)}; "
                "extra in current: "
                f"{_short_grid_difference(current_grid - reference_grid)}"
            )

    episodes_by_source: MutableMapping[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in episodes:
        episodes_by_source[str(row.get("_source_id", ""))].append(row)

    for episodes_path, metadata in metadata_by_file.items():
        if metadata is None:
            continue
        source_id = str(episodes_path)
        source_rows = episodes_by_source[source_id]
        metadata_location = episodes_path.with_name(
            "evaluation_metadata.json"
        )
        for row in source_rows:
            if not _canonical_seed(row.get("start_position_seed", "")):
                raise AggregationError(
                    f"{row.get('_source_location', 'episode row')}: missing "
                    "required start_position_seed for formal evaluation output"
                )
        if len(source_rows) != metadata["num_episodes"]:
            raise AggregationError(
                f"{metadata_location}: actual episode row count "
                f"{len(source_rows)} does not match metadata num_episodes="
                f"{metadata['num_episodes']}"
            )

        maps = {
            str(row.get("map_id", "")).strip() for row in source_rows
        }
        if len(maps) != metadata["num_maps"]:
            raise AggregationError(
                f"{metadata_location}: actual map count {len(maps)} does "
                f"not match metadata num_maps={metadata['num_maps']}"
            )

        eval_seeds = metadata["eval_seeds"]
        expected_per_map = {
            (eval_seed, str(episode_id))
            for episode_id, eval_seed in enumerate(eval_seeds)
        }
        for map_id in sorted(maps, key=_natural_key):
            actual_per_map = [
                (
                    _canonical_seed(row.get("eval_seed", "")),
                    _canonical_seed(row.get("episode_id", "")),
                )
                for row in source_rows
                if str(row.get("map_id", "")).strip() == map_id
            ]
            if (
                len(actual_per_map) != len(expected_per_map)
                or set(actual_per_map) != expected_per_map
            ):
                raise AggregationError(
                    f"{metadata_location}: map {map_id!r} does not realize "
                    "the metadata evaluation schedule; expected "
                    f"{sorted(expected_per_map, key=_episode_grid_sort_key)!r}, "
                    f"got {sorted(actual_per_map, key=_episode_grid_sort_key)!r}"
                )

    formal_source_ids = {
        str(path)
        for path, metadata in metadata_by_file.items()
        if metadata is not None
    }
    if formal_source_ids and model_identities:
        reference_grid = set(grids[model_identities[0]])
        for grid_identity in sorted(
            reference_grid,
            key=_episode_grid_sort_key,
        ):
            start_seeds = {
                model_identity: _canonical_seed(
                    grids[model_identity][grid_identity].get(
                        "start_position_seed",
                        "",
                    )
                )
                for model_identity in model_identities
                if str(
                    grids[model_identity][grid_identity].get(
                        "_source_id",
                        "",
                    )
                )
                in formal_source_ids
            }
            if len(set(start_seeds.values())) != 1:
                raise AggregationError(
                    "start_position_seed differs across model runs for "
                    f"episode grid {grid_identity!r}: {start_seeds!r}"
                )

            start_positions = {}
            for model_identity in model_identities:
                row = grids[model_identity][grid_identity]
                if (
                    str(row.get("_source_id", ""))
                    not in formal_source_ids
                    or str(row.get("termination_reason", "")).strip().lower()
                    == "exception"
                ):
                    continue
                start_x = _as_float(row.get("start_x"))
                start_y = _as_float(row.get("start_y"))
                if not math.isfinite(start_x) or not math.isfinite(start_y):
                    raise AggregationError(
                        f"{row.get('_source_location', 'episode row')}: "
                        "formal non-exception episode requires finite "
                        "start_x and start_y"
                    )
                start_positions[model_identity] = (start_x, start_y)
            if len(start_positions) > 1:
                reference_identity = next(iter(start_positions))
                reference_position = start_positions[reference_identity]
                mismatched_positions = {
                    identity: position
                    for identity, position in start_positions.items()
                    if not (
                        math.isclose(
                            position[0],
                            reference_position[0],
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                        and math.isclose(
                            position[1],
                            reference_position[1],
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                    )
                }
                if mismatched_positions:
                    raise AggregationError(
                        "recorded start_x/start_y differ across model runs "
                        f"for episode grid {grid_identity!r}: "
                        f"{start_positions!r}"
                    )

    covered_episode_identities = set()
    for row in coverage_rows:
        curve_identity = (
            str(row.get("_source_id", "")),
            _model_run_identity(row),
            _episode_grid_identity(row),
        )
        if curve_identity not in source_episode_identities:
            raise AggregationError(
                f"{row.get('_source_location', 'coverage curve row')}: "
                "coverage curve row references no episode identity "
                f"{curve_identity[1:]!r} in its paired episodes.csv"
            )
        covered_episode_identities.add(curve_identity)

    for row in episodes:
        if str(row.get("termination_reason", "")).strip().lower() == "exception":
            continue
        episode_identity = (
            str(row.get("_source_id", "")),
            _model_run_identity(row),
            _episode_grid_identity(row),
        )
        if episode_identity not in covered_episode_identities:
            raise AggregationError(
                f"{row.get('_source_location', 'episode row')}: coverage "
                "curve is missing for non-exception episode identity "
                f"{episode_identity[1:]!r}"
            )


def load_evaluation_inputs(
    inputs: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Load and canonicalize all paired episode and coverage CSV files."""

    episode_files = _discover_episode_files(inputs)
    metadata_by_file = _load_and_validate_evaluation_metadata(episode_files)
    episodes: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for episodes_path in episode_files:
        resolved_episodes_path = episodes_path.resolve()
        source_id = str(resolved_episodes_path)
        metadata = metadata_by_file[resolved_episodes_path]
        default_checkpoint_sha256 = (
            str(metadata["checkpoint_sha256"]) if metadata is not None else ""
        )
        _, raw_episodes = _read_csv(episodes_path)
        if not raw_episodes:
            raise AggregationError(f"No episode rows found in {episodes_path}")

        source_episodes = [
            _canonicalize_episode(
                row,
                source_id=source_id,
                location=f"{episodes_path}:row {index}",
                default_checkpoint_sha256=default_checkpoint_sha256,
            )
            for index, row in enumerate(raw_episodes, start=2)
        ]
        episodes.extend(source_episodes)

        curve_path = episodes_path.with_name("coverage_curve.csv")
        if not curve_path.is_file():
            raise AggregationError(
                f"Missing paired coverage curve file for {episodes_path}: "
                f"expected {curve_path}"
            )
        _, raw_curves = _read_csv(curve_path)

        seeds = sorted(
            {str(row["model_seed"]) for row in source_episodes},
            key=_natural_key,
        )
        if len(seeds) != 1 and any(
            not _nonempty(row, ("model_seed", "training_seed", "train_seed"))
            for row in raw_curves
        ):
            raise AggregationError(
                f"{curve_path}: model_seed is required because the paired "
                "episodes.csv contains more than one model seed"
            )
        default_seed = seeds[0] if len(seeds) == 1 else ""

        model_names = sorted(
            {
                str(row.get("model_name", "")).strip()
                for row in source_episodes
                if str(row.get("model_name", "")).strip()
            }
        )
        checkpoints = sorted(
            {
                str(row.get("checkpoint", "")).strip()
                for row in source_episodes
                if str(row.get("checkpoint", "")).strip()
            }
        )
        default_name = model_names[0] if len(model_names) == 1 else "model"
        default_checkpoint = checkpoints[0] if len(checkpoints) == 1 else ""
        checkpoint_candidates: MutableMapping[
            tuple[str, str],
            set[str],
        ] = defaultdict(set)
        checkpoint_by_model_run_candidates: MutableMapping[
            tuple[str, str, str],
            set[str],
        ] = defaultdict(set)
        checkpoint_sha256_candidates: MutableMapping[
            tuple[str, str],
            set[str],
        ] = defaultdict(set)
        for episode in source_episodes:
            model_key = (
                str(episode["model_name"]),
                str(episode["model_seed"]),
            )
            checkpoint_candidates[model_key].add(
                str(episode.get("checkpoint", "")).strip()
            )
            checkpoint_by_model_run_candidates[
                _model_run_identity(episode)
            ].add(str(episode.get("checkpoint", "")).strip())
            checkpoint_sha256_candidates[model_key].add(
                str(episode.get("checkpoint_sha256", "")).strip()
            )
        checkpoint_by_model = {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in checkpoint_candidates.items()
        }
        checkpoint_by_model_run = {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in checkpoint_by_model_run_candidates.items()
        }
        checkpoint_sha256_by_model = {
            key: next(iter(values)) if len(values) == 1 else None
            for key, values in checkpoint_sha256_candidates.items()
        }

        coverage_rows.extend(
            _canonicalize_curve_row(
                row,
                source_id=source_id,
                location=f"{curve_path}:row {index}",
                default_model_seed=default_seed,
                default_model_name=default_name,
                default_checkpoint=default_checkpoint,
                default_checkpoint_sha256=default_checkpoint_sha256,
                checkpoint_by_model=checkpoint_by_model,
                checkpoint_by_model_run=checkpoint_by_model_run,
                checkpoint_sha256_by_model=checkpoint_sha256_by_model,
            )
            for index, row in enumerate(raw_curves, start=2)
        )

    _validate_episode_grids_and_curves(
        episodes,
        coverage_rows,
        metadata_by_file,
    )
    return episodes, coverage_rows, episode_files


def _finite_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    successful_only: bool = False,
) -> np.ndarray:
    values = []
    for row in rows:
        if successful_only and not _row_success(row):
            continue
        value = _as_float(row.get(field))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _row_success(row: Mapping[str, Any]) -> bool:
    return _as_bool(
        row.get("success"),
        field="success",
        location=str(row.get("_source_location", "episode row")),
    )


def sample_variance(values: Iterable[float]) -> float:
    """Return the sample variance (ddof=1), or NaN for fewer than 2 values."""

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return math.nan
    return float(np.var(array, ddof=1))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else math.nan


def _distribution(
    values: np.ndarray,
    *,
    tracker: VarianceTracker,
    context: str,
) -> tuple[float, float, float, float]:
    mean = _mean(values)
    variance = sample_variance(values)
    if values.size < 2:
        tracker.note(context)
    std = math.sqrt(variance) if math.isfinite(variance) else math.nan
    return mean, variance, std, _median(values)


def _seed_for_label(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    label: str,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a mean CI with deterministic, label-derived randomness."""

    array = np.asarray(list(values), dtype=np.float64)
    array = np.sort(array[np.isfinite(array)])
    if not array.size:
        return math.nan, math.nan
    if array.size == 1:
        value = float(array[0])
        return value, value
    if bootstrap_samples <= 0:
        raise AggregationError("bootstrap_samples must be positive")
    if not math.isclose(confidence_level, 0.95):
        raise AggregationError(
            "confidence_level must be 0.95 because output columns are "
            "explicitly named ci95"
        )

    rng = np.random.default_rng(_seed_for_label(bootstrap_seed, label))
    means = np.empty(bootstrap_samples, dtype=np.float64)
    # Chunking bounds peak memory for experiments with thousands of maps.
    chunk_size = max(1, min(64, bootstrap_samples))
    offset = 0
    while offset < bootstrap_samples:
        count = min(chunk_size, bootstrap_samples - offset)
        sample_indices = rng.integers(
            0,
            array.size,
            size=(count, array.size),
        )
        means[offset : offset + count] = np.mean(
            array[sample_indices],
            axis=1,
        )
        offset += count

    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def _optional_boolean_rate(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    values = []
    for row in rows:
        raw = row.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            values.append(
                1.0
                if _as_bool(
                    raw,
                    field=field,
                    location=str(row.get("_source_location", "episode row")),
                )
                else 0.0
            )
        except AggregationError:
            raise
    return _mean(np.asarray(values, dtype=np.float64))


def summarize_episodes(
    rows: Sequence[Mapping[str, Any]],
    *,
    tracker: VarianceTracker | None = None,
    context: str = "episodes",
    bootstrap_samples: int | None = None,
    bootstrap_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute the common episode statistics for one aggregation group."""

    if not rows:
        raise AggregationError(f"Cannot summarize an empty group: {context}")
    if tracker is None:
        tracker = VarianceTracker()

    successes = np.asarray(
        [1.0 if _row_success(row) else 0.0 for row in rows],
        dtype=np.float64,
    )
    final_coverage = _finite_values(rows, "final_coverage")
    path_all = _finite_values(rows, "path_length")
    path_success = _finite_values(rows, "path_length", successful_only=True)
    steps_all = _finite_values(rows, "steps")
    steps_success = _finite_values(rows, "steps", successful_only=True)

    (
        mean_final,
        var_final,
        std_final,
        median_final,
    ) = _distribution(
        final_coverage,
        tracker=tracker,
        context=f"{context}: final_coverage",
    )
    mean_path, var_path, std_path, median_path = _distribution(
        path_all,
        tracker=tracker,
        context=f"{context}: path_length_all",
    )
    (
        mean_path_success,
        var_path_success,
        std_path_success,
        median_path_success,
    ) = _distribution(
        path_success,
        tracker=tracker,
        context=f"{context}: path_length_success",
    )
    mean_steps, var_steps, std_steps, median_steps = _distribution(
        steps_all,
        tracker=tracker,
        context=f"{context}: steps",
    )
    (
        mean_steps_success,
        var_steps_success,
        std_steps_success,
        median_steps_success,
    ) = _distribution(
        steps_success,
        tracker=tracker,
        context=f"{context}: steps_success",
    )

    result: dict[str, Any] = {
        "num_maps": len({str(row["map_id"]) for row in rows}),
        "num_episodes": len(rows),
        "num_successes": int(np.sum(successes)),
        "success_rate": _mean(successes),
        "mean_final_coverage": mean_final,
        "variance_final_coverage": var_final,
        "std_final_coverage": std_final,
        "median_final_coverage": median_final,
        "mean_path_length_all": mean_path,
        "variance_path_length_all": var_path,
        "std_path_length_all": std_path,
        "median_path_length_all": median_path,
        "mean_path_length_success": mean_path_success,
        "variance_path_length_success": var_path_success,
        "std_path_length_success": std_path_success,
        "median_path_length_success": median_path_success,
        "mean_steps": mean_steps,
        "variance_steps": var_steps,
        "std_steps": std_steps,
        "median_steps": median_steps,
        "mean_steps_success": mean_steps_success,
        "variance_steps_success": var_steps_success,
        "std_steps_success": std_steps_success,
        "median_steps_success": median_steps_success,
        "mean_path_length_to_target": _mean(
            _finite_values(rows, "path_length_to_target")
        ),
        "mean_steps_to_target": _mean(
            _finite_values(rows, "steps_to_target")
        ),
        "mean_elapsed_time_seconds": _mean(
            _finite_values(rows, "elapsed_time_seconds")
        ),
        "mean_inference_time_ms": _mean(
            _finite_values(rows, "mean_inference_time_ms")
        ),
        "mean_max_inference_time_ms": _mean(
            _finite_values(rows, "max_inference_time_ms")
        ),
        "coverage_success_rate": _optional_boolean_rate(
            rows,
            "coverage_success",
        ),
        "environment_success_rate": _optional_boolean_rate(
            rows,
            "environment_success",
        ),
    }

    if bootstrap_samples is not None:
        result[
            "success_rate_ci95_low"
        ], result["success_rate_ci95_high"] = bootstrap_mean_ci(
            successes,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{context}:success_rate",
            confidence_level=confidence_level,
        )
        result[
            "mean_path_length_all_ci95_low"
        ], result["mean_path_length_all_ci95_high"] = bootstrap_mean_ci(
            path_all,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{context}:mean_path_length_all",
            confidence_level=confidence_level,
        )
        result[
            "mean_final_coverage_ci95_low"
        ], result["mean_final_coverage_ci95_high"] = bootstrap_mean_ci(
            final_coverage,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"{context}:mean_final_coverage",
            confidence_level=confidence_level,
        )

    return result


def build_map_summary(
    episodes: Sequence[Mapping[str, Any]],
    *,
    tracker: VarianceTracker,
) -> list[dict[str, Any]]:
    groups: MutableMapping[
        tuple[tuple[str, str, str], str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in episodes:
        groups[(_model_run_identity(row), str(row["map_id"]))].append(row)

    output = []
    for identity, map_id in sorted(
        groups,
        key=lambda item: (
            _model_run_sort_key(item[0]),
            _natural_key(item[1]),
        ),
    ):
        rows = groups[(identity, map_id)]
        summary = _model_run_fields(identity, rows)
        summary["map_id"] = map_id
        summary.update(
            summarize_episodes(
                rows,
                tracker=tracker,
                context=(
                    f"model_name={identity[0]!r}, "
                    f"model_seed={identity[1]!r}, "
                    f"checkpoint_identity={identity[2]!r}, "
                    f"map_id={map_id!r}"
                ),
            )
        )
        output.append(summary)
    return output


def build_model_summary(
    episodes: Sequence[Mapping[str, Any]],
    *,
    tracker: VarianceTracker,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    groups: MutableMapping[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in episodes:
        groups[_model_run_identity(row)].append(row)

    output = []
    for identity in sorted(groups, key=_model_run_sort_key):
        rows = groups[identity]
        summary = _model_run_fields(identity, rows)
        summary.update(
            summarize_episodes(
                rows,
                tracker=tracker,
                context=(
                    f"model_name={identity[0]!r}, "
                    f"model_seed={identity[1]!r}, "
                    f"checkpoint_identity={identity[2]!r}"
                ),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        )
        output.append(summary)
    return output


def build_overall_summary(
    episodes: Sequence[Mapping[str, Any]],
    model_summaries: Sequence[Mapping[str, Any]],
    *,
    tracker: VarianceTracker,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    result = summarize_episodes(
        episodes,
        tracker=tracker,
        context="all episodes",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    result["num_model_seeds"] = len(
        {str(summary["model_seed"]) for summary in model_summaries}
    )
    result["num_model_runs"] = len(model_summaries)
    result["num_model_names"] = len(
        {str(summary["model_name"]) for summary in model_summaries}
    )
    result["bootstrap_seed"] = bootstrap_seed
    result["bootstrap_samples"] = bootstrap_samples
    result["confidence_level"] = confidence_level
    result["cross_model_resampling_unit"] = (
        "model_run (model_name, model_seed, checkpoint_sha256)"
    )
    result["model_run_identity_fields"] = (
        "model_name,model_seed,checkpoint_sha256"
    )

    cross_model_metrics = {
        "success_rate": "success_rate",
        "mean_path_length_all": "mean_path_length_all",
        "mean_final_coverage": "mean_final_coverage",
    }
    for output_name, model_field in cross_model_metrics.items():
        values = np.asarray(
            [
                _as_float(summary.get(model_field))
                for summary in model_summaries
                if math.isfinite(_as_float(summary.get(model_field)))
            ],
            dtype=np.float64,
        )
        mean, variance, std, _ = _distribution(
            values,
            tracker=tracker,
            context=f"across model runs: {output_name}",
        )
        low, high = bootstrap_mean_ci(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            label=f"across-model:{output_name}",
            confidence_level=confidence_level,
        )
        prefix = f"across_model_{output_name}"
        result[f"{prefix}_mean"] = mean
        result[f"{prefix}_variance"] = variance
        result[f"{prefix}_std"] = std
        result[f"{prefix}_ci95_low"] = low
        result[f"{prefix}_ci95_high"] = high

    return result


def _public_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields = {
        key
        for row in rows
        for key in row
        if not str(key).startswith("_")
    }
    return sorted(fields)


def _field_order(
    rows: Sequence[Mapping[str, Any]],
    preferred: Sequence[str],
) -> list[str]:
    fields = _public_fields(rows)
    field_set = set(fields)
    ordered = [field for field in preferred if field in field_set]
    ordered.extend(field for field in fields if field not in set(ordered))
    return ordered


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_fields: Sequence[str],
) -> None:
    fields = _field_order(rows, preferred_fields)
    if not fields:
        fields = list(preferred_fields)
    if not fields:
        raise AggregationError(f"Cannot write headerless CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(row.get(field, ""))
                    for field in fields
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None:
        return ""
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _save_placeholder(
    path: Path,
    *,
    title: str,
    message: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _model_groups(
    episodes: Sequence[Mapping[str, Any]],
) -> list[
    tuple[tuple[str, str, str], list[Mapping[str, Any]]]
]:
    groups: MutableMapping[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in episodes:
        groups[_model_run_identity(row)].append(row)
    return [
        (identity, groups[identity])
        for identity in sorted(groups, key=_model_run_sort_key)
    ]


def _plot_boxplot(
    episodes: Sequence[Mapping[str, Any]],
    *,
    field: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    data = []
    labels = []
    for identity, rows in _model_groups(episodes):
        values = _finite_values(rows, field)
        if not values.size:
            continue
        data.append(values)
        labels.append(f"{_model_run_label(identity)}\nn={values.size}")
    if not data:
        _save_placeholder(
            path,
            title=title,
            message=f"No finite {field} values",
            dpi=dpi,
        )
        return

    width = max(7.5, min(16.0, 1.2 * len(data) + 5.0))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    label_parameter = (
        "tick_labels"
        if "tick_labels" in inspect.signature(ax.boxplot).parameters
        else "labels"
    )
    ax.boxplot(
        data,
        showmeans=True,
        **{label_parameter: labels},
    )
    ax.set_title(f"{title} (all episodes, n={sum(len(x) for x in data)})")
    ax.set_xlabel("Model run (name and training seed)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _map_ids(
    map_summaries: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        {str(row["map_id"]) for row in map_summaries},
        key=_natural_key,
    )


def _set_map_ticks(ax: Any, x: np.ndarray, map_ids: Sequence[str]) -> None:
    if not map_ids:
        return
    if len(map_ids) <= 30:
        tick_indices = np.arange(len(map_ids))
    else:
        tick_indices = np.unique(
            np.linspace(0, len(map_ids) - 1, num=20, dtype=int)
        )
    ax.set_xticks(x[tick_indices])
    ax.set_xticklabels(
        [map_ids[index] for index in tick_indices],
        rotation=60,
        ha="right",
    )


def _plot_success_by_map(
    map_summaries: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    if not map_summaries:
        _save_placeholder(
            output_path,
            title="Success rate by map",
            message="No map summaries",
            dpi=dpi,
        )
        return

    map_ids = _map_ids(map_summaries)
    identities = sorted(
        {_model_run_identity(row) for row in map_summaries},
        key=_model_run_sort_key,
    )
    lookups = {
        identity: {
            str(row["map_id"]): _as_float(row.get("success_rate"))
            for row in map_summaries
            if _model_run_identity(row) == identity
        }
        for identity in identities
    }
    counts = {
        identity: sum(
            int(_as_float(row.get("num_episodes")))
            for row in map_summaries
            if _model_run_identity(row) == identity
            and math.isfinite(_as_float(row.get("num_episodes")))
        )
        for identity in identities
    }

    x = np.arange(len(map_ids), dtype=np.float64)
    figure_width = max(10.0, min(24.0, 0.22 * len(map_ids) + 7.0))
    fig, ax = plt.subplots(figsize=(figure_width, 6.0))
    if len(identities) == 1:
        identity = identities[0]
        values = [
            lookups[identity].get(map_id, math.nan) for map_id in map_ids
        ]
        ax.bar(
            x,
            values,
            width=0.8,
            label=(
                f"{_model_run_label(identity)}\n"
                f"n={counts[identity]}"
            ),
        )
    elif len(map_ids) <= 50:
        bar_width = 0.8 / len(identities)
        for index, identity in enumerate(identities):
            offset = (
                index - (len(identities) - 1) / 2.0
            ) * bar_width
            values = [
                lookups[identity].get(map_id, math.nan)
                for map_id in map_ids
            ]
            ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=(
                    f"{_model_run_label(identity)}\n"
                    f"n={counts[identity]}"
                ),
            )
    else:
        for identity in identities:
            values = [
                lookups[identity].get(map_id, math.nan)
                for map_id in map_ids
            ]
            ax.plot(
                x,
                values,
                marker=".",
                linewidth=1.0,
                label=(
                    f"{_model_run_label(identity)}\n"
                    f"n={counts[identity]}"
                ),
            )

    _set_map_ticks(ax, x, map_ids)
    total_episodes = sum(
        int(_as_float(row.get("num_episodes")))
        for row in map_summaries
        if math.isfinite(_as_float(row.get("num_episodes")))
    )
    ax.set_title(
        "Success rate by test map "
        f"({len(map_ids)} maps, "
        f"{total_episodes} model-run/map episodes)"
    )
    ax.set_xlabel("Map ID")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _trajectory_groups(
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[
    tuple[str, str, str],
    list[list[Mapping[str, Any]]],
]:
    episodes: MutableMapping[
        tuple[str, tuple[str, str, str], str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in coverage_rows:
        key = (
            str(row.get("_source_id", "")),
            _model_run_identity(row),
            str(row["map_id"]),
            str(row.get("eval_seed", "")),
            str(row["episode_id"]),
        )
        episodes[key].append(row)

    by_model_run: MutableMapping[
        tuple[str, str, str],
        list[list[Mapping[str, Any]]],
    ] = defaultdict(list)
    for key in sorted(
        episodes,
        key=lambda item: (
            item[0],
            _model_run_sort_key(item[1]),
            _natural_key(item[2]),
            _natural_key(item[3]),
            _natural_key(item[4]),
        ),
    ):
        by_model_run[key[1]].append(episodes[key])
    return dict(by_model_run)


def build_coverage_threshold_summary(
    episodes: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    tracker: VarianceTracker | None = None,
    thresholds: Sequence[float] = COVERAGE_STEP_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Summarize the first decision step reaching each coverage threshold.

    Step statistics are conditional on reaching the threshold.  ``num_reached``
    and ``reach_rate`` retain the information about episodes that never reach
    it, including exception episodes without a coverage curve.
    """

    if tracker is None:
        tracker = VarianceTracker()
    if not thresholds:
        raise AggregationError("At least one coverage threshold is required")

    normalized_thresholds = []
    for threshold in thresholds:
        numeric = float(threshold)
        if not math.isfinite(numeric) or not 0.0 < numeric <= 1.0:
            raise AggregationError(
                "Coverage thresholds must be finite and in (0, 1]"
            )
        normalized_thresholds.append(numeric)
    if len(set(normalized_thresholds)) != len(normalized_thresholds):
        raise AggregationError("Coverage thresholds must not contain duplicates")

    trajectories_by_model = _trajectory_groups(coverage_rows)
    output: list[dict[str, Any]] = []
    for identity, episode_rows in _model_groups(episodes):
        trajectories = trajectories_by_model.get(identity, [])
        for threshold in sorted(normalized_thresholds):
            first_steps = []
            for trajectory in trajectories:
                crossing_steps = [
                    _as_float(row.get("step"))
                    for row in trajectory
                    if (
                        math.isfinite(_as_float(row.get("step")))
                        and math.isfinite(_as_float(row.get("coverage")))
                        and _as_float(row.get("coverage")) >= threshold
                    )
                ]
                if crossing_steps:
                    first_steps.append(min(crossing_steps))

            values = np.asarray(first_steps, dtype=np.float64)
            variance = sample_variance(values)
            if values.size < 2:
                tracker.note(
                    f"model_name={identity[0]!r}, "
                    f"model_seed={identity[1]!r}, "
                    f"coverage_threshold={threshold:.2f}: "
                    "steps_to_threshold"
                )
            row = _model_run_fields(identity, episode_rows)
            row.update(
                {
                    "coverage_threshold": threshold,
                    "coverage_threshold_percent": int(round(100 * threshold)),
                    "num_episodes": len(episode_rows),
                    "num_reached": int(values.size),
                    "reach_rate": (
                        float(values.size / len(episode_rows))
                        if episode_rows
                        else math.nan
                    ),
                    "mean_steps_to_threshold": _mean(values),
                    "variance_steps_to_threshold": variance,
                    "std_steps_to_threshold": (
                        math.sqrt(variance)
                        if math.isfinite(variance)
                        else math.nan
                    ),
                }
            )
            output.append(row)
    return output


def _plot_coverage_threshold_steps(
    summaries: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot mean and sample variance of first-crossing decision steps."""

    if not summaries:
        _save_placeholder(
            output_path,
            title="Steps to coverage thresholds",
            message="No coverage-threshold summaries",
            dpi=dpi,
        )
        return

    thresholds = sorted(
        {
            _as_float(row.get("coverage_threshold"))
            for row in summaries
            if math.isfinite(_as_float(row.get("coverage_threshold")))
        }
    )
    identities = sorted(
        {_model_run_identity(row) for row in summaries},
        key=_model_run_sort_key,
    )
    if not thresholds or not identities:
        _save_placeholder(
            output_path,
            title="Steps to coverage thresholds",
            message="No valid coverage-threshold summaries",
            dpi=dpi,
        )
        return

    lookup = {
        (_model_run_identity(row), _as_float(row.get("coverage_threshold"))): row
        for row in summaries
    }
    x = np.arange(len(thresholds), dtype=np.float64)
    width = 0.8 / len(identities)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    fields = (
        ("mean_steps_to_threshold", "Mean decision steps"),
        ("variance_steps_to_threshold", "Decision-step sample variance"),
    )
    for axis, (field, ylabel) in zip(axes, fields):
        for index, identity in enumerate(identities):
            offset = (index - (len(identities) - 1) / 2.0) * width
            values = [
                _as_float(lookup.get((identity, threshold), {}).get(field))
                for threshold in thresholds
            ]
            axis.bar(
                x + offset,
                values,
                width=width,
                label=_model_run_label(identity),
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_title(
        "Decision steps to first reach coverage thresholds "
        "(reached episodes only)"
    )
    axes[0].legend(loc="best", fontsize="small")
    axes[1].set_xlabel("Coverage threshold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{threshold:.0%}" for threshold in thresholds])
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _xy_for_trajectory(
    trajectory: Sequence[Mapping[str, Any]],
    x_field: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    points: dict[float, float] = {}
    for row in trajectory:
        x_value = _as_float(row.get(x_field))
        coverage = _as_float(row.get("coverage"))
        if math.isfinite(x_value) and math.isfinite(coverage):
            points[x_value] = coverage
    if not points:
        return None
    x = np.asarray(sorted(points), dtype=np.float64)
    y = np.asarray([points[value] for value in x], dtype=np.float64)
    return x, y


def _plot_coverage_curve(
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    x_field: str,
    output_path: Path,
    curve_points: int,
    dpi: int,
) -> None:
    title_axis = (
        ("Coverage vs traveled distance", "Traveled distance")
        if x_field == "path_length"
        else ("Coverage vs decision step", "Decision step")
    )
    by_model_run = _trajectory_groups(coverage_rows)
    prepared: dict[
        tuple[str, str, str],
        list[tuple[np.ndarray, np.ndarray]],
    ] = {}
    global_max = 0.0
    for identity, trajectories in by_model_run.items():
        valid = []
        for trajectory in trajectories:
            xy = _xy_for_trajectory(trajectory, x_field)
            if xy is None:
                continue
            valid.append(xy)
            global_max = max(global_max, float(xy[0][-1]))
        if valid:
            prepared[identity] = valid

    if not prepared:
        _save_placeholder(
            output_path,
            title=title_axis[0],
            message="No valid coverage-curve rows",
            dpi=dpi,
        )
        return

    if x_field == "step":
        grid = np.arange(0, int(math.ceil(global_max)) + 1, dtype=np.float64)
    else:
        grid = np.linspace(0.0, global_max, num=curve_points)
    if not grid.size:
        grid = np.asarray([0.0])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for identity in sorted(prepared, key=_model_run_sort_key):
        interpolated = np.stack(
            [
                np.interp(
                    grid,
                    x,
                    y,
                    left=float(y[0]),
                    right=float(y[-1]),
                )
                for x, y in prepared[identity]
            ],
            axis=0,
        )
        mean = np.mean(interpolated, axis=0)
        if interpolated.shape[0] >= 2:
            std = np.std(interpolated, axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)
        lower = np.maximum(0.0, mean - std)
        upper = np.minimum(1.0, mean + std)
        ax.plot(
            grid,
            mean,
            linewidth=2,
            label=(
                f"{_model_run_label(identity)}\n"
                f"n={interpolated.shape[0]}"
            ),
        )
        ax.fill_between(grid, lower, upper, alpha=0.18)

    total_trajectories = sum(len(items) for items in prepared.values())
    ax.set_title(
        f"{title_axis[0]} (mean ± sample SD, n={total_trajectories})"
    )
    ax.set_xlabel(title_axis[1])
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def generate_plots(
    episodes: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    map_summaries: Sequence[Mapping[str, Any]],
    coverage_threshold_summaries: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    curve_points: int,
    dpi: int,
) -> list[Path]:
    """Generate all required figures from the loaded CSV records."""

    outputs = [
        output_dir / "path_length_boxplot.png",
        output_dir / "final_coverage_boxplot.png",
        output_dir / "success_rate_by_map.png",
        output_dir / "coverage_vs_distance.png",
        output_dir / "coverage_vs_step.png",
        output_dir / "coverage_threshold_steps_bar.png",
    ]
    _plot_boxplot(
        episodes,
        field="path_length",
        ylabel="Path length",
        title="Episode path length by model run",
        path=outputs[0],
        dpi=dpi,
    )
    _plot_boxplot(
        episodes,
        field="final_coverage",
        ylabel="Final coverage",
        title="Final coverage by model run",
        path=outputs[1],
        dpi=dpi,
    )
    _plot_success_by_map(
        map_summaries,
        output_path=outputs[2],
        dpi=dpi,
    )
    _plot_coverage_curve(
        coverage_rows,
        x_field="path_length",
        output_path=outputs[3],
        curve_points=curve_points,
        dpi=dpi,
    )
    _plot_coverage_curve(
        coverage_rows,
        x_field="step",
        output_path=outputs[4],
        curve_points=curve_points,
        dpi=dpi,
    )
    _plot_coverage_threshold_steps(
        coverage_threshold_summaries,
        output_path=outputs[5],
        dpi=dpi,
    )
    return outputs


def aggregate_results(
    inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    curve_points: int = 101,
    figure_dpi: int = 160,
) -> dict[str, Path]:
    """Aggregate evaluation results, write summaries, and create plots."""

    if bootstrap_samples <= 0:
        raise AggregationError("bootstrap_samples must be positive")
    if not math.isclose(confidence_level, 0.95):
        raise AggregationError(
            "confidence_level must be 0.95 because output columns are "
            "explicitly named ci95"
        )
    if curve_points < 2:
        raise AggregationError("curve_points must be at least 2")
    if figure_dpi <= 0:
        raise AggregationError("figure_dpi must be positive")

    episodes, coverage_rows, episode_files = load_evaluation_inputs(inputs)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    tracker = VarianceTracker()
    map_summaries = build_map_summary(episodes, tracker=tracker)
    model_summaries = build_model_summary(
        episodes,
        tracker=tracker,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    overall_summary = build_overall_summary(
        episodes,
        model_summaries,
        tracker=tracker,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    coverage_threshold_summaries = build_coverage_threshold_summary(
        episodes,
        coverage_rows,
        tracker=tracker,
    )

    paths = {
        "episodes": destination / "episodes.csv",
        "map_summary": destination / "map_summary.csv",
        "model_summary": destination / "model_summary.csv",
        "overall_summary_csv": destination / "overall_summary.csv",
        "overall_summary_json": destination / "overall_summary.json",
        "coverage_curve": destination / "coverage_curve.csv",
        "coverage_threshold_summary": (
            destination / "coverage_threshold_summary.csv"
        ),
    }
    _write_csv(
        paths["episodes"],
        episodes,
        preferred_fields=EPISODE_CANONICAL_FIELDS,
    )
    _write_csv(
        paths["coverage_curve"],
        coverage_rows,
        preferred_fields=COVERAGE_CANONICAL_FIELDS,
    )
    _write_csv(
        paths["coverage_threshold_summary"],
        coverage_threshold_summaries,
        preferred_fields=COVERAGE_THRESHOLD_FIELDS,
    )
    _write_csv(
        paths["map_summary"],
        map_summaries,
        preferred_fields=[
            "model_seed",
            "model_name",
            "checkpoint",
            "checkpoint_sha256",
            "map_id",
            *SUMMARY_FIELDS,
        ],
    )
    _write_csv(
        paths["model_summary"],
        model_summaries,
        preferred_fields=[
            "model_seed",
            "model_name",
            "checkpoint",
            "checkpoint_sha256",
            *SUMMARY_FIELDS,
        ],
    )
    _write_csv(
        paths["overall_summary_csv"],
        [overall_summary],
        preferred_fields=[
            *SUMMARY_FIELDS,
            "num_model_seeds",
            "num_model_runs",
            "num_model_names",
            "bootstrap_seed",
            "bootstrap_samples",
            "confidence_level",
            "cross_model_resampling_unit",
            "model_run_identity_fields",
        ],
    )
    with paths["overall_summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(overall_summary),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    plot_paths = generate_plots(
        episodes,
        coverage_rows,
        map_summaries,
        coverage_threshold_summaries,
        output_dir=destination,
        curve_points=curve_points,
        dpi=figure_dpi,
    )
    for plot_path in plot_paths:
        paths[plot_path.stem] = plot_path

    tracker.report()
    print(
        f"Aggregated {len(episodes)} episodes from {len(episode_files)} "
        f"episodes.csv files ({len(model_summaries)} model runs, "
        f"{overall_summary['num_model_seeds']} unique model seeds, "
        f"{overall_summary['num_maps']} maps)."
    )
    print(f"Results written to: {destination.resolve()}")
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one or more map-split evaluation outputs without "
            "pandas. Each input is an evaluation directory or episodes.csv; "
            "coverage_curve.csv is loaded from the same directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/aggregate_results.py eval/seed_0 eval/seed_1 "
            "eval/seed_2 --output-dir experiment/summary\n"
            "  python scripts/aggregate_results.py "
            "eval/seed_0/episodes.csv --output-dir eval/seed_0/summary\n\n"
            "The unqualified overall statistics pool all episodes. "
            "Per-model-run episode bootstrap intervals are in "
            "model_summary.csv; "
            "overall_summary.* additionally reports across_model_* intervals "
            "obtained by resampling independently trained model runs "
            "(model name, training seed, and checkpoint SHA-256) as units."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Evaluation directories or paths to episodes.csv files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory for combined CSV files, summaries, and PNG figures.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Fixed bootstrap seed (default: 42).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2_000,
        help="Number of bootstrap replicates (default: 2000).",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=101,
        help="Grid points for coverage-vs-distance curves (default: 101).",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=160,
        help="PNG resolution in dots per inch (default: 160).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        aggregate_results(
            args.inputs,
            args.output_dir,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_samples=args.bootstrap_samples,
            curve_points=args.curve_points,
            figure_dpi=args.figure_dpi,
        )
    except AggregationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
