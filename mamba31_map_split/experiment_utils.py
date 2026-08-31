"""Shared helpers for reproducible map-split experiments.

This module intentionally has no dependency on Ray or the policy networks so
that manifests and result files can be validated on machines without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SUPPORTED_MAP_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def stable_seed(*parts: object, modulo: int = 2**32) -> int:
    """Derive a process-independent integer seed from arbitrary values."""

    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels.

    Ray training remains asynchronous, so this controls random streams but
    cannot make worker completion order bitwise deterministic.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            warnings.warn(
                "This PyTorch version cannot enable deterministic algorithms "
                "in warn-only mode; CUDA determinism may be incomplete.",
                RuntimeWarning,
            )


def capture_rng_state() -> Dict[str, Any]:
    """Return serializable RNG state suitable for a PyTorch checkpoint."""

    import torch

    numpy_state = np.random.get_state()
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a random state produced by :func:`capture_rng_state`."""

    if not isinstance(state, Mapping):
        raise TypeError("checkpoint rng_state must be a mapping")

    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"checkpoint rng_state is missing fields: {missing}")

    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("checkpoint NumPy RNG state must be a mapping")
    numpy_required = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    numpy_missing = sorted(numpy_required.difference(numpy_state))
    if numpy_missing:
        raise ValueError(
            "checkpoint NumPy RNG state is missing fields: "
            f"{numpy_missing}"
        )

    random.setstate(state["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )

    import torch

    torch_cpu_state = state["torch_cpu"]
    if not torch.is_tensor(torch_cpu_state):
        torch_cpu_state = torch.as_tensor(
            torch_cpu_state,
            dtype=torch.uint8,
        )
    torch.set_rng_state(
        torch_cpu_state.detach().to(
            device="cpu",
            dtype=torch.uint8,
        )
    )
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) != torch.cuda.device_count():
            warnings.warn(
                "Checkpoint CUDA RNG state count does not match the current "
                "CUDA device count; CUDA RNG state was not restored.",
                RuntimeWarning,
            )
        else:
            normalized_cuda_state = [
                (
                    item
                    if torch.is_tensor(item)
                    else torch.as_tensor(item, dtype=torch.uint8)
                )
                .detach()
                .to(device="cpu", dtype=torch.uint8)
                for item in cuda_state
            ]
            torch.cuda.set_rng_state_all(normalized_cuda_state)


def git_commit_hash(repo_root: Path | str) -> str:
    """Return the current Git commit, or ``unknown`` outside a Git checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(repo_root)),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: Path | str) -> str:
    """Hash either one checkpoint file or a directory tree deterministically."""

    artifact = Path(path).expanduser()
    if artifact.is_file():
        return sha256_file(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(
            f"checkpoint artifact does not exist: {artifact}"
        )

    digest = hashlib.sha256()
    digest.update(b"checkpoint-directory-v1\0")
    entries = sorted(
        artifact.rglob("*"),
        key=lambda item: item.relative_to(artifact).as_posix(),
    )
    for entry in entries:
        relative = entry.relative_to(artifact).as_posix().encode("utf-8")
        if entry.is_symlink():
            target = entry.resolve()
            if not target.is_file():
                raise ValueError(
                    "checkpoint directory contains a broken or non-file "
                    f"symbolic link: {entry}"
                )
            digest.update(b"L\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(sha256_file(target).encode("ascii"))
            digest.update(b"\n")
        elif entry.is_file():
            digest.update(b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(sha256_file(entry).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Atomically write a human-readable JSON document."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    temporary.replace(output)


def _validate_map_entries(name: str, entries: Any) -> List[str]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest field {name!r} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in entries):
        raise ValueError(f"manifest field {name!r} contains an invalid path")
    if len(entries) != len(set(entries)):
        raise ValueError(f"manifest field {name!r} contains duplicate paths")
    return entries


def load_split_manifest(
    split_file: Path | str,
    split_name: str,
    *,
    require_complete_dataset: bool = True,
) -> Tuple[Dict[str, Any], List[Path]]:
    """Load and fully validate a train/test map manifest.

    Paths inside the manifest are relative to ``dataset_root`` unless already
    absolute. Both split lists are checked even when only one is requested.
    """

    manifest_path = Path(split_file).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"split manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("split manifest root must be a JSON object")

    if split_name not in {"train", "test"}:
        raise ValueError(
            f"split_name must be 'train' or 'test', got {split_name!r}"
        )

    train_entries = _validate_map_entries(
        "train_maps", manifest.get("train_maps")
    )
    test_entries = _validate_map_entries("test_maps", manifest.get("test_maps"))
    dataset_value = manifest.get("dataset_root")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("manifest field 'dataset_root' must be a path string")
    dataset_root = Path(dataset_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (manifest_path.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"manifest dataset_root does not exist: {dataset_root}"
        )

    def resolve_entry(entry: str) -> Path:
        candidate = Path(entry).expanduser()
        path = (
            candidate
            if candidate.is_absolute()
            else dataset_root / candidate
        ).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"manifest map escapes dataset_root: {entry}"
            ) from exc
        return path

    train_paths = [resolve_entry(entry) for entry in train_entries]
    test_paths = [resolve_entry(entry) for entry in test_entries]
    if len(train_paths) != len(set(train_paths)):
        raise ValueError(
            "manifest train_maps contains paths that resolve to the same file"
        )
    if len(test_paths) != len(set(test_paths)):
        raise ValueError(
            "manifest test_maps contains paths that resolve to the same file"
        )
    canonical_overlap = set(train_paths).intersection(test_paths)
    if canonical_overlap:
        preview = ", ".join(
            str(path) for path in sorted(canonical_overlap)[:5]
        )
        raise ValueError(
            f"train/test map overlap detected after path resolution: {preview}"
        )

    all_entries = train_entries + test_entries
    all_paths = train_paths + test_paths
    if require_complete_dataset:
        discovered = {
            path.resolve().relative_to(dataset_root).as_posix()
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_MAP_SUFFIXES
        }
        listed = {
            path.relative_to(dataset_root).as_posix()
            for path in all_paths
        }
        missing_from_manifest = sorted(discovered - listed)
        extra_in_manifest = sorted(listed - discovered)
        if missing_from_manifest or extra_in_manifest:
            raise ValueError(
                "manifest does not exactly cover the current map dataset: "
                f"unlisted={missing_from_manifest[:5]}, "
                f"not_in_dataset={extra_in_manifest[:5]}"
            )

    resolved = dict(zip(all_entries, all_paths))
    for entry, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"manifest map does not exist: {path}")

    expected_hashes = manifest.get("file_sha256")
    if expected_hashes is not None:
        if not isinstance(expected_hashes, dict):
            raise ValueError("manifest field 'file_sha256' must be an object")
        if set(expected_hashes) != set(all_entries):
            raise ValueError(
                "manifest file_sha256 must contain every listed map exactly once"
            )
        actual_hashes: Dict[str, str] = {}
        for entry, path in resolved.items():
            actual_hashes[entry] = sha256_file(path)
        hash_mismatches = [
            entry
            for entry in all_entries
            if expected_hashes.get(entry) != actual_hashes[entry]
        ]
        if hash_mismatches:
            raise ValueError(
                "manifest file hashes do not match the current dataset: "
                + ", ".join(sorted(hash_mismatches)[:5])
            )

        expected_inventory = manifest.get("dataset_inventory_sha256")
        if expected_inventory is not None:
            inventory = hashlib.sha256()
            for entry in sorted(all_entries):
                inventory.update(entry.encode("utf-8"))
                inventory.update(b"\0")
                inventory.update(actual_hashes[entry].encode("ascii"))
                inventory.update(b"\n")
            if inventory.hexdigest() != expected_inventory:
                raise ValueError(
                    "manifest dataset inventory digest does not match the "
                    "current dataset"
                )

    map_groups = manifest.get("map_groups")
    if map_groups is not None:
        if not isinstance(map_groups, dict):
            raise ValueError("manifest field 'map_groups' must be an object")
        if set(map_groups) != set(all_entries):
            raise ValueError(
                "manifest map_groups must contain every listed map exactly once"
            )
        if not all(
            isinstance(group_id, str) and group_id
            for group_id in map_groups.values()
        ):
            raise ValueError(
                "manifest map_groups values must be non-empty strings"
            )
        train_groups = {map_groups.get(entry) for entry in train_entries}
        test_groups = {map_groups.get(entry) for entry in test_entries}
        leaked_groups = train_groups.intersection(test_groups)
        if leaked_groups:
            raise ValueError(
                "base-map group leakage detected: "
                + ", ".join(sorted(map(str, leaked_groups))[:5])
            )

    selected_entries = (
        train_entries if split_name == "train" else test_entries
    )
    manifest["_manifest_path"] = str(manifest_path.resolve())
    manifest["_dataset_root"] = str(dataset_root)
    return manifest, [resolved[entry] for entry in selected_entries]


def trajectory_path_length(positions: Sequence[Sequence[float]]) -> float:
    """Sum Euclidean distance between consecutive 2-D robot positions."""

    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.size == 0:
        return 0.0
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("positions must have shape [N, 2]")
    if len(coordinates) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum())


def coverage_success(coverage: float, target_coverage: float = 0.99) -> bool:
    """Return the primary evaluation success predicate."""

    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    return bool(float(coverage) >= float(target_coverage))


def checkpoint_model_seed(checkpoint: Mapping[str, Any]) -> int:
    metadata = checkpoint.get("experiment_metadata", {})
    value = metadata.get("training_seed", checkpoint.get("training_seed", -1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
