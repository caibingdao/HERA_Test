#!/usr/bin/env python3
"""Create a deterministic, leakage-safe train/test map manifest.

The ARiADNE maps encode both occupancy and the robot start marker in the same
image.  A byte hash alone is therefore not sufficient to identify variants of
one underlying map.  This script computes two identities:

* a raw SHA-256 hash, used to report byte-for-byte copies; and
* a semantic occupancy hash which ignores the start marker and is invariant to
  uniform pixel scaling, translation, 90-degree rotations, and reflections.

Every discovered map path remains in the manifest.  Maps with the same group
identity are assigned together so duplicate or transformed files cannot leak
between training and testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_MAP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}
KNOWN_CATEGORIES = ("easy", "medium", "complex", "random")


@dataclass(frozen=True)
class MapRecord:
    """Audited metadata for one map file."""

    relative_path: str
    raw_sha256: str
    semantic_sha256: str
    width: int
    height: int
    logical_width: int
    logical_height: int
    pixel_scale_x: int
    pixel_scale_y: int
    category: Optional[str]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_maps(map_root: Path | str) -> List[Path]:
    """Return all supported map images below ``map_root`` in stable order."""

    root = Path(map_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"map root does not exist or is not a directory: {root}")

    maps = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_MAP_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not maps:
        suffixes = ", ".join(sorted(SUPPORTED_MAP_SUFFIXES))
        raise ValueError(f"no map images ({suffixes}) found below {root}")
    return maps


def infer_category(relative_path: Path | str) -> Optional[str]:
    """Infer only explicit category tokens; never invent difficulty labels."""

    value = Path(relative_path).as_posix().lower()
    tokens = set(token for token in re.split(r"[^a-z0-9]+", value) if token)
    matches = [category for category in KNOWN_CATEGORIES if category in tokens]
    if len(matches) > 1:
        raise ValueError(
            f"map path has conflicting category labels {matches}: {relative_path}"
        )
    return matches[0] if matches else None


def semantic_free_mask(image: np.ndarray) -> np.ndarray:
    """Convert an image using the same free-space thresholds as ``Env``.

    ``Env.import_ground_truth`` converts an image to grayscale, truncates it to
    integer values in [0, 255], and considers values above 150 or in [50, 80]
    free.  The latter range covers common colored robot markers.  Applying the
    same predicate makes the semantic identity insensitive to the marker while
    preserving the occupancy map used by the simulator.
    """

    array = np.asarray(image)
    if array.ndim == 2:
        source = array.astype(np.float64, copy=False)
        if np.issubdtype(array.dtype, np.integer):
            maximum = float(np.iinfo(array.dtype).max)
            gray = source * (255.0 / maximum)
        elif np.issubdtype(array.dtype, np.bool_):
            gray = source * 255.0
        else:
            finite = source[np.isfinite(source)]
            if finite.size == 0:
                raise ValueError("map contains no finite pixels")
            gray = source * 255.0 if float(finite.max()) <= 1.0 else source
    elif array.ndim == 3 and array.shape[2] >= 3:
        rgb_source = array[..., :3]
        rgb = rgb_source.astype(np.float64, copy=False)
        if np.issubdtype(rgb_source.dtype, np.integer):
            maximum = float(np.iinfo(rgb_source.dtype).max)
            rgb = rgb * (255.0 / maximum)
        elif np.issubdtype(rgb_source.dtype, np.bool_):
            rgb = rgb * 255.0
        else:
            finite = rgb[np.isfinite(rgb)]
            if finite.size == 0:
                raise ValueError("map contains no finite pixels")
            if float(finite.max()) <= 1.0:
                rgb = rgb * 255.0

        # Coefficients used by skimage.color.rgb2gray, which is what
        # skimage.io.imread(..., as_gray=True) applies in the environment.
        gray = (
            rgb[..., 0] * 0.2125
            + rgb[..., 1] * 0.7154
            + rgb[..., 2] * 0.0721
        )
    else:
        raise ValueError(
            "expected a 2-D grayscale or RGB/RGBA image, "
            f"received shape {array.shape}"
        )

    if not np.all(np.isfinite(gray)):
        raise ValueError("map contains NaN or infinite pixel values")
    gray_integer = gray.astype(np.int64)
    free = (gray_integer > 150) | (
        (gray_integer >= 50) & (gray_integer <= 80)
    )
    if not np.any(free):
        raise ValueError("map contains no free cells under the environment thresholds")
    return np.asarray(free, dtype=np.bool_)


def _remove_uniform_color_scale(
    image: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """Losslessly collapse a nearest-neighbour-scaled categorical image.

    The union of all color-transition coordinates defines a shared pixel grid.
    Its greatest common divisor is safe to remove: by construction there is no
    color transition inside any resulting block.  Performing this before the
    grayscale conversion makes the 16-pixel ARIADNE map tiles inexpensive to
    audit while retaining a scale-1 fallback for ordinary images.
    """

    height, width = image.shape[:2]
    if image.ndim == 2:
        comparison_image = image
    elif (
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 4
        and image.flags.c_contiguous
    ):
        # PIL supplies RGBA as four contiguous bytes.  Viewing a pixel as one
        # uint32 preserves equality and avoids four separate comparisons.
        comparison_image = image.view(np.uint32).reshape(height, width)
    else:
        comparison_image = image

    if comparison_image.ndim == 2:
        horizontal_transition = (
            np.any(
                comparison_image[:, 1:] != comparison_image[:, :-1],
                axis=0,
            )
            if width > 1
            else np.empty(0, dtype=np.bool_)
        )
        vertical_transition = (
            np.any(
                comparison_image[1:, :] != comparison_image[:-1, :],
                axis=1,
            )
            if height > 1
            else np.empty(0, dtype=np.bool_)
        )
    else:
        horizontal_transition = (
            np.any(
                comparison_image[:, 1:, :] != comparison_image[:, :-1, :],
                axis=(0, 2),
            )
            if width > 1
            else np.empty(0, dtype=np.bool_)
        )
        vertical_transition = (
            np.any(
                comparison_image[1:, :, :] != comparison_image[:-1, :, :],
                axis=(1, 2),
            )
            if height > 1
            else np.empty(0, dtype=np.bool_)
        )

    scale_x = width
    for coordinate in np.flatnonzero(horizontal_transition) + 1:
        scale_x = math.gcd(scale_x, int(coordinate))
    scale_y = height
    for coordinate in np.flatnonzero(vertical_transition) + 1:
        scale_y = math.gcd(scale_y, int(coordinate))
    scale_x = max(scale_x, 1)
    scale_y = max(scale_y, 1)
    reduced = np.ascontiguousarray(image[::scale_y, ::scale_x])
    return reduced, scale_x, scale_y


def _crop_to_free(mask: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("cannot canonicalize a map without free cells")
    return np.ascontiguousarray(
        mask[
            int(rows.min()) : int(rows.max()) + 1,
            int(columns.min()) : int(columns.max()) + 1,
        ]
    )


def _common_pixel_scale(mask: np.ndarray) -> Tuple[int, int]:
    """Find a uniform nearest-neighbour pixel scale for a binary map.

    After removing the constant outer border, all horizontal and vertical
    transition coordinates of a uniformly scaled categorical image share a
    common divisor.  Verification in ``_remove_uniform_pixel_scale`` prevents
    an unsafe reduction when the inferred grid is not actually uniform.
    """

    height, width = mask.shape
    horizontal_changes = (
        np.flatnonzero(np.any(mask[:, 1:] != mask[:, :-1], axis=0)) + 1
        if width > 1
        else np.empty(0, dtype=int)
    )
    vertical_changes = (
        np.flatnonzero(np.any(mask[1:, :] != mask[:-1, :], axis=1)) + 1
        if height > 1
        else np.empty(0, dtype=int)
    )

    scale_x = width
    for coordinate in horizontal_changes:
        scale_x = math.gcd(scale_x, int(coordinate))
    scale_y = height
    for coordinate in vertical_changes:
        scale_y = math.gcd(scale_y, int(coordinate))
    return max(scale_x, 1), max(scale_y, 1)


def _remove_uniform_pixel_scale(
    mask: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    cropped = _crop_to_free(mask)
    scale_x, scale_y = _common_pixel_scale(cropped)
    height, width = cropped.shape

    if height % scale_y != 0 or width % scale_x != 0:
        return cropped, 1, 1

    blocks = cropped.reshape(
        height // scale_y,
        scale_y,
        width // scale_x,
        scale_x,
    )
    logical = blocks[:, 0, :, 0]
    reconstructed = np.repeat(
        np.repeat(logical, scale_y, axis=0), scale_x, axis=1
    )
    if not np.array_equal(cropped, reconstructed):
        return cropped, 1, 1
    return np.ascontiguousarray(logical), scale_x, scale_y


def canonical_semantic_identity(
    free_mask: np.ndarray,
) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
    """Return a D4-, translation-, and integer-scale-invariant map identity."""

    logical, scale_x, scale_y = _remove_uniform_pixel_scale(free_mask)
    candidates: List[bytes] = []
    for rotation in range(4):
        rotated = np.rot90(logical, rotation)
        for transformed in (rotated, np.fliplr(rotated)):
            normalized = _crop_to_free(transformed)
            height, width = normalized.shape
            payload = (
                struct.pack(">II", height, width)
                + np.packbits(normalized.reshape(-1)).tobytes()
            )
            candidates.append(payload)
    canonical = min(candidates)
    digest = hashlib.sha256(canonical).hexdigest()
    return digest, logical.shape, (scale_x, scale_y)


def analyze_map(path: Path | str, map_root: Path | str) -> MapRecord:
    """Decode and fingerprint one map, raising a path-specific clear error."""

    root = Path(map_root).expanduser().resolve()
    map_path = Path(path).expanduser().resolve()
    try:
        relative = map_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"map escapes map root {root}: {map_path}") from exc

    try:
        with Image.open(map_path) as source:
            source.load()
            if source.mode in {"1", "L", "I", "I;16", "F"}:
                image = np.asarray(source)
            else:
                image = np.asarray(source.convert("RGBA"))
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"failed to decode map image {map_path}: {exc}") from exc
    if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"map image has invalid shape {image.shape}: {map_path}")

    try:
        reduced_image, raw_scale_x, raw_scale_y = (
            _remove_uniform_color_scale(image)
        )
        free_mask = semantic_free_mask(reduced_image)
        semantic_hash, logical_shape, pixel_scale = canonical_semantic_identity(
            free_mask
        )
    except ValueError as exc:
        raise ValueError(f"invalid map {map_path}: {exc}") from exc

    return MapRecord(
        relative_path=relative,
        raw_sha256=sha256_file(map_path),
        semantic_sha256=semantic_hash,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        logical_width=int(logical_shape[1]),
        logical_height=int(logical_shape[0]),
        pixel_scale_x=int(raw_scale_x * pixel_scale[0]),
        pixel_scale_y=int(raw_scale_y * pixel_scale[1]),
        category=infer_category(relative),
    )


def scan_map_records(map_root: Path | str) -> List[MapRecord]:
    """Discover, validate, and fingerprint every map below ``map_root``."""

    root = Path(map_root).expanduser().resolve()
    return [analyze_map(path, root) for path in discover_maps(root)]


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _choose_train_groups(
    group_ids: Sequence[str],
    group_members: Mapping[str, Sequence[str]],
    train_ratio: float,
    rng: random.Random,
) -> Set[str]:
    """Choose whole groups while targeting a ratio by number of map paths."""

    shuffled = list(sorted(group_ids))
    rng.shuffle(shuffled)
    total_weight = sum(len(group_members[group_id]) for group_id in shuffled)
    target_weight = total_weight * train_ratio
    train_groups: Set[str] = set()
    train_weight = 0

    for group_id in shuffled:
        weight = len(group_members[group_id])
        current_error = abs(train_weight - target_weight)
        candidate_error = abs(train_weight + weight - target_weight)
        if candidate_error < current_error:
            train_groups.add(group_id)
            train_weight += weight

    # Where possible, represent every stratum on both sides.  This is
    # especially important for small labeled categories.
    if len(shuffled) > 1 and not train_groups:
        best = min(
            shuffled,
            key=lambda group_id: (
                abs(len(group_members[group_id]) - target_weight),
                shuffled.index(group_id),
            ),
        )
        train_groups.add(best)
    if len(shuffled) > 1 and len(train_groups) == len(shuffled):
        best = min(
            shuffled,
            key=lambda group_id: (
                abs(
                    train_weight
                    - len(group_members[group_id])
                    - target_weight
                ),
                shuffled.index(group_id),
            ),
        )
        train_groups.remove(best)
    return train_groups


def split_grouped_maps(
    group_members: Mapping[str, Sequence[str]],
    *,
    train_ratio: float,
    seed: int,
    group_categories: Optional[Mapping[str, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split path groups deterministically, optionally within category strata."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if len(group_members) < 2:
        raise ValueError("at least two independent map groups are required")
    if any(not members for members in group_members.values()):
        raise ValueError("map groups must not be empty")

    if group_categories is None:
        strata = {"__all__": sorted(group_members)}
    else:
        missing = set(group_members).difference(group_categories)
        if missing:
            raise ValueError(
                "category mapping is missing map groups: "
                + ", ".join(sorted(missing)[:5])
            )
        strata: Dict[str, List[str]] = defaultdict(list)
        for group_id in sorted(group_members):
            strata[group_categories[group_id]].append(group_id)

    train_groups: Set[str] = set()
    for category, group_ids in sorted(strata.items()):
        stratum_seed = (
            seed
            if group_categories is None
            else _derive_seed(seed, category)
        )
        train_groups.update(
            _choose_train_groups(
                group_ids,
                group_members,
                train_ratio,
                random.Random(stratum_seed),
            )
        )

    all_groups = set(group_members)
    test_groups = all_groups - train_groups

    # If every category contains a single group, per-stratum choices can make
    # one global side empty.  Move exactly one whole group in that corner case.
    if not train_groups or not test_groups:
        source = test_groups if not train_groups else train_groups
        total_maps = sum(len(members) for members in group_members.values())
        current_train_maps = sum(
            len(group_members[group_id]) for group_id in train_groups
        )
        target = total_maps * train_ratio
        candidates = sorted(source)
        if not train_groups:
            moved = min(
                candidates,
                key=lambda group_id: (
                    abs(
                        current_train_maps
                        + len(group_members[group_id])
                        - target
                    ),
                    group_id,
                ),
            )
            train_groups.add(moved)
            test_groups.remove(moved)
        else:
            moved = min(
                candidates,
                key=lambda group_id: (
                    abs(
                        current_train_maps
                        - len(group_members[group_id])
                        - target
                    ),
                    group_id,
                ),
            )
            train_groups.remove(moved)
            test_groups.add(moved)

    train_maps = sorted(
        path
        for group_id in train_groups
        for path in group_members[group_id]
    )
    test_maps = sorted(
        path
        for group_id in test_groups
        for path in group_members[group_id]
    )
    return train_maps, test_maps


def _relative_dataset_root(map_root: Path, output: Optional[Path]) -> str:
    if output is None:
        return str(map_root)
    relative = os.path.relpath(map_root, output.parent.resolve())
    return Path(relative).as_posix()


def _inventory_digest(records: Sequence[MapRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.raw_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _category_statistics(
    records: Sequence[MapRecord],
    train_maps: Sequence[str],
    test_maps: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    category_by_path = {
        record.relative_path: record.category or "unlabeled"
        for record in records
    }
    train_counts = Counter(category_by_path[path] for path in train_maps)
    test_counts = Counter(category_by_path[path] for path in test_maps)
    total_counts = Counter(category_by_path.values())
    return {
        category: {
            "total": int(total_counts[category]),
            "train": int(train_counts[category]),
            "test": int(test_counts[category]),
        }
        for category in sorted(total_counts)
    }


def build_split_manifest(
    map_root: Path | str,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    output: Optional[Path | str] = None,
    group_by_base_map: bool = True,
    stratify: bool = True,
    records: Optional[Sequence[MapRecord]] = None,
) -> Dict[str, Any]:
    """Build and validate an in-memory split manifest.

    Supplying pre-scanned ``records`` is useful for tests that need to build
    several seeds without decoding the dataset repeatedly.
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    root = Path(map_root).expanduser().resolve()
    if records is None:
        audited_records = scan_map_records(root)
    else:
        audited_records = sorted(records, key=lambda item: item.relative_path)
        if not audited_records:
            raise ValueError("records must contain at least one map")

    output_path = (
        None if output is None else Path(output).expanduser().resolve()
    )
    group_members: Dict[str, List[str]] = defaultdict(list)
    raw_hash_members: Dict[str, List[str]] = defaultdict(list)
    map_groups: Dict[str, str] = {}

    for record in audited_records:
        identity = (
            f"semantic-sha256:{record.semantic_sha256}"
            if group_by_base_map
            else f"content-sha256:{record.raw_sha256}"
        )
        group_members[identity].append(record.relative_path)
        raw_hash_members[record.raw_sha256].append(record.relative_path)
        map_groups[record.relative_path] = identity

    for members in group_members.values():
        members.sort()
    duplicate_groups = [
        {"sha256": digest, "maps": sorted(members)}
        for digest, members in sorted(raw_hash_members.items())
        if len(members) > 1
    ]

    categories_present = {
        record.category for record in audited_records if record.category
    }
    unlabeled_count = sum(
        record.category is None for record in audited_records
    )
    category_by_path = {
        record.relative_path: record.category for record in audited_records
    }
    group_categories: Dict[str, str] = {}
    category_conflicts: Dict[str, List[str]] = {}
    for group_id, members in sorted(group_members.items()):
        member_categories = {
            category_by_path[path]
            for path in members
            if category_by_path[path] is not None
        }
        if len(member_categories) == 1 and all(
            category_by_path[path] is not None for path in members
        ):
            group_categories[group_id] = next(iter(member_categories))
        elif len(member_categories) > 1:
            category_conflicts[group_id] = sorted(member_categories)

    if not stratify:
        stratification_applied = False
        stratification_reason = "stratification disabled by command line"
    elif not categories_present:
        stratification_applied = False
        stratification_reason = "no explicit category labels found"
    elif unlabeled_count:
        stratification_applied = False
        stratification_reason = (
            f"{unlabeled_count} map(s) have no explicit category label"
        )
    elif category_conflicts:
        stratification_applied = False
        stratification_reason = (
            "one or more base-map groups contain conflicting category labels"
        )
    elif len(group_categories) != len(group_members):
        stratification_applied = False
        stratification_reason = "not every map group has a category label"
    else:
        stratification_applied = True
        stratification_reason = "explicit category labels used"

    train_maps, test_maps = split_grouped_maps(
        group_members,
        train_ratio=train_ratio,
        seed=seed,
        group_categories=group_categories if stratification_applied else None,
    )

    category_stats = _category_statistics(
        audited_records, train_maps, test_maps
    )
    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "split_seed": seed,
        "train_ratio": float(train_ratio),
        "actual_train_ratio": len(train_maps) / len(audited_records),
        "dataset_root": _relative_dataset_root(root, output_path),
        "dataset_inventory_sha256": _inventory_digest(audited_records),
        "group_by_base_map": bool(group_by_base_map),
        "grouping_method": (
            "semantic occupancy; start-marker, translation, integer-scale, "
            "rotation, and reflection invariant"
            if group_by_base_map
            else "raw content SHA-256 (exact copies still grouped)"
        ),
        "stratify_requested": bool(stratify),
        "stratification_applied": stratification_applied,
        "stratification_reason": stratification_reason,
        "category_labels": sorted(categories_present),
        "map_categories": {
            record.relative_path: record.category
            for record in audited_records
            if record.category is not None
        },
        "category_conflicts": category_conflicts,
        "train_maps": train_maps,
        "test_maps": test_maps,
        "map_groups": dict(sorted(map_groups.items())),
        "map_group_members": {
            group_id: members
            for group_id, members in sorted(group_members.items())
        },
        "duplicate_groups": duplicate_groups,
        "file_sha256": {
            record.relative_path: record.raw_sha256
            for record in audited_records
        },
        "statistics": {
            "total_maps": len(audited_records),
            "train_maps": len(train_maps),
            "test_maps": len(test_maps),
            "total_groups": len(group_members),
            "variant_groups": sum(
                len(members) > 1 for members in group_members.values()
            ),
            "duplicate_groups": len(duplicate_groups),
            "files_in_duplicate_groups": sum(
                len(group["maps"]) for group in duplicate_groups
            ),
            "redundant_duplicate_files": sum(
                len(group["maps"]) - 1 for group in duplicate_groups
            ),
            "train_test_overlap": False,
            "all_maps_included": True,
            "categories": category_stats,
        },
    }
    validate_manifest(manifest, root, records=audited_records)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    map_root: Path | str,
    *,
    records: Optional[Sequence[MapRecord]] = None,
) -> None:
    """Validate completeness, existence, hashes, and group isolation."""

    root = Path(map_root).expanduser().resolve()
    audited_records = (
        list(records) if records is not None else scan_map_records(root)
    )
    expected = {record.relative_path for record in audited_records}
    record_by_path = {
        record.relative_path: record for record in audited_records
    }

    train_maps = manifest.get("train_maps")
    test_maps = manifest.get("test_maps")
    if not isinstance(train_maps, list) or not train_maps:
        raise ValueError("manifest train_maps must be a non-empty list")
    if not isinstance(test_maps, list) or not test_maps:
        raise ValueError("manifest test_maps must be a non-empty list")
    if not all(isinstance(path, str) and path for path in train_maps + test_maps):
        raise ValueError("manifest map entries must be non-empty strings")
    if len(train_maps) != len(set(train_maps)):
        raise ValueError("manifest train_maps contains duplicate paths")
    if len(test_maps) != len(set(test_maps)):
        raise ValueError("manifest test_maps contains duplicate paths")

    train_set = set(train_maps)
    test_set = set(test_maps)
    overlap = train_set & test_set
    if overlap:
        raise ValueError(
            "train/test map overlap detected: "
            + ", ".join(sorted(overlap)[:5])
        )
    listed = train_set | test_set
    missing = expected - listed
    extra = listed - expected
    if missing or extra:
        raise ValueError(
            "manifest does not exactly cover the map dataset: "
            f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )

    for relative_path in sorted(listed):
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"manifest map escapes dataset root: {relative_path}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"manifest map does not exist: {candidate}"
            )

    map_groups = manifest.get("map_groups")
    if not isinstance(map_groups, dict):
        raise ValueError("manifest map_groups must be an object")
    if set(map_groups) != expected:
        raise ValueError("manifest map_groups must contain every map exactly once")
    train_groups = {map_groups[path] for path in train_set}
    test_groups = {map_groups[path] for path in test_set}
    leaked_groups = train_groups & test_groups
    if leaked_groups:
        raise ValueError(
            "base-map group leakage detected: "
            + ", ".join(sorted(map(str, leaked_groups))[:5])
        )

    file_hashes = manifest.get("file_sha256")
    if not isinstance(file_hashes, dict) or set(file_hashes) != expected:
        raise ValueError("manifest file_sha256 must contain every map exactly once")
    mismatches = [
        path
        for path, record in record_by_path.items()
        if file_hashes.get(path) != record.raw_sha256
    ]
    if mismatches:
        raise ValueError(
            "manifest file hashes do not match the dataset: "
            + ", ".join(sorted(mismatches)[:5])
        )

    # Raw copies must remain together even if semantic grouping was disabled.
    paths_by_hash: Dict[str, List[str]] = defaultdict(list)
    for record in audited_records:
        paths_by_hash[record.raw_sha256].append(record.relative_path)
    for digest, paths in paths_by_hash.items():
        sides = {
            "train" if path in train_set else "test"
            for path in paths
        }
        if len(sides) > 1:
            raise ValueError(
                f"duplicate content group leaked across splits ({digest}): "
                + ", ".join(sorted(paths))
            )

    if manifest.get("dataset_inventory_sha256") != _inventory_digest(
        audited_records
    ):
        raise ValueError("manifest dataset inventory digest is incorrect")


def write_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Path:
    """Atomically write a stable, human-readable JSON manifest."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            manifest,
            stream,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        stream.write("\n")
    temporary.replace(output)
    return output


def create_split_manifest(
    map_root: Path | str,
    output: Path | str,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    group_by_base_map: bool = True,
    stratify: bool = True,
) -> Dict[str, Any]:
    """Build, validate, write, reload, and revalidate a split manifest."""

    root = Path(map_root).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    records = scan_map_records(root)
    manifest = build_split_manifest(
        root,
        train_ratio=train_ratio,
        seed=seed,
        output=output_path,
        group_by_base_map=group_by_base_map,
        stratify=stratify,
        records=records,
    )
    write_manifest(output_path, manifest)
    with output_path.open("r", encoding="utf-8") as stream:
        reloaded = json.load(stream)
    validate_manifest(reloaded, root, records=records)
    return reloaded


def validate_manifest_file(path: Path | str) -> Dict[str, Any]:
    """Load and validate a manifest using its stored dataset root."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"split manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc
    dataset_value = manifest.get("dataset_root")
    if not isinstance(dataset_value, str) or not dataset_value:
        raise ValueError("manifest dataset_root must be a non-empty path string")
    dataset_root = Path(dataset_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (manifest_path.parent / dataset_root).resolve()
    validate_manifest(manifest, dataset_root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible train/test split while keeping duplicate "
            "and transformed base-map variants on the same side."
        )
    )
    parser.add_argument(
        "--map-root",
        default="maps",
        help="map dataset directory (default: maps)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="fraction of map paths assigned to training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fixed split random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "output JSON path (default: "
            "experiment/map_splits/map_split_seed_<seed>.json)"
        ),
    )
    parser.add_argument(
        "--group-by-base-map",
        dest="group_by_base_map",
        action="store_true",
        default=True,
        help="group start/scale/rotation/mirror variants (default: enabled)",
    )
    parser.add_argument(
        "--no-group-by-base-map",
        dest="group_by_base_map",
        action="store_false",
        help="only group byte-identical copies",
    )
    parser.add_argument(
        "--stratify",
        dest="stratify",
        action="store_true",
        default=True,
        help="stratify when real category labels exist (default: enabled)",
    )
    parser.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="disable category stratification",
    )
    return parser


def _print_statistics(manifest: Mapping[str, Any], output: Path) -> None:
    statistics = manifest["statistics"]
    print(f"Total maps: {statistics['total_maps']}")
    print(f"Train maps: {statistics['train_maps']}")
    print(f"Test maps: {statistics['test_maps']}")
    print(f"Base-map groups: {statistics['total_groups']}")
    print(
        "Exact duplicate groups: "
        f"{statistics['duplicate_groups']} "
        f"({statistics['files_in_duplicate_groups']} files)"
    )
    print(
        "Stratification applied: "
        f"{str(manifest['stratification_applied']).lower()} "
        f"({manifest['stratification_reason']})"
    )
    for category, counts in statistics["categories"].items():
        print(
            f"Category {category}: total={counts['total']}, "
            f"train={counts['train']}, test={counts['test']}"
        )
    print(
        "Train/test overlap: "
        f"{str(statistics['train_test_overlap']).lower()}"
    )
    print(f"Split manifest: {output}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = (
        Path(args.output)
        if args.output
        else Path(
            "experiment/map_splits",
            f"map_split_seed_{args.seed}.json",
        )
    )
    try:
        manifest = create_split_manifest(
            args.map_root,
            output,
            train_ratio=args.train_ratio,
            seed=args.seed,
            group_by_base_map=args.group_by_base_map,
            stratify=args.stratify,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    _print_statistics(manifest, output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
