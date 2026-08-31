#!/usr/bin/env python3
"""Annotate a legacy Mamba 3.1 checkpoint for split-training resume.

The existing checkpoint contains the policy, critics, optimizers, and SAC
state, but predates the map-split provenance fields.  This tool copies it to a
new run directory and adds only the provenance needed by the split-aware
driver; the original checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Import project modules from the integration root when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import git_commit_hash, load_split_manifest, sha256_file
from parameter import RUN_FOLDER_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--split-name", default="train", choices=("train",))
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.training_seed < 0:
        raise ValueError("--training-seed must be non-negative")
    source = args.checkpoint.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    split_file = args.split_file.expanduser().resolve()
    manifest, train_maps = load_split_manifest(split_file, args.split_name)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint root must be a mapping")
    if checkpoint.get("use_mamba") is not True:
        raise ValueError("the supplied checkpoint is not an ARiADNE-Mamba checkpoint")
    if "policy_model" not in checkpoint:
        raise KeyError("checkpoint is missing policy_model")

    existing_metadata = checkpoint.get("experiment_metadata")
    if isinstance(existing_metadata, dict):
        required = {
            "split_manifest_sha256": sha256_file(split_file),
            "split_name": args.split_name,
            "training_seed": args.training_seed,
        }
        mismatches = {
            key: (existing_metadata.get(key), value)
            for key, value in required.items()
            if existing_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "checkpoint already has incompatible split provenance: "
                f"{mismatches}"
            )
        metadata = dict(existing_metadata)
    else:
        metadata = {
            "git_commit": git_commit_hash(Path(__file__).resolve().parents[1]),
            "split_manifest": str(split_file),
            "split_manifest_sha256": sha256_file(split_file),
            "split_seed": manifest.get("split_seed"),
            "split_name": args.split_name,
            "training_seed": args.training_seed,
            "total_environment_steps": None,
            "requested_total_env_steps": None,
            "model_name": RUN_FOLDER_NAME,
            "model_source_ref": "ARiADNE_MAMBA_3.1",
            "model_source_commit": "local-checkpoint",
            "use_mamba": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "legacy_resume_source": str(source),
            "legacy_resume_source_sha256": sha256_file(source),
            "train_map_count": len(train_maps),
        }
        checkpoint["experiment_metadata"] = metadata

    # A legacy checkpoint has no split-run step counter.  The launcher treats
    # its requested budget as additional environment steps from this point.
    checkpoint.setdefault("completed_env_steps", 0)
    checkpoint.setdefault("completed_episodes", 0)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "checkpoint.pth"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    temporary = destination.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    print(f"Prepared split-resume checkpoint: {destination}")
    print(f"Train maps: {len(train_maps)}")
    print(f"Original checkpoint preserved: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
