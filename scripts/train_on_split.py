#!/usr/bin/env python3
"""Train the existing ARIADNE policy using only manifest training maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original ARIADNE/Mamba training loop on the train side "
            "of a fixed map manifest."
        )
    )
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument(
        "--split-name",
        default="train",
        choices=("train",),
        help="Training is deliberately restricted to the train split.",
    )
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--total-env-steps",
        type=positive_int,
        required=True,
        help="Stop after exactly this many completed environment decisions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint/log directory. Defaults to "
            "experiment/checkpoints/<model>/seed_<seed>."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted split run from a full training checkpoint. "
            "When --output-dir is omitted, the checkpoint parent directory "
            "is reused."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.training_seed < 0:
        raise SystemExit("--training-seed must be non-negative")

    # Import after argument validation because driver imports the project-wide
    # USE_MAMBA setting and builds the matching Ray actor class at import time.
    from driver import main as train
    from parameter import RUN_FOLDER_NAME

    resume_checkpoint = (
        args.resume_checkpoint.expanduser().resolve()
        if args.resume_checkpoint is not None
        else None
    )
    if (
        resume_checkpoint is not None
        and not resume_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            f"resume checkpoint does not exist: {resume_checkpoint}"
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            resume_checkpoint.parent
            if resume_checkpoint is not None
            else (
                PROJECT_ROOT
                / "experiment"
                / "checkpoints"
                / RUN_FOLDER_NAME
                / f"seed_{args.training_seed}"
            )
        )
    output_dir = output_dir.expanduser().resolve()

    if resume_checkpoint is None:
        if output_dir.exists() and (
            not output_dir.is_dir() or any(output_dir.iterdir())
        ):
            raise FileExistsError(
                "fresh split training requires an empty output directory; "
                f"choose a new --output-dir or archive the existing run: "
                f"{output_dir}"
            )
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"resume output directory does not exist: {output_dir}"
            )
        if resume_checkpoint.parent != output_dir:
            raise ValueError(
                "--output-dir must be the resume checkpoint's parent "
                "directory so checkpoints and TensorBoard logs remain in "
                f"one run: checkpoint={resume_checkpoint}, "
                f"output_dir={output_dir}"
            )
        print(
            "Resuming interrupted training from "
            f"{resume_checkpoint}"
        )

    final_checkpoint = train(
        split_file=args.split_file,
        split_name=args.split_name,
        training_seed=args.training_seed,
        total_env_steps=args.total_env_steps,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
        load_model=resume_checkpoint is not None,
    )
    if final_checkpoint is None:
        raise RuntimeError("training stopped before a final checkpoint was saved")
    print(f"Final checkpoint: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
