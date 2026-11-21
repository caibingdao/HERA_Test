"""Evaluate the nearest-frontier planner with the shared held-out protocol."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    folder = Path(__file__).resolve().parent
    workspace = folder.parent
    project = workspace / "ARiADNE_MAMBA_Test/mamba31_map_split"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-output-dir", type=Path, required=True)
    parser.add_argument("--limit-maps", type=int)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(workspace), MPLBACKEND="Agg")
    command = [
        sys.executable, str(project / "scripts/evaluate_single_seed_greedy.py"),
        "--checkpoint", str(folder / "specification.json"),
        "--adapter", "ARiADNE_NEAREST_FRONTIER.adapter:create_adapter",
        "--split-file", str(project / "experiment/map_splits/map_split_seed_42.json"),
        "--global-seed", "4777", "--max-steps", "128", "--target-coverage", "0.99",
        "--output-dir", str(args.output_dir.resolve()),
        "--visual-output-dir", str(args.visual_output_dir.resolve()),
    ]
    if args.limit_maps is not None:
        command.extend(["--limit-maps", str(args.limit_maps)])
    subprocess.run(command, cwd=project, env=env, check=True)


if __name__ == "__main__":
    main()
