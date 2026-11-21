"""Evaluate the ETH ASL RH-NBVP planar planner with the shared held-out protocol."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


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
        "--adapter", "ARiADNE_NBVP_OFFICIAL_2D.adapter:create_adapter",
        "--split-file", str(project / "experiment/map_splits/map_split_seed_42.json"),
        "--global-seed", "4777", "--max-steps", "128", "--target-coverage", "0.99",
        "--output-dir", str(args.output_dir.resolve()),
        "--visual-output-dir", str(args.visual_output_dir.resolve()),
    ]
    if args.limit_maps is not None:
        command.extend(["--limit-maps", str(args.limit_maps)])
    with tempfile.TemporaryDirectory(prefix="rh_nbvp_geometry_") as temporary:
        library = Path(temporary) / "visibility.so"
        subprocess.run(["g++", "-O3", "-std=c++17", "-fPIC", "-shared",
                        str(folder / "visibility.cpp"), "-o", str(library)], check=True)
        env["RH_NBVP_GEOMETRY_LIB"] = str(library)
        subprocess.run(command, cwd=project, env=env, check=True)
    if args.limit_maps is None:
        from summarize import summarize
        summarize(args.output_dir.resolve(), args.visual_output_dir.resolve())


if __name__ == "__main__":
    main()
