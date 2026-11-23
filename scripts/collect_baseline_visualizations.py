"""Collect audited baseline results under HERA/results/visualizations."""
import argparse
import csv
import filecmp
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


BASELINES = {
    "nearest_frontier": ("ARiADNE_NEAREST_FRONTIER", "ariadne-nearest-frontier-4777-summary.service"),
    "nbvp_official_2d": ("ARiADNE_NBVP_OFFICIAL_2D", "ariadne-rh-nbvp-2d-4777.service"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", choices=BASELINES)
    args = parser.parse_args()
    hera = Path(__file__).resolve().parents[1]
    folder, unit = BASELINES[args.baseline]
    source = hera.parent / folder / "results"
    data = source / "data/single_seed_4777"
    visual = source / "visualizations/single_seed_4777"
    summary_file = data / "summary.json"
    while not summary_file.is_file():
        check = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit])
        if check.returncode:
            # Check again in case the evaluator just wrote the file and exited.
            if not summary_file.is_file():
                raise RuntimeError(f"{unit} stopped without an audited summary")
            break
        time.sleep(20)

    summary = json.loads(summary_file.read_text())
    if summary["maps"] != 1133 or summary["paired_starts_validated"] != 1133:
        raise ValueError("Expected a complete suite of 1133 paired maps")
    with (data / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1133 or len({row["map_id"] for row in rows}) != 1133 or any(
        row["eval_seed"] != "4777" or row["error"] for row in rows
    ):
        raise ValueError("Unexpected map count, seed, or errors")
    figures = sorted(visual.glob("*.png"))
    if len(figures) != 6:
        raise ValueError("Expected six evaluation plots")
    files = figures + [data / name for name in (
        "episodes.csv", "coverage_curve.csv", "evaluation_metadata.json", "summary.json"
    )]
    destination = hera / "results/visualizations" / args.baseline / "single_seed_4777"
    if destination.exists():
        if all((destination / item.name).is_file() and
               filecmp.cmp(item, destination / item.name, shallow=False) for item in files):
            print(f"Already collected: {destination}")
            return
        raise FileExistsError(f"Different artifacts already exist in {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collect-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        for item in files:
            shutil.copy2(item, staging / item.name)
        staging.rename(destination)
    print(f"Collected {len(files)} files: {destination}")


if __name__ == "__main__":
    main()
