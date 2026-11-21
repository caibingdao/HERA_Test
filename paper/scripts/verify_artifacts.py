#!/usr/bin/env python3
"""Verify the HERA final-version test artifact bundle.

Run from the repository root:
    python3 paper/scripts/verify_artifacts.py

Checks:
  - 9 HERA result dirs with required files and 1133 episode rows;
  - 10 baseline dirs with required files and 1133 episode rows;
  - 1133 test maps match test_maps.csv SHA256;
  - derived tables have the expected row counts;
  - no excluded weights/diagnostics/TensorBoard/training logs are present.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRORS: list[str] = []


def fail(msg: str) -> None:
    ERRORS.append(msg)


def count_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def check_hera() -> int:
    dirs = sorted((ROOT / "results/data").glob("*/*/single_seed_4777"))
    if len(dirs) != 9:
        fail(f"HERA result dirs: expected 9, found {len(dirs)}")
    for d in dirs:
        for name in ("episodes.csv", "coverage_curve.csv", "evaluation_metadata.json",
                     "hera_summary.json", "publication_complete.json"):
            if not (d / name).exists():
                fail(f"missing {d.relative_to(ROOT) / name}")
        ep = d / "episodes.csv"
        if ep.exists():
            n = count_rows(ep)
            if n != 1133:
                fail(f"{ep.relative_to(ROOT)}: expected 1133 data rows, found {n}")
        summary = d / "hera_summary.json"
        if summary.exists():
            s = json.load(open(summary, encoding="utf-8"))
            if s.get("maps") != 1133:
                fail(f"{summary.relative_to(ROOT)}: maps != 1133")
            if s.get("successes") is None or s.get("mean_path_all") is None:
                fail(f"{summary.relative_to(ROOT)}: missing success/path fields")
    return len(dirs)


def check_baselines() -> int:
    dirs = sorted(p for p in (ROOT / "paper/baselines").iterdir()
                  if p.is_dir() and (p / "episodes.csv").exists())
    if len(dirs) != 10:
        fail(f"baseline dirs: expected 10, found {len(dirs)}")
    for d in dirs:
        for name in ("episodes.csv", "coverage_curve.csv", "evaluation_metadata.json"):
            if not (d / name).exists():
                fail(f"missing {d.relative_to(ROOT) / name}")
        n = count_rows(d / "episodes.csv")
        if n != 1133:
            fail(f"{d.relative_to(ROOT)}/episodes.csv: expected 1133 rows, found {n}")
    return len(dirs)


def check_maps() -> int:
    rows = list(csv.DictReader(open(ROOT / "paper/maps/test_maps.csv", newline="", encoding="utf-8")))
    if len(rows) != 1133:
        fail(f"test_maps.csv: expected 1133 rows, found {len(rows)}")
    bad = []
    for r in rows:
        p = ROOT / "paper/maps/test_maps" / r["map_id"]
        if not p.exists():
            bad.append(r["map_id"])
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != r["sha256"]:
            bad.append(r["map_id"])
    if bad:
        fail(f"test maps missing/hash mismatch: {len(bad)}; first={bad[:5]}")
    return len(rows) - len(bad)


def check_tables() -> None:
    expected = {
        "hera_final_results.csv": 9,
        "model_selection.csv": 4,
        "baseline_results.csv": 7,
        "checkpoint_manifest.csv": 9,
    }
    for name, n in expected.items():
        p = ROOT / "paper/tables" / name
        if not p.exists():
            fail(f"missing table {p.relative_to(ROOT)}")
            continue
        got = count_rows(p)
        if got != n:
            fail(f"{name}: expected {n} rows, found {got}")


def check_exclusions() -> None:
    forbidden = []
    for p in ROOT.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file() and (
            p.suffix == ".pth"
            or p.name.endswith(".pth.previous")
            or p.name.endswith(".jsonl")
            or p.name == "train.log"
            or p.name.startswith("events.out.tfevents")
        ):
            forbidden.append(str(p.relative_to(ROOT)))
    if forbidden:
        fail(f"excluded artifacts present: {forbidden[:10]}")


def main() -> int:
    n_hera = check_hera()
    n_base = check_baselines()
    n_maps = check_maps()
    check_tables()
    check_exclusions()
    print(f"HERA experiments: {n_hera}/9")
    print(f"Baselines: {n_base}/10")
    print(f"Test maps verified: {n_maps}/1133")
    if ERRORS:
        print("\nFAILURES:")
        for e in ERRORS:
            print(" -", e)
        return 1
    print("\nPASS: all artifact checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
