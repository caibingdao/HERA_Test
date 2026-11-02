#!/usr/bin/env python3
"""Rebuild paper/tables/*.csv from raw HERA and baseline artifacts.

Run from the repository root:
    python3 paper/scripts/build_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "tables"

BASELINE_SPEC = [
    ("nbvp", "reports/classical_baselines_full_restart_20261019/nbvp", "paper/baselines/nbvp",
     "deterministic explicit planner"),
    ("tare_local_classical", "reports/classical_baselines_full_restart_20261019/tare_local",
     "paper/baselines/tare_local_classical", "deterministic explicit planner"),
    ("tare_repaired",
     "reports/tare_repaired_full_20261020/visualizations/tare_local/default/single_seed_4777/data",
     "paper/baselines/tare_repaired", "official TARE seeded randomized coverage + TSP; repaired run"),
    ("tare_official",
     "reports/tare_official_full_20261020/visualizations/tare_local/default/single_seed_4777/data",
     "paper/baselines/tare_official", "official TARE run; near-zero success, kept for transparency"),
    ("cnn", "ARiADNE_MAMBA_3.1_TOKEN21/results/data/cnn/recovery_on/single_seed_4777",
     "paper/baselines/cnn", "greedy argmax"),
    ("dsvp", "ARiADNE_MAMBA_3.1_TOKEN21/results/data/dsvp/default/single_seed_4777",
     "paper/baselines/dsvp", "seeded dynamic RRT; 2D port, not full ROS deployment"),
    ("ariadne_original", "ariadne", "paper/baselines/ariadne_original", "original ARiADNE baseline"),
]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise SystemExit(f"no rows for {path}")
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_hera_results() -> None:
    runmap = {}
    for rp in glob.glob(str(ROOT / "results/runs/*.json")):
        rd = json.load(open(rp, encoding="utf-8"))
        runmap[os.path.basename(rp)[:-5].removesuffix("_seed4777")] = rd

    rows = []
    pattern = str(ROOT / "results/data/*/*/single_seed_4777/hera_summary.json")
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p, encoding="utf-8"))
        rel = os.path.relpath(os.path.dirname(p), ROOT)
        parts = rel.split(os.sep)
        exp = parts[2]
        recovery = parts[3]
        run = runmap.get(exp, {})
        mode = d.get("recovery") or run.get("recovery")
        if not mode:
            mode = "off" if "recovery_off" in recovery else "on"
        training_episodes = d.get("training_episodes") or run.get("training_episodes")
        if training_episodes is None:
            training_episodes = 10000 if ("10k" in exp or exp in {"recovery_streak2", "recovery_streak3"}) else ""
        rows.append({
            "experiment": exp,
            "recovery": mode,
            "streak": d.get("streak", run.get("streak")),
            "training_episodes": training_episodes,
            "maps": d.get("maps"),
            "successes": d.get("successes"),
            "success_rate": d.get("success_rate"),
            "mean_coverage": d.get("mean_coverage"),
            "mean_path_all_m": d.get("mean_path_all"),
            "mean_path_success_m": d.get("mean_path_success"),
            "mean_steps": d.get("mean_steps"),
            "recovery_triggers": d.get("recovery_triggers"),
            "checkpoint_sha256": d.get("checkpoint_sha256"),
            "source_dir": rel,
        })
    write_csv(TABLES / "hera_final_results.csv", rows)


def build_model_selection() -> None:
    md = (ROOT / "results/comparison/model_selection_20261026.md").read_text(encoding="utf-8").splitlines()
    rows = []
    for line in md:
        if line.startswith("|recovery_streak"):
            parts = [x.strip() for x in line.strip("|").split("|")]
            rows.append({
                "version": parts[0],
                "successes": parts[1],
                "mean_path_m": parts[2],
                "mean_steps": parts[3],
                "mean_episode_seconds": parts[4],
                "recovery_triggers": parts[5],
            })
    runmap = {}
    for p in glob.glob(str(ROOT / "results/runs/*.json")):
        d = json.load(open(p, encoding="utf-8"))
        runmap[os.path.basename(p)[:-5]] = d
    for r in rows:
        d = runmap.get(r["version"] + "_seed4777", {})
        r["training_episodes"] = d.get("training_episodes", 10000)
        r["checkpoint_sha256"] = d.get("checkpoint_sha256", "")
    write_csv(TABLES / "model_selection.csv", rows,
              ["version", "successes", "mean_path_m", "mean_steps", "mean_episode_seconds",
               "recovery_triggers", "training_episodes", "checkpoint_sha256"])


def build_baselines() -> None:
    all_rows = list(csv.DictReader(open(TABLES / "all_results.csv", newline="", encoding="utf-8")))
    by_exp = {r["experiment"]: r for r in all_rows}
    rows = []
    for name, exp, local, caveat in BASELINE_SPEC:
        r = by_exp.get(exp)
        if not r:
            raise SystemExit(f"missing baseline row: {name} ({exp})")
        rows.append({
            "display_name": name,
            "experiment": r["experiment"],
            "episodes": r["episodes"],
            "successes": r["successes"],
            "success_rate_pct": r["success_rate_pct"],
            "coverage_pct": r["coverage_pct"],
            "path_length_all_m": r["path_length_all_m"],
            "path_length_success_m": r["path_length_success_m"],
            "steps_all": r["steps_all"],
            "inference_ms_episode_mean": r["inference_ms_episode_mean"],
            "action_selection": r["action_selection"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "source": r["source"],
            "local_path": local,
            "caveat": caveat,
        })
    write_csv(TABLES / "baseline_results.csv", rows)


def build_checkpoints() -> None:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "results/runs/*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        rows.append({
            "run": os.path.basename(p)[:-5],
            "checkpoint_sha256": d.get("checkpoint_sha256", ""),
            "training_episodes": d.get("training_episodes", ""),
            "streak": d.get("streak", ""),
            "recovery": d.get("recovery", ""),
            "split_sha256": d.get("split_sha256", ""),
            "maps": d.get("maps", ""),
            "global_seed": d.get("global_seed", ""),
            "action_selection": d.get("action_selection", ""),
        })
    write_csv(TABLES / "checkpoint_manifest.csv", rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    build_hera_results()
    build_model_selection()
    build_baselines()
    build_checkpoints()
    print("rebuilt paper/tables: hera_final_results.csv, model_selection.csv, "
          "baseline_results.csv, checkpoint_manifest.csv")


if __name__ == "__main__":
    main()
