#!/usr/bin/env python3
"""Paired successful path lengths with completion constraint and map-cluster CIs."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def truth(value):
    if value not in ("True", "False", "true", "false", "1", "0"):
        raise ValueError(f"invalid boolean: {value}")
    return value in ("True", "true", "1")


def read_run(directory, mode):
    metadata = json.loads((directory / "evaluation_metadata.json").read_text())
    adapter = metadata["adapter_metadata"]
    if adapter.get("temporal_pose_encoding") != mode:
        raise ValueError(f"{directory}: unexpected pose semantics")
    if adapter.get("recovery_enabled") is not False:
        raise ValueError("paired pose report requires recovery_off")
    if adapter.get("temporal_pose_override_allowed", False):
        raise ValueError("checkpoint semantic override invalidates paired experiment")
    with (directory / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = tuple(row[k] for k in ("model_seed", "map_id", "eval_seed", "start_position_seed"))
        if key in result:
            raise ValueError(f"duplicate paired key: {key}")
        success = truth(row["success"])
        length = float(row["path_length"])
        if success and (not math.isfinite(length) or length < 0):
            raise ValueError(f"successful episode has invalid path length: {key}")
        result[key] = row
    return metadata, result


def descriptive(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {"n": len(x), "mean": float(x.mean()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None,
            "std": float(x.std(ddof=1)) if len(x) > 1 else None}


def summarize(values, map_ids, samples, seed):
    values = np.asarray(values, dtype=float)
    ids = sorted(set(map_ids))
    clusters = [np.flatnonzero(np.asarray(map_ids) == key) for key in ids]
    stats = descriptive(values)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(samples):
        indices = np.concatenate([clusters[i] for i in rng.integers(len(ids), size=len(ids))])
        s = descriptive(values[indices])
        if s["mean"] is not None:
            bootstrap.append([s["mean"], s["median"]])
    if len(clusters) > 1 and bootstrap:
        bounds = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        stats.update(mean_ci95=bounds[:, 0].tolist(), median_ci95=bounds[:, 1].tolist())
    else:
        stats.update(mean_ci95=None, median_ci95=None)
    stats["bootstrap_nonempty_samples"] = len(bootstrap)
    return stats


def compare(absolute, delta, output, samples=10000, seed=42):
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    ma, a = read_run(absolute, "absolute")
    mb, b = read_run(delta, "delta_local")
    for key in ("split_manifest_sha256", "eval_seeds", "episodes_per_map", "deterministic",
                "max_steps", "target_coverage", "max_path_length", "evaluation_mode"):
        if ma.get(key) != mb.get(key):
            raise ValueError(f"evaluation protocol mismatch: {key}")
    if not a or a.keys() != b.keys():
        raise ValueError("paired evaluation grids must be identical and nonempty")
    if output.exists():
        raise FileExistsError(output)
    pairs = []
    for key in sorted(a):
        ar, br = a[key], b[key]
        for field in ("start_x", "start_y"):
            if not math.isclose(float(ar[field]), float(br[field]), abs_tol=1e-8):
                raise ValueError(f"unpaired starting location: {key}")
        sa, sb = truth(ar["success"]), truth(br["success"])
        la, lb = float(ar["path_length"]), float(br["path_length"])
        pairs.append(dict(zip(("model_seed", "map_id", "eval_seed", "start_position_seed"), key),
            absolute_success=sa, delta_success=sb, both_success=sa and sb,
            absolute_success_path=la if sa else math.nan,
            delta_success_path=lb if sb else math.nan,
            paired_path_delta=lb-la if sa and sb else math.nan,
            absolute_steps=float(ar["steps"]), delta_steps=float(br["steps"]),
            absolute_inference_ms=float(ar["mean_inference_time_ms"]),
            delta_inference_ms=float(br["mean_inference_time_ms"])))
    map_ids = [r["map_id"] for r in pairs]
    summary = {
        "primary_metric": "successful episode path length in metres",
        "constraint": "completion rate at identical coverage threshold and budget",
        "paired_delta_sign": "delta_local minus absolute; negative path delta is shorter",
        "paired_path_population": "only map/start/seed pairs where BOTH models completed",
        "ci_method": "paired cluster bootstrap over maps; all seeds of a map resampled together",
        "training_seed_limitation": "CIs condition on supplied training seeds, not unseen retraining variability",
        "bootstrap_samples": samples, "bootstrap_seed": seed,
        "pairs": len(pairs), "maps": len(set(map_ids)),
        "both_completed": sum(r["both_success"] for r in pairs),
        "absolute_only_completed": sum(r["absolute_success"] and not r["delta_success"] for r in pairs),
        "delta_only_completed": sum(r["delta_success"] and not r["absolute_success"] for r in pairs),
        "neither_completed": sum(not r["absolute_success"] and not r["delta_success"] for r in pairs),
    }
    for field in ("absolute_success_path", "delta_success_path", "paired_path_delta",
                  "absolute_success", "delta_success"):
        summary[field] = summarize([r[field] for r in pairs], map_ids, samples, seed)
    summary["paired_completion_delta"] = summarize(
        [int(r["delta_success"])-int(r["absolute_success"]) for r in pairs],
        map_ids, samples, seed)
    summary["auxiliary"] = {field: descriptive([r[field] for r in pairs]) for field in (
        "absolute_steps", "delta_steps", "absolute_inference_ms", "delta_inference_ms")}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    map_rows = []
    for map_id in sorted(set(map_ids)):
        rows = [r for r in pairs if r["map_id"] == map_id]
        record = {"map_id": map_id, "pairs": len(rows)}
        for field in ("absolute_success", "delta_success", "absolute_success_path",
                      "delta_success_path", "paired_path_delta"):
            record.update({f"{field}_{k}": v for k, v in descriptive([r[field] for r in rows]).items()})
        map_rows.append(record)
    with (output / "map_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(map_rows[0]))
        writer.writeheader()
        writer.writerows(map_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--absolute-results", type=Path, required=True)
    p.add_argument("--delta-results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    a = p.parse_args()
    print(json.dumps(compare(a.absolute_results, a.delta_results, a.output_dir,
                             a.bootstrap_samples, a.bootstrap_seed), indent=2))


if __name__ == "__main__":
    main()
