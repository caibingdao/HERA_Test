"""Verify and summarize the complete shared-protocol RH-NBVP 2D run."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(data, images):
    root = Path(__file__).resolve().parent
    reference = root.parent / (
        "ARiADNE_MAMBA_3.1_TOKEN21/results/data/ariadne/single_seed_greedy_4777"
    )
    metadata = json.loads((data / "evaluation_metadata.json").read_text())
    original = json.loads((reference / "evaluation_metadata.json").read_text())
    for field in ("split_manifest_sha256", "global_seed", "target_coverage",
                  "max_steps", "max_path_length", "episodes_per_map"):
        if metadata[field] != original[field]:
            raise ValueError(f"Protocol differs from reference in {field}")
    if metadata["checkpoint_sha256"] != hashlib.sha256((root / "specification.json").read_bytes()).hexdigest():
        raise ValueError("Planner specification does not match the evaluation")
    if metadata["action_selection"] != "seeded_rh_nbvp_2d":
        raise ValueError("Unexpected planner policy")
    if metadata["adapter_metadata"]["recovery_enabled"]:
        raise ValueError("Recovery was enabled")
    with (data / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (reference / "episodes.csv").open() as stream:
        expected = list(csv.DictReader(stream))
    fields = ("map_id", "eval_seed", "start_cell_x", "start_cell_y")
    if len(rows) != 1133 or len({r["map_id"] for r in rows}) != 1133:
        raise ValueError("Expected one result for each of the 1133 maps")
    if [tuple(r[k] for k in fields) for r in rows] != [tuple(r[k] for k in fields) for r in expected]:
        raise ValueError("Map order, seeds or start cells differ from the paired reference")
    if any(r["error"] or r["recovery_enabled"] != "False" or
           int(r["recovery_steps"]) or int(r["recovery_trigger_count"]) for r in rows):
        raise ValueError("Unexpected episode error or recovery intervention")
    if len(list(images.glob("*.png"))) != 6:
        raise ValueError("Missing evaluation figures")

    successes = [r for r in rows if r["success"] == "True"]
    summary = {
        "model": "RH-NBVP 2D",
        "maps": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows),
        "mean_path_all_m": mean(float(r["path_length"]) for r in rows),
        "mean_path_success_m": mean(float(r["path_length"]) for r in successes) if successes else None,
        "mean_steps": mean(int(r["steps"]) for r in rows),
        "mean_final_coverage": mean(float(r["final_coverage"]) for r in rows),
        "paired_starts_validated": len(rows),
        "recovery_triggers": 0,
        "native_backtracks": sum(row["native_backtracks"] for row in metadata["adapter_metadata"]["native_diagnostics"]),
    }
    temporary = data / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(data / "summary.json")
    print(json.dumps(summary, indent=2))

