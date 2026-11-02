#!/usr/bin/env python3
"""Build EXPORT_MANIFEST.json and CHECKSUMS.sha256 for this artifact branch.

Run from the repository root:
    python3 paper/scripts/build_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/HERA")
SKIP_FILES = {"EXPORT_MANIFEST.json", "CHECKSUMS.sha256"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def included_files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and ".git" not in p.parts and p.name not in SKIP_FILES:
            out.append(p)
    return out


def exclude_reason(rel: str) -> str | None:
    parts = rel.split(os.sep)
    name = parts[-1]
    if "__pycache__" in parts or name.endswith(".pyc"):
        return "runtime cache"
    if parts[0] == "gifs":
        return "GIF media"
    if parts[0] == "logs":
        return "service/orchestration log"
    if "model" in parts:
        return "training checkpoint"
    if "train" in parts:
        return "TensorBoard/training directory"
    if name == "recovery_diagnostics.jsonl":
        return "per-step training diagnostics"
    if name in {"train.log", "plots.log"} or (len(parts) >= 2 and parts[-2].endswith("_control") and name.endswith(".log")):
        return "training/control log"
    if ".before_resume_" in name or ".before_restart_" in name:
        return "intermediate pre-resume/restart backup"
    if "matplotlib" in parts:
        return "Matplotlib font cache"
    return "not selected for the paper artifact"


def excluded_hera_files(included_rel: set[str]) -> list[dict]:
    out = []
    if not SOURCE.exists():
        return out
    for p in sorted(SOURCE.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(SOURCE))
        if rel in included_rel:
            continue
        reason = exclude_reason(rel)
        if reason:
            out.append({"path": rel, "bytes": p.stat().st_size, "reason": reason})
    return out


def main() -> None:
    inc = included_files()
    inc_rel = {str(p.relative_to(ROOT)) for p in inc}
    manifest = {
        "source_directory": "HERA + paper artifacts",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "included_file_count": len(inc),
        "included_files": [
            {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in inc
        ],
        "excluded_files": excluded_hera_files(inc_rel),
        "excluded_external": [
            {"path": "ARiADNE_MAMBA_Test/maps (train_maps)", "reason": "4530 training maps, ~43.8 MB; only 1133 test maps are included"},
            {"path": "HERA/experiments/**/model", "reason": "checkpoints, ~1.23 GB including .previous"},
            {"path": "HERA/experiments/**/recovery_diagnostics.jsonl", "reason": "per-step diagnostics, ~1.68 GB"},
            {"path": "HERA/gifs", "reason": "GIF media, ~3.0 GB"},
            {"path": "datasets/, map-houseexpo/, maps.zip", "reason": "not part of the 1133-map / seed-4777 protocol"},
            {"path": "ARiADNE_MAMBA_3.1_TOKEN21 full_test/no_* raw dirs", "reason": "historical TOKEN21 results, not final HERA; summaries remain in paper/tables/all_results.csv"},
        ],
        "note": "All HERA test raw results are included. Checkpoints, per-step diagnostics, TensorBoard events, training/control logs, GIFs, training maps, and duplicate historical results are excluded; see paper/docs/EXCLUSIONS.md.",
    }
    (ROOT / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or ".git" in p.parts or p.name == "CHECKSUMS.sha256":
            continue
        lines.append(f"{sha256(p)}  {p.relative_to(ROOT)}")
    (ROOT / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote EXPORT_MANIFEST.json ({len(inc)} included files) and CHECKSUMS.sha256 ({len(lines)} files)")


if __name__ == "__main__":
    main()
