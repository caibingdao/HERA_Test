# Data availability

> **2026-11-17 论文结果索引：** 本次确定使用的六组见 [paper/selected_results/README.md](paper/selected_results/README.md)。原始测试 CSV/JSON 及已有图复用，本次补齐原版 ARiADNE 的 6 张测试图；不新增训练日志或训练过程记录。

## Included in this branch

- **HERA test results:** `results/data/` contains 9 experiments, each with `episodes.csv` (1133 data rows), `coverage_curve.csv`, `evaluation_metadata.json`, `hera_summary.json`, and `publication_complete.json`.
- **HERA evaluation provenance:** `results/comparison/`, `results/runs/`, `results/logs/`, `results/cache/`, and `results/visualizations/`.
- **Baseline raw results:** `paper/baselines/` contains NBVP, TARE (classical/repaired/official), DSVP, CNN, original ARiADNE, RH-NBVP 2D, Nearest Frontier, and ARiADNE + Recovery 13k, each with 1133 evaluation episodes. The three added result sets include their evaluation plots.
- **Test maps:** `paper/maps/test_maps/` contains 1133 PNGs; `paper/maps/test_maps.csv` lists `map_id` and SHA256; `paper/maps/map_split_seed_42.json` defines the train/test split (4530/1133).
- **Consolidated tables:** `paper/tables/hera_final_results.csv`, `model_selection.csv`, `baseline_results.csv`, `checkpoint_manifest.csv`.
- **Paper context and limitations:** `PAPER_CONTEXT.md`, `paper/docs/LIMITATIONS.md`, `paper/docs/BASELINES.md`.

## Not included

- Training weights (`*.pth`), per-step diagnostics (`recovery_diagnostics.jsonl`), TensorBoard events, training/control logs, GIFs, and the 4530 training maps. See `paper/docs/EXCLUSIONS.md`.
- `paper/tables/checkpoint_manifest.csv` records the SHA256 of every evaluated checkpoint; the weight files themselves must be obtained from the original workspace or an external archive.

## Protocol

- Test maps: 1133, split seed 42, map split SHA256 `8397d4d1a3b1e6f57fa604b23f59284352a2e6dea6686f55ff62d0f531fcf0c0`.
- Evaluation: global seed 4777, greedy argmax, one episode per map, target coverage 0.99, max 128 decisions, seeded map-marker starts.
- HERA experiments are single test seed 4777 and single training seed 4777; this is not a multi-seed statistical study.

## Verification

From the repository root:

```bash
sha256sum -c CHECKSUMS.sha256
python3 paper/scripts/verify_artifacts.py
```

`verify_artifacts.py` checks the 9 HERA result sets, 10 baseline sets, and 1133 test-map hashes.

## Licensing

The map set and some baseline components are third-party. Redistribution terms are not declared in this branch. See `THIRD_PARTY_NOTICES.md`; confirm the upstream terms before public redistribution or citation.
