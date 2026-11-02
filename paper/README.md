# paper/ — 论文实验材料包

本目录是 HERA 最终版本的论文实验材料，和根目录的 HERA 源码/测试结果配套使用。所有数字以根目录 `results/` 和本目录 `baselines/` 的原始文件为准；本目录的 CSV 是可核对的重算表。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 协议、旧版实验汇总、旧版实验章节草稿、架构对比、基线说明、限制、排除清单 |
| `tables/` | HERA 汇总表、模型选型表、基线汇总表、checkpoint 清单、旧版总表、配对地图 |
| `audit/` | 旧 bundle 的协议选择、起点配对、排除/重复清单、发现清单 |
| `baselines/` | NBVP、TARE、DSVP、CNN、原始 ARiADNE 的原始评测结果（各 1133 回合） |
| `maps/` | split seed 42 划分清单、测试地图清单、1133 张测试地图 PNG |
| `evaluator/` | 共享评测脚本与 `evaluation_adapters/`（不含完整上游仓库） |
| `scripts/` | 重算表格与校验产物的脚本 |
| `FINAL_MODEL.json` | 当前主模型与候选 |

## 快速校验

```bash
# 在仓库根目录执行
sha256sum -c CHECKSUMS.sha256
python3 paper/scripts/verify_artifacts.py
```

`verify_artifacts.py` 会检查：

- 9 个 HERA 实验的 `episodes.csv` 是否为 1133 条数据；
- 7 个基线目录的 `episodes.csv` 是否为 1133 条数据；
- 1133 张测试地图的 SHA256 是否与 `maps/test_maps.csv` 一致；
- `hera_final_results.csv`、`baseline_results.csv` 是否能从原始 JSON/CSV 重算一致。

## 口径提醒

- `docs/experiment_summary.md` 是 2026-10-23 的旧 TOKEN21 汇总，只用于协议、基线和 caveats，不是 HERA 最终结果。
- `docs/experiment_section_draft_20260915.md` 是更早的实验章节草稿，结构与写法可参考，数字已过时。
- 主模型候选见 `FINAL_MODEL.json`；写主结果前必须固定一个 checkpoint。
- 原始 `run_config.json`/`evaluation_metadata.json` 保留绝对路径和 GPU UUID，用于溯源；公开脱敏版不要直接修改这些原始记录。
