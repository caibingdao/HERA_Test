# HERA final version 1 — 论文实验材料与原始测试结果

本分支（`hera_final_version1`）是 HERA 最终方法源码、全部 HERA 测试原始结果、论文实验材料、基线原始结果和测试地图的冻结快照。分支根目录保持 HERA 源码布局，论文材料集中在 `paper/`。

## 1. 这个分支包含什么

- **HERA 最终源码**：根目录的 `model.py`、`agent.py`、`worker.py`、`runner.py`、`driver.py`、`env.py`、`sensor.py`、`parameter.py`、`node_manager.py`、`replay_buffer.py`、`temporal_pose.py`、`local_revisit_recovery.py`、`recovery_artifacts.py` 等，以及 `scripts/`、`models/`、`configs/`、`tools/`。
- **HERA 全部测试原始结果**：`results/data/` 下 9 个实验目录，每个包含 `episodes.csv`、`coverage_curve.csv`、`evaluation_metadata.json`、`hera_summary.json`、`publication_complete.json`；另有 `results/comparison/`、`results/runs/`、`results/logs/`、`results/cache/` 和 `results/visualizations/`。
- **训练过程配置与训练回合表**：`experiments/` 下各实验的 `run_config.json`、`complete.json`、`continuation.json`、`episodes.csv`、`plots/`、控制元数据。不包含训练权重、逐步诊断、TensorBoard 事件和训练日志。
- **论文材料**：`paper/docs/`、`paper/tables/`、`paper/audit/`。
- **基线原始结果**：`paper/baselines/` 下 NBVP、TARE（classical / repaired / official）、DSVP、CNN、原始 ARiADNE，每项 1133 回合。
- **测试地图**：`paper/maps/test_maps/` 下 1133 张测试地图 PNG，SHA256 与 `paper/maps/test_maps.csv` 一致；划分清单为 `paper/maps/map_split_seed_42.json`。
- **评测器适配层**：`paper/evaluator/` 下共享评测脚本和 `evaluation_adapters/`。

## 2. 快速导航

| 目标 | 先读 |
|---|---|
| 论文实验部分怎么写、每个数字看哪里 | [PAPER_CONTEXT.md](PAPER_CONTEXT.md) |
| HERA 最终方法定义 | [README_HERA.md](README_HERA.md) |
| 测试协议与结果文件语义 | [results/README.md](results/README.md) |
| 四个最终 checkpoint 的选型 | [results/comparison/model_selection_20261026.md](results/comparison/model_selection_20261026.md) |
| 所有 HERA 实验汇总表 | [paper/tables/hera_final_results.csv](paper/tables/hera_final_results.csv) |
| 基线汇总表 | [paper/tables/baseline_results.csv](paper/tables/baseline_results.csv) |
| 旧版实验汇总（历史口径） | [paper/docs/experiment_summary.md](paper/docs/experiment_summary.md) |
| 论文实验部分旧草稿（结构参考） | [paper/docs/experiment_section_draft_20260915.md](paper/docs/experiment_section_draft_20260915.md) |
| 方法/架构对比 | [paper/docs/architecture_comparison.md](paper/docs/architecture_comparison.md) |
| 已知限制和写作口径 | [paper/docs/LIMITATIONS.md](paper/docs/LIMITATIONS.md) |
| 哪些原始文件没有放进仓库 | [paper/docs/EXCLUSIONS.md](paper/docs/EXCLUSIONS.md) |

## 3. 主模型口径

`paper/FINAL_MODEL.json` 记录当前默认主模型和候选：

- 默认（与根目录 `baseline.json` 一致）：`recovery_streak2_seed4777_10k_20261024`，10000 训练回合，recovery 门槛 2/2，测试平均路径 251.643 m。
- 备选：`recovery_streak3_seed4777_10k_20261024`，10000 训练回合，门槛 3/3，测试平均路径 253.563 m，但平均决策数少约 22.7%；`model_selection_20261026.md` 从系统效率角度推荐它。

写论文主结果前必须二选一，并在 `paper/FINAL_MODEL.json` 中固定。两组不能混用为同一模型。

## 4. 数据完整性

- 9 个 HERA 测试实验，每个 `episodes.csv` 均为 1133 行数据（加表头 1134 行），`publication_complete.json` 存在。
- 7 个基线原始目录，每个 `episodes.csv` 均为 1133 行数据。
- 1133 张测试地图与 `paper/maps/test_maps.csv` 的 SHA256 逐一匹配。
- 根目录 `CHECKSUMS.sha256` 覆盖本分支所有纳入文件；可用以下命令校验：

```bash
sha256sum -c CHECKSUMS.sha256
```

- `EXPORT_MANIFEST.json` 记录纳入文件清单和排除文件及原因。

## 5. 明确未纳入的内容

详见 [paper/docs/EXCLUSIONS.md](paper/docs/EXCLUSIONS.md)。主要包括：

- 训练权重 `*.pth`（HERA 9 个约 650 MB）。
- `recovery_diagnostics.jsonl`（逐步诊断，约 2.0 GB）。
- TensorBoard 事件、训练日志、控制日志。
- GIF 演示文件、训练地图（4530 张）、完整上游基线仓库。
- 五 seed、indoors、自生成地图、utility 等不属于本论文 1133 图 / seed 4777 协议的结果。
- 旧 TOKEN21/full_test 的原始结果文件（其汇总数字保留在 `paper/tables/all_results.csv`，仅作历史对照）。

## 6. 数据、许可与未脱敏说明

- 1133 张测试地图来自原始 ARiADNE 地图集；公开再分发前请确认上游数据许可。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。
- NBVP、TARE、DSVP、CNN 的完整上游代码未复制，仅保留评测适配层和原始结果；上游地址与本地 commit 见 [paper/docs/BASELINES.md](paper/docs/BASELINES.md)。
- 本分支未改动任何原始结果文件；`paper/` 下的 CSV 是由这些原始文件重算/整理的派生表。
- **本分支是未脱敏的溯源归档。** 原始 `episodes.csv`、`evaluation_metadata.json`、`run_config.json`、`baseline.json`、`results/cache/`、`results/logs/`、`paper/audit/` 等文件保留原始绝对路径（`/data/home/lzy/...`）、GPU UUID 和主机名；这是为了可追溯性，不是脱敏发布版。若需要公开脱敏版，请另建分支/快照生成 redacted 副本，不要修改本分支的原始记录。
- 本分支没有 `LICENSE` 文件；在公开再分发或引用前，作者需要为自己的代码添加许可证，并确认上游 ARiADNE/TARE/DSVP/NBVP/CNN 及地图集的许可。当前状态见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 7. 快照来源

- 源码来源：本工作区 `HERA/`（最终版本，含 `new_free_cell_count_v1` 反馈定义）。
- 结果来源：`HERA/results/` 与 `HERA/experiments/`。
- 论文材料来源：`paper_experiments_seed4777_1133_20261023/`、`reports/paper_experiments_20260915/`、`ARIADNE_TO_MAMBA_HEURISTIC_ARCHITECTURE_COMPARISON.md`。
- 基线来源：`reports/classical_baselines_full_restart_20261019/`、`reports/tare_repaired_full_20261020/`、`reports/tare_official_full_20261020/`、`ARiADNE_MAMBA_3.1_TOKEN21/results/data/`。
