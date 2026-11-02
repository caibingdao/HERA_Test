# docs/ — 论文材料说明

| 文件 | 状态 | 用途 |
|---|---|---|
| `bundle_README.md` | 原始 | 旧实验包 `paper_experiments_seed4777_1133_20261023/README.md` 的副本 |
| `experiment_summary.md` | **历史材料** | 2026-10-23 的 TOKEN21 时代实验汇总；只用于协议、基线、caveats，不能当 HERA 最终结果 |
| `experiment_section_draft_20260915.md` | **历史草稿** | 2026-09-15 的实验章节准备文档；结构可参考，数字已过时 |
| `architecture_comparison.md` | 方法材料 | 原始 ARiADNE → ARiADNE-Mamba → Heuristic Recovery 的架构对比 |
| `BASELINES.md` | 本分支生成 | 基线原始结果、来源、checkpoint SHA256、上游 commit、写作 caveats |
| `LIMITATIONS.md` | 本分支生成 | 统计范围、非纯消融、模型选型、耗时口径、环境/数据限制 |
| `EXCLUSIONS.md` | 本分支生成 | 未纳入本分支的训练产物/地图/历史结果及原因 |

## 建议读法

1. 先读 `../PAPER_CONTEXT.md`（仓库根目录）和 `LIMITATIONS.md`。
2. `experiment_summary.md` 只读第 1、4、5、6 节的协议/基线/caveats；不要复制其中的 TOKEN21 主表。
3. `experiment_section_draft_20260915.md` 只学章节结构；所有数字重新从 `../tables/` 和根目录 `results/` 取。
4. `architecture_comparison.md` 用于方法/架构叙述，不用于结果数字。
