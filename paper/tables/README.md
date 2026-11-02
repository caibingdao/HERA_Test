# tables/ — 汇总表说明

所有表由 `paper/scripts/build_tables.py` 从原始 JSON/CSV 重算；`paper/scripts/verify_artifacts.py` 可复核。

| 文件 | 内容 | 是否可直接用于论文 |
|---|---|---|
| `hera_final_results.csv` | 9 个 HERA 测试实验的汇总（成功数、覆盖率、路径、步数、recovery 触发、checkpoint SHA256） | 是；主结果需先按 `FINAL_MODEL.json` 固定主模型 |
| `model_selection.csv` | streak2/3 × 10k/12k 四个 checkpoint 的选型对照（从 `model_selection_20261026.md` 解析） | 是；仅作选型，不与其他表混行 |
| `baseline_results.csv` | 7 个基线方法的汇总、checkpoint SHA256、原始目录和 caveat | 是；注意 TARE official 是失败版本 |
| `checkpoint_manifest.csv` | 9 个 HERA 评测 run 的 checkpoint SHA256、训练回合、strike、split hash | 用于溯源/校验 |
| `all_results.csv` | 旧 bundle 的全部合规实验总表（含 TOKEN21 历史行） | **历史对照**；不能把 TOKEN21 行当 HERA 最终结果 |
| `recovery_paired_maps.csv` | 旧 bundle 的 full_test recovery 逐图配对 | 历史对照；HERA 自己的配对见 `results/comparison/*/paired_maps.csv` |
| `recovery_pair_summary.json` | 旧 bundle 的配对汇总 | 历史对照 |

## 字段口径

- `episodes`：1133（每图 1 回合）。
- `success_rate`：0–1；旧表 `success_rate_pct` 为百分数。
- `mean_coverage`：`final_coverage` 的回合平均。
- `mean_path_all_m`：1133 回合全部回合平均。
- `mean_path_success_m`：成功回合平均；全部成功时与 `mean_path_all_m` 相同。
- `mean_steps`：平均决策步数（不是路径长度）。
- `recovery_triggers`：recovery 控制器触发次数。
- `checkpoint_sha256`：评测时加载的 checkpoint 哈希；权重文件本身不在本分支。
