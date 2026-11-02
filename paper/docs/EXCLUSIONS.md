# 未纳入本分支的原始文件及原因

本分支的目标是“论文可核对 + 全部 HERA 测试原始结果”。以下文件或数据被有意排除；需要时从原始工作区或上游获取，不要因为本分支缺失就推断原实验没有这些产物。

## 1. HERA 训练/运行产物

| 内容 | 规模（原始工作区） | 排除原因 |
|---|---:|---|
| `HERA/experiments/**/model/*.pth`、`*.pth.previous` | 约 1.23 GB（18 个文件，单文件约 71.5 MB） | 训练权重；GitHub 单文件/仓库体积不适合，且不是测试原始结果 |
| `HERA/experiments/**/recovery_diagnostics.jsonl` | 约 1.68 GB（9 个文件） | 逐步 recovery 诊断；训练期逐 step 日志，不是测试结果 |
| `HERA/experiments/**/train/events.out.tfevents*` | 约 4.8 MB | TensorBoard 二进制；需要的曲线应导出为 CSV |
| `HERA/experiments/**/train.log`、`_control/*.log` | 约 16.1 MB | 训练/控制日志，不是测试结果 |
| `HERA/gifs/**` | 约 3.0 GB | 演示动画；论文插图只用静态图或外链视频 |
| `HERA/logs/**` | 约 0.2 MB | 服务/编排日志；测试评测日志已放在 `results/logs/` |
| `__pycache__/`、`.pytest_cache/`、`.vscode/` | 小 | 运行时缓存/本地配置 |
| `experiments/**/matplotlib/` | 小 | Matplotlib 字体缓存 |
| `experiments/**/episodes.csv.before_resume_*` | 小 | 续训前的中间 CSV 备份，不是最终训练回合表 |

## 2. 地图与数据

| 内容 | 规模 | 排除原因 |
|---|---:|---|
| 训练地图 4530 张 | 约 43.8 MB | 本分支只保留 1133 张测试地图；训练划分见 `paper/maps/map_split_seed_42.json` |
| 完整地图集 5663 张 | 约 67 MB | 与训练地图/上游地图集重复；测试地图已单独保留 |
| `datasets/indoors/`、`map-houseexpo/`、`maps.zip` | 数十 MB | 不属于本论文 1133 图 / seed 4777 协议；第三方许可需单独确认 |
| 自生成 dungeon 地图 | — | 不属于本论文主实验协议 |
| `HERA_token5/` 及其变体 | — | 未完成的 token 变体，不是最终方法 |

## 3. 历史/重复结果

| 内容 | 排除原因 |
|---|---|
| `ARiADNE_MAMBA_3.1_TOKEN21/` 的 full_test/no_target/no_motion/no_feedback 原始目录 | TOKEN21 时代结果，非 HERA 最终结果；汇总数字保留在 `paper/tables/all_results.csv` |
| `ARiADNE_MAMBA_3.1/`、`ARiADNE_ORIGINAL_PARALLEL/`、`ARiADNE_MAMBA_Test/mamba31_map_split` 的历史代码 | 与 HERA 源码重复/历史版本，容易混版本 |
| 五 seed（100–104）、indoors、自生成地图、utility_1/utility_10 结果 | 不属于本论文 1133 图 / seed 4777 单 seed 协议 |
| smoke、probe、failed runs、`failed_runs/` | 非正式结果 |
| `reports/*/comparison/coverage_curve.csv` 等合并曲线 | 约 139.8 MB，是多个模型合并的派生曲线；各模型原始 `coverage_curve.csv` 已放在 `paper/baselines/` |
| `paper_experiments_seed4777_1133_20261023/sources/` | 1.9 GB，含权重/地图/重复代码；本分支只保留 docs/tables/audit 和选定的基线原始结果 |
| `training_runs/` | 1.6 GB，训练运行目录；配置/回合表已按实验复制到 `experiments/` |

## 4. 基线

- 只保留 NBVP、TARE（classical/repaired/official）、DSVP、CNN、原始 ARiADNE 的 1133 图 / seed 4777 原始评测产物。
- 不复制 NBVP/TARE/DSVP/CNN 的完整上游仓库；保留 `paper/evaluator/evaluation_adapters/` 和 `paper/docs/BASELINES.md` 中的上游地址/commit。
- 五 seed、indoors、自生成地图、utility 基线结果未纳入。

## 5. 需要完整权重时

权重不在本分支。获取方式：

1. 从原始工作区 `HERA/experiments/<experiment>/model/checkpoint.pth` 取。
2. 用 `paper/tables/checkpoint_manifest.csv` 中的 SHA256 校验。
3. 不要把旧 TOKEN21 checkpoint 当作 HERA 新语义下的最终模型；`README_HERA.md` 说明旧覆盖率反馈 checkpoint 会被加载检查拒绝。
