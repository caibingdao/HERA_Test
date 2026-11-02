# 基线原始结果与来源

本目录只保留论文 1133 图 / seed 4777 / greedy 协议下的基线原始评测产物。完整上游代码未复制；下表给出上游地址、本地 commit 和本分支的原始结果目录。

## 1. 结果汇总

数据来源：`paper/tables/baseline_results.csv`（由旧 bundle 的 `all_results.csv` 和本分支原始 `episodes.csv` 核对）。路径长度为 1133 回合全部回合平均；成功路径为成功回合平均。

| 方法 | 成功/1133 | 成功率 | 覆盖率 | 全部路径 m | 成功路径 m | 平均步数 | 平均推理 ms | 动作选择 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NBVP | 785 | 69.29% | 92.14% | 295.04 | 237.62 | 56.07 | 38.61 | deterministic explicit planner |
| TARE local (classical run) | 1069 | 94.35% | 98.58% | 235.49 | 232.04 | 50.01 | 25.63 | deterministic explicit planner |
| TARE repaired | 1125 | 99.29% | 99.52% | 360.23 | 357.47 | 67.88 | 128.95 | official TARE seeded randomized coverage + TSP |
| TARE official | 3 | 0.26% | 30.72% | 27.15 | 90.69 | 3.71 | 17.33 | official TARE seeded randomized coverage + TSP |
| DSVP | 1133 | 100.00% | 99.63% | 270.75 | 270.75 | 38.80 | 379.61 | seeded dynamic RRT, official gain/relocation |
| CNN | 1007 | 88.88% | 98.09% | 340.12 | 307.41 | 64.91 | 10.83 | greedy argmax |
| ARiADNE (original) | 1114 | 98.32% | 99.42% | 273.49 | 258.81 | 32.54 | 6.72 | greedy argmax |

## 2. 原始目录与来源

| 方法 | 本分支原始目录 | 原始来源 | checkpoint SHA256 |
|---|---|---|---|
| NBVP | `paper/baselines/nbvp/` | `reports/classical_baselines_full_restart_20261019/nbvp` | `72e896971d276ed1cec29d2e1fed8fdb41395057ecae0a9af936416b1de6b900` |
| TARE local (classical) | `paper/baselines/tare_local_classical/` | `reports/classical_baselines_full_restart_20261019/tare_local` | `33f6f4959a35c4db168be3b266fd76aa4003df518c311cee6a5a6714be0e913e` |
| TARE repaired | `paper/baselines/tare_repaired/` | `reports/tare_repaired_full_20261020/visualizations/tare_local/default/single_seed_4777/data` | `2ac05474db604c930f8cf7db30e9f718aacbb7509783209b88335632b63a4e69` |
| TARE official | `paper/baselines/tare_official/` | `reports/tare_official_full_20261020/visualizations/tare_local/default/single_seed_4777/data` | `6dc3e37f650cf9afe5d6bbc16711d0ff834b5254197fc04a9ce372cd9d95d887` |
| DSVP | `paper/baselines/dsvp/` | `ARiADNE_MAMBA_3.1_TOKEN21/results/data/dsvp/default/single_seed_4777` | `4556090c58e429f77601649b07e502f57db192153547914da1831f1948bbef91` |
| CNN | `paper/baselines/cnn/` | `ARiADNE_MAMBA_3.1_TOKEN21/results/data/cnn/recovery_on/single_seed_4777` | `5eb11b7d0e54913a051e7a7ecff0f49bfebb73703b5a6bcfd6467c96f9e632ce` |
| ARiADNE (original) | `paper/baselines/ariadne_original/` | `ARiADNE_MAMBA_3.1_TOKEN21/results/data/ariadne/single_seed_greedy_4777` | `6ac1827a3dbdd816febd074817fb3116466002f3deb70474d0da643957db8b19` |

每个目录至少包含 `episodes.csv`、`coverage_curve.csv`、`evaluation_metadata.json`；CNN/DSVP 另含 `summary.json`/`publication_complete.json`。

## 3. 上游代码

| 方法 | 上游仓库 | 本地参考 commit |
|---|---|---|
| CNN | https://github.com/caibingdao/ARiADNE-new_version.git | `9e7f4c7` (update cnn) |
| NBVP | https://github.com/caibingdao/ARiADNE-new_version.git | `4019ec5` (nbvp local modifications) |
| TARE local | https://github.com/caibingdao/ARiADNE-new_version.git | `fde7253` (tare local modifications) |
| DSVP | https://github.com/HongbiaoZ/dsv_planner.git | `5ad721d` |
| 原始 ARiADNE | https://github.com/marmotlab/ARiADNE | 上游公开实现；本工作区的 ARiADNE 基线结果见上表 |

评测适配层位于 `paper/evaluator/evaluation_adapters/`，包括 `ariadne.py`、`classical.py`、`cnn.py`、`dsvp.py`、`protocol.py`、`graph_padding.py`。这些适配层定义动作选择、起点和统一评测接口；完整上游仓库请按上表地址获取。

## 4. 写作 caveats

- NBVP/TARE 的 `action_selection` 是确定性显式规划器，不应标成“greedy 神经策略”。
- TARE official 的 3/1133 是官方版本在该二维评测器上的失败结果，保留用于透明性；不能代表 TARE 的正常性能。TARE repaired 是修复版，路径和推理耗时更高。
- DSVP 是二维移植版，不是完整 ROS 部署；其 379.61 ms 的平均推理耗时包含 RRT 规划，不是纯网络前向。
- 所有基线都是 1133 图、每图 1 次、seed 4777；没有多 seed 重复，不能据此声称统计显著性。
- TARE repaired/official 含固定 seed 的随机覆盖与 TSP，不是与 HERA 相同意义上的 greedy argmax。
