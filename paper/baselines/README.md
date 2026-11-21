# baselines/ — 基线原始评测结果

每个方法目录保留 1133 图 / seed 4777 协议的原始产物：

- `episodes.csv`：1133 行逐图结果；
- `coverage_curve.csv`：逐步覆盖率/路径曲线；
- `evaluation_metadata.json`：协议、起点、checkpoint 信息；
- 部分目录含 `summary.json`、`summary.csv`、`publication_complete.json`。

| 目录 | 方法 | 说明 |
|---|---|---|
| `nbvp/` | NBVP | 确定性显式规划器 |
| [nbvp_official_2d/](nbvp_official_2d/) | RH-NBVP 二维移植 | [实现](nbvp_official_2d/implementation/ARiADNE_NBVP_OFFICIAL_2D/) · [测试图](nbvp_official_2d/visualizations/) |
| [nearest_frontier/](nearest_frontier/) | Nearest Frontier | [实现](nearest_frontier/implementation/ARiADNE_NEAREST_FRONTIER/) · [测试图](nearest_frontier/visualizations/) |
| [ariadne_recovery_on_13000/](ariadne_recovery_on_13000/) | ARiADNE + Recovery（13,000 回合） | [测试图](ariadne_recovery_on_13000/visualizations/) |
| `tare_local_classical/` | TARE local（classical run） | 确定性显式规划器 |
| `tare_repaired/` | TARE repaired | 官方 TARE 的修复版；路径/耗时较高 |
| `tare_official/` | TARE official | 官方版本失败结果（3/1133），保留用于透明性 |
| `dsvp/` | DSVP | 二维移植版，不是完整 ROS 部署 |
| `cnn/` | CNN | greedy argmax |
| `ariadne_original/` | 原始 ARiADNE | greedy argmax |

汇总、checkpoint SHA256、上游地址和 caveats 见 `../docs/BASELINES.md` 与 `../tables/baseline_results.csv`。

旧版七组的汇总表保留其历史版本。新增三组沿用原测试集，均核对了 1133 个逐图起点。

Nearest Frontier 的全量路径样本方差为 **5705.264935 m²**，1119 个成功回合的样本方差为 **5300.595901 m²**，均由 `episodes.csv` 的 `path_length` 按 `n − 1` 重算。ARiADNE + Recovery 13k 的恢复机制在全部 1133 个回合中启用，逐回合 `recovery_trigger_count` 合计 **0 次**。

两种规划器的实现文件保持评测时的版本。复现实验时，将实现目录放回原工作区根目录，与 `ARiADNE_MAMBA_Test/mamba31_map_split` 共享评测器同级部署。
