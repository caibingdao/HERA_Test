# baselines/ — 基线原始评测结果

每个方法目录保留 1133 图 / seed 4777 协议的原始产物：

- `episodes.csv`：1133 行逐图结果；
- `coverage_curve.csv`：逐步覆盖率/路径曲线；
- `evaluation_metadata.json`：协议、起点、checkpoint 信息；
- 部分目录含 `summary.json`、`summary.csv`、`publication_complete.json`。

| 目录 | 方法 | 说明 |
|---|---|---|
| `nbvp/` | NBVP | 确定性显式规划器 |
| `tare_local_classical/` | TARE local（classical run） | 确定性显式规划器 |
| `tare_repaired/` | TARE repaired | 官方 TARE 的修复版；路径/耗时较高 |
| `tare_official/` | TARE official | 官方版本失败结果（3/1133），保留用于透明性 |
| `dsvp/` | DSVP | 二维移植版，不是完整 ROS 部署 |
| `cnn/` | CNN | greedy argmax |
| `ariadne_original/` | 原始 ARiADNE | greedy argmax |

汇总、checkpoint SHA256、上游地址和 caveats 见 `../docs/BASELINES.md` 与 `../tables/baseline_results.csv`。

这些目录来自旧 bundle 的基线结果；它们不是 HERA 最终模型的消融。不要把基线行与 HERA 的 `results/data/` 行混为同一实验。
