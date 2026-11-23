# 评测结果说明

本目录保存 HERA 完全体基线（`recovery_streak2_seed4777_10k_20261024`，checkpoint SHA256 `8d126a76…6de568bb`）的离线评测产物。

## full_test_seed4777/

1133 张 test 地图的完整测试结果，协议见 `evaluation_metadata.json`：

- `episodes.csv`：逐图结果（起点、seed、路径长度、决策数、覆盖率、成功、触发次数等）。
- `coverage_curve.csv`：逐图逐决策的覆盖率曲线。
- `hera_summary.json`：汇总指标（1133/1133 成功，平均路径 251.643 m，平均决策 45.591，平均覆盖率 99.58%，触发 16 次）。
- `evaluation_metadata.json`：checkpoint、split 哈希、adapter 配置、recovery 参数、评测协议与耗时。
- `publication_complete.json`：发布校验记录（1133 张地图全部校验通过）。

## visualizations/

由上述结果生成的 6 张图：总览、覆盖率阈值、效率分布、探索进度、地图可靠性、失败分析。

## comparison/

- `model_selection_20261026.md`：同批四个最终 checkpoint 的选型对照与建议。
- `summary.json`：3 步 vs 2 步 10k 权重的评测汇总。
- `paired_maps.csv`：1133 张地图逐图配对路径差。
- `README.md`：对照实验协议说明。

## recovery_streak2_seed4777.json

完整测试启动配置：checkpoint 路径/哈希、split 哈希、地图数、seed、greedy 设置、recovery 配置、输出目录。

注意：本包未打包生成这些结果所需的共享评测器（`mamba31_map_split`，约 7.3 GB）。评测脚本原样保留在 `scripts/`，用于追溯；重跑完整测试需在工作区按原路径提供该评测器与地图数据。
