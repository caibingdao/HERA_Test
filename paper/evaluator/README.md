# evaluator/ — 共享评测器适配层

本目录保留 HERA/基线评测所需的共享评测脚本和适配层，不包含完整上游基线仓库。

## 文件

- `scripts/evaluate_single_seed_greedy.py`：单 seed greedy 评测入口。
- `scripts/evaluate_on_split.py`：按 split 评测。
- `scripts/run_classical_baselines.py`：经典基线评测入口。
- `scripts/split_maps.py`：地图划分脚本（生成 `map_split_seed_42.json` 的逻辑）。
- `scripts/aggregate_results.py`：结果汇总。
- `evaluation_adapters/ariadne.py`：ARiADNE/HERA 图与动作适配。
- `evaluation_adapters/classical.py`：NBVP/TARE 等经典规划器适配。
- `evaluation_adapters/cnn.py`：CNN 适配。
- `evaluation_adapters/dsvp.py`：DSVP 二维移植适配。
- `evaluation_adapters/protocol.py`：统一协议/起点/指标接口。
- `evaluation_adapters/graph_padding.py`：图 padding 工具。

## 运行前提

这些脚本原本位于完整的 `ARiADNE_MAMBA_Test/mamba31_map_split/` 工程中，依赖 `env.py`、`sensor.py`、`parameter.py`、`node_manager.py`、`quads.py`、`model.py` 等模块。单独复制本目录不能直接运行；需要把它们放回完整评测器树，或把根目录 HERA 源码作为模型端，与完整评测器一起使用。

评测协议见根目录 `results/README.md`；地图划分见 `../maps/map_split_seed_42.json`。
