# audit/ — 旧 bundle 的核验记录

这些文件来自 `paper_experiments_seed4777_1133_20261023/audit/`，用于追溯旧 bundle 的纳入/排除规则和起点配对。

| 文件 | 内容 |
|---|---|
| `selected_protocols.json` | 纳入实验的协议、checkpoint、训练配置选择（约 1.9 MB） |
| `start_pairing.csv` | 各模型逐图起点核对 |
| `excluded_or_duplicate.json` | 被排除或判为重复实验的清单 |
| `discovery_inventory.json` | 扫描到的候选结果清单 |
| `export_environment.json` | 导出时环境；**不是**历史训练环境 |

注意：这些记录对应 2026-10-23 的 TOKEN21 时代 bundle；HERA 最终结果在根目录 `results/`，不要用本目录的协议选择替代 HERA 的 `evaluation_metadata.json`。
