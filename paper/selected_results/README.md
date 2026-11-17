# 论文最终选定的六组测试结果（2026-11-17）

本索引固定本次论文使用的六组实验。只包含**测试原始 CSV、测试配置/汇总 JSON、可视化 PNG**，不新增训练日志、逐步训练诊断、TensorBoard 或其他运行日志。已有数据按 SHA256 核对后复用，不重复提交。

| 实验 | 训练回合 | 训练 seed | 成功数/1133 | 全部地图平均路径/m | 平均决策步数 | recovery 触发 |
|---|---:|---:|---:|---:|---:|---:|
| ariadne_original | 原记录未确认 | 0 | 1114 | 273.491418 | 32.536628 | 0 |
| ablation_no_recovery_10000 | 10000 | 4777 | 1122 | 259.202999 | 34.060018 | 0 |
| ablation_no_feedback_7500 | 7500 | 4777 | 1133 | 256.894129 | 34.264784 | 9 |
| ablation_no_motion_7500 | 7500 | 4777 | 1133 | 253.844182 | 43.023831 | 2 |
| ablation_no_target_7000 | 7000 | 4777 | 1133 | 257.194898 | 39.760812 | 4 |
| recovery_streak2 | 10000 | 4777 | 1133 | 251.642954 | 45.591350 | 16 |

## 数据与可视化位置

| 实验 | 原始测试数据 | 原始测试图 |
|---|---|---|
| ariadne_original | [CSV / JSON](../baselines/ariadne_original) | [6 PNG](../../results/visualizations/ariadne_original/single_seed_greedy_4777) |
| ablation_no_recovery_10000 | [CSV / JSON](../../results/data/ablation_no_recovery_10000/recovery_off/single_seed_4777) | [6 PNG](../../results/visualizations/ablation_no_recovery_10000/recovery_off/single_seed_4777) |
| ablation_no_feedback_7500 | [CSV / JSON](../../results/data/ablation_no_feedback_7500/recovery_on/single_seed_4777) | [6 PNG](../../results/visualizations/ablation_no_feedback_7500/recovery_on/single_seed_4777) |
| ablation_no_motion_7500 | [CSV / JSON](../../results/data/ablation_no_motion_7500/recovery_on/single_seed_4777) | [6 PNG](../../results/visualizations/ablation_no_motion_7500/recovery_on/single_seed_4777) |
| ablation_no_target_7000 | [CSV / JSON](../../results/data/ablation_no_target_7000/recovery_on/single_seed_4777) | [6 PNG](../../results/visualizations/ablation_no_target_7000/recovery_on/single_seed_4777) |
| recovery_streak2 | [CSV / JSON](../../results/data/recovery_streak2/recovery_on/single_seed_4777) | [6 PNG](../../results/visualizations/recovery_streak2/recovery_on/single_seed_4777) |

共 **28 个原始 CSV/JSON 文件、36 张 PNG**。原版 ARiADNE 的 3 个 CSV/metadata 文件和五组 HERA 的 25 个 CSV/JSON 文件、30 张图均已存在且与本地相同；本次只补齐缺失的原版 ARiADNE 6 张图，并添加本索引及校验清单。

## 统一测试协议

六组均为原 1133 张测试地图，每图一次，**评测 seed 4777、greedy argmax、99% 目标覆盖率、最多 128 次决策、无路径长度上限**。全局随机数在模型加载后初始化一次，地图之间不重新设置；1133 个逐图起点和地图顺序已完全配对。划分为 4530 训练图/1133 测试图，见 [map_split_seed_42.json](../maps/map_split_seed_42.json)，SHA256：`8397d4d1a3b1e6f57fa604b23f59284352a2e6dea6686f55ff62d0f531fcf0c0`。

`summary.csv` 从原始 episodes.csv 重算，包含全部地图和成功地图的路径均值。主表采用全部地图均值。原版 ARiADNE 的 metadata 记录**训练模型 seed=0**，评测 seed=4777；HERA 五组训练和评测 seed 均为4777。原版训练回合数未经原记录确认，故留空，不猜测。

## 版本与写作口径

- 主模型固定为 `recovery_streak2` **10k**，checkpoint SHA256 为 `8d126a76433fc33b8a61ddc4e9288ea30e141e11257c759b12e99d5c6de568bb`。不混入12k或streak3。
- `no_target_7000` 是21D token中的16D目标特征通道置零，不是Token5或旧no_target7500。
- `no_feedback_7500` 屏蔽 token 的 reward/新增自由栅格输入；RL reward仍计算，recovery仍使用原始新增自由栅格数。
- recovery-off10000是另行训练的模型，不是对同一checkpoint仅切换推理开关。
- 训练预算分别为10000、7500、7500、7000及主模型10000，不应声称同预算消融。单次训练/测试不代表多训练seed统计。
- 旧分支中的其他历史对照保留；本次论文选择以本目录selection.json为准。

## 核验

```bash
python3 paper/scripts/verify_selected_results.py
sha256sum -c CHECKSUMS.sha256
```

- [summary.csv](summary.csv)：六组论文汇总。
- [selection.json](selection.json)：精确选择、协议、checkpoint和存储位置。
- [start_pairing.csv](start_pairing.csv)：1133张地图的共同起点。
- [artifact_manifest.json](artifact_manifest.json)：每个原始文件的来源、字节数、SHA256和复用/新增状态。
