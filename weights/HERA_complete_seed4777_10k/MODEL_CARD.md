# Model Card — HERA Complete (recovery_streak2, 10k)

## 基本信息

| 项目 | 值 |
|---|---|
| 模型名称 | HERA 完全体（recovery on，21D motion token，frontier escape） |
| 实验目录 | `recovery_streak2_seed4777_10k_20261024` |
| 权重文件 | `checkpoint.pth` |
| 文件大小 | 71,573,717 字节 |
| SHA256 | `8d126a76433fc33b8a61ddc4e9288ea30e141e11257c759b12e99d5c6de568bb` |
| 训练回合 | 10,000（从零训练，精确回合预算） |
| 训练 seed | 4777 |
| 采集器 | 原 `driver.py` 五 worker Ray 采样 + 独立 GPU learner |
| 训练地图 | 4,530 张（`map_split_seed_42.json` 的 train 划分，SHA256 `8397d4d1…fcf0c0`） |
| 源文件快照 | `run_config.json` 的 `source_sha256`（本包程序哈希与之逐项一致） |

## 模型语义

- **反馈信号**：`new_free_cell_count_v1`。每步统计 belief map 中 UNKNOWN→FREE 的原始栅格数，不做面积归一化；已占据格与已知 FREE 的重复观测不计入。
- **位姿**：`delta_local`（`egocentric_delta_pose_v2`），局部位移按 `TEMPORAL_DELTA_POSITION_SCALE=4.0` 缩放，转角由实际执行前后位姿计算。
- **时序 token**：21 维。features、behavior_reward、map_delta（原始新增自由格）、motion_delta、reward 全部启用；frontier_delta 固定关闭。
- **网络**：Graph Transformer（6 层 encoder / 1 层 decoder，128 维）+ 官方 Mamba 时序分支（seq 16，d_model 128，d_state 16，d_conv 4，expand 2）+ concat-MLP 融合；K=25，节点 padding 360。
- **Recovery**：Local Revisit Recovery V4.0，区域重复与点重复门槛均为 2 步；历史长度 10、图半径 8 m、最低计数 7、分数阈值 0.7、进度窗口 8、新增自由格阈值 1.0；frontier-driven escape（最小距离 12 m / 最小回环距离 8 m / 距离容差 1 m）。

## 关键超参数（节选自 run_config.json）

| 参数 | 值 |
|---|---|
| `CELL_SIZE` | 0.4 m |
| `NODE_RESOLUTION` | 4.0 m |
| `SENSOR_RANGE` | 16 m |
| `MAX_EPISODE_STEP` | 128 |
| `BATCH_SIZE` | 32 |
| `REPLAY_SIZE` | 3000 |
| `LR` | 1e-5 |
| `GAMMA` | 1.0 |
| `LEARNER_UPDATES_PER_TRAIN` | 8 |
| `USE_BF16` | true |
| `NUM_META_AGENT` | 5 |
| `TEMPORAL_SEQUENCE_LENGTH` | 16 |
| `TEMPORAL_TOKEN_DIM` | 128 |
| `ACTION_EMBEDDING_DIM` | 16 |

完整参数见同目录 `run_config.json`。

## 训练结果

| 指标 | 值 |
|---|---|
| 完成回合 | 10,000 |
| 成功回合 | 9,924 |
| 覆盖率 ≥99% 回合 | 9,959 |
| 平均覆盖率 | 99.90% |
| Recovery 触发次数 | 467 |
| Recovery 步数 | 4,037 |
| 总步数 | 467,172 |

训练曲线见 `training_dashboard.png`，逐回合记录见 `episodes.csv`，完整日志见 `train.log`。

## 完整测试结果

协议：1133 张 test 地图，单全局 seed 4777，greedy argmax，按清单顺序单进程执行，起点由 seeded map marker 选择，目标覆盖率 99%，最多 128 步，recovery 开启（2/2 门槛），逐图起点与历史参考一致。

| 指标 | 值 |
|---|---|
| 成功数 | 1133 / 1133 |
| 平均路径长度 | 251.643 m |
| 平均决策数 | 45.591 |
| 平均覆盖率 | 99.58% |
| Recovery 触发次数 | 16 |
| 评测耗时 | 4101 s |

结果文件位于本包 `evaluation/full_test_seed4777/` 与 `evaluation/visualizations/`。

## 同批最终 checkpoint 对照

来自 `evaluation/comparison/model_selection_20261026.md`：

| 版本 | 成功数 | 平均路径 m | 平均决策数 | 平均回合秒 | 触发次数 |
|---|---:|---:|---:|---:|---:|
| recovery_streak3 10k | 1133/1133 | 253.563 | 35.244 | 2.856 | 1 |
| **recovery_streak2 10k（本权重）** | **1133/1133** | **251.643** | **45.591** | **3.618** | **16** |
| recovery_streak3 12k | 1133/1133 | 253.233 | 41.457 | 3.311 | 15 |
| recovery_streak2 12k | 1133/1133 | 255.558 | 49.730 | 3.925 | 24 |

`baseline.json` 以平均路径最短为准则选择本权重。若以决策数/运行效率优先，3 步门槛 10k 是另一候选；两者为独立训练模型，差异不能单独归因于门槛。

## 使用限制

- 权重仅在与上述语义（Mamba、delta_local、21D token、`new_free_cell_count_v1`、recovery 2/2）一致的配置下有效；`driver.py::validate_checkpoint_mode` 会拒绝旧覆盖率反馈 checkpoint。
- 训练与评测均在 2D 栅格仿真中完成，**尚未在 Gazebo 中验证**。迁移到 Gazebo 时传感器噪声、地图分辨率、里程计误差、运动执行误差都可能改变表现，需要重新做闭环测试。
- 评测仅使用单一训练 seed（4777）和单一评测 seed；成功率 1133/1133 只代表该测试集与该协议，不代表任意环境的可靠性。
- 不应直接用于安全关键场景；实际部署需加入碰撞保护、定位失效处理和人工接管。

## 来源与完整性

- 权重来自 `HERA/experiments/recovery_streak2_seed4777_10k_20261024/model/checkpoint.pth`。
- SHA256 与 `baseline.json`、`run_config.json`、`evaluation` 元数据三处记录一致。
- 本包程序文件哈希与 `run_config.json` 的 `source_sha256` 全部一致，确保程序与权重为同一训练快照。
- 复制日期：2026-11-23。
