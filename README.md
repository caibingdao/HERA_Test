# HERA_Gazebo_Test

HERA 的独立程序副本，附带训练完成的 HERA「完全体」模型权重，用于 Gazebo 集成测试与离线复现。

- 实现基线：工作区同级目录 `HERA/`（最终方法源码，2026-10-23 版本）。
- 复制日期：2026-11-23。
- 权重来源实验：`HERA/experiments/recovery_streak2_seed4777_10k_20261024`。
- 本目录不依赖原工作区路径，程序可直接以本目录为根运行；仅「1133 图完整评测脚本」仍依赖工作区外部的共享评测器（见第 6 节）。

## 1. HERA 完全体基线

`baseline.json` 定义的基线（选择准则 `mean_path_all`）：

| 项目 | 值 |
|---|---|
| 权重文件 | `weights/HERA_complete_seed4777_10k/checkpoint.pth` |
| 文件大小 | 71,573,717 字节 |
| SHA256 | `8d126a76433fc33b8a61ddc4e9288ea30e141e11257c759b12e99d5c6de568bb` |
| 训练回合 | 10,000 episodes，从零训练，seed 4777 |
| Recovery | 开启；区域重复/点重复连续门槛 2/2 |
| Token | 21 维；features / behavior_reward / map_delta / motion_delta / reward 全开，frontier_delta 关闭 |
| 反馈信号 | `new_free_cell_count_v1`（每步 UNKNOWN→FREE 的原始栅格计数，不做面积归一化） |
| 位姿编码 | `delta_local`，`temporal_delta_position_scale=4.0`（`egocentric_delta_pose_v2`） |
| Mamba | `mamba-ssm` 官方实现；d_model 128 / d_state 16 / d_conv 4 / expand 2 |

完整测试（1133 张 test 地图，seed 4777，greedy，99% 覆盖率目标，最多 128 步，recovery 开启）：

| 指标 | 结果 |
|---|---|
| 成功数 | 1133 / 1133 |
| 平均路径长度 | 251.643 m |
| 平均决策数 | 45.591 |
| 平均覆盖率 | 99.58% |
| Recovery 触发次数 | 16 |
| 评测总耗时 | 4101 s（约 3.62 s/图） |

> `baseline.json` 依据平均路径最短选择 2 步门槛 10k 权重。`evaluation/comparison/model_selection_20261026.md` 记录了四个最终 checkpoint 的对照；若更看重决策数与运行效率，3 步门槛 10k 是另一候选。两者权重与训练轨迹不同，不是同一 checkpoint 换测试门槛。

## 2. 目录结构

```
HERA_Gazebo_Test/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── baseline.json              # 官方基线声明（权重路径与 SHA256）
├── agent.py                   # 策略智能体：观测构建、动作选择、recovery 接入
├── driver.py                  # 训练主程序；含 checkpoint 构建/校验/加载函数
├── env.py                     # 2D 栅格探索环境（Gazebo 集成时需替换）
├── model.py                   # PolicyNet / QNet 网络定义
├── models/                    # Mamba 编码器、历史编码器、融合层、rollout 缓存
├── runner.py / worker.py      # Ray 采样器与单回合交互循环
├── replay_buffer.py           # 轨迹回放缓冲
├── local_revisit_recovery.py  # Loop Recovery V4.0（区域重复 + 点重复 + 逃逸）
├── recovery_artifacts.py      # 触发回合的工件记录
├── parameter.py               # 全部超参数与开关（环境变量可覆盖）
├── node_manager.py / quads.py / sensor.py / utils.py / temporal_pose.py
├── test_checkpoint.py         # 单 checkpoint 冒烟评测入口
├── training_budget.py / training_profiler.py
├── scripts/                   # 训练、评测、绘图、消融脚本（原样保留）
├── tools/                     # 辅助工具
├── configs/pose_ablation/     # 位姿消融配置
├── docs/                      # 原 HERA 文档（README_HERA、CHANGELOG、技术分析等）
├── weights/HERA_complete_seed4777_10k/
│   ├── checkpoint.pth         # HERA 完全体权重
│   ├── MODEL_CARD.md          # 权重卡：配置、指标、来源、使用限制
│   ├── run_config.json        # 训练超参数与源文件 SHA256
│   ├── launch.json / complete.json / train.log / episodes.csv
│   └── training_dashboard.png # 训练曲线
└── evaluation/
    ├── recovery_streak2_seed4777.json    # 完整测试启动配置
    ├── full_test_seed4777/               # 1133 图逐图结果与汇总
    ├── visualizations/                   # 评测图表（6 张）
    └── comparison/                       # 四模型选型对照
```

## 3. 环境依赖

```bash
pip install -r requirements.txt
```

要点：

- 需要 NVIDIA CUDA GPU；`mamba-ssm` 必须在其支持的 Linux + CUDA 环境编译安装（会依赖 `causal-conv1d`）。
- 训练使用 Ray：默认 5 个 rollout worker + 1 个 learner，双 GPU 布局见 `parameter.py` 与 `scripts/train_pose_original_parallel.py`。
- 推理同样要求 CUDA（`model.py` 的 Mamba 分支与 `driver.py` 的校验均假定 GPU 环境）。

## 4. 加载权重

在包根目录执行（模块为扁平导入，必须从本目录运行）：

```python
import torch
from driver import build_policy_network, load_model_state_dict, validate_checkpoint_mode

device = torch.device("cuda")
checkpoint = torch.load(
    "weights/HERA_complete_seed4777_10k/checkpoint.pth",
    map_location=device,
)
validate_checkpoint_mode(checkpoint)          # 校验 Mamba / pose / token / recovery 语义
policy = build_policy_network(device)
load_model_state_dict(policy, checkpoint["policy_model"], "policy")
policy.eval()
```

`validate_checkpoint_mode` 会拒绝以下不匹配：旧覆盖率反馈 checkpoint、位姿编码不一致、token 开关不一致、Mamba 配置不一致。权重对应的默认开关已在 `parameter.py` 中，无需额外设置：

- `USE_MAMBA=True`
- `ARIADNE_USE_RECOVERY` 默认 on
- `ARIADNE_RECOVERY_TRIGGER_STEPS` / `ARIADNE_POINT_REPEAT_TRIGGER_STEPS` 默认 2
- `ARIADNE_TEMPORAL_POSE_ENCODING` 默认 `delta_local`
- token 各开关默认全开（frontier_delta 固定关闭）

若需显式声明，可设置：

```bash
export ARIADNE_USE_RECOVERY=1
export ARIADNE_RECOVERY_TRIGGER_STEPS=2
export ARIADNE_POINT_REPEAT_TRIGGER_STEPS=2
export ARIADNE_TEMPORAL_POSE_ENCODING=delta_local
```

## 5. Gazebo 集成要点

本包内的 `Env` 是 2D 栅格仿真器，不是 Gazebo 接口。接入 Gazebo 时保留策略侧（`Agent`、`PolicyNet`、`LocalRevisitDetector`、历史 token 与 recovery），替换环境侧（`Env` 的栅格真值 / 传感器更新 / 运动执行）：

1. **地图与分辨率**：belief map 取值 `FREE=255`、`OCCUPIED=1`、`UNKNOWN=127`，栅格 `CELL_SIZE=0.4 m`，节点分辨率 `NODE_RESOLUTION=4.0 m`，传感器范围 16 m（`parameter.py`）。
2. **观测接口**：`Agent` 输出六元组
   `(node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask)`；
   `Agent.select_next_waypoint(observation, history_batch)` 返回世界坐标下的下一个 waypoint（米）与动作下标；贪心评测用 `logp.argmax` 替代多项式采样。
3. **历史 token**：每步需要按实际执行结果生成 21 维 token，包括局部位移/转角、reward、新增自由格原始计数 `new_free_cell_count`。`Agent.update_observation()` 与 `models/rollout_history_cache.py` 是生成与缓存入口；不要用未执行的动作或地图面积归一化替代。
4. **位姿语义**：`delta_local` + scale 4.0；heading 取最近一次实际位移方向，初始为 0。位姿必须由「实际执行前后」计算，不能使用动作提议值。
5. **Recovery**：保持区域重复/点重复 2/2 门槛、窗口 8、新增自由格阈值 1.0、frontier escape 参数不变；这些值已写入 checkpoint 的评测元数据。关闭或改动会导致与基线不一致。
6. **坐标系**：栅格坐标约定 `x=column, y=row`，世界单位米；`belief_origin_x/y` 由起始栅格计算。Gazebo 接入时统一到该约定。

## 6. 复现与评测

单 checkpoint 冒烟评测（需要地图）：

```bash
python test_checkpoint.py \
  --checkpoint weights/HERA_complete_seed4777_10k/checkpoint.pth \
  --episodes 10 --seed 4777
```

`Env` 查找地图的顺序：

1. `ARIADNE_MAP_SPLIT_FILE`（训练/评测清单）+ `ARIADNE_MAP_DIR` 指向地图目录；
2. 否则读取包内 `maps/` 目录（本包未打包地图图片）。

1133 图完整评测脚本 `scripts/run_full_test_eval.py`、`scripts/evaluate_recovery_streak.py` 等原样保留，但它们调用工作区外部的共享评测器 `ARiADNE_MAMBA_Test/mamba31_map_split`（约 7.3 GB，未打包）。没有该评测器时，请直接使用 `evaluation/full_test_seed4777/` 中已生成的逐图结果与汇总。

训练入口为 `scripts/train_pose_original_parallel.py`（基线实验实际使用）或 `driver.py`；训练同样需要外部地图清单与地图图片。默认双 GPU 布局：`CUDA_VISIBLE_DEVICES` 第 0 张做 rollout，第 1 张做 learner；轨迹记录、缓存与 profiler 开关见 `parameter.py`。

## 7. 数据与完整性

- `weights/HERA_complete_seed4777_10k/checkpoint.pth` 的 SHA256 与 `baseline.json`、`run_config.json`、评测元数据中的记录完全一致。
- 本包根目录与 `models/`、`scripts/` 中的 Python 源文件，其 SHA256 与 `weights/.../run_config.json` 的 `source_sha256` 全部一致，保证「程序 ↔ 权重」来自同一次训练快照。
- 完整测试结果包含 1133 张图的逐图记录；`publication_complete.json` 记录了校验通过的地图数。
- 训练结束记录：10,000 回合，成功 9,924，覆盖率 ≥99% 回合 9,959，平均覆盖率 99.90%，recovery 触发 467 次。

## 8. 未包含内容

- 训练 GIF、触发回合工件、逐步 recovery 诊断（原目录约 3 GB + 282 MB）。
- 其他消融实验的 checkpoint 与日志（`experiments/` 下除基线外的目录）。
- 共享评测器 `mamba31_map_split`、地图图片数据、`__pycache__`、原始 Git 历史（源目录不是 Git 仓库）。
- 本包不删除、不修改原 `HERA/` 目录。

## 9. 来源

| 内容 | 来源路径 |
|---|---|
| 全部程序 | `/data/home/lzy/cbd/code/ARiADNE_MAMBA_Test/HERA/` |
| 基线权重 | `HERA/experiments/recovery_streak2_seed4777_10k_20261024/model/checkpoint.pth` |
| 基线声明 | `HERA/baseline.json` |
| 完整测试 | `HERA/results/data/recovery_streak2/recovery_on/single_seed_4777/` |
| 模型选型对照 | `HERA/results/comparison/model_selection_20261026.md` |

上游 HERA 说明见 `docs/README_HERA.md`，ARiADNE 原项目说明见 `docs/README.md`。
