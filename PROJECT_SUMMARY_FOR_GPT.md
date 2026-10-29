# ARiADNE-MAMBA 项目与性能问题总结

> 用途：将本文档直接发给其他 GPT/技术人员，协助分析单张 RTX 4090 上的训练速度、显存占用和代码优化方案。
>
> 项目实际运行目录：`/users/cbd/code/ARiADNE_MAMBA_from_laptop`
>
> 状态采样时间：2026-07-25（CST）

## 1. 项目目标

本项目基于 ICRA 2023 的 ARiADNE 自主探索强化学习项目，在原有注意力图网络和离散 SAC 训练框架上增加了 Mamba 时间历史分支。

原始策略主要根据当前的拓扑图观测选择下一个探索节点。当前改造版希望进一步利用最近若干步的历史信息，包括：

- 历史图状态；
- 已选择的动作节点；
- 机器人位置与朝向；
- reward；
- coverage 增量；
- frontier 数量增量。

这些历史信息被编码成时间 token，输入 Mamba，再与当前图状态融合，以改善策略对探索过程时序信息的利用。

## 2. 硬件与当前运行状态

### 2.1 硬件

- GPU：NVIDIA GeForce RTX 4090
- 显存总量：24564 MiB
- NVIDIA 驱动：570.133.20
- 系统内存：62 GiB
- Swap：2 GiB

### 2.2 当前配置的实测状态

当前 `parameter.py` 已调整为：

```python
BATCH_SIZE = 64
NUM_META_AGENT = 8
RUNNER_GPU_FRACTION = 0.125
TEMPORAL_SEQUENCE_LENGTH = 12
NODE_PADDING_SIZE = 360
USE_GPU = True
USE_GPU_GLOBAL = True
```

2026-07-25 20:44:11 的运行快照：

- GPU 显存：12273 / 24564 MiB
- 剩余显存：11817 MiB
- GPU 利用率：100%
- GPU 温度：65°C
- GPU 功耗：约 287 W / 450 W
- 主训练 Python 进程：约 7630 MiB
- 8 个 `ray::RLRunner`：每个约 508 MiB，合计约 4064 MiB
- 系统内存：已用约 30 GiB，可用约 28 GiB
- Swap：已使用约 737 MiB

### 2.3 此前较重配置的实测状态

此前实际运行过：

```python
BATCH_SIZE = 64
NUM_META_AGENT = 16
RUNNER_GPU_FRACTION = 0.0625
TEMPORAL_SEQUENCE_LENGTH = 16
NODE_PADDING_SIZE = 360
```

当时的运行快照：

- GPU 显存：17390 / 24564 MiB
- 剩余显存：约 6700 MiB
- GPU 利用率：98%
- 主训练进程：约 8652 MiB
- 16 个 rollout worker：每个约 508 MiB，合计约 8128 MiB

该配置能够启动和运行，但显存余量较小，训练速度明显下降。

注意：`RUNNER_GPU_FRACTION` 是 Ray 的 GPU 调度资源配额，不是显存硬限制。设置为 `0.0625` 不表示 worker 最多只能使用 4090 显存的 6.25%。实测每个 worker 仍占用约 508 MiB。

## 3. 主要训练参数

当前参数如下：

```python
# 环境与回放
MAX_EPISODE_STEP = 128
REPLAY_SIZE = 3000
MINIMUM_BUFFER_SIZE = 2000

# SAC
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 1

# Ray rollout
NUM_META_AGENT = 8
RUNNER_GPU_FRACTION = 0.125
RAY_OBJECT_STORE_MEMORY = 3 * 1024 ** 3

# 时间历史
COLLECT_TRAJECTORY = True
TEMPORAL_SEQUENCE_LENGTH = 12
TEMPORAL_TOKEN_DIM = 128
ACTION_EMBEDDING_DIM = 16
USE_ROLLOUT_HISTORY_CACHE = True
USE_LEARNER_HISTORY_DEDUP = False

# 图网络
NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 25
NODE_PADDING_SIZE = 360

# Mamba
MAMBA_D_MODEL = 128
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
FUSION_HIDDEN_DIM = 128

# GPU
USE_GPU = True
USE_GPU_GLOBAL = True
ENABLE_MULTI_GPU_DATA_PARALLEL = False
```

每当满足训练条件后，`driver.py` 会连续执行 8 次 `sac_update`。

训练目前没有使用 AMP、FP16 或 BF16；网络、激活和优化器均按普通 FP32 路径运行。

## 4. 代码模块结构

### 4.1 训练和并行采样

- `driver.py`
  - 创建全局 PolicyNet、两个 QNet 和两个 Target QNet；
  - 创建 Adam 优化器；
  - 启动 Ray rollout workers；
  - 管理 replay buffer；
  - 采样 batch；
  - 执行离散 SAC 更新；
  - 同步 policy 权重并保存 checkpoint。

- `runner.py`
  - 定义 Ray `RLRunner`；
  - 每个 runner 持有一份本地 PolicyNet；
  - Mamba 模式下强制 rollout 使用 CUDA；
  - 每轮接收主进程同步的 policy 权重；
  - 每个 runner 申请一部分 Ray GPU 调度资源。

- `worker.py`
  - 与环境交互并完成一个 episode；
  - 生成当前和下一状态观测；
  - 保存动作、reward、done、姿态、coverage/frontier 变化；
  - 使用 rollout history cache，避免 rollout 阶段每一步从头重建全部历史；
  - 将完整 episode trajectory 返回给主训练进程。

### 4.2 环境和图构建

- `env.py`
  - 自主探索环境、地图和 reward 逻辑。

- `agent.py`
  - 维护机器人规划状态；
  - 调用 PolicyNet 选择下一个 waypoint。

- `node_manager.py`
  - 维护 informative graph；
  - 更新节点、utility 和邻接关系。

- `sensor.py`
  - 模拟激光雷达感知。

- `utils.py`
  - 地图、frontier、坐标转换等辅助逻辑。

- `quads.py`
  - 四叉树空间索引。

### 4.3 模型

- `model.py`
  - 自定义多头注意力；
  - 6 层图 Encoder；
  - 1 层 Decoder；
  - PolicyNet；
  - QNet；
  - 当前图状态与历史 Mamba 特征的融合入口。

- `models/history_encoder.py`
  - 将每个历史 transition 转换为时间 token；
  - 每个历史步首先通过当前网络自己的完整图 Encoder；
  - 对图节点做 masked mean pooling；
  - 提取被选动作节点的特征；
  - 拼接姿态、reward、coverage/frontier 增量；
  - 输出 128 维 token；
  - 历史图计算按 16 个图为一组分块；
  - 训练状态下使用 activation checkpointing 节省显存，但反向传播时会重新计算。

- `models/mamba_encoder.py`
  - 使用官方 `mamba_ssm.Mamba`；
  - 当前只有 1 个 Mamba block；
  - 输入形状为 `[batch, history_length, 128]`；
  - 使用最后一个有效位置作为历史摘要。

- `models/fusion.py`
  - 将当前图状态向量与 Mamba 历史向量拼接；
  - 经过 MLP 和 LayerNorm，输出 128 维融合特征。

### 4.4 Replay 与历史缓存

- `replay_buffer.py`
  - 以完整 episode 为单位保存 trajectory；
  - 按 transition 采样；
  - 为当前状态和下一状态分别构造左侧 padding 的历史序列；
  - batch 搬运到训练 GPU。

- `models/rollout_history_cache.py`
  - 只优化 rollout 推理路径；
  - 缓存最近的图 embedding、动作特征和 token；
  - policy 权重更新时使缓存失效；
  - 不能直接消除 learner 训练阶段的重复图编码。

## 5. 关键数据流

```text
Ray Worker 与环境交互
    ↓
当前图观测 + 动作 + reward + 位姿 + coverage/frontier 变化
    ↓
完整 EpisodeTrajectory 返回主进程
    ↓
TrajectoryReplayBuffer
    ↓
采样 transition，并构造 current_history / next_history
    ↓
每一个历史步重新经过对应 Policy/Q 网络的 6 层图 Encoder
    ↓
历史 transition → 128 维 token
    ↓
Mamba 处理时间序列
    ↓
Mamba 历史摘要与当前图状态融合
    ↓
Policy/Q 输出
    ↓
离散 SAC 更新
```

## 6. 当前最主要的性能问题

用户观察到：将历史长度从 8 增加到 16 后，训练速度下降很多，下降幅度在体感上明显大于预期。

初步判断：真正的主要开销很可能不是单层 Mamba 本身，而是历史 token 的生成方式。

### 6.1 历史中的每一步都会重新执行完整图编码

历史输入并不是已经预计算好的轻量 token。`HistoryStateEncoder` 会将：

```text
[batch, sequence, nodes, features]
```

展平为：

```text
[batch × sequence, nodes, features]
```

然后让每个历史图状态经过对应网络的 6 层图注意力 Encoder。

在 `BATCH_SIZE=64` 时：

- 序列 8：一次历史编码涉及 512 个图状态；
- 序列 12：一次历史编码涉及 768 个图状态；
- 序列 16：一次历史编码涉及 1024 个图状态。

每个图最多 padding 到 360 个节点，而图注意力包含与节点数平方相关的 attention tensor。因此，序列长度翻倍实际上会让大量完整图 Encoder 计算翻倍，并不只是让 Mamba 多处理 8 个 token。

### 6.2 一个 SAC update 会重复调用多个完整网络

当前一次 `sac_update` 大致执行以下模型调用：

1. 当前状态 Q1，无梯度；
2. 当前状态 Q2，无梯度；
3. 当前状态 Policy，有梯度；
4. 下一状态 Policy，无梯度；
5. 下一状态 Target Q1，无梯度；
6. 下一状态 Target Q2，无梯度；
7. 当前状态 Q1，有梯度；
8. 当前状态 Q2，有梯度。

即一次 SAC update 约有 8 次完整模型前向，其中训练路径还包含 3 次反向传播。每个模型拥有自己的图 Encoder、历史 token Encoder 和 Mamba 分支。

外层训练循环又固定执行：

```python
for j in range(8):
    sac_update(...)
```

因此一次进入训练阶段约产生 64 次完整模型调用。历史长度增加带来的开销被这一结构显著放大。

### 6.3 Activation checkpointing 用计算换显存

历史图编码使用 PyTorch activation checkpointing。它有助于控制 `batch × sequence × graph` 带来的显存峰值，但反向传播时会重新执行部分前向计算，因此可能进一步降低训练速度。

### 6.4 当前没有混合精度

代码没有使用：

- `torch.autocast`；
- BF16/FP16；
- `GradScaler`。

RTX 4090 的 Tensor Core 未被充分利用。全部采用 FP32 会增加激活显存、显存带宽压力和计算时间。

### 6.5 多 worker 占用显存，但不一定是 learner 变慢的唯一原因

每个 GPU rollout worker 实测约占 508 MiB：

- 8 workers：约 4.1 GiB；
- 16 workers：约 8.1 GiB。

减少 worker 能明显释放显存，但如果 GPU 已经被 learner 的图编码完全占满，GPU rollout 与 learner 同时竞争算力，也可能降低单次训练更新速度。

## 7. 已实现的优化

- rollout 历史缓存已经启用：

```python
USE_ROLLOUT_HISTORY_CACHE = True
```

这可以避免 rollout 每一步重复重算此前的历史 token。

- learner 历史去重目前关闭：

```python
USE_LEARNER_HISTORY_DEDUP = False
```

代码中明确指出，当前 SAC 更新里 current/next history 会由参数不同的 Policy、Q1、Q2 和 Target Q 网络编码，不能简单地共享同一个带梯度的编码结果。

- 历史图编码使用了分块：

```python
graph_batch_size = 16
```

- 历史图训练路径使用 activation checkpointing，以降低显存。

- 多 GPU DataParallel 默认关闭，当前机器只有一张 4090。

## 8. 当前拟解决的问题

希望分析并回答以下问题：

1. 为什么历史长度从 8 增至 16 后，训练速度下降如此明显？
2. 瓶颈主要位于：
   - 历史图 Encoder；
   - Mamba；
   - replay batch 拼装和 CPU→GPU 传输；
   - activation checkpointing 重算；
   - 还是 learner 与多个 GPU rollout worker 的算力竞争？
3. 如何在不破坏 SAC 梯度正确性和不同网络参数独立性的前提下，减少历史图的重复编码？
4. 是否可以安全地：
   - 为 replay buffer 存储预计算图特征；
   - 使用滞后 encoder、冻结 encoder 或 target encoder；
   - 合并某些 Q/Policy 的历史编码；
   - 对 current/next history 的重叠部分复用计算；
   - 将图历史压缩为轻量状态后再存储？
5. RTX 4090 上是否适合启用 BF16 autocast？Mamba CUDA kernel、图注意力和离散 SAC loss 是否存在数值稳定性注意事项？
6. activation checkpointing 在当前 batch/sequence 规模下是否得不偿失？应如何通过 profiling 决定开启或关闭？
7. `BATCH_SIZE`、`TEMPORAL_SEQUENCE_LENGTH`、`NUM_META_AGENT` 和每轮 `sac_update` 次数应如何联合配置？
8. 如果希望保留长度 16，最有价值的代码级优化顺序是什么？
9. 应使用哪些 PyTorch profiler/Nsight 指标验证瓶颈，而不是仅根据 GPU 利用率判断？
10. 能否设计一套严格的 A/B benchmark，分别测试：
    - sequence 8 / 12 / 16；
    - batch 16 / 32 / 64；
    - checkpoint 开/关；
    - BF16 开/关；
    - 0 / 4 / 8 个 GPU rollout workers；
    并记录每个 SAC update 耗时、历史编码耗时、峰值显存、samples/s 和 episodes/hour？

## 9. 当前建议的临时配置

如果优先保证单张 4090 稳定运行，可先使用：

```python
BATCH_SIZE = 32
NUM_META_AGENT = 8
RUNNER_GPU_FRACTION = 0.125
TEMPORAL_SEQUENCE_LENGTH = 8  # 或 12 做折中
NODE_PADDING_SIZE = 360
```

如果研究目标明确要求长度 16，建议先降低 batch：

```python
BATCH_SIZE = 16  # 或 32
TEMPORAL_SEQUENCE_LENGTH = 16
NUM_META_AGENT = 4  # profiling 时可先减少 GPU 算力竞争
RUNNER_GPU_FRACTION = 0.25
```

上述只是保守运行参数，不代表最终最优吞吐配置。最终应由分模块 profiler 和固定 workload benchmark 决定。

## 10. 额外注意事项

- 修改 `parameter.py` 不会改变已经运行的 Python/Ray 进程，必须重启训练。
- checkpoint 保存了 Mamba 配置，包括序列长度；改变序列长度后，当前严格配置校验可能拒绝加载旧 checkpoint。
- `LOAD_MODEL=False` 表示当前运行不续训旧 checkpoint。
- 项目目录中存在另一个 `/users/cbd/code/ARiADNE_MAMBA`，但本文分析针对实际运行的 `ARiADNE_MAMBA_from_laptop`。
- 原版 README 所说的约 8 GB 显存需求不适用于当前加入 Mamba、历史图重编码、较大 batch 和多个 GPU rollout worker 后的配置。

## 11. 希望获得的最终输出

请基于以上结构给出：

1. 按可能性排序的性能瓶颈诊断；
2. 对梯度正确性有无影响的优化分类；
3. 最小侵入式优化方案；
4. 中等改造方案；
5. 激进架构改造方案；
6. 一套可执行的 profiling 与 A/B benchmark 计划；
7. RTX 4090 上推荐的最终参数组合；
8. 对“长度 16 是否值得保留”的判断标准。
