# ARiADNE、ARiADNE–Mamba 与 Heuristic Recovery 版本技术分析

## 摘要

本文档基于当前工作区的实际代码，对三个版本进行逐层技术对比：

1. **原始 ARiADNE**：以当前拓扑图为输入、由图注意力网络直接产生候选节点策略；
2. **ARiADNE–Mamba**：保留原空间图注意力分支，并增加用于建模探索历史的 Mamba 时间分支；
3. **ARiADNE–Mamba–Heuristic（最终版本）**：在双分支学习策略之外增加事件触发式启发式逃逸控制器，并保留针对 Mamba 训练路径的计算优化。

最终模型的核心变化可以概括为：

> 使用图注意力编码当前空间结构，使用 Mamba 编码最近探索历史，在 Actor 和 Critic 内分别融合空间—时间表征；正常情况下仍由离散 SAC 策略决策，仅当检测到“持续重复访问且探索进展停滞”时，启发式控制器才临时接管动作，选择面向 frontier 的逃逸目标并执行 committed Dijkstra recovery。

需要特别澄清：实现并不是“两套完整策略网络分别输出动作再投票”，而是**一个空间图注意力分支与一个时间 Mamba 分支并行提取表征，融合后由同一个策略头输出动作**。

---

## 1. 对比对象与代码依据

| 版本 | 当前目录 | 作用 |
|---|---|---|
| ARiADNE | `ARiADNE_ORIGINAL_PARALLEL/` | 原始空间图注意力基线 |
| ARiADNE–Mamba 1.1 | Git 历史提交 `b41a6c2` 及之前 | Mamba 初始实现，用于计算优化前基线 |
| 优化后的 ARiADNE–Mamba | `ARiADNE_MAMBA_Test/` 与最终仓库的性能优化提交 | 无 heuristic recovery 的时空融合策略 |
| ARiADNE–Mamba–Heuristic | `ARiADNE_MAMBA_3.1/` | 最终模型：优化后的 Mamba 策略 + Loop Recovery V4 |

主要实现依据包括：

- 原始与最终网络：`model.py`；
- 时间 token：`models/history_encoder.py`；
- Mamba：`models/mamba_encoder.py`；
- 空间—时间融合：`models/fusion.py`；
- episode-aware replay：`replay_buffer.py`；
- 在线历史缓存：`models/rollout_history_cache.py`；
- 启发式控制器：`local_revisit_recovery.py`；
- 策略/恢复动作切换：`worker.py`；
- 优化实测：`PERFORMANCE_OPTIMIZATION_REPORT.md`。

---

## 2. 总体架构演进

### 2.1 原始 ARiADNE

```text
当前 belief map
      ↓
informative graph 构建
节点特征 + 邻接关系 + 当前节点 + 候选邻居
      ↓
6 层图注意力 Encoder
      ↓
1 层当前节点 Decoder
      ↓
当前状态特征
      ├──────────────→ Pointer policy → 候选动作概率
      └──────────────→ Q head         → 候选动作 Q 值
```

原模型只显式编码当前时刻的图状态。历史影响只能通过当前地图、节点 utility 等环境状态间接留下，策略网络本身没有专门的时间序列记忆模块。

### 2.2 ARiADNE–Mamba

```text
                   当前图观测
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
当前空间图注意力分支             最近 L 步轨迹
6-layer Graph Encoder                │
1-layer Decoder               HistoryStateEncoder
         │                    每步生成 128-d token
         │                           │
         │                    1-layer causal Mamba
         │                           │
         ▼                           ▼
   graph feature g_t          temporal feature h_t
          [128]                     [128]
         └─────────────┬─────────────┘
                       ▼
              concat + MLP + LayerNorm
                       ▼
              fused state z_t [128]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Pointer policy        Q-value head
```

Mamba 不替换 ARiADNE 的图注意力，而是补充其缺失的时间建模能力。空间分支回答“当前图上哪里值得去”，时间分支回答“机器人最近如何到达这里、采取过哪些动作、探索进展是否持续”。

### 2.3 最终 ARiADNE–Mamba–Heuristic

```text
                         当前图与历史轨迹
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   ARiADNE–Mamba SAC policy                 Loop Recovery V4 detector
   输出 learned policy action              检测重复访问 + 低覆盖增益
            │                                       │
            │                               trigger? ── no ──┐
            │                                       │         │
            │                                      yes        │
            │                                       ▼         │
            │                         frontier grouping/ranking│
            │                                       ▼         │
            │                         locked escape target     │
            │                                       ▼         │
            │                         committed Dijkstra action│
            │                                       │         │
            └────────────────── action selector ◄───┴─────────┘
                                │
                                ▼
                          executed action
                                │
                 replay 中记录真实执行动作及
                    is_recovery_action 标志
```

Heuristic 模块位于策略输出与环境执行之间，是一个**稀疏触发的安全/恢复层**，不是第二个学习策略，也不改变正常状态下的 SAC 动作分布。

---

## 3. 原始 ARiADNE 的空间策略网络

### 3.1 输入表示

每个 graph node 的输入维数为 4，原始网络接收：

- 节点特征 `node_inputs`；
- 节点 padding mask；
- 由图邻接关系形成的 attention mask；
- 当前机器人所在节点索引；
- 当前节点的候选邻居索引；
- 候选邻居 padding mask。

候选动作不是连续控制量，而是从当前 graph node 的邻接候选中选择下一 waypoint。

### 3.2 图编码器

节点特征先由线性层投影到 128 维，再进入 6 层、8-head 的图注意力 Encoder。邻接 mask 限制注意力传播范围，因此网络在 informative graph 上进行结构化信息聚合，而不是对所有地图栅格直接决策。

### 3.3 当前状态解码

网络从编码后的全部节点中 gather 当前节点特征，并通过 1 层 Decoder 让当前节点再次查询全图记忆。随后拼接：

```text
原始当前节点 embedding || Decoder 增强后的当前节点 embedding
```

再用线性层压回 128 维，得到当前空间状态特征 `g_t`。

### 3.4 Actor 与 Critic

- **PolicyNet**：以 `g_t` 为 query，以候选邻居节点特征为 key，通过 single-head pointer attention 输出各候选动作的 `log_softmax` 概率；
- **QNet**：将 `g_t` 复制到每个候选动作，与相应邻居特征拼接，再经线性层输出逐动作 Q 值；
- 离散 SAC 使用一个 Actor、两个独立 Critic 和两个 Target Critic。

原始模型的限制不是空间表达不足，而是没有显式状态去区分“几何位置相近但历史行为完全不同”的情形，例如正常回访与低进展循环。

---

## 4. 改进一：图注意力 + Mamba 的并行时空双分支

### 4.1 时间输入不是简单的位置序列

对于最近 `L=16` 个决策步，每一步的 temporal token 同时包含：

1. **历史图全局特征**：该历史观测通过当前网络自己的 6 层图 Encoder，再对有效节点做 masked mean pooling，得到 128 维 graph embedding；
2. **历史动作特征**：从该步的候选邻居中取出实际被选择的节点 embedding，再投影到 16 维；
3. **机器人位姿**：`x, y, heading` 三个量；
4. **即时反馈**：reward；
5. **探索进展**：coverage 增量；
6. **边界变化**：frontier 数量增量。

拼接后的维数为：

```text
128（图）+ 16（动作）+ 3（位姿）+ 1（reward）
+ 1（Δcoverage）+ 1（Δfrontier）= 150
```

该 150 维向量先经过 LayerNorm，再通过：

```text
Linear(150, 128) → SiLU → Linear(128, 128) → LayerNorm
```

形成每个历史时刻的 128 维 token。位姿、reward 和 frontier 增量在进入投影前使用固定尺度归一化，以减少不同物理量的量纲差异。

### 4.2 Mamba 时间编码器

最终实现使用官方 `mamba_ssm.Mamba`，配置为：

| 参数 | 值 |
|---|---:|
| sequence length | 16 |
| `d_model` | 128 |
| `d_state` | 16 |
| `d_conv` | 4 |
| `expand` | 2 |
| Mamba blocks | 1 |

序列采用左 padding，padding 不参与有效历史建模。Mamba 按因果顺序处理 token，并取最后一个有效时刻的 hidden state 作为时间摘要 `h_t ∈ R^128`。因此 `h_t` 仅依赖当前决策之前已经发生的轨迹，不使用未来信息。

### 4.3 空间—时间融合

当前图分支产生 `g_t ∈ R^128`，Mamba 分支产生 `h_t ∈ R^128`。二者通过：

```text
z_t = LayerNorm(Linear(SiLU(Linear([g_t || h_t]))))
```

融合层的具体形状为：

```text
Linear(256, 128) → SiLU → Linear(128, 128) → LayerNorm
```

输出仍为 128 维，因此原 ARiADNE 的 pointer policy head 和 Q-value head 可以保持不变。这个设计最大限度地保留了原模型的空间决策接口，只替换状态表征 `g_t` 为时空表征 `z_t`。

### 4.4 融合同时进入 Actor 和 Critic

时间分支不是只加在 Actor 上。Policy、Q1、Q2、Target Q1 和 Target Q2 都具有对应的 temporal branch 和 fusion layer，并各自持有参数。其原因是：

- Actor 需要根据历史选择动作；
- Critic 也必须在同一个时空状态定义下评价动作；
- Bellman target 使用下一状态历史 `history_(t+1)`；
- 不在不同网络之间复用带梯度 embedding，避免破坏独立 Critic 的梯度语义。

### 4.5 Episode-aware 时间回放

原始单步 replay 被扩展为完整 episode trajectory。每个 transition 除原始 observation/action/reward/next observation/done 外，还保存位姿、coverage 增量、frontier 增量、step index 和 `is_recovery_action`。

采样时构造：

- 当前历史 `history_t`；
- 当前 transition；
- 下一历史 `history_(t+1)`。

历史严格限制在同一 episode 内；episode 起始不足 16 步时左侧 padding，不允许把上一个 episode 的轨迹拼入当前序列。该约束对避免伪时间相关性非常重要。

### 4.6 这一改变解决了什么

与原始 ARiADNE 相比，双分支网络能够显式利用：

- 最近访问模式与动作往返模式；
- coverage/frontier 的短期变化趋势；
- 相似当前图状态背后的不同到达历史；
- 动作、奖励、位姿与图结构之间的时序依赖。

因此，提升来源不是简单地用 Mamba 替换 attention，而是将原来的**当前空间状态决策**扩展为**当前空间结构 + 有限历史动态**的联合决策。

---

## 5. 改进二：事件触发式启发式逃逸机制

### 5.1 模块定位

Loop Recovery V4 仅在检测到局部循环时临时接管动作。正常状态的执行源是 SAC；恢复状态的执行源是 heuristic controller：

```text
if recovery_triggered:
    execute next waypoint on locked Dijkstra path
else:
    execute ARiADNE–Mamba policy action
```

因此最终系统是“学习策略为主、启发式恢复为辅”的 hybrid policy，而不是永久混合两种动作概率。

### 5.2 双证据循环检测

检测器使用最近 10 个历史 graph node（不包含当前节点）和最近 8 步 coverage gain。触发必须满足低探索进展，并由以下任一重复访问分支连续成立 3 步。

#### A. 区域重复访问

- 当前节点周围图半径：`2 × NODE_RESOLUTION = 8 m`；
- 最近 10 个历史节点中，至少 7 个落在该 graph neighborhood；
- 等价于 regional revisit score 至少为 0.7。

#### B. 点重复访问

将 10 步历史映射到当前 graph node 并统计频次，要求：

- 单个最高频节点至少出现 3 次；
- 最高频的 3 个节点合计至少出现 8 次。

这一分支用于捕获两点往返、三点往返以及跨相邻小区域的循环，弥补单纯“当前位置邻域比例”可能产生的漏检。

#### C. 低进展门控

两个重复访问分支都必须同时满足：

```text
最近 8 步平均 Δcoverage < 0.001
```

因此正常的回访、经过枢纽节点或回程路径不会仅因重复位置而触发恢复。

#### D. 独立连续计数器

区域证据和点证据分别维护连续 3 步计数器，二者最后做 OR。它们不能在不同时间步交替“凑满”同一个 streak，从而降低提前误触发概率。

### 5.3 Frontier-driven escape target

触发后不是简单选一个最远节点。目标选择流程为：

1. 从 belief map 提取 frontier cells；
2. 按物理 8-connectivity 将 frontier 聚成 group；
3. 对每个 group，将其质心映射到当前图上的可达 target node；
4. 由 Dijkstra 得到当前位置到 target 的导航距离 `D`；
5. 计算 target 相对历史循环区域的距离 `L`；
6. 过滤 `D < 12 m` 或 `L < 8 m` 的候选；
7. 对通过过滤的候选计算：

```text
U = I / (D + 1 m)
```

其中 `I` 是 frontier group 的 cell 数，代表潜在信息量。候选排序优先级为 utility、frontier 数量、离循环区域距离，并在相同条件下偏向更短导航距离。

这个目标函数在“逃得足够远”与“仍朝向高信息区域探索”之间做平衡，避免 heuristic recovery 退化为无目的远离。

### 5.4 多级 fallback

- 如果存在满足距离约束的 frontier candidate，选择 utility 最大者；
- 如果存在 frontier candidate 但全部被距离过滤，选择全体中 utility 最大者，记为 `frontier_filter_fallback`；
- 如果没有可用 frontier 且探索尚未完成，在可达图节点中选择离历史循环区域最远者，记为 `graph_fallback`；
- 可通过配置关闭 frontier escape，退化为 V2.1 的邻域外最近低访问节点，用于消融实验。

### 5.5 Locked target 与 committed recovery

一旦选定目标，恢复过程会锁定同一个空间目标，并沿 Dijkstra 路径逐步执行：

- 不允许策略在恢复期间每步重新选目标；
- 动态图变化导致下一跳失效时，重新把同一空间目标映射到当前图并规划；
- 只有目标不可达、连续路径失败、达到最大 32 个 recovery steps、episode 结束或到达目标时才退出；
- 到达目标的容差为 `0.5 × NODE_RESOLUTION`；
- 退出后清空位置和 coverage 窗口，必须收集新的完整证据才能再次触发。

这种 committed execution 避免了 learned policy 的短期偏好再次把机器人拉回循环区域。

### 5.6 与 SAC/Mamba 的交互

恢复动作被映射回当前 observation 中真实存在的候选 action index。写入 replay 的是**实际执行的 recovery action**，而不是策略本来可能采样的反事实动作；`is_recovery_action` 仅作为 metadata 保留。

当前 SAC loss 不根据该标志过滤或重新加权样本。因此 heuristic 会改变 learner 看到的行为数据分布，但没有修改 reward、Bellman target、actor loss 或 critic loss。从系统实现看，它属于 action-level intervention，而不是新的 loss function。

---

## 6. 改进三：Mamba 引入后的计算优化

### 6.1 初始 Mamba 1.1 的主要瓶颈

初版中，历史序列每一个时刻都需要重新执行完整的 6 层图 Encoder。对于 batch size `B`、历史长度 `L`，Policy、Q1、Q2 和 target 网络都会在 SAC 更新中处理大量历史图。因此主要开销近似随历史图数量和图 attention 的二次节点复杂度增长。

受控 profiling 证明瓶颈不是单层 Mamba：在 B32-L16 初始基线中：

| 分段 | 时间 |
|---|---:|
| 总 SAC update | 3999.25 ms |
| history graph encode | 2062.8 ms |
| Mamba forward | 23.2 ms |

历史长度从 8 增至 16 时，history graph encode 从约 1026 ms 增至 2063 ms，而 Mamba 仅从约 14 ms 增至 23 ms。优化方向因此集中在历史图生成，而不是削减 Mamba 状态空间。

### 6.2 跳过 padding 历史图

episode 前若干步的 16-step history 含有大量左 padding。初版仍将这些 padding position 展平后送入图 Encoder。

优化后先从 `[B, L]` padding mask 提取有效位置，只编码真实历史图；编码结果再用 tensor `index_copy` 放回原序列位置，全 padding batch 直接返回零。这样既保持 token 与梯度语义，也消除了无意义的 attention 计算。

在 B32-L16 累积消融中，该项使 update 从 3325.7 ms 降至 1876.6 ms，较上一阶段下降 43.6%。收益大小会随 early-episode 样本比例变化。

### 6.3 动态裁剪历史图节点

图 observation 统一 padding 到 360 nodes，而很多真实图远小于该上限。全尺寸 self-attention 会对 padding 节点产生近似 `O(360²)` 的无效矩阵计算。

优化后，每个 history graph chunk 统计该 chunk 的最大有效节点数 `N_max`，并同步裁剪：

- node inputs；
- node padding mask；
- 两个维度上的 edge/attention mask。

再执行图编码。candidate index 仍经过边界检查，masked mean pooling 和实际动作节点特征保持一致。

该项使 B32-L16 从 1876.6 ms 进一步降至 690.5 ms，较上一阶段下降 63.2%，是单项收益最大的优化之一。其收益取决于真实节点数与 360 上限之间的差距。

### 6.4 Boolean mask 与稳定的混合精度路径

- node/edge masks 从数值 tensor 改为 `torch.bool`；
- mask 合并由加法改为逻辑 OR；
- masked fill 使用当前 dtype 的最小有限值；
- learner forward 启用 BF16 autocast；
- pointer 的 `log_softmax`、policy probabilities、entropy、Bellman target、MSE、temperature loss 显式回到 FP32。

BF16 不是严格数值等价优化，但已完成连续 500 次 SAC update 的有限值检查，无 NaN、Inf 或 OOM。它属于训练系统优化，不改变模型拓扑结构。

### 6.5 关闭 activation checkpointing

初版对 history graph chunk 使用 activation checkpointing，以较低显存换取反向重算。实测在当前显存条件下关闭 checkpoint 更合适：B32-L16 从 690.7 ms 降至 492.8 ms，代价是 reserved memory 增加。

这不是删减网络，而是保留前向激活，避免 backward 重新运行图 Encoder。

### 6.6 Union history collate

初版分别构造 `history_t` 和 `history_(t+1)`，其中绝大部分窗口重叠。优化后一次构造长度 `L+1` 的 union window：

```text
current history = union[:, :-1]
next history    = union[:, 1:]
```

二者共享底层 storage，并只进行一次 device transfer。该优化将受控实验的 H2D 从约 13.9 ms 降至 7.8 ms；由于总耗时由模型计算主导，整体 update 时间变化不大，但减少了重复 collate 和传输。

### 6.7 Mamba pooling 向量化

不同有效历史长度仍按组运行 Mamba，以保持左 padding 与因果顺序；组内结果由 Python 逐行写回改为 tensor `index_copy`，减少 host-side loop 和小粒度赋值。

### 6.8 Rollout 历史缓存

在线 rollout 中，策略每一步原本会重新编码最近 L 个历史图。缓存机制保存已经生成的：

- graph embedding；
- candidate/action feature；
- temporal token。

新决策只编码并追加最新一步，再构造定长 Mamba 输入。缓存绑定 episode id、策略权重版本、transition 连续性、device 和 dtype；权重更新或 episode 切换时失效。缓存内容 `detach`，只优化推理，不用于 learner 反向传播，因此不会把旧策略 embedding 错用于训练梯度。

### 6.9 优化器与并行资源细节

- `optimizer.zero_grad(set_to_none=True)` 避免无必要的梯度清零写操作；
- Policy/Q 权重分发给 Ray worker 前转到 CPU，减少不必要的 GPU 间复制；
- Ray worker GPU fraction 只用于调度，不被误认为显存硬限制；
- profiling 表明 8 workers 会使 learner update 变慢约 77%–114%，4 workers 是单卡实验中更平衡的配置；
- Q current-forward reuse 虽额外提速约 14%，但 B32-L16 reserved memory 达约 18.4 GiB，最终默认关闭；
- non-blocking transfer 在 replay tensors 未系统性 pin memory 时无可靠收益，最终关闭。

### 6.10 总体计算收益

在 RTX 4090、B32-L16、learner-only、固定 replay workload 的受控 benchmark 中：

```text
初始 Mamba 1.1 计算路径：3999.25 ± 0.13 ms/update
优化后核心配置：         405.60 ± 0.64 ms/update
```

即 update latency 降低约 89.9%，吞吐约提高 9.86 倍。该数字是**训练 update 的受控 microbenchmark**，不等同于端到端 wall-clock 训练提速；真实训练还受到 rollout worker、环境模拟、有效节点数分布和 replay 中 early-step 比例影响。

---

## 7. 三个模型的逐项对比

| 对比项 | 原始 ARiADNE | ARiADNE–Mamba | 最终 ARiADNE–Mamba–Heuristic |
|---|---|---|---|
| 当前空间编码 | 6-layer、8-head 图 attention | 保留 | 保留 |
| 当前节点解码 | 1-layer attention decoder | 保留 | 保留 |
| 时间建模 | 无显式时间分支 | 1-layer causal Mamba | 同左 |
| 历史长度 | 单时刻 | 16 | 16 |
| 历史 token | 无 | 图、动作、位姿、reward、Δcoverage、Δfrontier | 同左 |
| 空间—时间融合 | 无 | concat + 2-layer MLP + LayerNorm | 同左 |
| Actor 使用历史 | 否 | 是 | 是 |
| 双 Critic 使用历史 | 否 | 是 | 是 |
| 回放结构 | 单 transition | episode-aware aligned histories | 同左，额外记录 recovery flag |
| 正常动作源 | SAC policy | SAC + Mamba policy | SAC + Mamba policy |
| 循环检测 | 无 | 无 | regional revisit + point repeat + low progress |
| 逃逸目标 | 无 | 无 | frontier utility + graph fallback |
| 恢复执行 | 无 | 无 | locked target + committed Dijkstra |
| 是否修改 SAC loss | — | 保持离散 SAC | 保持离散 SAC |
| history padding 跳过 | 不适用 | 优化前无、优化后有 | 有 |
| 动态节点裁剪 | 不适用 | 优化前无、优化后有 | 有 |
| BF16 learner | 否 | 优化后有 | 有 |
| rollout token cache | 不适用 | 有 | 有 |

---

## 8. 最终模型的实际决策流程

一次决策可以准确描述为：

1. 从 belief map 更新 informative graph、node utility、frontier 和候选邻居；
2. 图注意力 Encoder/Decoder 产生当前空间状态 `g_t`；
3. 将最近 16 步的历史图、动作、位姿和探索反馈编码为 temporal tokens；
4. Mamba 产生因果时间摘要 `h_t`；
5. 融合得到 `z_t = Fusion(g_t, h_t)`；
6. PolicyNet 基于 `z_t` 与候选邻居特征输出动作分布；
7. 同时，Loop Recovery V4 使用位置历史和 coverage 增量检查是否持续停滞；
8. 未触发时执行 SAC 策略动作；触发时选择 frontier escape target，并执行锁定 Dijkstra 路径的下一跳；
9. 环境返回 reward 与新地图状态；
10. replay 保存实际执行动作及对应的 `is_recovery_action`；
11. learner 使用 `history_t` 和 `history_(t+1)`执行原离散 SAC 更新。

---

## 9. 技术边界与容易误读之处

1. 不应写成“用 Mamba 替换 Transformer/attention”。实际实现保留并依赖原 6 层图 attention；Mamba 是并行时间分支。
2. 不应写成“两套策略网络投票”。两条分支先融合，再由同一个 policy head 输出动作。
3. 不应把 heuristic 称为 learned fallback policy。它是确定性的 detector、target selector 与 path controller。
4. 不应声称 recovery transitions 被过滤。当前它们仍进入 replay 并参与原 SAC 更新。
5. 9.86 倍不是端到端训练速度提升，而是固定 workload 下 learner update microbenchmark 的吞吐提升。
6. 不应把 BF16 描述为严格数学等价；它是经稳定性验证的系统优化。
7. 不应声称 Q-forward reuse 已用于最终配置。该方案因显存开销默认关闭。
8. 当前 pure-policy 与 recovery 的 `overall_summary.json` 是两个模型合并后的总体统计，不能直接作为二者差值；探索效果增益需要从逐模型、逐地图、同 seed 的 episode 表进行 paired analysis。

---

## 10. 用于验证各模块作用的技术测试矩阵

为隔离各模块带来的结构和运行行为变化，可采用以下测试组合：

| 实验 | Graph attention | Mamba | Recovery | 计算优化 | 目的 |
|---|---:|---:|---:|---:|---|
| A | ✓ | × | × | — | 原始 ARiADNE 基线 |
| B | ✓ | ✓ | × | × | Mamba 1.1 原始计算路径 |
| C | ✓ | ✓ | × | ✓ | 纯学习策略及等价计算优化 |
| D | ✓ | ✓ | regional only | ✓ | 区域循环检测贡献 |
| E | ✓ | ✓ | point-repeat only | ✓ | 点重复分支贡献 |
| F | ✓ | ✓ | nearest-node escape | ✓ | 旧 escape target 基线 |
| G | ✓ | ✓ | frontier recovery V4 | ✓ | 最终完整模型 |

探索效果应按相同 map split 和相同 evaluation seeds 配对计算：success rate、final coverage、path length、decision steps、recovery trigger rate、target reach rate、20/30-step retrigger rate，以及每次 recovery 后的 coverage gain。计算效率应独立统计 update latency、history graph latency、samples/s、显存和 rollout throughput。

---

## 11. 最终系统技术定义

最终模型是一个以 ARiADNE 图注意力为空间骨干、以 Mamba 为历史动态编码器、以融合后的时空特征驱动离散 SAC Actor–Critic，并由低频事件触发的 frontier-driven committed recovery 提供停滞逃逸能力的混合式自主探索系统。
