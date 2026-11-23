# ARiADNE-MAMBA 单卡 RTX 4090 性能优化报告

## 1. 执行结论

本轮已完成 profiling、低风险等价优化、BF16、checkpoint 对比、Q forward 复用、current/next union history、Ray worker 竞争测试和完整回归测试。

主要结论：

1. 历史长度从 8 增加到 16 后变慢的主因不是 Mamba，而是每个历史时刻都重新执行 6 层图 Encoder。
2. 在 B32 下，原始 L8 为约 2115 ms/update，L16 为约 3999 ms/update；history graph encode 从约 1026 ms 增至约 2063 ms，几乎正好翻倍。
3. Mamba 在 B32-L16 原始基线中仅约 23 ms/update，远小于 history graph encode 的约 2063 ms。
4. 最大的两项收益来自：
   - 跳过左侧 padding 历史图；
   - 将每个 history chunk 从 360 个 padding 节点裁剪到该 chunk 的真实最大节点数。
5. B32-L16 核心 benchmark 从 `3999.25 ± 0.13 ms` 降至 `405.60 ± 0.64 ms`，下降约 89.9%。
6. Q forward 复用虽可继续加速约 14%，但 B32-L16 learner 峰值 reserved 达到约 18.4 GiB；加上 Ray workers 后风险过高，因此最终默认关闭。
7. 4 个 GPU rollout workers 会使 learner update 变慢约 22%～24%；8 个 workers 通常使 learner 变慢约 77%～114%。4 workers 是更均衡的单卡配置。
8. 最终默认选择“保留 L16 的稳健配置”：B16-L16、4 workers、BF16、关闭 history checkpoint、跳 padding、动态裁节点、chunk32、union history 开启、Q reuse 关闭。

## 2. 硬件与软件环境

测试时间：2026-07-25，Asia/Shanghai。

### 硬件

- GPU：NVIDIA GeForce RTX 4090
- GPU 显存：24564 MiB
- NVIDIA 驱动：570.133.20
- CPU：Intel Core i9-14900K
- 可见逻辑 CPU：32
- 系统内存：62 GiB
- Swap：2 GiB

### 软件

- Python：3.11.15
- PyTorch：2.7.1+cu118
- PyTorch CUDA runtime：11.8
- cuDNN：9.1.0
- Ray：2.10.0
- mamba-ssm：2.3.2.post1
- GPU BF16 支持：是

本轮没有升级 PyTorch、CUDA、Mamba、Triton 或 Conda 依赖。

## 3. 分支与提交

工作分支：

```text
perf/learner-history-optimization
```

提交按独立阶段拆分：

```text
9d89e2e perf: add learner timing and profiling benchmark
a21f0ff perf: store graph masks as boolean tensors
dbd65ce perf: skip padded temporal graph encodes
894b966 perf: trim padded nodes inside history graph chunks
e6c1170 perf: remove per-row Python writes in Mamba pooling
778efa7 perf: clear learner gradients with set_to_none
e7cb589 perf: add optional BF16 autocast for learner forwards
471b364 perf: make history checkpointing configurable
c5df7d8 perf: optionally reuse current critic forwards
101d833 perf: collate aligned histories from one union window
4e720da test: make regression suite configuration invariant
c02c2c0 bench: allow low-overhead matrix timing without Kineto
2f87061 bench: measure learner competition with Ray rollout workers
fbd31b5 perf: select measured RTX 4090 training defaults
06f220c fix: keep configured BF16 off CPU regression paths
```

## 4. 修改的代码

### `parameter.py`

新增集中开关：

```python
ENABLE_TRAIN_PROFILING
PROFILE_WARMUP_UPDATES
PROFILE_ACTIVE_UPDATES
PROFILE_LOG_INTERVAL

USE_BF16
CHECK_TRAIN_FINITE
USE_HISTORY_CHECKPOINT
HISTORY_GRAPH_BATCH_SIZE
SKIP_PADDED_HISTORY_GRAPHS
TRIM_HISTORY_GRAPH_NODES
REUSE_CURRENT_Q_FORWARD
USE_NON_BLOCKING_TRANSFER
USE_UNION_HISTORY_COLLATE
```

所有高风险或实验性优化都可以独立开关。

### `training_profiler.py`

新增：

- CPU `time.perf_counter()` 分段计时；
- CUDA Event 分段计时；
- 仅 profiling 开启时执行 CUDA synchronize；
- mean、P50、P95；
- peak allocated/reserved memory；
- contextvar 深层计时，使 history encoder 和 Mamba 无需改变训练接口。

### `benchmarks/profile_sac_update.py`

新增可复现 benchmark：

- 固定随机种子；
- 固定 replay 数据和采样 seed；
- 可配置 batch、sequence、checkpoint、BF16、graph chunk；
- 可配置 skip padding、trim nodes、Q reuse、union history；
- 20 次 warm-up、50 或更多正式更新；
- CSV 输出；
- 可选 Kineto CPU/CUDA trace；
- `profile_memory=True`；
- `record_shapes=False`；
- `with_stack=False`；
- 支持 0/4/8 个真实 Ray `RLRunner` 并发；
- 记录 learner updates/s、episodes/hour、environment steps/hour。

### `agent.py` 和 `model.py`

- `node_padding_mask`、`edge_mask`、`edge_padding_mask` 改为 `torch.bool`；
- `current_index`、`current_edge` 保持 `torch.long`；
- MultiHeadAttention 使用逻辑 OR 合并 mask；
- Pointer log-softmax 在 BF16 autocast 下显式回到 FP32。

### `models/history_encoder.py`

新增：

- padding 历史位置过滤；
- 有效位置 `index_copy` 恢复；
- 全 padding batch 安全返回零；
- history chunk 动态节点裁剪；
- selected action node 裁剪边界检查；
- graph chunk size 配置；
- activation checkpoint 配置；
- history graph encode 和 token projection 分段计时。

### `models/mamba_encoder.py`

- 保留按有效长度分组；
- 删除逐 row Python 写回；
- 使用张量 `index_copy`；
- 保持 Mamba 因果顺序；
- BF16/FP32 输出 dtype 兼容；
- 增加 Mamba forward 分段计时。

### `driver.py`

- SAC 分阶段 profiling；
- BF16 forward autocast；
- policy、Q、target 输出回到 FP32 后计算 loss；
- FP32 entropy、Bellman target、MSE 和 log-alpha；
- 可选 finite 检查；
- `zero_grad(set_to_none=True)`；
- 可选 Q1/Q2 当前状态 forward 复用；
- union history 与 non-blocking transfer 接入。

### `replay_buffer.py`

- 新增 `UnionHistoryBatch`；
- 一次构造长度 L+1 的 union window；
- current history 使用 `[:, :-1]`；
- next history 使用 `[:, 1:]`；
- union 一次 `.to(device)` 后 current/next 共享 storage；
- 保持 episode 边界和 early-step 左 padding。

### `runner.py`

- 将 history 优化配置传入每个本地 PolicyNet；
- 显式 `use_mamba=False` 时不会错误继承 Mamba rollout cache。

## 5. 正确性与梯度语义

| 优化 | 梯度/语义结论 | 默认状态 |
|---|---|---|
| bool mask | FP32 输出和输入梯度严格一致 | 开启 |
| 跳过 padding 历史图 | 有效位置输出与梯度一致；padding token 仍为零 | 开启 |
| 动态节点裁剪 | 有效节点、pool、action feature 和 token 梯度一致 | 开启 |
| Mamba `index_copy` | 输出和梯度一致，因果顺序不变 | 开启 |
| `set_to_none=True` | 优化目标和梯度不变 | 开启 |
| union history | current/next 所有字段逐元素一致，episode 边界不变 | 开启 |
| checkpoint 开关 | 开/关输出与梯度一致；关闭增加激活显存 | 关闭 |
| Q forward 复用 | FP32 loss、grad norm、更新后权重一致 | 关闭 |
| BF16 | 非严格数值等价；loss 仍为 FP32，需训练曲线验证 | 开启 |

没有实施：

- Policy/Q/Target 共享 trainable encoder；
- replay 存储 stale embedding；
- 冻结历史 encoder；
- 更改 reward；
- 更改 SAC target；
- 更改 Mamba 因果顺序；
- 更改网络层数或隐藏维度；
- 稀疏第三方 kernel。

## 6. 测试结果

最终完整测试：

```text
44 passed, 11 subtests passed
```

覆盖新增测试：

- `tests/test_bool_masks.py`
- `tests/test_history_skip_padding.py`
- `tests/test_history_dynamic_trim.py`
- `tests/test_mamba_grouped_pooling.py`
- `tests/test_history_checkpoint_config.py`
- `tests/test_sac_q_forward_reuse.py`
- `tests/test_aligned_union_history.py`
- `tests/test_bf16_sac_smoke.py`

BF16 长程验收：

- 同一组模型和优化器连续执行 500 次 SAC update；
- 每次检查 policy loss、Q losses、alpha loss、entropy、value；
- 无 NaN/Inf；
- 无 OOM；
- 均值 222.41 ms/update；
- P95 224.36 ms；
- peak reserved 约 3310 MiB。

## 7. Benchmark 方法

固定条件：

- workers=0 的 learner-only 测试；
- 固定模型初始化；
- 固定随机 seed；
- 固定 replay 数据；
- 固定 sample seed；
- 节点 padding 上限 360；
- benchmark 中真实有效节点数设为 180；
- 20 次 warm-up；
- 50 次正式 update；
- 核心 baseline 和最终候选各重复 3 次。

注意：

- 有效节点 180 是固定、可重复 workload，不代表所有真实 episode 都恰好有 180 个节点；
- 动态裁剪实际收益取决于真实节点数分布；
- skip-padding 收益取决于 replay 中 early-step 样本比例；
- PyTorch peak memory 是当前进程 allocator 指标，不包含所有 Ray actor CUDA context；
- 第一组 B16-L8 生成了完整 Kineto trace，其他矩阵主要使用低干扰 CUDA Event。

## 8. 原始 baseline

原始模型逻辑提交：`9d89e2e`。

| Batch | Sequence | Mean ms/update | History graph ms | Mamba ms | H2D ms | Reserved MiB | Samples/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 2412.6 | 1068.4 | 261.1* | 92.7 | 2090 | 6.63 |
| 16 | 16 | 3111.7 | 1492.4 | 227.4 | 85.1 | 2342 | 5.14 |
| 32 | 8 | 2115.2 | 1026.2 | 14.0 | 55.4 | 3026 | 15.13 |
| 32 | 16 | 3999.25 ± 0.13 | 2062.8 | 23.2 | 约 106 | 3536 | 8.00 |
| 64 | 8 | 4191.3 | 2056.6 | 16.7 | 108.1 | 5134 | 15.27 |
| 64 | 16 | 7957.8 | 4113.2 | 28.9 | 212.7 | 6156 | 8.04 |

`*` B16-L8 是开启完整 Kineto memory trace 的代表组，Mamba 分段受 profiler 干扰较明显，不宜与低干扰组直接比较。

最可靠的 L8/L16 对照是 B32：

```text
total update:          2115 ms → 3999 ms
history graph encode:  1026 ms → 2063 ms
Mamba:                   14 ms →   23 ms
```

这说明 L16 变慢主要来自历史图 Encoder。

## 9. B32-L16 累积消融

| 阶段 | Mean ms | 相对上一阶段 | History graph ms | H2D ms | Reserved MiB |
|---|---:|---:|---:|---:|---:|
| 原始 baseline | 3999.2 | — | 2062.8 | 约 106 | 3536 |
| bool mask 等基础改动 | 3325.7 | -16.8% | 1602.2 | 13.3 | 2784 |
| + skip padded history | 1876.6 | -43.6% | 872.4 | 13.4 | 3110 |
| + trim history nodes | 690.5 | -63.2% | 271.7 | 13.4 | 2226 |
| + BF16，checkpoint on | 690.7 | 基本不变 | 256.5 | 14.8 | 2502 |
| + checkpoint off | 492.8 | -28.6% | 201.1 | 14.3 | 6408 |
| + Q forward reuse | 423.6 | -14.0% | 158.4 | 13.9 | 18400 |
| + union history | 428.3 | 无总耗时收益 | 161.7 | 7.8 | 18332 |

结论：

- skip padding 和 trim nodes 是最重要的改动；
- union history 明显减少 H2D，但模型计算主导时总耗时改善很小；
- Q reuse 的显存代价过高；
- checkpoint 应在当前最终配置中关闭；
- BF16 在 checkpoint 开启时没有收益，但在 checkpoint 关闭时有收益。

## 10. 最终候选，3 次重复

配置共同项：

```text
checkpoint=False
BF16=True
graph_batch_size=32
skip_padding=True
trim_nodes=True
reuse_q_forward=False
union_history=True
workers=0
```

| 配置 | Mean ± std ms | History graph ms | Reserved MiB | Samples/s |
|---|---:|---:|---:|---:|
| B16-L16 | 222.37 ± 0.67 | 73.4 | 3310 | 71.95 |
| B32-L16 | 405.60 ± 0.64 | 139.3 | 6134 | 78.90 |
| B32-L12 | 366.66 ± 0.42 | 119.5 | 5564 | 87.28 |

B32-L16 相对原始同配置：

```text
3999.25 ms → 405.60 ms
下降约 89.9%
约 9.86× update throughput
```

达到并显著超过“至少下降 25%，理想下降 35%～50%”的目标。

## 11. Ray worker 竞争

| Learner 配置 | Workers | ms/update | 相对 workers=0 | Episodes/hour | Env steps/hour |
|---|---:|---:|---:|---:|---:|
| B32-L12 | 0 | 366.7 | — | — | — |
| B32-L12 | 4 | 452.7 | +23.5% | 2081 | 262531 |
| B32-L12 | 8 | 647.5 | +76.6% | 4281 | 545749 |
| B32-L16 | 0 | 405.6 | — | — | — |
| B32-L16 | 4 | 495.9 | +22.3% | 1974 | 248119 |
| B32-L16 | 8 | 722.2 | +78.1% | 4318 | 552746 |
| B16-L16 | 0 | 222.4 | — | — | — |
| B16-L16 | 4 | 274.6 | +23.5% | 2038 | 249466 |
| B16-L16 | 8 | 476.8 | +114.4% | 4803 | 608976 |

结论：

- 8 workers 接近翻倍 rollout 吞吐，但严重抢占 learner；
- 4 workers 使 learner 慢约两成，整体更均衡；
- 如果 replay 数据长期不足，可以临时用 8 workers；
- 如果 learner 是瓶颈，应使用 4 workers。

## 12. 最终默认参数

最终写入 `parameter.py` 的稳健 L16 配置：

```python
BATCH_SIZE = 16
TEMPORAL_SEQUENCE_LENGTH = 16

NUM_META_AGENT = 4
RUNNER_GPU_FRACTION = 0.25

USE_BF16 = True
USE_HISTORY_CHECKPOINT = False
HISTORY_GRAPH_BATCH_SIZE = 32
SKIP_PADDED_HISTORY_GRAPHS = True
TRIM_HISTORY_GRAPH_NODES = True
REUSE_CURRENT_Q_FORWARD = False
USE_NON_BLOCKING_TRANSFER = False
USE_UNION_HISTORY_COLLATE = True
```

选择理由：

- 保留研究所需的 L16；
- 4 workers 比 8 workers 更平衡；
- B16 learner-only reserved 约 3.31 GiB；
- 加上 4 个实测约 508 MiB/worker，仍有较大显存余量；
- 500 次连续 BF16 update 稳定；
- 不启用高显存 Q reuse。

如果更重视 learner 样本吞吐，并可接受更高显存：

```python
BATCH_SIZE = 32
TEMPORAL_SEQUENCE_LENGTH = 16
NUM_META_AGENT = 4
```

如果允许 L12：

```python
BATCH_SIZE = 32
TEMPORAL_SEQUENCE_LENGTH = 12
NUM_META_AGENT = 4
```

这是当前 samples/s 最高的候选。

## 13. 回退方法

恢复原始计算路径：

```python
USE_BF16 = False
USE_HISTORY_CHECKPOINT = True
HISTORY_GRAPH_BATCH_SIZE = 16
SKIP_PADDED_HISTORY_GRAPHS = False
TRIM_HISTORY_GRAPH_NODES = False
REUSE_CURRENT_Q_FORWARD = False
USE_NON_BLOCKING_TRANSFER = False
USE_UNION_HISTORY_COLLATE = False
```

bool mask、Mamba `index_copy` 和 `set_to_none=True` 已通过严格回归，不提供运行时回退开关；如需完全恢复，可回退对应独立 Git commit。

## 14. 已知风险

1. BF16 已通过 500 update smoke，但还需要比较相同环境步数下的长期 reward、success rate 和 exploration curve。
2. 动态裁节点每个 chunk 使用一次 `.item()` 获取最大有效节点数，会产生 CUDA 同步；当前 benchmark 中收益远大于该开销。
3. 实际地图的节点数若接近 360，动态裁剪收益会下降。
4. checkpoint 关闭后的真实训练整卡显存应继续观察；learner-only allocator 指标不包含所有 Ray actor context。
5. Q forward reuse 虽保持梯度语义，但 B32 峰值显存过高，默认关闭。
6. union history 当前减少 H2D，但 CPU collate 仍可继续优化；尚未引入 pinned-memory replay。
7. `USE_NON_BLOCKING_TRANSFER=False` 保持关闭，因为 replay CPU tensor 当前没有系统性 pin memory，单独开启不会可靠提升。
8. 修改 sequence length 会触发现有 checkpoint Mamba 配置严格校验；L8/L12 checkpoint 不能直接当作 L16 checkpoint 加载。

## 15. 未实施的后续架构方案

按执行约束，本轮未直接实现：

- trainable encoder 共享；
- stale replay embedding；
- EMA/frozen history encoder；
- adjacency-list 稀疏 attention；
- 中央 GPU inference actor；
- 轻量统计 token 替代完整图历史。

后续最值得研究的是稀疏图 attention。当前动态裁剪已经证明 N² 图 attention 是核心热点；若真实节点数仍较大，基于邻接表的稀疏 attention 可能继续带来明显收益，但必须另建实验分支并做训练效果 ablation。

## 16. 产物

CSV：

- `benchmarks/results/baseline_phase1.csv`
- `benchmarks/results/optimized_ablation.csv`
- `benchmarks/results/final_candidates.csv`
- `benchmarks/results/confirmed_candidates.csv`
- `benchmarks/results/worker_competition.csv`
- `benchmarks/results/bf16_stability_500.csv`

Kineto trace：

```text
benchmarks/results/baseline_phase1.trace.json.gz
```

压缩后约 487 MiB，未加入 Git。

说明：

- 核心 B32-L16 baseline 和三个最终候选均完成 3 次确认；
- 探索性完整 batch/sequence 矩阵和 worker 竞争组各完成 1 次正式测量；
- 未把单次探索矩阵伪装成 3 次重复结果。
