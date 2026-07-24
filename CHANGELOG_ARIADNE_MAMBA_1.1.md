# ARiADNE Mamba 1.1 更新日志

> 更新日期：2026-07-24  
> GitHub 仓库：`caibingdao/ARiADNE_MAMBA`  
> 开发分支：`ariadne_mamba_1.1`  
> 对比基线：官方 ARiADNE 项目及仓库 `main` 分支  
> 草稿 PR：<https://github.com/caibingdao/ARiADNE_MAMBA/pull/1>

## 1. 版本目标

本版本在保留 ARiADNE 原有图结构自主探索方法的基础上，引入 Mamba
时间序列分支，并将时间特征同时融合到离散 SAC 的策略网络和两个 Q
网络中。随后围绕单张 RTX 4060 Laptop GPU 完成了低风险资源配置和
rollout 历史缓存优化。

本次工作的原则是：

- 不重写原有地图、图构建、候选节点和路径规划模块；
- 保留原有非 Mamba 单步经验回放路径，便于回退和对照；
- Mamba 模式使用 episode-aware 时间序列回放，禁止历史跨越 episode；
- 优化项使用独立开关控制，并在不满足约束时尽早报错；
- 保留 Ubuntu 20.04，不升级系统 glibc，避免破坏 ROS 环境。

## 2. 当前系统架构

### 2.1 总体训练流程

```text
                         ┌──────────────────────────────┐
                         │        driver.py / Learner   │
                         │ Policy + Q1 + Q2 + Target Q  │
                         └──────────────┬───────────────┘
                                        │ 策略权重
                                        ▼
┌──────────────────┐    Ray     ┌──────────────────────────────┐
│ RLRunner 0        │◄──────────►│ RLRunner 1 ... RLRunner 3   │
│ Agent + Worker    │            │ Agent + Worker              │
└────────┬─────────┘            └─────────────┬────────────────┘
         │ 环境交互、图观测、动作、奖励          │
         └────────────────┬────────────────────┘
                          ▼
              ┌────────────────────────┐
              │ TrajectoryReplayBuffer │
              │ 按 episode 保存轨迹     │
              └───────────┬────────────┘
                          │ 对齐采样
                          ▼
             history_t + transition_t + history_t+1
                          │
                          ▼
                 离散 SAC 参数更新
```

`driver.py` 是训练主控：

1. 校验 Mamba、历史缓存和 Ray GPU 配置；
2. 初始化 Policy、Q1、Q2、Target Q1、Target Q2 和温度参数；
3. 根据 `LOAD_MODEL` 决定是否恢复模型及优化器状态；
4. 创建多个 Ray `RLRunner` 并分发最新策略权重；
5. 收集 episode 轨迹和探索指标；
6. 从时间序列回放缓冲区采样对齐的当前/下一历史；
7. 执行离散 SAC 更新；
8. 写入 TensorBoard，并定期保存完整 checkpoint。

### 2.2 图空间分支

原有 ARiADNE 图分支继续负责空间结构建模：

- `worker.py`：环境交互、地图更新和 episode 数据生成；
- `agent.py`：维护自主探索 Agent，并调用策略网络选择候选节点；
- `node_manager.py`：图节点、邻接关系和候选节点管理；
- `model.py`：图编码器、解码器和注意力层；
- `PolicyNet`：输出候选动作的对数概率；
- `QNet`：输出候选动作对应的 Q 值。

空间图分支仍然是当前状态表示的主体，Mamba 分支并不替代它，而是提供
最近若干决策步的时间上下文。

### 2.3 Mamba 时间分支

时间分支的数据流如下：

```text
历史观测序列
    │
    ▼
HistoryStateEncoder
    │  图状态、动作、奖励、位姿、覆盖变化、frontier 变化
    ▼
固定宽度时间 token，形状 [B, 8, 128]
    │
    ▼
MambaTemporalEncoder
    │  取每行最后一个有效时间步
    ▼
时间特征，形状 [B, 128]
    │
    ├──────────────┐
    ▼              ▼
图状态特征       GraphMambaFusion
 [B, 128]          concat + MLP
    └──────────────┬───────────────┘
                   ▼
            融合状态 [B, 128]
                   │
          ┌────────┴────────┐
          ▼                 ▼
       PolicyNet          QNet
```

核心模块：

- `models/history_encoder.py`
  - 将每个历史决策步转换为固定宽度 token；
  - 处理图状态、已选动作及辅助标量特征；
  - 使用 padding mask 区分左侧补零和有效历史。
- `models/mamba_encoder.py`
  - 封装官方 `mamba-ssm` 的 `Mamba` 模块；
  - 保持因果顺序；
  - 对每条序列取最后一个有效隐状态作为时间摘要。
- `models/fusion.py`
  - 将图空间特征和 Mamba 时间特征拼接；
  - 通过 MLP 映射回原图嵌入宽度。
- `model.py`
  - `TemporalFusionNetwork` 为 Policy/Q 网络提供共享的时间分支接口；
  - Policy、Q1 和 Q2 仍分别持有自己的网络参数，不共享同一计算图。

### 2.4 时间序列回放

`replay_buffer.py` 新增 episode-aware 轨迹结构：

- `TrajectoryStep`：保存一个 transition；
- `EpisodeTrajectory`：保存同一 episode 内的连续 transition；
- `TrajectoryReplayBuffer`：以 transition 数量为容量限制，按完整 episode
  管理轨迹；
- `HistorySequenceBatch`：保存带 padding mask 的历史序列；
- `AlignedTransitionBatch`：同时提供：
  - 当前观测 `observation_t`；
  - 动作、奖励和 done；
  - 下一观测 `observation_t+1`；
  - 当前历史 `history_t`；
  - 向后平移一步的下一历史 `history_t+1`。

采样历史时不会跨越 episode 边界。episode 开头不足 8 步的部分在左侧
补零，并用 `padding_mask=True` 标记；有效时间步为 `False`。

### 2.5 Rollout 历史缓存

`models/rollout_history_cache.py` 用于减少在线动作选择阶段的重复计算。

未启用缓存时，每个决策步都需要重新编码最近 8 步的全部历史。启用
`USE_ROLLOUT_HISTORY_CACHE=True` 后，Runner 会保存已经计算出的：

- 图状态嵌入；
- 候选动作特征；
- 已选动作特征；
- 对应的时间 token。

缓存使用长度受限的 `deque`，新决策只追加一个 token，再构造固定长度
Mamba 输入。以下情况会使缓存失效并重新开始：

- episode 改变；
- 策略权重版本改变；
- transition id 不连续；
- device 或 dtype 改变；
- Mamba 开关或序列/嵌入维度改变。

缓存内容在写入时执行 `detach()`，只用于 rollout 推理，不参与 learner
反向传播。因此该优化不会复用旧策略的梯度图。

## 3. 本版本完成的主要改变

### 3.1 Mamba + SAC Fusion

对应提交：`97f524d Complete Mamba SAC fusion integration`

- 为 Policy、Q1、Q2 和 Target Q 网络加入可配置的 Mamba 时间分支；
- 加入图特征与时间特征融合层；
- 新增 episode-aware 时间序列回放；
- SAC 更新同时使用 `history_t` 和 `history_t+1`；
- checkpoint 新增：
  - Mamba 模式标记；
  - Mamba 结构参数；
  - Target Q1/Q2 状态；
  - target 更新计数器；
- 加载 checkpoint 时严格检查模型模式、结构参数和 state dict key；
- 保留 `USE_MAMBA=False` 的原始单 transition 训练路径。

### 3.2 单 GPU Ray 资源配置

对应提交：`e33b584 Optimize single GPU Ray runner allocation`

- `NUM_META_AGENT` 改为显式可配置；
- 每个 Runner 使用固定 `RUNNER_GPU_FRACTION`；
- 启动前验证：

  ```text
  NUM_META_AGENT × RUNNER_GPU_FRACTION <= 1
  ```

- 当前配置为 `4 × 0.25 = 1 GPU`；
- Ray object store 固定为 3 GiB，避免使用过高的自动估计值；
- 多 GPU `DataParallel` 改为显式开关，单 GPU 默认关闭；
- 策略权重传给 Runner 前统一转到 CPU，减少不必要的 GPU 间复制。

说明：Ray 的 GPU fraction 是任务调度配额，不等价于显存硬隔离。Learner
和 Runner 仍会在同一物理 GPU 上执行，因此不能仅通过 fraction 解决所有
显存或吞吐问题。

### 3.3 Rollout 历史缓存

对应提交：`24b80cb Add reversible rollout history embedding cache`

- 新增严格校验的 rollout 历史缓存；
- 在线决策阶段复用已经计算的图嵌入和 token；
- 缓存与 episode、策略版本、transition 顺序和 device/dtype 绑定；
- 保留关闭缓存后的原始历史编码路径；
- learner 历史去重仍保持关闭：

  ```python
  USE_LEARNER_HISTORY_DEDUP = False
  ```

原因是当前 SAC 更新中的 Policy、Q1、Q2 是独立网络，不能在不同网络之间
错误复用同一组带梯度时间特征。

### 3.4 1.1 默认训练参数

对应提交：`58123e1 Configure Mamba 1.1 training defaults`

当前关键默认值：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `USE_MAMBA` | `True` | 启用时间分支 |
| `LOAD_MODEL` | `True` | 从 checkpoint 恢复 |
| `NUM_META_AGENT` | `4` | Ray rollout worker 数 |
| `RUNNER_GPU_FRACTION` | `0.25` | 每个 Runner 的 GPU 调度份额 |
| `RAY_OBJECT_STORE_MEMORY` | `3 GiB` | Ray object store 大小 |
| `BATCH_SIZE` | `8` | SAC batch size |
| `REPLAY_SIZE` | `3000` | 回放容量（transition） |
| `MAMBA_SEQ_LEN` | `8` | 时间历史长度 |
| `MAMBA_D_MODEL` | `128` | 时间 token 宽度 |
| `MAMBA_D_STATE` | `16` | Mamba state size |
| `MAMBA_D_CONV` | `4` | Mamba local convolution width |
| `MAMBA_EXPAND` | `2` | Mamba expansion |
| `USE_ROLLOUT_HISTORY_CACHE` | `True` | 启用 rollout 缓存 |
| `USE_LEARNER_HISTORY_DEDUP` | `False` | learner 去重暂不启用 |
| `ENABLE_MULTI_GPU_DATA_PARALLEL` | `False` | 单 GPU 不使用 DP |

## 4. Ubuntu 20.04 Conda 环境

### 4.1 为什么需要本机重编译 Mamba

原 Docker 镜像中的 Mamba CUDA 扩展要求 glibc 2.32，而 Ubuntu 20.04
提供 glibc 2.31。直接复制扩展会出现：

```text
GLIBC_2.32 not found
```

本次没有升级系统 glibc，而是在独立 Conda 环境内安装 CUDA 11.8 编译
工具，并针对 RTX 4060 的 `sm_89` 重新编译
`mamba-ssm 2.3.2.post1`。重新编译后的扩展最高只要求 glibc 2.14，
因此可在 Ubuntu 20.04 上运行，同时保留 ROS 所需的系统环境。

本机环境路径：

```text
/home/caibingdao/.conda/envs/ariadne-mamba
```

主要版本：

```text
Python       3.11.13
PyTorch      2.7.1+cu118
mamba-ssm    2.3.2.post1
Ray          2.10.0
NumPy        1.26.4
scikit-image 0.22.0
matplotlib   3.8.4
TensorBoard  2.19.0
```

本机兼容 wheel 保存在：

```text
/home/caibingdao/.cache/ariadne-conda-build/wheels/
mamba_ssm-2.3.2.post1-cp311-cp311-linux_x86_64.whl
```

该 Conda 环境和 wheel 是本机资产，没有提交到 GitHub。

### 4.2 Conda 启动训练

不要与 Docker 训练同时启动，以免两个 Learner/Runner 集群争用同一张
GPU 和主机内存。

```bash
cd "/home/caibingdao/桌面/ARiADNE_MAMBA"

export PATH="/home/caibingdao/.conda/envs/ariadne-mamba/bin:$PATH"
source "/home/caibingdao/.conda/envs/ariadne-mamba/etc/profile.d/conda.sh"
conda activate "/home/caibingdao/.conda/envs/ariadne-mamba"

export PYTHONPATH="$PWD"
export MPLBACKEND=Agg

python driver.py
```

退出环境：

```bash
conda deactivate
```

### 4.3 TensorBoard

在另一个终端中激活同一 Conda 环境，然后运行：

```bash
cd "/home/caibingdao/桌面/ARiADNE_MAMBA"
tensorboard --logdir train --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

## 5. 输出和 checkpoint

Mamba 模式使用以下目录：

```text
model/ariadne1_mamba/checkpoint.pth
train/ariadne1_mamba/events.out.tfevents.*
gifs/ariadne1_mamba/
```

当前本机 checkpoint 已验证：

```text
episode: 928
use_mamba: True
sequence_length: 8
d_model: 128
d_state: 16
d_conv: 4
expand: 2
fusion_hidden_dim: 128
action_embedding_dim: 16
```

checkpoint、TensorBoard event、GIF、PNG 和 Python cache 没有上传到
GitHub。

重要：分支默认 `LOAD_MODEL=True`。在新机器或全新 clone 上，如果没有
兼容的 `model/ariadne1_mamba/checkpoint.pth`，启动会失败。此时应二选一：

1. 将兼容 checkpoint 放到上述路径后继续训练；
2. 把 `LOAD_MODEL` 改为 `False`，从 episode 0 开始训练。

## 6. 已完成验证

本次完成以下验证：

- `tests.test_temporal_pipeline`：4 项测试通过；
- 时间序列采样不会跨越 episode；
- `history_t` 与 `history_t+1` 正确错开一步；
- padding graph pooling 不受补零节点影响；
- 历史 token 维度固定；
- `driver.py` 导入和 Ray 单 GPU 配置校验通过；
- 项目 `MambaTemporalEncoder` CUDA 前向通过；
- Mamba CUDA 反向传播通过，输入和参数梯度均为有限值；
- episode 928 checkpoint 模式及 Mamba 结构参数校验通过；
- Conda 环境可识别 NVIDIA GeForce RTX 4060 Laptop GPU。

说明：本阶段没有进行严格的吞吐率、episode wall time 或显存峰值基准
测试。因此 rollout 缓存的目标是减少重复编码，但本文不宣称具体百分比
性能提升。

## 7. 已知问题和后续建议

### 7.1 主机内存与 Ray OOM

此前训练曾在约 31 GiB 主机内存上触发 Ray OOM monitor。增加
`RAY_memory_usage_threshold` 只能推迟终止，不能解决实际内存压力。

后续优先级：

1. 保持 `NUM_META_AGENT=4`，不要直接恢复到 8 或 15；
2. 使用 `free -h`、Ray 日志和进程 RSS 定位真实内存占用；
3. 分析 Worker 是否重复持有地图、网络或轨迹对象；
4. 检查 object store 中是否存在未释放的大对象；
5. 在有测量数据后再调整并发数和 object store 大小。

### 7.2 Learner 历史去重

`USE_LEARNER_HISTORY_DEDUP` 当前必须为 `False`。若继续优化，应先设计
Policy、Q1、Q2 各自网络内部的安全复用边界，并验证梯度与原实现一致，
不能直接跨网络共享编码结果。

### 7.3 Conda 可复现性

当前 Conda 环境来自已验证 Docker 镜像并在本机修复，尚未生成可在其他
机器一键重建的 `environment.yml`。后续可增加：

- 精确 Python/Conda 依赖锁定文件；
- Ubuntu 20.04 本机构建 Mamba wheel 的脚本；
- CUDA compute capability 检查；
- 环境导入和最小 CUDA smoke test。

### 7.4 训练产物管理

仓库当前没有统一忽略所有本地训练产物。后续建议完善 `.gitignore`，
至少覆盖：

```text
__pycache__/
*.pyc
model/
train/
gifs/
```

checkpoint 如需长期保存，建议使用 GitHub Release、Git LFS 或独立模型
存储，不要直接放入普通 Git commit。

## 8. 提交历史

`ariadne_mamba_1.1` 相对 `origin/main` 包含以下核心提交：

```text
97f524d Complete Mamba SAC fusion integration
e33b584 Optimize single GPU Ray runner allocation
24b80cb Add reversible rollout history embedding cache
58123e1 Configure Mamba 1.1 training defaults
```

发布前曾发现 `driver.py` 被编辑器中的旧内容覆盖，从 744 行缩短为
318 行。该异常版本没有提交；发布分支中保留的是删除前、包含完整
Mamba/SAC/缓存/checkpoint 校验逻辑的 744 行版本。

## 9. 快速检查清单

以后重新开始训练前，建议依次检查：

- [ ] 当前 Git 分支是否为 `ariadne_mamba_1.1`；
- [ ] Docker 训练是否已经停止，避免与 Conda 同时运行；
- [ ] Conda 环境是否正确激活；
- [ ] `torch.cuda.is_available()` 是否为 `True`；
- [ ] `NUM_META_AGENT × RUNNER_GPU_FRACTION <= 1`；
- [ ] `LOAD_MODEL=True` 时 checkpoint 是否存在且模式一致；
- [ ] 磁盘是否有足够空间保存 checkpoint、GIF 和 TensorBoard 日志；
- [ ] 主机可用内存是否足以启动 4 个 Runner；
- [ ] TensorBoard 是否指向 `train/`；
- [ ] 修改 Mamba 结构参数后，不要继续加载结构不匹配的旧 checkpoint。
