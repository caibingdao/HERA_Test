# Mamba 探索模型：Experiments 准备文档

数据截止：2026-09-15。本文是依据本地代码、训练配置、checkpoint 审计及逐 episode 评测记录形成的论文准备材料。表格数值重新从 episodes.csv 和 coverage_curve.csv 计算；未开展新的训练或评测。T01–T08 为本文整理编号，不是代码版本号。

## 1. 实验范围与证据层级

目前共收录 54 份完整汇总评测记录：36 份新 token 评测、18 份历史模型及协议评测。完整记录指有 evaluation_metadata.json、episodes.csv 与 coverage_curve.csv；隐藏的并行 worker 中间目录不重复计数。单种子和五种子分别汇总，greedy 与 stochastic 分开解读。不同目录可能复用 checkpoint 或相同评测，54 份记录不能表述为 54 次独立训练。

正文按三层组织：首先列出新 token 的评测结果；其次使用同一 OFF 训练 checkpoint、相同地图和起点做 recovery OFF/ON 配对；最后保留历史基线与协议实验作为完整档案。早期模型训练预算、token 版本和计时口径未全部追溯，不能直接据其排名声称公平的架构优势。

## 2. 实验场景与任务

### 2.1 二维未知环境探索

实验在二维占据栅格地图上开展。机器人从初始位置出发，通过有限范围感知更新 belief map，逐步构建可通行节点图并选择相邻节点作为下一航点，目标是在有限决策预算内覆盖未知自由空间。地图区分自由、占据和未知三种状态，分别编码为 255、1、127。训练环境将输入灰度地图按 2 倍 block-reduce/min 预处理，再按代码阈值转为占据地图。

这是地图驱动的二维探索实验；目前证据不足以将其描述为真实机器人部署、带动力学约束的连续控制实验或具有传感器噪声的物理仿真。地图来源、许可、场景类别以及物理面积分布尚未完成溯源，不宜自行补写为某个公开建筑数据集，也不能宣称具有已标注的走廊、办公室等类别分层。

### 2.2 数据划分与初始化

地图清单共 5,663 个文件，训练集 4,530，测试集 1,133，划分种子为 42。清单按语义占据结构分组，同组划分考虑起点标记、平移、整数尺度、旋转及反射不变性，以降低同一底图变体跨集合的风险。清单记载 5,658 个语义组，4 个重复/变体组，共涉及 9 个文件；训练和测试文件不重叠。未找到显式类别标签，因此没有实际执行类别分层。

新 token 模型均在该清单的 train 部分训练，测试使用 test 部分。训练按 episode 索引轮转访问配置中的地图列表。单种子协议使用全局 seed 4777；五种子协议分别使用 100、101、102、103、104，每张测试图在每个种子下评测一次。Greedy 固定动作选择规则，测试 seed 仍可影响初始化，不能将其理解为 stochastic 动作采样。恢复机制配对已检查实际起点坐标及起点栅格，而非只检查 seed 名字。

### 2.3 环境与图表示参数

| 项目 | 设置 |
| --- | --- |
| 占据栅格分辨率 | 0.4 m |
| 图节点间距 / frontier 栅格 | 4.0 m / 0.8 m |
| 感知半径 / utility 范围 | 16.0 m / 12.8 m |
| 局部地图更新跨度 | 80 m |
| 最小 utility 阈值 | 2 |
| 节点输入 | 相对 x、y，utility，访问标记，共 4D |
| 图节点 padding / 邻居槽位上限 | 360 / 25；不等于每个状态都有这些真实节点或边 |
| 每 episode 最大决策数 | 128 |
| 评测成功覆盖率阈值 | 99% |
| 评测路径长度硬上限 | 未设置 |

## 3. 评价指标与统计方式

覆盖率 C 为已观测自由栅格数除以完整地图自由栅格数。主要成功指标为最终 C ≥ 0.99。训练中的终止/完成奖励依据 utility 清空条件；其语义与评测覆盖率成功不完全相同，不能直接将训练 TensorBoard success 当作本表测试成功率。

成功率 SR = 成功 episode 数 / 总 episode 数。失败数报告整数，避免 99.9% 以上的差异被四舍五入成 100%。异常数统计 error 字段非空的 episode，与未达到覆盖目标的普通失败区分。

路径长度 L 为执行轨迹的累计行走距离，单位 m；决策步数为航点选择/执行次数，不是训练迭代数，也不是底层控制频率。正文报告所有 episode 的平均路径长度和平均步数，包含失败轨迹。该指标采用无条件平均口径，未排除失败episode。

最终覆盖率报告所有 episode 的平均值。覆盖阈值达成率使用 coverage_curve.csv 中每个 episode 的最大覆盖率（含初始状态），分别统计达到 60%、70%、80%、90%、95% 的比例。附录保留原图1对应的全部核心指标，包括决策步数和路径长度的样本方差（分母 N−1）、失败率及异常数。

单种子 N=1,133；五种子 N=5,665。五种子主表采用所有 episode 合并统计，per_seed_results.csv 另给出逐 seed 结果。因为每个 seed 的地图数相同，合并均值等于 seed 均值的平均；合并样本方差并不是五个 seed 均值的方差。新 token 训练 seed 均为 0；本文没有多训练 seed 的方差或显著性证据。重复测试同一地图也不应被当作完全独立的新地图样本。

Recovery 的直接收益使用配对指标：OFF 失败而 ON 成功记为挽救；OFF 成功而 ON 失败记为新增失败；同时报告双方都成功 episode 上的路径差。触发 episode 数表示至少触发一次 recovery 的轨迹数，不等于触发事件总数，也不能直接等同于所有真实回环数量。

当前适配器的推理计时覆盖 CUDA 同步的 select_next_waypoint，包含 recovery 检测/规划，不含 Env.step 和后续图更新。历史记录未全部核实一致的计时实现与独占硬件条件，因此本报告不将其混合列入推理速度排行榜。

## 4. 模型、训练与 recovery 配置

### 4.1 网络与训练方式

当前增强图节点特征用于候选航点表示，历史 token 经 Mamba 编码后与当前图状态融合，融合表示作为查询参与邻居动作选择。图特征的历史输入置零不代表去掉当前图注意力支路。保留目标特征的版本将历史实际选中目标节点的增强表示投影为 16D；该输入用于表示历史行动上下文。

| 配置 | 当前新 token 实验设置 |
| --- | --- |
| 优化算法 | 离散动作 SAC；policy 与双 Q 网络 |
| 图网络 | 128D；图 encoder 6 层、decoder 1 层，8 heads |
| Mamba | 序列长度16，d_model128，d_state16，d_conv4，expand2 |
| 融合 | 当前图状态128D与历史128D拼接，经MLP与LayerNorm得到128D |
| Replay / warmup / batch | 3000 / 2000 / 32 |
| 学习率 / 折扣因子 | 1e−5 / 1 |
| 更新频率 | 每次 rollout 完成后 learner 更新8次 |
| 采样 | 5个 Ray rollout workers |
| 数值精度 | BF16 混合精度 |
| 训练 seed | 0 |
| GPU组织 | rollout与learner分工；并非单模型两卡数据并行；各run具体绑定见配置附件 |

训练配置以各实验保存的 run_config.json 为准，而非 parameter.py 当前默认值。各版本从各自 token 掩码配置训练；T03/T04 等存在同版本续训，不能表述为只在完整 token checkpoint 测试时遮挡输入。两卡主要分担采样和学习，单个模型测试不要求同时占用两张显卡。当前资料未形成覆盖全部实验的统一硬件/驱动/依赖归档，正式论文需补齐硬件型号和版本记录；不能把今日机器环境当成所有历史实验环境。

### 4.2 原始 reward 与历史 token

训练奖励由行走代价、旧 frontier 被观测消除的收益及任务完成奖励组成。按当前实现可写为 r_t = −5·d_t/80 + |F_(t−1) \ F_t| / floor(16·3.14/0.8) + 20·I(任务完成)。其中 frontier 收益是旧 frontier 集合消失部分的数量，不是 frontier 总数的净变化，也不是直接的 Δcoverage。没有已完成的“检测到回环额外扣分”奖励设计实验。

所有新 token 采用 egocentric_delta_pose_v2 / delta_local：运动为局部坐标增量 Δx、Δy、Δθ，不是绝对 x、y、θ。原始拼接宽度始终为 150：池化图特征128D、运动3D、历史所选目标特征16D、reward1D、Δcoverage1D、Δfrontier1D。位置增量除以4，角增量除以π，reward除以20；Δfrontier采用带符号log1p缩放，Δcoverage保持原始增量。掩码在对应变换后、token LayerNorm前置零，维度不删除；下表“保留维度”指保留输入槽位，不是每步实际非零元素数。所有版本仍投影为128D送入Mamba。

| 编号 | 历史 token 保留内容 | 保留维度 | 已训练 recovery |
| --- | --- | --- | --- |
| T01 | full token | 150 | ON、OFF |
| T02 | no motion | 147 | OFF |
| T03 | motion+reward+Δcoverage+Δfrontier | 6 | OFF |
| T04 | graph+goal+motion | 147 | OFF |
| T05 | motion only | 3 | ON、OFF |
| T06 | motion+reward | 4 | ON、OFF |
| T07 | motion+Δcoverage+reward | 5 | ON、OFF |
| T08 | motion+Δcoverage+reward+goal | 21 | ON、OFF |

表中 graph 指池化图特征128D，goal 指历史选中目标特征16D，motion 指局部运动增量3D。T01还含Δfrontier。T02保留graph、goal、reward、Δcoverage、Δfrontier，移除motion。T04含graph、goal、motion，不含reward。T02/T03/T04不能仅凭文件夹简写推断 token 内容。

### 4.3 已评测 checkpoint 的训练预算

下表使用既有 checkpoint 审计与T08快照字段；“旧episode字段”属于调度计数，不能冒充精确完成轮数。ON/OFF分别指训练阶段的恢复设置。不同token训练预算并不完全一致，表格排名是现有系统结果，尚非严格等预算的因果消融。

| Token | 训练recovery | 完成episodes | 旧episode字段 | 环境步数 | learner更新 |
| --- | --- | --- | --- | --- | --- |
| T01 | OFF | 未记录 | 9568 | 411218 | 76384 |
| T01 | ON | 未记录 | 9824 | 412925 | 78432 |
| T02 | OFF | 未记录 | 10208 | 420852 | 81504 |
| T03 | OFF | 10000 | 10010 | 395353 | 79016 |
| T04 | OFF | 10000 | 10010 | 400301 | 79008 |
| T05 | OFF | 8347 | 8352 | 324048 | 66656 |
| T05 | ON | 8763 | 8768 | 338662 | 69984 |
| T06 | OFF | 10000 | 10000 | 413396 | 79880 |
| T06 | ON | 10203 | 10208 | 386163 | 81056 |
| T07 | OFF | 10000 | 10000 | 376345 | 79880 |
| T07 | ON | 10000 | 10000 | 387274 | 79880 |
| T08 | OFF | 10000 | 10000 | 375404 | 79880 |
| T08 | ON | 10000 | 10000 | 394789 | 79880 |

T05使用提前停止快照，OFF完成8,347、ON完成8,763轮；T06 ON训练至10,203轮，OFF为10,000轮。T01/T02旧快照没有精确完成轮数字段。训练轮数一致也不代表环境步数、梯度更新数或计算量完全一致。

### 4.4 Recovery V4

现有控制器为 local_revisit_recovery_v4。它结合局部区域反复访问、低覆盖增益与点重复模式检测停滞，触发后选择远离回环区域的frontier目标，并沿图路径执行逃脱。历史缓存记录实际执行动作及反馈，包含recovery动作。因此ON训练会改变采样状态、执行动作及后续学习分布；其效果不能用OFF训练模型在测试时开关recovery的结果替代。

| 参数组 | 设置 |
| --- | --- |
| 历史/局部重复 | 历史10；半径8m；最少访问7；score阈值0.7，退出0.5 |
| 低增益/触发 | 近8步覆盖增益阈值0.001；连续3步 |
| 点重复 | 单点至少3次；top3累计至少8次；连续3步 |
| 逃脱目标 | 最小逃脱距离12m；离回环区域至少8m |
| 结束限制 | 目标容差2m；恢复最多32步；路径失败上限3次 |

## 5. 新 token 实验结果

以下“训练/测试”均按 recovery 状态写为 OFF/OFF、OFF/ON、ON/ON。OFF/ON复用OFF训练checkpoint，只改变测试开关。全部新token评测均无运行异常；普通失败保留在统计内。没有完成的ON训练/OFF测试单元格不作推测。

### 5.1 单种子：seed 4777，N=1,133

| ID | Token | 训练/测试 | 策略 | 成功率% | 失败 | 路径m | 步数 | 覆盖率% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E006 | T01 | OFF/OFF | greedy | 99.029 | 11 | 260.68 | 36.23 | 99.480 |
| E008 | T01 | OFF/ON | greedy | 100.000 | 0 | 256.85 | 35.57 | 99.658 |
| E007 | T01 | ON/ON | greedy | 100.000 | 0 | 252.63 | 39.87 | 99.620 |
| E001 | T02 | OFF/OFF | greedy | 99.294 | 8 | 265.25 | 37.01 | 99.623 |
| E016 | T03 | OFF/OFF | greedy | 99.735 | 3 | 254.56 | 34.89 | 99.628 |
| E017 | T04 | OFF/OFF | greedy | 100.000 | 0 | 253.51 | 34.48 | 99.706 |
| E027 | T05 | OFF/OFF | greedy | 98.588 | 16 | 263.71 | 32.20 | 99.557 |
| E029 | T05 | OFF/ON | greedy | 100.000 | 0 | 254.71 | 31.22 | 99.712 |
| E028 | T05 | ON/ON | greedy | 100.000 | 0 | 254.65 | 33.48 | 99.693 |
| E033 | T06 | OFF/OFF | greedy | 99.735 | 3 | 253.02 | 34.38 | 99.631 |
| E035 | T06 | OFF/ON | greedy | 100.000 | 0 | 251.15 | 34.19 | 99.689 |
| E034 | T06 | ON/ON | greedy | 100.000 | 0 | 256.03 | 32.78 | 99.714 |
| E021 | T07 | OFF/OFF | greedy | 99.912 | 1 | 253.67 | 33.42 | 99.676 |
| E023 | T07 | OFF/ON | greedy | 100.000 | 0 | 252.92 | 33.35 | 99.696 |
| E022 | T07 | ON/ON | greedy | 100.000 | 0 | 254.29 | 36.89 | 99.661 |
| E040 | T08 | OFF/OFF | greedy | 100.000 | 0 | 257.18 | 29.39 | 99.712 |
| E042 | T08 | OFF/ON | greedy | 100.000 | 0 | 257.18 | 29.39 | 99.712 |
| E041 | T08 | ON/ON | greedy | 100.000 | 0 | 253.97 | 33.49 | 99.702 |

### 5.2 五种子：seeds 100–104，N=5,665

| ID | Token | 训练/测试 | 策略 | 成功率% | 失败 | 路径m | 步数 | 覆盖率% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E003 | T01 | OFF/OFF | greedy | 99.047 | 54 | 261.20 | 36.29 | 99.500 |
| E005 | T01 | OFF/ON | greedy | 100.000 | 0 | 256.41 | 35.63 | 99.668 |
| E004 | T01 | ON/ON | greedy | 100.000 | 0 | 252.79 | 40.07 | 99.623 |
| E036 | T02 | OFF/OFF | stochastic | 99.806 | 11 | 263.91 | 36.82 | 99.651 |
| E014 | T03 | OFF/OFF | greedy | 99.700 | 17 | 255.63 | 35.10 | 99.627 |
| E015 | T04 | OFF/OFF | greedy | 99.735 | 15 | 256.17 | 34.80 | 99.684 |
| E024 | T05 | OFF/OFF | greedy | 98.588 | 80 | 264.05 | 32.24 | 99.566 |
| E026 | T05 | OFF/ON | greedy | 100.000 | 0 | 254.40 | 31.23 | 99.709 |
| E025 | T05 | ON/ON | greedy | 100.000 | 0 | 256.40 | 33.76 | 99.707 |
| E030 | T06 | OFF/OFF | greedy | 99.647 | 20 | 256.01 | 34.79 | 99.628 |
| E032 | T06 | OFF/ON | greedy | 100.000 | 0 | 253.57 | 34.56 | 99.683 |
| E031 | T06 | ON/ON | greedy | 100.000 | 0 | 255.32 | 32.76 | 99.708 |
| E018 | T07 | OFF/OFF | greedy | 99.876 | 7 | 255.45 | 33.72 | 99.691 |
| E020 | T07 | OFF/ON | greedy | 99.982 | 1 | 254.84 | 33.66 | 99.706 |
| E019 | T07 | ON/ON | greedy | 99.982 | 1 | 253.37 | 36.91 | 99.652 |
| E037 | T08 | OFF/OFF | greedy | 99.982 | 1 | 257.75 | 29.51 | 99.711 |
| E039 | T08 | OFF/ON | greedy | 100.000 | 0 | 257.61 | 29.49 | 99.716 |
| E038 | T08 | ON/ON | greedy | 100.000 | 0 | 253.52 | 33.59 | 99.689 |

T02五种子记录实际是stochastic，即使上级可视化目录含greedy也不能列入greedy排名。单种子不是五种子表中的一个重复seed，两者不应计算为六个同协议独立训练结果。

### 5.3 同一 checkpoint 的 recovery 配对

下表全部通过checkpoint SHA256一致、episode键一致、起点坐标及起点栅格一致检查。ΔL = ON−OFF，负值表示恢复开启后平均路径更短。“共同成功ΔL”只在双方都成功的相同episode上计算，避免把新挽救的失败轨迹直接混入普通成功路径代价。

| Token | 种子数 | 失败OFF→ON | 挽救/新增失败 | 触发episodes | ΔL(m) | 共同成功ΔL(m) |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | 1 | 11→0 | 11/0 | 11 | -3.832 | 0.0000 |
| T05 | 1 | 16→0 | 16/0 | 16 | -8.999 | 0.0000 |
| T06 | 1 | 3→0 | 3/0 | 3 | -1.868 | 0.0000 |
| T07 | 1 | 1→0 | 1/0 | 1 | -0.750 | 0.0000 |
| T08 | 1 | 0→0 | 0/0 | 0 | 0.000 | 0.0000 |
| T01 | 5 | 54→0 | 54/0 | 54 | -4.794 | 0.0000 |
| T05 | 5 | 80→0 | 80/0 | 80 | -9.657 | 0.0000 |
| T06 | 5 | 20→0 | 20/0 | 20 | -2.433 | 0.0000 |
| T07 | 5 | 7→1 | 6/0 | 7 | -0.612 | 0.0056 |
| T08 | 5 | 1→0 | 1/0 | 1 | -0.138 | 0.0000 |

## 6. 历史基线与评测协议档案

以下18条保留seed数和动作策略。模型简称M0=ARiADNE-original，M1=ARiADNE-Mamba，M2=ARiADNE-Mamba-3.1，M2-R=同版本测试开启Recovery V4。原始完整模型名见all_results.csv。历史token统一标为未追溯；训练recovery未由本轮完整checkpoint审计确认，故不补写。表中的ON/OFF仅指评测metadata记录的测试recovery标志。历史模型同名并不保证相同checkpoint；具体路径与SHA见附件。此表用于完整保留原始评测结果。E002/E009、E011/E044的逐episode地图、seed、起点、成功、步数、路径及最终覆盖率相同，分别保留档案ID，不作为独立重复试验。

| ID | 模型 | 种子数 | 策略 | 测试R | SR% | 失败 | 路径m | 步数 | 覆盖% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E002 | M2 | 5 | greedy | OFF | 99.594 | 23 | 256.99 | 40.87 | 99.586 |
| E009 | M2 | 5 | greedy | OFF | 99.594 | 23 | 256.99 | 40.87 | 99.586 |
| E010 | M2 | 1 | greedy | OFF | 99.735 | 3 | 255.37 | 40.65 | 99.616 |
| E011 | M2-R | 5 | greedy | ON | 99.965 | 2 | 255.47 | 40.64 | 99.629 |
| E012 | M2-R | 5 | stochastic | ON | 99.982 | 1 | 256.98 | 40.76 | 99.630 |
| E013 | M2-R | 1 | stochastic | ON | 99.912 | 1 | 255.50 | 40.50 | 99.646 |
| E043 | M1 | 5 | greedy | OFF | 99.559 | 25 | 254.32 | 37.98 | 99.554 |
| E044 | M2-R | 5 | greedy | ON | 99.965 | 2 | 255.47 | 40.64 | 99.629 |
| E045 | M0 | 5 | greedy | OFF | 98.199 | 102 | 274.63 | 32.56 | 99.411 |
| E046 | M1 | 5 | stochastic | OFF | 99.982 | 1 | 254.83 | 37.61 | 99.631 |
| E047 | M2 | 5 | stochastic | OFF | 99.912 | 5 | 257.43 | 40.81 | 99.624 |
| E048 | M0 | 5 | stochastic | OFF | 99.241 | 43 | 270.67 | 32.00 | 99.648 |
| E049 | M1 | 1 | stochastic | OFF | 100.000 | 0 | 255.78 | 37.69 | 99.632 |
| E050 | M2 | 1 | stochastic | OFF | 100.000 | 0 | 256.20 | 40.57 | 99.636 |
| E051 | M0 | 1 | stochastic | OFF | 99.382 | 7 | 269.24 | 31.85 | 99.606 |
| E052 | M1 | 1 | greedy | OFF | 99.470 | 6 | 255.45 | 38.21 | 99.568 |
| E053 | M2-R | 1 | greedy | ON | 99.912 | 1 | 254.98 | 40.56 | 99.636 |
| E054 | M0 | 1 | greedy | OFF | 98.323 | 19 | 273.49 | 32.54 | 99.418 |

## 7. 已完成的训练耗时诊断数据

训练日志诊断窗口为10次warmup后50次更新。下表为既有诊断报告记录，单位ms。不同计时区间可能嵌套，不能将列出的子项简单相加。

| 运行/计时项 | T07 | T08 |
| --- | --- | --- |
| ON：总SAC更新 | 584.97 | 891.25 |
| OFF：总SAC更新 | 581.83 | 922.62 |
| ON：policy backward | 24.80 | 133.03 |
| ON：Q1 forward/backward | 100.86 | 203.87 |
| ON：Q2 forward/backward | 98.66 | 202.31 |
| ON：历史图forward | 386.26 | 387.72 |
| ON：Mamba | 24.40 | 17.75 |
| ON：Replay collate | 38.71 | 39.94 |
| ON：Replay sample | 0.79 | 5.03 |
| ON：Host→device | 15.61 | 15.67 |

合成输入微基准使用共享GPU0、batch32、历史长度16、padding360/有效节点180、BF16、历史图chunk32；4次warmup后测量12次。下表单次记录不作为独占硬件速度排名。Detach与Q forward复用均为诊断分支，未用于本报告任务结果表中的模型。

| 微基准配置 | 墙钟耗时ms |
| --- | --- |
| T08 | 727.0 |
| T07 | 519.8 |
| T08历史图detach | 519.2 |
| T08 Q当前forward复用 | 691.8 |
| T08重复测量 | 803.7 |

Q forward复用记录的峰值显存为16.75GiB，对应T08基准为5.74GiB。完整原始数据见reports/t08_speed_diagnosis_20260914/benchmark_results.json与live_log_profiles.json。

## 8. 实验完成情况与记录边界

现有新token测试组合包括：T01/T05/T06/T07/T08的OFF训练/OFF测试、OFF训练/ON测试及ON训练/ON测试；T02/T03/T04目前只有OFF训练/OFF测试。各组合的种子数与动作策略以结果表为准。尚无ON训练/OFF测试的完整对应单元格。

当前新token实验只有训练seed0；尚无独立多训练seed结果、同token的GRU/Transformer公平对照、回环惩罚reward实验或真实机器人验证。现有test已多轮参与模型设计与token筛选，不能视为从未用于选择的独立最终评估集。完整历史模型的训练预算、硬件环境与token定义尚未全部追溯，相关缺失字段不推定为与新模型一致。

本文件仅整理实验设置、指标口径和已有数据，不包含结果优劣分析、机制原因推断或最终token选择建议。

## 附录A. 图1完整统计指标

成功数、失败率可由正文N和失败数精确恢复；下表显式补齐。主要均值见第5、6节；样本方差与各覆盖阈值达成率在此列全。CSV保留未四舍五入数值。NaN表示该统计不可用，不能解释为0。

| ID | 地图/N | 成功数 | 失败率% | 异常 | 步数方差 | 路径方差m² |
| --- | --- | --- | --- | --- | --- | --- |
| E001 | 1133/1133 | 1125 | 0.7061 | 0 | 147.899 | 9727.554 |
| E002 | 1133/5665 | 5642 | 0.4060 | 0 | 140.574 | 6834.411 |
| E003 | 1133/5665 | 5611 | 0.9532 | 0 | 158.652 | 9523.927 |
| E004 | 1133/5665 | 5665 | 0.0000 | 0 | 112.613 | 5074.220 |
| E005 | 1133/5665 | 5665 | 0.0000 | 0 | 83.561 | 4869.313 |
| E006 | 1133/1133 | 1122 | 0.9709 | 0 | 161.130 | 8313.935 |
| E007 | 1133/1133 | 1133 | 0.0000 | 0 | 111.451 | 5108.859 |
| E008 | 1133/1133 | 1133 | 0.0000 | 0 | 85.204 | 5008.744 |
| E009 | 1133/5665 | 5642 | 0.4060 | 0 | 140.574 | 6834.411 |
| E010 | 1133/1133 | 1130 | 0.2648 | 0 | 130.720 | 5957.434 |
| E011 | 1133/5665 | 5663 | 0.0353 | 0 | 115.071 | 5468.861 |
| E012 | 1133/5665 | 5664 | 0.0177 | 0 | 119.396 | 5730.603 |
| E013 | 1133/1133 | 1132 | 0.0883 | 0 | 119.874 | 5763.032 |
| E014 | 1133/5665 | 5648 | 0.3001 | 0 | 108.736 | 7262.590 |
| E015 | 1133/5665 | 5650 | 0.2648 | 0 | 116.906 | 7462.218 |
| E016 | 1133/1133 | 1130 | 0.2648 | 0 | 104.524 | 6869.545 |
| E017 | 1133/1133 | 1133 | 0.0000 | 0 | 95.070 | 5372.196 |
| E018 | 1133/5665 | 5658 | 0.1236 | 0 | 95.446 | 5969.571 |
| E019 | 1133/5665 | 5664 | 0.0177 | 0 | 105.465 | 5445.675 |
| E020 | 1133/5665 | 5664 | 0.0177 | 0 | 87.199 | 5263.289 |
| E021 | 1133/1133 | 1132 | 0.0883 | 0 | 87.743 | 5830.368 |
| E022 | 1133/1133 | 1133 | 0.0000 | 0 | 105.878 | 5537.429 |
| E023 | 1133/1133 | 1133 | 0.0000 | 0 | 80.370 | 4837.796 |
| E024 | 1133/5665 | 5585 | 1.4122 | 0 | 207.662 | 16158.539 |
| E025 | 1133/5665 | 5665 | 0.0000 | 0 | 85.870 | 5244.676 |
| E026 | 1133/5665 | 5665 | 0.0000 | 0 | 86.655 | 5464.350 |
| E027 | 1133/1133 | 1117 | 1.4122 | 0 | 208.172 | 15272.917 |
| E028 | 1133/1133 | 1133 | 0.0000 | 0 | 83.928 | 5109.722 |
| E029 | 1133/1133 | 1133 | 0.0000 | 0 | 89.039 | 5650.075 |
| E030 | 1133/5665 | 5645 | 0.3530 | 0 | 118.075 | 8140.876 |
| E031 | 1133/5665 | 5665 | 0.0000 | 0 | 88.144 | 5148.264 |
| E032 | 1133/5665 | 5665 | 0.0000 | 0 | 90.153 | 5270.870 |
| E033 | 1133/1133 | 1130 | 0.2648 | 0 | 105.816 | 6861.777 |
| E034 | 1133/1133 | 1133 | 0.0000 | 0 | 92.838 | 5450.176 |
| E035 | 1133/1133 | 1133 | 0.0000 | 0 | 83.751 | 4844.456 |
| E036 | 1133/5665 | 5654 | 0.1942 | 0 | 114.479 | 7408.464 |
| E037 | 1133/5665 | 5664 | 0.0177 | 0 | 74.813 | 5281.466 |
| E038 | 1133/5665 | 5665 | 0.0000 | 0 | 82.921 | 4951.185 |
| E039 | 1133/5665 | 5665 | 0.0000 | 0 | 73.127 | 5144.878 |
| E040 | 1133/1133 | 1133 | 0.0000 | 0 | 70.029 | 4930.898 |
| E041 | 1133/1133 | 1133 | 0.0000 | 0 | 86.059 | 5194.731 |
| E042 | 1133/1133 | 1133 | 0.0000 | 0 | 70.029 | 4930.898 |
| E043 | 1133/5665 | 5640 | 0.4413 | 0 | 134.998 | 8229.765 |
| E044 | 1133/5665 | 5663 | 0.0353 | 0 | 115.071 | 5468.861 |
| E045 | 1133/5665 | 5563 | 1.8005 | 0 | 242.019 | 19976.868 |
| E046 | 1133/5665 | 5664 | 0.0177 | 0 | 108.235 | 5964.384 |
| E047 | 1133/5665 | 5660 | 0.0883 | 0 | 125.000 | 6073.679 |
| E048 | 1133/5665 | 5622 | 0.7590 | 0 | 166.235 | 13777.855 |
| E049 | 1133/1133 | 1133 | 0.0000 | 0 | 118.501 | 6711.339 |
| E050 | 1133/1133 | 1133 | 0.0000 | 0 | 122.468 | 5987.179 |
| E051 | 1133/1133 | 1126 | 0.6178 | 0 | 157.075 | 12321.716 |
| E052 | 1133/1133 | 1127 | 0.5296 | 0 | 145.766 | 8538.571 |
| E053 | 1133/1133 | 1132 | 0.0883 | 0 | 119.964 | 5692.581 |
| E054 | 1133/1133 | 1114 | 1.6770 | 0 | 232.553 | 17960.890 |

| ID | 达60% | 达70% | 达80% | 达90% | 达95% |
| --- | --- | --- | --- | --- | --- |
| E001 | 100.000% | 100.000% | 100.000% | 99.912% | 99.823% |
| E002 | 100.000% | 99.965% | 99.912% | 99.841% | 99.788% |
| E003 | 99.982% | 99.753% | 99.647% | 99.382% | 99.170% |
| E004 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E005 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E006 | 99.912% | 99.735% | 99.647% | 99.382% | 99.117% |
| E007 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E008 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E009 | 100.000% | 99.965% | 99.912% | 99.841% | 99.788% |
| E010 | 100.000% | 100.000% | 99.912% | 99.912% | 99.912% |
| E011 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E012 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E013 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E014 | 100.000% | 99.894% | 99.841% | 99.841% | 99.771% |
| E015 | 100.000% | 100.000% | 100.000% | 99.912% | 99.859% |
| E016 | 100.000% | 99.823% | 99.823% | 99.823% | 99.735% |
| E017 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E018 | 100.000% | 100.000% | 99.965% | 99.929% | 99.912% |
| E019 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E020 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E021 | 100.000% | 100.000% | 99.912% | 99.912% | 99.912% |
| E022 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E023 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E024 | 99.982% | 99.982% | 99.894% | 99.276% | 98.976% |
| E025 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E026 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E027 | 100.000% | 99.823% | 99.823% | 99.294% | 99.117% |
| E028 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E029 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E030 | 100.000% | 99.912% | 99.876% | 99.771% | 99.753% |
| E031 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E032 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E033 | 100.000% | 99.912% | 99.823% | 99.823% | 99.735% |
| E034 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E035 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E036 | 100.000% | 100.000% | 100.000% | 99.929% | 99.876% |
| E037 | 100.000% | 100.000% | 99.982% | 99.982% | 99.982% |
| E038 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E039 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E040 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E041 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E042 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E043 | 99.965% | 99.947% | 99.859% | 99.665% | 99.647% |
| E044 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E045 | 99.823% | 99.682% | 99.435% | 98.800% | 98.588% |
| E046 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E047 | 100.000% | 100.000% | 100.000% | 99.982% | 99.965% |
| E048 | 99.982% | 99.965% | 99.912% | 99.629% | 99.523% |
| E049 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E050 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E051 | 99.823% | 99.823% | 99.823% | 99.735% | 99.559% |
| E052 | 100.000% | 100.000% | 99.823% | 99.823% | 99.647% |
| E053 | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| E054 | 99.735% | 99.559% | 99.470% | 98.941% | 98.676% |

## 附录B. 数据来源与复核

所有路径相对项目根目录。all_results.csv包含每条评测的metadata目录、checkpoint路径、checkpoint SHA256、episodes.csv SHA256及split SHA256。per_seed_results.csv为逐测试seed统计，paired_recovery.csv为配对结果。audit.json记录计数和checkpoint检查结果；training_configs.json保存各新token训练配置快照。历史记录的checkpoint未在本次全部重新计算文件hash，附件中的历史hash来自原始metadata。

主要代码依据：ARiADNE_MAMBA_3.1/parameter.py、env.py、worker.py、model.py、models/history_encoder.py、models/mamba_encoder.py、models/fusion.py、local_revisit_recovery.py；评测依据ARiADNE_MAMBA_Test/mamba31_map_split/scripts/evaluate_on_split.py、evaluate_single_seed_greedy.py及evaluation_adapters/ariadne.py。划分依据experiment/map_splits/map_split_seed_42.json，SHA256为8397d4d1a3b1e6f57fa604b23f59284352a2e6dea6686f55ff62d0f531fcf0c0。

下面列出全部评测来源。路径对应原始图表和轨迹档案，附件保留ID映射，便于之后生成论文精简表而不丢弃不理想结果。

E001 — ARiADNE_MAMBA_3.1/experiments/pose_ablation_no_motion_delta_20260908/test_greedy_4777/results

E002 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_fresh_seed_0_v5_step539525

E003 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/five_seed_greedy_100_104/recovery_off

E004 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/five_seed_greedy_100_104/recovery_on

E005 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/five_seed_greedy_100_104/train_recovery_off_test_on

E006 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_off

E007 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_on

E008 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/train_recovery_off_test_on

E009 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_pure_policy_greedy_ablation/five_seed_greedy_100_104

E010 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_pure_policy_greedy_ablation/single_seed_greedy_4777

E011 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_recovery_v4_three_protocols/five_seed_greedy

E012 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_recovery_v4_three_protocols/five_seed_stochastic

E013 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_recovery_v4_three_protocols/single_seed_stochastic_4777

E014 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/five_seed_greedy_100_104/no_features_v2

E015 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/five_seed_greedy_100_104/no_map_delta_reward

E016 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_features_v2

E017 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_map_delta_reward

E018 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/five_seed_greedy_100_104/recovery_off

E019 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/five_seed_greedy_100_104/recovery_on

E020 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/five_seed_greedy_100_104/train_recovery_off_test_on

E021 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_off

E022 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_on

E023 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/train_recovery_off_test_on

E024 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/five_seed_greedy_100_104/recovery_off

E025 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/five_seed_greedy_100_104/recovery_on

E026 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/five_seed_greedy_100_104/train_recovery_off_test_on

E027 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_off

E028 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_on

E029 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/train_recovery_off_test_on

E030 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/five_seed_greedy_100_104/recovery_off

E031 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/five_seed_greedy_100_104/recovery_on

E032 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/five_seed_greedy_100_104/train_recovery_off_test_on

E033 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_off

E034 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_on

E035 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/train_recovery_off_test_on

E036 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/no_motion_delta_five_seed_stochastic_100_104_retry

E037 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/five_seed_greedy_100_104/recovery_off

E038 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/five_seed_greedy_100_104/recovery_on

E039 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/five_seed_greedy_100_104/train_recovery_off_test_on

E040 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_off

E041 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_on

E042 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/train_recovery_off_test_on

E043 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_greedy_100_104_recovery_v4/ariadne_mamba

E044 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_greedy_100_104_recovery_v4/ariadne_mamba_3_1_recovery_v4

E045 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_greedy_100_104_recovery_v4/ariadne_original

E046 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_stochastic_100_104/ariadne_mamba

E047 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_stochastic_100_104/ariadne_mamba_3_1

E048 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_five_seed_stochastic_100_104/ariadne_original

E049 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_4777/ariadne_mamba

E050 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_4777/ariadne_mamba_3_1

E051 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_4777/ariadne_original

E052 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba

E053 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba_3_1_recovery_v4

E054 — ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_original
