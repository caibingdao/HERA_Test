# Frontier-driven escape target selection 离线数据分析

分析日期：2026-08-14
分析对象：当前工作区中的 ARiADNE_MAMBA、`ariadne1_mamba_3.2` recovery 日志、地图与当前 checkpoint。未训练模型，未修改 SAC、recovery 或动作选择逻辑。

## 1. 数据范围与口径

### 可用数据

- 地图：`maps/` 下 5,663 张地图。
- 3.2 recovery 日志：202 个发生 trigger 的 episode，共 373 次 trigger。
- 3.1 recovery 日志：1,637 个 JSON，其中 21 个 episode 非空，共 30 次 trigger。3.1 与 3.2 detector 配置和日志字段不同，不混入尺度统计。
- 3.2 当前 checkpoint：episode 6,240。
- 完整逐帧 GIF 仅每 100 episode 保存；发生 trigger 且有 GIF 的历史 episode 只有 1000 和 3100，共 3 次 trigger。

### 关键数据限制

recovery JSON 只保存 trigger step、位置、旧 escape target 和恢复结果，不保存 trigger 时刻的 belief map、frontier、graph、完整轨迹或随机状态。因此，不能从现有 JSON 对全部 373 个历史 trigger 精确重建 frontier groups。相同 episode ID 用最终 checkpoint 重跑也不是历史复现，因为训练时策略参数在变化，且动作是随机采样的。

为避免把重跑结果冒充历史事实，本报告分成两种证据：

1. **尺度样本**：固定随机种子，用当前 3.2 checkpoint 离线重放日志中前 30 个 episode ID；覆盖 1,251 个决策状态和 10,123 个 frontier group。utility 选择对比使用第二次同口径重放的 1,170 个有效状态。重放没有改动或训练模型。
2. **历史 trigger 核验**：从 episode 1000、3100 的逐帧 GIF 恢复 trigger 帧。三个 trigger 的覆盖率均约 99.8%，画面中没有可用 frontier group；它们属于地图接近完成后的 loop，而非适合 frontier-driven recovery 的案例。

### 计算定义

- frontier：沿用项目 `get_frontier_in_map()` 的 0.8 m downsample 结果。
- group：0.8 m downsample 栅格上的 8 邻接连通分量，邻接阈值为 `sqrt(2) * 0.8 m`。
- `I_i`：group 中 downsample frontier 点数，不是原始 0.4 m frontier cell 数。
- `D_i`：机器人到可达 graph node 的 Dijkstra 距离，加该 node 到 group 最近 frontier 点的直线 connector，取最小值。这样解决了 frontier 点本身通常不是 graph node 的问题。
- 标准差采用总体标准差（`ddof=0`），P90 使用 NumPy 线性百分位。
- `epsilon` 仅用于数值保护；本样本最小 `D=1.697 m`，不主导结果。

## 2. Frontier group size distribution

样本：30 个离线重放 episode，1,251 个状态，10,123 个 group。

| 指标 | `|G_i|`（downsample frontier 点） |
|---|---:|
| min | 1 |
| max | 40 |
| mean | 4.542 |
| median | 3 |
| std | 3.594 |
| P90 | 9 |

分布明显右偏。典型 group 是 3–5 个点，90% 不超过 9 个点；40 属于长 frontier 边界的尾部值。直接使用 raw `I` 时，少量大 group 会显著支配线性 utility。

按 0.8 m frontier 间距粗略解释，`I=3`、`I=9` 分别对应约 2.4 m、7.2 m 的离散 frontier 长度；这只是数量尺度，不等同于可探索面积。

## 3. Frontier group navigation distance distribution

| 指标 | `D_i`（m） |
|---|---:|
| min | 1.697 |
| max | 55.437 |
| mean | 21.072 |
| median | 15.763 |
| std | 10.076 |
| P90 | 36.942 |

典型距离约 16–21 m，约为 4–5 个 `NODE_RESOLUTION=4 m`；P90 约 37 m。距离同样右偏，但量纲数值约是 `I` 的 3–5 倍，所以 `I - lambda D` 中的 `lambda` 不能直接取 1 或 2 作为中性初值。

基于 P90 的尺度比：

`I90 / D90 = 9 / 36.942 = 0.244 frontier-point/m`

因此 raw additive utility 中，`lambda≈0.2–0.5` 是合理起点；`lambda=1–2` 已明显偏向最近候选。

## 4. Triggered recovery episode analysis

### 历史日志总体特征

3.2 的 373 次 trigger：

| 指标 | 结果 |
|---|---:|
| trigger coverage min / mean / median / P90 / max | 0.0979 / 0.8277 / 0.9843 / 0.9998 / 1.0000 |
| trigger step median / P90 | 71 / 111.6 |
| target reached | 100% |
| recovery 后 coverage gain = 0 | 97.32% |
| recovery 后 coverage gain > 0 | 2.68% |
| 20 步内再次 trigger | 47.45% |

这里的“target reached 100%”不能解释为 recovery 有效：旧 controller 通常只选择约 8–12 m 外、1–2 个动作即可到达的最近 graph node。97.3% 零覆盖增量和 47.5% 快速重触发说明它经常只是走到 loop 邻域内的另一个节点。

### 有逐帧状态可核验的 trigger

| episode | trigger step | coverage | frontier groups | 候选排序 |
|---:|---:|---:|---:|---|
| 1000 | 67 | 0.998259 | 0 | 无候选 |
| 1000 | 82 | 0.998259 | 0 | 无候选 |
| 3100 | 67 | 0.998831 | 0 | 无候选 |

这三个事件说明实现时必须有明确 fallback：发生 loop trigger 并不保证 belief map 仍有 frontier。对这类接近完成的 episode，frontier-driven selector 不应返回任意值；应回退到非 frontier 的“离 loop 最远可达 graph node”，或者若探索终止条件已满足则直接结束 episode。

全部 373 个 trigger 的逐 group 表格无法由现有数据诚实地产生。要在后续实验中补齐，应在 trigger 当下额外保存：downsample frontier 坐标、graph node/edge、robot node、最近 8–16 个 graph positions，以及 belief map 或其哈希/压缩快照。

## 5. 不同 utility 的离线比较

下表在 1,170 个含 frontier 的离线状态上比较选择结果。`I/D` 作为 A；`C_ratio` 是逐状态 min-max 后的 `norm(I)/(norm(D)+1e-3)`；`C_diff` 是 `norm(I)-norm(D)`。`L` 是候选离当前 loop/机器人位置的距离代理。

| 方法 | 与 A 相同 | 与最大 I 相同 | 与最近 D 相同 | 所选 mean I | 所选 mean D (m) |
|---|---:|---:|---:|---:|---:|
| 最大 I | 73.59% | 100% | 29.15% | 10.215 | 19.418 |
| 最近 D | 44.79% | 29.15% | 100% | 5.551 | 13.578 |
| A: `I/D` | 100% | 73.59% | 44.79% | 9.554 | 16.176 |
| B, `lambda=0.1` | 83.25% | 88.80% | 34.19% | 10.168 | 18.235 |
| B, `lambda=0.5` | 90.94% | 71.97% | 46.58% | 9.516 | 15.767 |
| B, `lambda=1` | 82.99% | 60.17% | 56.84% | 8.763 | 14.696 |
| B, `lambda=2` | 70.09% | 48.89% | 69.06% | 7.836 | 14.017 |
| C ratio | 58.12% | 41.20% | 78.63% | 6.815 | 14.309 |
| C difference | 84.62% | 67.78% | 48.55% | 9.243 | 15.807 |

结论：

- A 在当前尺度上是合理 baseline，但仍有 44.8% 与最近 group 相同，作为 escape utility 略偏近。
- B 的 `lambda=0.5` 最接近 A；`lambda=1–2` 明显向最近 group 倾斜，和“逃离 loop”目标相反。
- 用户给出的 C 若按“归一化值相除”实现，会因为最近候选 `norm(D)=0` 而退化为最近 group 选择。样本中 78.6% 与最近 group 相同，不推荐。
- 将 C 改为 normalized difference 后行为稳定得多，但逐场景 min-max 对 outlier 和候选数量敏感。

### Loop-distance bonus

测试 `U=I/D + alpha*L`，其中 `L` 单位为 m：

| alpha | 与 A 相同 | 所选 mean D/L (m) | 判断 |
|---:|---:|---:|---|
| 0.01 | 88.80% | 18.142 | 温和增加逃离距离 |
| 0.05 | 46.92% | 26.543 | 明显偏向远端 |
| 0.10 | 31.45% | 29.190 | distance bonus 开始主导 |
| 0.25 | 21.54% | 30.254 | 过强 |
| 0.50 | 18.63% | 30.406 | 基本退化为选最远 |

由于 `I/D` 的典型值约为 `3/15.8=0.19`，而 raw `L` 是十几到几十米，`alpha=0.1` 绝不是“小修正”。raw 单位下推荐初值 `alpha=0.005–0.02 1/m`，中心值 0.01；若对 `L` 做 P90 normalization，则推荐无量纲权重 0.1–0.3。

更严格的 `L_i` 应定义为：候选 target node 到最近 loop-history graph node 的最短路距离，而不是到当前机器人位置的直线距离。这样可以区分“直线很远但仍在同一回廊/loop”与真正沿 graph 离开 loop 的候选。

## 6. 推荐 normalization 与参数范围

### 推荐 normalization

不建议逐场景 min-max ratio。推荐使用从离线数据固定下来的 robust scale，并在运行时 clip：

```text
I_hat = clip(I / I90, 0, 2)       # I90 = 9
D_hat = clip(D / D90, 0, 2)       # D90 = 36.94 m
L_hat = clip(L / D90, 0, 2)
score = I_hat - beta*D_hat + gamma*L_hat
```

固定 P90 normalization 有三个优点：不同 trigger 间分数可比；不会令最近候选的 denominator 变成 0；单个极端 group 不会重定义所有候选的尺度。生产参数可先圆整为 `I_scale=9`、`D_scale=L_scale=36 m`。

### 推荐初始范围

| 参数 | 推荐初值 | 建议扫描范围 |
|---|---:|---:|
| ratio epsilon（若保留 A，距离单位 m） | 1.0 m | 0.5–2.0 m |
| raw B lambda | 0.3 | 0.2–0.5 frontier-point/m |
| normalized distance beta | 0.7 | 0.5–1.0 |
| normalized loop bonus gamma | 0.2 | 0.1–0.3 |
| raw loop alpha（`I/D + alpha L`） | 0.01 1/m | 0.005–0.02 1/m |
| minimum useful group | 2 points | 1–3；不建议一开始硬过滤超过 2 |
| minimum escape distance | 12 m | 8–16 m（2–4 graph steps） |

`epsilon=1 m` 远小于典型 `D`，但能避免 frontier connector 恰好为 0 时出现极端 utility。不要用机器 epsilon 作为业务参数。

## 7. 哪个 utility 更适合当前地图

首选是 **固定 P90 normalization 的加权差**，而不是题目 C 的 normalized ratio：

```text
U = clip(I/9, 0, 2)
    - 0.7 * clip(D/36, 0, 2)
    + 0.2 * clip(L/36, 0, 2)
```

理由：当前地图中 `I` 与 `D` 都右偏且量纲相差明显；逃离 loop 还需要一个独立于“离机器人远”的 graph loop-distance 信号。加权差的每一项有清楚的上限和贡献，易诊断，也不会因最近候选 normalization 后为 0 而产生奇异值。

若第一版必须尽量简单，则使用 A 的轻微改版：

```text
U = I / (D + 1.0) + 0.01 * L
```

并要求 `D >= 8–12 m`；这比纯 `I/D` 更符合 escape，而不会像 `alpha>=0.05` 那样直接选最远 group。

## 8. 简单可实现的 escape target selection 建议

1. 提取 0.8 m downsample frontier，以 8 邻接聚类。
2. 对每组找 Dijkstra 可达、且能观测/连接到该 group 的 target graph node；不可达 group 丢弃。
3. 计算 `I`、graph `D`，以及 target 到最近 loop-history graph node 的最短路 `L`。
4. 优先排除 `D < 8–12 m` 的候选，除非所有候选都被排除；这个 guard 对避免“一步恢复、立即重触发”很重要。
5. 用上述 P90 normalized difference 排序；同分时依次选 `L` 大、`D` 小、group id 稳定者。
6. 选 group 后，不把 frontier cell 本身作为 waypoint；选择该 group 对应的最佳可达 graph node并锁定 target，沿 Dijkstra path 执行。
7. 若无 frontier group：在未达到终止条件时回退到 loop-history 外最远的可达 graph node；若探索已达到终止条件则结束 episode。

建议的第一组离线 A/B 参数是：`beta={0.5,0.7,1.0}`、`gamma={0.1,0.2,0.3}`、`D_min={8,12,16} m`。评价指标应以“恢复后 20/30 步内是否重触发”和“恢复后 coverage gain”作为主指标，而不是 target reached；旧数据已经表明 target reached 本身几乎没有区分力。

## 9. 最终结论

1. 当前环境的典型尺度：frontier group 中位数 3、均值 4.54、P90 9；graph navigation distance 中位数 15.76 m、均值 21.07 m、P90 36.94 m。
2. 推荐固定离线 P90 normalization，并 clip；不推荐逐 trigger min-max ratio。
3. raw additive distance 权重从 `lambda=0.2–0.5` 起步；raw loop bonus `alpha=0.005–0.02 1/m`；normalized 权重建议 `beta=0.5–1.0`、`gamma=0.1–0.3`。
4. 当前地图最适合 `normalized(I) - beta*normalized(D) + gamma*normalized(loop distance)`。若只能选题目 A–D 中原式，则 A 比 B(lambda>=1) 和 C ratio 更稳，D 仅在很小 alpha 下合适。
5. 简单落地版本：8 邻接 frontier group + 可达 graph target + `D_min=12 m` + P90 normalized difference；无 frontier 时必须有 graph escape fallback/探索终止分支。
