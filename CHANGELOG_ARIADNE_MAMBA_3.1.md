# ARiADNE Mamba 3.1 变更说明

## 版本定位

`ariadne_mamba_3.1` 基于 `ariadne_mamba_1.1_trained` 的代码能力创建，但不携带训练产物。本版本的核心目标是在不修改 ARiADNE 主网络、Mamba 结构和 SAC loss 的前提下，增加一个可关闭、低频触发、便于测试的局部重复访问检测与 heuristic recovery 机制。

该机制是临时安全/恢复层，不是第二套 learned policy。正常状态仍由原 ARiADNE+Mamba policy 决策；只有连续检测到“高局部重复访问 + 低探索进展”时，heuristic recovery 才暂时接管动作，离开重复区域后立即恢复原 policy。

## 1. 分支与数据清理

新分支从 `ariadne_mamba_1.1_trained` 创建，并删除了不应继续复制到开发分支的生成数据：

- `model/` 下的训练 checkpoint；
- `train/` 下的 TensorBoard event 日志；
- `gifs/ariadne*/` 下的逐轮训练可视化；
- `benchmarks/results/*.csv` 下的跑分结果；
- 仓库中误提交的 `__pycache__` 和 `.pyc` 文件。

保留了代码运行所需的 `maps/` 数据集，以及 README 使用的 `gifs/demo_1.gif` 至 `gifs/demo_4.gif`。

`.gitignore` 已补充训练输出、模型权重、训练可视化、benchmark CSV 和 Python cache 的忽略规则，避免后续再次误提交。

## 2. 局部重复访问检测

新增 `local_revisit_recovery.py`，其中 `LocalRevisitDetector` 封装 episode 内的检测和恢复状态。

### 历史窗口

使用 FIFO `deque` 保存最近 `LOCAL_REVISIT_HISTORY_LENGTH` 个机器人 world coordinates。

当前 `NodeManager` 在一个 episode 中增量维护坐标节点，不会每一步整体重建并重新编号。尽管节点坐标身份较稳定，本实现仍在计算时把历史坐标重新映射到当前 graph 中最近的合法节点，以避免动态图更新造成旧引用失效。

每次 `Worker.run_episode()` 开始都会显式调用 detector reset，确保历史、coverage、trigger counter、escape target 和统计不跨 episode 泄漏。

### 当前 graph neighborhood

局部邻域定义为从当前节点出发，weighted graph shortest-path distance 不超过 `LOCAL_REVISIT_GRAPH_RADIUS` 的所有节点。

复用 `NodeManager.Dijkstra()`，边权为相邻 graph node world coordinates 之间的欧氏距离，因此这里使用的是真实累计 graph distance，不是 hop 数，也不会把隔墙但欧氏距离接近的节点直接视为同一邻域。

`Dijkstra()` 增加了可选 `boundary` 参数，检测邻域时可以在距离超过半径后提前结束；选择 escape target 和重新规划时仍运行完整可达图搜索。

### Revisit score

若当前历史窗口为 `H_t`，当前 graph 邻域为 `N_R(v_t)`，则：

```text
revisit_score = 映射后落入 N_R(v_t) 的历史位置数 / 当前历史位置数
```

取值范围为 `[0, 1]`。只有历史窗口和 coverage 窗口都填满后，检测证据才具备触发资格，避免 episode 初期小样本比例产生误触发。

## 3. Coverage progress gate

没有引入新的 coverage 定义。每一步直接复用 `Env.get_exploration_rate()`：

```text
delta_coverage = exploration_rate_after - exploration_rate_before
recent_mean_coverage_gain = 最近 K 步 delta_coverage 的平均值
```

只有 revisit score 高且 recent mean coverage gain 低时才累计触发计数。这样可以降低死胡同正常返回、走廊回流和合法短路径复用带来的误触发。

本测试版本默认只使用 `revisit score + coverage gain`，没有加入 frontier gain、reward penalty、turn penalty、Gaussian heatmap、region clustering 或 learned recovery policy。

## 4. 连续确认与触发

进入 recovery 必须同时满足：

```text
revisit_score > LOCAL_REVISIT_SCORE_THRESHOLD
recent_mean_coverage_gain < LOCAL_REVISIT_COVERAGE_THRESHOLD
以上条件连续满足 LOCAL_REVISIT_TRIGGER_STEPS 个 decision steps
```

单步异常不会直接触发。每次真正进入 recovery 时，`recovery_trigger_count` 增加一次。

## 5. Escape target 与路径执行

触发后从当前可达 graph 中选择一个位于当前局部邻域之外的节点。第一版规则保持简单、确定且可解释：

1. 节点必须在当前 graph 中可达；
2. weighted graph distance 必须大于局部邻域半径；
3. 首先选择 graph distance 最近的合法 escape node；
4. 历史访问次数和坐标只用于等距离情况下的确定性 tie-break。

路径复用现有 `NodeManager.Dijkstra()` 与 `get_Dijkstra_path_and_dist()`。

进入 recovery 后固定 escape target，不再允许 policy 每一步重新选择目标。每个 decision step 执行路径的下一跳，并把该 waypoint 映射回 observation 中真实存在的 action index。

若动态图更新导致下一跳失效，会针对同一 escape target 重新规划。只有固定目标已经不可达时才终止本次 recovery，而不是擅自切换到另一个目标。

## 6. Recovery 退出与 hysteresis

满足任一条件时退出：

- 到达固定 escape target；
- `revisit_score < LOCAL_REVISIT_EXIT_THRESHOLD`；
- 固定 escape target 在更新后的 graph 中不可达。

退出阈值低于触发阈值。退出时会清空检测窗口并重新收集完整证据，避免短时间内反复出现 `policy -> recovery -> policy -> recovery`。

## 7. Policy、Mamba history 与 replay

正常模式完全沿用原 policy action sampling 路径。

Recovery 模式下：

- 不调用 policy 来采样目标；
- heuristic waypoint 被映射到当前 observation 的合法 action index；
- replay 保存实际执行的 action，而不是 policy 可能会采样的另一个 action；
- Mamba rollout history token 同样使用实际执行的 recovery action feature；
- `TrajectoryStep`、`SequenceBatch` 和 `AlignedTransitionBatch` 新增 `is_recovery_action` 标志；
- 非 Mamba 的旧 episode buffer 也增加对应标志槽位。

当前 recovery transition 会进入 replay buffer，并参与现有 SAC 训练。SAC loss 没有修改，也不会读取 `is_recovery_action`。这意味着 critic 会观察带 heuristic intervention 的 transition，actor 可能通过 critic 数据分布受到间接影响。正式研究应对“保留或过滤 recovery transition”进行消融实验。

## 8. 配置参数

所有配置集中在 `parameter.py`：

```python
USE_LOCAL_REVISIT_RECOVERY = False
LOCAL_REVISIT_HISTORY_LENGTH = 16
LOCAL_REVISIT_GRAPH_RADIUS = 2 * NODE_RESOLUTION
LOCAL_REVISIT_SCORE_THRESHOLD = 0.75
LOCAL_REVISIT_EXIT_THRESHOLD = 0.5
LOCAL_REVISIT_PROGRESS_WINDOW = 8
LOCAL_REVISIT_COVERAGE_THRESHOLD = 0.001
LOCAL_REVISIT_TRIGGER_STEPS = 3
LOCAL_REVISIT_DEBUG_VISUALIZATION = False
```

总开关默认关闭。关闭后 detector 不保存历史、不累计 coverage、不接管动作，policy 行为路径保持原样。

## 9. 日志与可视化

Worker 保存逐步诊断字段：

- `revisit_score`；
- `recent_mean_coverage_gain`；
- `revisit_trigger_counter`；
- `recovery_mode`；
- `recovery_trigger_count`；
- `recovery_steps`；
- `escape_target`；
- `escape_path_length`；
- `is_recovery_action`。

Episode 级统计包括：

- `total_recovery_triggers`；
- `total_recovery_steps`；
- `recovery_step_ratio`；
- episode 结束时的 detection 状态。

TensorBoard 新增：

- `Recovery/Triggers`；
- `Recovery/Steps`；
- `Recovery/Step Ratio`。

打开 `LOCAL_REVISIT_DEBUG_VISUALIZATION` 后，可以在已有 graph 图中显示历史位置、当前邻域、escape target、recovery path、revisit score 和 recovery mode，并输出逐步调试日志。

## 10. 测试

新增 `tests/test_local_revisit_recovery.py`，覆盖：

- Case A：持续向前移动，不触发 recovery；
- Case B：局部往返震荡且 coverage 接近零，连续确认后触发；
- Case C：revisit score 较高但 coverage 明显增长，不误触发；
- Case D：escape target 在邻域外、路径下一跳合法、recovery 期间 target 固定、到达后退出；
- Case E：总开关关闭后 detector 为 no-op。

扩展 `tests/test_temporal_pipeline.py`，验证 `is_recovery_action` 在 sequence replay 和 aligned transition replay 中均不会丢失。

本机可运行的 19 项 synthetic、replay、history、mask 和 history optimization 相关测试全部通过。全量测试中的其余部分需要当前 Windows 环境中未安装的 `mamba_ssm`，完整 Mamba 运行仍建议使用支持官方 `mamba-ssm` 的 Linux CUDA 环境。

## 11. 运行方式

运行 recovery 和 temporal replay 测试：

```powershell
conda activate test1
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -m unittest tests.test_local_revisit_recovery tests.test_temporal_pipeline -v
```

启用 recovery 测试版本：

```python
USE_LOCAL_REVISIT_RECOVERY = True
```

由于该分支已移除 checkpoint，从头训练时还需要设置：

```python
LOAD_MODEL = False
```

## 12. 测试版边界与后续研究项

以下内容是为了快速验证机制有效性而采用的测试版设计：

- 当前默认阈值与窗口长度；
- 最近邻坐标重新映射；
- 简单的最近 escape target 规则；
- 触发时完整 Dijkstra；
- recovery transition 默认保留在 replay；
- 轻量 debug visualization。

如果正式用于论文实验，建议进一步完成：

- 参数 sweep 和不同地图上的敏感性分析；
- recovery trigger rate、step ratio 与计算开销统计；
- 保留/过滤 recovery transition 的训练消融；
- 动态 graph 变化和目标失效的鲁棒性评估；
- 与无 recovery 原始 policy 的严格同 seed 对照；
- 对失败案例、成功逃逸距离和额外路径开销的定量分析。
