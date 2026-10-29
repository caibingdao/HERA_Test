# ARiADNE_MAMBA Frontier-driven Escape V3.0 版本总结

## 1. 版本目标

本版本在当前 ARiADNE_MAMBA 3.2 基础上，将 loop recovery 的 escape target 从“loop 邻域外最近 graph node”替换为“frontier group 驱动的可达 graph target”。

目标是验证以下组合是否比 nearest-node recovery 更稳定地离开局部循环：

```text
loop detector
    + frontier-driven escape target
    + locked target
    + committed Dijkstra recovery
```

正常状态仍由 SAC + Mamba policy 决策。只有原有 loop detector 达到 trigger 条件后，heuristic recovery 才接管动作。

## 2. 修改文件

| 文件 | 修改内容 |
|---|---|
| `parameter.py` | 增加 V3.0 开关及三个默认参数 |
| `local_revisit_recovery.py` | 增加 frontier group、candidate evaluation、filter、fallback 和目标元数据；保留 V2.1 committed recovery |
| `worker.py` | trigger 时传入当前 belief map frontier；扩展 trigger JSON 和逐步 action-source 诊断 |
| `recovery_artifacts.py` | 为每个触发 episode 缓存轻量 belief frames，并生成专用 GIF/PNG/JSON/summary |
| `driver.py` | 将 V3 target 类型、D/L、candidate 和 20/30-step retrigger 指标写入 TensorBoard |
| `tests/test_local_revisit_recovery.py` | 增加 V3.0 clustering、utility、filter fallback、no-frontier fallback、target lock/reached 和 ablation 测试 |
| `tests/test_recovery_visualization.py` | 验证新增日志字段、20/30-step retrigger 统计和既有 GIF 标记 |
| `tests/test_recovery_artifacts.py` | 验证非触发不落盘，以及触发 episode 的完整专用归档 |
| `tools/archive_recovery_episodes.py` | 将已有 recovery JSON、GIF 和 final PNG 按 episode 重新归档并生成索引 |
| `FRONTIER_DRIVEN_ESCAPE_V3_0_SUMMARY.md` | 本版本总结 |

既有 `env.py` recovery visualization 和 replay buffer 的 `is_recovery_action` 能力直接复用，不需要为 V3.0 改写。

## 3. Frontier-driven escape 流程

```text
SAC + Mamba 正常探索
        |
        v
原有 loop detector trigger
        |
        v
get_frontier_in_map(current belief map)
        |
        v
8-neighbor frontier connected components
        |
        v
每个 group 映射到附近的 Dijkstra 可达 graph node
        |
        v
计算 I、D、L、U
        |
        v
D >= 12 m 且 L >= 8 m 的 hard filter
        |
        v
选择最大 utility candidate
        |
        v
锁定 world-coordinate target
        |
        v
committed Dijkstra navigation
        |
        v
target reached / 异常退出
        |
        v
返回 SAC + Mamba
```

loop detector 的 revisit calculation、history window、coverage threshold、连续 trigger gate 均未修改。

## 4. Frontier extraction 与 clustering

### 4.1 Frontier extraction

Worker 在每个 decision step 已持有当前 belief map。进入 detector 前调用项目原有：

```python
get_frontier_in_map(self.env.belief_info)
```

V3.0 没有重新实现 frontier detection。输入选择器的是项目已经按 `FRONTIER_CELL_SIZE=0.8 m` downsample 的 frontier 坐标集合。

### 4.2 8-neighbor connected component

`LocalRevisitDetector.cluster_frontiers()` 使用物理坐标上的 8 邻接：

```text
abs(dx) <= FRONTIER_CELL_SIZE
and
abs(dy) <= FRONTIER_CELL_SIZE
```

通过 BFS 得到 connected components。每个 `FrontierGroup` 保存：

- `group_id`
- `frontier_points`
- `frontier_count`
- `centroid`

Frontier potential 定义为：

```text
I_i = frontier_count = |G_i|
```

group id 由排序后的 frontier 坐标确定，因而同一输入下结果稳定。

## 5. Frontier group 到 target node

Frontier cell 位于 free/unknown 边界，不直接作为 recovery goal。

对每个 group：

1. 从机器人当前位置执行完整 Dijkstra；
2. 仅保留 `distance < 1e8` 且不是当前节点的 graph node；
3. 选择欧氏距离 group centroid 最近的可达 node；
4. 距离相同时，优先 navigation distance 更短、坐标排序更小的 node。

最终 `locked_escape_target` 始终是可由现有 graph 和 Dijkstra path 到达的 node，而不是 frontier centroid。

## 6. Candidate evaluation

每个可达 frontier group candidate 计算：

### 6.1 Navigation distance

```text
D_i = graph_shortest_path(robot_node, candidate_target_node)
```

直接使用 NodeManager Dijkstra 返回的路径距离，单位为 meter。

### 6.2 Loop distance

历史 robot positions 先映射到当前 graph node。当前 robot node 也加入 loop node 集合。

选择器从这些 loop nodes 同时出发执行一次 multi-source Dijkstra，得到：

```text
L_i = graph_shortest_path(candidate_target_node, nearest_loop_node)
```

它比纯欧氏距离更适合回廊和障碍地图，也避免对每个 candidate 重复运行完整 Dijkstra。

### 6.3 Utility

V3.0 按要求使用简单固定公式：

```text
U_i = I_i / (D_i + ESCAPE_DISTANCE_EPS)
```

默认 `ESCAPE_DISTANCE_EPS=1.0 m`。

utility 相同时，依次使用以下稳定 tie-break：

1. frontier count 更大；
2. loop distance 更大；
3. navigation distance 更小；
4. group id 更小。

## 7. Candidate filtering

候选需同时满足：

```text
D_i >= ESCAPE_MIN_DISTANCE
L_i >= ESCAPE_MIN_LOOP_DISTANCE
```

默认值：

| 参数 | 默认值 | 单位 |
|---|---:|---|
| `USE_FRONTIER_ESCAPE` | `True` | - |
| `ESCAPE_MIN_DISTANCE` | `12.0` | m |
| `ESCAPE_MIN_LOOP_DISTANCE` | `8.0` | m |
| `ESCAPE_DISTANCE_EPS` | `1.0` | m |
| `SAVE_TRIGGERED_RECOVERY_ARTIFACTS` | `True` | - |
| `TRIGGERED_RECOVERY_GIF_DURATION` | `0.5` | s/frame |

如果至少一个 candidate 通过 hard filter，只在通过者中选最大 utility，日志中的 `target_type="frontier"`。

如果存在 frontier candidates，但全部被 hard filter 排除，不让 recovery 静默失败，而是在全部 candidates 中选择最大 utility，日志中的：

```text
target_type = "frontier_filter_fallback"
```

每个 candidate 的 `passed_min_distance` 和 `passed_min_loop_distance` 都写入日志，便于判断 fallback 原因。

## 8. No-frontier 与异常 fallback

### 8.1 没有 frontier，探索未完成

从当前 graph 中寻找可达且非当前节点的 target，按以下顺序最大化：

1. 到最近 loop node 的 graph distance；
2. 到当前 robot 的 navigation distance；
3. 稳定坐标顺序。

选中后记录：

```text
target_type = "graph_fallback"
escape_target_type = "graph_fallback"
```

因此只要 graph 中存在其他可达节点，不会因为没有 frontier 返回 `None`。

### 8.2 没有 frontier，探索已经完成

Worker 通过当前 graph utility 判断 exploration complete。此时不进入 recovery，允许原 episode termination 流程结束 episode，并在 detector 内保留 `exploration_complete` 原因。

### 8.3 Ablation fallback

设置：

```python
USE_FRONTIER_ESCAPE = False
```

恢复 V2.1 nearest-node escape target selection，日志记录：

```text
target_type = "nearest_node"
```

这使 frontier-driven 与 nearest-node 可以在不改代码的情况下做 ablation。

### 8.4 Recovery 异常退出

保留 committed recovery 的异常退出条件：

- `target_unreachable`
- `path_failure`
- `max_steps`
- `episode_end`

正常退出仍只依赖 target tolerance，未使用 revisit score 下降作为退出条件。

## 9. Target lock 与 committed recovery

选择完成后保存：

```text
locked_escape_target
escape_path
escape_target_type
selected candidate metadata
```

recovery 期间不会因为以下变化重新排名 frontier：

- revisit score 变化；
- frontier 变化；
- SAC 输出变化。

如果 graph 更新导致下一 waypoint 失效，只向同一个 locked world-coordinate target 重规划；不会选择新的 frontier group。到达 tolerance 内才以 `target_reached` 正常退出并恢复 SAC。

## 10. Recovery 日志格式

每次实际进入 recovery 时，episode JSON event 包含：

```json
{
  "episode_id": 42,
  "trigger_step": 67,
  "candidate_num": 7,
  "selected_group_id": 5,
  "frontier_count": 31,
  "navigation_distance": 14.6,
  "loop_distance": 14.60,
  "utility": 1.987,
  "target_node": [12.0, 8.0],
  "target_type": "frontier",
  "escape_target_type": "frontier",
  "candidates": [
    {
      "group_id": 5,
      "frontier_count": 31,
      "centroid": [11.135, 9.458],
      "D": 14.6,
      "L": 14.60,
      "utility": 1.987,
      "target_node": [12.0, 8.0],
      "passed_min_distance": true,
      "passed_min_loop_distance": true
    }
  ],
  "recovery_start_step": 67,
  "recovery_end_step": 70,
  "exit_reason": "target_reached"
}
```

示例数值来自真实 graph 冒烟测试；candidate 数组实际保存全部可达 groups，不只保存 selected group。

既有字段继续保留：

- trigger position、revisit count/score、coverage gain；
- recovery steps、target reached；
- recovery 前后 coverage；
- `steps_since_last_recovery_end`；
- `retrigger_within_20_steps`；
- `retrigger_within_30_steps`。

逐步 diagnostics 继续记录实际动作来源：

```text
action_source = "sac"
```

或：

```text
action_source = "recovery"
```

写入 replay 的 action 仍是实际执行 action；`is_recovery_action` metadata 保留，SAC loss 不读取它做额外分支。

## 11. GIF visualization

V3.0 复用并保持当前 recovery visualization：

- loop trigger：红色 `x`，编号 `R1`、`R2`……；
- locked escape target：绿色星号，编号 `T1`、`T2`……；
- recovery trajectory：gold 独立线段；
- 正常 SAC trajectory：原蓝色轨迹。

V3.0 target 仍通过统一的 `escape_target` event 字段传给 `env.py`，因此 frontier target、filter fallback、graph fallback 都能使用同一 GIF 绘制路径，不影响正常 SAC 轨迹。

### 11.1 所有触发 episode 的专用归档

旧逻辑只有 `episode % SAVE_IMG_GAP == 0` 时才创建标准三面板 GIF，因此历史 202 个触发 episode 中只有 2 个有 GIF。V3.0 增加独立开关：

```python
SAVE_TRIGGERED_RECOVERY_ARTIFACTS = True
TRIGGERED_RECOVERY_ARTIFACT_PATH = (
    f"{gifs_path}/frontier_escape_v3_triggered"
)
TRIGGERED_RECOVERY_GIF_DURATION = 0.5
```

每个 episode 在内存中缓存 uint8 belief frame 和机器人位置。没有 trigger 时，episode 结束后直接丢弃，不写文件；发生至少一次 trigger 时，自动建立：

```text
frontier_escape_v3_triggered/
└── episode_XXXXXX/
    ├── episode_XXXXXX_recovery.gif
    ├── episode_XXXXXX_recovery_final.png
    ├── episode_XXXXXX_recovery_events.json
    ├── episode_XXXXXX_recovery_steps.json
    └── episode_XXXXXX_recovery_summary.json
```

专用 GIF 展示完整 episode belief/trajectory、红色 trigger、绿色 target、gold recovery segment。summary 汇总 target/exit 类型、D/L、candidate/frontier 数、utility、20/30-step retrigger、action source 和 episode 指标。

已有 3.2 数据已整理到：

```text
gifs/ariadne1_mamba_3.2/legacy_triggered_recovery_archive/
```

该归档包含 202 个触发 episode、373 次 trigger 的独立目录和总索引。旧训练只实际保存了 episode 1000、3100 两个匹配 GIF，其余 200 个目录明确标记 `gif_available=false`，没有伪造缺失帧。

### 11.2 新增 TensorBoard 统计

- `Recovery/FrontierTargets`
- `Recovery/FilterFallbacks`
- `Recovery/GraphFallbacks`
- `Recovery/NearestNodeTargets`
- `Recovery/ReTriggerWithin20`
- `Recovery/ReTriggerWithin30`
- `Recovery/MeanCandidateNum`
- `Recovery/MeanFrontierCount`
- `Recovery/MeanNavigationDistance`
- `Recovery/MeanLoopDistance`

## 12. 测试结果

### 12.1 自动化测试

运行：

```bash
/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python -m pytest -q
```

结果：

```text
75 passed, 1 skipped, 1 warning, 12 subtests passed
```

warning 是 PyTorch 首次建立 CUDA/cuBLAS context 的提示，不是测试失败。

### 12.2 V3.0 定向测试

| 测试 | 验证内容 | 结果 |
|---|---|---|
| 多 frontier groups | 8 邻接聚类、utility、selected target 可达 | 通过 |
| candidates 全部过近 | hard filter 后仍按最大 utility fallback | 通过 |
| 无 frontier、未完成 | 选择离 loop 最远的可达 graph node | 通过 |
| 无 frontier、已完成 | 不接管，允许 episode termination | 通过 |
| target reached | locked target 不变，正确正常退出 | 通过 |
| ablation switch | `False` 恢复 nearest-node 规则 | 通过 |
| path invalid | 重规划到同一 locked target | 通过 |
| 20/30-step retrigger | 正确更新当前和前一次 event 的统计字段 | 通过 |
| action metadata | 实际 recovery action 保留 `is_recovery_action` | 通过 |
| GIF markers | trigger、target、recovery path 分色和编号 | 通过 |
| 触发 episode 专用归档 | GIF、final PNG、events、steps、summary 全部生成 | 通过 |
| 非触发 episode 归档 | 不写磁盘文件，buffer 可正确清理 | 通过 |

### 12.3 真实 graph 冒烟测试

使用 episode 1 初始 belief map 和真实 `Agent/NodeManager`，人为填满 detector evidence 后触发一次 V3.0：

```text
frontier points:       47
reachable candidates: 7
selected group:        5
target type:           frontier
target node:           [12.0, 8.0]
D:                     14.6 m
L:                     14.60 m
utility:               1.987
Dijkstra path:         successfully generated
```

说明选择器可以直接处理项目真实 frontier set、QuadTree graph 和 Dijkstra 接口，不只适用于 synthetic graph。

## 13. 明确未修改的部分

本次 Frontier-driven Escape V3.0 没有改变：

- ARIADNE encoder；
- ARIADNE decoder；
- Mamba encoder/fusion；
- temporal token definition；
- reward；
- SAC actor；
- SAC critic；
- SAC loss；
- replay sampling/training pipeline；
- loop detector 的 history、revisit、coverage 和 trigger condition。

本版本只改变 loop trigger 后的 recovery target selection，并扩展对应日志与既有 visualization 数据。

## 14. 后续实验建议

V3.0 的核心评价不应只看 `target_reached`，建议直接比较 `USE_FRONTIER_ESCAPE=True/False` 的：

1. recovery 后 20/30 steps 内 retrigger rate；
2. recovery 后 coverage gain；
3. mean navigation distance 和 recovery action 数；
4. `frontier`、`frontier_filter_fallback`、`graph_fallback` 占比；
5. `target_unreachable`、`path_failure`、`max_steps` 占比；
6. episode success rate 和 travel distance。

该实验用于回答本版本的唯一核心问题：

> loop detector + frontier-driven escape target + committed recovery，是否比 nearest-node escape 更稳定地脱离局部循环？
