# ARiADNE_MAMBA Loop Recovery V4.0 修改总结

## 1. 版本目标

V4.0 在 Frontier-driven Escape V3.0 的基础上修改 loop trigger，解决机器人在少数几个相邻或分散 graph node 之间来回扫描、但未始终落在“以当前位置为中心的局部区域”内时可能漏检的问题。

本版本保留：

- SAC + Mamba 正常探索；
- Frontier-driven escape target selection；
- locked target 与 committed Dijkstra recovery；
- recovery action 写入 replay 并参与现有 SAC 更新；
- 触发 episode 的 GIF、JSON、PNG 和 summary 专用归档。

本版本没有修改：

- ARIADNE encoder/decoder；
- Mamba、token definition 和 history pipeline；
- reward；
- SAC actor、critic、loss 或 replay sampling；
- frontier utility 与 recovery target ranking；
- recovery action 的执行和学习方式。

## 2. 离线分析依据

V4 参数来自对当前数据的逐帧离线回放：

- 27 个已发生 recovery trigger 的 episode；
- 103 个没有 recovery 记录的完整 episode；
- GIF 轨迹映射到项目 4 m graph node；
- 覆盖率条件沿用最近 8 步平均增量小于 `0.001`。

主要结论：

| 条件 | 单独覆盖已触发 episode | 对照集命中 | 结论 |
|---|---:|---:|---|
| 区域重复 `7/10`，连续 3 步 | 27/27 | 3/103 | 作为主分支 |
| `max >= 3, top2 >= 6`，连续 3 步 | 22/27 | 3/103 | 更灵敏但偏激进 |
| `max >= 3, top3 >= 8`，连续 3 步 | 20/27 | 2/103 | 更适合作为初始值 |
| 窗口内不同节点数 `<= 4` | 15/27 | 2/103 | 对跨小区域回环过于严格 |

点重复分支在对照集中额外发现了 3.3 episode 1100：候选 step 70 之后覆盖率到 episode 结束都没有增长，仍有约 203.9 m 路径，属于现有区域条件漏掉的真实停滞。

离线实验还表明，不能把区域条件和点条件先做逐步 OR 再共用一个连续计数器。episode 5 中，两种证据交替成立会让共享计数器在 step 12 提前触发，而两条分支各自稳定成立分别要到 step 55 和 step 57。因此 V4.0 使用两个独立连续计数器。

## 3. V4.0 触发流程

```text
最近 10 个历史 graph node（不包含当前 p[t]）
                |
        +-------+-------+
        |               |
        v               v
区域重复证据         点重复证据
revisit >= 7        max_count >= 3
                    top3_count >= 8
        |               |
        +-------+-------+
                |
最近 8 步平均 coverage gain < 0.001
                |
        +-------+-------+
        |               |
regional counter   point counter
连续 3 步          连续 3 步
        |               |
        +------- OR ----+
                |
          loop trigger
                |
    V3 frontier-driven target selection
                |
      committed Dijkstra recovery
```

## 4. 历史窗口与区域分支

运行参数从 V3 的保守 `16/13` 调整为离线分析推荐值：

```python
LOCAL_REVISIT_HISTORY_LENGTH = 10
LOCAL_REVISIT_MIN_COUNT = 7
LOCAL_REVISIT_PROGRESS_WINDOW = 8
LOCAL_REVISIT_COVERAGE_THRESHOLD = 0.001
LOCAL_REVISIT_TRIGGER_STEPS = 3
```

当前步 `p[t]` 不进入本步证据。detector 先将 `position_history` 中的 `p[t-10] ... p[t-1]` 映射到当前 graph，再计算有多少历史节点位于当前节点的 `8 m` graph neighborhood 内：

```text
regional_repeat = revisit_count >= 7
```

这对应 10 步窗口中的 70%。10 步整数窗口不能精确表达 75%；使用 8/10 的 80% 会漏掉多个已确认回环，因此采用 7/10。

## 5. 点重复分支

点重复只统计当前滑动窗口，不使用 episode-global `visit_counts`。

对映射后的 10 个历史节点执行 `Counter`，将频次降序记为：

```text
c1 >= c2 >= c3 >= ...
```

计算：

```text
max_count  = c1
top3_count = c1 + c2 + c3
```

默认条件：

```python
USE_POINT_REPEAT_DETECTION = True
POINT_REPEAT_MAX_COUNT = 3
POINT_REPEAT_TOP_K = 3
POINT_REPEAT_TOP_K_COUNT = 8
POINT_REPEAT_TRIGGER_STEPS = 3
```

即：

```text
point_repeat = max_count >= 3 and top3_count >= 8
```

它可以覆盖典型的：

- 两点往返：`5, 5`；
- 三点往返：`4, 3, 3`；
- 少量跨区节点：`3, 3, 2, 2`。

单独使用 `max_count` 容易把正常重复经过的路口当成 loop；单独限制 unique node 数又会漏掉跨相邻区域的回环。因此 V4 同时要求单点最低频次和前三点集中度。

## 6. 独立连续计数器

V4.0 分别维护：

```text
revisit_trigger_counter
point_repeat_trigger_counter
```

每条分支只由自己的证据更新：

```text
regional_abnormal = regional_repeat and low_progress
point_abnormal    = point_repeat and low_progress
```

最终触发条件：

```text
regional_counter >= 3 OR point_counter >= 3
```

不存在以下共享累计：

```text
counter += regional_abnormal OR point_abnormal
```

因此一种证据中断时不会由另一种证据替它延续 streak。

成功进入 recovery 后，两套计数器停止更新并保留触发时的值供事件日志记录；target selection 失败或离开 recovery 时，两套计数器都会清零。正常 recovery 结束后仍会清空 position history 和 coverage history，要求重新收集完整证据后才能再次触发。

## 7. Trigger source 与诊断字段

每次成功进入 recovery 会记录：

```text
trigger_source = regional
               | point_repeat
               | regional+point_repeat
```

逐步 diagnostics 和 trigger event JSON 新增：

- `point_repeat_max_count`
- `point_repeat_top_k_count`
- `point_repeat_unique_count`
- `point_repeat_trigger_counter`
- `regional_repeat_detected`
- `point_repeat_detected`
- `low_progress_detected`
- `trigger_source`

episode recovery summary 新增：

```text
trigger_source_counts
```

TensorBoard 新增：

- `Recovery/RegionalTriggers`
- `Recovery/PointRepeatTriggers`
- `Recovery/CombinedEvidenceTriggers`

这些字段可以区分 V4 的新增分支实际贡献了多少 trigger，并回看是否出现过早介入。

## 8. Frontier recovery 保持不变

V4 只改变 loop trigger。trigger 之后仍使用 V3.0：

```text
belief frontier
    -> 8-neighbor frontier groups
    -> reachable graph target
    -> D >= 12 m and L >= 8 m filter
    -> U = I / (D + 1 m)
    -> locked target
    -> committed Dijkstra recovery
```

无 frontier、filter fallback、graph fallback、target lock、target reached、max steps、path failure 等行为均未改变。

## 9. 输出目录与版本隔离

新启动的 V4 训练默认写入：

```text
model/ariadne1_mamba_3.4
train/ariadne1_mamba_3.4
gifs/ariadne1_mamba_3.4
```

发生 trigger 的 episode 专用归档写入：

```text
gifs/ariadne1_mamba_3.4/loop_recovery_v4_triggered/
```

已经启动的 3.3 Python 进程在启动时已加载旧参数，不会因为磁盘上的 `parameter.py` 更新而改变当前运行行为或输出目录。

## 10. 修改文件

| 文件 | V4.0 修改 |
|---|---|
| `parameter.py` | 切换 3.4 输出目录；设置 10/7 区域参数；增加 point-repeat 参数；更新 trigger artifact 目录 |
| `local_revisit_recovery.py` | 增加滑动窗口点频次统计、独立 point streak、trigger source 和诊断字段 |
| `worker.py` | 传入 V4 参数；扩展逐步日志、trigger JSON、episode metrics 和调试输出 |
| `driver.py` | 增加三类 trigger source TensorBoard 指标 |
| `recovery_artifacts.py` | 在 episode summary 中统计 trigger source |
| `tests/test_local_revisit_recovery.py` | 增加分散节点回环、低探索率 gate、独立计数器和滑动窗口测试 |
| `tests/test_recovery_visualization.py` | 验证 trigger JSON 中的 V4 字段 |
| `tests/test_recovery_artifacts.py` | 验证 summary 的 trigger source 统计 |
| `LOOP_RECOVERY_V4_0_SUMMARY.md` | 本文档 |

## 11. 测试结果

针对 detector、trigger logging、visualization 和 artifact archive 的定向测试结果：

```text
34 passed
```

项目完整测试集结果：

```text
79 passed, 1 skipped, 12 subtests passed
```

覆盖的 V4 关键行为包括：

- 分散在多个局部区域的高频节点可以由 point-repeat 分支触发；
- 区域 revisit 未达到 7 时，point-repeat 仍可独立工作；
- 有覆盖率增长时不触发；
- 区域和点证据交替时不会串接成连续 3 步；
- episode-global `visit_counts` 不参与点重复判定；
- trigger source 和点频次统计进入事件 JSON 与 summary；
- V3 frontier selection、fallback 和 committed recovery 既有测试继续通过。

## 12. 后续监控建议

V4 初次训练重点观察：

1. `PointRepeatTriggers / Triggers` 的比例；
2. `point_repeat` trigger 后的 coverage gain；
3. point-repeat trigger 是否集中在 episode 前 15 步；
4. 20/30 步内 retrigger；
5. `regional+point_repeat` 是否异常偏多；
6. 专用 GIF 中是否出现正常通过路口时被接管。

如果点重复仍然漏检，可离线评估将条件放宽为 `max >= 3 and top2 >= 6`；不建议在没有新日志验证前直接使用该激进条件，也不建议取消连续 3 步确认。
