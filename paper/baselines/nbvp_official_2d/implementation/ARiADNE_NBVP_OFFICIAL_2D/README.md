# RH-NBVP 二维移植

算法依据 [ETH ASL nbvplanner 源码](https://github.com/ethz-asl/nbvplanner/tree/5c07657af18f967d7049d187db0607024746a42c) 和 [Bircher et al., ICRA 2016](https://www.research-collection.ethz.ch/entities/publication/ae2aefd8-1e86-4601-bb36-fa175be18481)。移植对应 `RrtTree::initialize`、`iterate`、`gain`、`getBestEdge`、`memorizeBestBranch` 和 `nbvp.hpp` 的停止逻辑。

每次以当前位置为根，在二维矩形画布均匀采样，向欧氏距离最近的树节点连接。边和端点之后的 0.5 m 必须穿过已知自由栅格。沿分支累计收益：

```
distance(child) = distance(parent) + length(parent, child)
gain(child) = gain(parent) + visible_unknown_area(child) * exp(-0.5 * distance(child))
```

可见未知面积由 360°、16 m 范围内到未知栅格中心的无遮挡视线计算。视线遇已知障碍停止。执行最高收益分支的第一段，更新感知，然后重新验证并插入上一轮最佳分支的剩余节点，再继续采样。达到最少 15 个有效节点且找到正收益后选分支。超过 200 个有效节点仍无收益时结束。候选插入尝试次数上限沿用上游；触发上限时，在仍然可行的情况下按上游思路返回历史上一点。逐图记录此类原生回退次数。

二维环境采用点机器人和全向激光，去掉无人机高度、朝向采样、三维包围盒、ROS 控制及多机避让。候选边最大 4 m，对齐当前评测的节点尺度；上游示例为 1 m。范围使用仿真传感器的 16 m；上游 RotorS 示例为 1.5 m。两端落在同一栅格的连接会被拒绝。每次移动一条有效树边计作一次决策。规划器使用 belief 栅格和位姿，完整地图覆盖率仅由共享评测器计分及判断停止。

测试使用原 1133 张地图、全局 seed 4777、原起点、99% 目标覆盖率、128 次决策，HERA recovery 关闭。算法采样随机数从评测 seed 初始化一次并跨地图保持。

在 `ariadne_mamba` 环境运行，需有 `g++` 编译栅格几何库：

```bash
python ARiADNE_NBVP_OFFICIAL_2D/evaluate.py \
  --output-dir ARiADNE_NBVP_OFFICIAL_2D/results/data/single_seed_4777 \
  --visual-output-dir ARiADNE_NBVP_OFFICIAL_2D/results/visualizations/single_seed_4777
```

全量评测自动核验 1133 张地图的配对起点，输出逐图 CSV、覆盖率曲线、汇总和六张图。
