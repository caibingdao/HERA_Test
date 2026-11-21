# Nearest Frontier

受 [Yamauchi 1997](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated1/yamauchi_frontiers.pdf) 启发的独立二维栅格实现。未找到作者公开发布的原始源码。ROS 的 [explore_lite](https://github.com/hrnr/m-explore/blob/noetic-devel/explore/src/frontier_search.cpp) 默认按距离和前沿大小计算代价。[frontier_exploration_ros2](https://github.com/mertgulerx/frontier_exploration_ros2) 有 Nearest 模式，其 ROS 节点通过 Nav2 执行导航。

每次规划以当前观测栅格和机器人所在栅格为输入。与未知栅格四邻接的自由栅格是候选前沿。广度优先搜索只经过已知自由栅格，按路径长度选出最近的可达候选，排除机器人当前栅格。距离平局时，搜索固定按上、左、右、下扩展。执行最短路径的第一个直线段，最长 4 m，更新感知后重新选前沿。没有可移动到的前沿时结束探索。规划器无训练权重、信息收益权重或 recovery。

共享评测器负责按地图指定位姿初始化、记录真实覆盖率并根据 99% 目标覆盖率或 128 次决策上限结束回合。决策步为一次直线段运动，粒度可能与图节点方法不同。

在工作区根目录使用已有的 `ariadne_mamba` Python 环境运行。默认评测原测试集 1133 图、seed 4777，每张图一次并生成六张统计图：

```bash
python ARiADNE_NEAREST_FRONTIER/evaluate.py \
  --output-dir ARiADNE_NEAREST_FRONTIER/results/data/single_seed_4777 \
  --visual-output-dir ARiADNE_NEAREST_FRONTIER/results/visualizations/single_seed_4777
```

快速验证可用 `--limit-maps 3` 和独立输出目录。模型制品字段指向 `specification.json`，它记录规划器配置。
