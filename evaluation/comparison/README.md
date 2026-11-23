# HERA 连续门槛测试对照

两个独立训练模型，各10000训练回合；均开启recovery。1133测试地图，seed4777，greedy，相同起点，目标覆盖率99%，最多128步。paired_maps.csv为逐图配对，summary.json为重算汇总。差异包含训练轨迹及权重变化，不是相同checkpoint仅改变测试门槛的纯消融。
