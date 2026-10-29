# HERA 追加2000回合并自动测试

两组分别从各自10000回合的最终checkpoint分支续训到累计12000回合，保留3/3和2/2触发门槛、seed4777、recovery开启、原4530张训练地图及其他超参数。

续训目录：
- `recovery_streak3_seed4777_12k_20261025/`
- `recovery_streak2_seed4777_12k_20261025/`

每组continuation.json记录原checkpoint SHA256、原学习步数、原计数和恢复说明；恢复策略/critic/target/优化器/熵系数与计数。按已有恢复实现，回放缓冲区重新收集，未恢复原worker进行中状态，因此不是未中断训练的逐位复现。

新目录episodes.csv含原10000回合和新增2000回合；新增逐步诊断从10001回合开始。GIF位于HERA/gifs下同名12k目录，原10k GIF不复制也不删除。TensorBoard端口6006同时展示10k和12k曲线。

每组训练完成、累计计数严格验证为12000后，自动执行原1133地图、seed4777、greedy、同起点、99%覆盖率目标、128步上限、recovery开启的测试。分别恢复checkpoint的3步和2步门槛。

新测试结果：`HERA/results/data/recovery_streak{3,2}_12k/recovery_on/single_seed_4777/`；图表：`HERA/results/visualizations/recovery_streak{3,2}_12k/recovery_on/single_seed_4777/`。日志：`HERA/results/logs/recovery_streak{3,2}_12k_seed4777_20261025.log`。

两组测试均完成后自动生成`HERA/results/comparison/recovery_streak3_vs_streak2_12k_seed4777/`。旧10k checkpoint和测试结果全部保留。每组训练与测试均成功才会写training_and_test_complete.json。
