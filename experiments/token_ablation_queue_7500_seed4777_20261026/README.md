# HERA 三组 Token Ablation 顺序实验

用户指定基线：连续2步门槛、10000训练回合的完整HERA模型；选择依据为1133地图greedy测试的平均路径长度。基线与SHA256记录在HERA/baseline.json。

用户最新指定三组均先训练7500回合，全部从头初始化，不加载基线权重。除消融开关与训练预算外，使用相同seed4777、原4530训练地图、原五worker Ray采样/独立GPU学习器、优化器/学习率/回放/批次等设置。recovery开启，区域和点重复门槛均为2步，低进展阈值1原始新增自由格/步、窗口8。

顺序与开关：

1. no_target：仅关闭历史Token的action_features，16维目标特征置零；其他输入保留。
2. no_feedback：关闭历史Token的reward与map_delta（新增自由格计数），不改变训练奖励，也不关闭recovery的在线探索增益观测。
3. no_motion：关闭历史Token的motion_delta，局部位移/转角3维置零；其他输入保留。

保留21维原始Token布局，禁用分量置零。每组7500回合训练完成、验证checkpoint开关与预算后，立即自动进行原1133测试地图、seed4777、greedy、同起点、99%覆盖率目标、128步上限、recovery开启的测试。只有该组训练和测试都成功后，才启动下一组。每组开始前重新选择显存占用最低的两张GPU（先采样、后学习），不停止其他任务；出错则停止队列并保留现场。

位置：
- 当前状态：本目录status.json；固定计划：plan.json。
- 队列日志：HERA/logs/token_ablations_7500_20261026.service.log。
- 训练：HERA/experiments/ablation_{no_target,no_feedback,no_motion}_7500_seed4777_20261026/。
- 实时训练日志：对应同名_control/train.log；每组plots/index.html每30秒刷新。
- GIF：HERA/gifs下对应同名目录；普通训练GIF及逃逸触发GIF均开启。
- 测试数据：HERA/results/data/ablation_{name}_7500/recovery_on/single_seed_4777/。
- 测试图表：HERA/results/visualizations下对应相同层级。
- 测试日志：HERA/results/logs/ablation_{name}_7500_seed4777_20261026.log。
- TensorBoard仍使用6006端口，加入三组曲线，等待中的组在产生日志后出现。

每完成一组更新comparison.json/comparison.md。测试集仅用于评测，不进入训练地图。三组目前均为7500回合，而已选基线为10000回合；最终对照需披露预算差异。需要严格等预算结论时，应另外对齐预算，不能把当前差异完全归因于Token定义。


## 追加完整 Mamba、recovery 关闭组

用户追加第四组no_recovery：完整21维Token、Mamba开启，训练和测试均关闭recovery，其他训练设置不变；seed4777、从头10000回合。由单独等待服务在原三组训练与测试均完成后接续，不中断当前训练。开始前重新选择显存占用最低的GPU对。

- 排队状态：no_recovery_followup_status.json。
- 调度服务：hera-mamba-no-recovery-followup-10000-20261026.service。
- 调度日志：HERA/logs/no_recovery_followup_10000_20261026.service.log。
- 训练及GIF：HERA/experiments/ablation_no_recovery_10000_seed4777_20261026/、HERA/gifs下同名目录。
- 测试：HERA/results/data/ablation_no_recovery_10000/recovery_off/single_seed_4777/，1133地图、seed4777、greedy，保持其余协议。
- 验证checkpoint的use_mamba=True与use_local_revisit_recovery=False；完整Token各分量开启；评测recovery_enabled=False且所有回合触发/介入步数为0。
- 普通训练GIF保留，关闭recovery的实验不会产生逃逸触发GIF。
- TensorBoard6006加入no_recovery入口；实际产生日志后显示。

最新预算确认：no_target/no_feedback/no_motion仍各7500回合；完整Mamba、recovery-off为10000回合，与完整基线训练预算一致。
