# 写作时必须披露的限制

本文件汇总 HERA 最终版本论文实验部分的已知限制。写正文、附录和 rebuttal 时按此口径，不要只挑有利结果。

## 1. 统计范围

- 测试协议是 **单测试 seed 4777**、每图 1 次、1133 张原始测试地图。`episodes.csv` 的 1133 行是 1133 个地图回合，不是多次重复。
- 训练是 **单训练 seed 4777**；没有多训练 seed 重复。不能把单次训练/单测试 seed 的均值写成统计显著性结论。
- `results/comparison/*/paired_maps.csv` 的逐图配对只能描述配对差异；配对检验不能替代多 seed 方差分析。
- 旧 bundle 中的五 seed（100–104）、indoors、自生成地图、utility 结果不属于本论文 1133 图 / seed 4777 协议，未纳入本分支。

## 2. 对照不是纯消融

- `recovery_streak2` 与 `recovery_streak3` 是分别训练的完整模型，权重和训练轨迹不同；只能写“两个完整系统的对照”，不能写“同一模型只改测试门槛”。
- `recovery_streak*_12k` 是从各自 10k checkpoint 续训到 12k；续训使用新的回放收集，未恢复 worker 进行中状态，不是未中断训练的逐位复现。
- `ablation_no_recovery_10000` 是单独训练的 recovery-off 模型；不能写成“同一 checkpoint 只关闭 recovery 开关”。
- `ablation_no_target_7000` 与 `ablation_no_target_7500` 是两次不同训练；no_target 系列训练预算（7000/7500）与 full 模型（10000/12000）不同，必须披露。
- `ablation_no_motion`、`ablation_no_feedback` 均为 7500 回合训练，与 full 的 10000 回合不同。

## 3. 模型选型

- 根目录 `baseline.json` 选择 `recovery_streak2_seed4777_10k_20261024`，选择准则为 `mean_path_all`（最短平均路径）。
- `results/comparison/model_selection_20261026.md` 从系统运行效率角度推荐 `recovery_streak3_seed4777_10k_20261024`：路径长约 0.76%，但平均决策数少约 22.7%，平均回合运行时间少约 21.1%。
- 两者不是同一模型；论文主结果必须固定一个，并在 `paper/FINAL_MODEL.json` 中记录。不能把 streak2 的路径和 streak3 的步数拼成一行。
- 四个 checkpoint 成功率均为 1133/1133；相同成功率不能证明分布相同或更强抗回环能力。

## 4. 指标与耗时

- 覆盖率是 `final_coverage` 的回合平均；路径长度同时报告全部回合与成功回合，不能只报成功回合。
- `mean_inference_time_ms` 是各回合 `mean_inference_time_ms` 的平均，不是所有决策加权平均。
- 推理耗时包含 recovery 检测/规划等，不是纯网络前向；不同规划器决策粒度不同，不能直接当作网络速度对比。
- 平均回合运行秒受硬件和负载影响，不能当作真实机器人运动时间。
- 成功率使用原始 `success` 字段；TARE official 的 3/1133 是官方版本在该评测器上的失败结果，不能代表 TARE 正常性能。
- NBVP/TARE 是确定性显式规划器；DSVP 是二维移植版，不是完整 ROS 部署。

## 5. 文档/环境不一致

- `README_HERA.md` 中“本目录的模型尚未使用新反馈定义重新训练”写于 2026-10-23；`HERA/experiments/*/run_config.json` 显示 2026-10-24 起的正式训练为 `from_scratch` 且 `exploration_progress_signal = new_free_cell_count_v1`。写作以后者为准，并说明该句已过时。
- `run_config.json` 记录了 CUDA 设备和当时环境，但未完整记录操作系统与依赖版本；`paper/audit/export_environment.json` 是导出时环境，不是历史训练环境。
- 原始 `run_config.json`/`evaluation_metadata.json` 保留绝对路径和 GPU UUID，仅用于溯源；公开脱敏版不要修改原始记录。
- 旧版 `paper/docs/experiment_summary.md` 是 TOKEN21 时代材料，其中的 full_test/no_target/no_motion/no_feedback 数字不是 HERA 最终数字；只可引用其协议定义和 caveats。

## 6. 数据与许可

- 1133 张测试地图来自原始 ARiADNE 地图集；公开再分发前需确认上游数据许可。
- NBVP、TARE、DSVP、CNN 的完整上游代码未复制，仅保留适配层和原始结果；上游地址与 commit 见 `paper/docs/BASELINES.md`。
- 训练地图 4530 张未纳入本分支；划分定义在 `paper/maps/map_split_seed_42.json`，训练图需要从上游地图集补齐。
- 训练权重、逐步诊断、TensorBoard、GIF 和训练日志未纳入；需要时从原始工作区获取，不要用本分支缺失文件推断不存在。
