# HERA 最终模型完整测试

两个版本均使用各自10000训练回合后的最终checkpoint，recovery开启，恢复checkpoint中记录的区域/点重复连续门槛（3/3与2/2），新增自由格阈值1、窗口8。

协议与历史原始地图测试一致：1133张test地图、split_seed42的原清单，global seed4777，greedy argmax，每图一次，按清单顺序单进程执行，模型加载后一次初始化全局随机种子，地图间不重新播种，seeded_map_marker_selection起点，99%覆盖率目标，最多128次决策，不设路径长度上限。

- 实时日志：`logs/recovery_streak3_seed4777_20261025.log` 和 `logs/recovery_streak2_seed4777_20261025.log`。
- 启动配置/权重哈希：`runs/recovery_streak3_seed4777.json` 和 `runs/recovery_streak2_seed4777.json`。
- 进行中逐图进度：`data/recovery_streak{3,2}/recovery_on/.single_seed_4777.staging/progress.jsonl`（隐藏目录）。
- 测试完成后发布：`data/recovery_streak{3,2}/recovery_on/single_seed_4777/`，包含episodes.csv、coverage_curve.csv、evaluation_metadata.json、hera_summary.json等。
- 完成后图表：`visualizations/recovery_streak{3,2}/recovery_on/single_seed_4777/`。
- 两组均完成并通过校验后：`comparison/recovery_streak3_vs_streak2_seed4777/`包含配对地图表与汇总。

publication_complete.json仅在地图数量/集合、seed、协议、checkpoint未变化、1133个起点与原ARiADNE参考记录相同及无回合error等检查通过后生成。完整结果目录在测试结束前尚未发布，请通过实时日志或staging进度查看。

两个版本是分别训练的模型，权重及训练轨迹不同；该对照不等价于同一checkpoint只切换测试连续门槛。
