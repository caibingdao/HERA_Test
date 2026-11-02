# results/logs — 评测运行日志

本目录保存 9 个 HERA 测试实验的评测运行日志（每个实验一份）：

- `recovery_streak2_seed4777_20261025.log`
- `recovery_streak3_seed4777_20261025.log`
- `recovery_streak2_12k_seed4777_20261025.log`
- `recovery_streak3_12k_seed4777_20261025.log`
- `ablation_no_feedback_7500_seed4777_20261026.log`
- `ablation_no_motion_7500_seed4777_20261026.log`
- `ablation_no_target_7500_seed4777_20261026.log`
- `ablation_no_recovery_10000_seed4777_20261026.log`
- `ablation_no_target_7000_seed4777_20261101.log`

最后两份（no_recovery 10000 与 no_target 7000）来自对应实验 `_control/` 下的自动重试/尝试日志；日志开头包含启动配置，末尾有 `HERA_EVALUATION_COMPLETE` 汇总行。其余 7 份来自评测器的标准日志。日志中的绝对路径和 GPU UUID 保留原样，用于溯源。

注意：这些是评测运行日志，不是训练 TensorBoard 或逐步诊断；训练日志/逐步诊断未纳入本分支，见 `paper/docs/EXCLUSIONS.md`。
