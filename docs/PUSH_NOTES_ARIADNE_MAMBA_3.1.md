# ariadne_mamba_3.1 推送说明

## 推送目标

将本地分支 `ariadne_mamba_3.1` 首次推送到：

```text
origin/ariadne_mamba_3.1
```

## 本次推送内容

- 从 `ariadne_mamba_1.1_trained` 保留完整代码能力，但移除 checkpoint、TensorBoard 日志、逐轮训练 GIF/PNG、benchmark CSV 和 Python cache；
- 新增局部重复访问检测器；
- 使用当前 graph 的 weighted shortest-path neighborhood 计算 revisit score；
- 使用现有 exploration rate 的 coverage delta 作为低进展 gate；
- 增加连续多步确认，避免单步误触发；
- 新增固定 escape target 和 Dijkstra recovery path；
- recovery 结束后恢复原 policy，并通过较低退出阈值和重新收集证据实现 hysteresis；
- 为 recovery transition 增加 `is_recovery_action` 标志，保留实际执行 action 并维持原 SAC loss；
- 增加逐步诊断、episode 统计、TensorBoard 指标和可选调试可视化；
- 新增 A–E synthetic tests 和 replay marker 测试；
- 增加完整的 3.1 版本变更说明。

## 兼容性与默认行为

`USE_LOCAL_REVISIT_RECOVERY` 默认值为 `False`。未显式开启时，不保存 detector 状态、不接管动作，保持原 ARiADNE+Mamba policy 行为。

本次没有修改：

- ARiADNE encoder/decoder；
- Mamba token 或网络结构；
- SAC loss；
- reward；
- learned policy。

## 验证结果

- 5 个指定 synthetic recovery cases：通过；
- replay recovery marker：通过；
- temporal replay/history、union history、mask 和 history optimization 相关测试：通过；
- 共 19 项本机可运行测试通过；
- Python 语法检查通过；
- `git diff --check` 通过。

完整 Mamba 模型在本机 Windows 环境仍受缺少官方 `mamba_ssm` 限制；建议在 Linux CUDA 环境完成训练级验证。

## 建议提交标题

```text
feat: add local revisit heuristic recovery
```
