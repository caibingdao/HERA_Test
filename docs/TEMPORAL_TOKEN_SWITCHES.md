# Mamba 3.1 History Token 与 Recovery 开关

## Token 定义

本副本的原始 Mamba token 固定为 21 维，内容顺序为：

    局部运动 dx、dy、dtheta (3)
    + 当前实际目标节点特征 (16)
    + reward (1)
    + coverage 增量 (1)
    = 21

局部运动由同一个已执行 transition 的前后位姿计算：dx、dy 转到机器人局部坐标系后按位置尺度归一化，dtheta 按 π 归一化。目标节点特征来自 Attention 图编码后的实际执行目标节点，并经 Linear(128, 16) 投射。reward 按既有尺度归一化；coverage 增量保留原始比例值。

预处理及投射关系为：

    LayerNorm(21)
    → Linear(21, 128)
    → SiLU
    → Linear(128, 128)
    → LayerNorm(128)
    → Mamba (d_model=128)

全图池化特征和 frontier 数量增量不再进入 token。frontier 增量仍可留在轨迹和诊断数据中，供 recovery 与评估使用。恢复动作选出的真实目标特征照常写入历史 token。

## Token 消融开关

| 分组 | 内容 | 环境变量 | 默认值 |
|---|---|---|---:|
| 目标特征 | 目标节点 16 维投射特征 | ARIADNE_TOKEN_FEATURES / ARIADNE_TOKEN_ACTION_FEATURES | 1 |
| 行为与奖励 | 局部 dx、dy、dtheta 和 reward | ARIADNE_TOKEN_BEHAVIOR_REWARD | 1 |
| 地图增量 | coverage 增量 | ARIADNE_TOKEN_MAP_DELTA | 1 |
| 局部运动 | 仅局部 dx、dy、dtheta | ARIADNE_TOKEN_MOTION_DELTA | 1 |
| reward | 独立 reward mask | ARIADNE_TOKEN_REWARD | 1 |

关闭某组只将对应维度置零，21 维宽度和模型参数量保持不变。frontier 不属于这个 token 布局，USE_HISTORY_FRONTIER_DELTA_IN_TOKEN 固定为 False。

布尔值支持 1/0、true/false、yes/no、on/off；其他值会在启动时立即报错。ARIADNE_RUN_NAME 用来隔离模型、TensorBoard 与 GIF 输出目录。

## Checkpoint 兼容性

checkpoint 会保存 local_motion_target_reward_coverage_v1 布局标记和 21 维 raw input 宽度。原 150 维 token checkpoint 与本副本的投射层形状、位姿语义均不同，不能直接恢复；应从随机初始化重新训练。加载不兼容 checkpoint 时，结构配置校验会在载入网络权重前报错。

## 运行示例

完整 21 维 token：

    ARIADNE_RUN_NAME=token21_full python driver.py

仅保留运动、reward 和 coverage 增量：

    ARIADNE_RUN_NAME=token21_no_target ARIADNE_TOKEN_FEATURES=0 python driver.py

## Recovery

Recovery V4 继续在 Worker 的策略动作执行边界介入。启发式恢复目标及其实际执行动作仍会记录到 replay；Recovery 开关、逃逸目标选择与 SAC 更新路径保持原有逻辑。
