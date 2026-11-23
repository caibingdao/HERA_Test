# 历史 Mamba token：绝对位姿与同 transition 局部运动增量

## 范围与审计结论

仅本地修改训练主目录 `ARiADNE_MAMBA_3.1` 和测试部署副本 `ARiADNE_MAMBA_Test/mamba31_map_split`。不修改 baseline，不操作远程，不删除旧实验。正式训练尚未启动；短训练只用于验证程序，不能据此声称性能提升。

修改前真实路径：Worker 在决策前读取环境位置和自维护的 `robot_heading`，保存为 `TrajectoryStep.robot_pose`；执行动作后保存实际所选邻接节点 action、reward、coverage 差及 frontier 数量差。在线路径复用当次注意力图编码，动作结束后生成 token；replay 路径重新编码对应 observation，并按保存的 action 索引提取目标节点。两者通过 HistoryStateEncoder 将绝对 xy 除以 80、heading 环绕后除以 pi。旧 token 的位姿是 pre-action 状态，不是运动变化量。

环境没有独立物理 yaw 或转向仿真：x 对应图像列，y 对应图像行，正 y 在显示中向下。`atan2(dy,dx)` 的正角在显示中顺时针，在数值 xy 基中为标准正方向。因此以下 R(-theta) 变换与代码坐标约定一致。初始 heading 明确为 0；后续 heading 为实际非零位移方向。零位移保持原 heading。不虚构未执行的旋转。这是运动方向代理，不是真实车体姿态传感器。

Agent 负责图构建和节点选择，没有独立执行环境动作；实际动作边界位于 Worker/评估 adapter/渲染驱动。MambaEncoder 的 d_model、attention–Mamba fusion、pointer 和双 critic 输入维度均不需要改变。

## 精确公式

令 H_t 为动作前图的增强节点特征（每节点128D），g_t 为有效节点 masked mean（128D）；a_t 为实际执行的邻接目标索引，e_t=W_a H_t[a_t]+b_a（16D）。r_t 为该次环境返回 reward（包括原有流程对奖励的处理），C/F 为 coverage/frontier count。

旧位姿块：p_abs=[x_t/80, y_t/80, wrap(theta_t)/pi]。

新位姿块：

```text
dx_world = post.x - pre.x
dy_world = post.y - pre.y
dx_local = cos(pre.theta)*dx_world + sin(pre.theta)*dy_world
dy_local = -sin(pre.theta)*dx_world + cos(pre.theta)*dy_world
dtheta = atan2(sin(post.theta-pre.theta), cos(post.theta-pre.theta))
p_delta = [dx_local/S, dy_local/S, dtheta/pi]
S = TEMPORAL_DELTA_POSITION_SCALE，默认 NODE_RESOLUTION = 4
```

两种模式共同：

```text
raw_t = concat(g_t, p_abs 或 p_delta, e_t,
               r_t/20,
               C_(t+1)-C_t,
               sign(dF)*log(1+abs(dF))/log(1+360))
dF = F_(t+1)-F_t
dim(raw_t) = 128+3+16+1+1+1 = 150
token_t = LN128(Linear128(SiLU(Linear150→128(LN150(raw_t)))))
```

保持原有三个语义开关：features 控制 g/e；behavior_reward 控制三维位姿或运动及 reward；map_delta 控制 coverage/frontier。置零发生在各自归一化后、150D LayerNorm 前。原始特征块置零不意味着 LayerNorm 后对应通道仍为零，这是原有开关语义。新位姿对比两组均开启所有三个语义块，只改变位姿编码，并关闭 recovery。

后续增加了细粒度 `ARIADNE_TOKEN_MOTION_DELTA` 开关。关闭时仅将归一化后的 `[dx_local/S, dy_local/S, dtheta/pi]` 三维置零；reward、coverage increment、frontier increment及其余147维保持开启，token仍为150D并使用相同参数量。为避免与既有 `map_delta`（coverage/frontier）实验重名，该消融在目录和checkpoint说明中命名为 `no_motion_delta`。

## 时间对齐与两条数据路径

`execute_pose_transition` 紧贴真实 `Env.step`：读取实际 pre 位置与已有 heading，调用 step，读取真实 post 位置，然后用实际位移更新 heading。recovery 已先选定真正执行的 waypoint/action，因此不是记录被覆盖的 SAC 建议动作。

transition t 的 observation、g/e 为动作前；pre/post、r、dC、dF 均属于执行同一个动作产生的 transition。执行结束后才 append 历史。决策 t 只能看到历史 `[max(0,t-L),t)`；决策 t+1 可看到 t。没有把 post/outcome 输入产生动作 t 的决策。

Replay 保留 `robot_pose=pre_pose` 并新增 `post_pose`，不通过两个历史 pre pose 相减。SequenceBatch/HistorySequenceBatch/UnionHistoryBatch 的 post 字段与 observation/action 等使用相同 padding、切片和设备传输。union 的 L+1 窗口依旧拆成 current/next 两个 L 窗口。terminal transition 同样存储 post 和 outcome，done 保持控制 TD bootstrap；不把下一 episode 的首步接到上一 episode。首个决策历史为空，episode 结束清空在线 cache；Worker 不允许重复复用已经完成的 trajectory。

在线 cache 复用当前 actor 的 g/action feature，执行后调用同一个 HistoryStateEncoder；replay 从相同 observation 和实际 action 重建后调用相同编码器。相同权重、相同精度下专项测试比较 raw/projected token 与 actor 输出。训练后 replay 用更新后的当前网络重新编码是原有设计，不应要求它逐比特等于旧权重曾经缓存的 token。在线 cache 依然按 episode/policy_version 隔离，不支持中途修改网络语义而复用 cache。

legacy replay 可缺少 post，仅 absolute 能使用；delta 明确拒绝缺失 post。新增 finite 检查覆盖 pose、reward、原始/投影 token 与 cache 特征；非法数据会报错，不静默替换 NaN。

## 配置与 checkpoint

环境变量：`ARIADNE_TEMPORAL_POSE_ENCODING=absolute|delta_local`、`ARIADNE_TEMPORAL_DELTA_POSITION_SCALE=4`。全局默认 absolute 保持旧流程；新 delta 实验 JSON 显式启用 delta_local。

checkpoint 顶层保存 `temporal_pose_encoding`、`temporal_token_version`、`temporal_delta_position_scale`。版本分别 `absolute_pose_v1` / `egocentric_delta_pose_v2`。完全缺少三字段的 legacy checkpoint 解释为 absolute；部分缺失或模式/版本自相矛盾报错。版本不匹配、delta 尺度不匹配默认报错。仅显式 `ARIADNE_ALLOW_TEMPORAL_POSE_OVERRIDE=1`（测试 adapter 对应 allow_temporal_pose_override）允许有警告的跨语义加载；受控 paired 实验禁止 override。

测试 adapter 默认从 checkpoint 推断模式，也可显式要求模式以校验。训练主 driver、test_checkpoint、地图导出和部署测试路径均使用校验。部署副本的旧 driver 保持 absolute-only 并拒绝 delta checkpoint；新 delta 训练使用本目录的新入口。直接手写 PyTorch load_state_dict 绕过项目 loader 不可能由这些元数据自动防护。

## 独立受控实验

两个 JSON 仅 pose_encoding 不同。seed=4777、同一 seed42 split 的4530训练地图（1133测试地图）、800000环境步、recovery_off、相同超参数、2000步 warmup，此后每32环境步计划8次更新，总199496 learner updates。每个 episode 后执行累计到期更新。最终固定环境步 checkpoint 为唯一选择规则，不按测试集挑最优模型。每次新建输出目录，存在即拒绝；无 resume、无旧权重初始化。

为实现精确匹配预算，新增同步单 Worker collector，调用已有 SAC 更新函数；不是原始无限异步 Ray 训练驱动。两组彼此严格使用同一训练算法，但不能把与历史异步训练的差异归因于 token。固定环境步时两模型 episode 长度可能不同，因此已完成地图的频次不一定相同；地图全集与选图规则、episode seed 策略一致。checkpoint 保存源文件 hash 和完整配置，配对评估会校验。

在 `ARiADNE_MAMBA_3.1` 目录执行（两个命令可分别在终端运行）：

```bash
PY=/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python
CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/train_pose_ablation.py --config configs/pose_ablation/absolute_pose.json
CUDA_VISIBLE_DEVICES=1 "$PY" -u scripts/train_pose_ablation.py --config configs/pose_ablation/delta_local_pose.json
```

输出分别 `experiments/pose_ablation/absolute_pose/recovery_off/seed_4777` 和 `experiments/pose_ablation/delta_local_pose/recovery_off/seed_4777`，各自有 run_config.json、episodes.csv、train/TensorBoard、model/final_checkpoint.pth。可先追加 `--dry-run` 检查，不创建目录。扩展训练种子时两组同时使用相同 `--seed`。

训练完成后配对评估（全1133测试地图，贪婪，5个相同随机起点 seed）：

```bash
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/evaluate_pose_ablation.py \
  --absolute-run experiments/pose_ablation/absolute_pose/recovery_off/seed_4777 \
  --delta-run experiments/pose_ablation/delta_local_pose/recovery_off/seed_4777 \
  --output-dir experiments/pose_ablation/evaluation_seed4777 --workers 5
```

输出两模型独立 episode 结果、visualization/ 下分别标注模型的图，以及 paired_comparison/{paired_results.csv,map_summary.csv,summary.json}。99% coverage、128步预算；主指标为成功 episode 路径长度，同时报告 completion rate。paired 路径差仅在两模型都完成的相同 map/seed/start 上计算；单边成功/双失败数量单列，不能将失败的短路径视为更优。输出均值、中位数、样本标准差，均值及中位数95% CI 使用按地图聚类 bootstrap（10000次，同地图的全部 seed 一起采样）；同时报告 completion difference CI。步数和决策时间仅辅助。CI 条件于当前训练种子，不代表重新训练的不确定性。没有足够有效地图时 CI 为 null，CSV 失败路径为 NaN。

## 修改文件清单

主目录：temporal_pose.py（新增）；parameter.py；model.py；models/history_encoder.py；models/rollout_history_cache.py；replay_buffer.py；worker.py；driver.py；runner.py；scripts/export_paper_map.py；新增 configs/pose_ablation 两个 JSON；新增 scripts/train_pose_ablation.py、evaluate_pose_ablation.py、compare_pose_ablation.py；新增 tests/test_temporal_pose.py、test_pose_reporting.py；tests/test_single_gpu_optimization.py 修正 Ray GPU 物理编号断言；本文。

部署副本：temporal_pose.py（新增）；model.py；models/history_encoder.py；models/rollout_history_cache.py；replay_buffer.py；driver.py；evaluation_adapters/ariadne.py；scripts/render_failed_map_gifs.py；scripts/render_stochastic_failure_comparisons.py。Agent 和 models/mamba_encoder.py 经审计不需改维度或功能。

## 验证与限制

专项16项覆盖所要求的几何不变性、角度环绕、零位移、真实环境轴向、首步/reset、normal/recovery 实际动作、online/replay、150/128维、actor/双critic/targetcritic、BF16 SAC 更新及 checkpoint 版本。另有2项 paired 报告测试、部署副本6项兼容回归。完整测试命令与最终结果见后续验证记录。

仍需正式从头训练和全量配对评估才能判断性能；不预设 delta 优于 absolute。当前环境的 heading 是运动方向代理；若未来接入有真实 yaw 的机器人，应直接读取 pre/post yaw，而不是继续从平移估计。尺度4对应邻接步长量级，不对异常长 recovery 位移裁剪。GPU 浮点和 Mamba kernel 可能存在非逐比特复现差异，测试使用数值容差。

## 最终验证记录（2026-09-07）

以下命令在主目录执行，PY 为上面的 conda Python：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mamba31-pose-audit-mpl "$PY" -m unittest discover -s tests -q
"$PY" -m unittest discover -s tests -p test_pose_reporting.py -v
"$PY" scripts/train_pose_ablation.py --config configs/pose_ablation/absolute_pose.json --dry-run
"$PY" scripts/train_pose_ablation.py --config configs/pose_ablation/delta_local_pose.json --dry-run
git diff --check -- '*.py'
```

完整 CUDA 回归 **114/114 通过，64.670秒**（包含专项16项和报告2项）；独立报告测试2/2通过；两个 dry-run 均确认4530训练地图与独立输出目录；diff whitespace 检查通过。使用官方 mamba_ssm 2.3.2.post1，真实128步 episode 的 Mamba 前向/反向均 finite。日志 `/tmp/mamba31-pose-full-tests.log`。

部署副本目录执行相同 Python 的 `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mamba31-pose-audit-mpl "$PY" -m unittest discover -s tests -v`：6/6通过，日志 `/tmp/mamba31-pose-eval-tests.log`。

短训练命令（仅临时验证，目录已存在，不能原样重跑覆盖；重跑需换新的临时目录）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/train_pose_ablation.py --config configs/pose_ablation/absolute_pose.json --total-env-steps 2064 --output-dir /tmp/mamba31-pose-smoke.9MYUf7/absolute
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/train_pose_ablation.py --config configs/pose_ablation/delta_local_pose.json --total-env-steps 2064 --output-dir /tmp/mamba31-pose-smoke.9MYUf7/delta
```

两组均完成2064步、17个episode、16次SAC更新，独立保存 final_checkpoint.pth；读取两个 checkpoint 调用 `scripts.evaluate_pose_ablation.validate_pair` 通过，版本分别 absolute_pose_v1 与 egocentric_delta_pose_v2。短训练均未达到完成探索，不具有最终模型性能解释意义。日志位于同一临时父目录下 absolute.log / delta.log。

验证过程中沙箱不可访问 CUDA 导致过环境错误；提权在真实GPU重新验证通过。原 Ray 测试硬编码物理GPU编号0，在选择GPU3时误报，已改为核对 CUDA_VISIBLE_DEVICES 声明的实际编号，未修改资源分配行为。

评估 adapter 实测：两个短训练 checkpoint 自动识别各自语义，均在测试地图110.png、eval/start seed100上执行3步，recovery关闭、无错误、轨迹长度核对误差0。日志 `/tmp/mamba31-pose-adapter-smoke.log`。在部署副本目录执行：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mamba31-pose-audit-mpl "$PY" -c 'from pathlib import Path; from evaluation_adapters.ariadne import ARIADNEAdapter
for key in ("absolute","delta"):
 a=ARIADNEAdapter(checkpoint_path=Path("/tmp/mamba31-pose-smoke.9MYUf7")/key/"model/final_checkpoint.pth",deterministic=True,adapter_args={"enable_recovery":False}); r=a.evaluate_episode(map_path=Path("maps/110.png"),map_id="110.png",eval_seed=100,start_seed=100,episode_id=1,config={"target_coverage":0.99,"max_steps":3}); print(key,a.metadata(),r)
'
```

尚未执行800000步正式训练及1133地图全量评估；这里只确认实现、短训练与评估接入正确，不报告不存在的最终 ablation 结果。
