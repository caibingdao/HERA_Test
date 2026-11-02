# PAPER_CONTEXT — 论文实验部分的阅读索引与写作口径

本文件给写论文实验部分的 AI/作者使用。原则：**HERA 是唯一最终方法**；旧 TOKEN21/full_test 结果只作历史对照；每个数字必须能追溯到本仓库文件。

## 1. 建议阅读顺序

1. `README_PAPER.md` — 本分支包含什么、怎么校验。
2. `README_HERA.md` — 最终方法定义、token 反馈、recovery 门槛。
3. `results/README.md` — 测试协议和 `results/data/` 目录语义。
4. `results/comparison/model_selection_20261026.md` — 四个最终 checkpoint 的选型。
5. `paper/FINAL_MODEL.json` — 当前固定的主模型（写主结果前先确认）。
6. `paper/tables/hera_final_results.csv` — 9 个 HERA 实验的汇总。
7. `paper/tables/model_selection.csv` — 4 个 checkpoint 对照。
8. `paper/tables/baseline_results.csv` — 7 个基线汇总及原始目录。
9. `paper/docs/LIMITATIONS.md` — 写作时必须披露的限制。
10. 需要核对原始行时，再读对应的 `results/data/<experiment>/<recovery>/single_seed_4777/episodes.csv` 和 `hera_summary.json`。

## 2. 论文各节 → 文件映射

| 论文小节 | 主要文件 | 提取内容 / 注意事项 |
|---|---|---|
| 实验环境与任务 | `env.py`、`sensor.py`、`parameter.py`、`paper/maps/README.md`、`results/README.md` | 2D 占据栅格、0.4 m 栅格、4 m 节点间距、16 m 感知、frontier 0.8 m、utility 12.8 m；以源代码为准 |
| 数据划分与测试集 | `paper/maps/map_split_seed_42.json`、`paper/maps/test_maps.csv`、`paper/maps/test_maps/` | split seed 42，训练 4530 / 测试 1133；测试种子 4777 与划分种子不同 |
| 评测协议与指标 | `results/README.md`、`paper/audit/selected_protocols.json`、各实验 `evaluation_metadata.json` | 1133 图、每图 1 次、greedy argmax、目标覆盖 99%、最多 128 步、全局 seed 只初始化一次、seeded map-marker 起点 |
| 主结果 | `paper/tables/hera_final_results.csv`、`results/data/recovery_streak{2,3}/`、`results/comparison/` | HERA 主模型 vs 基线；路径长度分全部回合/成功回合；覆盖率取 `final_coverage` 回合平均 |
| 模型选型/训练预算 | `results/comparison/model_selection_20261026.md`、`paper/tables/model_selection.csv`、`results/runs/*.json` | streak2/3 × 10k/12k；不同训练权重，不是纯测试开关消融 |
| Token 消融 | `results/data/ablation_no_target_7000/`、`ablation_no_target_7500/`、`ablation_no_motion_7500/`、`ablation_no_feedback_7500/` | 四组均 1133 图、seed 4777；no_target 训练 7000/7500，full 训练 10000/12000，必须披露预算差异 |
| Recovery 消融 | `results/data/ablation_no_recovery_10000/`、`results/data/recovery_streak2/`、`local_revisit_recovery.py`、`recovery_artifacts.py` | recovery off 是单独训练的 checkpoint；不能写成“同一模型只关开关” |
| 与经典规划器对比 | `paper/tables/baseline_results.csv`、`paper/baselines/` | NBVP/TARE 为确定性显式规划器，不是神经 greedy；DSVP 为 2D 移植；TARE official 失败版本保留用于透明性 |
| 计算效率 | 各 `evaluation_metadata.json` 的 `mean_inference_time_ms`、`results/comparison/model_selection_20261026.md` | 耗时是回合平均，包含 recovery 检测/规划，不是纯网络前向；不同规划器决策粒度不同 |
| 失败案例分析 | `results/data/<experiment>/.../episodes.csv`、`results/visualizations/` | 按 `success`/`final_coverage` 筛失败图；不要用未记录的轨迹 |
| 方法描述 | `paper/docs/architecture_comparison.md`、`model.py`、`agent.py`、`worker.py`、`replay_buffer.py`、`temporal_pose.py` | 空间注意力分支 + Mamba 时序分支 + 融合 + SAC；21 维 token；episode-aware 回放 |
| 局限性与伦理/可复现性 | `paper/docs/LIMITATIONS.md`、`paper/docs/EXCLUSIONS.md` | 单测试 seed、单训练 seed、不同 checkpoint 对照、第三方地图许可 |

## 3. 写作硬性口径

1. **HERA 是最终方法**。`paper/docs/experiment_summary.md` 是 2026-10-23 的 TOKEN21 时代汇总，只能用于协议、基线、历史 caveats；不能把其中的 `full_test` 数字写成 HERA 最终结果。
2. **不要混用两个主模型候选**。`paper/FINAL_MODEL.json` 当前默认 `recovery_streak2_10k`；`model_selection_20261026.md` 从效率角度推荐 `recovery_streak3_10k`。主结果表必须固定一个。
3. **不同 checkpoint 的 recovery 对照不是纯消融**。streak2 与 streak3、recovery on 与 off 都是分别训练的权重；只能写“两个完整系统的对照”，不能写“同一模型只改门槛/开关”。
4. **单测试 seed 4777、单训练 seed 4777**。没有多训练 seed 重复，不能声称统计显著性；配对地图检验只能描述逐图差异。
5. **训练预算不同**。no_target 7000/7500 与 full/其他 10000/12000 不同；no_target_7000 与 no_target_7500 是两次不同训练。写作时必须列出。
6. **经典基线不是神经 greedy**。NBVP、TARE 的 `action_selection` 为确定性显式规划器；TARE 还含固定 seed 的随机覆盖/TSP。DSVP 是二维移植版，不是完整 ROS 部署。
7. **TARE official 结果接近全失败**（3/1133），保留它是为了透明性；不能用它代表 TARE 的正常性能。TARE repaired 是修复版，但路径长度和推理耗时明显更高。
8. **推理耗时不是纯网络前向**。`mean_inference_time_ms` 包含 recovery 检测/规划等；不同方法决策粒度不同，不能直接当作网络速度对比。
9. **覆盖率与路径长度口径**：覆盖率用 `final_coverage` 回合平均；路径长度同时报告全部回合和成功回合；不要只报成功回合。
10. **不发明数字**。每个数字必须来自 `episodes.csv`、`hera_summary.json`、`baseline_results.csv` 或 `all_results.csv`；引用时写明文件路径和列/字段。

## 4. 原始结果目录语义

```text
results/data/<experiment>/<recovery_mode>/single_seed_4777/
├── episodes.csv              # 1133 行逐图结果；主表重算依据
├── coverage_curve.csv        # 逐步覆盖率/路径曲线
├── evaluation_metadata.json  # 协议、checkpoint、起点、耗时边界
├── hera_summary.json         # 汇总指标（成功数、覆盖率、路径、步数、触发次数）
└── publication_complete.json # 完整性/发布校验
```

`results/comparison/` 是配对地图汇总；`results/runs/` 是每次评测的启动配置与 checkpoint SHA256；`results/logs/` 是 9 个最终评测的原始运行日志（其中 no_recovery 10000 与 no_target 7000 来自 `_control/` 自动重试日志）；`results/cache/` 是地图清单缓存。

## 5. 基线目录语义

`paper/baselines/<method>/` 只保留原始评测产物：`episodes.csv`、`coverage_curve.csv`、`evaluation_metadata.json`，部分含 `summary.json`/`publication_complete.json`。基线汇总和 checkpoint SHA256 见 `paper/tables/baseline_results.csv`；上游代码地址与本地 commit 见 `paper/docs/BASELINES.md`。

## 6. 不要读的内容

- `experiments/**/model/*.pth`、`*.pth.previous`：权重，单文件约 71.5 MB。
- `experiments/**/recovery_diagnostics.jsonl`：逐步诊断，合计约 2.0 GB。
- `experiments/**/train/events.out.tfevents*`：TensorBoard 二进制。
- `experiments/**/train.log`、`_control/*.log`：训练/控制日志。
- `gifs/**`：演示动画。
- 训练地图 4530 张：本分支只放 1133 张测试地图。
- 旧 TOKEN21/full_test 的原始目录：历史版本，容易与 HERA 最终结果混淆。

## 7. 已知文档不一致

- `README_HERA.md` 中有一句“本目录的模型尚未使用新反馈定义重新训练”，写于 2026-10-23；但 `HERA/experiments/*/run_config.json` 显示正式训练为 `from_scratch` 且 `exploration_progress_signal = new_free_cell_count_v1`（2026-10-24 起）。写作时以后者为准，并在论文中说明。
- `baseline.json` 选择 `recovery_streak2_10k`（最短路径），`model_selection_20261026.md` 综合效率推荐 `recovery_streak3_10k`。这是两个不同目标，不是矛盾数据；主模型必须固定。
