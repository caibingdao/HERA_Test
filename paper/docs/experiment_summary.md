# 论文实验材料：seed 4777 / greedy / 原始1133地图测试集

整理日期：2026-10-23。最终模型仓库为 ARiADNE_MAMBA_3.1_TOKEN21。所有表格由原始 episodes.csv 重新计算；保留源目录结构、原始覆盖率曲线、元数据、已存在的汇总及模型/规划器文件。

## 1. 测试配置、环境与指标

测试清单：ARiADNE_MAMBA_Test/mamba31_map_split/experiment/map_splits/map_split_seed_42.json；SHA256：8397d4d1a3b1e6f57fa604b23f59284352a2e6dea6686f55ff62d0f531fcf0c0。划分种子42，训练4530张，测试1133张；划分使用语义占据图分组，具体重复组及划分统计见原始清单。测试种子4777与划分种子、模型训练种子是不同概念。

统一测试模式 single_global_seed_greedy，eval_seeds=[4777]，每图1次，共1133回合；目标覆盖率0.99，最多128次决策，地图顺序由清单规定，模型加载后初始化一次全局seed，不在地图之间重新播种。神经策略动作选择 greedy_argmax。原始评测中的seeded_map_marker_selection起点及逐图起点核查见 audit/start_pairing.csv。

环境：二维占据栅格、栅格0.4米、节点间距4米、激光感知半径16米、frontier栅格0.8米、utility观测范围12.8米。具体实现以打包的 env.py、sensor.py、parameter.py 和各评测 metadata 为准。历史运行元数据记录CUDA设备，但未完整记录当时操作系统和依赖版本；audit/export_environment.json仅为导出时环境，不能当作历史运行环境。

成功率使用原始success字段；覆盖率为final_coverage的回合平均值；路径长度分别报告全部回合及成功回合平均值，单位米；推理耗时是各回合mean_inference_time_ms的平均值，非所有决策加权均值。不同规划器的耗时边界/决策粒度需结合adapter阅读，不直接视作纯网络前向耗时。单次训练、单测试seed的均值不代表多seed统计显著性。

## 2. 空间分支相关对照

列出ARiADNE及Token21 full_test recovery_off/on。ARiADNE与full_test是架构对照；若要把差异严格归因于“空间分支”，还需控制时序分支和其他结构变化，现有两组本身不能独立证明空间分支的因果贡献。

|实验|成功数/1133|成功率%|覆盖率%|全部路径m|成功路径m|平均步数|平均推理ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|ariadne|1114/1133|98.3230|99.4176|273.4914|258.8072|32.5366|6.7196|
|full_test_off|1117/1133|98.5878|99.4034|256.8740|247.2234|40.8597|18.3671|
|full_test_on|1133/1133|100.0000|99.6095|250.4509|250.4509|40.7414|13.9730|

## 3. Mamba token定义消融

full_test采用完整21维token：局部位移与转角3维、目标动作特征投影16维、reward 1维、覆盖率增量1维；delta_frontier不进入该布局，消融将相应分量置零且不改变维数。no_target对应action_features=false，使用7500 episode训练版本；no_feedback对应no_map_feedback（reward/map_delta关闭）；no_motion关闭motion_delta。原始开关、训练配置和checkpoint哈希见sources及audit/selected_protocols.json。序列长度16，投影维度128，Mamba d_state=16、d_conv=4、expand=2；局部运动编码delta_local，位置归一化尺度4米。其余细节以checkpoint评测元数据为准。

7500 episode是no_target的训练预算，不是测试回合数；该版本仍完整评测1133张地图。它与10000 episode模型的训练预算不同，解释token消融时应披露这一差异。

|实验|成功数/1133|成功率%|覆盖率%|全部路径m|成功路径m|平均步数|平均推理ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|full_test_on|1133/1133|100.0000|99.6095|250.4509|250.4509|40.7414|13.9730|
|no_target_7500|1133/1133|100.0000|99.5939|255.0266|255.0266|44.9223|41.0229|
|no_feedback|1133/1133|100.0000|99.6316|251.5369|251.5369|41.6646|14.3209|
|no_motion|1133/1133|100.0000|99.6048|252.0793|252.0793|46.3477|13.5518|

## 4. 启发式逃逸机制

控制器local_revisit_recovery_v4，包含局部重复检测、frontier逃逸和图回退。具体阈值完整保留在元数据的recovery_configuration。

|实验|成功数/1133|成功率%|覆盖率%|全部路径m|成功路径m|平均步数|平均推理ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|full_test_off|1117/1133|98.5878|99.4034|256.8740|247.2234|40.8597|18.3671|
|full_test_on|1133/1133|100.0000|99.6095|250.4509|250.4509|40.7414|13.9730|

逐图成功配对计数：{'off_False_on_False': 0, 'off_False_on_True': 16, 'off_True_on_False': 0, 'off_True_on_True': 1117}。逐图路径差与起点核查见tables/recovery_paired_maps.csv。

关键限制：当前full_test recovery_off与recovery_on引用不同训练目录且checkpoint SHA256不同，因此这组结果是两个checkpoint的恢复配置对照，不能直接写作“同一模型只切换测试时逃逸开关”的纯消融。历史目录中的train_recovery_off_test_on与recovery_off若哈希相同，可另作同checkpoint机制对照，完整列于总表和元数据。

## 5. 其他模型与历史实验

下表收录全部找到且通过1133地图ID、4777单种子、greedy评测模式核验的非utility实验；完全相同的episodes.csv副本只保留一份。包括旧Mamba/token变体、CNN、NBVP、TARE与DSVP。传统规划器的action_selection在all_results.csv保留原文：统一评测模式名称并不表示其内部算法使用神经网络argmax；TARE/DSVP含固定seed的随机规划，不应标为严格greedy神经策略。TARE历史/修复版本分别保留，DSVP为二维移植版，不能当成完整ROS部署。

|实验|成功数/1133|成功率%|覆盖率%|全部路径m|成功路径m|平均步数|平均推理ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|reports/classical_baselines_full_restart_20261019/nbvp|785/1133|69.2851|92.1360|295.0401|237.6237|56.0671|38.6129|
|reports/classical_baselines_full_restart_20261019/tare_local|1069/1133|94.3513|98.5811|235.4879|232.0390|50.0115|25.6281|
|reports/tare_repaired_full_20261020/visualizations/tare_local/default/single_seed_4777/data|1125/1133|99.2939|99.5214|360.2313|357.4714|67.8764|128.9524|
|reports/tare_official_full_20261020/visualizations/tare_local/default/single_seed_4777/data|3/1133|0.2648|30.7208|27.1500|90.6930|3.7105|17.3327|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/train_recovery_off_test_on|1133/1133|100.0000|99.6576|256.8505|256.8505|35.5737|13.7293|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_off|1122/1133|99.0291|99.4797|260.6823|255.0266|36.2268|13.5600|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_on|1133/1133|100.0000|99.6197|252.6265|252.6265|39.8694|13.3516|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/train_recovery_off_test_on|1133/1133|100.0000|99.6958|252.9242|252.9242|33.3548|13.5644|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_off|1132/1133|99.9117|99.6761|253.6745|252.7155|33.4166|13.5277|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_on|1133/1133|100.0000|99.6607|254.2862|254.2862|36.8914|14.2397|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_features_v2|1130/1133|99.7352|99.6275|254.5561|252.1805|34.8914|13.1508|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_map_delta_reward|1133/1133|100.0000|99.7062|253.5055|253.5055|34.4793|13.7988|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/train_recovery_off_test_on|1133/1133|100.0000|99.7120|257.1841|257.1841|29.3892|13.2460|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_off|1133/1133|100.0000|99.7120|257.1841|257.1841|29.3892|13.8105|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_on|1133/1133|100.0000|99.7021|253.9693|253.9693|33.4872|13.5464|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/train_recovery_off_test_on|1133/1133|100.0000|99.7118|254.7111|254.7111|31.2189|13.4805|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_off|1117/1133|98.5878|99.5570|263.7104|251.6210|32.2048|13.5100|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_on|1133/1133|100.0000|99.6926|254.6457|254.6457|33.4846|13.8043|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba_3_1_recovery_v4|1132/1133|99.9117|99.6358|254.9813|254.4316|40.5631|14.0260|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba|1127/1133|99.4704|99.5675|255.4546|251.3160|38.2065|13.7782|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_pure_policy_greedy_ablation/single_seed_greedy_4777|1130/1133|99.7352|99.6157|255.3729|254.0219|40.6531|13.8369|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/train_recovery_off_test_on|1133/1133|100.0000|99.6889|251.1514|251.1514|34.1898|13.3306|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_off|1130/1133|99.7352|99.6311|253.0197|250.6759|34.3822|13.6531|
|ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_on|1133/1133|100.0000|99.7143|256.0301|256.0301|32.7829|13.6968|
|ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_target/recovery_on/single_seed_4777|1133/1133|100.0000|99.5795|246.8354|246.8354|43.6558|14.6549|
|ARiADNE_MAMBA_3.1_TOKEN21/results/data/cnn/recovery_on/single_seed_4777|1007/1133|88.8791|98.0868|340.1250|307.4058|64.9135|10.8346|
|ARiADNE_MAMBA_3.1_TOKEN21/results/data/dsvp/default/single_seed_4777|1133/1133|100.0000|99.6276|270.7497|270.7497|38.7970|379.6116|
|ARiADNE_MAMBA_3.1/experiments/pose_ablation_no_motion_delta_20260908/test_greedy_4777/results|1125/1133|99.2939|99.6228|265.2535|259.8671|37.0106|13.6147|

## 6. 原始数据、完整性与使用方式

sources/保存纳入实验的原始文件及原路径：episodes.csv逐图结果、coverage_curve.csv逐步覆盖率/路径曲线、evaluation_metadata.json协议与checkpoint信息、已有summary和其他同目录产物。另含相关模型checkpoint/规划器spec、训练配置、测试地图1133张、评测器/模型/环境代码快照；训练配置仅作实验背景，不混入其他seed的测试结果。未生成不存在的轨迹或历史环境信息。

tables/all_results.csv为全部合规实验重算总表；tables/recovery_paired_maps.csv为full_test逐图配对；audit/excluded_or_duplicate.json列出本次扫描中不合规或重复条目；audit/start_pairing.csv逐模型核对起点。未纳入随机策略、五seed、自生成地图、indoors地图、smoke/probe和utility结果。MAPDIVIDER/LARGE_SCALE_DRL若未出现在总表，表示本次未发现满足上述协议及完整1133行的evaluation_metadata/episodes结果，不能用其他地图版本补齐。

SHA256SUMS.csv记录每个打包文件的长度与SHA256，可检验完整性。build_bundle.py为本次整理脚本；重新生成需要原始工程与扫描清单，压缩包内可直接使用文档和表格，无需重新运行测试。

## 7. 各实验数据定位

- **reports/classical_baselines_full_restart_20261019/nbvp**：`sources/reports/classical_baselines_full_restart_20261019/nbvp`；checkpoint SHA256 `72e896971d276ed1cec29d2e1fed8fdb41395057ecae0a9af936416b1de6b900`；动作 `deterministic explicit planner`。
- **reports/classical_baselines_full_restart_20261019/tare_local**：`sources/reports/classical_baselines_full_restart_20261019/tare_local`；checkpoint SHA256 `33f6f4959a35c4db168be3b266fd76aa4003df518c311cee6a5a6714be0e913e`；动作 `deterministic explicit planner`。
- **reports/tare_repaired_full_20261020/visualizations/tare_local/default/single_seed_4777/data**：`sources/reports/tare_repaired_full_20261020/visualizations/tare_local/default/single_seed_4777/data`；checkpoint SHA256 `2ac05474db604c930f8cf7db30e9f718aacbb7509783209b88335632b63a4e69`；动作 `official_tare_seeded_randomized_coverage_and_tsp`。
- **reports/tare_official_full_20261020/visualizations/tare_local/default/single_seed_4777/data**：`sources/reports/tare_official_full_20261020/visualizations/tare_local/default/single_seed_4777/data`；checkpoint SHA256 `6dc3e37f650cf9afe5d6bbc16711d0ff834b5254197fc04a9ce372cd9d95d887`；动作 `official_tare_seeded_randomized_coverage_and_tsp`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/train_recovery_off_test_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/train_recovery_off_test_on`；checkpoint SHA256 `4913c5946b84f1ea1ff9872419d5fa26f1970a93922d191983649b65fba58fb8`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_off**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_off`；checkpoint SHA256 `4913c5946b84f1ea1ff9872419d5fa26f1970a93922d191983649b65fba58fb8`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_new_token_final_test_20260908/single_seed_greedy_4777/recovery_on`；checkpoint SHA256 `cfff3b43866c16fbdd04b6a0deb1f534a38f3c40a65e2adc7ccab8212c866be3`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/train_recovery_off_test_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/train_recovery_off_test_on`；checkpoint SHA256 `a0b75055c0c65913591b38520a71e429c7e1d3fec7491fa6698e3509f9b1bdee`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_off**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_off`；checkpoint SHA256 `a0b75055c0c65913591b38520a71e429c7e1d3fec7491fa6698e3509f9b1bdee`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_coverage_reward_greedy_pair_20260913/single_seed_greedy_4777/recovery_on`；checkpoint SHA256 `ef8d4354c7736c5b9246c5257042ac9494ce076a653307d00005b5991dfcec27`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_features_v2**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_features_v2`；checkpoint SHA256 `22d68231e89eed5a23fc03f8da0e97c3b07d7c5f7bb3ea2d47c41fc2a5e9a667`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_map_delta_reward**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_token_ablation_greedy/single_seed_greedy_4777/no_map_delta_reward`；checkpoint SHA256 `940533e8800d5c55d92b712dea7dabe43b18670f3c76207603fd08b7e3e0a61d`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/train_recovery_off_test_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/train_recovery_off_test_on`；checkpoint SHA256 `9eb3df40ffe121ea14ba658c71475080452c130deebc0aef315e321b599eb70f`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_off**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_off`；checkpoint SHA256 `9eb3df40ffe121ea14ba658c71475080452c130deebc0aef315e321b599eb70f`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/target_motion_coverage_reward_greedy_pair_20260915/single_seed_greedy_4777/recovery_on`；checkpoint SHA256 `90a8ea5e600bca04a37031530c9cd2d16c5a6ea8e1fd44d043a56dd01d53448b`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/train_recovery_off_test_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/train_recovery_off_test_on`；checkpoint SHA256 `aa29dd84fc3d5657457a6c630e7ed24c2aaf503cbc08448b8d45b037b49efd5d`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_off**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_off`；checkpoint SHA256 `aa29dd84fc3d5657457a6c630e7ed24c2aaf503cbc08448b8d45b037b49efd5d`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_only_greedy_pair_20260911/single_seed_greedy_4777/recovery_on`；checkpoint SHA256 `56ee41419ec091f2aaf57d9015c120a7a2e08db86c967a418abfa472368278ba`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba_3_1_recovery_v4**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba_3_1_recovery_v4`；checkpoint SHA256 `c9c07d8ee39015c4965365189dfa2c2b8717772b90e35e82444f9b6771ecea2f`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/three_models_single_seed_greedy_4777/ariadne_mamba`；checkpoint SHA256 `5aaedb25811ffb589959d42dd7a7a3fd44e7b8fe518f5b234dacc1ccb1f3c12a`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_pure_policy_greedy_ablation/single_seed_greedy_4777**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/mamba31_pure_policy_greedy_ablation/single_seed_greedy_4777`；checkpoint SHA256 `c9c07d8ee39015c4965365189dfa2c2b8717772b90e35e82444f9b6771ecea2f`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/train_recovery_off_test_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/train_recovery_off_test_on`；checkpoint SHA256 `e60ffbfd048cde6278e6274c3778f093a5556f0602541cfcd1794ef88e1332c3`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_off**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_off`；checkpoint SHA256 `e60ffbfd048cde6278e6274c3778f093a5556f0602541cfcd1794ef88e1332c3`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_on**：`sources/ARiADNE_MAMBA_Test/mamba31_map_split/experiment/results/motion_reward_greedy_pair_20260912/single_seed_greedy_4777/recovery_on`；checkpoint SHA256 `af9da301a15fb5f33d4ec68469ccae29403b2f9c1a36d182f0e48b0e61c93289`；动作 `greedy_argmax`。
- **no_target_7500**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_target_7500/original_map_result/recovery_on/single_seed_4777`；checkpoint SHA256 `b9afc3d96fb3349225e24840e5d105865f41a58aac348e0d7a84bb9a9cc04b92`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_target/recovery_on/single_seed_4777**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_target/recovery_on/single_seed_4777`；checkpoint SHA256 `3a50dbcd2123114f95dffb554081a4b05474a08bcf8831ea7af981b1b2190fad`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_3.1_TOKEN21/results/data/cnn/recovery_on/single_seed_4777**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/cnn/recovery_on/single_seed_4777`；checkpoint SHA256 `5eb11b7d0e54913a051e7a7ecff0f49bfebb73703b5a6bcfd6467c96f9e632ce`；动作 `greedy_argmax`。
- **no_motion**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_motion/recovery_on/single_seed_4777`；checkpoint SHA256 `0a91eab4cba5422e33889a94d3ea2a9cdbe37cf3e34843c12d637c32cd169a59`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_3.1_TOKEN21/results/data/dsvp/default/single_seed_4777**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/dsvp/default/single_seed_4777`；checkpoint SHA256 `4556090c58e429f77601649b07e502f57db192153547914da1831f1948bbef91`；动作 `seeded_dynamic_rrt_official_gain_and_relocation`。
- **ariadne**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/ariadne/single_seed_greedy_4777`；checkpoint SHA256 `6ac1827a3dbdd816febd074817fb3116466002f3deb70474d0da643957db8b19`；动作 `greedy_argmax`。
- **no_feedback**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/no_map_feedback/recovery_on/single_seed_4777`；checkpoint SHA256 `cfab115ce1fff5dc98a0a26fb969ec6071a86acf9c276da27ce8dd087463dcfe`；动作 `greedy_argmax`。
- **full_test_off**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/full_test/recovery_off/single_seed_4777`；checkpoint SHA256 `07b76f906f360ce34030f7f07ba6d95772ca25ad6f9fde1811b5002002111dc8`；动作 `greedy_argmax`。
- **full_test_on**：`sources/ARiADNE_MAMBA_3.1_TOKEN21/results/data/full_test/recovery_on/single_seed_4777`；checkpoint SHA256 `548ecf895f9b0ed85e33f7dcb4edfb27e88df774bfb62260cb407748611faf14`；动作 `greedy_argmax`。
- **ARiADNE_MAMBA_3.1/experiments/pose_ablation_no_motion_delta_20260908/test_greedy_4777/results**：`sources/ARiADNE_MAMBA_3.1/experiments/pose_ablation_no_motion_delta_20260908/test_greedy_4777/results`；checkpoint SHA256 `093453195c0469e234c79283251db197b1f713424ff7d90b8ee475f69d309027`；动作 `greedy_argmax`。
