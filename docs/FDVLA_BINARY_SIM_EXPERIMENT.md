# FDVLA 稀疏二值奖励仿真实验协议

日期：2026-08-20（实验协议最近校正：2026-09-03）  
仓库：`/vepfs-mlp2/c20250301/240403026/async_vla/RLinf`  
Git 基线：`a6d8747a0661fe7ca7ff1a9f7a284d2d75594afa`

## 1. 研究问题与边界

本实验只研究：当动作 worker 完全不含本地 VLM，只消费独立冻结 semantic
server 已完成的、可能滞后的 packet 时，原生 PPO 能否用 LIBERO terminal-success
稀疏二值奖励适应真实语义延迟，并改善 wall-clock 学习效率。

这里的“异步”只表示 semantic computation 与 DiT action generation 的物理和时间
解耦，不表示异步优化器或异步 PPO。训练算法保持：

```yaml
loss_type: actor_critic
adv_type: gae
group_size: 1
```

Reward Model、shared-semantic reward、dense/progress/stage reward、step penalty、
GRPO、DAgger 和 simulator privileged state 均不进入训练。现有 shared-semantic
reward 代码未修改。

## 2. 二值奖励数据流

`LiberoEnv.step()` 不采用 LIBERO 返回的数值 reward，只读取 termination：

```text
termination
  -> first_success = termination AND NOT success_once
  -> reward = float(first_success)
  -> rollout buffer
  -> 原生 GAE
  -> 原生 actor_critic PPO
```

因此每个成功 episode 至多一个 `+1`，失败和纯 truncation 全为 `0`。
`env/libero_10_binary.yaml` 显式设置：

```yaml
use_rel_reward: false
reward_coef: 1.0
use_step_penalty: false
ignore_terminations: false
```

`reward.use_reward_model: false` 通过统一 gate 阻止 Reward Worker 启动；运行时
记录 invocation count 为 0。

## 3. 配对配置

Task-0正式比较固定为 GR00T N1.7、16-step action prediction horizon、2-step
execution horizon、4 denoising steps，以及相同 optimizer、batch、rollout、PPO epoch、
KL/clip/gamma/GAE lambda、task/reset/noise seed 和二值环境。C与D不强制使用字节
相同的SFT checkpoint：C checkpoint必须保留原生coupled前向且不构造delay adapter，D
checkpoint必须包含其架构必需的age/history adapter。两者来自记录清楚的预训练lineage，
并在独立selection noise seed、exact age 0、相同48个固定trial上联合选择到相近成功率；
checkpoint绝对路径、SHA256、SFT步数和数据范围均进入冻结manifest。

| 差异 | D-PPO | C-PPO |
| --- | --- | --- |
| execution mode | decoupled | coupled |
| semantic server | 开启 | 关闭 |
| local backbone | 删除 | 保留前向 |
| central cache/publisher | 开启 | 关闭 |
| semantic age | 自然异步或预注册 age | 当前帧；server age 控制禁用 |
| semantic replay assert | 开启 | 不适用 |
| 正式四卡 placement | 每卡一个冻结semantic server与一个DiT/env rank共置 | GPU 0–3各一个coupled rollout/env rank |

两种模式都只更新最后4个DiT block与value head；D额外更新其必要的age/history
adapter。coupled本地VLM存在但全部冻结且不构造delay adapter；decoupled actor/rollout
内VLM参数量为零。启动时硬性审计VLM、DiT、value head参数量和初始化fingerprint。
因此“相同可训练主干范围”指两臂共同的DiT tail-4/value边界，不伪称总可训练参数逐项相同。

## 4. 方法矩阵与预注册比较

`fdvla_binary_run_manifest.yaml` 注册 C-SFT、D-SFT、C-PPO、D-PPO、
D-PPO-Fresh、D-PPO-NoAge 和 D-PPO-NoHistory。主要终点固定为：

1. D-PPO 相对 D-SFT 的最终成功率；
2. D-PPO 相对 C-PPO 的 wall-clock learning-curve AUC；
3. actual semantic age 0、2、4、6、8、12 的成功率退化曲线。

`D-PPO-Fresh`训练时将random-age范围固定为`[0,0]`；评测时同时设置fixed age 0
并将eval random-age显式关闭为`[-1,-1]`。这是必要条件，因为模型在random-age
启用时会优先使用env调度的age；只写fixed age 0并不能保证Fresh。配合fetch target
和hard max age均为0，训练与评测都会阻塞等待当前帧的exact packet。七方法
`DRY_RUN=true`现在输出完整解析后的`PROTOCOL`行，可直接审计only-eval、train/eval
age、Fresh、NoAge和NoHistory开关，而不只显示下游脚本名。

固定 trial 的严格条件键覆盖
`(task_id, trial_id, requested_semantic_age, action_execution_horizon,
policy_noise_seed)`。结果工具报告 Wilson 95%
区间、配对 failure-to-success / success-to-failure、exact McNemar、frame/time AUC
和跨三个 seed 的 bootstrap 95% 区间，不允许丢弃负结果。

## 5. 语义 replay 与评测审计

Decoupled rollout 将实际使用的 `semantic_*` tensor、packet age、action history、
denoising chain 和 packet identity 写入 PPO buffer。actor update 直接读取这些
tensor；缺失或 fingerprint 不一致均硬失败，不重新调用 VLM。

评测 JSONL 除 task/trial/success 外写入 requested age、actual age、
bootstrap-clipped 和 policy-noise condition seed。actual age 从被选 packet 的
真实 age 计算，不以 requested age 代替。episode 开头因
`source_frame=max(0,t-d)` 产生的因果截断单独计数。


### 5.1 事后语义延迟增广（PosthocAug）

数据流为：rollout在动作边界只读取latest completed packet，不等待指定age，也不等待
当前观测的VLM完成；采样后，PosthocAug只在actor update中扫描已经写入PPO buffer的
同env、同episode generation semantic候选。拉大延迟使用更旧的buffer packet；缩小
延迟使用后续rollout row实际携带、但source frame仍不晚于被重配action frame的packet。
若该packet在原动作发生时尚未完成，则显式记录candidate_was_online_available=false；
没有合法候选的row直接置valid=false，不向semantic server补请求，也不等待或重跑VLM。
随后按真实frame差重算actual semantic age，写成PPO eligibility恒为false的Posthoc
buffer view，只供独立log-prob consistency辅助项。actor、critic、advantage、return和
old log-prob都不能把hindsight row当作on-policy样本。该路径不增加冻结VLM forward、
semantic RPC或tensor传输；额外成本仅来自buffer重配以及抽样的actor replay前向/反向，
仍必须单独报告wall-clock和峰值显存。

增广不会伪造age，也不会把requested age写成actual age。每个row按固定seed从
[-4,-2,+2,+4,+8]中选择目标变化：负数请求比rollout实际semantic更新鲜的packet，
正数请求更旧的packet；目标方向没有因果可用候选时，该row的valid mask为false。
候选tensor、attention/image mask、packet identity和semantic fingerprint作为一组
重配，当前state、action history、denoising chain和实际执行action保持不变。

PPO主损失仍只读取rollout当时真正消费的semantic及其old log-probability。增强row的
ppo_eligible与写入forward inputs的rollout_posthoc_ppo_eligible都被硬编码并断言为
false，禁止复用不匹配的behavior probability。可选辅助项只最小化同一denoising
transition在原semantic与增强semantic下的log-prob差，原分支作为stop-gradient target；
critic warmup期间辅助项关闭。两份C/D配置都包含完全相同、默认关闭的配置块，固定
delta grid为[-4,-2,2,4,8]，默认seed为9473，默认辅助权重为0.0。

正式的增广效果比较使用D-PPO-PosthocControl：它与PosthocAug完全相同地从rollout
buffer重配semantic，并在相同微批序号执行第二次log-prob前向，但辅助权重固定为0；
两臂均禁止boundary fresh-bank、exact-age训练以及actor update期间的server/VLM调用。
PosthocAug开发默认权重为0.01。为避免全buffer二次前向吞没解耦收益，
正式paired pilot预注册每8个微批重放1个。三个FSDP rank按相同全局微批序号确定性选择，
实际覆盖率必须为0.125；选中微批的辅助损失乘8，作为全buffer一致性目标的无偏随机估计。
Control执行相同的第二前向，但零权重分支不挂接辅助反向图，因为扩散尾部的非有限值即使
乘零也可能污染梯度。因此它是严格的“等额外前向/数据流control”，不是逐算子完全相同的
反向成本control；两臂仍分别报告actor update wall-clock。

开发测量表明，全覆盖版本在60个env、每卡2560个时序样本时，每卡产生1280次额外增广
微批前向，首轮actor update为332.5秒，因此被拒绝作为高效正式实现。最终采用1/8确定性
replay。2026-08-30的四卡Task-0配对smoke中，阻塞exact-age v9的10步中位rollout为
248.31秒、semantic fetch均值中位数为451.95毫秒；改为latest-only/buffer-only后，
PosthocControl rollout为101.86秒、fetch均值29.37毫秒，分别缩短58.98%和93.50%，
总step由v9中位507.21秒降至355.64秒（29.88%）。VLM queue latency仍为749.62毫秒，
说明VLM计算本身没有变快，而是被移出动作关键路径。PosthocAug rollout为102.03秒，
与Control相差0.17%；actor update分别为248.17秒和256.24秒。Control/Aug有效重配率
分别为0.893/0.896，fresher为0.325/0.329，older均为0.568，PPO eligible和nonfinite
count均为0；Aug加权辅助损失为8.74e-4。该smoke验证实现与效率契约，不构成成功率提升
结论。

普通D-PPO与PosthocControl的比较只用于量化buffer重配和replay前向成本，不能当作
纯学习消融；两者的冻结VLM forward与semantic RPC契约相同。论文中必须称其为“事后语义延迟增广的辅助/离线视图”，
不得称为额外on-policy PPO样本，也不得把它描述为异步PPO。

## 6. 可复现运行命令

先提供共同 checkpoint 的绝对路径：

```bash
export GR00T_MODEL_PATH=/absolute/path/to/common/checkpoint
export GR00T_BACKBONE_PATH=/absolute/path/to/frozen/backbone
```

查看但不启动：

```bash
METHOD=D-PPO STAGE=smoke DRY_RUN=true \
  bash examples/embodiment/run_fdvla_sim_matrix.sh
```

Task 0 两步 smoke：

```bash
METHOD=D-PPO STAGE=smoke DRY_RUN=false \
  bash examples/embodiment/run_fdvla_sim_matrix.sh
METHOD=C-PPO STAGE=smoke DRY_RUN=false \
  bash examples/embodiment/run_fdvla_sim_matrix.sh
```

Task 0 pilot 的单个预注册方法：

```bash
METHOD=D-PPO STAGE=pilot DRY_RUN=false \
  TRAIN_SEED=0 bash examples/embodiment/run_fdvla_sim_matrix.sh
```

正式 matrix launcher 默认每10个 update 固定评测并保存一次 checkpoint。D-method
pilot 的中间评测固定 requested age=3；训练 age 仍随机0--6。直接调用底层 runner
可用于开发，但不自动继承这组 formal-pilot 选择协议。

正式 checkpoint delay sweep：

```bash
POLICY_METHOD=D-PPO PPO_CKPT_PATH=/absolute/path/to/full_weights.pt DRY_RUN=false \
  bash examples/embodiment/eval_fdvla_delay_sweep.sh
POLICY_METHOD=D-SFT DRY_RUN=false \
  bash examples/embodiment/eval_fdvla_delay_sweep.sh
```

从一个或多个训练 run 自动生成固定评测 learning-curve CSV：

```bash
python examples/analysis/collect_fdvla_learning_curve.py \
  --run D-PPO:0=logs/20260821-06:10:05-libero_10_fdvla_binary_ppo \
  --output /absolute/external/results/fdvla_learning_curves.csv
```

成功率严格来自每个 `eval_trials_step_*.jsonl`，而不是可能受历史 trajectory
筛选影响的聚合标量。`environment_frames` 是每次固定评测前各训练 update 实际执行
control frame 的累计和；`wallclock_s` 从 step-0 固定评测完成时计零，并包括后续
rollout、actor update、weight sync 和中间评测。工具同时写出该定义的 metadata
JSON，多个方法/seed 可重复传入 `--run METHOD:SEED=RUN_DIR`。

工具同时审计各 checkpoint 相对 step 0 的 trial identity、requested semantic age
和 action horizon，并对不含 outcome 的
`(task_id, trial_id, policy_noise_seed, requested_semantic_age, action_execution_horizon)`
有序集合写入稳定 SHA256。正式曲线要求每个 run 的
`pairing_audit.formal_pairing=true`，且 C-PPO/D-PPO 全部 seed 的条件集合条数与
SHA256 完全一致。旧 runner 曾将
`eval_trials_step_10.jsonl` 对应的 TensorBoard eval 标量写在 step 9；当前源码已
统一为 completed-update step，汇总器对旧 run 显式写
`eval_step_alignment=legacy_step_minus_one`，不会静默错位。

三seed正式终点与学习曲线汇总使用：

```bash
python examples/analysis/summarize_fdvla_results.py \
  --eval D-SFT=/absolute/final/d_sft.jsonl \
  --eval C-PPO:0=/absolute/final/c_ppo_seed0.jsonl \
  --eval C-PPO:1=/absolute/final/c_ppo_seed1.jsonl \
  --eval C-PPO:2=/absolute/final/c_ppo_seed2.jsonl \
  --eval D-PPO:0=/absolute/final/d_ppo_seed0.jsonl \
  --eval D-PPO:1=/absolute/final/d_ppo_seed1.jsonl \
  --eval D-PPO:2=/absolute/final/d_ppo_seed2.jsonl \
  --learning-curves /absolute/external/results/fdvla_learning_curves.csv \
  --output-dir /absolute/external/results/fdvla_primary_summary
```

默认正式统计要求D-SFT一个共享冻结策略评测，C-PPO与D-PPO各恰好三个相同seed，
每个最终评测至少400个唯一trial，并且所有方法使用相同requested age、action horizon
和trial/noise身份。学习曲线必须包含相同三seed、每条至少两个严格递增点、从0 frame和
0 wall-clock开始，并自动读取collector生成的`.metadata.json`确认每个run的固定条件
配对成立及所有方法/seed的条件身份指纹完全相同。`D-PPO - C-PPO` AUC在每个seed两条曲线实际覆盖的共同wall-clock/frame
区间上插值积分，同时报告原始与按共同区间宽度归一化的AUC差，再对三个seed做
bootstrap 95%区间。重复`METHOD:SEED`、少于400 trial、混合condition、seed集合不匹配
或非formal学习曲线都会拒绝正式汇总。开发子集必须显式使用`--allow-incomplete`，其
`formal_statistical_summary_complete`恒为false；condition不匹配还必须额外显式传入
`--allow-condition-mismatch`。

单条训练 run 的机器可读健康度摘要：

```bash
python examples/analysis/summarize_fdvla_training_run.py \
  --run-dir logs/20260821-06:10:05-libero_10_fdvla_binary_ppo \
  --output-dir /absolute/external/results/task0_seed0_health
```

输出长表 scalar CSV 和 health JSON；后者汇总实际 control frames、frames/s、
episodes/hour、成功/失败 episode、PPO 有限性、local-VLM forward、峰值 rollout
显存、Reward Model 调用、semantic replay fingerprint mismatch 和
`cross_episode_packet_mismatch_count`。fresh process 会逐 fetch 比较 server 返回
的 episode generation 与 actor 请求；任何不一致都会先累计再硬失败，正常运行的
窗口指标必须恒为0。它是健康度与系统效率审计，不以训练 rollout success 代替固定
trial 评测。

每次启动自动保存 resolved config、Git SHA/status、初始 checkpoint和backbone的
路径/SHA256、可选 PPO checkpoint 路径/SHA256、artifact哈希算法标识和GPU硬件信息
到被 Git 忽略的`logs/fdvla_metadata/`。

## 7. 信息论离线分析

信息数据必须写到 Git 工作树以外。默认 dry-run 可先检查七个 age 的完整命令：

```bash
FDVLA_INFORMATION_OUTPUT_ROOT=/absolute/external/fdvla_information \
  DRY_RUN=true bash examples/analysis/run_fdvla_information_collection.sh
```

实际运行时，eval worker 在每个 action boundary 导出策略实际读取的 semantic
tensor/mask/source frame/version/actual age、当前图像与 state/history；raw NPZ 不含
reward、return、advantage 或 teacher action。episode 完成后才追加 outcome。独立离线
进程随后加载冻结 coupled teacher，在同步当前图像上生成 `A_t^sync`，只写作 probe
标签；metadata 保存 teacher checkpoint 绝对路径和 SHA256。

`collect_fdvla_information_dataset.py` 校验
`semantic_source_frame_id == frame_id - actual_age`，并按
`(task_id, init_state_id)` 分组切分，防止初始状态泄漏。
`train_fdvla_information_probe.py` 在训练集 teacher action chunk 上建立固定
K-means codebook，并训练容量匹配的 baseline/semantic probe 及时间打乱、跨 trial、
置零、task-only、state-history 负对照。时间打乱与跨 trial semantic 都被硬性限制在
相同 `(split, task_id)` 内，不能从 validation/test 泄漏到训练集；所有容量匹配 probe
使用相同初始化 seed。语言融合前 early visual feature 保持可选，不侵入主路径。

报告量为 held-out cross entropy 差：

```text
I_hat(d) = [CE(q0) - CE(qd)] / ln(2)
R(d) = max(I_hat(d), 0) / max(I_hat(0), epsilon)
```

它只能称为“条件任务信息的变分估计”或 “variational predictive-information
estimate”，不能称为精确 Shannon mutual information。VLM 冻结意味着 PPO
不会增加 semantic 本身的信息量；成功率改善最多说明策略更有效利用已有残余信息。

## 8. 当前 smoke 记录

共同初始 checkpoint：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/checkpoints/GR00T-N1.7-LIBERO/libero_spatial
canonical directory SHA256: 5c96b44e0cdb6b2804f67948385cc0bb7409efc218edf732cfb994bf5d346b04
legacy run-metadata SHA256: 0bb3ccdbfce7bc39b35dc88b5200e6258010da396c71bba90ded785157bfc0e9
```

D-PPO 两步 smoke：

- 日志：`logs/20260820-15:26:12-libero_10_fdvla_binary_ppo`；
- 两次 PPO update 完成，policy/value/ratio/KL/gradient/advantage/return 均有限；
- 32-frame 短 episode 未成功，观测 reward/return 全为 0，符合二值协议；
- decoupled actor/rollout 本地 VLM 参数为 0；
- DiT 可训练参数 1,092,186,496，value head 4,327,425；
- actor/rollout 初始化 fingerprint 均为
  `0d3b8e078a6c431bd7bd9a96dac91f1e33a9b43ef9c9d590be45e334ba05a8b3`；
- semantic replay fingerprint 未触发 mismatch 硬错误；
- Reward Worker 未启动，invocation count 为 0。

step 2 checkpoint：

```text
logs/20260820-15:26:12-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt
SHA256: a32e243087e71ed51e9751ee6ac10528a50c523c76c42aa40b681991eeae327d
```

与初始化权重的只读比较发现 456 个共同 DiT key；固定抽样前 64 个中 58 个发生
变化，最大绝对差 `2.384e-7`。checkpoint 中没有 backbone key。

fresh process 从该 checkpoint 成功加载并完成两个固定 Task 0 trial：

```text
logs/20260820-15:37:56-libero_10_fdvla_binary_ppo
```

两个 JSONL 记录均为 requested age 0、actual age 0、bootstrap false、
policy noise seed 1234；成功数为 0/2。该短评测只验证加载与审计契约，不是科学结果。

C-PPO 两步 smoke：

- 日志：`logs/20260820-15:46:25-libero_10_coupled_n1d7_binary_ppo`；
- 两次 PPO update、训练前与 step 2 评测均完成，指标有限；
- VLM 总参数 1,523,500,032、可训练参数 0；
- DiT 可训练参数 1,092,186,496，value head 4,327,425；
- 与 D-PPO 的初始化 fingerprint 完全相同；
- step 0/2 的四条评测记录均为 requested/actual age 0、bootstrap false、
  policy noise seed 1234；
- Reward Worker 未启动，invocation count 为 0；
- checkpoint SHA256 为
  `ed4aab66319ccf1ad649b1b7d474ad604f212b867e30ba818b6c57db897a1959`。

checkpoint 与初始权重的逐 tensor 只读比较显示：494 个共同 backbone tensor
全部 bitwise 不变；固定抽样的 64 个 DiT tensor 中 60 个变化，最大绝对差
`2.384e-7`。这同时验证 coupled 保留本地 VLM 前向但只更新 DiT/value 边界。

D-PPO-Fresh修复后的两步smoke：

```text
logs/20260821-1115-fdvla-fresh-smoke
```

- 使用与Task 0长跑相同的cached-semantic SFT初始化；当前canonical目录SHA256为
  `8154f8bde6480918e3b82595d41cbafedd5d032a83893e30c090ffcf2852c3ca`。
  该历史run metadata中的legacy SHA256为
  `37a258536859190bbf6c8afbc6a8d6527094cd089ef86f10553913d35fb0d7b8`；
- resolved config为train random age `[0,0]`、eval fixed age 0、eval random age
  `[-1,-1]`、fetch target/hard max age均为0；
- 完成2次PPO update、320个control frame和4个短episode；4个episode都未成功，
  reward均为0，未观察到二值协议之外的值；
- train和step-0/step-2 eval的每个实际动作boundary均记录requested age 0、actual
  age 0；eval JSONL的bootstrap、requested/actual mismatch均为0；
- 所有必需TensorBoard标量有限，semantic replay fingerprint mismatch为0，
  cross-episode packet mismatch为0，local VLM forward为0，Reward Model invocation为0；
- actor/rollout VLM参数均为0；初始fingerprint相同。初始SFT与step-2 checkpoint的
  64个确定性共有DiT小张量样本中55个变化，最大绝对差`2.384e-7`；
- 观测aggregate control throughput为9.61 frame/s，peak rollout GPU memory为
  3180.60 MiB。这是2-env smoke profiling，不用于方法间效率结论。

step-2 checkpoint：

```text
logs/20260821-1115-fdvla-fresh-smoke/
fdvla_binary_d_ppo_fresh_task0_smoke/checkpoints/global_step_2/
actor/model_state_dict/full_weights.pt
SHA256: 3a6220cdc8f5a0ab4697081d22453868750bde79599cd1ff615b46eab3d23c6b
```

独立fresh process从上述SHA checkpoint成功加载，并在新Ray进程和新semantic server
端口完成2个固定Task 0 trial：

```text
logs/20260821-1120-fdvla-fresh-step2-load
```

两条JSONL均为requested/actual age 0、actual min/mean/max均为0、bootstrap 0、
age mismatch 0、cross-episode mismatch 0、local VLM forward 0。该评测记录的
action-boundary blocking p50/p95为144.81/215.73 ms，直接体现Fresh等待当前帧
semantic的代价。中央latest-cache路径会先记录一次stale fallback警告，但随后
`fetch_exact`取得当前帧；进入策略和JSONL审计的最终semantic均为actual age 0。
2-trial成功率为0只反映32-frame短smoke，不能作为性能结论。

信息采集链路另完成一次 2-env、32-frame、fixed-age-0 eval-only smoke：

- raw 输出位于 `/tmp/fdvla_information_export_smoke`，两个完整 episode NPZ；
- 每个 episode 两个 action boundary，semantic shape 为 `(2, 160, 2048)`；
- actual age 全为 0，source-frame 一致性校验通过；
- 冻结 coupled teacher 在独立进程成功生成 4 个 `(16, 7)` action 标签，全部有限；
- labelled 输出为 `/tmp/fdvla_information_label_smoke.npz`，未写入 Git。

同一信息采集 smoke 的 semantic server 文本日志可由
`summarize_fdvla_system_profile.py` 机器汇总。64 个 control frame 中记录 3 次 VLM
forward（46.875 次/1000 frame）；含首次 warm-up 的 forward p50/p95 为
52.60/843.38 ms，queue latency p50/p95 为 75.24/897.48 ms。这些只是 profiling
链路验证，不用于 coupled/decoupled 性能比较。

后续 run 的 TensorBoard 还会按统计窗口写入 semantic fetch、action-boundary
blocking、扣除 semantic 的 DiT generation、semantic queue、semantic age 的
mean/p50/p95/count，以及 local VLM forward count、control frame count 和 rollout
GPU peak memory。RLinf 原生 timer/训练指标继续记录 rollout、actor update、weight
sync、evaluation、KL/clip/gradient/value/advantage/return。

在 `logs/20260820-16:44:57-libero_10_fdvla_binary_ppo` 的 2-env、32-frame
D-SFT eval-only profiling smoke 中，上述 TensorBoard 标签已实际写入：action
generation p50/p95 为 368.22/547.09 ms，semantic fetch p50/p95 为
167.84/247.84 ms，扣除 semantic 后的 DiT p50/p95 为 200.38/299.25 ms，rollout
GPU peak memory 为 3154.58 MiB。action-boundary blocking 明确定义为动作关键路径
上的完整 semantic 路径（包括 exact-age 等待或 coupled 本地 VLM），并由单元测试
验证；这些数值同样只证明记录链路可用，不是正式方法间效率结果。

## 9. 当前结论与未运行项

配对的 C-PPO 与 D-PPO 已完成二值奖励两步 smoke；D-PPO 的独立 semantic server、
rollout semantic replay、参数冻结、checkpoint 保存和 fresh-process 加载已形成
Task 0 最小闭环。

一次使用错误初始化的严格二值 50-update 开发运行位于：

```text
logs/20260821-01:32:01-libero_10_fdvla_binary_ppo
```

它把原始 `libero_spatial` checkpoint 直接用于 LIBERO-10 Task 0。step
0/10/20/30/40/50 的固定评测全部为 0/48，训练期间 `env/success_once`、
`env/return` 和 `env/reward` 也始终为 0。这不是“二值奖励算法失败”的证据，而是
初始化/任务不匹配导致全程没有 terminal-success 策略信号；该负结果保留且不删除。

作为历史对照，旧 decoupled env-reward 长跑
`logs/20260808-15:29:43-libero_spatial_ditonly_ppo_semantic_server_clean`
明确设置 `reward.use_reward_model=false`、`use_rel_reward=true`、
`reward_coef=1.0`、无 step penalty，并保持原生 actor-critic/GAE/group-size-1。
环境封装调用 `self.env.step(actions)` 后把 simulator 数值 `_reward` 明确丢弃，
再由 termination 重建奖励；因此它没有使用 LIBERO 数值 dense reward，而是对
termination 指示量做时间差分；成功
标志单调时近似首次成功 `+1`，但仍不满足本实验更强的 first-success 硬门控契约。
该运行用 `libero_spatial` checkpoint 训练同一 `libero_spatial` suite，训练窗口
success 从5.0%上升至90.0%（79个已记录 return update，最大93.27%）。另一条
`logs/20260808-09:32:19-libero_spatial_ditonly_ppo_semantic_server_clean` 从7.41%
上升至88.98%（67个已记录 return update，最大96.97%）。两者均为
`global_batch_size=960`、`rollout_epoch=1`，而不是当前 pilot 的384和4。这说明
旧运行具备初始正 episode 且 checkpoint
与任务匹配；不能将其可训练性归因于 dense reward。

旧运行的 eval logger 只写出成功完成的 trajectory，因而其少量 `eval=100%` 记录
不是固定 trial 成功率，不能用于与当前48-trial配对评测比较。

正确的共同初始化改为 LIBERO-10 cached-semantic SFT checkpoint：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/
libero10_decoupled_sft_0to8_rebuilt/
S_libero10_decoupled_0to8_scalar_age_s4_a4/
libero10_delay0to8_s4_a4_4gpu/checkpoint-5000
canonical directory SHA256: 8154f8bde6480918e3b82595d41cbafedd5d032a83893e30c090ffcf2852c3ca
legacy run-metadata SHA256: 37a258536859190bbf6c8afbc6a8d6527094cd089ef86f10553913d35fb0d7b8
```

该冻结 D-SFT 策略在48个固定 Task 0 trial、随机 exact age 0--6、16-step
execution horizon、policy-noise seed 2026 下得到 40/48（83.33%）成功。此结果证明
初始化在当前二值协议下能产生足够正 episode，但尚不证明 PPO 有改进。

用户授权的 D-PPO seed-0 50-update pilot 于 2026-08-21 06:09 UTC 启动，并于
10:13 UTC 完成：

```text
logs/20260821-06:10:05-libero_10_fdvla_binary_ppo
logs/fdvla_metadata/20260821T060900Z_D-PPO
```

50次 PPO update 共执行1,134,080个实际 control frame，完成9,062个 episode，
7,023成功、2,039失败；聚合吞吐78.70 control frames/s，完成 episode 吞吐
2,263.93/hour。所有已记录 log-probability、advantage、return 和 loss 均有限；
local VLM forward、Reward Model 调用、semantic replay fingerprint mismatch 均为0。
rollout窗口成功率均值为77.46%，范围70.45%--83.60%；该统计与固定trial评测分开
报告。

开发评测曲线为：

| update | 实际 control frames | step-0后 wall-clock | success | Wilson 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 s | 40/48 = 83.33% | 70.42%--91.30% |
| 10 | 232,960 | 2,898.05 s | 42/48 = 87.50% | 75.30%--94.14% |
| 20 | 458,240 | 5,731.50 s | 41/48 = 85.42% | 72.83%--92.75% |
| 30 | 683,520 | 8,547.42 s | 41/48 = 85.42% | 72.83%--92.75% |
| 40 | 908,800 | 11,462.71 s | 42/48 = 87.50% | 75.30%--94.14% |
| 50 | 1,134,080 | 14,397.88 s | 43/48 = 89.58% | 77.83%--95.47% |

终点相对初始化增加3个成功trial，即绝对+6.25个百分点；开发性配对记录为
failure-to-success=3、success-to-failure=0、exact McNemar p=0.25。该变化是积极
开发信号，但尚不显著，也不是正式配对结论：终点48个trial中有20个requested和
actual age与step 0不同。原因是该已启动进程仍加载旧 random-age evaluator，
它按 stage action-boundary stream 取样，策略变化后同一 trial 不保证落在同一 stream
位置。因此所有step-0对后续checkpoint的比较均标记 `formal_pairing=false`，只能
作为开发信号，不能声称 PPO 已显著提升，也不能进入正式 learning-curve AUC。
旧进程还把step 10--50 JSONL的eval标量写在TensorBoard step 9--49；汇总器已显式标记
`legacy_step_minus_one`，当前源码的后续fresh run已修正。

最终权重为：

```text
logs/20260821-06:10:05-libero_10_fdvla_binary_ppo/
fdvla_binary_d_ppo_cached_sft_task0_seed0/checkpoints/global_step_50/
actor/model_state_dict/full_weights.pt
SHA256: b9687d28c68d0e9ca1517b2d28a16bed7fa72c18a606484e540729d5b03d9dfe
size: 3,276,766,913 bytes
```

训练进程结束后，使用全新Python、Ray、semantic server和DiT-only worker加载该
checkpoint，并在Task 0、2个fixed trial、exact age 0下完成2/2成功：

```text
logs/20260821-1015-fdvla-task0-seed0-fresh-load
```

fresh-process启动日志确认decoupled execution、无本地VLM、VLM总参数和可训练参数
均为0、checkpoint override成功加载；requested/actual age mismatch为0，
cross-episode packet mismatch为0，local VLM forward为0，Reward Model关闭。

未来由 matrix launcher 启动的 D-method pilot 保留训练 age 0--6，但中间固定评测
预注册为 requested age 3，并禁用 eval random-age，从而使所有 checkpoint 的
`(task, trial, requested age, horizon, noise)` 完全一致。随机 age 0--6 与自然异步
调度保留为独立部署评测；最终 fixed-age sweep 仍按0/2/4/6/8/12等条件严格配对。

已经完成少量单 seed、50-trial 的 D-PPO-vs-D-SFT 固定 age/action-horizon
开发性配对点，但尚未完成其他训练 seed、C-PPO 等配对 pilot、独立最终
400-trial 评测、完整 LIBERO-10、正式40-cell实时性曲面或完整 information probe。
本轮仍没有可用于论文主结论的三 seed D-PPO-vs-D-SFT 或 D-PPO-vs-C-PPO
learning curve；现有开发点不能替代正式曲线与预注册终点评测。

另有一组Task 0、K=8、60 train env、48 fixed eval trial、各50 update的单seed
C/D开发配对，位于
logs/fdvla_matched_start/20260825_task0_k8_formal_3seed_lr1e8_warmup100_v1/
seed0/ppo_compare/。

两臂都使用307.2万control frame、global batch 384和相同trial/noise集合。C-PPO从
15/48上升到19/48，D-PPO从19/48上升到21/48；因此两臂都呈净上升，但48-trial波动
仍很大。C总用时17724.27秒，D总用时15854.70秒；按相同frame预算，D wall-clock少
10.55%，聚合吞吐从173.32增至193.76 frame/s（+11.79%）。rollout worker峰值显存
从7097.59 MiB降至3343.90 MiB（-52.89%）。

这组结果只能作为支持性开发证据：C/D分别使用独立的matched-start SFT checkpoint
（SHA256分别为4d9297...和d3b4ca...），step 0并不相同；coupled当前帧与decoupled
自然异步的semantic condition本来就不同，跨方法condition SHA256不相同；D的step 40
还有1条actual-age不配对。因此其学习曲线被审计为非formal，不能据此宣称统计显著、
不能代替三seed同协议实验，也不能把rollout显存下降表述成四卡系统总显存下降。

## 10. 本轮验收结果

- 新增及相关 LIBERO/semantic/shared-semantic/EnvWorker/PPO 回归：214 passed，
  2个第三方warning；
- `ruff check` 通过，全部32个修改 Python 文件均已格式化；
- 全部 Python 入口 `py_compile` 通过；
- 9个相关Shell入口`bash -n`通过；七方法解析后协议dry-run均通过，其中Fresh明确为
  train age `[0,0]`、eval fixed age 0且random age关闭；
- coupled/decoupled 两份 Hydra 配置均完整 resolve（各356行）；
- D-PPO 与 C-PPO 两步 smoke 均无 NaN/Inf，actor 权重变化且 VLM 不变；
- D-PPO checkpoint 已在 fresh process 中加载并完成 exact-age-0 评测。
- D-PPO-Fresh两步smoke、exact-age-0训练/评测、actor参数变化和step-2
  fresh-process checkpoint加载均通过；
- D-SFT和D-PPO各自预注册8×5响应曲面完整dry-run（各40格）通过；schema-v3
  checkpoint/backbone/code身份resume与跨方法provenance拒绝测试通过；两种策略在同一
  age-4×horizon-4、同一trial/noise条件下的fresh-process配对smoke通过，逐格
  validation、manifest写入与安全resume均通过；完整synthetic 2×40×400正式汇总、
  deadline、CSV/JSON和热力图端到端验收通过，低于400-trial的开发格无法标成formal。
- 总结果工具的synthetic正式协议验收通过：D-SFT共享评测、C-PPO/D-PPO各3个匹配
  seed、每个最终评测400个trial、学习曲线pairing metadata、跨方法/seed条件身份
  SHA256以及共同wall-clock/frame积分区间均通过；真实age-8单seed开发结果保留统计，但因缺方法/seed、少于400 trial
  和无正式学习曲线被正确标为`formal_statistical_summary_complete=false`。

未运行项仍按第 9 节保留，不以 smoke 结果替代正式科学结果。

## 11. Semantic age × action replanning 实时性响应曲面

主实时性实验固定控制频率为 20 Hz，并独立扫描：

```text
semantic age d_s:       0, 1, 2, 4, 6, 8, 12, 16 frames
action horizon K_a:     1, 2, 4, 8, 16 frames
prediction horizon:     始终为 16 frames
```

`semantic_eval_fixed_age_frames` 通过 packet history 和 `fetch_exact` 选择真实
来源帧，不通过 sleep 模拟。`eval_execution_horizon` 只截取16步预测的执行前缀；
DiT prediction horizon 和 checkpoint 结构不变。eval worker 根据 `K_a` 重算
action boundary 数量，因此40个条件执行相同的最大控制帧预算。

当 `d_s >= K_a` 时，semantic observation 按
`K_a - (d_s mod K_a)` 的 chunk 内相位发布，保留到后续 boundary 使用。episode
开头因果上不存在的历史仍截断到 bootstrap packet，并分别写出 requested age、
actual age 和 bootstrap-clipped。每行还写出 `action_execution_horizon`。surface
配对键为：

```text
(task_id, trial_id, requested_semantic_age,
 action_execution_horizon, policy_noise_seed)
```

actual age 与 bootstrap-clipped 是逐 trial 审计字段。汇总器以
`requested/actual mismatch boundaries - bootstrap-clipped boundaries`审计非因果
截断之外的actual-age违约；两种策略因episode长度不同产生的actual-age mean差异
单列为轨迹相关描述量，不误记为实验条件不配对。正式汇总
还硬性要求同一方法所有40格，以及D-SFT/D-PPO之间，使用完全相同的
`(task_id, trial_id, policy_noise_seed)`集合；每格各自达到400条但trial集合不同仍会
失败。

完整40-cell矩阵默认只 dry-run：

```bash
POLICY_METHOD=D-SFT DRY_RUN=true \
  bash examples/embodiment/eval_fdvla_realtime_surface.sh
```

正式运行需要显式关闭 dry-run；D-PPO 还必须提供选定 checkpoint：

```bash
POLICY_METHOD=D-SFT DRY_RUN=false \
  bash examples/embodiment/eval_fdvla_realtime_surface.sh

POLICY_METHOD=D-PPO PPO_CKPT_PATH=/absolute/path/to/full_weights.pt \
  DRY_RUN=false bash examples/embodiment/eval_fdvla_realtime_surface.sh
```

可用 `SEMANTIC_AGES="0 4 8"`、`ACTION_HORIZONS="1 4 16"` 运行开发子集；正式
结果必须补齐预注册40格。每个 cell 写到独立目录，runner 成功后自动生成
`surface_cells.tsv`，并在写manifest行之前运行唯一键和条件验证。现有manifest默认
拒绝覆盖；中断后只有显式设置同一个绝对`FDVLA_SURFACE_ROOT`和
`RESUME_SURFACE=true`才会恢复。schema-v3 manifest逐行绑定policy checkpoint、共同
初始checkpoint和backbone的绝对路径与SHA256，以及Git SHA、当前tracked/untracked
源码工作树指纹、16-step prediction horizon和20 Hz控制频率。artifact SHA使用
`file-raw+dir-logical-deref-sha256-v1`：单文件使用标准raw SHA256，目录绑定相对
逻辑路径和解引用后的文件内容，因而覆盖Hugging Face symlink snapshot；空目录或
断链直接失败。恢复路径重新验证
已完成cell；上述身份、task filter、noise seed集合或JSONL路径任一不一致都会在
启动GPU前失败，防止中断恢复时把不同策略或代码版本混入同一曲面。汇总器还会在
D-SFT/D-PPO两份manifest之间核对共同初始化、backbone、代码、trial条件和时间尺度，
只允许两者的policy checkpoint身份不同。汇总器可直接读取一个或多个
manifest，也保留显式 `--cell` 入口：

审计发现旧目录哈希函数只枚举普通文件；冻结Cosmos backbone snapshot的15个入口
全部是symlink，因此旧值没有绑定backbone内容。该问题不改变历史训练或评测数值，
但历史manifest/metadata不能据此声称backbone内容级provenance。当前canonical
backbone SHA256为
`fd4ba5e5215d5861387cf7bca0cf0df1f231a5be6deff5a8673862dcb1ca4220`；schema-v3
显式记录上述算法名并拒绝旧schema resume。symlink-only内容变化、relocation、空目录、
断链和symlink loop均有可执行回归测试。

```bash
python examples/analysis/summarize_fdvla_realtime_surface.py \
  --manifest /absolute/surface_root/surface_cells.tsv \
  --manifest /absolute/second_method/surface_cells.tsv \
  --output-dir /path/to/surface_summary

RESUME_SURFACE=true FDVLA_SURFACE_ROOT=/absolute/existing_surface_root \
  POLICY_METHOD=D-SFT DRY_RUN=false \
  bash examples/embodiment/eval_fdvla_realtime_surface.sh

python examples/analysis/summarize_fdvla_realtime_surface.py \
  --cell D-SFT:0:1=/path/to/sft_age0_ka1.jsonl \
  --cell D-PPO:0:1=/path/to/ppo_age0_ka1.jsonl \
  --allow-incomplete \
  --output-dir /path/to/surface_summary
```

显式`--cell`和旧schema manifest没有checkpoint/code provenance，只允许开发验证并
强制`formal_matrix_complete=false`；正式汇总必须使用新runner生成的schema-v3
manifest。schema升级前已经完成的2-trial smoke仍是有效的链路证据，但不能用新
launcher原地resume，也不能作为正式provenance-complete曲面。

正式 surface runner 和汇总器均默认要求每格至少400个唯一 trial。LIBERO Task 0
只有50个固定初始化状态，不能用重复 eval epoch 伪造更多 trial；runner 因此默认用
8个固定且不同的策略噪声种子（2026--2033）分别覆盖50个初始化状态，再合并为
`(task_id, trial_id, policy_noise_seed)` 唯一的400条记录。默认52个 eval env 用于覆盖
50个状态（多出的环境记录会被 trial 去重）；合并后不足400会立即失败。正式4-rank
placement必须用52而不是51，因为decoupled launcher会把环境数向下取整为rank数的倍数，
51会静默变成48。种子集合可用
`SURFACE_POLICY_NOISE_SEEDS` 显式指定，方法间必须保持完全一致。脚本默认显式设置
`SURFACE_TASK_ID_FILTER='[0]'`；完整 suite 必须另建预注册运行并显式覆盖，不能与
Task 0 的400-trial结果混在同一 manifest。manifest 同时记录 task filter、全部策略
噪声种子和合并 JSONL 的绝对路径。正式52-env值使用独立的
`SURFACE_EVAL_NUM_ENVS`，不会被调用者残留的通用 `EVAL_NUM_ENVS` 静默改成48。
runner在任何模型启动前拒绝重复或非整数age、重复/非正/超过16步预测长度的action
horizon、非16步prediction horizon、非20 Hz控制频率、重复noise seed，以及
`EVAL_ROLLOUT_EPOCH != 1`；正式样本扩展只能使用
不同noise seed，不能重复同一eval epoch。
开发子集必须显式设置
`SURFACE_MIN_TRIALS=48`，汇总时同时传 `--min-trials-per-cell 48 --allow-incomplete`。
汇总器另以不可由CLI验证阈值降低的`formal_trial_audit`检查每个实际cell至少有400条
唯一记录，并写出`required_trials_per_cell`、`minimum_observed_trials`和所有不足格。
`--allow-incomplete`会显式设置`development_mode=true`并强制
`formal_matrix_complete=false`；即使开发者收集齐40格，也不能用48-trial阈值误标为
正式矩阵。

输出包括成功率热力图、Wilson 95% 区间、逐 cell McNemar 转移以及
`tau_s`、`tau_a`、时间尺度不对称比。`tau95/tau90/tau50` 同时报告点估计和
Wilson 下界版本，并换算 `semantic_min_update_hz` 与 `action_min_replan_hz`；零帧
deadline 的最低频率写 `null`。没有测试点通过门槛时写 `null`，不外推 deadline。正式汇总默认
硬性要求D-SFT与D-PPO各自完整的预注册8×5矩阵；缺方法、缺格或额外格都会失败。
开发子集必须显式传入`--allow-incomplete`，输出会标记
`formal_matrix_complete=false`。论文主deadline严格按预注册公式，取所有已测点中
满足阈值的最大age/period；另行输出从`P(0,1)`开始连续通过阈值的敏感性版本，便于
识别非单调噪声反弹，但不替代主定义。没有`P(0,1)`的开发子集明确写
`deadline.available=false`，不再报错或伪造deadline。即使已有`P(0,1)`，只要
`K_a=1`上的预注册semantic age轴或`age=0`上的预注册action horizon轴缺少任一点，
也必须写`deadline.available=false`；`deadline.axis_audit`机器可读地列出每条轴的
expected、observed与missing点。额外的非预注册点不会参与deadline计算。

若目标只是估计两类deadline，而不是绘制完整8×5响应曲面，可使用低成本轴模式：

```bash
DRY_RUN=true \
SURFACE_SWEEP_MODE=deadline_axes \
POLICY_METHOD=D-SFT \
bash examples/embodiment/eval_fdvla_realtime_surface.sh
```

默认预注册集合下，该模式仅枚举`(age, K_a=1)`的8格与`(age=0, K_a)`的5格并集，
共12格/方法，`P(0,1)`只运行一次；`cartesian`仍为默认且枚举40格/方法。正式运行仍需
分别对D-SFT和D-PPO提供checkpoint、每格至少400个固定配对trial，并在汇总时传
`--allow-incomplete`（因为没有内部28格，完整矩阵标志应保持false）。deadline是否
可用独立由`axis_audit.complete`决定；轴模式不改变trial、provenance和二值奖励契约。

使用明确标记为synthetic的临时数据完成了一次正式汇总器端到端验收：D-SFT和
D-PPO各40格、每格400个严格相同的`(task, trial, noise)`身份，共80格和40个跨方法
配对条件。非开发模式成功输出`formal_matrix_complete=true`、完整provenance/trial
审计、`tau95/tau90/tau50`的点估计与Wilson下界、CSV、JSON和两张热力图。相同代码
重汇总真实age-8与age-12开发格时则分别记录`minimum_observed_trials=50`、
`formal_trial_audit.complete=false`、`development_mode=true`和
`formal_matrix_complete=false`。synthetic数值只验证工具链，绝不作为科学结果。

正式 fresh-process surface JSONL 还会逐 episode 写出 semantic boundary 总数、
bootstrap-clipped boundary 数、requested/actual-age mismatch boundary 数，以及
actual-age 的 episode 均值/最小/最大值。汇总同时报告整格 boundary-level 计数，
`bootstrap_clipped_trials`定义为episode内至少一个boundary被截断；终止boundary的
旧布尔值另记为`terminal_boundary_bootstrap_clipped_trials`，不再冒充整段episode
的bootstrap统计。

低成本端到端surface smoke已通过：

```text
logs/fdvla_realtime_surface/20260821T1036Z_D-SFT_smoke
method: D-SFT
cell: semantic age 4, action execution horizon 4
fixed trials: 2
success: 2/2
semantic boundaries: 154
bootstrap-clipped boundaries: 2（每个episode起点各1）
requested/actual mismatch boundaries: 2
actual age: min 0, boundary mean 3.9481, max 4

logs/fdvla_realtime_surface/20260821T1050Z_D-PPO_smoke
method: D-PPO（fresh process加载global_step_50 checkpoint）
cell: semantic age 4, action execution horizon 4
fixed trials: 2（与D-SFT完全相同）
success: 2/2
semantic boundaries: 158
bootstrap-clipped boundaries: 2（每个episode起点各1）
requested/actual mismatch boundaries: 2
actual age: min 0, boundary mean 3.9494, max 4
```

运行日志确认模型始终预测16步而每次只执行4步前缀，起点因果截断为age 0，之后
每个action boundary均通过`fetch_exact`取得age 4；manifest、合并JSONL和逐格
validation JSON均成功写出。对同一root执行`RESUME_SURFACE=true`只重新验证并跳过
该cell，修改noise seed后则以resume signature mismatch退出码2拒绝。这两份记录
生成于正式provenance schema绑定之前；schema-v3版本另有可执行测试证明checkpoint
SHA变化会在模型启动前被拒绝，而完整匹配的身份会继续进入JSONL存在性和逐格验证。

两份manifest合并后的开发配对审计确认：2个
`(task_id, trial_id, policy_noise_seed)`身份完全一致，非bootstrap actual-age违约为0，
bootstrap条件不匹配为0，D-SFT到D-PPO的failure-to-success与success-to-failure均为0。
两条策略轨迹的episode长度不同，因此2个trial的actual-age mean略有不同；这不改变
除首边界外均为exact age 4的事实。该结果只证明单格实验链路与配对契约，不能替代
每格400个trial、D-SFT/D-PPO各40格的正式响应曲面，也不支持性能优劣结论。

为直接检查Task 0在随机age 0--6下83.33%的初始成功率是否存在天花板，额外运行了
一个预注册困难格的schema-v3开发评测：fixed semantic age 12、execution horizon
16、prediction horizon 16、policy-noise condition seed 2026。两种方法分别在全新
Ray和semantic server进程中评测相同50个固定初始化状态：

| method | success | Wilson 95% CI | semantic boundaries | bootstrap boundaries |
|---|---:|---:|---:|---:|
| D-SFT | 2/50 = 4.0% | 1.10%--13.46% | 1,485 | 50 |
| D-PPO step 50 | 1/50 = 2.0% | 0.35%--10.50% | 1,495 | 50 |

日志和汇总位于：

```text
logs/fdvla_realtime_surface/20260821_D-SFT_age12_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_D-PPO_age12_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_age12_ka16_pair_summary_schema3
```

schema-v3审计确认共同初始checkpoint、backbone、Git SHA、运行时worktree SHA、task、
trial、noise、prediction horizon和control frequency完全一致，仅最终policy checkpoint
按方法合法不同。50个trial身份完全配对；D-SFT到D-PPO的failure-to-success为0、
success-to-failure为1，exact McNemar `p=1.0`。每个episode只有首action boundary因
因果限制从requested age 12截断到actual age 0，此后均exact age 12；两种方法的
非bootstrap actual-age契约违约均为0，bootstrap条件不匹配为0。

因此较困难的age/action条件确实消除了容易条件下的天花板，但该50-update PPO只在
age 0--6上训练，并未在这个age-12外推格改善成功率。该单格结果是应保留的负开发
结果，不能证明RL普遍退化，也不能计算`tau_s`、`tau_a`或时间尺度不对称比；汇总器
正确输出`formal_matrix_complete=false`和`deadline.available=false`。正式结论仍需
预注册完整矩阵和每格400个配对trial。

为避免只观察容易条件的天花板和age-12条件的地板，另在开始前固定了一个唯一的
中间开发格：fixed semantic age 8、execution horizon 16、prediction horizon 16、
policy-noise condition seed 2026、Task 0的相同50个固定初始化状态。age 8既是cached
SFT的训练年龄上界，又超出本次PPO的age 0--6训练范围；它不是根据结果挑选的最优格。

| method | success | Wilson 95% CI | semantic boundaries | bootstrap boundaries |
|---|---:|---:|---:|---:|
| D-SFT | 16/50 = 32.0% | 20.76%--45.81% | 1,407 | 50 |
| D-PPO step 50 | 17/50 = 34.0% | 22.44%--47.85% | 1,385 | 50 |

日志和配对汇总位于：

```text
logs/fdvla_realtime_surface/20260821_D-SFT_age8_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_D-PPO_age8_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_age8_ka16_pair_summary_schema3
```

两份schema-v3 manifest的初始checkpoint、backbone、Git SHA、worktree SHA、task、trial、
noise、prediction horizon和control frequency完全一致；50个trial身份完全配对，出处
审计无不匹配。D-SFT到D-PPO有7个failure-to-success和6个success-to-failure，净差
仅1个trial，exact McNemar `p=1.0`；两个Wilson区间高度重叠。因此该格证明age 8、
horizon 16是介于随机age 0--6容易条件和age 12地板条件之间的有辨别力开发难度，
但不提供PPO优于SFT的统计证据。每个episode只有首边界因因果bootstrap从requested
age 8截断到actual age 0，此后均exact age 8；非bootstrap actual-age契约违约为0，
bootstrap条件不匹配为0。完整矩阵仍未运行，不能据此选择checkpoint或宣称普遍提升。

为补齐训练年龄范围上界处的可辨别开发点，随后运行预注册网格中的 fixed semantic
age 6、execution horizon 16、prediction horizon 16、policy-noise condition seed
2026。该条件在启动前固定；Task 0 的50个初始化状态、源码工作树和模型出处均与方法
内严格绑定，并非根据结果选择。

| method | success | Wilson 95% CI | semantic boundaries | bootstrap boundaries |
|---|---:|---:|---:|---:|
| D-SFT | 27/50 = 54.0% | 40.40%--67.03% | 1,262 | 50 |
| D-PPO step 50 | 26/50 = 52.0% | 38.51%--65.20% | 1,261 | 50 |

日志和配对汇总位于：

```text
logs/fdvla_realtime_surface/20260821_D-SFT_age6_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_D-PPO_age6_ka16_dev_schema3
logs/fdvla_realtime_surface/20260821_age6_ka16_pair_summary_schema3
```

两份manifest的schema版本、artifact哈希算法、共同初始checkpoint、backbone、Git SHA、
worktree SHA、Task 0 filter、50个trial/noise身份、prediction horizon和20 Hz控制频率
完全一致，仅policy checkpoint按方法不同。D-SFT到D-PPO有5个
failure-to-success和6个success-to-failure，净差为-1个trial，exact McNemar
`p=1.0`；Wilson区间高度重叠。每个episode只有首boundary因果bootstrap为actual
age 0，之后均exact age 6；两种方法的非bootstrap actual-age违约、bootstrap条件
不匹配和cross-episode packet mismatch均为0。该点位于age 8的32%--34%与旧随机
age 0--6的83.33%--89.58%之间，进一步说明旧随机条件的初始成功率偏高；但50-trial
单seed开发结果仍不构成D-PPO优于或劣于D-SFT的正式证据。

### age 6 下 execution horizon 4/8/16 的严格配对开发切片

为直接检查16-step prediction chunk是否被执行过长，在保持 prediction horizon=16、
fixed semantic age=6、Task 0、policy-noise condition seed=2026和50个固定初始化状态
不变时，只扫描实际 action execution/replanning horizon 4、8和16。首次真实运行暴露
出一个预取错误：fixed-age evaluator 用 prediction horizon 16 而不是实际 execution
horizon 4 计算下一次 semantic source frame，导致短周期运行在
`current_frame=8, source_frame=2`无法取得 exact packet。失败运行保留在：

```text
logs/fdvla_realtime_surface/20260821_D-SFT_age6_ka4_16_dev_schema3_v2
```

修复后，下一 boundary 的预取按 `eval_execution_horizon` 推进；prediction head和
checkpoint结构仍保持16步。单元测试显式设置 prediction=16、execution=4，验证预取
使用后者。修复后的严格配对结果为：

| method | execution horizon | success | Wilson 95% CI | eval wall-clock | rollout predict | semantic fetch |
|---|---:|---:|---:|---:|---:|---:|
| D-SFT | 4 | 40/50 = 80.0% | 66.96%--88.76% | 99.45 s | 51.99 s | 39.02 s |
| D-SFT | 8 | 39/50 = 78.0% | 64.76%--87.25% | 57.50 s | 11.07 s | 3.41 s |
| D-SFT | 16 | 27/50 = 54.0% | 40.40%--67.03% | 50.88 s | 6.31 s | 2.41 s |
| D-PPO step 50 | 4 | 35/50 = 70.0% | 56.25%--80.90% | 99.23 s | 52.42 s | 38.06 s |
| D-PPO step 50 | 8 | 39/50 = 78.0% | 64.76%--87.25% | 59.60 s | 11.59 s | 4.40 s |
| D-PPO step 50 | 16 | 26/50 = 52.0% | 38.51%--65.20% | 51.94 s | 6.90 s | 2.67 s |

同一方法内逐 trial 比较，D-SFT 从 horizon 4到8有6个failure-to-success、7个
success-to-failure，exact McNemar `p=1.0`；从8到16为5和17，`p=0.016901`。
D-PPO从4到8为9和5，`p=0.423950`；从8到16为6和19，`p=0.014633`。因此在这个
单 seed 开发切片中，horizon 8到16的成功率下降在两种方法内均达到配对显著性；
16步实际执行过长，而4相对8没有显著成功率优势。

代价同样明确：horizon 4相对8把本次 fresh-process eval wall-clock提高约67%--73%，
而成功率没有提高（D-PPO还下降8个百分点）。每个rollout rank记录的action boundary
在horizon 4/8/16下分别为120/60/30；D-SFT的semantic fetch mean分别为
325.14/56.88/80.46 ms，D-PPO为317.18/73.25/88.83 ms。horizon 8允许server预取
跟上控制边界，在该硬件放置下形成明显的系统甜点。六格均为8160个控制帧、本地VLM
forward=0、cross-episode packet mismatch=0。K=4时每个episode前两个boundary
（current frame 0和4）受因果限制而bootstrap截断，K=8和16时只有首boundary截断；
扣除这些bootstrap后actual-age契约违约均为0。

日志和汇总位于：

```text
logs/fdvla_realtime_surface/20260821_D-SFT_age6_ka4_16_dev_schema3_v4
logs/fdvla_realtime_surface/20260821_D-SFT_age6_ka8_dev_schema3_v1
logs/fdvla_realtime_surface/20260821_D-PPO_age6_ka4_16_dev_schema3_v4
logs/fdvla_realtime_surface/20260821_D-PPO_age6_ka8_dev_schema3_v1
logs/fdvla_realtime_surface/20260821_age6_ka4_8_16_pair_pareto_schema3_v2
```

四份schema-v3 manifest的共同初始checkpoint、backbone、Git SHA、运行时worktree SHA、
task/trial/noise、prediction horizon和control frequency完全一致，50个trial身份审计
完整，仅最终policy checkpoint按方法合法不同。方法间比较在horizon 4下为D-SFT到
D-PPO failure-to-success=2、success-to-failure=7、`p=0.179688`；horizon 8为6、6、
`p=1.0`；horizon 16为5、6、`p=1.0`。PPO只在horizon 8与SFT同率，三格均没有
PPO优于SFT的统计证据，不能据此声称binary PPO已经适应该延迟。

实时性汇总器会按每个`noise_*`运行读取TensorBoard最新事件；同一noise目录即使因
restart留下多个event文件，也只取每个标签wall-time最新的值，不会重复累计。CSV/JSON
现自动包含`eval_wall_clock_s`、每trial wall-clock、successful trials/hour、rollout
predict、semantic fetch、control frames和正确性计数，并生成
`fdvla_realtime_success_vs_wall_clock.png`。本切片D-SFT在horizon 4/8/16下的成功
trial吞吐分别为1448/2442/1910 per hour，D-PPO为1270/2356/1802 per hour；自动
Pareto审计将D-PPO horizon 4判为被horizon 8严格支配，D-PPO前沿为`[8,16]`。
D-SFT因horizon 4仍多1个成功trial，数学前沿为`[4,8,16]`，但8是实际knee point。
这些是固定策略评测吞吐，不是PPO训练learning-curve wall-clock效率，二者不得混称。

该切片说明“execution horizon 16对成功率过高”和“execution horizon 4的关键路径
更贵”同时成立。horizon 8是当前单seed开发切片的实用knee point：D-SFT只比4少1个
成功trial却快42%，D-PPO则同时比4成功率高8个百分点且快40%。这仍没有证明8是
全局最优；正式响应曲面应保留1/2/4/8/16并报告success-vs-wall-clock Pareto关系。

动作计算—执行 FIFO 延迟 `eval_action_apply_delay_frames` 没有混入主二维实验。
它需要逐环境、跨 episode 清零的 action queue 与独立 bootstrap 审计，将作为附加
实验单独实现；当前代码不会把较小 `K_a` 错称为 action apply delay。

## 弱初始化去天花板开发 pilot（2026-08-21）

### 预注册筛选与初始化锁定

强 cached-semantic SFT 初始化在 fixed age 6、execution horizon 8 的开发评测中已达到
约78%，留给binary PPO的提升空间过小。为避免在看到正式评测结果后挑选起点，先在
`fdvla_binary_run_manifest.yaml`中锁定Task 0、fixed age 6、execution horizon 8、
prediction horizon 16、policy-noise seed 12026、50个唯一trial和目标成功率45%的
筛选规则，再在相同worktree上评测三个同系列checkpoint。结果产生前固定的选择规则
是最小化与45%的绝对距离，正确性计数必须全部为0；seed 12026不得进入后续评测。

代码冻结筛选位于：

```text
logs/fdvla_initial_checkpoint_screen/20260821_codefrozen_v2_age6_ka8_noise12026
```

worktree SHA256为
`8fa77902ed3d4d1c2ce1f8113434790f5750260819be63480a2e38d866d79ab0`。结果为
weak1000 15/50（30%）、weak1500 40/50（80%）、weak1000plus500 8/50（16%）；
三者的非bootstrap age mismatch、cross-episode packet mismatch和local VLM forward
均为0。按预注册规则选择weak1000，并在后续D-SFT/D-PPO方法中锁定为共同初始化：

```text
/vepfs-mlp2/c20250301/240403026/async_vla/async_libero_runs/
libero10_decoupled_sft_0to8_rebuilt/
W1000_libero10_decoupled_0to8_scalar_age_s4_a4/
libero10_delay0to8_s4_a4_weak1000/checkpoint-1000
SHA256: edff533a5c7b9472739577f0b4de8c0bb39f6a93815aad30ec40784538d1adbf
```

### D-PPO seed 0，50次更新

开发运行位于：

```text
logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo
experiment: fdvla_binary_d_ppo_weak1000_task0_seed0
```

训练保持原生`actor_critic`、GAE和`group_size=1`，训练semantic age随机0--6；固定
评测使用Task 0、trial 0--47、noise seed 2026、exact age 6、execution horizon 8和
prediction horizon 16。该评测seed与筛选seed 12026隔离。标准learning-curve输出为：

```text
logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/
analysis_final/learning_curve.csv
```

| update | actual control frames | wall-clock since step-0 eval | success | Wilson 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.00 s | 13/48 = 27.08% | 16.57%--41.00% |
| 10 | 232,960 | 2,732.15 s | 15/48 = 31.25% | 19.95%--45.33% |
| 20 | 458,240 | 5,470.48 s | 17/48 = 35.42% | 23.43%--49.56% |
| 30 | 683,520 | 8,249.76 s | 15/48 = 31.25% | 19.95%--45.33% |
| 40 | 908,800 | 11,023.06 s | 12/48 = 25.00% | 14.92%--38.78% |
| 50 | 1,134,080 | 13,782.44 s | 18/48 = 37.50% | 25.22%--51.64% |

六次评测的48个`(task_id, trial_id, policy_noise_seed, requested_semantic_age,
action_execution_horizon)`身份完全一致，配对身份SHA256为
`778fa050ab209bc76382a7a07b843022c6a9e36d1b203f2305e2ff78b8b1e89f`。
step 0到50有9个failure-to-success和4个success-to-failure，exact McNemar
`p=0.266846`。终点相对共同D-SFT初始化增加5个成功trial，即绝对+10.42个百分点，
但48-trial、单训练seed的区间重叠且配对检验不显著，不能声称binary PPO已经可靠
提升性能。曲线在step 40回落到25%后于step 50恢复到37.5%，也说明开发评测方差和
训练漂移都不可忽略；没有按中间峰值事后截断或删除负点。

50次更新共执行1,134,080个control frame，完成7,847个episode，其中4,094成功、
3,753失败；平均82.22 control frames/s和2,047.96 episodes/hour。全程50个PPO点的
required scalar均存在，NaN/Inf计数为0；cross-episode packet mismatch、rollout/train
semantic fingerprint mismatch、local VLM forward和Reward Model invocation均为0。
rollout峰值GPU显存为3,416.06 MiB。终点KL为`-4.74e-05`（有限采样估计可轻微为负）、
clip fraction 0.0057、ratio 1.002、gradient norm 7.945、value loss 0.0335、
explained variance 0.3197，未见数值发散。

最终checkpoint为：

```text
logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/
fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50/
actor/model_state_dict/full_weights.pt
SHA256: ade2858d1bf13e062947e227d00d12159c4dbdcc08b1464f7d087497cbafca8f
```

### fresh-process严格复评

训练进程完全退出后，从上述`full_weights.pt`在新Ray/semantic-server进程中重新加载，
运行相同48个配对trial。schema-v3 manifest、逐trial JSONL和验证摘要位于：

```text
logs/fdvla_realtime_surface/20260821_weak1000_step50_fresh_age6_ka8_noise2026
```

三个rollout rank均报告override checkpoint加载成功，VLM总参数和可训练参数均为0，
checkpoint中需丢弃的冻结backbone tensor为0。fresh process仍为18/48（37.5%，
Wilson 95% CI 25.22%--51.64%）；与训练内step-50 JSONL相比，48个身份、48个success
布尔值以及全部semantic age/boundary审计字段逐项相等，outcome mismatch为0，
非bootstrap requested/actual age mismatch为0。该次评测执行7,680个control frame，
eval wall-clock 54.36 s、rollout predict 12.12 s、semantic fetch 4.49 s，local VLM
forward和cross-episode packet mismatch均为0。

这项结果证明弱初始化的50-update D-PPO闭环可复现、checkpoint可在新进程加载且评测
确定性可复核；它不证明正式方法优势。正式结论仍需要锁定同一weak1000初始化后完成
D-SFT、C-PPO及D-PPO的3个训练seed、独立checkpoint选择记录和至少400个最终固定
trial。当前未自动启动这些高成本矩阵。

### 同一worktree的fresh D-SFT--D-PPO配对闭环

为避免把训练内step-0直接当作独立D-SFT评测，另在fresh process中加载共同
weak1000初始化，使用与step-50 D-PPO完全相同的Task 0、trial 0--47、noise seed
2026、exact age 6、execution horizon 8和prediction horizon 16重新评测。随后又在同一
worktree快照上重跑D-PPO单格；未绕过响应曲面汇总器对worktree provenance的严格
检查。两个schema-v3 manifest分别位于：

```text
logs/fdvla_realtime_surface/20260821_weak1000_dsft_fresh_age6_ka8_noise2026
logs/fdvla_realtime_surface/20260821_weak1000_dppo_step50_currenttree_age6_ka8_noise2026
```

两者共同记录Git SHA
`a6d8747a0661fe7ca7ff1a9f7a284d2d75594afa`、worktree SHA256
`cf2ed5bb3f1ec211e649b110bf545ca8a5f1a27b7d2313580b6dbe09340b4b31`、initial
checkpoint SHA256 `edff533a5c7b9472739577f0b4de8c0bb39f6a93815aad30ec40784538d1adbf`
和backbone SHA256 `fd4ba5e5215d5861387cf7bca0cf0df1f231a5be6deff5a8673862dcb1ca4220`。
manifest provenance audit、48个跨方法trial identity以及各方法内部identity audit均
完整且无不匹配。

| 方法 | 成功率 | Wilson 95% CI | eval wall-clock | successful trials/hour |
|---|---:|---:|---:|---:|
| D-SFT | 13/48 = 27.08% | 16.57%--41.00% | 53.80 s | 869.83 |
| D-PPO | 18/48 = 37.50% | 25.22%--51.64% | 52.34 s | 1,238.10 |

D-SFT到D-PPO有9个failure-to-success和4个success-to-failure，绝对差
+10.42个百分点，exact McNemar `p=0.266846`；requested/actual age的非bootstrap
contract violation、cross-episode packet mismatch和local VLM forward均为0。
D-SFT fresh JSONL与训练内step-0 JSONL逐字节相同（SHA256
`8349a68461351a0c303aca713c8bd3404e395c1c9b4e2f4fd23ce11995002113`），D-PPO fresh
JSONL与训练内step-50 JSONL也逐字节相同（SHA256
`b60ccaa8fd7010601fc436a6f9beca4110a325dd0387184caa981d8ac630a215`）。统一摘要与图表
位于：

```text
logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/analysis_final/
realtime_weak1000_paired_age6_ka8
```

该摘要因每方法只有48个trial、D-PPO只有一个训练seed且仅覆盖一个响应曲面格，强制
保留`development_mode=true`和`formal_matrix_complete=false`；deadline因缺少完整
预注册轴及`P(0,1)`基准而明确不可用。这是严格可复核的开发闭环，不是论文主结果。

上述分散产物可用只读验收器重新核对；它会在任一训练健康度、pairing、provenance、
fresh-process字节一致性、checkpoint哈希或非正式结果标签不满足时返回非零退出码：

```bash
python examples/analysis/audit_fdvla_development_closure.py \
  --run-dir logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo \
  --training-health logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/analysis_final/fdvla_training_health.json \
  --learning-curve-metadata logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/analysis_final/learning_curve_paired.csv.metadata.json \
  --surface-summary logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/analysis_final/realtime_weak1000_paired_age6_ka8/fdvla_realtime_surface.json \
  --dsft-fresh-jsonl logs/fdvla_realtime_surface/20260821_weak1000_dsft_fresh_age6_ka8_noise2026/age_6_ka_8/eval_trials_combined.jsonl \
  --dppo-fresh-jsonl logs/fdvla_realtime_surface/20260821_weak1000_dppo_step50_currenttree_age6_ka8_noise2026/age_6_ka_8/eval_trials_combined.jsonl \
  --checkpoint logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/fdvla_binary_d_ppo_weak1000_task0_seed0/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
  --output logs/20260821-weak1000-dppo-seed0-libero_10_fdvla_binary_ppo/analysis_final/development_closure_audit.json
```

该命令已对真实产物执行；9个门槛全部通过并输出
`development_closure_complete=true`，同时固定写出`scientific_result_formal=false`和
仍缺的正式实验要求，避免把开发闭环误报为论文结果。

## 2026-08-30：四卡 semantic/DiT placement 配对筛查

为判断60个并行LIBERO环境下的主要rollout瓶颈，固定同一weak1000 D-SFT
checkpoint、Task 0、60个train env、prediction horizon 8、实际execution horizon
`K=8`、denoising steps 4、1个rollout epoch、global batch 192、1个PPO update和四张
A100，对三种物理布局做单seed开发筛查。所有布局均使用真正非阻塞latest packet；
3S1D和2S2D中单个DiT rank分别通过20-row有序微批处理60和30个环境。

| 布局 | semantic server | DiT rollout rank | actor rank | rollout | actor update | full step | semantic age mean / p50 / p95 | train success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1S3D | 1 | 3 | 4 | 44.38 s | 46.29 s | 102.21 s | 9.02 / 10 / 14 | 13/60 |
| 2S2D | 2 | 2 | 4 | 53.78 s | 48.04 s | 113.46 s | 4.59 / 4 / 11 | 6/60 |
| 3S1D | 3 | 1 | 3 | 86.54 s | 61.35 s | 158.22 s | 3.82 / 3 / 11 | 11/60 |

相对1S3D，2S2D的rollout慢21.2%，3S1D慢95.0%；按固定15,360个名义control
frame折算，三者分别为346.1、285.6和177.5 frame/s。增加semantic server确实把
平均semantic age从9.02帧降至4.59和3.82帧，但把三张DiT卡缩到一张后，有序微批
使action generation成为主要吞吐瓶颈。因此，在本次60-env仿真配置中，1S3D是
wall-clock速度最优布局，2S2D是semantic freshness与吞吐的折中，3S1D不支持
“更多server会提高总体吞吐”的假设。该结论只针对多环境仿真吞吐，不能直接外推为
单机器人端侧算力结论。

三个PPO update均完成，NaN/Inf、cross-episode packet mismatch、rollout/train
semantic fingerprint mismatch、trainable VLM参数和Reward Model调用均为0。表中的
train success是单次rollout中终止episode的开发统计，不是固定trial策略评测，不能据此
比较三种布局的最终任务性能；需要多update、相同固定trial的独立评测才能判断semantic
freshness带来的性能收益。有效产物位于：

```text
logs/fdvla_placement_matrix/20260830_matched_k8_v3/
```

## 2026-09-03：BF16 tail-4 batching 计时协议修正

首次计时运行`20260903_task0_k2_bf16_tail4_v3`按每候选5个update执行。框架仍在
最后一步进行终评，训练汇总器按预注册规则排除被终评包围的训练事件；因此扣除前2个
warm-up update后，每个候选只剩2个无评测污染的稳定样本，低于selector硬编码的3个
样本门槛。该问题在`batch1`完成、`batch2`刚启动时发现；v3立即停止，所有日志保留，
且不得进入候选排序。

修正版统一将每候选改为6个update，并显式记录`2 warm-up + 1 final-eval-enclosing`
排除规则，从而恰好留下3个稳定样本。候选集合、排序指标、tie-break、模型、数据、seed
与健康门槛均不变；该修正不读取也不使用任何候选成功率。
