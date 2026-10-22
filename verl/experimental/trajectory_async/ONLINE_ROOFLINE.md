# 在线 Decode Roofline 容量辨识

2026-10-20：真实 Laminar/vLLM 路径默认启用。目标是估计当前模型、设备、TP 和
执行模式下的经验 decode 延迟膝点，并约束轨迹重调度；不将统计断点直接称为
硬件从 HBM-bound 到 compute-bound 的证明。

## 数据路径与开销

训练器通过 vLLM 的 worker extension RPC 安装 verl 内的探针，不修改已安装的
vLLM、vllm-ascend 或 torch。常态每 8 个 execute step 采样一次；新 batch/context 档位短时密集采样，使用 CUDA/NPU
事件包住 worker execute 与分离的 sample_tokens。之后用 query() 检查事件完成，
不增加 synchronize()、张量 CPU 拷贝或逐步 TP collective。

device elapsed_ms 是 worker 设备流上的服务时间，包括计时区间内的等待，
不是客户端端到端 TPOT；host_enqueue_ms 仅作诊断，不冒充设备执行时间。
控制器约每 5 秒读取遥测时，server 批量提取已完成样本，按版本和 step 对齐所有 TP
rank，检查 batch/context/mode/padding 一致，取最慢 rank 的 elapsed_ms。
缺 rank 的样本有界等待，不使用 rank 0 值替代。每次每 rank 最多返回 512 条。

只保留明确的纯 decode：没有新请求、encoder 输入或 speculative tokens；每个
调度请求恰好 1 token，且 computed_tokens 已达到该请求 prompt 长度。
因此 chunked prefill 的最后一个 token 也不会被当作 decode。
图模式和 padded batch 从 forward context 读取；未知模式不参与拟合。
首个 shape 的若干样本、图捕获及 backend compilation 计数变化的样本被排除。
当前 Ascend NPUModelRunner V1 的 _model_forward 适配已经接入；其他 runner 必须能
提供相同模式信息，或显式 enforce_eager，否则报告不可用而保留原容量。

采样仅覆盖每个 replica 最早 3 个实际加载的不同 policy version，之后停止。
这是开销窗口，不是保证 3 轮收敛。每个 batch/context/mode 桶最多保留 128 条，
每个 worker 和每个 replica 的拟合器分别最多 16384 条；pending device events 默认 64。
拟合放在控制器 CPU 线程中，默认间隔至少 10 秒，不阻塞 decode worker 做回归。
新增档位的密集阶段最多尝试 `2 * min_samples_per_bin + warmup_per_shape` 步，
默认每步采样，之后回到 sample_every。选择时刻只依赖调度 CPU 元数据和尝试计数，
不依赖各 rank 的设备完成速度；pending/cache/总样本预算仍生效。
节流期间收到的样本会标记为待拟合，即使之后没有新样本也会刷新。
一次被拒绝的拟合会打断 stable_fits 连续成功计数。
校准数据不落盘，事件日志仅记录聚合统计与模型判定。

## 拟合、交叉验证与拒绝条件

- 按执行模式和最大单请求上下文的 4096-token 区间分层；不混合不同 TP replica。
  c 明确定义为本步全部 decode 请求的有效上下文 token 总数，而不是 KV 占用率。
- 每个实际 batch 桶至少 16 点，统计中位数和 P95。至少 6 个不同 batch，
  断点两边各至少 2 个档位；1..8 的观测不能用于声称辨识出了 B=32。
- 拟合连续 hinge 模型：
  `t = a + s_b*b + s_c*c/4096 + s_h*max(0,b-B)`，所有斜率非负。
  同时比较无 hinge 的模型；默认 RSS 改善至少 35%、前后总斜率比至少 2，
  且 BIC 改善大于 2，才接受存在断点的证据。
- 对 TPS=b/t 的归一化差值曲线执行 Kneedle 局部极值和阈值穿越检查。
  简单的“边际增益下降到 30%”不冒称 Kneedle。两种估计明显不一致时不采用。
- 按 `(policy_version, step//128)` 分块 bootstrap，避免把相邻步当作独立样本。
  至少 8 个块，默认 64 次重采样；有效拟合少于 80% 或 95% CI 太宽时拒绝。
  两次带有新观测的稳定估计后才冻结该 profile。
- 另查 P95 延迟约束：相对于该分层最小已观测 batch 的 P95，默认容忍 25%。
  若没有 b=1，不将基线标为 b=1；25% 是可配置工程策略，不是 Roofline 定义。

独立经验延迟界：即使没有可信膝点，也可在足够档位、样本和独立块覆盖后，
用两次含新观测的稳定估计建立 latency_guard。它按当前分层最小已观测 batch 的 P95
和 latency_tolerance 选取达标的最大已观测工作点，独立参与 admission/repack 限制。
建立后只允许在该标定窗口内收紧；旧上限本身不再达标时，退到当前仍达标的观测点。
非单调的坏 shape 保留在 unsafe_observed_batches，不掩盖异常，也不以孤立 shape
推断单调性能曲线。该界是经验工作点，不保证所有更小 shape 或未来请求的 P95；
它不等于膝点，也不是分布无关的延迟 SLA。

状态可能为 insufficient_coverage、no_observed_knee、kneedle_unconfirmed、
estimators_disagree、insufficient_independent_blocks、unstable_bootstrap、
wide_confidence_interval、candidate 或 ready。拒绝辨识是正常结果，不能强行输出 B。

## 调度接入

admission 的 batch_per_replica、引擎 max_num_seqs 与 repack_max_running 分开。
校准可信时取以下最小值，并仅选择已观测 batch，不外推：

`B_safe = min(engine_limit, configured_repack_limit, online_limit, KV_SSM_limit,
              knee_CI_lower, P95_latency_limit)`

Best-Fit 对每个目的实例单独检查，包含本轮已经预约的请求数和迁入后的最大上下文。
KV/SSM 容量由当前已加载引擎的真实缓存组和页数给出，不再仅靠固定请求数。
膝点证据不足时不施加虚构的膝点界；独立 latency_guard ready 时仍应用经验延迟界。
上下文/模式未覆盖、探针不可用或两者均不可信时，不宣称已校准延迟限制；
在线探索仍受物理页预算和引擎上限约束，延迟保证尚未校准。
启用在线容量时，配置的 batch_per_replica 是并发起点，可用实际训练请求逐级增加，
不注入 dummy 请求，也不在线修改引擎硬上限。关闭在线容量时保留原有配置行为。

## 配置和观测

```yaml
trainer:
  v1:
    laminar:
      batch_per_replica: 8  # 在线 admission 初始值
      online_capacity:
        enabled: true
        kv_usage_limit: 0.9
        healthy_polls: 3
        growth_factor: 1.25
        backoff_factor: 0.75
        load_ratio: 0.75
        context_bucket_tokens: 4096
        max_observation_age_s: 15.0
      repack_max_running: 0  # 0 表示取 engine max_num_seqs 为 fallback
      roofline:
        enabled: true
        calibration_rounds: 3
        sample_every: 8
        coverage_sample_every: 1
        context_bucket_tokens: 4096
        latency_tolerance: 0.25
```

完整参数见 ppo_trainer.yaml。关闭 Roofline 使用
`trainer.v1.laminar.roofline.enabled=False`，在线容量随之停止。
仅关闭主动并发探索可用 `trainer.v1.laminar.online_capacity.enabled=False`。
controller.status 的 roofline 字段包含分层模型、样本数、拒绝原因、膝点、CI 和容量。
scheduler.jsonl 的 roofline_update/replica_snapshot 记录拟合和当前 repack_limit；
replica_snapshot 还记录探针错误、采集窗口、模式和上下文。

修改设备、模型、量化、TP、图策略或 kernel 后应新建任务重新校准，不复用旧 profile。
没有覆盖拐点时先区分引擎上限、物理页预算、请求供应不足和样本不足。
Qwen3.5 Laminar launcher 的引擎探索硬上限默认 64，可用
`LAMINAR_ENGINE_MAX_SEQS=32` 在启动前修改；默认 admission 仍从 8 开始。
通用入口需显式设置 rollout.max_num_seqs。用户最后提供的 Hydra override 优先。
设置硬上限为 8 时，不会为了标定自动突破 8。

参考：[Laminar §5.2](https://arxiv.org/html/2510.12633v1#S5.SS2)、
[Kneedle 原论文](https://www.cs.williams.edu/~jeannie/papers/kneedle-simplex11.pdf)。


## 2026-10-20 验收记录

- 最终 V1 + Qwen3.5 decoder CPU 回归：**315 passed，2 skipped**。覆盖明确膝点、
  无膝点、上下文干扰、P95 毛刺、分块置信度、TP 慢 rank、后台关闭时只标定、
  显式重调度兼容性、内存精确版本恢复与根端点原引用复用。
- 16 NPU 任务 `qwen35_9b_memory_roofline_20261020_v2` **SUCCEEDED**：2 次更新、
  64 条训练样本及 4 条验证，版本 0/1/2，事件审计一致。四个 TP2 replica 共收到
  **562 条**对齐后的纯 decode 样本，与两 rank 的 sampled 数逐实例一致，零探针错误。
- 有足够样本的 batch 档位主要为 8，另一个实例覆盖 2；所有 profile 正确返回
  `insufficient_coverage`，保留上限 8，**没有声称已找到真实硬件膝点**。
  不同 batch/context 全覆盖及 24K 标定仍需后续工作负载验证。
- 未创建模型 tensor 备份、catalog.json 或 partial_trajectories.sqlite。
  无数据库意味着这次只能做事件谱系审计，不能声称做过离线持久化 token/chunk 审计。
- 默认内存模式真实 Mooncake TCP 故障探针覆盖 intermediate/master/endpoint/root endpoint，
  两个精确版本和旧版本 GC 均通过；1 个物理节点、3 个逻辑 relay。catalog 不作自动重启。
- 256-token 任务验证系统执行和采集接口；不证明 24K 学习效果、RDMA/multi-node scaling
  或正式性能收益。该旧基线发布总耗时约 30–33 秒；后续发布优化已降至两次更新平均 12.36 秒，
  计时边界与实测限制见 [WEIGHT_PUBLICATION.md](WEIGHT_PUBLICATION.md)。

结果报告：`/tmp/verl-laminar-runs/qwen35_9b_memory_roofline_20261020_v2/final_memory_roofline_report.json`。


## 在线 KV/SSM 容量与默认值逐级探索

每个 worker 在探针安装时读取当前 engine 的 KVCacheConfig；这些页已经由当前模型、
dtype、TP、算子 workspace 和 NPU 可用显存的启动 profile 实际分配。
不根据模型名称查固定并发，也不通过短序列的占用比例外推 24K 容量。

对每个 TP rank 计算：

```
pages_per_request = sum(ceil(group.max_memory_usage_bytes(max_model_len) / group.page_size_bytes))
request_fraction = pages_per_request / (num_blocks - 1)
B_KV = floor(kv_usage_limit / max_rank(request_fraction))
```

共享 pool 的各缓存组计入总页需求。Mamba/SSM 使用引擎 spec 的驻留页契约，覆盖
none/align/all 和 speculative 预留；不能把 recurrent state 当成普通 attention KV token。
以 max_model_len 预留未来增长，结果会比只看当前上下文更保守。
独立工作区/状态缓冲的并发约束继续由启动时分配的 engine max_num_seqs 保证。
TP rank 缺失、重复或布局不一致时，容量明确不可用，不平均不同 rank 的容量。
首个真实请求前即应用可用的池上限，避免先发满默认 batch 再退避。

online_capacity 为每个 replica 单独维护当前并发。连续 3 次新鲜、足够负载的观测后，
按 1.25 倍向上探索；达到物理/引擎边界时停止。KV 超安全水位、scheduler 排队或
新发生的预占会立即乘以 0.75 退避。重复、过期、异常遥测不会推动增长；进入新
context/mode 层时从保守起点重新观察，探针窗口结束后停止主动增长。
两次可信 Roofline 估计产生 ready 后，其安全上限即时参与裁剪。

增长只可补入同一最新 policy version 的健康、非 drain batch；新版本、迁移源、
迁移预约和当前新发现的 KV ramp-down 都保持原批次关闭。下调不取消已有轨迹，
只限制后续 admission/迁入，并等待自然释放。故障接管也受当前并发上限约束。
迁移按目的设备 `current_usage + arrivals * destination_request_fraction` 保守核算，
不相加不同容量设备的源/目的归一化 KV 比例；无法取得目的布局时不授权迁移。

`kv_usage_limit=0.9` 是预留 10% 的工程策略，不是模型容量或观测出来的 Roofline。
物理 KV 池在引擎启动时分配，本实现在线调整 admission/迁移预算，不热扩容已分配显存。
KV 页预算和实测服务时间膝点共同限制并发；内存先成为瓶颈时，可能始终观测不到膝点。

观测：scheduler.jsonl 新增 capacity_update；replica_snapshot 新增 admission_limit 和
capacity（初值、当前值、物理上限、池描述、退避原因）。roofline_update 的 coverage
包含全部已观测档位、合格档位数、未满桶的缺样本数和独立时间块数。
这些数据只用于在线状态和聚合事件日志，默认不落盘恢复数据。

2026-10-20 新一轮验收：完整 CPU 回归 **367 passed、2 skipped**。
16 NPU / Qwen3.5-9B / 24K thinking 预算任务
`qwen35_9b_online_capacity_24k_20261020_v2` **SUCCEEDED**，一次训练更新，事件审计一致。
首轮任务因 launcher 多余参数在 Hydra 解析阶段退出；已修正，并通过完整配置检查。

- 四个 TP2 replica 均实际探索过 8 / 10 / 13 / 17 / 22 / 28 / 35 / 44 / 45。
  各 rank 可用页 2811，每条满上下文请求预留 55 页，90% 预算得到 B_KV=45；
  实际 active+reserved 没有超过 45，未发生 OOM。
- 共接纳 **8814 条** TP 对齐的纯 decode 样本；已覆盖 1–45 范围内多个 batch 档位，
  探针无错误。低上下文主要返回 no_observed_knee，另一个 profile 的分段断点
  没有通过 Kneedle。较长上下文仍有 insufficient_coverage，**没有任何 ready 膝点**。
  因此 45 只标记为物理页预算上限，不宣称是 Roofline 或有新的 P95 保证。
- 完成 59 条非探针响应，长度 2018–13001；
  32 条进入训练、52 条训练到达。训练选中响应均值 5016.78、最长 11496 token，
  无响应被 24K 预算截断。本轮没有验证集评估，组内 advantage 和梯度为零，
  不证明学习收益，也没有验证所有 24K 上下文层的标定。
- 发布版本 0/1，old_log_probs 使用真实 rollout logprob；没有未完成的权重 pull。
  有一次迁移 preflight、没有正式迁移。本轮未注入故障或制造缓存压力；退避分支由 CPU 回归覆盖。
- 未生成恢复模型文件、catalog 或轨迹 SQLite 文件；仅做在线状态和事件审计。
  结束时 121 条请求没有终态记录；100% 训练完成之后，CANN/TBE 守护进程清理出现
  ProcessLookupError 和 parent-disappeared 日志。Ray 作业退出成功，但退出清理仍须改进。

报告：`/tmp/verl-laminar-runs/qwen35_9b_online_capacity_24k_20261020_v2/final_online_capacity_report.json`。
采样开销 A/B、24K 各层充分覆盖、正式稳态吞吐、跨节点 RDMA 和 32/64 卡 scaling 仍待验收。


在线标定的受控硬件覆盖、开销配对和真实缓存压力验收方法见
[CALIBRATION_ACCEPTANCE.md](CALIBRATION_ACCEPTANCE.md)。覆盖通过与膝点辨识单独报告。


受控标定验收已完成：七个上下文层×九个 batch 覆盖通过，采样开销的 95% CI 上界均低于
5%，真实 KV 高水位退避通过。验收发现的无膝点 P95 保护缺口已修复，并通过独立真卡复验。
没有将缓存上限或经验延迟界标为硬件膝点；完整数据与适用边界见
[CALIBRATION_ACCEPTANCE.md](CALIBRATION_ACCEPTANCE.md)。
