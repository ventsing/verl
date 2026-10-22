# Laminar 论文对照与后续开发清单

核对日期：2026-10-20。范围是 `trainer_mode=laminar` 的真实 vLLM/Ascend
执行路径；CPU demo、P2P mock 和未接入当前任务的适配器不算已经完成的生产能力。
参考 [Laminar 原文](https://arxiv.org/html/2510.12633v1)，重点是 §§3–8。
下列优先级是本项目的开发建议，不是论文规定的实施顺序。

## 已实现并经过真实硬件验证

16 NPU、Qwen3.5-9B、thinking、24576 response token 的任务完成 4 次更新和
8 条验证响应，Ray 状态 SUCCEEDED，退出码 0。以下机制已验证：

- 单轨迹写入经验队列；GRPO 保留完整组，够一个训练 batch 即采样更新。
- prompt 生产与 actor 训练独立运行，队列有背压。
- actor 发布 CPU 权重快照，不等待全体 rollout 完成或统一更新。
- rollout 按自身 batch 边界独立拉权重；同一时刻允许不同副本使用不同版本。
- 同版本重调度，保留已生成 token、behavior logprob、剩余长度预算和请求身份。
- 周期性及权重发布后触发 repack；已接入真实 KV/running/waiting 指标，
  使用 Best-Fit、目标容量预约和完成/取消竞态处理。
- 每条轨迹的 policy version 固定；old_log_probs 直接使用实际 rollout logprob。
- 最终事件审计通过，43 次正式迁移，15 次拉权重时其他请求仍在运行。

这些是系统执行和一致性证据，不等于复现了论文的吞吐收益或收敛结果。

## 需要开发或补齐的项目

P0 表示跨节点 scaling 或可信性能对比的前置工作；P1 表示论文完整能力和
长期运行的重要工作；P2 表示后续任务/并行布局扩展。

| 优先级 | 项目与论文依据 | 当前真实实现 | 后续开发及验收 |
|---|---|---|---|
| P0，跨节点时 | 分层 relay 与跨机器权重分发，§§4.2、7 | 原 `cpu_relay` 仍单节点；新增 `mooncake_relay` 已实现节点 relay、原生分块链式传播和本地加载，并通过同机三进程 TCP 探针；跨节点/RDMA 待验收 | 见 [Mooncake 部署记录](MOONCAKE_RELAY.md)。下一步在实际多节点上验证 RDMA、独立 pull 和网络吞吐；仍需部署兼容依赖、补 TP 预重分片及节点失联后的链重配置，再报告 32/64 卡性能。 |
| P0 | 并发与容量标定，§5.2 | 在线 Roofline 已接入纯 decode 设备采样、上下文/模式分层、分段拟合、Kneedle、置信度和目的端容量；新增按实际 KV/SSM 页预算的在线容量控制，从默认 batch 逐级探索、遇压力退避，覆盖度诊断和延迟拟合刷新已接入 | 见 [ONLINE_ROOFLINE.md](ONLINE_ROOFLINE.md)。已完成 16 NPU 七上下文层×九 batch 的受控覆盖、有效采样开销配对和真实 95% KV 高水位退避，见 [CALIBRATION_ACCEPTANCE.md](CALIBRATION_ACCEPTANCE.md)。无可信膝点，保持否定结果；新增独立 P95 界并通过三轮真卡复验。仍需图模式、其他 TP、align/all/speculative 与长期真实 RL 调度性能验收。保留当前 ramp-down 状态记忆适配，并与论文逐轮下降判据进行对照。迁移已改为目的端页需求投影，不再相加不同容量设备的归一化比例；更多 TP/布局仍需验收。 |
| P0 | 性能采集和消融，§§6、8 | 有版本、拉取、迁移和训练计时日志；只有短任务执行证明，没有正式 A/B | 记录 actor update 完成时间；分别统计轨迹完成时的 inherent staleness 与训练采样时 staleness。补 NPU 利用率、队列深度、等待完整组的时间、KV 均值及迁移全程成本。做 repack on/off、异步/同步权重路径对照；明确冷启动与验证耗时的处理。按论文方法固定全局 batch 做 strong scaling，训练/推理比例单独标定和报告。论文性能实验先 warmup 10 次，再平均 5 次；当前 4 步不满足该测量协议。 |
| 已完成当前单轮范围 | 进行中轨迹的 partial response pool，§§3.1–3.3、7 | 独立 CPU actor/SQLite :memory: 默认保存流式 token/logprob 增量、固定版本、序号、epoch 和预算；支持确认、幂等、旧写入 fence 与最终结果重放；落盘默认关闭 | 见 [PARTIAL_RECOVERY.md](PARTIAL_RECOVERY.md)。CPU/Ray 进程故障探针、16 NPU 真 vLLM 终止及 8 条轨迹接管已通过；首轮累计列表别名竞态已修复。默认内存服务须存活；终态记录清理、持久化/异步备份和完整 checkpoint 一致恢复均为遗留事项。 |
| 部分完成，P1 | rollout/relay 故障恢复及弹性，§§3.3、4.3 | 已有 heartbeat/停滞检测、失败副本隔离、健康实例按原版本接管，必要时重载保留的旧权重；新增版本数/字节预算，发布前背压并核算 Mooncake staging | relay 进程级链重建、同节点 native master 主备已实现；catalog 默认内存，显式磁盘模式支持 catalog 重建，见 [RELAY_RECOVERY.md](RELAY_RECOVERY.md)。仍需同卡自动重建 TP 实例、动态副本/节点资源管理、节点失联后的跨机器重配置，以及整个 Ray 集群的一致恢复。vLLM 驻留旧权重作为第二备份来源已记录，尚未实现从实例导出重建 relay。 |
| P1 | 低开销迁移，§§5.1、8.4 | 同版本 abort + prefix re-prefill；会重算已有前缀来恢复 KV 和 Qwen3.5 recurrent state；只有迁移计数、prefix 长度和部分首 token 计时 | 先测 abort 到目的端首个新 token 的全程延迟、重算 token 数、释放副本数量及净吞吐收益，再决定实现缓存/状态传输。若做直接迁移，需要覆盖 KV、SSM、调度器和采样续接状态，并通过 token/logprob/预算一致性及错误回滚测试。论文没有给出 Qwen3.5 SSM 迁移接口；不能把原生 KV/SSM 传输写成论文明确要求或已完成能力。 |
| 部分完成，P1 | 权重发布与加载的内存/拷贝优化，§§4.2、7 | 已实现有界并行 SHA256、同版本不可变源引用的校验复用、发布各阶段计时，见 [WEIGHT_PUBLICATION.md](WEIGHT_PUBLICATION.md)。各 FSDP rank 仍参与 full_tensor，rank 0 导出完整 CPU 张量；消费端复制并加载，尚无 relay 侧 TP 预先 reshard | 评估从 actor 分片到 relay 的导出方案、relay 侧 TP 布局转换、预分配 pinned shared memory、分块与复用；支持超过单 bucket 的大张量。区分 export、reshard、broadcast、pull、device load 的耗时，避免将不同硬件上的论文绝对秒数直接当验收门槛。 |
| P1 | 检查点与一致恢复，§3.3 | 继承 verl 的 actor/optimizer/dataloader 和可选 TQ checkpoint/reissue；当前 launcher 关闭保存及 resume；initial_experience_dir 仅是版本 0 的诊断回放 | 联调非零训练版本的保存/恢复，恢复权重版本服务、已完成经验、进行中任务及去重状态；trainer 恢复时让健康 rollout 保持运行。不能把现有的 prompt 重新派发或完成经验备份当成原轨迹部分恢复。验收一次真实更新后中断、恢复、继续更新，并审计版本与数据使用。 |
| P2 | 工具调用、多轮与更多并行布局，§§3.1、8 | 当前入口是 single_turn_agent；迁移拒绝多模态，PD/engine DP>1/PP>1 也被拒绝 | 按任务保存工具调用结果、会话/环境恢复信息与生成阶段；定义状态迁移接口，再拓展并行布局并验证跨 rank 协议。它们不是当前单轮 16 卡流程的阻塞项；DP/PP/多模态也不应一概称为论文必须实现的功能。 |

## 暂缓开发：可靠性落盘

用户于 2026-10-20 要求先不落盘，以免可靠性开销抵消异步收益。默认关闭 relay
整模型备份、catalog fsync 和轨迹 SQLite 文件，保留内存版本/引用及 fencing。
遗留事项：后台持久化队列及确认语义、备份前内存保底、跨故障域副本、完整任务恢复。
可选同步磁盘路径仍保留，**异步备份尚未实现**；它不再是当前优先开发项。
另需缩短目前约 30 秒的全 native 链重建，验证局部恢复的旧写入隔离。

## 模型效果验证：与系统功能缺口分开记录

本轮验证 1/8 正确，7/8 达到 24576 上限；没有同配置的初始模型验证对照。
后续需要固定验证集的初始/训练后对照，增加样本量，统计答案是否完成、EOS、
截断、重复与分组奖励方差，再决定延长训练或调整采样、数据和奖励。
这些是学习效果与实验设计工作，不能直接归因于 Laminar 异步机制，也不能仅
根据训练 loss 下降宣称已经复现论文收敛。

## 建议实施顺序

1. 保持 16 卡，补并发标定、真实 repack A/B、完整指标和模型验证对照。
2. 若 32/64 卡跨节点，先验证现有分层 relay 的双节点放置和真实 RDMA，再补节点失联重配置。
3. 验收已开发的 partial response pool 与健康实例接管，继续同卡重建、relay 恢复提速和 checkpoint 联调。
4. 在统一协议下运行 16/32/64 卡 strong-scaling 测量及长期训练；将性能对照与
   容错注入实验分开，避免混入不同工作负载。
5. 根据迁移成本测量决定是否开发原生 KV/SSM 传输，再扩展工具调用等任务。

论文的采样策略扩展在 §6 被明确列为正交的后续方向，因此优先经验回放或修改
GRPO advantage 并非复现本论文系统设计的必做项。

## 代码定位

- `laminar_controller.py`：版本固定、拉权重边界、Best-Fit、预约和重调度。
- `cpu_relay.py`：单节点 CPU 快照、版本发布及加载 lease。
- `prompt_feeder.py` / `tq_sampler.py`：生产背压、轨迹到达及组装采样。
- `../../trainer/ppo/v1/trainer_laminar.py`：单节点限制、训练 hook 和保存/清理。
- `../../../monitor_qwen35_longtail.py` / `../../../audit_laminar_run.py`：
  当前观测和只读审计；需要在这些记录上补齐正式性能报告。
