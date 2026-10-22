# 在线标定验收

本验收区分机制正确、覆盖充分、膝点可辨识和测量开销。没有可信膝点时保留否定结果，
不把 KV 预算或引擎上限写成 Roofline。验收使用独立受控负载，不向训练经验队列写数据。

## 受控真实硬件矩阵

工具：仓库根目录 `probe_online_calibration.py`。当前模型 Qwen3.5-9B BF16，
TP2、enforce_eager、chunked prefill、无 prefix cache，与当前训练 rollout 的关键路径一致。
每个 TP2 实例单独拟合，不混合不同 replica 的设备时间。

- 上下文：1024、5120、9216、13312、17408、21504、25088 token，覆盖七个 4096-token 层。
- batch：1、2、4、8、16、24、32、40、45。
- 每个矩阵点强制生成 512 token、忽略 EOS，两轮按相反 batch 顺序运行。
- 所有矩阵点先检查该实例真实 KV/SSM 满上下文页预算；超预算明确失败。
- 只将 TP 对齐的纯 decode 样本交给生产同一 OnlineRoofline；prefill、编译、warmup
  仍由生产探针过滤。第一轮和第二轮分别拟合，保持默认样本/置信度阈值。

每个目标层必须具备全部计划 batch 的有效桶和至少 8 个独立时间块，覆盖才算通过。
覆盖通过与 ready 膝点是两个独立结果。no_observed_knee、Kneedle 未确认或 CI 不稳定
都会如实记录，不能放宽阈值来制造膝点。结论仅适用于当前模型、TP、算子和硬件范围。

## 开销配对

在 1K、13K、25K 三个上下文的同一引擎、相同 batch=16 和输入上交替执行 off/on、
on/off，最终八组配对。全部引擎初始化完成后统一开始暖机，避免将其他引擎启动干扰混入开销。引擎和 kernel 已暖机，prefix cache 始终关闭；原生统计记录器两边相同。
比较完整请求墙钟和连续完整 batch 的纯 decode 客户端输出间隔，后者不是设备事件时间。
prefill 边界、非完整 batch 和混合步不进入 decode 间隔统计。

off 仍保留一次 wrapper 分支调用，跳过采样工作；因此这里测量采样的增量开销，
不是完全删除所有扩展代码后的框架基线。必须验证 on 有实际设备事件样本、off 没有，
且配对 greedy 输出 token 的 SHA256 完全相同。

生产探针的每桶上限可能使标定结束后的采样停止。因此每组独立测试通过 harness 的
字符串 RPC 的 fresh_measurement 参数清空测量桶/计数，保留权重版本、step 和 shape warmup；不修改生产 controller，
也不重开失败或超过版本窗口的探针。原初轮开销数据没有强制这个条件，不作为最终验收依据。

按整个配对 run 做 bootstrap，而不是把相关的相邻 decode 步当独立重复。
默认工程容忍度 5%：95% CI 上界不超过 5% 才通过；下界超过 5% 为失败，其余为不确定。
记录生产拟合器在真实样本上的 CPU 耗时；不将 CPU 时间直接等同于 GPU 吞吐损失。
这不替代完整 RL、repack 或 scaling 性能 A/B。

## KV/SSM 压力退避

隔离的 TP2 引擎使用受限 logical cache pool；保留 vLLM 原生 allocator 和 preemption，
不制造物理 HBM OOM。测试才会有意超过应用层保守 admission 预算，生产路径仍执行上限。

首轮 256 页池：实际使用率最高 83.137%，没有预占；排队触发在线 governor 4→3→2→1，
之后健康观测恢复到 4。此结果只证明真实排队退避，不证明 90% KV 高水位分支。

复测改为 224 页池，按原生真实水位激活受测 governor，保留原生 queue 数值，
确保观察到超过 90% 的实际 KV 使用率或真实预占，记录及时退避与后续有界 admission。
这用于隔离高水位分支，未改变生产 always-on governor 的启用时机。
原生未出现预占时，不宣称已经验证真实预占分支；已有 CPU 用例仍覆盖该分支。

## 运行与产物

现有 Ray cluster 的 NPU 必须空闲；工具只分配自己的 actor，不重启 cluster。
默认七个上下文引擎和一个压力引擎共用 16 NPU。

```bash
python3 -u probe_online_calibration.py --output-dir /tmp/calibration-acceptance

# 独立复测开销及高水位，使用 8 张空闲 NPU。
python3 -u probe_online_calibration.py \
  --output-dir /tmp/calibration-followup \
  --contexts 1024 13312 25088 --batches 16 --benchmark-only --pairs 8 --no-pressure
```

progress.jsonl 和 report.json 记录配置、聚合分桶、拒绝原因、配对结果及压力反馈。
不保存原始 decode 样本、模型备份或轨迹恢复数据库。
默认推理执行为 eager，本验收不推断 CUDA/ACLGraph、其他 TP、模型或硬件上的结果。

当前轮次产物：

- 覆盖矩阵：`/tmp/verl-calibration-acceptance-20261020`。
- 初轮复测因回调序列化限制退出，没有有效配对结论；已改用字符串 RPC。
- 有效采样的独立开销/压力复测：`/tmp/verl-calibration-followup-20261020_v2`。
- v2 有效采样复测：13K 开销判据通过，1K/25K 的 95% CI 上界超过 5%，按不确定记录。
- 统一初始化屏障后的开销复测：`/tmp/verl-calibration-followup-20261020_v3`。
- 最终汇总报告：`/tmp/verl-calibration-final-20261020/report.json`。


## 验收发现的 P95 保护缺口

13K 矩阵中 B=45 的 P95 约为 B=1 的 1.355 倍，而膝点拟合返回 no_observed_knee。
原路径仅在可信膝点之后施加 P95 界，无法独立保护这类工作点。现新增
EmpiricalLatencyGuard：相同分层两次稳定且有新观测后单独生效，经同一 limit() 接口
约束 online capacity 和 repack；物理膝点状态仍保持未辨识。

使用两轮真实聚合数据和采样修订号回放，13K 的经验工作点上限为 40，其余六层为 45。
不达标的中间 shape 同时记录，不能宣称所有小于该工作点的 shape 都满足 25%。
此处是原始硬件测量的聚合回放与 CPU 集成验证，不冒充修复后的完整 RL 真卡重跑。
新 guard 的长期运行、重调度净收益和不同硬件布局仍需后续性能验收。


## 最终结果

受控机制验收通过，膝点辨识仍为“观测范围内未确认”。

- 16 NPU、七个上下文层、九个计划 batch、两轮反向顺序，63 个计划点全部覆盖通过。
  共 12042 条 TP 对齐纯 decode 样本；各层保留 64–78 个独立块。
- 真实 KV 高水位最高 95.067%，在 92.825% 和 94.619% 处记录到 3→2→1 退避；
  后续 admission 为 3，实际 KV 最高 71.300%。未发生 native preemption，不能称该分支已真机验收。
- 新增独立 P95 保护的 13K 真卡复验完成三轮，六个计划 batch 覆盖通过。
  `latency_guard=ready`，上限 32，仅取已观测工作点；膝点仍为 no_observed_knee。
- 完整 CPU 回归 **378 passed、2 skipped**；Ruff、语法和 diff 检查通过。

有效采样对照（每层八组；负开销按噪声解释，不声称加速）：

| 上下文 | decode 间隔中位开销 | 95% CI 上界 | 完整请求中位开销 | 95% CI 上界 |
|---|---:|---:|---:|---:|
| 1024 | 0.024% | 2.043% | 0.178% | 1.338% |
| 13312 | 0.152% | 1.385% | 0.011% | 0.737% |
| 25088 | -0.820% | 2.262% | -0.485% | 1.627% |

原矩阵拟合 CPU 单次 4.175–22.047 ms；新增 guard 的真实拟合为 2.574–7.348 ms。
不把这些 CPU 时间直接换算为训练吞吐收益。结论范围为本模型、BF16、TP2、eager、
Mamba cache mode=none；图模式、align/all/speculative、其他 TP 和长期 RL 调度效果仍待验收。
