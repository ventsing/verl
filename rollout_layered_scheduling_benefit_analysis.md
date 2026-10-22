# verl 分层调度方案理论收益分析

> 本文是 `rollout_layered_scheduling_design.md` 的姊妹篇:设计文档回答"怎么建",本文回答"**为什么值得建、收益有多大、什么时候不收益**"。所有机制引用均指向设计文档章节(§N)或已落地代码;实现状态以 2025-12 快照为准。

---

## 0. 分析框架:钱花在哪,浪费在哪

一次 RL 训练的墙钟时间可分解为:

```
T_wall = Σ_steps (T_train + T_rollout + T_sync) + Σ_jobs T_build

T_rollout = T_prefill_bulk + T_decode_bulk + T_tail      ← L1/L2 作用区
T_build   = 进程 + CUDA context + NCCL + 权重 + 编译      ← P4b 作用区
T_sync    = trainer → rollout 权重推送                    ← P4a 作用区
```

四类气泡,各自对应一层:

| 气泡 | 定义 | 出现位置 | 回收层 |
|---|---|---|---|
| A 波尾气泡 | 同一 wave 内,快完成副本空转等最慢副本 | 每个 generation wave 的最后 20-40% 时间 | L1(防放大)+ L2(直接回收) |
| B 相位气泡 | colocate 下,训练相位推理面闲 / rollout 相位训练面闲 | 每个 step 的相位切换 | L2 + §10.6 hybrid 镜像 |
| C 跨 job 气泡 | job A 的训练相位 = job B 的扩容窗口 | 多租户集群 | L3 broker |
| D 构建气泡 | 实例冷启动(进程/编译/权重)68-495s | 每次 job/副本创建 | P4b 共享池 |

**收益 = Σ 回收气泡 × 替代用途价值**。替代用途的价值排序:

```
他 job 的 rollout > 本 job 训练算力(colocate deactivate)> 其他 job > 节能
```

后续各节按层展开,每节给出:机理 → 形式化模型 → 数值例 → 收益性质。

---

## 1. L1(token 路由):preemption 概率模型

### 机理

vLLM 副本 KV 容量为 C( tokens),在途 token 负载 X_i 超过 C 触发 preemption:被逐请求**重算 prefill** 并掉入队尾——既付 O(L) 重算成本,又把该请求变成新的 straggler。verl 既有指标 `agent_loop/num_preempted` 度量的正是这一现象(设计 §7 已列为验收口径)。

### 形式化

每副本分到的请求集合为 S_i,X_i = Σ_{j∈S_i} L_j。

**count 路由**(现状):条数均衡 n_i ≈ n,负载方差只由单请求方差累积:

```
Var_count(X_i) = n · σ_L²
```

**token 路由**(P1):按 hint = prompt + response budget 做加权均衡,负载方差退化为 bin packing 残差,受单请求上界约束:

```
Var_token(X_i) ≈ (max_j L_j)²    (残差上界;一维首次适应的界更紧,见 [Garey-Johnson])
```

**数值例**:C = 100k tokens,n = 16,轨迹长 μ = 4k、σ = 6k(重尾,max/min ≈ 30×):

| 路由 | σ_X | P(X > C) = P(Z > (C−X̄)/σ) |
|---|---|---|
| count | √(16×36M) = 24k | P(Z > 1.5) ≈ **6.7% / 副本 / wave** |
| token | ≈ 8k | P(Z > 4.5) ≈ **≈ 0** |

两个补充性质:

1. **hint 是上界估计**(实际生成 ≤ budget)→ 均衡偏保守,不会欠分配;系统性漂移由 `rebase_loads` 治愈(设计 §3.4);
2. **无 hint 请求退化 hint=1** → 与现状完全等价,收益下界 0(见 §8)。

### 收益性质

L1 不直接缩短 wave,而是消除"preemption → 重算 → 掉队尾"的**正反馈尾部放大**。这是乘性效应:一次 preemption 既是 O(L) 重算成本,又制造新的 straggler,喂给 L2 更难处理的尾。消除放大项后,L2 面对的尾部分布更干净、可预测。

---

## 2. L2(波尾收缩):makespan 与机会成本模型

### 浪费结构

wave 完成时刻 T = max_i B_i(B_i 为副本 i 的批量完成时刻)。尾部闲置:

```
W_tail = Σ_i (T − B_i) · g_i        (g_i = 副本 i 的 GPU 数)
w = W_tail / Σ_i T · g_i            (尾部浪费率)
```

重尾分布(20% 轨迹长 3-5×)下,w 典型 **30-50%**(Laminar Fig. 9 的 KV 生命周期 ramp-down 段即此)。

### 收缩的净收益条件(设计 §4.5 的形式化)

L2 将 n 个副本收缩到 k 个 keeper,当且仅当:

```
(n − k) · E[T_tail_remaining] · V_alt
    >   Σ_{victims} residual_tokens · (1 − h) / prefill_tps    (re-prefill 成本)
      + last_deactivate_duration_s                              (切换代价)
```

其中 h = prefix cache / Mooncake 命中率,V_alt = 替代用途单位时间价值(§0 排序)。

**关键结构性收益**:迁移发生在**同版本组内**(§4.4 重试解耦后),成本只有 re-prefill——没有权重语义代价、没有样本损失、不依赖 partial_rollout。这是相对 Laminar repack 的同等条件(Laminar §5 步骤①同样按权重版本分组)。

### 门控保证:收益下界为 0

| 门 | 不满足时 | 后果 |
|---|---|---|
| G1 full_determinism | 整体拒绝 | 无收益,无损失(正确性优先) |
| G2 n > min_keep | no-op | 单副本/已收缩到位时零动作 |
| G3 波尾条件(count≤K 且 kv_util≤U,**全部**副本) | no-op | 防单点噪声触发全局动作 |
| 版本组内无 keeper | 降级 drain-wait | 不 abort,自然排空 |
| net-benefit ≤ margin | no-op | 不值得就不动 |
| 迁移失败/超时 | 回滚(步骤 e) | 退回现状,无半迁移状态 |

**L2 的期望收益 = Σ P(门全过) × 收益(该场景),每项的下界都是 0 而非负。**

### Amdahl 上界(诚实估计)

设 rollout 占 step 时间的 60%、其中尾部浪费 35%,则尾部回收最多贡献 f = 0.21:

```
S_step ≤ 1 / (1 − 0.21) ≈ 1.27×
```

L2 单独不是 5× 故事(见 §8 与 Laminar 对比)。但它解锁 L3 的跨 job 复用(§3),且 colocate 模式下释放副本直接缩短 T_train(deactivate 既有流程),实际系数优于独立 rollout 池形态。

### 数值例

8 副本 wave,尾部 40s,2 个 victim 各剩 1 条 30s 长尾:

- 自然排空:16 GPU × 40s = 640 GPU-s 闲置;
- 收缩到 6 keeper(迁移 2 条长尾,各 re-prefill 2k tokens ≈ 1s @ 2k tok/s):
  victim 释放提前 ~35s → 回收 16 × 35 = 560 GPU-s,代价 2 × 1s × 8 GPU = 16 GPU-s(keeper 侧 prefill),净 ~**544 GPU-s / wave**;
- 若 colocate 且训练面可即时吃下:这 544 GPU-s 直接换成训练算力。

---

## 3. L3(共享池):统计复用 + 冷启动摊销

### (a) 构建气泡摊销(P4b 已落地部分)

每次副本构建 B ≈ 68-495s(进程+CUDA+NCCL+权重+编译),池化后切换 s ≈ 10-60s(同基座、编译缓存命中)。每小时回收:

```
GPU-s / h = λ_jobs × n_replicas × g × (B − s)
```

**数值例**:λ = 6 jobs/h(评测/消融/小 finetune 典型节奏),4 副本 × 8 GPU,B = 300s,s = 30s:

```
6 × 4 × 8 × 270s = 51,840 GPU-s/h ≈ 持续 14.4 张 GPU 的等效算力
```

对短 job 尤其重:5 分钟评测 + 5 分钟构建 = **50% 的开销直接归零**。池生命周期越长、job 越短,摊销越彻底(池寿命 → ∞ 时构建成本 → 0)。

### (b) 统计复用(跨 job 气泡 C)

m 个 job 各有占空比 p 的 bursty 需求。专用配置需要 m 份峰值容量;池化容量由峰值需求和相关性 ρ 决定:

```
专用容量 = m × peak        池化容量 ≈ m·p + z·√(m·p·(1−p)) · √((1+(m−1)ρ)/m) + 驻留余量
```

数值例(m = 8, p = 0.5):

| 相关性 | 池化需求 | vs 专用 8 份 |
|---|---|---|
| ρ = 0(不相关) | ≈ 6.8 | **−15%** |
| ρ < 0(相位错开,async job 天然如此) | → 4 | **−50%** |

RL job 有相位结构(训练相位不需要 rollout GPU),broker 看到的是真实租约需求而非峰值配置——**把"按峰值配"改为"按均值+波动配"**的经典统计复用。job 间相关性越差,增益越大;极限情形(全部同步相位)增益为 0。

### (c) 切换成本分层(驻留登记 + 编译缓存)

broker 记录 actor_group → (model, version, loaded_at),放置优先级:

```
同版本原样复用 0s  >  同基座 switch_model 10-60s(编译缓存命中)  >  跨基座 30-180s(可能 + 编译)
```

编译缓存 hash **不含权重值**(multi-task-infer 设计),同架构不同权重必命中——缓存粘性让放置策略天然偏向同架构 job 群。**多租户同基座**(同一 base model 的多组超参/消融实验,RL 调参的典型形态)是甜点场景:基座只装一次,后续 job 全走 10-60s 通道。

### 驻留的显存代价(完整记账)

空闲副本持有权重(GB 级)而非 sleep/offload,是**用显存换 10-60s 切换**。LRU 驱逐(按 loaded_at + 池显存水位)在压力下归还。这是 L3 的经营决策项,不是免费收益。

---

## 4. P4a(存储权重通道):同步解耦

**现状(trainer-push)**:n 副本 × 串行 ZMQ/bucketed 推送在训练关键路径上;trainer 必须等全部副本确认。

**存储通道**:trainer **写一份**到 NFS/Mooncake(RDMA ~9 GB/s)即返回,n 副本各自**并行**拉取:

```
T_sync:  O(n × 推送)  →  O(1 写 + max_i 拉取_i)
```

同步时间下降之外的两个结构收益:trainer 不必为慢副本在线等待(解耦);与 P4b 组合时,新入池副本的权重供给不再依赖 trainer 在线(§10.4 集成点④)。

---

## 5. 重试解耦(已实现):使能性收益,条件概率视角

这一项的收益不是直接吞吐,而是**解锁整条 L2/L3 链的前置条件**:

```
P(系统可用迁移 | 未解耦) = P(partial_rollout = True)     ← 仅 async 训练形态
P(系统可用迁移 | 已解耦) = 1                              ← 由 drain 状态背书
```

未解耦时,sync 训练(partial_rollout=False)使用迁移会**丢样本**——不是慢,是训练数据静默损坏。解耦把迁移许可从"算法容忍度"(partial_rollout)移到"路由层状态"(LB `is_request_migrating` 查询):

- 一次 RPC,仅在 abort 路径(正常完成零开销,有热路径测试守护);
- 查询可靠性有结构保证:sticky 惰性失效恰好发生在下一次 acquire,查询窗口在 abort 返回与重试之间,条目必然仍指向 victim;
- 保守回退:stock LB 恒 False、自定义 LB 无方法 → 行为与现状逐字节一致。

**收益语义**:L2/L3 的全部收益(§2 §3)从"仅 async 训练可用"推广到"任意训练模式可用"。乘性而非加性:

```
E[系统收益] = P(可用) × E[收益 | 可用]    ← P(可用) 从 p(async) 升到 1
```

---

## 6. 质量/正确性收益:不是收益,是前提,但值得估值

### (a) on-policy 度(staleness)

释放的副本更快拉取最新权重 → inherent staleness k ↓(设计 §4.3 已补:权重更新完成后立即触发收缩)。重要性比 π_{θ_t}/π_{θ_{t−k}} 的方差随 k 增长;经验上 k 小则 off-policy 梯度估计行为良好。Laminar 实测 inherent staleness 稳定 **< 3**(其 Fig. 10)。

**收益语义**:同样的 GPU 时间产出更 on-policy 的样本——这是**收敛速度**轴上的收益,墙钟吞吐之外的第二次轴。两者不可直接换算,但对最终"达到目标奖励所需总 GPU 时"都有效。

### (b) 红线消除静默污染(期望值视角)

dummy 权重副本服务一个 wave = 数千条带错误 `global_steps` 的样本进 replay buffer → 梯度偏置,**直到收敛发散才可能被发现**(损失量级:GPU-周)。红线(fill 必填、失败即抛、打标+清 KV,设计 §10.5)把这类故障从"静默损坏"转为"秒级 fail-fast":

```
E[红线价值] = P(填充故障) × cost(静默污染) − cost(fail-fast) ≈ P × GPU-周 − GPU-秒 ≈ P × GPU-周
```

这是负收益转零的保险,不是收益项;但方案总价值必须计入——尤其共享池引入了权重不在 trainer 直接控制下的新故障面,红线的边际价值随 L3 上线而**上升**。

---

## 7. 汇总:收益矩阵

| 层 | 收益机制 | 量级(理论) | 依赖条件 | 状态 |
|---|---|---|---|---|
| L1 | preemption 概率 ↓(方差结构改善) | P(preempt) 6.7%→≈0;尾部放大消除 | 重尾负载、多副本 | **已实现** |
| P4a | 权重同步 O(n)→O(1)+并行拉 | 同步时间显著 ↓(RDMA 9 GB/s);trainer 解耦 | 存储可达 | **已实现** |
| P4b | 构建气泡摊销 | ~14 GPU 等效/集群(示例参数);短 job 50% 开销归零 | 池寿命长、fork 分支 | **已实现(骨架)** |
| 解耦 | 迁移对 sync 训练解锁 | P(可用迁移) 从 p(async) → 1(乘性) | — | **已实现** |
| L2 | 尾部机会成本回收 | step 上界 ~1.27×(典型参数);净 ~544 GPU-s/wave(数值例) | 有替代用途、net-benefit 过线 | 设计(P3) |
| L3 | 统计复用 + 驻留复用 | 容量 15-50% ↓(相关性依赖);切换 0-60s | 多 job、broker | 设计(P4c) |
| 质量 | staleness ↓ + 红线保险 | k<3 量级(锚点);E[保险] ≈ P×GPU-周 | 权重更新触发收缩 | 设计(随 P3)/已实现(红线) |

典型部署形态的合成(凸组合,权重由场景决定):

- **多租户同基座、短 job 群**(评测/消融):构建摊销 + 统计复用主导 → 集群吞吐 +30-50%;
- **单 job colocate、重尾负载**:L1 + L2 主导 → step time −20-30%;
- **async 多 job、独立 rollout 池**:全链生效 → 上限最高,但依赖 P3/P4c 落地。

---

## 8. 理论上界与无收益场景(诚实边界)

Amdahl 纪律:本方案**不是** 5.48× 类故事。Laminar 的 5.48× 来自彻底打破全局同步锁步(架构级:relay 层让 rollout 随时拉权重),而 verl 已有 async 模式,本方案是在既有架构内的分层改进。逐项无收益场景:

| 场景 | 无效层 | 原因 | 系统行为 |
|---|---|---|---|
| 单副本 job | L1, L2 | 无路由选择、无 keeper/victim | G2 门 no-op |
| 轨迹长度均匀 | L2 | 无尾可收 | G3 门 no-op |
| GPU 独占 + 无他 job + 非 colocate | L2, L3 | 释放副本无替代用途 | 仅省能耗(若计量含成本仍是收益) |
| full_determinism | L2, L3 | hash 路由要求稳定成员集 | G1 整体 gate(正确性换性能的显式选择) |
| 超长 prompt 无 prefix 共享 | L2(部分) | re-prefill 成本可能超过尾收益 | net-benefit 门 no-op |
| 全同步相位 job 群 | L3(b) | ρ→1,统计复用消失 | 池退化为共享构建(§3(a) 仍有效) |
| PyPI vLLM(非 fork) | P4b | 无 external_actors 插座 | 启动 fail-fast,带分支指引 |

**所有门控的语义统一为:不满足则收益下界为 0,不为负。** 方案的期望收益是各场景收益的凸组合,由部署形态决定权重——不存在"开启后反而变慢"的参数区间(除 §9 的固定开销项)。

---

## 9. 引入的成本(完整记账)

| 成本项 | 量级 | 性质 |
|---|---|---|
| 解耦查询 RPC | +1 RPC,**仅 abort 路径**(热路径零开销,有测试守护) | 已实现,固定小 |
| L2 tick | 每 1-5s 两个快照 RPC | 决策频率 << 请求频率,可忽略 |
| 驻留显存 | 空闲副本持权重 GB 级 | 用显存换 10-60s 切换;LRU 驱逐兜底 |
| 迁移 re-prefill | residual × (1−h) / prefill_tps | 已入净收益公式,不白付 |
| 协调复杂度 | membership lock、drain-wait、回滚路径 | 真实故障面;换来每次动作可回滚(失败模式=退回现状,非半迁移状态) |
| broker 单点 | 控制面集中 | 数据面直连不经 broker;TTL 心跳兜底 |
| 编译缓存本地 miss(当前 P4b 边界) | 各副本本地编译 | fork EngineCore 未透传 cache_manager;P4b-next 修复 |

---

## 10. 一句话总结

L1 消除尾部放大(方差结构),P4a/b 消除构建与同步气泡(摊销+解耦),重试解耦把迁移能力从 async 专属推广到全域(乘性使能),L2/L3 回收时间线上本就存在的 30-50% 尾部与相位闲置(机会成本)——**每一层的收益下界都由门控保证为 0 而非负,上界由部署形态决定**;典型参数下 step 级 1.2-1.3×、集群容量级 15-50%、短 job 构建开销 50%→0,staleness 轴另有收敛收益。

---

## 参考

- 设计文档:`rollout_layered_scheduling_design.md`(§3 L1、§4 L2、§10 L3、§10.5 红线)
- Laminar: A Scalable Asynchronous RL Post-Training Framework(EuroSys'26,arXiv:2510.12633)——同版本主动迁移(repack)、inherent staleness <3、5.48×(架构级上界参照)
- Garey & Johnson, Computers and Intractability —— bin packing 残差界
- verl 既有指标:`agent_loop/num_preempted`、`timing_s/gen`、`DynamicScheduleContext.last_deactivate_duration_s`
