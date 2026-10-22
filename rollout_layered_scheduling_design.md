# verl Rollout 分层调度设计提案

> 状态:设计稿(未实现)。基于 main 分支当前代码(截至 `a0bd149a`)。
> 姊妹篇:`rollout_layered_scheduling_benefit_analysis.md` —— 本设计的理论收益分析(机理建模、数值例、Amdahl 上界、无收益场景与完整成本记账)。本文回答"怎么建",彼篇回答"为什么值得建、收益有多大、什么时候不收益"。
> 讨论脉络:请求级 least-in-flight 均衡 → KV 层面不均衡 → 副本粒度"波尾搁浅/让渡" → 分层架构 → **多 RL 任务共享 rollout 池**。
>
> 关联材料:`~/source/opensource/ai/llm/vllm/multi-task-infer`——vLLM ExternalExecutor 插件
> (Actor 池化 / 模型热切换 / 编译缓存共享 / 存储后端权重加载),已实现且与 verl
> `CheckpointEngineWithCache` 接口兼容。本设计将其定位为 L3 的物理基础设施层(§10)。

---

## 0. 摘要

把 rollout 调度显式拆成四层,每层独立的决策对象、时间尺度、信号精度和失败代价:

- **L1 路由层**(已有,升级):`GlobalRequestLoadBalancer` 从"数请求"升级为"数 token",新增 drain 状态与负载校准;
- **L2 编排层**(新增):事件驱动的波尾收缩(tail consolidation)与副本让渡,复用现有 abort+retry 迁移机制;
- **L0**:引擎内部调度不动;
- **L3 租户层**(本轮设计,§10):多 RL 任务共享 rollout 池,分两个子层——**L3a 容量层**(SharedRolloutBroker:跨 job 租约/公平/计量)与 **L3b 基础设施层**(multi-task-infer 的 ActorPoolManager / 热切换 / 编译缓存 / StorageCheckpointEngine,已实现)。L2 波尾收缩的输出(drain 完的副本)直接作为 L3b 热切换的输入。

核心设计原则:

1. **改视图,不替路由**:L2 只通过成员集 / drain 集 / 权重偏置影响 L1,永不逐条指定请求去向;
2. **快层近似,慢层校准**:L1 用本地 token 记账(内存操作),L2 用引擎实测周期性 rebase;
3. **正确性由不变式兜底**:abort+retry+token 预算语义不变,调度只影响性能,不影响样本正确性;
4. **默认即现状**:所有新行为 opt-in,关闭时精确退化为当前行为。

---

## 1. 问题定义

| 编号 | 问题 | 根因 | 归属层 |
|---|---|---|---|
| P1 | KV 层面不均衡:least-in-flight 均衡请求条数,RL rollout 的 KV 占用重尾 | 计数是一维投影 | L1 |
| P2 | 波尾搁浅:burst 均分后长尾散布,每副本 1-2 条 in-flight 但利用率极低;"计数看不出来"导致副本无法让渡 | 缺少残余工作量信号 + 让渡原语 | L2 |
| P3 | 编排只有步驱动:deactivate/activate 在训练步边界触发,看不见波尾事件 | 缺事件驱动回路 | L2 |
| P4 | 多 RL 任务共享:多个训练 job 的 rollout 资源无法互通,各 job 独占的池在波尾闲置 | rollout 池是 job 内组件,无跨 job 租约/驻留登记/公平性机制 | L3 |

非目标:引擎内部调度(L0 的 continuous batching / preemption / prefix cache)、RL 正确性语义、PD 分离架构本身。

关键事实依据(现状代码):

- 迁移机制已存在:`FullyAsyncLLMServerClient` 捕获 abort 后带 `prompt + 已生成 token + 剩余预算`重放(`verl/workers/rollout/llm_server.py:410-466`);`rebalance_requests` 已串起 clear-sticky → abort → drain-wait → resume 四步(`verl/experimental/fully_async_policy/fully_async_rollouter.py:1211-1272`);
- abort 丢弃门:`partial_rollout` 关闭时 aborted 样本不重试、直接丢弃(`llm_server.py:518-520`)——注意该门约束的是**跨版本**场景的样本容忍,而非迁移本身:同版本副本间的迁移零版本混布,不需要 partial_rollout(见 §4.4 概念澄清;Laminar 的 repack 即同版本主动迁移);
- 策略插件已就位:`DynamicSchedulePolicyBase` + `register_policy`(`verl/experimental/fully_async_policy/dynamic_schedule/base.py:109-174`),输入口 `DynamicScheduleContext`(`base.py:57-106`);
- 权重版本混布已有容忍:`min/max_global_steps` 透传(`llm_server.py:444-470`),trainer 侧按 staleness 处理;
- Mooncake 共享 prefix store 的定位即降低跨副本迁移的 re-prefill 成本(`docs/perf/rollout_kv_offload.md`);
- `StorageCheckpointEngine`(multi-task-infer)实现 verl `CheckpointEngineWithCache` 兼容接口,README 明确给出 `CheckpointEngineRegistry.register("storage")` 的接入方式;`checkpoint_manager_class` FQN 扩展点(`rollout.py:238`)是现成接入缝;
- multi-task-infer `switch_model` 已实现:worker 侧重建模型 + 存储加载首选 / weight_transfer 备选 / dummy 兜底,KV cache 重分配,耗时 10-60s(对比重建实例 68-495s),前置条件"无 in-flight 请求"由 L2 drain 满足。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│ L3a 容量层(新建):SharedRolloutBroker —— 集群级 Ray 命名 actor     │
│   LeaseManager:租约 acquire/renew/preempt                       │
│   WeightResidencyRegistry:actor-group→(model, version) 驻留登记 │
│   FairScheduler:GPU-time 加权公平/配额;Accounting:计量          │
└───────────────▲─────────────────────────────────┬───────────────┘
                │ 池状态/驻留上报                  │ 池操作(租约驱动 acquire/release/switch)
┌───────────────┴─────────────────────────────────▼───────────────┐
│ L3b 基础设施层(已实现):multi-task-infer ExternalExecutor 插件     │
│   ActorPoolManager:预启动 actor 池(绑 GPU、NCCL 预热)            │
│   ExternalWorkerActor:switch_model 热切换(10-60s)/reset(5-10s) │
│   CacheManagerActor:编译缓存共享(本地→池→编译锁)                 │
│   StorageCheckpointEngine:NFS/Mooncake(RDMA ~9 GB/s,           │
│     兼容 verl CheckpointEngineWithCache 接口)                    │
└───────────────▲─────────────────────────────────┬───────────────┘
                │ actor handles(数据面直连:       │ 容量申报/让渡/续期
                │  executor 构建/switch/权重拉取)  │  (控制面,经 broker)
┌───────────────┴─────────────────────────────────▼───────────────┐
│ L2 编排层(新增,每 job 一套)                                       │
│  OrchestrationLoop(事件驱动 tick,秒级)                          │
│   输入:LB 快照(residual 记账)+ EngineStats(实测)+ 训练侧需求    │
│   决策:波尾收缩 / 让渡 / rebalance(复用 net-benefit 骨架)        │
│   执行:begin_drain → 选择性 abort → wait → release(lease)        │
│  DynamicResourceController(步驱动,已有,保留)                    │
└───────────────▲─────────────────────────────────┬───────────────┘
                │ 上行:replica 负载快照            │ 下行:成员/drain/bias/sticky-clear
┌───────────────┴─────────────────────────────────▼───────────────┐
│ L1 路由层(升级,每 job 一个 LB actor;域=(job, version|adapter))    │
│  GlobalRequestLoadBalancer:token 加权评分 + 粘性 + drain 集       │
│  输入:acquire(kv_hint) / release(actual_kv)                     │
└───────────────▲─────────────────────────────────┬───────────────┘
                │ TokenOutput(含 num_preempted)   │ sleep/wake/abort/profile
┌───────────────┴─────────────────────────────────▼───────────────┐
│ L0 引擎层(不动):continuous batching / preemption / prefix cache │
│   引擎实例运行在 L3b 租出的 actor group 上                         │
└──────────────────────────────────────────────────────────────────┘
```

| 层 | 决策对象 | 时钟 | 信号 | 机制 | verl 落点 |
|---|---|---|---|---|---|
| L0 | 副本内请求调度 | iteration | token 预算/KV 水位 | 引擎自带 | vLLM/SGLang |
| L1 | 单请求去向 | 每请求(毫秒) | 本地 token 记账 + sticky | 评分选副本 | `GlobalRequestLoadBalancer` |
| L2 | 副本池成员/状态 | 事件+步(秒~分) | 残余工作量+引擎实测 | drain/收缩/让渡 | Rollouter + Policy |
| L3a | 池跨 job 容量分配 | 秒~作业级 | 计量/驻留/公平 | Lease + 驻留感知放置 | SharedRolloutBroker(§10.3) |
| L3b | actor 池/热切换/缓存/存储 | 事件(切换 10-60s) | 池状态/驻留 | acquire/release/switch_model | multi-task-infer(§10.1) |

---

## 3. L1 路由层设计

### 3.1 状态扩展

```python
class GlobalRequestLoadBalancer:
    def __init__(self, servers, max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
                 full_determinism=False, strategy="token"):
        self._servers: dict[str, ActorHandle]            # 不变
        self._inflight_requests: dict[str, int]          # 不变(计数,兼容指标)
        self._inflight_tokens: dict[str, int]            # 新增:token 加权负载
        self._request_book: dict[str, tuple[str, int]]   # 新增:request_id → (sid, hint)
        self._calibration: dict[str, list[int, int]]     # 新增:Σhint, Σactual(EMA 用)
        self._draining: set[str] = set()                 # 新增:L2 拥有的 drain 集
        self._bias: dict[str, float] = {}                # 新增:L2 可选评分偏置
        self._request_id_to_server: LRUCache             # 不变
        self._full_determinism: bool                     # 不变
        self._strategy: str = strategy                   # "token" | "count"
```

### 3.2 API 变更(全部向后兼容)

```python
def acquire_server(self, request_id: str,
                   kv_hint: int | None = None) -> tuple[str, ActorHandle]:
    # 粘性路径:命中且目标未 drain → 照旧(计数与 token 各加 hint);
    #          目标已 drain → 删除粘性条目,落入重选(复用 remove 的惰性失效模式)
    # 重选路径:候选 = 非 drain 副本;
    #          strategy=="token" → score = inflight_tokens × (1 + bias)
    #          strategy=="count" → score = inflight_requests(现状)
    #          平局先比计数再比插入序;注册 _request_book[request_id] = (sid, hint)

def release_server(self, server_id: str, request_id: str | None = None,
                   actual_kv: int | None = None) -> None:
    # hint = _request_book.pop(request_id)[1] if request_id else 1
    # 计数与 token 各减 hint(精确,无漂移);
    # actual_kv 提供时累积进 _calibration(仅供 L2 校准和指标,不进记账)

def rebase_loads(self, loads: dict[str, int]) -> None:
    # L2 周期调用:用引擎实测的每副本真实占用量覆盖 _inflight_tokens
    # (快速路由用近似,慢速回路用真值;彻底消除 hint 泄漏/漂移)

def begin_drain(self, server_ids: list[str]) -> None: ...
def end_drain(self, server_ids: list[str]) -> None: ...
def set_bias(self, server_id: str, bias: float) -> None: ...
```

`get_status()` 扩展返回每副本 `{count, tokens, draining}`,供 L2 快照。

### 3.3 客户端改动(`LLMServerClient` / `FullyAsyncLLMServerClient`)

```python
# generate() 内,acquire 前:
kv_hint = len(prompt_ids) + self._effective_max_tokens(sampling_params)
# _effective_max_tokens: sampling_params 的 max_tokens/max_new_tokens,
# 缺省回退 rollout.response_length(FullyAsync 重试循环里预算已被改写为
# "剩余量"、prompt 已带上已生成 token,hint 自然收敛到真实终长)

# finally 中:
self._release_server(server_id, request_id=request_id,
                     actual_kv=len(prompt_ids) + len(output.token_ids))
```

改动面极小:`acquire_server` 全仓库只有一个调用点(`llm_server.py:221-229`)。

### 3.4 退化与已知近似

- **无 hint 调用方**(第三方子类/旧客户端):hint 缺省 1 → token 模式精确退化为 count 模式;
- **前缀共享高估**:n 条同 prompt 请求共置一个副本时,prefix block 共享,Σhint > 实际 KV。方向保守(提前绕开),由 L2 的 rebase 修正,不需要 L1 解决;
- **full_determinism**:hash 路由路径完全不动(`llm_server.py:103-107`),且 L2 整层被 gate(见 §4.6)。

---

## 4. L2 编排层设计

### 4.1 组件与部署位置

新增 `OrchestrationLoop`:一个 asyncio 后台任务,随 `FullyAsyncRollouter.create()` 启动(它持有 `llm_server_manager`,是唯一天然能同时触达 LB 与 replica 的位置):

```
tick(每 poll_interval_s):
  1. 快照:lb.get_status() + EngineStatsProvider.get_stats()(并行)
  2. 决策:policy.on_orchestration_tick(ctx) → Noop | Consolidate | Cede
  3. 执行:持 membership_lock(与 add/remove/activate/deactivate 互斥)
```

与现有步驱动钩子的关系:**两个时钟,两个 Context**。`DynamicScheduleContext`(步驱动,训练供需)保持不变;新增 `OrchestrationContext`(事件驱动,池内状态)。两者都喂给同一个 policy 实例。

### 4.2 引擎统计适配器

```python
@dataclass
class EngineStats:
    kv_util: float | None          # 0..1
    kv_capacity_tokens: int | None
    running: int | None
    waiting: int | None
    decode_tps: float | None
    preempted_recent: int | None

class EngineStatsProvider(Protocol):
    async def get_stats(self) -> EngineStats: ...
```

- vLLM 适配:经 `server_handle.collective_rpc` / 引擎 metrics;SGLang 适配:`get_server_info` 路径;
- **所有字段可缺省**:取不到的字段为 None,决策降级(见 §4.5 容量检查);API 跨版本漂移只损失精度,不损失正确性;
- 顺带产出:`preempted_recent` 持续高的副本 → `set_bias(sid, +0.3)` 临时降权(L1 立即少路由过去),比等 L2 收缩快一个数量级。

### 4.3 波尾收缩算法

```
consolidate(min_keep):
  前置门(全过才继续):
    G1 full_determinism == False
    G2 active 副本数 > min_keep
    G3 波尾条件:∀ active sid: count(sid) ≤ K 且 kv_util(sid) ≤ U
       (即"每个副本都只剩少量长尾、整机利用率低")
  0. 版本分组:active 副本按 global_steps 分组(version cohort);
     收缩只在组内规划——victim 与 keeper 必须同版本(Laminar repack
     step ① 同款:先按权重版本分组,再在组内迁移)
  1. 快照 residual:组内 active 副本按 inflight_tokens 降序
  2. keeper 选择(贪心,组内):取前 k 个,使得
       Σ_{keepers} headroom(s) ≥ capacity_safety × Σ_{victims} residual(s)
     其中 headroom(s) = kv_capacity_tokens(s) × (1 - kv_util(s));
     kv_capacity 未知 → 直接放弃收缩(降级为仅观测)
     k == n(选不出 victim)→ Noop
  3. 执行(仅对 victims):
     a. lb.begin_drain(victims)                  # 切断新路由(含粘性)
     b. await gather(replica.abort_all_requests())  # 现有原语,粒度恰好是整副本
        # 客户端捕获 abort → sleep 1s → 重试 → acquire 落在 keeper
        # (victims 已 drain,重试不会回流;keeper 与 victim 同版本
        #  → 续生成零版本混布,与 partial_rollout 无关,见 §4.4)
     c. 轮询 lb.get_status() 直到 victims 的 count == 0(超时 drain_timeout_s;
        复用 rebalance_requests 第 3 步的 drain-wait 模式)
     d. 成功:lb.remove_servers(victims) → cede 回调,三个去向按配置:
        i.  本 job 保留(收缩省出的容量,仅降低成本);
        ii. 交还 trainer 的 deactivate 流程(dynamic_schedule 既有);
        iii.释放 L3 租约(broker 再分配;副本驻留权重保留至被驱逐,
            让渡成本从"sleep+offload"降为"drain+lease release",§10.2)
     e. 失败/超时回滚:replica.resume_generation() + lb.end_drain(victims)
        (请求已随 retry 落在 keeper,回滚只恢复路由资格)
```

触发时机两源(Laminar §5 同款):周期 tick(§4.1)+ **权重更新完成后立即**——被释放的 victim 尽快拉取最新版本,提升 on-policy 度。组内无同版本 keeper 可选 → 该组退化为 drain-wait(只 begin_drain 自然排空,不 abort)。

### 4.4 迁移安全性门(关键正确性约束)

**概念澄清**:迁移与 partial_rollout **正交**。partial_rollout 的语义是"通过 abort+重计算解决**跨权重版本**样本"(trainer 侧的版本容忍机制);而**同版本副本间的迁移天然零版本混布**——续生成用的还是同一份权重,只是换了引擎实例,重付的只有 re-prefill 成本。Laminar(EuroSys'26,arXiv:2510.12633)的 dynamic repack 即此模式:rollout manager 先按 model weight version 分组(§5 步骤①),再在**组内**主动迁移在途轨迹;其 staleness 是轨迹固有属性(生成耗时 vs trainer 推进速度),与迁移动作无关。

| 门 | 规则 | 依据 |
|---|---|---|
| 版本一致性(迁移的真正前置) | keeper 与 victim 必须同版本组(§4.3 步骤 0)。同版本续生成零混布,**无需 partial_rollout 背书**;组内选不出同版本 keeper → 该组退化为 drain-wait(自然排空,不 abort) | Laminar §5 步骤①;verl 版本号 = server 侧 `global_steps` |
| 重试许可解耦(**已实现**) | abort 重试许可由 LB 的 drain 状态背书,不再与 partial_rollout 硬耦合:客户端在 abort 路径查询 LB 插座 `is_request_migrating(request_id)`(sticky 指向的 server 在 drain 集或已移除 → 迁移性 abort → 允许重试;否则按 partial_rollout 门)。stock LB 恒 False、自定义 LB 无此方法(AttributeError)保守回退 → **默认行为与现状完全一致**;查询仅在 abort 路径发生(正常完成零开销)。安全性前提:L2 membership lock 保证 drain 与权重更新互斥,故"sticky 在 drain"不可能掩盖并发版本变更 | `llm_server.py`(基类 LB 插座 + `FullyAsyncLLMServerClient._abort_was_migration`);`load_balancer.py`(`TokenWeightedLoadBalancer.is_request_migrating` 覆写) |
| 跨版本样本容忍(既有问题,与迁移无关) | 权重更新打断在途(update_weights 先 abort_replicas 全副本)产生的 mixed 版本样本,才归 partial_rollout/重计算与 staleness 容忍管;`min/max_global_steps` 标注机制既有 | `llm_server.py:496-503` |
| 预算不变式 | 迁移后样本 token 总长 ≤ 原 max_tokens(与版本无关,迁移与重试共用) | FullyAsync 重试携带剩余预算(`llm_server.py:451-452`) |
| 迁移成本 | re-prefill token 量 × (1 − mooncake/前缀缓存命中率);Laminar 用 partial response pool 在目标副本重生成,同款成本结构 | `docs/perf/rollout_kv_offload.md` |

### 4.5 决策模型

复用 `default_policy.py:185-189` 的 net-benefit 骨架,替换输入信号:

```
benefit  = E[自然排空时间] − E[迁移后排空时间]        # 来自 residual / decode_tps
cost     = Σ re-prefill tokens / prefill_tps(keepers) # Mooncake 折价
           + last_deactivate_duration_s(已有字段)     # 让渡切换本身的代价
决策     = benefit − cost > margin 才执行
```

### 4.6 策略接口扩展

```python
class DynamicSchedulePolicyBase:
    # 现有五个钩子不动(步驱动)
    def on_orchestration_tick(self, ctx: OrchestrationContext)
        -> OrchestrationDecision | None: ...   # 新增,默认 Noop

@dataclass
class OrchestrationContext:
    lb_status: dict[str, ReplicaLoad]    # {count, tokens, draining}
    engine: dict[str, EngineStats]
    versions: dict[str, int | None]      # sid → global_steps(版本分组输入,§4.3 步骤 0)
    min_keep: int
    partial_rollout: bool
    full_determinism: bool
```

默认实现 `@register_policy("tail_consolidation")` 仅用上述阈值;自定义策略可完全接管。

---

## 5. 层间契约汇总

| 方向 | 接口 | 频率 | 语义 |
|---|---|---|---|
| Client → L1 | `acquire(kv_hint)` / `release(actual_kv)` | 每请求 | 近似成本申报 |
| L1 → L2 | `get_status()` | 每 tick | 记账快照 |
| L2 → L1 | `add/remove_servers`、`begin/end_drain`、`set_bias`、`clear_sticky_cache`、`rebase_loads` | 事件 | 只改路由视图 |
| L0 → L2 | `EngineStatsProvider.get_stats()` | 每 tick(1-5s) | 实测真值 |
| L2 → L0 | `sleep/wake_up/abort_all_requests/resume_generation` | 事件 | 生命周期 |
| Trainer → L2 | `DynamicScheduleContext`(步) | 每步 | 供需信号 |
| L2 → Trainer | deactivate/activate 既有流程 | 事件/步 | 资源交还 |
| L2 → L3a | `request_capacity(spec)` / `release(lease)`(附 net-benefit 证据与期望时长) | 事件 | 容量请求/让渡(控制面) |
| L3a → L2 | Lease 授予/续期/抢占(grace)(附 actor handles 与驻留提示) | 秒~分钟 | 池划分与抢占 |
| L2 → L3b | 数据面直连:executor 构建 / `switch_model` / `StorageCheckpointEngine` 拉权重 | 事件 | 副本物理生命周期 |
| L3b → L3a | 池状态/权重驻留上报 | 秒 | 驻留感知放置的输入 |

**契约红线**:L2 永不调用 L1 的 `acquire_server`;L1 永不改变成员集;L0 的指标只进 L2,不进请求关键路径;L3a(broker)永不进入请求路径——授予后数据面直连 L3b。

---

## 6. 配置面

沿用现有风格(`rollout.py:79` 的 FQN 插件先例):

```yaml
actor_rollout_ref:
  rollout:
    load_balancer:
      strategy: token          # count | token;缺省 hint 时 token≡count
      # class: "mypkg.MyLB"    # 全量自定义(走 load_balancer_cls 既有参数)

async_training:
  dynamic_schedule_policy: tail_consolidation   # 已有键(base.py:30)
  orchestration:
    enable: true                 # 总开关;false = 整层关闭,精确现状
    poll_interval_s: 2.0
    consolidation:
      min_keep_replicas: 1
      tail_max_inflight: 2       # K:波尾判定,每副本在途上限
      tail_kv_util: 0.25         # U:波尾判定,利用率上限
      mode: auto                 # auto | abort_migrate | drain_wait
                                 # auto 门控:full_determinism 拒绝 + 版本组可用性
                                 # (同版本迁移无需 partial_rollout,见 §4.4)
      drain_timeout_s: 60
      capacity_safety: 0.7

    shared_pool:                       # L3b;不配置 = 现状(自建 standalone 副本)
      enable: false                    # 开启前 fail-fast 检测 vLLM external_actors 支持
      # broker_name: shared_rollout_broker   # L3a Ray 命名 actor
      # model_family: qwen3-8b         # 驻留放置提示(同基座优先 switch 而非跨架构)
      weight_source: auto              # trainer_push(最fresh)| storage_pull | auto(按 staleness 决定)
      allow_dummy_fallback: false      # RL 红线:dummy 仅用于初始化建结构,权重填充
                                       # (storage_pull / weight_transfer 重排)失败必须抛错(§10.5)
```

---

## 7. 可观测性

- **L1**:`rollout_lb/inflight_tokens`(per replica)、hint 校准比(Σactual/Σhint)、粘性命中率、drain 副本数;
- **L2**:收缩事件(victims/迁移请求数/估计 re-prefill 量)、让渡耗时、回滚次数、tick 决策分布;
- **L3**:lease 利用率(授予/持有/排队)、驻留命中率(授予时权重已就位的比例,直接反映放置质量)、切换耗时分布(storage_pull vs trainer_push 分开统计)、编译缓存命中率、抢占次数与 grace 超时次数;
- **复用既有**:`agent_loop/num_preempted`(预期显著下降)、`agent_loop/slowest/*`、`timing_s/gen`、`timing_s/switch_to_rollout`、`DynamicScheduleContext.last_deactivate_duration_s`。

验收口径:重尾 prompt 合成负载下,`num_preempted/max` 下降、波尾时刻到"副本可让渡"的延迟从 O(最长尾)降到 O(迁移时间)。

---

## 8. 故障模式与降级

| 故障 | 影响 | 处置 |
|---|---|---|
| hint 泄漏(client 崩在 acquire/release 之间) | token 记账虚高 | L2 周期 `rebase_loads` 用引擎真值覆盖;现状的计数泄漏同样受益 |
| LB actor 重启 | 路由状态丢失 | Ray actor 重建后由 L2 立即 rebase + 重建 drain 集(成员集由 manager 持有,可重放) |
| drain 超时(重试未如期落地) | victim 挂起 | 回滚:end_drain + resume_generation;计数已含超时统计 |
| 引擎 stats 取不到 | 容量检查失效 | 降级为仅观测(不做收缩);bias/rebase 停用,记数记账仍有效 |
| keeper 被 retry 涌入打满 | KV 压力转移 | G3 波尾门 + capacity_safety 预留;`preempted_recent` 触发 bias 降权;abort 本身受引擎 ack 背压 |
| 确定性冲突 | — | G1 硬门:full_determinism 下 L2 整层禁用 |
| L3b 权重填充失败(storage pull 与 weight_transfer 重排均失败) | — | **RL 红线:必须抛错,不得以 dummy 权重进入服务**;L2 回滚让渡(§10.5) |
| broker(L3a)失效 | 租约仲裁停摆 | Ray 命名 actor 自动重启;租约表自 L3b 池状态重建;job 侧租约 TTL 过期自行 drain 退出 |
| actor 死亡(L3b) | 该副本/租约失效 | 池重建 actor(预热件);受影响 lease 作废,broker 重分配 |
| vLLM 缺 `external_actors` 参数 | shared_pool 不可用 | 启动 fail-fast(捕获 TypeError 并指引 fork 分支),不进入半初始化状态 |

---

## 9. 分阶段落地与测试

| 阶段 | 内容 | 独立价值 | 测试 |
|---|---|---|---|
| P1 | L1 token 评分 + hint/release + 校准统计 | 直接缓解 P1 | 单元:纯类可测(`GlobalRequestLoadBalancer` 是 plain class)——无 hint 退化等价 count、粘性+drain 惰性失效、记账精确性、rebase;e2e:复用 `tests/experimental/agent_loop/test_basic_agent_loop.py` / `test_standalone_rollout.py` 挂重尾长度 fixture |
| P2 | EngineStatsProvider 两适配器 + tick 回路(**仅观测模式**,只打日志不动作) | 波尾可见性(P2 的"看不出来"先变成"看得见") | 适配器 mock;stats 缺省降级路径 |
| P3 | OrchestrationLoop + 波尾收缩策略 + 回滚(**客户端 abort-重试许可解耦已提前实现**:LB `is_request_migrating` 插座 + drain 感知覆写,§4.4/§9.1 插座 7;剩事件驱动回路与版本组收缩算法本体) | 让渡能力(P2/P3) | 单元(已实现):解耦门四分支(迁移重试/权重更新丢弃/partial_rollout 无关性/v1 恒重试)、drain-移除窗口、预算不变式、热路径零查询;集成(待 P3):同版本迁移样本完整性(TokenOutput 长度 ≤ response_length、global_steps 单调)、版本组 keeper 选择(混布版本下只组内迁移)、partial_rollout 开启下另测跨版本重试路径;故障注入:drain 超时、stats 缺失、membership 并发 |
| P4a | `StorageCheckpointEngine` 注册进 `CheckpointEngineRegistry`(`checkpoint_manager_class` FQN,`checkpoint_engine/base.py:49`) | 单 job 即有价值:副本冷启动权重走存储,trainer 不必在线推送 | 接口兼容 round-trip(send→receive→get_weights);复用现有 checkpoint engine 测试模式 |
| P4b | `PooledRolloutReplica`(`init_pooled` 第四条 init 路径)+ LLMServerManager shared_pool 模式 | 单 job 消费共享池:副本构建 68-495s → 10-60s。**骨架已实现(§9.2)**:握手/租约生命周期/红线全通;每步权重同步留 P4b-next | 单元:fail-fast 矩阵 + init_pooled 时序 + 填充失败断言(已实现);e2e:池上副本跑通 rollout;`switch_model` 热切换断言(P4b-next) |
| P4c | SharedRolloutBroker(lease/驻留/公平/计量)+ 抢占协议 | 跨 job 共享:波尾让渡的容量即时复用 | 多 job 集成:让渡→再分配→热切换全链路;grace 抢占;租约 TTL 失联;驻留放置命中率 |

依赖关系:P1 独立;P2→P3;P4a 独立(可与 P1 并行);P4b 骨架(冷启动+租约生命周期)不依赖 P3——粗粒度整副本释放无在途问题;P4b-next(每步热切换)依赖 P3(drain 原语)+ P4a;P4c 依赖 P4b。L3b(multi-task-infer)已实现,不在本仓库阶段内。

每阶段均为独立 PR 量级;P1 不依赖 P2/P3,可先行合入。

### 9.1 P1/P4a 落地(插件化,已完成实现)

实现遵循"插件优先"原则:全部新行为放在插件包内,verl 主干只加了三个默认关闭的插座。

**插件包:`verl/experimental/rollout_scheduling/`**

| 文件 | 内容 | 对应设计 |
|---|---|---|
| `load_balancer.py` | `TokenWeightedLoadBalancer`:`_inflight_tokens`/`_request_book`/`_draining`/`_bias`/校准累加器;`acquire_server(request_id, kv_hint)`、`release_server(server_id, request_id, actual_kv)`、`begin_drain`/`end_drain`/`set_bias`/`rebase_loads`、扩展 `get_status` | §3 全部 |
| `client.py` | `_HintedAcquireMixin`(`LLMServerClient` 子类,mixin)+ `HintedLLMServerClient` + `HintedFullyAsyncLLMServerClient` + `upgrade_client_class`;ContextVar 传递 hint,per-attempt 估算 | §3.3 |
| `storage_checkpoint.py` | `register_storage_checkpoint_engine()`:导入 multi-task-infer 的 `StorageCheckpointEngine` 并注册进 `CheckpointEngineRegistry`(幂等;模块级自动注册走 verl 现有 `custom_backend_module` 钩子,显式调用缺依赖时 ImportError 带指引) | §10.4 集成点 1(P4a) |
| `tests/experimental/rollout_scheduling/` | LB 单测(token 评分、退化、粘性、drain、记账精确性、rebase、determinism、status 兼容)+ 客户端单测(hint 数学、P+B 不变量、并发隔离、upgrader 映射) | §9 P1 测试项 |

**verl 主干插座(`verl/workers/rollout/llm_server.py`,全部加法、默认关闭)**

1. `GlobalRequestLoadBalancer.acquire_server/release_server` 接受(并忽略)可选 `kv_hint`/`request_id`/`actual_kv` —— 使任意 LB×客户端组合互操作安全;
2. `LLMServerManager.__init__` 从 `VERL_ROLLOUT_LOAD_BALANCER_CLS`(FQN)解析 LB 类:显式参数 > env > 默认,带 `issubclass` 校验;
3. client-class upgrader 注册表(`register_client_class_upgrader` + `get_client` 内查询);插件包导入时注册(hinted 客户端对 stock LB 无害,因为插座 1 忽略多余 kwarg);
4. `full_determinism` 转发给接受该参数的 LB 子类(`inspect.signature` 探测);
5. `LLMServerManager.create` 内 FQN 分发(`rollout.llm_server_manager_class` 命名子类时透明实例化,§9.2);
6. `RolloutConfig` 加 `llm_server_manager_class` / `shared_pool` 字段(§9.2);
7. `GlobalRequestLoadBalancer.is_request_migrating(request_id)` 插座(默认恒 False)+ `FullyAsyncLLMServerClient` abort 重试许可解耦(§4.4):stock LB 与自定义 LB 均保持现状行为,只有 drain 感知的 LB(P1)返回 True 才启用同版本迁移重试;查询仅在 abort 路径发生。

**启用方式(单一开关)**:

```bash
export VERL_ROLLOUT_LOAD_BALANCER_CLS=verl.experimental.rollout_scheduling.load_balancer.TokenWeightedLoadBalancer
```

时序安全性:所有 trainer 路径(`ray_trainer.py:951→962`、`trainer_base.py:353→375`、`trainer_colocate_async.py:34`、`trainer_separate_async.py:132→182`)都先构造 `LLMServerManager`(此时 FQN 导入插件包 → 注册 upgrader)再调 `get_client`,upgrader 注册必然先于客户端创建。

**关键实现决策**

- **FullyAsync 重试兼容(MRO)**:`HintedFullyAsyncLLMServerClient(FullyAsyncLLMServerClient, _HintedAcquireMixin)` 的 MRO 为 `[HintedFully, FullyAsync, Mixin, LLMServerClient]`——`only_hybrid` 重试包装层位于 hinted acquire **外层**(重试重跑的是带 hint 的 acquire);每次尝试的 `super().generate` 落到 Mixin,用当次 prompt/budget 重新估算。**P+B 不变量**:重试时 prompt 增长量 == budget 缩减量,故每次尝试的 hint 恒等于 `prompt_len + original_budget`,估算精确。已用无依赖脚本验证 MRO 调用链。
- **ContextVar 而非实例属性**:`LLMServerClient.generate` 并发运行于同一 worker 的多个 asyncio task,实例属性会串任务;`contextvars` 在 task 边界天然隔离。
- **upgrader 精确匹配**:只映射 `LLMServerClient`/`FullyAsyncLLMServerClient` 两个精确类,自定义客户端子类原样通过(安全退化为 count 路由)。
- **校准不阻塞主路径**:v1 不做 per-request `actual_kv` 上报(避免每请求一次额外 RPC);漂移由 `rebase_loads` 治愈,`hint_sum/actual_sum` 供 L2 观测。

### 9.2 P4b 落地(共享池副本,已完成实现——首次真实握手)

P4b 把设计 §10.4 的集成点②(`external_actors` 物理连接)落成代码:verl 副本第一次真正建立在 multi-task-infer 预启动的 actor 之上,**租约即副本生命周期**。

**插件包新增**

| 文件 | 内容 |
|---|---|
| `pooled_replica.py` | `PooledvLLMHttpServer`(单 CPU 前端:引擎构造改为 `AsyncLLM(executor_class=ExternalExecutor, external_actors=...)`——fork 的构造器插座,`from_vllm_config` 无此参数;跳过 `monkey_patch_model` 启动 RPC)、`PooledvLLMReplica`(第四条 init 路径 `init_pooled(actors, fill)`:launch → fill → 打标 → 清 KV;`release_to_pool`)、`PooledLLMServerManager`(pre_start → 逐副本 acquire → init_pooled)、`check_pooled_prerequisites`(fail-fast 矩阵) |
| `tests/.../test_pooled_replica.py` | fail-fast 矩阵 / init_pooled 时序 / 填充失败永不打标 / 释放路径 / 拓扑与必填校验 / `create` FQN 分发 / 引擎构造参数 |

**verl 主干新增插座(两处,全部加法、默认关闭)**

5. `LLMServerManager.create` 内 FQN 分发:`rollout.llm_server_manager_class` 命名子类且调用方直接用基类时,透明实例化子类(单点分发,零 trainer 改动;同 `agent_loop_manager_class` 惯例);
6. `RolloutConfig` 加 `llm_server_manager_class` / `shared_pool` 两个 Optional 字段(镜像 `checkpoint_manager_class` 先例:yaml 经 `+` override 设置)。

**数据流**:`pre_start(num_actors=num_replicas*tp*pp, devices_per_node)` → 每副本 `acquire(tp,pp)`(world_size 个单卡 actor)→ `init_pooled`:
① 单前端 actor(无 NodeAffinity/无 per-node 服务器——GPU 归池的 placement group);② 存储填充 `load_model_from_storage`(asyncio.gather,任一失败即抛——脚手架永不进 LB);③ `set_global_steps` 打标;④ `clear_kv_cache`。释放:杀前端 → `actor.reset()` → `pool.release`。

**红线落实(§10.5)**

- `fill` 为**必填结构化参数**(非可选 flag):无 `checkpoint_path`/`global_steps` 在 manager 层即拒绝;填充失败异常直接冒出——`allow_dummy_fallback: false` 由类型与校验双保险;
- 填充成功后 `set_global_steps` + `clear_kv_cache` 顺序固定(先打标后清缓存,失败路径两者都不发生);
- `full_determinism` 在 prerequisites 即拒绝(池成员变动与 hash 路由互斥)。

**P4b 的诚实边界(重要)**

- **每步权重同步未接**:`CheckpointEngineManager.update_weights` 走 trainer-push(依赖 `replica.workers`,池化副本为空)。候选通道:executor 级 `switch_model`(需 fork 侧 EngineCore 透传——当前 fork 未暴露)、worker 级存储拉取(可经 `collective_rpc` 到达任意 worker 方法)、每步 `fill_weights_from_storage` 重放。**v1 池化副本服务填充时的 checkpoint**:适用冻结策略 rollout / 评测 / 温池验证;完整训练循环等通道落地(故本阶段为"骨架":握手 + 生命周期 + 红线全通,权重步进通道留 P4b-next)。
- **verl worker 扩展缺失**:池化 worker 是插件侧裸 `WorkerWrapperBase`,无 `vLLMColocateWorkerExtension`(OOV logits 掩码、MoE loader 补丁、LoRA hijack、FP8 补丁);LoRA 在 prerequisites 直接拒绝,其余启动告警。
- **编译缓存未通池**:池创建 `CacheManagerActor`,但 fork 的 `EngineCore` 构造 executor 不透传 `cache_manager`,各副本本地编译。
- **拓扑**:dp=1/副本(横向扩用多副本)、hybrid worker group 不入池(§10.6)、PD disagg 拒绝;fork 分支检测 fail-fast(`AsyncLLM.__init__` 无 `external_actors` 参数即 TypeError 带分支指引)。

---

## 10. L3 多 RL 任务共享层设计

### 10.1 子层划分与组件映射

L3 拆成两个子层,职责与实现状态不同:

| 子层 | 职责 | 状态 |
|---|---|---|
| **L3a 容量层** | 跨 job 容量仲裁:租约、驻留登记、公平性、计量 | 新建(本设计) |
| **L3b 基础设施层** | 物理副本生命周期:actor 池、热切换、编译缓存、存储权重 | **已实现**(multi-task-infer) |

multi-task-infer 组件与本设计的映射:

| multi-task-infer 组件 | 能力 | 在本设计中的角色 | verl 现状对应物 |
|---|---|---|---|
| `ActorPoolManager` | 预启动 Ray actor 池(绑 GPU、NCCL 预热),`acquire(tp,pp)`/`release` | 共享物理资源的最小单位:**一个 actor group = 一个 rollout 副本的引擎侧** | `RolloutReplica.init_standalone` 自建 `RayWorkerGroup`(`replica.py:189`) |
| `ExternalWorkerActor.switch_model` | 不重启进程换模型(10-60s,对比重建 68-495s),含 KV cache 重分配 | 让渡后副本换租户权重的**主通路** | 无对应(现状靠 sleep/wake + update_weights) |
| `CacheManagerActor` | 编译缓存跨节点共享(本地→池→编译锁三级懒加载) | 切换成本摊薄:同架构不同权重 → 编译缓存必命中 | 无对应 |
| `StorageCheckpointEngine` | NFS/Mooncake(RDMA ~9 GB/s)权重加载,接口兼容 `CheckpointEngineWithCache` | 权重供给的存储通道;经 `checkpoint_manager_class` FQN(`rollout.py:238`)接入 `CheckpointEngineRegistry`(`checkpoint_engine/base.py:49`) | `CheckpointEngineRegistry` 既有抽象("naive" 等后端同机制注册) |

**关键咬合点**:multi-task-infer 的 `switch_model` 前置条件是"无 in-flight 请求"(其 design.md 场景 5.3)——这恰好是 L2 波尾收缩(§4.3)的输出状态。没有 L2,共享池只能等整实例空闲后粗粒度切换;有了 L2,让渡发生在波尾(秒级),切换成本被摊进原本就要闲置的时间窗。**L2 是 L3b 在 RL 场景可用的前提,L3b 是 L2 让渡价值的放大器**——两者互为充分理由。

### 10.2 端到端时序:让渡 → 再分配 → 热切换

```
job A(持有副本 R,基座 M@vA)
  │ ① L2 tick:波尾判定(§4.3 G3)→ victim = R
  │ ② begin_drain(R) → abort+retry 迁往 keeper → drain 完成
  │    (迁移发生在 job A 的**同版本组内**——续生成零版本混布,§4.4;
  │     此时 R 满足 switch_model 的"无 in-flight"前置;
  │     迁移与后续的版本切换在时间上不相交:② 完成后 R 才无在途,
  │     ③ 之后才可能换版本/换租户)
  │ ③ lease.release(R) → L3a(附驻留事实:R 当前载有 M@vA)
  ▼
L3a SharedRolloutBroker
  │ ④ 驻留感知放置:等待队列里谁最该拿到 R?
  │    · job B 同基座 M → switch_model(M@vB)≈10-60s(编译缓存命中)
  │    · job B 跨基座   → switch + 可能编译(30-180s,首次)
  │    · job A 马上回来 → 0s 原样复用(驻留保留)
  │ ⑤ lease.grant(R → job B,附 actor handles)
  ▼
job B(数据面直连 L3b,不经 broker)
  │ ⑥ ExternalExecutor 在 R 的 actor group 上构建 vLLM 实例:
  │    权重 = StorageCheckpointEngine 从 NFS/Mooncake 拉 M@vB(trainer 不必在线)
  │    编译 = CacheManagerActor 命中 → 跳过
  │ ⑦ job B 的 LB add_servers(R) → 流量进入
```

时间预算:② O(长尾迁移,秒级) + ⑤⑥ 10-60s(缓存命中)——对比现状"job B 自建 standalone 副本"的 68-495s。注意 verl 侧还有既有通道可比:hybrid 副本激活 = weight sync(秒级,但要求 trainer 在线且共享 GPU);共享池 = 存储拉取(10-60s,但解耦 trainer、独占 GPU)。

### 10.3 Lease 协议(L3a:SharedRolloutBroker)

**控制面集中、数据面直连**(与"改视图,不替路由"同构):所有容量请求经 broker;授予后 job 直接持有 actor handles 构建 vLLM 实例、挂到自己的 LB,broker 永不进入请求路径。

```python
@dataclass
class LeaseSpec:
    tp: int; pp: int
    model: str; version_hint: int | None   # 驻留放置提示
    duration_hint_s: float                 # 由 §4.5 net-benefit 折算
    priority: int; preemptible: bool

class SharedRolloutBroker:                  # ray.remote(name="shared_rollout_broker")
    def request_capacity(self, job_id, spec: LeaseSpec) -> Lease | Queued
    def release(self, lease_id, residency: ResidencyFact) -> None
    def renew(self, lease_id) -> bool       # TTL 心跳;失联租约过期自动回收
    def preempt(self, lease_id, grace_s) -> PreemptionTicket
```

- **抢占协议**:broker 发 `preempt(grace_s)` → 目标 job 的 L2 将其视为一次外部触发的收缩,走 §4.3 同一流程(begin_drain → abort+retry → release);仅当 grace 超时才允许强制(租约作废,job 侧请求 abort 重试到别处)。**抢占复用收缩算法,不另造一套**;
- **驻留登记**:`actor_group → (model, version, loaded_at)`。放置优先级:同版本原样复用(0s)> 同基座 switch(10-60s)> 跨基座 switch(可能 +编译)。驻留保留有显存代价 → LRU 驱逐(按 loaded_at + 池显存水位),驱逐动作 = `actor.reset()`(5-10s);
- **公平性**:账本记 `Σ(租约时长 × world_size)` per job,加权公平 + 配额上限 + 优先级;
- **计量**:per-job 租约时间、token 量、抢占/被抢占次数 → 进 RLInsightLogger。

### 10.4 verl 侧集成点

| 集成点 | 改动 | 依赖 |
|---|---|---|
| ① 存储引擎注册 `checkpoint_manager_class: "storage"` | 把 `StorageCheckpointEngine` 注册进 `CheckpointEngineRegistry`(插件 README 已验证接口兼容:`prepare/build_topology/init_process_group/finalize/send_weights/receive_weights/get_weights`)。**已实现(P4a,§9.1)** | 无(独立有价值) |
| ② `PooledRolloutReplica` | `RolloutReplica` 第四条 init 路径 `init_pooled(actors)`(现有 hybrid/colocated/standalone 三条之外,`replica.py:131-226`);`LLMServerManager` shared_pool 模式:**副本生命周期 = 租约生命周期**(grant→构建 server actor;release→`actor.reset()`+`pool.release`)。**已实现(P4b 骨架,§9.2):握手/生命周期/红线全通;drain 联动与热切换留后续** | 全量让渡需 P3 drain;粗粒度"整副本释放"不需要 |
| ③ switch_model 联动 | `ExternalExecutor.switch_model` 的"调用方保证无在途"前置条件 ↔ L2 drain 输出(§10.2 步骤 4 的咬合点);另需 fork 侧暴露 executor 句柄的通道(EngineCore 透传,当前 AsyncLLM 路径不可达 executor 对象) | P3 + fork 侧改动 |
| ④ 权重双通道 | trainer-push(现有 bucketed ZMQ / delta flush,最 fresh;池化副本因 worker 扩展缺失暂不可用)与 storage-pull(解耦 trainer,`load_model_from_storage` 已可用);`async_training.staleness_threshold` 决定可接受的 pull 版本下界 | 无 |
| ⑤ 热切换入口 | `CheckpointEngineManager.update_weights` 的 pooled 分支:副本已 drain → `switch_model`(而非 sleep/wake+update);未 drain → 先走 L2 收缩。**P4b-next 的核心** | P3 + ③ |

### 10.5 RL 特有的正确性红线

普通多任务推理与 RL 共享池的关键差异——**推理服务错权重是质量事故,RL 错权重是训练污染**:

1. **dummy 仅作初始化脚手架,不得成为服务状态**:multi-task-infer 的加载链是 dummy 初始化(建模型结构)→ 权重填充(storage pull 首选 / weight_transfer 参数重排备选)。RL 集成保留这条链,但**填充(storage pull 与重排均)失败必须抛错**(`allow_dummy_fallback: false`,§6)——假权重生成的样本会带错误 `global_steps` 混进 replay buffer 污染训练。不变式:**进入 LB 的副本,其权重必来自某次成功填充,且已 `set_global_steps` 打标**;由 L2 回滚让渡、broker 重新放置;
2. **版本标记**:storage-pull 的副本必须以 checkpoint 路径中的 `global_steps` 打标(`set_global_steps`,`vllm_async_server.py:890` 现有机制),不得沿用 actor 上残留的旧版本号;`switch_model` 后 KV cache 由插件重分配,verl 侧仍须补 `clear_kv_cache`——旧权重的 KV 不可复用;
3. **full_determinism 互斥**:hash 路由要求稳定成员集,与租约动态进出矛盾 → 共享池模式整体 gate(与 §4.3 G1 同源)。

### 10.6 部署约束

- vLLM 须用 fork 的 `feature/external-executor` 分支(`external_actors` 参数未上游,PyPI 包会 TypeError)→ shared_pool 启动时 fail-fast 检测并给出指引(插件 README 已有此警告);
- multi-task-infer 为 vLLM 专属 → 共享池首期限定 `rollout.name == "vllm"`;
- **hybrid 副本首期不进池(边界,非永久禁令)**:hybrid 与训练共享 GPU,成员进出必须与训练步的 deactivate/activate 对齐,协调面比 standalone 复杂,首期不做。**放开路径已具备,不必新造机制**:借出前用 L2 收缩原语 drain(在途请求迁往本 job 的 keeper 或池内副本)→ 训练忙时借出、activate 前归还——本质是 dynamic_schedule "训练空闲期激活 hybrid 参与 rollout" 的镜像(训练忙时 hybrid 的推理面借给池,纯增益)。约束:借用窗口上界 = 训练步节奏,归还超时由 broker 抢占协议兜底;权重同步仍走 trainer-push(hybrid 与 trainer 同 GPU,不必绕存储);
- `CacheManagerActor` 随池创建,集群级单例;编译缓存 hash 含(模型架构, TP/PP/DP, cudagraph 配置)——**不含权重值**,这是"同基座多 RL job 必命中"的根据。

### 10.7 LoRA 细粒度共享(远期方向,仅记录)

多租户同基座时,vLLM multi-LoRA 可把"分时共享"升级为"并发共享":base 权重驻留一份,adapters per job,零切换成本。verl 已有 `lora_as_adapter` 雏形但 `max_loras: 1`(`vllm_async_server.py:405-413`)。**前提是算法侧配合**:RL 每步全权重更新,LoRA 化需要专门的 RL 算法。定位为算法+系统联合的远期方向,不阻塞本设计。

---

## 11. 贡献合规(按 AGENTS.md)

- 提 PR 前执行查重:`gh issue view` / `gh pr list --search`,关键词:`load balanc* rollout`、`KV-aware routing`、`tail consolidation`、`dynamic resource scheduling`;
- 本设计为 substantive work,满足"非 busywork"要求;须有人类提交者端到理解并运行测试;PR 描述须包含查重结论、测试命令与结果、AI 辅助声明;
- 实现应置于 `verl/experimental/` 起步(与 `dynamic_schedule`、`fully_async_policy` 同级),稳定后再考虑提升默认值;
- P4(共享池)依赖 vLLM fork 的 `feature/external-executor` 分支(`external_actors` 未上游)——该分支的上游化进度是 P4b/P4c 的外部风险,P4a 不受影响;
- 涉及 multi-task-infer 侧的改动(如 RL 红线要求的 dummy 兜底开关)在插件仓库内进行,两仓 PR 各自独立可测。
