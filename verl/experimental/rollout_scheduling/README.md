# rollout_scheduling — verl 分层调度插件

实现 `rollout_layered_scheduling_design.md`(仓库根目录)的 L1 路由层升级、P4a 存储引擎注册与 P4b 共享池副本。全部为插件:verl 主干只增加了默认关闭的插座(见下),不设置环境变量/配置时行为与主线完全一致。

## 组件

| 组件 | 作用 | 设计章节 |
|---|---|---|
| `TokenWeightedLoadBalancer` | 按**在途 token 估计值**(而非请求条数)选副本;带 sticky session、drain 集合、偏置、rebase、校准遥测、`is_request_migrating` drain 状态查询(迁移重试解耦) | §3/§4.4 |
| `HintedLLMServerClient` / `HintedFullyAsyncLLMServerClient` | acquire 时传 `kv_hint`(prompt + response budget),release 时传 `request_id` 精确销账 | §3.3 |
| `register_storage_checkpoint_engine()` | 把 multi-task-infer 的 `StorageCheckpointEngine` 注册进 verl 的 `CheckpointEngineRegistry` | §10.4(P4a) |
| `PooledLLMServerManager` / `PooledvLLMReplica` / `PooledvLLMHttpServer` | 从 multi-task-infer 池租 actor 构建 rollout 副本:verl ↔ 池的首次真实握手(租约=副本生命周期) | §10.4(P4b,集成点②) |

## 启用 L1(单一开关)

```bash
export VERL_ROLLOUT_LOAD_BALANCER_CLS=\
verl.experimental.rollout_scheduling.load_balancer.TokenWeightedLoadBalancer
```

该 env 由 `LLMServerManager.__init__` 解析(显式 `load_balancer_cls` 参数优先):导入插件包 → 注册 client upgrader → 所有 `get_client()` 返回 hinted 客户端。两条路都开启后:

- 带 hint 的请求按 token 权重路由;
- 客户端不传 hint(或使用未升级的自定义客户端)时,退化为与现状**完全等价**的 count 路由(hint=1);
- `full_determinism` 语义不变(hash 路由;drain/bias 由 L2 侧保证不在该模式下使用)。

## 启用 P4a(存储权重引擎)

**方式一:纯 config**(走 verl 现有 `custom_backend_module` 钩子,推荐):

```yaml
actor_rollout_ref:
  rollout:
    checkpoint_engine:
      custom_backend_module: verl.experimental.rollout_scheduling.storage_checkpoint
      backend: storage
```

`CheckpointEngineManager` 在实例化 backend 前导入该模块 → 触发模块级注册。

**方式二:训练脚本显式调用**:

```python
from verl.experimental.rollout_scheduling import register_storage_checkpoint_engine

register_storage_checkpoint_engine()
```

两种方式都幂等。显式调用在缺 `vllm_external_executor` 依赖时抛 `ImportError`(带安装指引)——**fail-fast,不做静默降级**(设计 §10.5);import 路径只告警,后续 `backend: storage` 查找自然报错。

## 启用 P4b(共享池副本)

前提:multi-task-infer 插件(`vllm_external_executor`)可导入,**且 vLLM 为 fork 分支 `feature/external-executor`**(其 `AsyncLLM.__init__` 才有 `external_actors` 参数;PyPI 版会 TypeError——启动即 fail-fast,报错带分支指引)。

```bash
# 训练命令追加 override(`+` 新增键,与 checkpoint_manager_class 同惯例)
+actor_rollout_ref.rollout.llm_server_manager_class=\
verl.experimental.rollout_scheduling.pooled_replica.PooledLLMServerManager \
'+actor_rollout_ref.rollout.shared_pool={num_replicas:1,devices:[0,1],fill:{checkpoint_path:model/step_100,global_steps:100,storage_backend:nfs,storage_config:{base_path:/shared/ckpt}}}'
```

或在实验 yaml 中:

```yaml
actor_rollout_ref:
  rollout:
    llm_server_manager_class: verl.experimental.rollout_scheduling.pooled_replica.PooledLLMServerManager
    shared_pool:
      num_replicas: 1            # 必填:副本数 = 池租约数
      devices: [0, 1]            # 可选:默认 range(num_replicas*tp*pp)
      warmup_distributed: true   # pre_start 时 NCCL 预热
      shared_cache_dir: null     # 编译缓存 NFS 目录(见下方已知边界)
      fill:                      # 必填(红线 §10.5:填充分配,失败即抛)
        checkpoint_path: model/step_100
        global_steps: 100
        storage_backend: nfs
        storage_config: {base_path: /shared/ckpt}
```

配置经 `LLMServerManager.create` 的 FQN 插座分发到子类(单点,零 trainer 改动;同 `agent_loop_manager_class` 惯例)。数据流:`ActorPoolManager.pre_start(num_actors=num_replicas*tp*pp)` → 每副本 `acquire(tp,pp)` → `PooledvLLMReplica.init_pooled(actors, fill)`:

1. 起**单个** CPU 前端 `PooledvLLMHttpServer`(无 NodeAffinity,GPU 归池的 placement group);
2. 引擎构造走 `AsyncLLM(executor_class=ExternalExecutor, external_actors=...)`(fork 的构造器插座;`from_vllm_config` 不带该参数);
3. 存储填充:每个 actor `load_model_from_storage`(任一失败→抛错,脚手架**永不进 LB**)→ `set_global_steps` 打标 → `clear_kv_cache`;
4. 释放:`release_replica/release_all` → 杀前端 → `actor.reset()` → `pool.release`(租约=副本生命周期)。

### P4b 已知边界(诚实清单)

- **每步权重同步未接**:`CheckpointEngineManager.update_weights` 依赖 `replica.workers` 的 trainer-push,池化副本 `workers=[]`。后续三选一:executor 级 `switch_model`(需 fork 侧 EngineCore 透传)、worker 级 `load_model_via_weight_transfer`(可经 `collective_rpc` 到达)、每步存储拉取。**当前池化副本服务的是填充的那个 checkpoint**——冻结策略 rollout / 评测 / 温池验证场景;完整 RL 训练循环等上述通道落地。
- **verl worker 扩展缺失**:池化 worker 是插件的裸 `WorkerWrapperBase`,无 `vLLMColocateWorkerExtension` → OOV logits 掩码(视觉占位 token)、MoE weight-loader 补丁、LoRA hijack、在线 FP8 补丁均不可用(启动时会告警;LoRA 直接 fail-fast 拒绝)。
- **编译缓存未通池**:池创建 `CacheManagerActor`,但 fork 的 `EngineCore` 构造 executor 时不透传 `cache_manager` → 各副本本地编译。
- **拓扑限制**:每副本 dp=1(用多副本横向扩)、hybrid worker group 不入池(§10.6 边界)、PD disagg / full_determinism 拒绝。池本身按多任务推理当前 API 为单设备列表语义。

## 测试

```bash
pytest tests/experimental/rollout_scheduling/ -v
```

纯逻辑测试:LB 是 plain class(fake actor handle 即可),客户端用 fake LB/server 句柄,均不需要 Ray 集群。覆盖:

- 无 hint 退化等价 count;token 评分改变路由;tie-break 确定性;
- sticky session(含 drain 惰性失效);drain 全排除时报错文案与基类一致(客户端匹配依赖);
- release 带 `request_id` 精确销账;legacy release 退化(−1,由 rebase 治愈);
- FullyAsync abort-retry 下 **P+B 不变量**(每次尝试 hint 恒定);
- ContextVar 并发任务隔离;upgrader 精确匹配;插件包注册副作用;
- 迁移重试解耦(`test_retry_decoupling.py`):LB drain 查询语义(draining/移除/active/无 sticky/end_drain 回滚/重试后落 keeper)、客户端四分支(迁移重试、权重更新丢弃、partial_rollout 无关、v1 恒重试)、自定义 LB 保守回退、预算不变式、热路径零查询;
- P4b:prerequisites fail-fast 矩阵 / init_pooled 时序(launch→fill→打标→清 KV)/ 填充失败永不打标 / 释放路径 / 拓扑与必填校验 / `create` FQN 分发 / 引擎构造参数与 stock-vLLM fail-fast。

## 迁移重试解耦(§4.4,已实现)

**概念**:迁移与 `partial_rollout` 正交。partial_rollout 解决**跨权重版本**样本(abort+重计算,trainer 侧);同版本副本间的迁移(Laminar dynamic repack 同款)零版本混布,重付的只有 re-prefill。原实现把"abort 后是否重试"与 partial_rollout 硬耦合,导致非 partial_rollout 模式下迁移样本被丢——已解耦:

- **主干插座 7**(`llm_server.py`,默认行为与现状完全一致):基类 LB 新增 `is_request_migrating(request_id)`(恒 False);`FullyAsyncLLMServerClient` 的 abort 重试门改为——partial_rollout=False 时**仅当** LB 证实该请求的 sticky server 在 drain 集或已移除(迁移性 abort)才重试,否则维持丢弃语义(权重更新 abort)。查询只发生在 abort 路径(正常完成零额外 RPC);自定义 LB 无此方法时保守回退。
- **P1 覆写**(`load_balancer.py`):`TokenWeightedLoadBalancer.is_request_migrating` —— sticky 指向的 server 在 `_draining` 或不在 active 池(drain 完成后 remove 已发生、客户端仍在 1s 重试睡眠内的窗口)→ True。
- **安全性前提**:L2 的 membership lock 使 drain 与权重更新互斥,"sticky 在 drain"不可能掩盖并发版本变更(§4.3/§4.4)。

效果:同版本迁移的样本完整性与训练模式解耦——**sync 训练也能安全使用波尾收缩迁移**,迁移许可由路由层状态而非算法容忍度决定。

## 后续阶段(见设计文档 §9)

P2(EngineStats 观测)、P3(收缩策略)、P4b-next(每步权重同步通道)、P4c(SharedRolloutBroker)在此包内继续扩展;L2 策略将挂接 `DynamicSchedulePolicyBase` + `register_policy`。
