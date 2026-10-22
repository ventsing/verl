# 进行中轨迹存储与故障接管

适用 `trainer_mode=laminar`，CPU/Mooncake relay、文本、单轮、engine DP=1、PP=1。
模型权重版本必须与 rollout 的行为 logprob 一致。下面是当前实现与验收边界。

**2026-10-20：默认使用 SQLite :memory:，journal_mode=MEMORY、synchronous=OFF、
temp_store=MEMORY，不创建数据库/锁文件，不执行恢复数据 fsync。**ACK 表示该独立
CPU actor 已接收的事务，不再表示磁盘持久化。存储 actor 禁止自动重启为空库；
它丢失时明确失败。健康 store 存活期间，rollout 的原版本接管和 epoch fence 保持。
落盘及完整任务恢复已登记为遗留事项；以下磁盘协议仅适用于显式 persist_to_disk=True。

## 数据通道与恢复协议

```mermaid
sequenceDiagram
    participant C as Laminar controller
    participant V as vLLM source
    participant P as Partial store CPU actor
    participant R as CPU/Mooncake relay
    participant D as Healthy vLLM target
    C->>P: create immutable request + begin_attempt(epoch)
    C->>V: generate(version, epoch, remaining budget)
    loop Every 128 tokens or 1 second with output
        V->>P: append(offset, token delta, behavior logprob delta)
        P-->>V: ACK after store transaction (memory by default)
    end
    V--xC: Actor death / RPC loss / heartbeat or progress timeout
    C->>P: fence old epoch and read committed prefix
    C->>D: Reserve healthy capacity for original version
    opt Target has newer weights
        C->>R: acquire exact old version
        R->>D: Reload original weights on every TP rank
    end
    C->>D: Re-prefill original prompt + committed tokens; decode remaining budget
    D->>P: Stream new epoch, starting at committed offset
    P-->>C: Acknowledged terminal state
```

- `partial_store.py` 的独立 CPU Ray actor 保存原 prompt、采样参数、响应预算、
  固定版本、attempt epoch、连续 token/logprob chunk 和终止状态。
- **可选磁盘模式**的 SQLite 使用 rollback journal、`synchronous=FULL`。只有提交成功后才确认；
  流式调用等待 ACK，形成背压。相同 chunk 的重试幂等，冲突、越界、乱序、缺失
  logprob、非有限 logprob、旧 epoch 和跨版本更新均被拒绝。
- **可选磁盘模式**的存储 actor 最多自动重启 3 次，同一个数据库重新打开。文件系统必须支持 SQLite
  文件锁和 fsync；WAL 不用于共享 NFS。数据库默认在本次任务的 event_dir 中。
  本地 `/tmp` 路径只支持该节点内的进程恢复，节点/磁盘丢失需要共享持久盘。
  数据库同时持有 writer 文件锁，拒绝另一个 live store 写同一路径。
- vLLM 使用 CUMULATIVE 输出，只提交新增后缀。其 token 列表引用会在异步 ACK
  期间增长，因此必须先冻结待提交的列表，再等待存储；不能事后用 live list
  的长度更新确认序号。仅在需要提交时才提取整段 logprob，避免逐 token 序列化。
- 故障时先在存储中提交 fence，再接管。旧进程即使复活也不能写入新 attempt。
  目标实例版本相同可直接接续；不同则在无活动请求时加载精确旧版本。
  **不会切换成最新权重，也不会使用当前 actor 重算 old_log_probs。**
- 恢复只重算 prompt/prefix 的 KV/SSM，已确认 token/logprob 不改动；未确认后缀
  可以重新采样。因此保证已确认前缀和行为策略一致，不保证故障前后随机序列逐位相同。
- 终止标记和最后一块在同一事务提交。最终 RPC 丢失时从存储返回完整结果，
  不重新生成。正在运行的相同 request ID 被拒绝；已完成 ID 可幂等读取。
  这是生成结果幂等，不是对 TQ/optimizer 的跨系统 exactly-once 事务。
- 普通调用保持唯一的逻辑 ID，避免 full_determinism 下验证重复使用旧结果。
  显式重新挂接使用 controller 的同一个 ID，或客户端采样参数
  `_laminar_trajectory_id`；输入不一致时拒绝。
- policy/token/logprob 一致性错误会让 controller 进入 failure 状态，训练停止，
  不仅仅驱逐某个 GRPO 组。可恢复的 rollout 故障会隔离失败副本并接管；如果
  接管目标加载旧权重时也失败，继续选择下一个健康目标。

## 默认参数

配置位于 `trainer.v1.laminar`，两个 relay 后端通用：

```yaml
keep_last_versions: 2
weight_budget:
  max_versions: 4
  max_bytes: 103079215104 # 每个 relay 96 GiB
  capacity_timeout_s: 600.0
partial_store:
  enabled: true
  persist_to_disk: false
  path: null             # 仅 persist_to_disk=True 时使用；默认 :memory:
  max_active: 4096
  chunk_tokens: 128
  flush_interval_s: 1.0
  heartbeat_interval_s: 5.0
  heartbeat_timeout_s: 10.0
  heartbeat_failures: 2
  progress_timeout_s: 600.0
  recovery_timeout_s: 300.0
  max_retries: 3
```

1 秒刷新是在下一次 vLLM 输出到来时触发，不是静默 prefill 期间的定时 flush。
无输出超过 progress_timeout 时按停滞处理。长 prefill 应相应提高该阈值。
Mooncake acquire_timeout 默认 900 秒，覆盖 600 秒容量背压及传输等待。
`max_active` 达到上限会报错；正常 prompt feeder 的有界组缓冲应低于此上限。

## 权重何时清理，以及如何避免长尾造成 OOM

不使用固定 TTL 删除旧权重。controller 在每轮调度中（默认约 5 秒及事件唤醒）
更新保留集合：最新版本、健康副本当前版本、存储中所有尚未结束的轨迹版本。
所有节点 relay 都保留这些版本，leaf 尚未收到数据时也可先建立保留意图。
轨迹完成且没有健康副本继续使用后，释放保留 owner；relay 在 owner 和读取
lease 全释放、且该版本不在最近 `keep_last_versions` 中时清理快照。
因此清理延迟还包括副本 drain、拉取新权重和未完成的网络传输时间，不承诺固定 5 秒删除。

`weight_budget` 给每个 relay 设置版本数和字节数上限。达到上限时阻塞**新权重发布**，
健康 rollout 继续生成，完成旧轨迹并释放历史版本。发布者在 full_tensor/创建下一份
Ray 权重对象之前等待；首次导出还逐张量检查。保留版本和 staging 预算不够时
不能通过删除活跃旧版本继续。等待超过 capacity_timeout 会明确失败，保留恢复所需数据。
配置必须容纳最近保留版本和一份新版本；Mooncake 接收端还需同时容纳 native
receive buffer 和本地不可变 Ray copy。

预算核算包括：

- CPU relay 的不可变模型快照及将导出的下一版本；
- Mooncake master 与 Ray 共用的源缓冲，按版本避免重复计算；
- leaf 的已完成快照、原生接收缓冲及即将创建的本地 Ray 副本；
- 仍在传输或注销失败而隔离的缓冲。另有 max_pending_versions=2 的传输并发限制。

预算是上述权重 payload 的上限，**不是整个进程/机器 RSS 的硬上限**。Python/Ray
元数据、trainer 内存、设备权重、消费端张量转换和其他进程需要单独预留空间。
96 GiB 是本任务 9B BF16 的默认值，不适用于任意模型/机器。按各节点可用主机和
Ray object-store 内存配置，并观察 `accounted_bytes`、`capacity_waiters`、
`capacity_wait_seconds`、`retention_owners`。不要设置只允许两个快照而没有新版本 staging 空间的预算。

终止任务留下的未完成前缀不会因超时自动删除。显式丢弃已放弃的轨迹时，先确保
它已取消且调用完成，再执行 `await controller.abandon.remote(request_id, reason="...")`。
这会 fence 旧写入、保留废弃记录和原因、释放该轨迹对旧版本的引用；不能再接续该 ID。
`await controller.durability_status.remote()` 返回 pool 计数、未完成请求摘要和 relay
预算状态。已完成 token 记录暂不自动压缩/删除；内存模式下需观察长期运行容量，
终态确认后的有界清理仍是遗留事项。可选磁盘模式还需管理文件保留。

## vLLM 驻留权重作为备份：已记录的后续设计

用户提出健康 vLLM 实例中驻留的旧版模型也可以充当备份。这是可用的第二来源：
同版本健康实例直接接管已经实现；从实例导出权重重建 CPU relay 尚未实现。
后续导出协议必须核对全部 TP rank 的版本、冻结版本切换边界、恢复对应布局，并记录
副本位置和健康租约。最后一个驻留该版本的实例消失前，仍需有独立 CPU/持久化备份。
不能仅凭“某个 vLLM 还驻留旧权重”就释放唯一的故障恢复副本，也不应每条轨迹保存一整份模型。

## 验证记录

- CPU 回归包含真实 SQLite 进程强退与事务回滚、幂等、epoch fence、旧版本接管、
  心跳/进度超时、迁移、丢失最终响应、取消竞态、二次目标故障、预算背压与 staging 核算。
- `python3 -u probe_partial_recovery.py` 在现有 Ray 上只创建并终止自己的 actor：
  重启存储进程后保留已确认前缀；终止生成进程后，从已升级至 v1 的健康实例
  重载 v0 并续接，8 个 token 与 8 个 logprob 无重复或丢失。
  该探针使用确定性的 CPU token，不是模型推理测试。
- 首轮真实 16 NPU/Qwen3.5-9B/Mooncake TCP、256 response token 任务完成两次更新
  和 4 条验证，但发现一次 live cumulative list 竞态，随后修复并增加回归。
  该轮不能作为无错误验收结果。
- 修复后的普通复测 `qwen35_9b_partial_recovery_smoke_20261019_v2`：16 NPU、两次
  更新、64 条 train selected、4 条 val，SUCCEEDED/exit 0，无轨迹错误。166 个
  已完成响应、2649 个持久化 chunk 和 42496 个 token 通过独立审计。
- 真模型故障注入 `qwen35_9b_partial_recovery_fault_20261019`：终止该任务的
  `vllm_server_0_0`，目标轨迹已有 32 个确认 token、版本 v0。该实例的 8 条
  进行中轨迹全部由其他实例按 v0 接管；目标轨迹最终为 256 token。随后两次
  更新和 4 条验证全部完成，SUCCEEDED/exit 0，没有 prompt 驱逐错误。
  142 个完成响应、2273 个 chunk、36352 个 token 通过持久化及队列谱系审计；
  64 条 train selected、4 条 val selected，正常关闭时剩余 24 条未完成请求。
- 本次真卡任务是单物理节点、原生 Mooncake TCP、256 token 的执行/容错验收，
  不代表 RDMA、多节点或 24K 长期训练性能/学习效果已经验收。目标已升级后重载
  原权重版本的场景由真实 Ray 进程探针和 CPU 权重加载测试单独覆盖。

```bash
python3 -m pytest -q -o cache_dir=/tmp/verl-partial-tests \
  tests/trainer/ppo/v1 tests/models/test_qwen35_decoder_compat_on_cpu.py
python3 -u probe_partial_recovery.py
```

## 仍在边界之外的恢复

当前恢复目标是 **rollout 进程故障下，健康实例接管进行中的轨迹**。没有实现：

- 故障 TP/NPU 实例在同卡上的自动重建、自动扩缩容和节点级资源重新分配；
- 机器失联后的 relay 链重配置、跨机器 coordinator 选主以及整个 Ray/object store 故障后的历史权重再生；
  已实现的进程级 native 链重建和同节点 master 切换见 RELAY_RECOVERY.md。
- controller/trainer 自动重建及 actor/optimizer/TQ/inflight 的一致 checkpoint。
  重新挂接数据库还要求精确的旧权重仍可取得；只保存 token 不能恢复丢失的模型版本。
- 原生 KV/SSM/RNG 跨实例拷贝、多轮工具状态和多模态轨迹。

这些与当前的进行中轨迹存储/健康实例接管分开验收，不把数据库重开当成整集群灾难恢复。


## 默认内存模式的恢复边界与遗留事项

relay 权重/catalog 和 partial store 默认均不落盘。native/endpoint 和 vLLM 进程恢复
仍依赖存活的内存管理服务；不宣称 catalog、partial-store、机器或整个 Ray 集群丢失后
可以继续原任务。落盘/异步备份、终态数据清理和完整 checkpoint 一致恢复暂缓开发。
relay 进程恢复见 [RELAY_RECOVERY.md](RELAY_RECOVERY.md)。
