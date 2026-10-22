# 权重发布优化

2026-10-20：默认内存恢复模式使用有界并行 SHA256，并复用 catalog 对不可变源对象的验证。
权重仍在完整 CPU 快照、catalog 提交和本地 endpoint 安装后才确认发布；不等待所有 rollout
同时加载。远端广播及每个 rollout 的设备加载另行计时。这里的发布时间不代表全体实例加载完成。

## 关键路径

原任务 `qwen35_9b_memory_roofline_20261020_v2` 的两次训练后发布分别为
33.284 / 30.638 秒。relay 事件显示每版 catalog 摘要约 11.7–11.8 秒，随后本地提交
约 12.9–13.4 秒；后者包含对同一组源对象重复扫描 SHA256。

本次改动：

1. catalog 首次校验按张量并行计算 SHA256，保留张量顺序与原有摘要语义，不复制 payload。
   总量不足 64 MiB 或不足两个张量时保持串行；线程数限制为 1–32，默认 4。
2. 本地 endpoint 询问 catalog：版本必须已提交，节点、名称、shape、dtype、字节数、
   SHA256 及每个实际 Ray ObjectRef ID 都必须与 catalog 保存的原始源一致。
   只有这些条件全部成立，才复用已验证结果，避免第二次读取全模型。
3. 新引用、可变 numpy 数组、接收节点、不同版本或 schema 均不走上述快路径。
   网络传输的逐块校验、版本/epoch 隔离、预算与 lease 清理规则保持原有语义。
4. 相同已提交版本、相同不可变源引用的 RPC 重试可以复用首次结果；不同内容仍拒绝冲突。
   校验过程被取消时，不将未提交数组长期 pin 在 catalog 中。

## 配置

```yaml
trainer:
  v1:
    laminar:
      mooncake:
        recovery:
          persist_to_disk: false
          hash_workers: 4
          reuse_verified_source: true
```

catalog 和每个 endpoint 各预约 `hash_workers` 个 Ray CPU，native core 仍预约一个。
排查时可设 `hash_workers: 1`、`reuse_verified_source: false` 恢复串行、完整校验路径。
显式磁盘模式仍执行其原有落盘和校验流程，不使用内存源引用认证快路径。
默认不新增模型文件、catalog 或轨迹数据库；异步持久化备份仍是遗留事项。

## 计时口径

训练日志的 `laminar/publish_*` 指标包含：

| 指标后缀 | 含义 |
| --- | --- |
| `seconds` | 从发布函数进入到本地提交确认，含容量等待 |
| `capacity_seconds` | 导出前和导出中的预算背压等待 |
| `export_seconds` | 参数迭代、CPU 转换、Ray put 和 Python bookkeeping，不含容量等待 |
| `iterator_seconds` | 等待参数迭代器，包括 FSDP gather 的可见耗时 |
| `cpu_copy_seconds` / `ray_put_seconds` | CPU 转换与 Ray object-store 写入 |
| `relay_seconds` | 发送 catalog commit 到收到本地确认的 RPC 时间 |
| `catalog_fetch_seconds` / `catalog_hash_seconds` | catalog 解析源对象及校验 |
| `catalog_control_seconds` | catalog 元数据提交、endpoint 同步和查询 |
| `native_source_seconds` | native 源提交，包含下面的原生与 endpoint 阶段 |
| `native_resolve_seconds` / `native_prepare_seconds` | native 解析源对象、准备和注册源内存 |
| `endpoint_install_seconds` / `endpoint_hash_seconds` | 本地 endpoint 安装及其中摘要时间 |
| `endpoint_checksums_reused` / `catalog_hash_reused` | 是否复用既有校验，0 或 1 |
| `catalog_wait_seconds` | catalog 进入处理前的容量、操作串行化与恢复等待 |

这些指标存在父子包含关系，不能全部相加。`published` relay 事件也保留 catalog/native
阶段计时，初始化版本可从该事件查看。多节点广播、消费者 pull 和 vLLM device load
不属于发布确认的端到端边界，不能由 `publish_seconds` 推断其耗时。

## 验证

- 完整 V1 + Qwen3.5 decoder CPU 回归：335 passed、2 skipped。
  新增用例覆盖并行摘要顺序、CPU 快照隔离、阶段计时、非主 rank 消费迭代器，
  以及 ref、版本、节点、epoch、shape、dtype、名称、字节数、摘要变化时拒绝复用。
- 真实 Qwen3.5-9B safetensors 驻内存摘要基准，775 张量、19,306,216,416 字节：
  单线程 16.102 / 14.141 秒；4 线程 3.014 / 3.454 秒，中位数加速 4.68 倍，
  所有 SHA256 按顺序完全一致。此基准包含模型文件的全部原始 dtype，
  与训练发布时转换为 BF16 的 payload 字节数不同；不包含读盘或模型导出时间。
- 原生 Mooncake TCP 故障注入通过 intermediate、master、endpoint、root endpoint
  四项，两个精确版本恢复和旧版本 GC 均通过，报告目录
  `/tmp/verl-relay-recovery-784pcmkl`。这是一台物理机器的三个逻辑 relay，非跨机 RDMA 证据。

真实 16 NPU 任务 `qwen35_9b_publish_fast_20261020` **SUCCEEDED**，8 FSDP2 训练卡 +
4 个 TP2 rollout，Qwen3.5-9B 每版发布 18,819,627,488 字节。与上述基线相同的
两次更新、64 条训练样本、4 条验证、256-token smoke 配置，结果如下（秒）：

| 阶段 | 更新 1 | 更新 2 |
| --- | ---: | ---: |
| 基线完整发布 | 33.284 | 30.638 |
| 优化后完整发布 | 13.970 | 10.744 |
| 导出 | 9.383 | 5.585 |
| 其中 CPU 转换 | 6.926 | 2.595 |
| 其中 Ray put | 2.012 | 2.491 |
| catalog 摘要 | 3.875 | 3.926 |
| 本地 endpoint 安装 | 0.064 | 0.483 |

两次观测平均从 31.961 秒降至 12.357 秒，发布延迟减少 61.3%，
约 2.59 倍加速。初始化及两次更新的 endpoint 均复用校验，endpoint 摘要耗时为零。
剩余重点是 CPU 转换、Ray put 及 FSDP 导出布局；本次未实现 pinned buffer 池或 TP 预重分片。

事件审计一致：版本 0/1/2、174 条完成事件、一次 preflight 迁移、无未完成 pull。
结束时另有 32 条请求没有 terminal 记录，终止清理仍是遗留事项，不能宣称全部请求闭环。
未生成恢复模型文件或数据库；内存模式无法提供离线持久化 token/chunk 审计。

报告：`/tmp/verl-laminar-runs/qwen35_9b_publish_fast_20261020/final_publish_speed_report.json`。
这两次短更新是发布路径实测，不能替代 24K 长尾、非零梯度学习、稳态训练吞吐 A/B、
多物理节点 RDMA 或 32/64 卡 scaling 验证。
