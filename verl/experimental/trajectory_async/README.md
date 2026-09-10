# Trajectory-level async RL on verl (experimental)

An implementation of trajectory-level asynchronous RL training after the
Laminar design (arXiv 2510.12633): single-response delivery granularity,
trainer-side GRPO group reassembly, a hierarchical relay tier for
asynchronous weight synchronization over verl's P2P checkpoint engines,
and KVCache-aware trajectory repacking.

**Real components only.** Everything in this package is deployable code
or a pure algorithm; nothing simulates the system to pretend a feature
exists. What is not implemented yet is a tracked TODO (below), not a
mock. The only test double is `FakeP2PBackend` — a bytes-level transport
for CPU tests of the real store/tier orchestration (real deployments use
the kimi/mooncake adapters).

## Module map

| Module | Role |
|---|---|
| `types.py` | data model: `TrajectorySample` (one response: uid, traj_index, model_version, reward, attempts, payload), `GroupRecord`, `TrajectoryStatus` |
| `group_aggregator.py` | trainer-side GRPO group reassembly from per-trajectory rows; FAILED-sentinel eviction protocol; late-arrival guard; buffer limits |
| `group_collector.py` | consumption core: `TrajectoryBatchCollector` forms exact mini-batches of complete, fresh groups under staleness control, with full accounting (trained / evicted / dropped-stale / leftover / incomplete); `row_from_sample_batch` adapts verl `DataProto` rows (traj_index / model_version / rollout_failed / uid); `grpo_group_advantages` (group-preserving contract) |
| `mini_batcher.py` | exact-size mini-batch formation over completed groups |
| `async_trainer.py` | `TrajectoryAsyncTrainer(FullyAsyncTrainer)` — the real separate-deployment trainer (ray remote), overrides sample intake to route through the collector; emits `trajectory_async/*` metrics |
| `versioned_weight_store.py` | multi-version, pull-based weight store over P2P checkpoint engines: `VersionedWeightStore` (registry, retention GC, per-consumer state/lag) + `KimiP2PBackend` / `MooncakeP2PBackend` adapters + `make_p2p_backend` factory (`--p2p-backend`) + `FakeP2PBackend` (CPU tests) |
| `relay_tier.py` | hierarchical relay tier (Laminar §4): `RelayService` / `RelayNode` (one per rollout machine; master stage = the whole actor stall; background chunk-pipelined chain distribution; anytime local pull of the latest complete version); trainer-side `format_fn` (HF format) + `reshard_fn` (rollout TP layout) hooks. Also the repack execution seam: `RolloutRepackExecutor` over `RolloutReplicaHandle` replicas (recompute or KV-transfer prefill, pluggable `kv_transfer_fn`) |
| `repack.py` | active scheduling (Laminar §5): `ReplicaState` idleness (KVCache ramp-down), Algorithm 1 `best_fit_consolidation` (pure function), `RepackManager` (periodic + post-update triggers, drives any `RepackExecutor`), `MigrationResult` |
| `relay_controller.py` | the Ray-native control plane of the versioned pull path: `RelayController` (version registry, retention, pull policy, metrics — CPU-testable) + `build_relay_controller` / `make_relay_controller_actor` wiring it over a real `CheckpointEngineManager` |
| `rollout_producer.py` | `TrajectoryLevelRollouter(FullyAsyncRollouter)` — the trajectory-level producer: one queue message per response (uid / traj_index / group_size / model_version stamped), FAILED rows on failure, batch-boundary weight pulls |
| `trajectory_async_main.py` | the launcher: `TrajectoryAsyncTaskRunner` wiring `TrajectoryLevelRollouter` + `TrajectoryAsyncTrainer` + MessageQueue + the relay controller (mirrors `fully_async_main.py`) |

## Relation to the v1 separate-async stack (TransferQueue replay buffer)

verl's newer separate-async path (`verl/trainer/ppo/v1/`) stores
experiences in a TransferQueue-backed `ReplayBuffer` (per-trajectory
keys `{uid}_{session_id}_{index}`, group re-assembly, oldest-first
sampling, staleness eviction `max_off_policy_threshold` drop/wait, DAPO
filter, failure eviction, refill, custom-sampler hook) fed by
`agent_loop_tq.py` (one `kv_put` per agent-loop output) — i.e. the
Laminar experience buffer and the trajectory-level producer exist
upstream on that path.

Positioning: `TrajectoryAsyncTrainer` extends the `fully_async_policy`
**MessageQueue** path, where the collector provides the group
re-assembly + staleness control. On the v1 TransferQueue path the
collector is redundant. What this package adds on EITHER path: the
**relay tier** (weights) and the **repack** algorithm + executor, plus
the alignment audit. Combining the relay tier + repack with the v1 stack
is the most direct route to a full Laminar deployment.

## Real-engine wiring guide

### Weights: multi-version pull (P0 wiring — implemented)

The stock `CheckpointEngineManager.update_weights` is a one-shot collective
broadcast: every actor rank `send_weights` (register + gather + UNregister),
every rollout rank receives, all in lockstep. The versioned pull path
replaces it (kimi engine only for now):

* **engine** (`kimi_checkpoint_engine.py`): `stage_version` registers the
  actor's CPU shards under `"{ckpt}:v{version}"` and does NOT unregister —
  the registered shards ARE the relay memory, pullable until retention
  retires them (`unstage_version`). `gather_metas` is collective, so the
  rollout ranks run `gather_version_metas` concurrently and every rank
  snapshots the version's metas; `receive_weights_version` pulls from the
  snapshot (anytime, repeatable, no re-gather).
* **workers**: `TrainingWorker.stage_weights_version` (actor side) and
  `CheckpointEngineWorker.gather_version_metas` / `pull_weights_version`
  (rollout side) expose the engine methods to worker-group dispatch.
* **controller** (`relay_controller.py`): a Ray actor holding version
  registry + retention (`keep_last`) + pull sequencing. `publish` fires
  the actor stage and the rollout gather CONCURRENTLY — that metadata
  phase is the whole trainer-side stall; `pull` runs the stock
  abort → release-kv → load → resume sequencing, but at a moment the
  ROLLOUT side chooses.
* **trainer** (`async_trainer.py`): with `async_training.weight_store`
  set, `_setup_checkpoint_manager` builds the controller and injects it
  into the rollouter; `_fit_update_weights` publishes per version
  instead of the collective push.
* **producer** (`rollout_producer.py`): before submitting each prompt
  group, `_maybe_pull_weights` pulls if the trainer published a newer
  version — the batch-boundary pull — and stamps every row with the
  version it generates under.

P0 topology (known limit, next cluster item): the stock kimi engine group
spans actor + ALL rollout ranks and `receive_tensor` barriers on it, so
pulls are fleet-synchronized (every replica pulls the same version at the
same moment, chosen by the rollout side — with ONE rollout replica this is
exactly per-replica anytime pulling). Per-replica process groups — true
per-replica, any-version pulls — plus chunk-pipelined distribution are
what `relay_tier.py` (the CPU-verified tier) models and the remaining
cluster work; all engine-group traffic is serialized behind one lock until
then. The mooncake engine raises `NotImplementedError` on `stage_version`
(per-version RDMA staging buffers still to design — TODO below).

Run it (see `examples/trajectory_async/`):

```bash
python -m verl.experimental.trajectory_async.trajectory_async_main \
    async_training.weight_store.backend=kimi \
    rollout.checkpoint_engine.backend=kimi_ckpt_engine \
    [examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh for the full recipe]
```

`async_training.weight_store=null` falls back to the stock push path;
`async_training.trajectory_group_assembly=False` falls back to stock
group-level consumption (either producer granularity works on the
trainer side either way).

### Repack: active scheduling over real rollouts

`RolloutRepackExecutor` binds Algorithm 1 to replicas that implement
`RolloutReplicaHandle` (vLLM-stats mapping: `kv_cache_usage` × capacity →
`kv_used_tokens`, `max_num_seqs` → `batch_limit`, scheduler request
states → `running_requests` / `remove_request` / `admit_request`, a
collective-rpc weight reload → `pull_weights`). Migration semantics:
move each running request to its destination — recompute prefill
(portable: resend prompt + partial response) or `kv_transfer_fn` (real
KV movement); freed sources immediately pull the latest weights.

## Cluster TODO list (ordered)

**P0 — make the real path runnable** ✅ DONE (this branch)

1. ~~Launcher~~ — `trajectory_async_main.py` + hydra config
   (`examples/trajectory_async/config/`) + run script
   (`examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh`).
2. ~~Deployment-path decision~~ — MessageQueue path (`fully_async_policy`):
   `TrajectoryLevelRollouter` emits 1-row messages (per-row generation
   tasks, per-row delivery, FAILED rows); re-targeting the v1
   TransferQueue stack (where only the relay tier + repack would port)
   remains a documented alternative, not needed for this path.
3. ~~Worker-side stage_version~~ — kimi engine `stage_version` /
   `gather_version_metas` / `receive_weights_version` / `unstage_version` /
   `drop_version` + `TrainingWorker.stage_weights_version`.
4. ~~Replica-side pull driver~~ —
   `CheckpointEngineWorker.pull_weights_version` + `RelayControllerActor.pull`
   (abort → release-kv → load → resume), invoked at group-submission
   boundaries by `TrajectoryLevelRollouter._maybe_pull_weights`.
5. ~~kimi read-side placement~~ — `KimiP2PBackend.read_into` now uses
   `consumer_ctx["engine"]` (the receiver's parameter server drives the
   pull), matching the mooncake adapter and stock semantics.

**P1 — validate the written code on a real machine**

6. Full-run of `examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh`:
   the launcher, the producer's per-row messages against a real ALM/vLLM,
   the kimi stage/gather/pull collectives, and the trainer's group
   re-assembly end to end (syntax + CPU logic verified; the cluster run
   is the actual gate).
7. `row_from_sample_batch` against real DataProto (`union(position)`
   slicing, `non_tensor_batch` `.item()` paths) — including the FAILED
   row shape (`DataProto(non_tensor_batch=...)` with no tensor batch).
8. Pinned-memory accounting of `keep_last` versions on the actor ranks
   (each version pins a full CPU shard copy per rank).
9. mooncake `stage_version`: per-version RDMA staging buffers + runtime
   `batch_register_memory`/`unregister_memory` semantics + per-version
   buffer descriptor distribution (the engine currently raises
   NotImplementedError by design).
10. kimi per-replica process-group topology: today pulls are
   fleet-synchronized on the single engine group; one group per replica
   (plus serialized-but-per-replica receive) gives true per-replica,
   any-version pulls — the P0 controller serializes behind one lock
   until then.
11. Concurrent collective safety: publish while a pull is in flight
   (gather_metas vs receive_tensor on the same group) — currently
   prevented by the controller lock; verify whether the kimi store
   tolerates overlap before relaxing it.
12. Relay-tier chain distribution (`relay_tier.py`, CPU-verified) on real
   transports: one relay engine per rollout machine, master-side
   format/reshard hooks, chunk-pipelined chain — replaces the flat
   fleet pull once multi-machine.
13. Collector behavior under real queue semantics (cloudpickle'd
   samples, `put_sample(None)` termination).

**P2 — close the remaining gaps vs the paper**

14. `RolloutReplicaHandle` implementation against the real rollout stack
    (vLLM/sglang scheduler APIs) + a real `kv_transfer_fn`; wire
    `RepackManager` + `RolloutRepackExecutor` into the trainer's update
    path (post-publish trigger) and the rollout side.
15. Relay tier elasticity: per the paper's §4.3, O(1) chain rebuild on
    relay failure, master failover (deliberately deferred with fault
    tolerance as a whole).
16. Partial response pool + fault tolerance (paper §3.3): stream
    in-progress trajectories centrally; on replica failure redirect to a
    same-version replica reusing partial progress. The single biggest
    remaining design pillar with no counterpart here. (FAILED-row
    delivery at trajectory granularity already landed with the producer;
    the pool + redirect is the missing part.)

## Relation to the Laminar paper (alignment status)

| Paper element | Status |
|---|---|
| §3 trajectory-level asynchrony, no lockstep; emergent per-trajectory staleness (§6, no static k) | ✅ collector + trainer (MessageQueue path); ✅ upstream v1 (TransferQueue) |
| §3.1 experience buffer (sampling/eviction) | ✅ upstream v1 `ReplayBuffer`; ⚠️ here: collector is FIFO + staleness refusal only |
| §3.1 prompt pool | ✅ upstream v1 (streaming dataloader + refill) |
| §3.1 partial response pool (fault-tolerance substrate) | ❌ TODO-16 |
| §3.2 workflow steps ④-⑦ (interleaved train/publish/background distribute/anytime pull) | ✅ P0 wiring (`relay_controller.py` + `rollout_producer.py` batch-boundary pulls); chain distribution = TODO-12 |
| §3.3 + §4.3 fault tolerance (heartbeat failover, chain rebuild, master failover, checkpoint recovery) | ❌ TODO-15 (deferred) |
| §4.2 relay hierarchy: master + per-machine relays, resharding, chain-pipelined broadcast, PCIe local pull | ⚠️ P0 flat path via `relay_controller.py` (versioned stage + fleet pull, live); per-machine chain tier = `relay_tier.py` (CPU-verified) = TODO-12 |
| §4.2 actor stall = single push to master | ✅ `publish` returns after the master stage |
| §5 repack: triggers, version grouping, KVCache idleness, Algorithm 1 Best-Fit + CanFit(`C_max` ∧ `B`), freed sources pull fresh weights | ✅ `repack.py` + `RolloutRepackExecutor`; real-stack handle TODO-14 |
| §8 convergence guarantees / multi-iteration training | ❌ out of scope here (no model, no gradient step) |

## Test coverage

`tests/experimental/trajectory_async/` (87 tests; 80 stdlib-only + 2
torch-gated adapter smokes + 4 ray-gated wiring smokes + 1 torch-gated
kimi read-placement smoke, all skipping gracefully without their deps):

* aggregator: completion order, duplicates, FAILED-eviction protocol,
  late-arrival guard, buffer limits, record invariants;
* collector: group/trajectory/mixed granularity, eviction accounting,
  staleness refusal, reconciliation identity, DataProto row adapter;
* mini-batcher: exact sizes, group advantages preserved;
* weight store: publish/manifest, retention GC, pinned/latest pulls,
  concurrent pulls, per-consumer accounting, backend factory switching;
* relay tier: chain distribution, chunk pipelining faster than
  sequential hops, actor stall = master stage only, anytime local pull,
  startup-edge wait, retention, HF-format/TP-reshard hooks, version
  monotonicity;
* repack: Algorithm 1 pure-function tests (Best-Fit, CanFit, busy/empty
  exclusion), manager triggers, consolidation over the real executor,
  exception survival;
* relay controller (stdlib): version registry, duplicate-publish
  idempotence, keep_last retention (never retires the latest), pull
  defaults + explicit older versions + retired-version LookupError,
  publish-failure cleanliness, strict publish/pull serialization;
* adapter smokes (torch-gated, full machine): the real
  KimiP2PBackend / MooncakeP2PBackend against duck-typed stub engines,
  including the consumer-engine placement of kimi `read_into`;
* wiring smoke (ray-gated, full install): the P0 launch path imports
  (producer/trainer/launcher/engine-worker methods) and the FAILED-row
  message contract round-trips through `row_from_sample_batch`.

```bash
python3 -m unittest discover -s tests/experimental/trajectory_async -t .
```
