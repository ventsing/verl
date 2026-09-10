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
| `group_aggregator.py` | trainer-side GRPO group reassembly from per-trajectory rows; FAILED-sentinel resolution — strict eviction (default) OR long-tail survivor delivery (`min_group_survivors`: one straggler must not poison its n−1 healthy siblings); `group_deadline_s` bounded head-of-line blocking; late-arrival guard; buffer limits |
| `group_collector.py` | consumption core: `TrajectoryBatchCollector` forms exact mini-batches of complete, fresh groups under staleness control, with full accounting (trained / evicted / dropped-stale / leftover / incomplete); `row_from_sample_batch` adapts verl `DataProto` rows (traj_index / model_version / rollout_failed / uid); `grpo_group_advantages` (group-preserving contract) |
| `mini_batcher.py` | exact-size mini-batch formation over completed groups |
| `async_trainer.py` | `TrajectoryAsyncTrainer(FullyAsyncTrainer)` — the real separate-deployment trainer (ray remote), overrides sample intake to route through the collector; emits `trajectory_async/*` metrics |
| `versioned_weight_store.py` | multi-version, pull-based weight store over P2P checkpoint engines: `VersionedWeightStore` (registry, retention GC, per-consumer state/lag) + `KimiP2PBackend` / `MooncakeP2PBackend` adapters + `make_p2p_backend` factory (`--p2p-backend`) + `FakeP2PBackend` (CPU tests) |
| `relay_tier.py` | hierarchical relay tier (Laminar §4): `RelayService` / `RelayNode` (one per rollout machine; master stage = the whole actor stall; background chunk-pipelined chain distribution; anytime local pull of the latest complete version); trainer-side `format_fn` (HF format) + `reshard_fn` (rollout TP layout) hooks. Also the repack execution seam: `RolloutRepackExecutor` over `RolloutReplicaHandle` replicas (recompute or KV-transfer prefill, pluggable `kv_transfer_fn`) |
| `repack.py` | active scheduling (Laminar §5): `ReplicaState` idleness (KVCache ramp-down), Algorithm 1 `best_fit_consolidation` (pure function), `RepackManager` (periodic + post-update triggers, drives any `RepackExecutor`), `MigrationResult` |
| `relay_controller.py` | the Ray-native control plane of the versioned pull path: `RelayController` (version registry, retention + staged-bytes quota, fleet AND per-replica pull drivers with per-replica locking, metrics — CPU-testable) + `derive_replica_partition` + `build_relay_controller` / `make_relay_controller_actor` wiring it over a real `CheckpointEngineManager` |
| `rollout_producer.py` | `TrajectoryLevelRollouter(FullyAsyncRollouter)` — the trajectory-level producer: one queue message per response (uid / traj_index / group_size / model_version / attempts stamped), bounded row retries (`row_retry.py` policy; retried rows re-stamp the version they actually generated under), FAILED rows on budget exhaustion, batch-boundary weight pulls |
| `row_retry.py` | bounded per-row generation retry policy (stdlib-pure, CPU-testable): attempt budget, per-attempt version stamping, FAILED-sentinel delivery on exhaustion — the cheapest long-tail mitigation (a transient single-response failure stops terminating whole groups) |
| `partial_pool.py` | Partial Response Pool (paper §3.1 substrate): central, version-gated store of in-progress trajectories (same-version redirect rule enforced on every read; TTL + LRU + byte-quota bounded) — the producer's row retries consult it; token-level writers are the documented external seam |
| `fault_tolerance.py` | fleet + relay fault tolerance (§3.3 heartbeat + §4.3 master failover): `ReplicaHealthMonitor` (consecutive-failure strikes, hysteresis), `RelaySupervisor` (controller heartbeat → recreate + recover + re-attach every consumer; replica probes → retire from routing / revive), `rebuild_chain` (O(dead) relay-chain splice, pure core) |
| `repack_bridge.py` | the repack closed loop over the real rollout fleet (paper §5): `RolloutReplicaView` (LB in-flight probe + per-replica pull), `FleetRepackExecutor` — idle-replica refresh after each publish PLUS cross-replica migration as the DRAIN LIFECYCLE (`begin_drain` steering → optional hard abort with client-side transparent resume → completion watcher pulls fresh weights into emptied sources and `end_drain`s them back to routing), real async fleet snapshots feeding the Best-Fit planner, `build_repack_controller` / `make_repack_controller_actor` |
| `staleness_correction.py` | loss-side version-staleness correction: per-trajectory reweighting by version age (`staleness_weights`), adaptive clip scaling (`adaptive_clip_scale`), version-cohort GRPO baselines (`cohort_advantages`), cohort diagnostics; `apply_staleness_correction` attaches `staleness_weights` / `cliprange_scale` batch columns consumed by the policy loss (see below) |
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

### Phased v1-integration route (data plane: TransferQueue; control + parameter planes: this package)

Grounded in verified upstream facts: the v1 TQ writer already stamps
per-trajectory weight versions (`agent_loop_tq.py` tags
`min/max_global_steps`, sourced from the rollout client's
`extra_fields` — the engine's bound version is observable on that path
today), the replay buffer already filters on them
(`max_off_policy_threshold`), and the v1 trainer's weight sync is the
synchronous `update_weights` collective — the exact actor stall the
relay tier removes. The phases:

* **Phase 0 — zero-architecture staleness correction.** The version
  signal exists; what is missing is only consuming it at the v1 loss:
  `staleness_correction.apply_staleness_correction` is
  path-agnostic (reads `model_version` per row, computes
  `τ = v_trainer − v_rollout` in trainer memory, no relay push needed
  for the signal — the relay tier distributes weights; it does not
  need to distribute versions). A v1-side adapter maps the TQ tags to
  the row field and calls it before the loss.
* **Phase 1 — relay-tier mount.** `RelayControllerActor` +
  `VersionedWeightStore` are already standalone actors: the v1 trainer
  publishes after staging (it does not wait for fleet-wide reload),
  and a v1 rollout-side batch-boundary pull driver (the same seam as
  this package's producer) fetches per replica. Per-replica process
  groups (landed here, P0 item 10) are the prerequisite that keeps
  pulls from degrading to global-barrier broadcasts.
* **Phase 2 — repack as the resident monitor.** `notify_update` on
  publish + best-fit drain (landed here) with the new
  `repack.drain_deadline_s`: soft drain FIRST (in-flight finishes,
  zero recompute), escalate to abort only past the deadline — the
  v1-integration posture of drain-first with abort as the bounded
  fallback, so frequent publishes cannot thrash long generations into
  repeated re-prefill.

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

Topology (LANDED, cluster validation pending): `derive_replica_partition`
+ `build_process_group(replica_partition=...)` install one engine
subgroup per rollout replica; `receive_weights_version(version,
replica_id=...)` pulls over that subgroup alone (H2D bucket partition +
barriers touch only the replica's ranks), and `RelayController.
pull_replica` drives it with per-replica locks — distinct replicas pull
CONCURRENTLY; fleet pulls and publishes take all locks (ordered,
deadlock-free — the concurrent-collective-safety question is TODO-11).
The producer's batch-boundary fleet pull is unchanged. Chunk-pipelined
chain distribution remains `relay_tier.py`'s CPU-verified model
(TODO-12). The mooncake engine raises `NotImplementedError` on
`stage_version` (per-version RDMA staging buffers still to design —
TODO below).

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

### Repack: active scheduling over real rollouts (drain lifecycle — implemented)

`FleetRepackExecutor` binds Algorithm 1 to the real fleet through the
rollouter's LB + manager seams (committed sockets: `remove_servers` /
`add_servers`, server-handle RPCs; the stock `LLMServerManager` exposes
`server_handles` / `server_addresses` as parallel lists — replica index
== LB server index). The loop:

* **triggers**: `notify_update()` after every publish + a periodic
  `check_interval_s` tick (post-publish refresh of idle lagging
  replicas + migration planning over version-consistent idle groups).
* **migration = the drain lifecycle**: `begin_drain` steers NEW work
  away from consolidation sources (crash-safe: sources are marked
  BEFORE any abort, so a watcher always exists); soft mode lets
  in-flight finish; hard mode (`hard_drain=true`, requires the rollout
  stack's abort-resume semantics — L2 scheduling or
  `partial_rollout=true`) aborts in-flight on sources and clients
  transparently resume them on other replicas (recompute prefill —
  real KV movement is NOT implemented: the pluggable `kv_transfer_fn`
  hook exists only in `relay_tier.py`'s CPU model, the honest
  boundary); the completion watcher then pulls the fresh version into
  the emptied source over ITS engine subgroup and `end_drain` returns
  it to routing.
* **escalation**: `drain_deadline_s` bounds the soft wait — past the
  deadline the source's in-flight is aborted (once per source), so one
  long-tail generation cannot pin a migration.
* **execution-time re-checks**: CanFit is re-verified against live
  in-flight at execution; retired (dead) replicas are excluded from
  every path; `batch_bound` gates planning when no honest capacity
  exists.

`repack.enabled=false` disables the loop; `server_ids` overrides the
replica-index → LB-server mapping when the deployment order differs.

### Fault tolerance: heartbeat failover + replica retire/revive (implemented)

`fault_tolerance.py` + the trainer's supervisor wiring
(`async_training.fault_tolerance.*`, on by default):

* **controller failover (§4.3 master)**: `RelaySupervisor` heartbeats
  the `RelayControllerActor` (`ping`); on actor death it recreates the
  controller via a factory, RECOVERS state (`recover`: the trainer's
  `current_param_version` is the authority, engines keep their staged
  shards, replica versions re-sync on the next pull — the registry is
  derived data, so NO separate async checkpoint exists by design), and
  re-attaches every consumer (the producer's pull hook, the repack
  controller actor). A failed resurrection is counted and retried on
  the next heartbeat — the supervisor itself never crashes the run.
* **replica heartbeat (§3.3)**: liveness probes are a harmless
  `abort_request` RPC per server handle (a dead actor raises; a live
  one returns a not-found no-op). `ReplicaHealthMonitor` declares a
  server dead after `failure_threshold` consecutive failures; dead
  servers are RETIRED from LB routing (`remove_servers`) and from
  every repack lifecycle path (no refresh, non-routable, drains
  released, migration pairs declined); a restarted server is REVIVED
  (`add_servers`, zero in-flight — least-loaded routing refills it).
* **failure redirect**: a row that fails on a dead replica is retried
  by the row-retry loop onto healthy replicas (recompute prefill); with
  the partial pool enabled the retry resumes from same-version pooled
  progress instead of restarting.
* **chain rebuild (§4.3)**: `rebuild_chain` — the O(dead) splice
  (exclude dead ranks, neighbors reconnect, live ranks never
  re-receive) as a pure core; engine-side re-registration for the
  mooncake rank chain is the TODO-15 tail (kimi needs no chain:
  per-rank P2P reads are already failure-isolated per replica).

### Partial response pool: version-gated store + retry consumer (implemented substrate)

`partial_pool.py` + `async_training.partial_pool.*`: a central Ray
actor holding in-progress trajectories. Every read enforces the
same-version redirect rule (a cross-version partial is refused AND
dropped — cross-version resume would mix weight versions WITHIN one
trajectory, the exact property trajectory-level delivery protects);
TTL + LRU + byte-quota bound the store (never the just-put entry).
The consumer is wired: a row retry consults the pool — a same-version
hit becomes a resume hint on the retry (`row_retry.py`'s
`pool_consult_fn` → `generate_fn(row, hint)`; the producer attaches it
as `non_tensor_batch["resume_tokens"]`, which resume-aware rollout
paths honor and stock agent loops ignore), a miss restarts clean, and
a pool error never breaks a retry. The token-level WRITER is the
deliberately-external seam: partials must be checkpointed where tokens
are observable (the LLM client's generation loop or a resume-aware
agent loop — shared rollout infrastructure outside this package); the
pool actor's `put` RPC is the seam.

### Long-tail group policy: retries, survivors, deadlines (implemented)

Three knobs (`async_training.*`) so one straggler cannot poison its
group: `row_max_attempts` (bounded per-row generation retries;
retried rows re-stamp the version they actually generated under — a
mixed-version group flows through `version_span` + staleness
correction), `min_group_survivors` (deliver a terminally-failed
group's survivors as a trainable PARTIAL group; null = strict
eviction, the default; clamped ≥ 2 — a lone survivor has no
group-relative signal), and `group_deadline_s` (wall-clock bound on
head-of-line blocking; late siblings are counted and dropped).

## Configuration reference (`async_training.*`)

| Knob | Default | Meaning |
|---|---|---|
| `trajectory_group_assembly` | `True` | trainer-side GRPO re-assembly from per-trajectory messages (`False` = stock group-level consumption) |
| `staleness_drop` | `null` | per-trajectory staleness refusal (versions behind current); null = emergent staleness, rely on loss-side correction |
| `weight_store` | `backend: kimi` | versioned pull path; `null` = stock push-based sync |
| `weight_store.keep_last` | `2` | retained staged versions |
| `weight_store.max_staged_bytes` | `null` | host pinned-memory quota (retires oldest, never the latest) |
| `row_max_attempts` | `2` | per-row generation retry budget (1 = single-shot) |
| `min_group_survivors` | `null` | survivor-delivery threshold (null = strict eviction) |
| `group_deadline_s` | `null` | head-of-line blocking bound |
| `partial_pool.enabled` | `false` | attach the partial response pool actor |
| `partial_pool.max_entries` / `.max_bytes` / `.ttl_s` | `4096` / `null` / `null` | pool bounds |
| `fault_tolerance.enabled` | `true` | supervisor: controller failover + replica retire/revive |
| `fault_tolerance.heartbeat_s` | `10.0` | heartbeat + probe cadence |
| `fault_tolerance.failure_threshold` | `3` | consecutive probe failures before retire |
| `repack.enabled` | `True` | the closed loop (post-publish + periodic triggers) |
| `repack.check_interval_s` | `5.0` | periodic trigger cadence |
| `repack.min_group_candidates` | `2` | min idle candidates for consolidation planning |
| `repack.hard_drain` | `false` | abort in-flight on drained sources (requires abort-resume semantics; with `partial_rollout=false` on the stock client aborted requests are DROPPED) |
| `repack.drain_deadline_s` | `null` | soft-drain deadline before abort escalation |
| `repack.batch_bound` | `rollout.max_num_seqs` | CanFit decode batch capacity |
| `repack.kv_per_request` | `null` | KV-token estimate per in-flight request |
| `staleness_correction.mode` | `decay` | loss-side reweighting (`none` / `decay` / `exp`); full knob set in the yaml + staleness section below |

## Cluster TODO list (ordered)

Status at a glance: the core closed loop, fault tolerance (§3.3
heartbeat/retire + §4.4 controller failover), and the partial-pool
substrate are LANDED. P0 is complete; in P1 the per-replica process
groups and pinned-memory (staged-bytes) accounting have landed — what
remains is cluster validation and mooncake's per-version RDMA staging;
in P2 the two biggest open gaps are the relay-chain distribution's
real deployment and per-token KV introspection. For positioning vs
the v1 separate-async stack see the section above: this package's
incremental value on ANY path is the relay tier (weights) and the
repack algorithm + executor.

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

**P1 — validate the written code on a real machine** — PARTIALLY
LANDED HERE: per-replica process groups (item 10) and pinned-memory /
staged-bytes accounting (item 8) are done; the remaining items are the
cluster validation run itself (6) and mooncake per-version RDMA
staging (9).

6. Full-run of `examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh`:
   the launcher, the producer's per-row messages against a real ALM/vLLM,
   the kimi stage/gather/pull collectives, and the trainer's group
   re-assembly end to end (syntax + CPU logic verified; the cluster run
   is the actual gate).
7. `row_from_sample_batch` against real DataProto (`union(position)`
   slicing, `non_tensor_batch` `.item()` paths) — including the FAILED
   row shape (`DataProto(non_tensor_batch=...)` with no tensor batch).
8. ~~Pinned-memory accounting~~ — DONE: `stage_version` reports per-rank
   staged bytes, `RelayController` sums them and enforces
   `async_training.weight_store.max_staged_bytes` (retiring oldest live
   versions beyond `keep_last` AND the byte quota, never the latest);
   `relay/staged_bytes` / `relay/quota_retires` metrics. Cluster
   validation of the accounting itself remains (run with the metric on).
9. mooncake `stage_version`: per-version RDMA staging buffers + runtime
   `batch_register_memory`/`unregister_memory` semantics + per-version
   buffer descriptor distribution (the engine currently raises
   NotImplementedError by design).
10. ~~kimi per-replica process-group topology~~ — DONE: `init_process_group`
   takes a `replica_partition` (derived from the replicas' worker order;
   `derive_replica_partition`) and installs one subgroup per replica;
   `receive_weights_version(version, replica_id=...)` pulls over that
   subgroup alone (H2D bucket partition + barriers touch only the
   replica's ranks). `RelayController.pull_replica` drives it with
   per-replica locks — distinct replicas pull CONCURRENTLY; fleet pulls
   and publishes take all locks. Fleet-synchronized batch-boundary pulls
   (producer) are unchanged. Cluster validation pending (TODO-6).
11. Concurrent collective safety: fleet pulls/publishes take ALL
   per-replica locks (ordered — deadlock-free); per-replica pulls take
   only their own, so replicas pull concurrently on disjoint subgroups.
   Overlap of publish (global gather) with a per-replica pull is still
   driver-serialized (publish takes all locks); verify whether the kimi
   store tolerates overlapping gather_metas + a subgroup receive_tensor
   before relaxing further.
12. Relay-tier chain distribution (`relay_tier.py`, CPU-verified) on real
   transports: one relay engine per rollout machine, master-side
   format/reshard hooks, chunk-pipelined chain — replaces the flat
   fleet pull once multi-machine. Deadlock-safety rules for the binding:
   (i) LINEAR chain only, never a ring (no wait-for cycles); (ii) bounded
   in-flight per hop with an explicit completion signal before buffer
   reuse (the stock mooncake engine's magic-word double-buffering is the
   reference pattern); (iii) retire a version's staged chunks only after
   per-version completion tracking (a downstream RDMA read of an evicted
   buffer is an error, not a stall); (iv) keep the controller's
   stable-id lock ordering and never call back into a lock holder from
   inside a locked section.
13. Collector behavior under real queue semantics (cloudpickle'd
   samples, `put_sample(None)` termination).

**P2 — close the remaining gaps vs the paper** — MOSTLY LANDED:
items 14 (repack execution), 15 (fault tolerance), 16 (partial pool
substrate + long-tail mitigations) are live; the two biggest open gaps
are the relay-chain distribution on real transports (12) and per-token
KV introspection (13's real-queue semantics and 9's mooncake staging
also remain).

14. `RolloutReplicaHandle` against the real rollout stack — MOSTLY
    LIVE: the closed loop is wired (`repack_bridge.py`:
    `RepackControllerActor` owns the manager loop; the trainer's
    `_publish_versioned_weights` tail fires `notify_update`;
    `async_training.repack` config), and BOTH §5 mechanisms now execute:
    (a) idle-replica refresh (replicas with LB in-flight == 0 lagging
    the fresh version pull it per-replica right after a publish) and
    (b) CROSS-REPLICA REQUEST MIGRATION as the drain lifecycle — the
    engine pieces already existed scattered (LB drain sockets
    `begin_drain`/`end_drain`, server-level `abort_all_requests`, the
    fully-async client's transparent abort-resume with recompute
    prefill); what was missing was the driver. `FleetRepackExecutor`
    now supplies it: plans execute as `begin_drain(sources)` (new work
    steers to the planner's CanFit-verified destinations) → soft mode
    (default) lets in-flight requests FINISH on their source under the
    version they started on (no version mixing, no lost work) while
    hard mode (`repack.hard_drain=true`) aborts them for client-side
    resume elsewhere → the completion watcher pulls fresh weights into
    emptied sources and returns them to routing. Execution-time CanFit
    re-checks reject pairs on live counts; a crashing abort RPC degrades
    to soft instead of stranding a drained source; capability probes
    (drain socket absent → plan declined, `repack/migrations_declined`)
    keep the clean-branch behavior honest. Remaining boundaries:
    per-request DIRECTED placement is the LB's, not the executor's (the
    plan's Best-Fit math gates safety; actual redirect placement is
    least-loaded); hard mode requires abort-resume semantics on the
    rollout stack (L2 scheduling, or `partial_rollout=true` — with
    `partial_rollout=false` the stock client DROPS aborted requests);
    per-token KV introspection (KV columns are linear in the in-flight
    count, CanFit collapses onto the true batch bound
    `rollout.max_num_seqs`); a real `kv_transfer_fn` (KV blocks never
    travel — recompute prefill is the accepted default).
15. Relay tier elasticity — LANDED (§4.3 + §3.3): master failover is
    `RelaySupervisor` (fault_tolerance.py): heartbeats the
    RelayControllerActor; on actor death it recreates the controller
    via a factory, RECOVERS state (`RelayController.recover`: the
    trainer is the authority on the published version; engines keep
    their staged shards; per-replica versions re-sync on the next
    pull — the registry is derived data by design, no async
    checkpoint needed), and re-attaches every consumer (producer
    batch-boundary pulls, the repack controller actor). Replica-level
    fault tolerance: liveness probes (a harmless `abort_request` RPC
    per server handle — dead actors raise, live ones no-op) with
    consecutive-failure strikes; dead replicas are RETIRED from LB
    routing (`remove_servers`, a committed socket) and from every
    repack lifecycle path; restarted replicas REVIVE (re-added with
    zero in-flight, least-loaded routing refills them). Chain rebuild:
    `rebuild_chain` implements the O(dead) splice (exclude dead ranks,
    neighbors reconnect, live ranks never re-receive) — engine-side
    RDMA re-registration for the mooncake rank chain is the remaining
    cluster-validated TODO; the kimi path needs no chain (per-rank
    P2P reads are already failure-isolated per replica).
16. Partial response pool (paper §3.1/§3.3) — SUBSTRATE LANDED,
    WRITER SEAM EXTERNAL: stream in-progress trajectories centrally; on
    replica failure redirect to a same-version replica reusing partial
    progress. Scoping: the pool is a RELIABILITY substrate, not a
    requirement of trajectory-level async RL. And note the plain
    sentinel+eviction story was NOT enough for the true long tail — one
    straggler used to poison its n−1 healthy siblings (compute waste ∝
    group size, head-of-line blocking until the straggler settled, and a
    length-correlated dropout: failures concentrate on long generations,
    so the effective training distribution skewed short — a milder
    cousin of the paper's Appendix C critique of partial-rollout mixing).
    The landed mitigations: bounded ROW RETRIES (`row_max_attempts`,
    preserves the length distribution up to budget exhaustion; retried
    rows re-stamp the version they actually generated under, so a
    mixed-version group is exactly what version_span + loss-side
    staleness correction handle), SURVIVOR DELIVERY
    (`min_group_survivors`, opt-in — deliver the settled survivors as a
    trainable partial group instead of evicting; group_size keeps the
    original rollout.n; advantage normalization runs over survivors), and
    a GROUP DEADLINE (`group_deadline_s`, bounds head-of-line blocking;
    late siblings are counted and dropped — the accepted cost of a
    bounded wait). The pool itself has now LANDED as the substrate
    (`partial_pool.py`): a central, version-gated store of in-progress
    trajectories (same-version reads only — a cross-version partial is
    refused AND dropped, the same-version redirect rule; TTL + LRU +
    byte-quota bounded) with the consumer wired (row retries consult
    it: a same-version hit becomes a resume hint on the retry, a miss
    restarts clean; a pool error can never break a retry). The §3.3
    failure-redirect half is fully live WITHOUT the pool: probe →
    retire from routing → the row-retry loop redirects onto healthy
    replicas (recompute prefill). The one deliberately-external seam
    is the token-level WRITER: checkpointing partials as they are
    generated must sit where tokens are observable (the LLM client's
    generation loop or a resume-aware agent loop — shared rollout
    infrastructure outside this package; the pool actor's `put` RPC is
    the seam). Until a writer lands, the pool serves redirects at
    whole-trajectory retry granularity.

## Staleness correction (loss-side)

Honest positioning first: **the Laminar paper derives no importance-sampling
bias or convergence bound** — its Appendix D analyzes chain-broadcast
latency, and its §8.2 comparison is empirical. Its actual recipe is
trajectory version-atomicity (one weight version per trajectory — unlike
partial-rollout systems that mix versions *within* a trajectory), a bounded
observed staleness (≤4 in its runs), and a larger mini-batch (2048) to
stabilize off-policy training. Appendix C explicitly lists IS-based
experience sampling as future work.

What the loss path already has (no new code needed): with
`algorithm.rollout_correction.bypass_mode=True` (the fully-async default),
`old_log_probs := rollout_log_probs`, so the PPO ratio is
π_current/π_{v_i} — the per-token cross-version importance ratio — and its
clipping IS truncated importance sampling. The correction exists; it is
version-blind.

What `staleness_correction.py` adds (grounded in the staleness-aware
training literature — gap-aware gradient-staleness mitigation, SAPipe-style
staleness-aware reweighting, TIS/V-trace variance control):

1. **Staleness reweighting** (`async_training.staleness_correction`, default
   in the example config): per-trajectory weight `w(age)=1/(1+λ·age)` or
   `exp(-λ·age)`, self-normalized to mean 1. Reweights *representation* —
   how much each version cohort contributes to the update. Composes with,
   never double-counts, the π-ratio IS inside the clip (it is a function of
   version distance, not of the ratio).
2. **Adaptive clipping** (default OFF): per-trajectory clip scale
   `ε_i = ε·clamp(1+γ·age, min, max)`; γ>0 widens the trust region with age
   (counteracts clip saturation shrinking stale gradients), γ<0 tightens it.
3. **Version-cohort baselines** (default OFF, experimental): mixed-version
   groups (version_span>0) normalize advantages within same-version cohorts.
   Any baseline keeps the IS-weighted estimator unbiased; a cohort baseline
   removes between-version reward drift from the control variate at the cost
   of the between-cohort signal. Empirical tradeoff, hence opt-in.
4. **Cohort diagnostics**: `trajectory_async/stale_*` metrics (age mean/max,
   cohort count, fresh fraction, weight spread, clip scale).

Consumer-side `max_staleness_drop` (hard truncation in the collector) and
this loss-side layer are complementary: the former bounds the worst-case
off-policy distance, the latter reweights what is admitted.

## Relation to the Laminar paper (alignment status)

| Paper element | Status |
|---|---|
| §3 trajectory-level asynchrony, no lockstep; emergent per-trajectory staleness (§6, no static k) | ✅ collector + trainer (MessageQueue path); ✅ upstream v1 (TransferQueue) |
| §3.1 experience buffer (sampling/eviction) | ✅ upstream v1 `ReplayBuffer`; ⚠️ here: collector is FIFO + staleness refusal only |
| §3.1 prompt pool | ✅ upstream v1 (streaming dataloader + refill) |
| §3.1 partial response pool (fault-tolerance substrate) | ✅ substrate + retry consumer (`partial_pool.py`); token-level writer = documented external seam (TODO-16 tail) |
| §3.2 workflow steps ④-⑦ (interleaved train/publish/background distribute/anytime pull) | ✅ P0 wiring (`relay_controller.py` + `rollout_producer.py` batch-boundary pulls); chain distribution = TODO-12 |
| §3.3 + §4.3 fault tolerance (heartbeat failover, chain rebuild, master failover, checkpoint recovery) | ✅ LANDED (`fault_tolerance.py`: supervisor failover + `recover`, probe/retire/revive; `rebuild_chain` pure core); engine-side RDMA re-registration = TODO-15 tail; checkpoint recovery = standard path + `recover` (no separate async checkpoint by design) |
| §4.2 relay hierarchy: master + per-machine relays, resharding, chain-pipelined broadcast, PCIe local pull | ⚠️ flat path via `relay_controller.py` (versioned stage + fleet pull + PER-REPLICA subgroup pulls, live); per-machine chain tier = `relay_tier.py` (CPU-verified) = TODO-12 |
| §4.2 actor stall = single push to master | ✅ `publish` returns after the master stage |
| §5 repack: triggers, version grouping, KVCache idleness, Algorithm 1 Best-Fit + CanFit(`C_max` ∧ `B`), freed sources pull fresh weights | ⚠️ closed loop LIVE (`repack_bridge.py`: post-publish + periodic triggers; idle replicas refresh to fresh versions per-replica); algorithm + executor CPU-verified; request migration + KV introspection = TODO-14 |
| §8 convergence / off-policy stability under staleness | ⚠️ paper itself derives no bound (App. D = broadcast latency; App. C lists IS-based experience sampling as future work); our mitigation = bounded staleness (collector) + loss-side version-staleness correction (`staleness_correction.py`) |

## Test coverage

`tests/experimental/trajectory_async/` (217 tests; 201 stdlib-only +
3 torch-gated adapter smokes + 2 torch-gated staleness batch-application
smokes + 11 ray-gated wiring smokes, all skipping gracefully without
their deps):

* aggregator: completion order, duplicates, FAILED-eviction protocol,
  late-arrival guard, buffer limits, record invariants;
* relay controller per-replica path: partition derivation (contiguous
  blocks, workerless replicas skipped, single-replica degenerate),
  `pull_replica` version tracking / idempotence / error paths,
  concurrent per-replica pulls (lock overlap asserted), fleet pull
  fan-out, staged-bytes quota (rolling retirement, never the latest);
* repack bridge: idle-replica refresh (lagging pulls latest, busy/fresh/
  unknown skipped, one failure never stops the rest), migration
  declining (never half-migrates), manager hook + notify-update wakeup,
  conservative no-mapping wiring;
* partial response pool: version-gated reads (same-version reuse,
  mismatch refused-and-dropped, complete entries terminal), TTL expiry,
  LRU/byte-quota eviction (never the just-put), no version regression
  on overwrite, discard lifecycle, snapshot metrics; retry-consumer
  hook (hints flow to retries only, first attempts clean, pool errors
  never break retries);
* fault tolerance: health monitor (threshold strikes, success-resets,
  retire dedup, revive), supervisor (heartbeat failover recovers +
  re-attaches consumers, failed resurrection survived and counted,
  probe retire/revive lifecycle, retire fires once), chain rebuild
  (identity, middle splice, multiple deaths, all-dead empty, unknown
  deads ignored), controller recover semantics, bridge retire/revive
  (no refresh, non-routable snapshots, drain release, migrate-pair
  decline, idempotence);
* repack drain escalation: soft-drain deadline (abort past deadline,
  once per source; no deadline keeps static semantics; inside the
  window natural completion wins; abort failure degrades to soft);
* repack drain lifecycle: soft plan execution (drain steering, deferred
  emptiness), hard mode (abort counts, drain-before-abort ordering,
  crash-degrades-to-soft), execution-time CanFit rejections, no-drain
  and no-LB-socket declines, completion watcher (pull + end_drain,
  busy sources wait, hard-mode resume ordering, no double-refresh,
  failure retried), real snapshots (inflight/KV-linear/pulling/routable,
  kv_prev decline tracking, fleet util from cache, no-batch-bound
  degradation), end-to-end planning over real snapshots, config
  passthrough;
* long-tail mitigation: row-retry policy (success / retry-then-success /
  budget exhaustion delivering the FAILED sentinel / single-shot legacy
  semantics / delivery-error propagation / per-attempt version stamping),
  survivor delivery (threshold satisfied, below threshold, clamp to 2,
  strict default, late-sibling drops, snapshot counters), group deadline
  (partial resolve, evict without a survivors policy, no premature fire,
  late stragglers counted), collector end-to-end (partial group trains,
  reconciliation identity holds, attempts threading);
* staleness correction: weight families (decay/exp/none), normalization
  invariants, adaptive clip bounds + caps, cohort baselines (mixed-version
  groups, singleton fallback, degenerate rewards), diagnostics, config
  parsing, torch batch application (weight column masking/normalization,
  clip column, cohort advantage rewrite);
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
