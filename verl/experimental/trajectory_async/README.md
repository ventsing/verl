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

## Mathematical reward and longer thinking

The [reward stability guide](REWARD_STABILITY.md) records the 42-update
diagnosis, final-answer-only mathematical verifier, audited reference repair,
ready-group mixing, raw/censored reward diagnostics and the 32K reasoning preset.
It includes the measured 48K backward OOM and separates offline regrading from
new-policy learning evidence.

## Positioning: what already exists in verl

| Existing | Streaming unit | Where groups are formed |
|---|---|---|
| `trainer.v1.trainer_mode=separate_async/colocate_async` | prompt group (`ReplayBufferAsync` refills per group) | rollout side |
| `verl.experimental.fully_async_policy` | prompt group — "sample" = 1 prompt × `rollout.n` rows, delivered once all `n` responses settled | rollout side (`RolloutSample.full_batch`) |
| **this package** | **single response** — each of the `n` responses crosses the queue the moment it finishes | **trainer side** (`GroupAggregator`) |

`fully_async_policy` already streams sample-by-sample, so the long tail
*across* prompts is largely hidden. What remains is the long tail *within* a
group: a group is delivered when its slowest of `n` responses finishes, and
a terminally-failed response traditionally costs its whole group. This
package shrinks the delivery/retry unit from `n` responses to 1.

## Architecture

The current v1 Laminar path uses TransferQueue for trajectory delivery:

```text
AgentLoopWorkerTQ
  generation → reward → per-response TransferQueue write
                                │
                                ▼
TrajectoryReplayBuffer → GroupAggregator → complete-group training batch
                                │
                                ▼
PPOTrainerLaminar → actor update → CPU/Mooncake relay publication
                                │
                                ▼
LaminarController → independent rollout-replica weight pulls and repacking
```

`agent_loop_tq.py` is the producer; TransferQueue provides storage and delivery,
while `tq_sampler.py` selects complete groups on the trainer side. The separate
MessageQueue integration uses `TrajectoryLevelRollouter` and
`TrajectoryAsyncTrainer` with `TrajectoryBatchCollector`. The retired in-process
CPU demonstration trainer is not part of either supported execution path.


Components (all under `verl/experimental/trajectory_async/`):

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

## Real Qwen3 run through TransferQueue

The launchers are grouped under
[`examples/trajectory_async/`](../../../examples/trajectory_async/README.md).
The commands below run from the repository root.

For the local Qwen3-4B checkpoint on a single 16-NPU Ascend machine:

```bash
bash examples/trajectory_async/run_qwen3_4b_16npu_trajectory_async.sh
# Or select GSPO:
ALGORITHM=gspo bash examples/trajectory_async/run_qwen3_4b_16npu_trajectory_async.sh
```

The launcher reuses the installed Ascend environment and the GSM8K preparation
in `examples/trajectory_async/run_qwen3_4b_16npu.sh`. Defaults are 8 trainer NPUs, 8 standalone rollout
NPUs, TP=2, four responses per prompt, and two training steps. It uses V1's
`separate_async` trainer with `TrajectoryReplayBuffer` from `tq_sampler.py`.
`--check-config` resolves the configuration without starting workers.

Start from a shell with the CANN environment configured. The launcher forwards
`PYTHONPATH`, `LD_LIBRARY_PATH`, toolkit paths, and the jemalloc `LD_PRELOAD` into
Ray's runtime environment, including when attaching to an existing cluster.
On this machine, missing jemalloc aborts vLLM model inspection; missing CANN's
Python path prevents rollout workers from importing `acl`.
The launcher sets `additional_config.weight_nz_mode=0` for vLLM-Ascend's RL
sleep/wake path. Its weight-transfer bucket defaults to 1024 MiB, large enough
for Qwen3-4B's approximately 742 MiB embedding tensor; the preflight checks this
against safetensors headers before starting workers. Override the bucket with
`UPDATE_WEIGHTS_BUCKET_MB` when using another model.

`AgentLoopWorkerTQ` writes each scored response independently to TransferQueue;
the custom sampler observes those arrivals while sibling requests are still
running and feeds their metadata to the real `GroupAggregator`. Tensor payloads
stay in TransferQueue. GRPO/GSPO updates consume complete groups, ordered by
their observed completion time. A terminal prompt marker is required before
retiring the group, preventing producer writes after its metadata is cleared.
Partial groups are never padded into training batches. Failed/stale training
groups use the existing async sampler's eviction and replacement policy;
validation failures raise an error.

This path supports single-output agent sessions. It uses the normal checkpoint
manager for weight updates (`nccl` selects the HCCL implementation on Ascend).
It does not enable the multi-version P2P weight store or active repacking from
the separate MessageQueue integration. Metrics under `trajectory/train/aggregator/*` report received
trajectories, partial groups, and completed groups.

### Laminar semantics and the current real-engine gap

The reference is [Laminar: A Scalable Asynchronous RL Post-Training
Framework](https://arxiv.org/html/2510.12633v1), particularly §§3–6.
The table below describes the original `separate_async` Ascend launchers
(including `examples/trajectory_async/run_qwen35_9b_16npu_trajectory_async_24k.sh`). The new `laminar`
entry point in the next section replaces this execution path; the CPU demo
has been retired.

| Requirement | Current real Ascend path |
|---|---|
| Consume completed experiences once a training batch is available, independently of other in-flight generation (§3.2). | Implemented by `TrajectoryReplayBuffer`, subject to complete GRPO/GSPO groups and the configured batch size. |
| Publish weights to a CPU relay; each rollout pulls independently at its own batch boundary or after repack releases it (§4). | Not connected. V1 `separate_async` still uses the shared checkpoint manager to pause generation and synchronize weights. |
| Keep one policy version throughout each trajectory, including after migration (§§2.4, 5.1). | Not guaranteed. `FullyAsyncLLMServerClient` can resume an aborted response after a global weight update, potentially mixing policy versions. |
| Consolidate running long-tail trajectories onto fewer replicas of the same version, freeing sources for new weights and prompts (§5). | Only modeled by `MultiReplicaEngine` and `RepackManager`; no real migration adapter is connected. |
| Detect ramp-down from KV usage, and respect both KV capacity and the decode batch bound when packing (§5.2). | The old CPU demonstration is retired; use the current Laminar integration and its hardware validation below. |

Repack is consolidation of **in-progress** work, rather than redistribution
of pending prompts. Sources stop admitting fresh work, unfinished trajectories
move to compatible destinations, and released sources pull the latest weights
before admitting a new batch. The paper triggers this periodically (for
example every five seconds) and immediately after a training weight update.
It does not require a global rollout drain before training can proceed.

For the current 16-NPU placement, the intended independently managed rollout
units are the four standalone TP=2 replicas on eight NPUs; the other eight
NPUs train with FSDP2. A real integration must replace the shared weight-update
barrier with CPU snapshot publication and per-replica loading, connect backend
progress/capacity telemetry, and implement running-request migration while
preserving token IDs, behavior log probabilities, remaining token budget,
request identity, and policy version. Qwen3.5's recurrent attention state must
also be considered: a PD-transfer API alone does not establish support for
migrating arbitrary running requests. Abort followed by re-prefill must be
reported as such, with its cost measured.

Acceptance on real hardware requires event traces demonstrating an actor
update while an earlier long trajectory remains active, one replica updating
without pausing the others, and a same-version migration that releases a
source without losing or duplicating a response. Record per-request policy
versions, migration events, per-replica loads, and training/publish/pull times;
CPU timing-mock results cannot establish these properties. These original
launchers provide streaming group consumption with partial rollout, not the
complete Laminar execution model.

```bash
python3 -m pytest -q tests/trainer/ppo/v1/test_trajectory_sampler_on_cpu.py
```

### Single-node Laminar controller with real servers

The paper comparison, remaining implementation work and acceptance criteria are
tracked in [Laminar development gaps](LAMINAR_GAPS.md).

The `mooncake_relay` backend adds node-affine CPU relays and native Mooncake
chunk-pipelined propagation. See [Mooncake relay deployment and validation](MOONCAKE_RELAY.md)
for the multi-node entry point, isolated CPU dependency setup and tested limits.
The Mooncake backend passed a real three-process CPU/TCP chain probe and a
16-NPU two-update smoke run (64 training and four validation responses; exit 0,
trace audit consistent). The final V1/decoder suite passed 233 tests, with two
skipped. Only one physical node was available; multi-node/RDMA and scaling
remain unverified. The short 256-token run is an execution test, not a learning
quality or 24K throughput result.

The dedicated entry point keeps eight FSDP2 trainer NPUs and four standalone
TP=2 rollout replicas on the other eight NPUs. No vLLM engines are launched on
trainer cards; validation uses the independent rollout pool. It uses Qwen3.5-9B, DeepScaleR, thinking,
a 2048-token prompt limit and a 24576-token response budget:

```bash
bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_24k.sh --check-config
bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_24k.sh
# Optional sequence-level GSPO loss:
ALGORITHM=gspo bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_24k.sh
```

For a run managed by the existing Ray Jobs service, independent of the launching
terminal, use the same CANN-enabled Python environment:

```bash
python3 submit_qwen35_laminar.py --smoke
# Full 24576-token run; submit after the smoke task has finished:
python3 submit_qwen35_laminar.py
python3 monitor_qwen35_longtail.py
# Include this job's current backend generation lengths:
python3 monitor_qwen35_longtail.py --live
```

The submission helper forwards the notebook's CANN paths and selected launch
settings to the job entrypoint. It restores the launching shell's NPU visibility
for the CPU driver's device-count preflight; Ray still assigns NPU visibility
to training and rollout workers. It does not restart Ray or install dependencies.
Use `--dashboard URL` for another existing dashboard and `--` before Hydra
overrides. Environment variables such as `ALGORITHM=gspo` still apply; `--smoke`
explicitly selects 256 tokens, two updates, 64 training prompts and four
validation prompts. The run state records the job ID and dashboard; the monitor
queries its status, exit code and failure message directly without routing the
local dashboard through the external notebook proxy. `driver_alive` remains a
separate PID check. Submission metadata is also saved under the output directory.
`completed_steps` and `training_updates` are parsed from completed training
metric records, independently of validation exports. `rollout_files` lists the
exported JSONL filenames separately. Update records include gradient norm,
advantages, peak actor memory and generation/reference/update/publication
latencies. A zero-gradient cache diagnostic is therefore visible directly.
The monitor regression suite passed six tests, including duplicate console
records, incomplete steps and malformed/nonfinite timing values.
The optional `--live` query selects only named vLLM actors belonging to the
recorded Ray job. It reports policy versions, KV usage, running/waiting counts
and `segment_generated_tokens`, even before long responses finish. These token
counts cover the current backend segment; after a migration they exclude the
prefix replayed from the source. Completed full-response lengths remain in the
`train`/`val` summaries. `--ray-address` selects the cluster used for actor RPCs
(default `auto`, the local single-node cluster).

An independent trace auditor checks the scheduler's per-replica version history
and joins completed requests to queue arrivals and selections:

```bash
python3 audit_laminar_run.py --output-dir outputs/YOUR_RUN
# For a short 256-token run, add: --response-budget 256
```

It rejects pulls while a request remains active, cross-version migration,
changed queue versions/token counts/migration histories, missing lineage,
duplicate selections, future-policy training samples and stale validation
samples. Queue logs are read before the scheduler snapshot to support live
auditing. `consistent: true` means the observed records are consistent; it does
not mean an in-progress job has finished. Requests without terminal records
may still be running or may have been cancelled at shutdown. This audit reads
metadata rather than the actual logprob tensors; per-segment token/logprob
length and finiteness checks remain in the controller, and exact old-logprob
reuse is covered by the trainer tests.

For local diagnostic backups of completed experiences, including original
`rollout_log_probs`, use:

```bash
python3 snapshot_laminar_experiences.py          # One read-only pass
python3 -u snapshot_laminar_experiences.py --watch
```

The helper verifies that the queue controller belongs to the recorded running
job, constructs an independent `TransferQueueClient` from its connection
configuration, and only uses metadata/data reads. It never enters the global
`tq.init()`/`tq.close()` lifecycle or clears queue entries. Closing it only
releases its own client connections and Ray driver. Completed rows are written
under `outputs/RUN/experiences/{train,val}/KEY.pt`, after checking exact response
and behavior-logprob lengths, finite logprobs and a single policy version.
Writes are flushed/fsynced and renamed from `.partial`; existing completed
files are skipped. Each file contains the original tag, full CPU TensorDict
and source run/model metadata. These are experience backups for diagnosis;
they do not include actor/optimizer weights, recover unfinished generation or
implement automatic trainer resume. Read `--watch` progress to see how many
rows were actually saved; rows retired before a snapshot are not guaranteed
to be captured.

`trainer.v1.trainer_mode=laminar` selects `PPOTrainerLaminar`. Configuration
validation rejects unsupported deployments instead of silently falling back
to global synchronization. The original `separate_async` entry points remain
available for comparison.

The real-server path is composed of:

| Component | Responsibility |
|---|---|
| `cpu_relay.py` | Publish immutable CPU snapshots into Ray shared memory; atomically expose the new version; retain manifests while consumers hold read leases. All FSDP ranks exhaust the parameter iterator, but only rank zero publishes. |
| `laminar_controller.py` | Own per-replica batch admission, independent drain/pull boundaries, version checks and running-request repacking. |
| `vllm_telemetry.py` | Collect actual vLLM scheduler KV usage and running/waiting counts. |
| `trainer_laminar.py` | Publish after an actor update without invoking `CheckpointEngineManager.update_weights`; consume complete groups through the existing TQ sampler. |
| `prompt_feeder.py` | Replenish a bounded prompt/experience backlog independently while the actor trains; propagate producer errors to the sampler. |

After a replica accepts `trainer.v1.laminar.batch_per_replica` requests (default
8), it closes admission for that batch. Newly published weights also close
old-version replicas to fresh admission. Once its active requests and migration
reservations both reach zero, the replica independently loads the latest
committed CPU manifest and reopens. Other replicas continue generating with
their own policy versions throughout that load. Publication waits for the CPU
snapshot, not for rollout consumption. Validation requests require the current
published version; older training trajectories continue on other replicas.
The 16-NPU launcher's `max_num_seqs=8` is a validation concurrency limit, not
a measured Ascend roofline optimum. The paper's packing bound `B` is chosen
from its hardware's memory/compute transition. Here the controller conservatively
uses the engine's configured request capacity as that bound. For example, two
replicas with six and five remaining requests cannot be consolidated into one
eight-request engine even if their combined KV usage is small. Engine capacity
and admission batch quota are separate settings; higher concurrency requires
an actual throughput/latency measurement on the target NPU configuration.
Relay consumers keep tensors on CPU until the vLLM transfer adapter consumes
them. This avoids a redundant CPU-to-NPU-to-CPU staging round trip when the
Ascend adapter uses shared memory instead of device IPC.

The trainer starts a background prompt producer after warmup. Its target,
`trainer.v1.laminar.prompt_buffer_groups` (default 64), counts pending, running
and completed-but-unsampled prompt groups. Completed groups therefore apply
backpressure instead of allowing unlimited experience accumulation. Dataset
iteration and checkpoint state reads share a lock. The launcher uses
`data.dataloader_num_workers=0` and `data.gen_batch_size=1` so a producer thread
can cycle the dataset and replace failed groups without forking worker
processes. Producer failures interrupt sampler polling instead of leaving
training waiting indefinitely. Production stops at the last update; shutdown
cancels pending agent sessions and controller requests without inventing empty
completed trajectories.

Repack checks run every `trainer.v1.laminar.repack_interval_s` (default 5 s)
and after publication. Only closed, nonempty replicas with an observed KV ramp-down,
no backend waiting requests, and fewer than the configured decode batch bound
are candidates. Once a KV decrease establishes that drain phase, the real
controller remembers it until the next batch boundary; new waiting requests
or high KV pressure clear it. This adaptation lets replicas whose completions
fall in different polling windows consolidate, rather than requiring their
KV drops to coincide in one check. Packing operates within one policy-version group, applies
projected KV and request limits, reserves destination capacity before cancelling
sources, and avoids transfer chains in one plan. A source can pull new weights
only after its last request has left. A request that finishes during migration
releases its reservation and produces exactly one final response.

**Behavior policy and old log probabilities.** This entry point enables
`rollout.calculate_log_probs=True` and
`algorithm.rollout_correction.bypass_mode=True`. Each response token keeps the
vLLM log probability from its actual generating policy; those values become
`old_log_probs` without an actor recomputation. Every trajectory must have
`min_global_steps == max_global_steps`; a version mismatch, missing logprob,
non-finite value, or token/logprob length mismatch raises an error. Different
trajectories in the same training batch may have different policy versions.
The reference model still computes `ref_log_prob` when KL loss is enabled.
GRPO and GSPO keep their respective configured policy-loss functions.

For probability alignment with the training forward, sampling uses
`processed_logprobs`, `top_p=1`, `top_k=-1`, and repetition penalty 1. Temperature
remains configurable and is applied by both inference and training. The mode
rejects top-p/top-k truncation or a repetition penalty because the current actor
forward does not reproduce those transforms. `max_off_policy_strategy=none`
disables both fixed-age eviction and the stale-inflight waiting barrier;
failure handling and complete-group requirements still apply.

**Transport and remaining scope.** CPU relay transport uses the single-node Ray
object store; it does not implement the paper's multi-node UCX/RDMA relay chain.
Running-request migration currently uses **abort and re-prefill under the same
policy version**. Completed token/logprob segments and the remaining generation
budget are preserved; the destination rebuilds KV and Qwen3.5 recurrent state
from the prefix. This can cost substantial compute for 24K trajectories and
must not be reported as a direct KV/SSM transfer. Multi-node relay operation is
available through `mooncake_relay` (see the deployment guide); this original
`cpu_relay` transport remains single-node. PD, engine DP/PP greater than one,
multimodal migration, durable partial-response recovery and native KV/SSM
migration remain outside these paths.

Inspect the run through the existing state-file monitor:

```bash
python3 monitor_qwen35_longtail.py
```

`outputs/RUN_NAME/trajectories/scheduler.jsonl` records `published`, `pull_begin`,
`pull_end`, `request_begin`, `migration`, `request_end`, and controller failures.
Completed TQ rows carry `trajectory_request_id` and `migrations`; arrival logs
include `scheduler_request_id` so selected training responses can be joined
to the scheduling trace without matching by token length.
Periodic `replica_snapshot` events retain actual KV utilization, prior usage,
per-replica versions, active/reserved counts and backend waiting counts so a
missing repack can be diagnosed without guessing. The monitor summarizes
publication versions, event counts and pulls made while other requests remain
active; `first_errors` retains the original exception when shutdown errors
would otherwise obscure it at the end of the log.
`pull_begin.other_active` shows whether other replicas were still generating;
`migration.transport=recompute` and `prefix_tokens` expose cache rebuild work.
Training metrics include `laminar/publish_seconds`, `laminar/publish_bytes`,
pull/migration counts and `laminar/old_log_probs_from_rollout`.

The launcher enables `trainer.v1.laminar.run_migration_probe=True`. Before
training admission, it runs two 256-token text requests with EOS ignored,
waits for at least 16 generated tokens, then performs a capacity-checked
same-version migration. Both outputs must retain 256 tokens and 256 finite
behavior logprobs, and exactly one response must report a migration. These
requests are marked `is_probe` in scheduler events, bypass reward/TQ training,
and use separate preflight counters; they do not count as training migration
evidence. Disable this preflight explicitly with
`trainer.v1.laminar.run_migration_probe=False` if needed. It tests backend
continuation, while automatic repack still requires the normal KV ramp-down
conditions. `continuation_segment.prefill_seconds` records the backend's
measured prefill duration when available.

Validation commands:

```bash
python3 -m pytest -q -o cache_dir=/tmp/laminar_pytest_cache \
    tests/trainer/ppo/v1/test_laminar_on_cpu.py \
    tests/trainer/ppo/v1/test_trajectory_sampler_on_cpu.py
python3 -u probe_laminar_relay.py
```

The CPU tests cover independent pulls during active generation, same-version
migration, token/logprob/budget preservation, completion/cancel races, packing
capacity, read leases, no fixed-staleness barrier and exact old-logprob reuse.
The relay probe uses separate actors on the existing Ray cluster and checks
immutable old/new snapshots and scalar/BF16 round trips. These checks validate
control-plane and shared-memory behavior; NPU execution, full-model loading,
training overlap and actual migration cost require real-run traces.

The 2026-10-16 24K run in
`outputs/qwen35_9b_grpo_laminar_thinking24k_20261016_193957` completed all four
initial TP=2 weight loads in 11.9–13.4 seconds. It observed 34 trajectories,
including 18 at the 24576-token cap, and selected 32 responses for training.
It completed **zero actor updates**: an inherited `on_sample_end` accessed
`_step_sample_wait_seconds` before the Laminar step hook had initialized it.
The Laminar step hook now preserves the parent hook's accounting initialization
while remaining in fixed TRAINER mode; a regression test exercises the actual
step/sample hook sequence. Later `Laminar controller stopped` messages from
that run are shutdown effects, not the initial cause.

To verify the update pipeline before waiting for another full-length batch:

```bash
MAX_RESPONSE_LENGTH=256 TOTAL_STEPS=2 TRAIN_SAMPLES=64 VAL_SAMPLES=4 \
    bash examples/trajectory_async/run_qwen35_9b_16npu_laminar_24k.sh trainer.test_freq=2
python3 monitor_qwen35_longtail.py --response-budget 256
```

This is a 16-NPU/full-model smoke test, not evidence of long-tail training
quality. Restore the default 24576-token budget for the long-tail run.
The launcher records its actual response budget in the run state; the explicit
monitor override also supports older state files.

The full-model 16-NPU smoke run
`outputs/qwen35_9b_laminar_smoke256_20261019_0116` exited with code 0: two actor
updates completed, 64 responses were selected for training, four validation
responses completed, and versions 0, 1, 2 were published. It preceded the
continuous prompt producer and real-backend migration preflight. It verifies
the update/logprob/pull path, not automatic long-tail repacking or learning
quality. The shared V1 trainer tests plus the Qwen3.5 decoder compatibility
tests passed with 191 passed and 2 skipped on 2026-10-19.

The later run `outputs/qwen35_9b_laminar_stream_smoke256_20261019_014640`
passed the real-NPU migration preflight: two 256-token responses, one migration
from replica 0 to replica 1 at policy version 0, and a 40-token total prefix
(prompt plus generated tokens).
The preflight took 16.0 seconds; the continuation segment reported 0.145 seconds
to its first token. This is one measured portable re-prefill, not a native
KV/SSM transfer benchmark. The run recorded update 1 and published version 2
while 32 trajectories remained active, but did not finish validation or dump
update 2. Ray recorded the driver's shutdown and subsequent worker cleanup;
the precise trigger is unconfirmed. Its partial trace must not be counted as
a successful two-update run. New runs can use Ray Jobs to retain an independent
authoritative task status and exit code.

The Ray-managed run
`outputs/qwen35_9b_laminar_ray_smoke256_20261019_021001` subsequently finished
with Ray Job status `SUCCEEDED` and driver exit code **0**. It includes the
continuous prompt producer, CPU-only relay consumption and real migration
preflight. Both updates and four validation responses were dumped; 64 training
responses were selected. The trace recorded versions 0, 1, 2, eight independent
pulls while other replicas still had active requests, and one **automatic
training** migration in addition to the separate preflight migration. The
automatic migration preserved version 0 and completed its last three tokens
after rebuilding the prefix on replica 2. This verifies the migration and
shutdown paths but does not measure long-prefix migration performance or
learning quality: all training responses hit this smoke test's 256-token cap.
Native CANN `TBE Subprocess ... main process disappeared` cleanup messages were
printed during worker teardown despite the successful exit code.

The subsequent 24K run is
`outputs/qwen35_9b_laminar_ray_thinking24k_20261019_021847`, submitted through
the same Ray Jobs service with DeepScaleR, thinking enabled, a 24576-token
response budget, four planned updates and eight validation prompts. Its real
migration preflight passed before production generation began. At the
2026-10-19 02:32 UTC observation, eight training responses had completed
(1593–6675 tokens) and 24 remained in flight at roughly 6.6K–6.8K tokens. No
training update had completed yet. This is a timestamped progress observation,
not a claim that 24K actor training or the full job has passed; use the monitor
for current results.

At 02:33 UTC the same 24K run had completed 12 responses (1593–7779 tokens)
and automatically moved four running training trajectories from replica 3 to
replica 2. Their total replayed prefixes were 7524–7556 tokens; all remained
on version 0. Replica 3 then admitted eight new requests while the migrated
and other old requests continued. This is real long-prefix repacking and
replica reuse before the first training update. Continuation timing is written
when the corresponding backend segment completes, so those four migration
costs were not yet available at this observation.

The first of those migrated trajectories subsequently completed on version 0.
Its continuation had 5599 tokens and reported 0.435 seconds to its first token
after replaying a 7525-token total prefix. At 02:41 UTC, the run had observed
19 completed training responses, up to 14540 tokens, and two complete groups;
the first update still awaited the configured eight complete groups. The
independent auditor passed both this partial run and the completed Ray-managed
256-token run, including the latter's 64 training selections and four fresh
validation selections.

That 24K run later exited before its first update at 02:42 UTC. The immediate
error was `Finished prompts lack 4 trajectories` for a hexadecimal test UID
outside the training data. Concurrent CPU integration tests had called
`tq.init()` on the training cluster. The installed TransferQueue uses the fixed
actor `TransferQueueController` in namespace `transfer_queue`; per-test data
partitions did not isolate its controller or `close()` lifecycle. The trajectory
sampler had also applied its rollout group size to unrelated test partitions.
This was test interference, not a missing response in one of the recorded
training groups.

The two CPU integration-test modules now depend on an isolated local Ray
fixture, with a separate short `/tmp` root, 16 CPU resources, zero NPU/GPU
resources and teardown limited to that instance. The existing training-cluster
discovery files remain unchanged. The trajectory sampler only ingests the V1
`train`/`val` partitions; a regression verifies that a foreign one-response
finished group cannot break a valid training group. The missing-response check
still rejects malformed groups within a training partition. After this fix,
the complete V1/Qwen3.5 CPU suite passed **201 tests, with 2 skipped**. Other
test suites calling unisolated `tq.init()`/`tq.close()` must not share an active
training cluster.

### 16-NPU long-sequence validation and scaling baseline

Keep the current correctness baseline at 16 NPUs: eight FSDP2 trainer ranks
and four TP=2 rollout replicas on eight independent inference NPUs. First
complete fresh 24K training updates, asynchronous publication/pulls, validation
and migration with trace auditing. Only then compare 32/64-NPU deployments.
Preserve model, dataset, thinking/sampling settings and length distribution;
state the trainer/rollout allocation and global batch for each comparison.
Report trained tokens per wall-clock second, update latency, NPU/KV utilization,
publication/pull time, migration/re-prefill overhead and policy staleness.
Cached-experience diagnostics must not be used as throughput or learning results.
The current CPU relay is single-node; multi-node 32/64-card execution still
requires a transport/backend and deployment changes before claiming scaling.

The fresh 24K run `qwen35_9b_laminar_ray_thinking24k_20261019_024948`
observed 40 completed responses (863–24576 tokens, mean 10184.875) and selected
32 for the first update (mean 9705.594). It performed 19 production migrations,
in addition to the preflight probe. Reference evaluation passed, but actor
backward did **not** finish: dense-logit NPU cross entropy requested another
11.41 GiB with only 11.08 GiB free. No actor update completed in that run.
A later shutdown log write also hit the shared filesystem quota. Cleanup now
preserves the original training exception while attempting all shutdown steps.

The Qwen3.5 launcher now uses `model.use_fused_kernels=True` and
`model.fused_kernel_options.impl_backend=torch`: the output projection computes
logprobs/entropy in token chunks instead of materializing whole-sequence logits.
The Torch backend calls the `lm_head` module so FSDP's unshard, CPU-offload and
backward hooks execute. Directly reading an offloaded head's weight had failed
the first replay attempt with CPU/NPU device mismatch. A two-NPU head probe
passed value and gradient comparisons with CPU offload both enabled and disabled:

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 probe_qwen35_fsdp_head.py
python3 -m pytest -q tests/models/test_qwen35_fused_head_on_cpu.py
```

A separate output-head-only 24576-token forward/backward probe, with the real
4096 hidden width and 248320 vocabulary size, completed with finite values and
gradients at 9.88 GiB peak allocation. This alone does not validate whole-model
24K training. The earlier complete CPU suite passed 210 tests with 2 skipped;
the additional fused-head regressions are recorded separately.

Shared storage returned `EDQUOT` during experience backup and source writes.
The affected launcher/config files were restored from local backups and
validated. Subsequent source writes use same-directory temporary files,
fsync, byte verification and atomic replacement. Two failed run directories
were archived under `/tmp/verl-laminar-failed-runs` with their original output
paths retained as symlinks. New runtime artifacts can use local storage:

```bash
python3 submit_qwen35_laminar.py --output-root /tmp/verl-laminar-runs
# A separate read-only sidecar can also select a local backup destination:
python3 snapshot_laminar_experiences.py --watch --backup-dir /tmp/verl-experiences
```

For a memory/correctness diagnostic only, `initial_experience_dir` imports
complete initial-policy groups from these trusted local `.pt` backups. It
requires the same resolved model directory, unchanged checkpoint files, original
version-zero tags, finite behavior logprobs and exact response/logprob lengths.
It rejects resume after actor updates. Original `rollout_log_probs` are retained,
never recomputed from a newer actor. Groups are selected longest-first to stress
memory; incomplete groups are omitted. The auditor links each `request_import`
to the original run's completed request. This is not general checkpoint resume:
actor/optimizer state and unfinished generations are not restored.

```bash
# One initial-checkpoint update from 4 groups × 4 responses; no validation.
env TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 TOTAL_STEPS=1 VAL_SAMPLES=4 \
  python3 submit_qwen35_laminar.py --output-root /tmp/verl-laminar-runs -- \
  trainer.v1.laminar.initial_experience_dir=/tmp/YOUR_BACKUP/experiences \
  trainer.test_freq=-1
```

The diagnostic run `qwen35_9b_laminar_replay24k_20261019_fsdpfix`
finished **SUCCEEDED, exit 0**, with an independently consistent trace and
published versions 0/1. Its imported batch contains 16 responses
(863–24101 tokens, mean 12354.125). Reference evaluation took 23.48 s,
actor forward/backward 129.43 s, and CPU publication 6.04 s; reported peak
actor allocation was 26.14 GiB. All four groups had uniform reward 1, so
advantages, loss and gradient norm were zero: this verifies the memory and
execution path, not a nonzero policy update or learning progress. The two-NPU
head probe separately checked nonzero gradients against an unsharded baseline.
The additional fused-head and Ulysses routing suite passed 27 tests.

The subsequent fresh run
`qwen35_9b_laminar_ray_thinking24k_20261019_060254` uses the default
32-response update batch, four updates and eight validation prompts, without
importing cached experiences. Its first fresh-data update completed with
loss 0.0518833 and gradient norm 0.138672; rewards averaged 0.8125 and group
advantages included both signs. Selected responses spanned 901–24576 tokens
(mean 8949.438). Actor peak allocation was 30.07 GiB; reference evaluation
took 31.15 s, actor update 204.38 s and CPU publication 6.12 s. The original
vLLM behavior logprobs were used (`old_log_probs_from_rollout=1`).

Version 1 was published with 28 generation requests still active. Replica 3
then loaded version 1 in 9.63 s while 18 requests remained active on other
replicas; replica 2 followed in 10.89 s with 24 other requests active. A later
snapshot showed replicas 0/1 still finishing version-zero work and replicas
2/3 generating under version 1. The independent trace audit passed with
19 production migrations and 32 selected training trajectories at that point.
This validates a nonzero long-sequence update, asynchronous publication and
independent pulls while older trajectories continue.

The complete run subsequently finished **SUCCEEDED, exit 0**, in 128.66 minutes,
including initialization and final validation. All four updates had nonzero
gradients (0.1387, 0.0986, 0.1855, 0.3242), with no training OOM. Peak actor
allocation was 34.47 GiB and peak reservation 43.65 GiB. Training selected
128 fresh responses (mean 15607.99 tokens), of which 51 reached the 24576-token
budget. The final trace audit passed: published versions 0–4, 43 production
migrations plus one preflight migration, 15 pulls overlapping other active
requests, and no unfinished pulls. All eight validation responses used version 4.
There were 32 requests without terminal records at shutdown; these are not
counted as completed or trained responses.

Final validation accuracy was **1/8 (12.5%)**. Seven of eight validation responses
reached the token budget; the remaining response had 22055 tokens. This is a
successful execution/correctness test, not evidence of model-quality improvement.
No matching initial-model validation baseline was collected, and eight prompts
are too few for a strong quality conclusion. Long, repetitive reasoning and
budget exhaustion remain concerns before extending training or interpreting
scaling results. Post-exit CANN TBE subprocess cleanup messages were present,
while Ray's authoritative job status remained successful with exit code zero.
The full monitor and terminal trace audit are saved in the run's
`final_run_report.json` under `/tmp/verl-laminar-runs`.

### Thinking and natural long tails (24K)

For Qwen3.5-9B, download the official weights from ModelScope using the
ModelArts proxy (the downloader overrides inherited proxies only in its own
requests and subprocesses):

```bash
python3 -u download_qwen35_9b.py
# Use --direct to disable proxies, or --proxy URL to select another proxy.
bash examples/trajectory_async/run_qwen35_9b_16npu_trajectory_async_24k.sh --prepare-only
bash examples/trajectory_async/run_qwen35_9b_16npu_trajectory_async_24k.sh --check-config
bash examples/trajectory_async/run_qwen35_9b_16npu_trajectory_async_24k.sh
```

This configuration enables thinking, uses a 24,576-token **response budget**
plus 2,048 prompt tokens (26,624 total), and replaces GSM8K with DeepScaleR's
AIME/AMC/Omni-MATH/Still problem mix. It does not force a minimum output
length or ignore EOS: the resulting length distribution must be measured.
DeepScaleR is deduplicated by normalized problem hash before a deterministic
256-question held-out split. Source answers and solutions are never included
in prompts. Data resides under `datasets_for_train/deepscaler`; `manifest.json`
records the source hash, split sizes, and boxed-answer MATH scorer.

Defaults are four training steps, eight prompt groups per batch, four responses
per prompt, TP=2, eight concurrent sequences per inference engine, and chunked
prefill. Qwen3.5 follows the local Ascend example's unpacked microbatch path
and uses a 2048 MiB weight-transfer bucket for its larger embedding. The
Ascend launcher explicitly sets both Mamba cache dtypes to BF16: this machine's
`aclnnRecurrentGatedDeltaRule` rejects the model's default FP32 recurrent state.
It also forwards `VLLM_LOGGING_LEVEL=INFO` into Ray workers for initialization logs.
The
smaller Qwen3-4B remains available through
`examples/trajectory_async/run_qwen3_4b_16npu_trajectory_async_24k.sh` with the same data and token budget.
Set `TOTAL_STEPS` for a longer run; the 24K launchers use the entire training
split. Set `QWEN_OUTPUT_DIR` to override the default `outputs/$RUN_NAME` directory.

Every response observed by the sampler is appended once to
`trajectories/train.jsonl`, including arrivals from incomplete groups. Separate
selection events identify responses consumed by training. `observed_at` is the
trainer's polling time, not the exact generation finish time. Selected batches
also report P50/P90/P95/P99 and the fractions reaching 4K/8K/16K/24K under
`trajectory/train/length/*`; responses reaching the budget may be truncated.
Selected prompt/response texts are saved under `rollouts/`.

```bash
python3 summarize_trajectory_lengths.py outputs/RUN_NAME/trajectories/train.jsonl
tail -f outputs/RUN_NAME/train.log
python3 monitor_qwen35_longtail.py
```

The Qwen3.5 launcher records the latest run in `/tmp/qwen35_longtail_run.json`
for the monitor. Set `QWEN_STATE_FILE` to isolate concurrent experiments.
The decoder patch supports both Transformers' newer `block_type` and older
`layer_type` attributes. Before a long sampling run, check the decoder's
outputs/gradients and the full patched model's tiny NPU forward/backward:

```bash
python3 -m pytest -q tests/models/test_qwen35_decoder_compat_on_cpu.py
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 python3 -u probe_qwen35_training.py
```

The NPU probe uses a small randomly initialized model at 128 and 24,576 tokens;
the full 9B FSDP2 training run is still required to validate memory and synchronization.

The summary distinguishes all observed arrivals from consumed responses;
completion-order sampling can favor shorter groups, so the two distributions
must not be conflated. In-flight responses not yet observed are absent from
these statistics.

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

Topology — per-replica subgroups are LANDED but activate only with
**≥2 worker-bearing rollout replicas** (`derive_replica_partition`
returns None below that: a single replica IS the whole fleet, scoped
pulls are meaningless there by design). NOTE the shipped example runs
`rollout.nnodes=1` — one replica — so the per-replica path never
activates in that deployment; the repack bridge's idle refresh probes
once, latches off cleanly (`repack/per_replica_pulls: 0`), and the
producer's fleet pull covers weight updates. Where subgroups ARE
installed: `receive_weights_version(version, replica_id=...)` pulls
over that subgroup alone (H2D bucket partition + barriers touch only
the replica's ranks), and `RelayController.pull_replica` drives it
with per-replica locks — distinct replicas pull CONCURRENTLY; fleet
pulls and publishes take all locks (ordered, deadlock-free — the
concurrent-collective-safety question is TODO-11). The ONLY per-replica
pull caller today is the repack bridge's idle refresh (post-publish +
periodic) — Laminar's anytime pull, realized for idle replicas.

**The known P0-path gap, framed as two breakthrough points** (the
producer's batch-boundary pull is FLEET-SYNCHRONIZED:
`_maybe_pull_weights` → `controller.pull()` aborts in-flight, releases
KV, runs the collective receive, and resumes on ALL replicas per
version bump; on one replica that coincides with per-replica
semantics, at ≥2 it stalls the whole fleet per publish):

1. **Communication-domain split (the prerequisite) — LANDED at the
   manager-level split point**: `CheckpointEngineManager
   .build_process_group(replica_partition=...)` installs one
   `dist.new_group` subgroup per replica (not inside the engine — the
   partition flows manager → topology → `init_process_group`).
   Subgroup composition note: each subgroup contains the replica's
   ROLLOUT ranks only — actor ranks are deliberately NOT in the pull
   group (kimi's P2P store: shards are registered into the store at
   STAGE time; a pull READS the store, so the pull group needs no
   actor membership — the barriers and H2D bucket partition touch
   only the replica's ranks). Without this split, pushing pull
   control down would be "logically async, physically synchronized"
   (a global barrier under every scoped call).
   Elastic safety: the engine's init guard now FAILS LOUD on a
   re-init with a CHANGED partition (silently keeping stale subgroups
   would route per-replica pulls over the wrong topology); a
   same-partition re-init (the failover factory path) is an
   idempotent no-op. True dynamic group rebuild remains a cluster
   TODO (NCCL group-teardown traps).
2. **Pushing pull control down to the replica — the open half.** The
   centralized fallback is RETAINED by design (the producer's fleet
   pull doubles as the emergency/timeout convergence path). What
   exists today at replica granularity: the repack idle refresh —
   per-replica DECISION + per-replica EXECUTION over the subgroups;
   only the observation is centralized polling. The full
   self-triggered form (a server/client-side loop calling
   `relay_controller.pull_replica(self.id)` when the replica's own
   scheduler goes idle or hits an internal drain state) additionally
   needs per-row version attribution from the engine and
   version-aware routing (new work may only go to replicas at/above
   the group's target version) — an LB/manager seam, i.e. the L2
   scheduling stack's territory.

**Verify before refactoring (the 2-replica experiment).** Before any
further push-down work, run the 2-replica experiment — the full
runbook is cluster TODO-6 Tier 2 (launch command, log markers, pass/
fail reading): the controller emits `PULL_REPLICA_BEGIN/END
replica=...` interval logs (per-replica windows spanning the full
abort → release KV → subgroup receive → resume sequence), and
OVERLAPPING windows across replicas = the subgroups are live and
per-replica pulls are truly concurrent, while DISJOINT windows = a
global barrier remains (revisit the split). The per-rank
`receive v{N} done ... time cost` engine lines corroborate.
Chunk-pipelined chain distribution remains
`relay_tier.py`'s CPU-verified model (TODO-12). The mooncake engine now
implements the same versioned method set as kimi (TODO-9 below): its
per-version staging buffers replace the rank CHAIN entirely — pulls are
direct per-consumer `transfer_sync_read`s, so mooncake needs NO
per-replica subgroups (each replica's pull is engine-side disjoint by
construction; `replica_id` is accepted for driver-dispatch parity and
otherwise unused).

Run it (see `examples/trajectory_async/`):

```bash
python -m verl.experimental.trajectory_async.trajectory_async_main \
    async_training.weight_store.backend=kimi \
    rollout.checkpoint_engine.backend=kimi_ckpt_engine \
    [examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh for the full recipe]
```

Both versioned engines work: `weight_store.backend=kimi` pairs with
`checkpoint_engine.backend=kimi_ckpt_engine` (per-version CPU shards +
P2P-store registration, pulls over per-replica subgroups); `mooncake`
pairs with `checkpoint_engine.backend=mooncake` (per-version pinned
host staging buffers + direct `transfer_sync_read` pulls — see item 9
below for what is engine-implemented vs cluster-pending).
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
  boundary; the upgrade path is items 17/18 with recompute as the
  PERMANENT fallback); the completion watcher then pulls the fresh version into
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
| `weight_store` | `backend: kimi` | versioned pull path (`kimi` or `mooncake`); `null` = stock push-based sync |
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
groups and pinned-memory (staged-bytes) accounting have landed
(subgroups activate at ≥2 rollout replicas — the shipped single-replica
example never exercises them, and the producer's batch-boundary pull
stays fleet-synchronized either way) — what
remains is cluster validation (BOTH engines now implement the versioned
protocol; mooncake's landed in item 9);
in P2 the two biggest open gaps are the relay-chain distribution's
real deployment and per-token KV introspection + KV direct migration
(items 17/18 below). For positioning vs
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
LANDED HERE: per-replica process groups (item 10), pinned-memory /
staged-bytes accounting (item 8), and the mooncake versioned path
(item 9) are done; the remaining item is the cluster validation run
itself (6) — now covering BOTH engines' stage/pull cycles.

6. Full-run cluster validation of
   `examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh` — the
   actual gate for every path in this package (syntax + CPU logic are
   verified; the cluster run is the only honest evidence). Runbook:

   **Launch** (the interesting topology is 2 rollout replicas — it
   activates the per-replica subgroups a single-replica run never
   exercises):

   ```bash
   pip install "checkpoint-engine[p2p]"   # the kimi engine
   NNODES_ROLLOUT=2 bash examples/trajectory_async/dapo_qwen25_math_7b_traj_async.sh
   ```

   **Tier 1 — the closed loop (the minimum pass).** One full step cycle
   (publish → per-row generation → group re-assembly → trainer update)
   completes, observably:
   * startup wiring logs fire — "relay controller attached
     (backend=kimi, ...)", "fault tolerance attached (heartbeat ...)",
     "partial response pool attached" (when enabled);
   * every publish takes the stage-only path and updates
     `relay/publishes` / `relay/staged_bytes`; per-rank engine lines
     `Rank N receive v{V} done ... time cost ... bandwidth` appear for
     each pull;
   * the trainer consumes complete groups (version-span /
     staleness-accounted) and steps; teardown is clean — bounded
     collector leftovers, supervisor + repack actors stopped.

   **Tier 2 — the 2-replica concurrency verdict (the watershed).** With
   repack enabled (default), a lagging IDLE replica is refreshed
   per-replica after each publish ("repack refresh: replica N ... ->
   vV"): grep the relay-controller actor logs for
   `PULL_REPLICA_BEGIN/END replica=0|1` — the windows span the full
   abort → release KV → subgroup receive → resume sequence.
   OVERLAPPING windows = the communication-domain split is live and
   per-replica pulls are truly concurrent; DISJOINT windows = a global
   barrier remains — capture the log and revisit the split BEFORE any
   pull-control push-down work (the wiring guide's two-breakthrough
   framing). `relay/replica_pulls` / `relay/replica_pull_last_s`
   corroborate. NOTE: the producer's batch-boundary pull is
   fleet-synchronized by design — PULL_REPLICA lines come from the
   repack idle refresh only, so keep a replica idle across a publish
   (small gen batch or a long generation elsewhere) to exercise it.
   Also capture the `repack/*` counters (`kv_util_delta_mean`,
   `sources_emptied`, `migrations_declined`, `overhead_total_s`) on
   this run — under today's count-denominated model they are the
   BASELINE that item 17's per-token introspection must beat, and the
   acceptance gate for 17/18 is defined against them.

   **Tier 3 — fault injection (a second run).**
   * `ray kill` the relay-controller actor → the supervisor logs
     "relay controller failed over (attempt N ...)", re-attaches the
     producer + repack consumers, and the next publish succeeds (state
     rebuilt via `recover` — no async checkpoint exists by design);
   * `ray kill` a rollout server actor → after `failure_threshold`
     probe strikes the supervisor logs "retired dead replicas [...]";
     row retries redirect onto the healthy replica (recompute
     prefill); a restarted server is re-added on its next successful
     probe.
   Every injection must leave the run ALIVE — the supervisor survives
   even a failed controller resurrection (counted, retried next
   heartbeat).

   **Known non-failures (expected — do not file):** a single-replica
   run never logs PULL_REPLICA (per-replica path inert by design —
   `repack/per_replica_pulls: 0`); migration without LB drain sockets
   logs "repack declined ... begin_drain unsupported" and degrades to
   idle refresh only; KV never actually
   moves — recompute prefill is the only migration mode; hard_drain
   requires abort-resume semantics (L2 or partial_rollout=true) —
   keep it off on the stock client.
7. `row_from_sample_batch` against real DataProto (`union(position)`
   slicing, `non_tensor_batch` `.item()` paths) — including the FAILED
   row shape (`DataProto(non_tensor_batch=...)` with no tensor batch).
8. ~~Pinned-memory accounting~~ — DONE: `stage_version` reports per-rank
   staged bytes, `RelayController` sums them and enforces
   `async_training.weight_store.max_staged_bytes` (retiring oldest live
   versions beyond `keep_last` AND the byte quota, never the latest);
   `relay/staged_bytes` / `relay/quota_retires` metrics. Cluster
   validation of the accounting itself remains (run with the metric on).
9. ~~mooncake `stage_version`~~ — DONE (engine-level): the mooncake
   engine implements the full versioned method set with the same
   contracts as kimi — `stage_version` packs whole-tensor buckets (the
   stock packing convention) into a per-version PINNED HOST buffer and
   registers it with `batch_register_memory` (kept registered — the
   buffer IS the relay memory); the descriptor reaches rollout ranks
   through the store `all_gather_obj` rendezvous (the same pattern the
   topology init uses), concurrent with the actor-side stage (mirrors
   kimi's `gather_metas`); `receive_weights_version` streams bucket by
   bucket via direct `transfer_sync_read` from the staged buffer — no
   rank chain, no barrier, per-consumer by construction, hence NO
   replica subgroups (the driver scopes a pull by dispatching to one
   replica's worker group); `unstage_version`/`drop_version` retire.
   `build_topology` accepts (and ignores) `replica_partition`, and
   `async_trainer` accepts `weight_store.backend=mooncake` (paired with
   `checkpoint_engine.backend=mooncake`). The mooncake ADAPTER
   (`MooncakeP2PBackend`, the package-level store) gained the same
   whole-tensor bucket packing and per-bucket streaming reads — fixing
   a real multi-chunk bug where every bucket overwrote the same
   consumer offset and the reassembly read past the last chunk. HONEST
   BOUNDARY: the protocol smokes are torch/mooncake-gated (not executed
   on the CPU-only dev box — they run on the first full install); the
   runtime `batch_register_memory`/`unregister_memory` semantics across
   mooncake-transfer-engine versions and the RDMA transport itself are
   CLUSTER-PENDING — validate in runbook item 6 before trusting at
   scale. Shared exposure (identical to kimi, not mooncake-specific):
   a DEAD rollout rank breaks the descriptor rendezvous (the store
   ``all_gather_obj`` misses a participant) exactly as it breaks kimi's
   ``gather_metas`` — engine-store rebuild after replica death is the
   TODO-15 cluster item. The P0 single-sender convention is kept: actor engine rank 0
   stages; other actor ranks drain (same as stock `send_weights`).
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
KV introspection + KV direct migration (17/18; 13's real-queue
semantics also remains; 9's mooncake staging is engine-implemented,
cluster-pending).

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
    `rollout.max_num_seqs` — item 17); a real `kv_transfer_fn` (KV
    blocks never travel — recompute prefill is the accepted default,
    item 18).
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
17. per-token KV introspection — OPEN (upgrades the repack planner's
    KV columns AND the honesty of its benefit metrics):
    * today the placement model is COUNT-denominated by design —
      `kv_used = inflight × kv_per_request`, `C_max = batch_bound ×
      kv_per_request` (repack_bridge.py's `ReplicaState` snapshot), so
      CanFit's KV constraint collapses onto the true batch bound B;
      the `RepackStats` rounds (`kv_util_before/after`,
      `kv_util_delta_mean`) are count-proportional, not bytes;
    * target: per-REQUEST KV token accounting from the engine plus the
      real KV pool size for C_max.
    Design constraints (fixed; do not revisit):
    (a) DUAL CONSTRAINT STAYS — B (`rollout.max_num_seqs`) is a genuine
        engine roofline, not an approximation to replace: CanFit keeps
        BOTH `C_max` (KVCache) and B (paper Algorithm 1 line 9);
    (b) ONE PRIMITIVE, TWO CONSUMERS — the engine-side per-request
        metrics RPC is SHARED with the L2 layered-scheduling stack
        (the same seam that supplies `begin_drain`/`end_drain`): design
        it once there, consume it here; do not build a second
        introspection channel;
    (c) CAPABILITY-PROBED UPGRADE — the pattern already proven twice
        (`supports_pull_replica`, the `_PullUnavailable` latch): RPC
        absent → fall back to the count-linear estimate and SAY SO in
        stats (e.g. `repack/kv_model: count|tokens`); the stock stack
        always runs. This is an upgrade path, never a hard cutover.
    Division of labor: engine + `llm_server` RPC = L2-side (external);
    the probe, planner consumption, and stats = this package.
18. KV direct migration — OPEN (one SEMANTIC decision required before
    any code):
    * the seam exists: `RolloutRepackExecutor(kv_transfer_fn=...)`
      (relay_tier.py) — default `None` → abort-resume with recompute
      prefill, the accepted default on every path;
    * SEMANTIC DECISION (prerequisite): SAME-VERSION constraint — KV
      blocks are computed under the SOURCE replica's weights; resuming
      on a different-version replica continues generation under a
      different policy (token-level drift, mid-trajectory). The pool
      precedent redirects SAME-VERSION only (cross-version partials
      refused AND dropped), and soft-mode drain guarantees no version
      mixing. KV migration must route to a same-version destination —
      or the drift must be EXPLICITLY accepted and folded into
      staleness accounting. Decide before writing code; do not
      discover it in a cluster run;
    * scope: PREFIX-KV first (prompt KV is the bulk of the recompute
      cost and needs no request-state reconstruction); full
      decode-state migration (block tables + sampler state) is a later
      engine-coupled step;
    * transport options to evaluate SIDE BY SIDE:
      (a) engine-native connectors (vLLM `kv_connector` / sglang KV
          transfer) — built for PD-disaggregation prefill→decode
          handoff, NOT cross-replica decode continuation;
      (b) the package's own mooncake path — registered staging buffers
          + `transfer_sync_read` direct reads, the same machinery as
          the versioned weights (item 9): KV blocks are just bytes, no
          engine-internal API dependency;
    * dependency order: 17 (per-request KV sizes) ∥ 16's writer seam
      (request handoff) → 18. NOT linear after 17;
    * INVARIANT: recompute prefill stays the PERMANENT fallback —
      `kv_transfer_fn` is a capability-gated fast path, never a
      replacement; a failure mid-migration falls back to abort-resume
      + recompute and must never strand a request (the drain
      lifecycle's crash-safety rules apply unchanged).

    **Repack benefit acceptance (the 17/18 gate)** — the metrics that
    define "real migration benefit", all captured on the TODO-6
    cluster run (the count-model values recorded there are the
    BASELINE 17 must improve):
    * p99 tail trajectory time (watch its interaction with
      `group_deadline_s` drops);
    * `repack/kv_util_delta_mean` (honest bytes after 17);
    * `sources_emptied` vs `migrations_declined` (plans that execute
      vs plans the missing capability killed);
    * `repack/overhead_total_s` as a share of the run.
    Phase order: runbook 6 first (baseline), then 17, then 18 —
    building 18 on an unvalidated cluster stack compounds risk
    exactly the way this package refuses to.

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
| §3.2 workflow steps ④-⑦ (interleaved train/publish/background distribute/anytime pull) | ✅ versioned path live; ⚠️ the producer's batch-boundary pull is FLEET-synchronized (the known P0 gap vs per-replica anytime pull — see the wiring guide's topology paragraph); per-replica anytime pull exists today only via the repack idle refresh (≥2 replicas); chain distribution = TODO-12 |
| §3.3 + §4.3 fault tolerance (heartbeat failover, chain rebuild, master failover, checkpoint recovery) | ✅ LANDED (`fault_tolerance.py`: supervisor failover + `recover`, probe/retire/revive; `rebuild_chain` pure core); engine-side RDMA re-registration = TODO-15 tail; checkpoint recovery = standard path + `recover` (no separate async checkpoint by design) |
| §4.2 relay hierarchy: master + per-machine relays, resharding, chain-pipelined broadcast, PCIe local pull | ⚠️ flat path via `relay_controller.py` (versioned stage + fleet pull + PER-REPLICA subgroup pulls, live); per-machine chain tier = `relay_tier.py` (CPU-verified) = TODO-12 |
| §4.2 actor stall = single push to master | ✅ `publish` returns after the master stage |
| §5 repack: triggers, version grouping, KVCache idleness, Algorithm 1 Best-Fit + CanFit(`C_max` ∧ `B`), freed sources pull fresh weights | ⚠️ closed loop LIVE (`repack_bridge.py`: post-publish + periodic triggers; idle replicas refresh to fresh versions per-replica); algorithm + executor CPU-verified; request migration + KV introspection = TODO-14 |
| §8 convergence / off-policy stability under staleness | ⚠️ paper itself derives no bound (App. D = broadcast latency; App. C lists IS-based experience sampling as future work); our mitigation = bounded staleness (collector) + loss-side version-staleness correction (`staleness_correction.py`) |

## Test coverage

`tests/experimental/trajectory_async/` (227 tests; 208 stdlib-only +
3 torch-gated adapter smokes + 2 torch-gated staleness batch-application
smokes + 14 ray/mooncake-gated smokes, all skipping gracefully without
their deps):

* aggregator: completion order, duplicates, FAILED-eviction protocol,
  late-arrival guard, buffer limits, record invariants;
<<<<<<< HEAD
* mooncake versioned engine protocol (torch+mooncake-gated): stage ->
  descriptor rendezvous -> multi-bucket direct reads with REAL bytes
  flowing -> repeatable pull -> LookupError on unknown version ->
  unstage/drop; whole-tensor bucket layouts; `build_topology` accepting
  (and ignoring) `replica_partition`; the ADAPTER smoke exercises the
  same packing/streaming end to end including the multi-bucket case the
  old single-chunk smoke missed;
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
* kimi topology guard (ray-gated): same-partition re-init is an
  idempotent no-op, changed/flat-to-partitioned re-init fails loud
  (elastic safety); per-replica pull durations recorded in snapshot
  (the 2-replica concurrency-validation window);
* per-replica pull capability degradation: unavailable pulls latch off
  on first discovery (never retried, never counted as failures), the
  capability metric reflects wiring, a drained source with unavailable
  pulls completes its drain instead of stranding, controller
  capability probe;
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
=======
* mini-batcher: exact batch emission, eviction passthrough, drain;
* rollouter: trajectory-mode early delivery vs group-mode held delivery
  (scripted-engine timing asserts), cross-mode data equivalence, retry
  semantics, all-or-nothing baseline comparison, sentinel accounting;
* end-to-end: cross-mode trained-data equivalence, accounting identities,
  staleness-drop path, failed-group exclusion, both preprocess policies;
* Laminar scheduling layer: relay timing (actor stall = master hop only,
  chain propagation, PCIe-only late pulls, divergent per-relay versions),
  engine lifecycle (batch-quota gating, KVCache plateau/ramp-down
  detection, drain-triggered weight pulls, migration incl. CanFit and
  capacity rejection), Algorithm 1 (smallest-source/fullest-destination
  packing, roofline and KV capacity constraints, busy/empty replica
  exclusion), manager loop (periodic + post-update triggers, straggler
  consolidation end-to-end, exception survival);
* multi-version weight store: publish/manifest/latest invariants
  (monotonic, unique), retention eviction (unstage + failed reads of
  evicted versions), independent per-consumer pulls with lag accounting,
  pinned pulls, concurrent publish/pull interleavings, byte accounting,
  fake-backend registered-memory lifecycle, the demo-engine relay
  contract over the real store orchestration, and the backend-selection
  factory (dispatch, name normalization, engine requirement errors);
* adapter smokes (torch-gated, run on a full machine): the real
  KimiP2PBackend / MooncakeP2PBackend against duck-typed stub engines —
  register-without-unregister retention, per-version staging buffers,
  pinned pulls through the stub `receive_tensor` / `transfer_sync_read`,
  chunk accounting, and eviction unregistering exactly the evicted
  version.


## 进行中轨迹持久化与故障接管（2026-10-19）

Laminar 默认启用独立 CPU partial store。真实 vLLM 在生成中提交 token/logprob
增量，事务确认后才能作为恢复前缀；故障恢复使用原权重版本，并对旧 attempt fence。
新增 per-relay 版本数与字节预算，包含 Mooncake 接收/本地快照 staging；到达预算时
背压权重发布，保留活跃旧版本。vLLM 驻留权重作为未来备份来源也已记入设计。

协议、清理条件、配置、测试和恢复边界见 [PARTIAL_RECOVERY.md](PARTIAL_RECOVERY.md)。


最终真卡故障验收：`qwen35_9b_partial_recovery_fault_20261019` 使用 16 NPU、
Qwen3.5-9B、原生 Mooncake TCP，主动终止一台 vLLM server 进程；该实例的
8 条轨迹按原 v0 版本成功接管，随后完成两次更新及 4 条验证，Ray SUCCEEDED、
退出码 0，没有轨迹驱逐错误。持久化/队列独立审计一致：64 train selected、
4 val selected、142 completed、2273 chunks、36352 tokens；正常退出取消剩余
24 条请求。256 token 任务只验证执行和容错；本轮不声称 24K 学习效果或多节点/RDMA 性能。

结果：`/tmp/verl-laminar-runs/qwen35_9b_partial_recovery_fault_20261019/final_partial_recovery_report.json`。


## Relay 自身容错与恢复

Mooncake 默认启用 relay 进程恢复层：独立节点 endpoint 保管模型对象，带预算的
内存 catalog 保管完整权重版本和 lease/owner，native master 支持同节点主备切换。
恢复时隔离旧 topology epoch、重启原生传输进程并重建链，保留健康缓存，补传缺失版本。
健康 vLLM 已有轨迹继续生成。2026-10-20 起默认关闭恢复数据落盘，catalog 丢失时明确失败。

实现、参数、测试及节点级恢复边界见 [RELAY_RECOVERY.md](RELAY_RECOVERY.md)。


## 默认不落盘与在线 Roofline（2026-10-20）

为避免可靠性 I/O 抹去异步收益，relay 权重备份/catalog 和 partial trajectory store
默认改为内存模式：两个配置中的 persist_to_disk 均为 false。精确版本、epoch、
lease/owner 和容量背压仍保留；内存管理服务丢失后的恢复明确作为遗留事项。
可选磁盘代码保留，异步落盘尚未开发。事件/训练日志仍用于分析。

在线 Roofline 默认启用：纯 decode 设备事件采样、图捕获过滤、TP 最慢 rank 合并、
上下文/模式分层、连续分段拟合、Kneedle、分块 bootstrap 和 P95 延迟约束。
可信后用于目的实例的重调度容量，覆盖不足则保留配置上限，不外推并发。
协议、默认参数、诊断状态和验收边界见 [ONLINE_ROOFLINE.md](ONLINE_ROOFLINE.md)。

最终回归 315 项通过、2 项跳过；16 NPU 内存/Roofline 短任务完成 2 次更新和 4 条验证，
562 条 TP 对齐样本收发一致。实际 batch 覆盖不足，保持上限 8，未声称已找到膝点。
完整验收范围和性能边界见 ONLINE_ROOFLINE.md 的验收记录。


2026-10-20 权重发布优化：默认使用 4 线程 SHA256，并对 catalog 已校验的同版本、
同 schema、同一组不可变 Ray ObjectRef 复用本地验证结果，避免重复扫描全模型。
新增导出、CPU 拷贝、Ray put、catalog/native/endpoint 分阶段计时。配置、验证边界和
实测结果见 [WEIGHT_PUBLICATION.md](WEIGHT_PUBLICATION.md)。


2026-10-20 在线容量更新：KV/SSM 使用当前已加载模型在各 TP rank 的实际缓存页预算，
Roofline 使用纯 decode 观测；admission 从 batch_per_replica 初值逐级探索并遇压力退避。
Qwen3.5 Laminar launcher 的引擎硬上限默认预留 64（LAMINAR_ENGINE_MAX_SEQS 可改），
初始并发仍为 8；不在线热扩容物理缓存池。配置、预算公式和验收边界见
[ONLINE_ROOFLINE.md](ONLINE_ROOFLINE.md)。


在线标定的受控硬件覆盖、开销配对和真实缓存压力验收方法见
[CALIBRATION_ACCEPTANCE.md](CALIBRATION_ACCEPTANCE.md)。覆盖通过与膝点辨识单独报告。

Strict synchronous 8+8 reasoning control and comparison curves: [SYNC_COMPARISON.md](SYNC_COMPARISON.md).
>>>>>>> 46d068d7 (Implemented Laminar-stype trajectory-level asynchronous trraining for verl  which including:)
