# Trajectory-Level Asynchronous RL (experimental)

A minimal, self-contained implementation of **trajectory-level asynchronous
RL** on verl: the streaming unit between rollout and training is a *single
response* (one trajectory of one prompt), not a prompt group and not a
batch. Groups are re-assembled on the trainer side for group-based
advantage estimation (GRPO/DAPO), so algorithm semantics stay identical to
synchronous training.

The package is stdlib-only at its core (`asyncio` + `dataclasses`), runs and
is unit-tested on a bare CPU machine, and ships an A/B demo that replays an
*identical* workload (same response lengths, same failures) under both
delivery granularities. Real-engine (GPU) wiring is a thin adapter — see
the [wiring guide](#real-engine-wiring-guide) below.

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

```
rollout side                              train side
───────────                              ──────────
PromptSource
   │ one prompt = n independent requests
   ▼
TrajectoryRollouter ──┐
  per trajectory:     │  TrajectorySample          TrajectoryTrainer
  gen → reward → push │  (single response,          consume (FIFO)
                      │   opaque payload,              │
                      ▼   version, reward)             ▼
               TrajectoryQueue ──────────────►  optional per-trajectory
               (asyncio in-process;             preprocessing stage
                reuse fully_async_policy's        (policy: per-trajectory
                MessageQueue actor for            | on-group-complete)
                multi-process)                       │
                                                     ▼
                                              GroupAggregator        ┌──────────────┐
                                              (uid → n/n rows)  ────►│ MiniBatcher  │
                                              evicts groups whose     │ K groups =   │
                                              siblings died           │ one update   │
                                                                      └──────┬───────┘
                                                                             ▼
                                                              update (serialized), version += 1,
                                                              GRPO group advantages, staleness
                                                              accounting, weight-sync hook
```

Components (all under `verl/experimental/trajectory_async/`):

| Module | Role |
|---|---|
| `types.py` | `TrajectorySample` (minimum transmission unit — the analogue of `RolloutSample` shrunk to one response), `GroupRecord` (reassembled group with version-span/staleness stats) |
| `trajectory_queue.py` | in-process asyncio queue with close semantics + stats |
| `group_aggregator.py` | trainer-side 攒组: per-uid buffering, exactly-once completion emission in completion order, duplicate tolerance, eviction on terminal failure / staleness / buffer limit |
| `mini_batcher.py` | collects complete groups, emits mini-batches of `ppo_mini_batch_size` groups |
| `rollouter.py` | both delivery modes over one generation backend; per-trajectory retries; reward scoring mirrors `AgentLoopWorker` (per-row tasks) |
| `trainer.py` | consumption pipeline: queue → preprocessing (two policies) → aggregation → mini-batch → update; GRPO advantage helper; staleness drop; version timeline + inherent-staleness / tokens-per-iteration metrics |
| `mock_rollout.py` | slot-bounded fake inference server: continuous batching, lognormal long-tail lengths, deterministic per-request failures, version tracking |
| `multi_replica_engine.py` | multi-replica rollout cluster mock: per-replica KVCache lifecycle (ramp-up/plateau/ramp-down), per-activation batch quota, roofline decode-batch bound `B`, per-replica weight versions, in-flight migration API |
| `weight_relay.py` | hierarchical parameter service **timing mock** (Laminar §4): actor→master push (the only actor stall), background chain broadcast, per-replica pull-anytime — used by the CPU demo's default `--weight-store relay` |
| `versioned_weight_store.py` | **real** multi-version, pull-based weight management over P2P checkpoint engines: `VersionedWeightStore` (registry, retention GC, per-consumer state) + `KimiP2PBackend` / `MooncakeP2PBackend` adapters + `FakeP2PBackend` (owner-memory + remote-read semantics with bytes payloads) |
| `repack.py` | active scheduling (Laminar §5): idleness detection (KVCache ramp-down), Algorithm 1 Best-Fit trajectory consolidation within weight-version groups, periodic + post-update triggers |
| `group_collector.py` | the consumption core of the REAL trainer, factored stdlib-only: `TrajectoryBatchCollector` accepts both producer granularities (group-level `RolloutSample` and trajectory-level single rows), re-assembles groups, emits exactly-`ppo_mini_batch_size` fresh batches; `row_from_sample_batch` adapts real DataProto rows |
| `async_trainer.py` | **the real trainer** — `TrajectoryAsyncTrainer(FullyAsyncTrainer)`: separate deployment (inherits `SeparateRayPPOTrainer` semantics: `Role.Actor` training workers, rollout replicas on the rollout side, message-queue intake, `CheckpointEngineManager` weight sync), overriding `_get_samples_from_queue` with group-aware trajectory-level collection + `trajectory_async/*` metrics, and a multi-version pull-based weight extension point |
| `run_demo.py` | CLI A/B benchmark with data-equivalence verification (`--compare` delivery granularity, `--compare-repack` active scheduling) |

### The real trainer vs. the mock

`trainer.py` + `run_demo.py` simulate the consumption contract on CPU;
`async_trainer.py` deploys it on the real stack, structured after
`verl/experimental/fully_async_policy/fully_async_trainer.py`:

```python
@ray.remote(num_cpus=10)
class TrajectoryAsyncTrainer(FullyAsyncTrainer):   # SeparateRayPPOTrainer lineage
    # separate deployment: Role.Actor workers here, rollout replicas on the
    # rollout side (rollouter + AgentLoopManager), message queue in between.

    async def _get_samples_from_queue(self):
        # stock: collect by COUNT — one message = one whole prompt group
        # (rollout.n rows), the slowest response gates every batch.
        # ours: every message is split into per-trajectory rows and fed to
        # TrajectoryBatchCollector, which re-assembles GRPO groups and only
        # emits a batch of ppo_mini_batch_size COMPLETE, FRESH groups — so a
        # trajectory-level producer (one response per message) drops in,
        # while the stock group-level producer keeps working unchanged.
        ...
    async def _fit_update_weights(self):
        # default: stock CheckpointEngineManager push (versioned by
        # global_steps). async_training.weight_store.backend selects the
        # multi-version pull path (stage-only publish, per-replica pulls
        # at batch boundaries) — the README wiring guide is the recipe.
        ...
```

`TrajectoryBatchCollector` (the part worth testing without a cluster)
is stdlib-only and covered by `test_group_collector.py`: both producer
granularities through one path, complete-group-only batches, staleness
refusal without batch shrinkage, and the reconciliation identity
`trained + evicted + dropped_stale + leftover + incomplete == groups
started`.

## Correctness invariants

* A group is trained **iff** all `n` trajectories arrived and none
  terminally failed; evicted groups are accounted, never trained.
* Both delivery modes over the same seeds train on **identical data**
  (same responses, lengths, rewards, attempt counts) — verified
  automatically by the demo and by unit tests. Only timing may differ.
* `FAILED` sentinels for a group are emitted only after all sibling tasks
  settled, so no trajectory can arrive after its group's eviction (the
  aggregator additionally keeps a bounded dead-uid guard).
* Updates are serialized (one policy update at a time); preprocessing and
  aggregation keep flowing during an update.
* Staleness is accounted per trajectory (`model_version` at request start):
  `GroupRecord.staleness/oldest_staleness/version_span` give the exact
  off-policy distance of every update.
* Complete groups that never fill a mini-batch are **counted**
  (`trainer/groups_leftover`), never silently lost: trained + stale-dropped
  + evicted + leftover reconciles with the number of prompted groups.

## The honest benefit model

Measured on the included A/B demo (same workload both modes; H20-scale
ratios are not claimed — this is a CPU mock):

**1. Failure/retry granularity (the big one).** A terminally-failed
response costs 1 response, not its group's `n − 1` completed siblings:

```bash
python -m verl.experimental.trajectory_async.run_demo --compare \
    --failure-rate 0.08 --group-retry none
```

```
groups trained                          8             24   <- trajectory
groups dropped on rollout side         14              0   <- trajectory
wasted trajectory attempts (dropped)   93              0   <- trajectory
mean group staleness (versions)       0.5            2.5
```

Trajectory-level delivery saved 16 of 24 groups and wasted zero completed
work, at the cost of *higher staleness*: the straggler groups it saves were
generated under older versions. That staleness/coverage trade-off is the
knob `--staleness-drop` controls.

**2. Timing is *not* a benefit by itself under group aggregation.** With no
trainer-side per-trajectory work and no failures, both modes are
timing-equivalent by design: rewards already overlap generation inside
`AgentLoopWorker` (per-row tasks), and a group becomes trainable when its
slowest response arrives — regardless of where the waiting happens. The
default `--compare` run shows exactly this (identical wall time and data).
Anyone promising large throughput gains from delivery granularity alone,
under strict GRPO group semantics, is selling snake oil.

**3. Trainer-side preprocessing is where timing re-enters.** If the trainer
does per-response work (old-log-prob compute, tokenization, trainer-side
rewards), *when* that work starts matters:

* `preprocess_policy=per-trajectory` — starts on arrival, overlaps the
  siblings' remaining generation: **better wall time**, but early responses
  of slow groups head-of-line-block fast groups in the preprocess queue
  (**later first mini-batch**), and the work is speculative (wasted when
  the group later dies).
* `preprocess_policy=on-group-complete` — starts when the group completed
  at the aggregator: no speculation, earliest first mini-batch, identical
  timing to group-level delivery while keeping the failure/memory
  granularity benefits.

```bash
python -m verl.experimental.trajectory_async.run_demo --compare \
    --preprocess-latency-s 0.5 --num-prompts 48 --mini-batch-groups 6
# wall time 14.05s vs 13.84s (trajectory wins)
# time to first mini-batch 3.02s vs 5.28s (HOL blocking — try
#   --preprocess-policy on-group-complete to remove it)
```

**4. Structural benefits (not timed by the mock).** Queue/memory units are
single responses (n× finer, smoother occupancy); per-trajectory version
labels enable exact intra-group off-policy accounting — the observability
`fully_async/partial/*` metrics approximate at group level.

**Costs.** Intra-group version mixing becomes possible (weight sync while
siblings still generate — harmless for reward-based advantages, visible in
`version_span`); the trainer holds partial groups in memory; more queue
operations; saved stragglers raise average staleness (point 1).

## Quick start

```bash
# unit tests (bare CPython is enough; no numpy/ray/torch needed)
python -m unittest discover -s tests/experimental/trajectory_async -t .

# end-to-end scenarios as runnable scripts (examples/gspo_trainer style,
# every knob an env var; see examples/trajectory_async/README.md)
bash examples/trajectory_async/run_repack_ab.sh              # active-migration A/B
bash examples/trajectory_async/run_failure_isolation.sh      # retry vs all-or-nothing
bash examples/trajectory_async/run_staleness_control.sh      # freshness bound
bash examples/trajectory_async/run_weight_store_p2p.sh       # weight plane (fake|kimi|mooncake)

# single-mode run with per-event timeline
python -m verl.experimental.trajectory_async.run_demo --mode trajectory

# honest A/B with data-equivalence verification
python -m verl.experimental.trajectory_async.run_demo --compare

# Laminar-style cluster: 4 rollout replicas + relay weights + repack
python -m verl.experimental.trajectory_async.run_demo --mode trajectory \
    --replicas 4 --num-prompts 48 --mini-batch-groups 8 --update-time-s 0.5

# same, but weights flow through the real multi-version pull-based store
# (--p2p-backend fake|kimi|mooncake; kimi/mooncake need a cluster engine)
python -m verl.experimental.trajectory_async.run_demo --mode trajectory \
    --replicas 4 --weight-store p2p --p2p-backend fake --num-prompts 48 \
    --mini-batch-groups 8 --update-time-s 0.5

# active-scheduling A/B: repack off vs on, same workload
python -m verl.experimental.trajectory_async.run_demo \
    --replicas 4 --num-prompts 48 --mini-batch-groups 8 --update-time-s 0.5 \
    --compare-repack
```

> On a machine without verl's heavy dependencies, `import verl` fails in
> `verl/__init__.py` (numpy/ray). The test suite bootstraps around it via
> `tests/experimental/trajectory_async/_bootstrap.py` (a package stub that
> skips the heavy `__init__`). To run the demo on such a machine:
>
> ```bash
> python3 -c "import sys; sys.path.insert(0, 'tests/experimental/trajectory_async'); \
>     import _bootstrap; from verl.experimental.trajectory_async import run_demo; \
>     run_demo.main()" --compare
> ```

Key knobs (see `run_demo.py --help` for all): `--n` (rollout.n),
`--mini-batch-groups` (ppo_mini_batch_size in groups), `--length-sigma`
(long-tail severity), `--failure-rate`, `--max-retries`,
`--preprocess-latency-s`/`--preprocess-policy`, `--update-time-s`,
`--staleness-drop`, `--group-retry {per-trajectory,none}` (the
all-or-nothing baseline mirrors the granularity a group-level delivery
operates at); multi-replica layer: `--replicas`, `--batch-per-replica`
(assignment quota per activation), `--max-running` (roofline bound `B`),
`--kv-capacity-tokens` (`C_max`), `--repack {on,off}`,
`--repack-interval-s`, `--repack-overhead-s`, `--actor-stall-s`,
`--relay-hop-s`, `--pcie-pull-s`.

## Active scheduling à la Laminar (multi-replica layer)

Everything above (`--replicas 1`, the default) is **passive**: FIFO
delivery, static admission semaphores, reactive staleness drop. The
multi-replica layer adds the *active* scheduling of
[Laminar](https://arxiv.org/abs/2510.12633) (ByteDance Seed + HKU,
EuroSys'26 — the paper behind
[《Laminar: 让RL后训练不再等最慢轨迹》](https://zhuanlan.zhihu.com/p/1967188401481557802) and
[《解锁5.48倍吞吐量》](https://zhuanlan.zhihu.com/p/2074054741554996645)),
mapped 1:1 onto mock components:

| Laminar concept | This implementation |
|---|---|
| trajectory-level asynchrony (generate/consume each trajectory independently) | the whole package (delivery unit = single response) |
| relay workers: actor pushes to ONE master relay, chain-pipelined RDMA broadcast to per-machine relays (pinned CPU memory), rollout pulls anytime (§4) | **real path**: `VersionedWeightStore` + `KimiP2PBackend`/`MooncakeP2PBackend` (see the next section); **timing mock**: `WeightRelayService` — `publish()` blocks only `--actor-stall-s`, chain arrival at relay *i* = publish + `--relay-hop-s`·(i+1), `pull(replica)` waits for chain propagation + `--pcie-pull-s` |
| rollout pulls weights at batch completion / when released (no lockstep, version mixing across rollouts) | `MultiReplicaEngine._drain_cycle`: when a replica's activation drains (or a repack empties it) it pulls the latest version, then becomes routable again |
| dynamic repack (§5): periodic check (e.g. 5s) + immediate trigger after each weight update | `RepackManager.run()` loop (`--repack-interval-s`) + `notify_update()` called from the trainer's publish path |
| idleness metric: KVCache in ramp-down (`C_used < min(C_max, C_prev)`) and remaining requests < roofline bound `B` (§5.2) | `ReplicaState.idle_candidate` — the mock adds the equivalent direct signal `num_waiting == 0` because per-request KV grows with decode progress, so a flat-low straggler never shows a strict decline |
| Algorithm 1 Best-Fit consolidation within a weight-version group; sources sorted by footprint; destination = becomes most densely packed; `CanFit`: `C_load ≤ C_max ∧ N_load ≤ B` | `repack.best_fit_consolidation()` — same pseudocode (line-for-line comments), run per version group by `RepackManager.repack_once()` |
| freed rollouts pull the latest weights and produce fresh on-policy trajectories | migration empties a source → its drain cycle pulls a newer version → the rollouter routes new prompts there under the fresh version |

**Metrics** follow the paper's evaluation section: the headline number is
throughput in **tokens/s per RL iteration** (tokens in the trained batch ÷
interval between consecutive actor update completions —
`trainer/tokens_per_s`), plus **inherent staleness** per trajectory
(trainer version at the trajectory's finish time minus the version that
generated it; Laminar reports typically < 3), avg/peak **KVCache
utilization**, actor stall total, pull chain-wait, and the repack
activity counters. **Active migration is measured in KV terms**
(`repack/*`): `requests_moved`, `kv_tokens_moved` (the KVCache footprint
that traveled), `sources_released` (planned) vs `sources_emptied`
(actually freed — the execution-time `CanFit` re-check can reject part of
a plan), and a per-round history `repack/rounds` recording each round's
plan, what moved, and fleet KVCache utilization **before → after** the
migration (plus idle-replica count before/after) — the direct causal
measure of what one repack round did to KV utilization.

Measured with the A/B above (CPU mock, 4 replicas, 48 prompts × 8
responses, lognormal σ=1 lengths):

```
metric                                  repack off   repack on
wall time (s)                              43.201      35.363  <- repack wins
mean throughput (tok/s)                   28964        32986  <- repack wins (+14%)
avg KVCache utilization                    0.491       0.593  <- repack wins (+21%)
mean inherent staleness (versions)         0.388       0.471  <- no-repack wins
mean group staleness (versions)            1.125       1.312  <- no-repack wins
```

The scenario script `examples/trajectory_async/run_repack_ab.sh` runs the
same A/B with tighter straggling (quota 12/replica) and prints the full
per-round KV effect, e.g.:

```
wall time (s)                                      72.063         44.451  <- repack wins
mean throughput (tok/s)                           14824          31210     <- repack wins
avg KVCache utilization                            0.2821         0.4821  <- repack wins
repack: 23 rounds, 203 trajectories moved, 228640 KV tokens migrated,
        22/28 planned sources actually emptied, 11.5s total overhead
round 3: plan=[[1,3],[0,2]] moved=16 reqs (15872 KV tokens),
         KV util 0.374 -> 0.454, idle replicas 0 -> 2
```

The same shape as the paper's §8 (repack: +26% generation throughput,
+14.8% KVCache utilization, 0.69s overhead): throughput and utilization
up, a slightly staler tail as the price — the straggler trajectories that
get consolidated finish later, and freed replicas pull *newer* weights so
the system mixes more versions. `--compare-repack` also verifies data
equivalence: repack only moves in-flight work between same-version
replicas, so both runs train on identical data.

**Fidelity notes** (where the mock deliberately simplifies): no KV
preemption is modeled — admission reserves prompt + mean length, so
lognormal overshoot can transiently push utilization past `C_max`;
per-request decode rate is constant below `B` (the roofline observation);
migration moves request state instantly at a flat `--repack-overhead-s`;
the relay is a timing model, not a real RDMA/UCX fabric; and the trainer
is one in-process actor, so `complete_time`/version timelines share one
clock (in a real deployment the rollouter would stamp versions from the
synchronizer, as `fully_async_policy` does today).

## Multi-version weight management (the real path)

`weight_relay.py` is a timing model. The deployable version is
`versioned_weight_store.py`: a **pull-based, multi-version weight store**
built on the two verl checkpoint engines that already speak peer-to-peer —
`kimi_ckpt_engine` (KIMICheckpointEngine: actor shards registered in a P2P
store, receivers read owner memory directly via `receive_tensor`) and
`mooncake` (MooncakeCheckpointEngine: RDMA `TransferEngine` with registered
buffers and direct `transfer_sync_read` between sessions).

**What the store adds over the stock engines** (stock = one-shot
collective broadcast, buffer reused immediately):

1. **Versioned retention.** Each published version gets a
   version-scoped name (`actor:v{N}`) and *stays in P2P-readable memory*
   until the retention policy evicts it (`keep_last`, default 2).
   - kimi: `register_checkpoint("actor:v3", cpu_shards)` + `gather_metas`
     — and, unlike stock `send_weights`, **no unregister**; the actor's
     registered CPU shards *are* the relay memory (Laminar's pinned-CPU
     placement).
   - mooncake: per-version RDMA-registered staging buffers (instead of
     reusing the transient `buf`), advertised as
     `(session_id, ptr, nbytes, buckets)` in the manifest.
2. **Pull, not broadcast.** `publish` returns as soon as the version is
   staged — the actor stall is just offload + register. Every replica
   pulls *when it decides to* (batch boundary / repack release) via a
   direct peer read:
   - kimi: the (already-patched) `receive_tensor` scoped to that
     replica's own `ranks`/`ranks_group` — requires building one process
     group **per replica** instead of the stock single group over
     actor + all rollout workers;
   - mooncake: `transfer_sync_read` from the actor's registered session
     into the replica's local registered buffer — no chain, no barrier,
     no coordination with other replicas.
3. **Per-consumer accounting.** The store tracks each replica's current
   version, pull count, bytes and **lag** (`latest − consumer_version`)
   — the input to version-group repacking and the staleness metrics, in
   one place.

```python
# actor side (replaces CheckpointEngineManager.update_weights' barrier):
manifest = await store.publish(global_steps, actor.get_per_tensor_param())

# replica side, at its own batch boundary (or when a repack releases it):
version = await store.pull(
    f"replica-{replica_id}",
    consumer_ctx={"ranks_group": my_group, "ranks": my_ranks},  # kimi
    sink=server_adapter.load_weights_via_sink,                 # or mooncake ctx
)
```

Cross-process, the store is the control plane: wrap it in a Ray actor
whose methods are exactly `publish` / `latest_version` / `pull` /
`release` (the surface is kept Ray-actor-friendly for that reason);
version identity is the `global_steps` that already flows through the
`CheckpointEngineManager.update_weights` stack.

**Switching backends** happens at one point — `make_p2p_backend` (CLI:
`--p2p-backend {fake,kimi,mooncake}`):

```python
from verl.experimental.trajectory_async import make_p2p_backend, VersionedWeightStore

# CPU demo / tests: no cluster needed, latencies model the transport
backend = make_p2p_backend("fake", stage_latency_s=0.5, read_latency_s=0.2)

# real cluster: wrap the stock engine constructed the standard way
engine = CheckpointEngineRegistry.new("kimi_ckpt_engine", bucket_size=..., **engine_kwargs)
#   ... prepare() / build_topology() / init_process_group() as CheckpointEngineManager does
#   (kimi additionally needs one process group PER REPLICA for per-consumer pulls)
backend = make_p2p_backend("kimi", engine=engine)

engine = CheckpointEngineRegistry.new("mooncake", bucket_size=..., **engine_kwargs)
backend = make_p2p_backend("mooncake", engine=engine,
                           staging_device="cpu",   # pinned host memory (Laminar relay placement)
                           chunk_bytes=None)       # defaults to the engine's bucket size

store = VersionedWeightStore(backend, keep_last=2)   # same store either way
```

Selecting `kimi`/`mooncake` without an engine raises `ValueError` with
the exact wiring — so the CPU demo exits with instructions instead of
failing deep inside the transport.

**What is tested vs. what needs a cluster.** The orchestration —
registry, monotonic versioning, retention eviction (reads of evicted
versions fail), independent per-consumer pulls, pinned pulls, concurrent
publish/pull interleavings, byte accounting, the demo-engine contract,
and the backend-selection factory — runs and is unit-tested over
`FakeP2PBackend`, which reproduces the owner-memory + remote-read
lifecycle with bytes payloads. The kimi/mooncake adapters isolate every
real engine call behind lazy imports and are written against
`verl/checkpoint_engine/kimi_checkpoint_engine.py` and
`mooncake_checkpoint_engine.py` as checked in; they need torch + RDMA +
the engine packages to execute and should be validated on a cluster
before use (notably: mooncake's `unregister_memory` name, and kimi's
per-replica process-group topology).

The CPU demo can run the *real* orchestration over the fake transport:

```bash
python -m verl.experimental.trajectory_async.run_demo --mode trajectory \
    --replicas 4 --weight-store p2p --compare-repack \
    --num-prompts 48 --mini-batch-groups 8 --update-time-s 0.5
# report adds store/* stats, e.g.:
#   store/retained_versions: [3, 4]
#   store/consumers: replica-0={'version': 3, 'lag': 1, ...}, ...
```

with `--keep-last-versions` (retention) and `--weight-mb` (fake payload
size) as knobs; `--weight-store relay` (default) keeps the chain-broadcast
timing model instead.

## Debugging on a real machine

### Which layer needs which dependency

| Layer | Needs | How |
|---|---|---|
| L0 orchestration: store, repack, engine, demo, all CPU tests | nothing (stdlib) | `python -m unittest discover -s tests/experimental/trajectory_async -t .` |
| L1 engine import checks | pip packages only (below) | `python -c ...` snippets |
| L2 adapter smoke: real kimi/mooncake backend code over stub engines | torch (+ the two packages for the engines they stub) | `python -m unittest tests.experimental.trajectory_async.test_adapter_smoke -v` |
| L3 real transfers | GPUs + RDMA cluster | deployment wiring + the validation points below |

### Installing the engines

```bash
# kimi backend — PyPI 'checkpoint-engine' (github MoonshotAI/checkpoint-engine);
# the [p2p] extra pulls mooncake-transfer-engine>=0.3.5, which is exactly
# the p2p store KimiP2PBackend wraps
pip install "checkpoint-engine[p2p]"

# mooncake backend — PyPI 'mooncake-transfer-engine' (github kvcache-ai/Mooncake)
pip install mooncake-transfer-engine
```

* ⚠️ **do NOT `pip install mooncake`** — that PyPI name is an unrelated
  project (a spoken-language tool). The transfer engine is
  `mooncake-transfer-engine`.
* Both adapters additionally need torch (checkpoint-engine's floor is
  2.5.0), and verl's engine classes need ray plus vllm or sglang (kimi's
  parameter server uses the vllm NCCL backend; mooncake's
  `StatelessProcessGroup` import falls back to sglang).
* mooncake transfers need RDMA — verl's engine hardcodes the `"rdma"`
  protocol. For single-machine debugging without RDMA NICs, soft-RoCE
  works: `sudo rdma link add rxe0 type rxe netdev <nic>` (then the
  `device_name` argument selects it).
* Wheels: mooncake-transfer-engine ships cp310–cp313 manylinux x86_64 /
  aarch64.

### L1: import checks

```bash
python -c "from mooncake.engine import TransferEngine; print('mooncake ok')"
python -c "from checkpoint_engine.ps import ParameterServer; print('kimi ps ok')"
python - <<'EOF'
from verl.checkpoint_engine import CheckpointEngineRegistry
for backend in ("kimi_ckpt_engine", "mooncake"):
    try:
        CheckpointEngineRegistry.get(backend)
        print(backend, "registered")
    except ValueError as e:
        print(backend, "MISSING:", e)
EOF
```

### L3: what only a cluster can validate

1. **kimi per-replica topology** — `KimiP2PBackend.read_into` passes one
   replica's own `ranks`/`ranks_group` to `receive_tensor`; the stock
   manager builds a single group over actor + all rollout workers, so a
   per-replica variant of `build_topology` must be built and exercised.
2. **mooncake `unregister_memory`** — the name/semantics of the
   TransferEngine unregister counterpart to `batch_register_memory`.
3. **concurrent direct reads** — several replicas issuing
   `transfer_sync_read` against the actor's registered memory at once.

### Logging and breakpoints

* `VERL_LOGGING_LEVEL=INFO` (or `DEBUG`) — the stock engines' knob;
  the trajectory_async modules use standard `logging` too.
* `store.snapshot()` (per-consumer version/lag/bytes),
  `engine.stats.snapshot()`, `repack.stats.snapshot()` — call at any
  await point.
* Useful breakpoints: `VersionedWeightStore.publish` / `.pull`,
  `KimiP2PBackend.stage` / `.read_into`, `MooncakeP2PBackend.stage` /
  `.read_into`, `MultiReplicaEngine._drain_cycle` (the pull timing),
  `RepackManager.repack_once` (the scheduling decision).

## Real-engine wiring guide

The data plane is engine-agnostic: `TrajectorySample.payload` is opaque and
the rollouter only needs an object with
`async generate(seed, prompt_tokens) -> (num_tokens, latency_s,
model_version)`. To move from the mock to a real deployment:

1. **Engine adapter.** Replace `MockRolloutEngine` with a thin wrapper over
   `FullyAsyncAgentLoopManager` (`verl/experimental/fully_async_policy/fully_async_rollouter.py`).
   The whole trajectory-level trick at the rollout side is submitting each
   sibling as its own single-row request instead of one `n`-row request:

   ```python
   # per trajectory task (uid, traj_index):
   gen_input = single_row_dataproto            # repeat(1), not repeat(n)
   output = await manager.generate_sequences_single(gen_input)
   traj = TrajectorySample(uid=uid, traj_index=i, group_size=n,
                           payload=output,          # 1-row DataProto
                           model_version=version_at_submit,
                           num_tokens=response_len)
   ```

   Prefix caching makes the shared prompt cheap across the `n` requests.

2. **Queue.** For multi-process deployment reuse
   `fully_async_policy`'s `MessageQueue` Ray actor verbatim —
   `put_sample`/`get_sample` are payload-agnostic; serialize
   `TrajectorySample` with `ray.cloudpickle` exactly as `RolloutSample` is
   today. `max_queue_size` then counts *trajectories* (n× finer units).

3. **Group assembly.** A completed `GroupRecord` with `n` single-row
   `DataProto` payloads is assembled with `DataProto.concat` — the same
   shape `RolloutSample.full_batch` has after generation, so
   `assemble_batch_from_rollout_samples`-style postprocessing applies
   unchanged.

4. **Rewards** stay rollouter-side per trajectory (already the
   `AgentLoopWorker` behavior: per-row `_compute_score`).

5. **Weight sync / partial rollout.** For multi-replica deployments the
   weight path is the [multi-version store above](#multi-version-weight-management-the-real-path):
   the trainer publishes each version to the P2P store (kimi registered
   shards / mooncake staging buffers), each replica pulls at its own
   batch boundary, and the repack manager decides when a released
   replica refreshes. In verl terms this replaces
   `CheckpointEngineManager.update_weights`'s global barrier
   (`fully_async_policy`'s `ParameterSynchronizer` cadence included) with
   one publisher + per-replica pulls. Note the interplay with
   trajectory-level delivery: aborting in-flight work at a sync boundary
   costs strictly less than at group granularity (already-safe
   trajectories have already crossed the queue), and under pull-based
   sync a replica never aborts for weights at all — it finishes its
   batch first, then pulls. The `version_span` metric quantifies the
   within-group mixing that the resulting version skew introduces.

6. **Critic / replay / DAPO filters** attach at the mini-batch stage
   (`MiniBatcher` output), where the data shape is identical to
   `fully_async_policy`'s consumption point.

## Relation to the async-RL literature

This is the "group-preserving" corner of the design space: AReaL-style
sample-level async training additionally decouples *advantage estimation*
from the group (running baselines / decoupled PPO) so training can start
before groups complete. That changes the algorithm; it is deliberately out
of scope here. `GroupAggregator` is the seam where such an estimator would
plug in — everything upstream (delivery, retries, versioning) already works
at trajectory granularity.

Laminar is the systems-side complement: it keeps group-based algorithms
intact and buys throughput with cluster-level engineering — trajectory-level
streaming, relay-based asynchronous weight sync, and KVCache-aware repack.
The multi-replica layer above is a faithful mock of exactly that systems
design; a real deployment would swap `WeightRelayService` for the actual
parameter service and `MultiReplicaEngine` for rollout replicas behind a
routing manager.

## Test coverage

`tests/experimental/trajectory_async/` (82 tests; 80 stdlib-only + 2
torch-gated adapter smokes that skip without torch and run the real
kimi/mooncake adapter code over stub engines on a full machine):

* aggregator: completion order, duplicates, FAILED-eviction protocol,
  late-arrival guard, buffer limits, record invariants;
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
