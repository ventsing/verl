# trajectory_async scenario scripts

End-to-end scenarios for the trajectory-level async pipeline
(`verl/experimental/trajectory_async`), following the conventions of
`examples/gspo_trainer`: every knob is an environment variable, every
script runs the full pipeline and prints a metric report.

The scenarios run the **real orchestration** — rollouter → trajectory
queue → trainer-side group aggregation → mini-batcher → GRPO-style
update → multi-version weight store → multi-replica routing → repack —
on a CPU mock rollout (no GPU/cluster needed; stdlib-only Python, the
launcher falls back to the test-suite package stub when verl's heavy
imports are unavailable). See the package
[README](../../verl/experimental/trajectory_async/README.md) for what is
simulated vs. real.

## Scenarios

| script | shows | key metrics | exit code |
|---|---|---|---|
| `run_vs_sync.sh` | the paper's headline A/B: synchronous RL (wait for the whole batch, then train) vs trajectory-level async on the same replicas | wall time, time-to-first-update, end-to-end tokens/s, inherent staleness | 0 unless the pipelines train on **different data** (hard invariant) |
| `run_repack_ab.sh` | active migration (Laminar repack): same workload, repack off vs on | throughput, avg/peak KVCache util, per-round migration KV effect (`repack/rounds`: plan, requests + KV tokens moved, sources emptied, fleet KV util before→after), inherent staleness | 0 unless repack **changes trained data** (hard invariant) |
| `run_failure_isolation.sh` | trajectory- vs group-level delivery: phase 1 proves data equivalence with no failures; phase 2 shows per-trajectory retry saving groups an all-or-nothing baseline loses | groups trained, groups dropped, wasted attempts | phase 1 enforces equivalence |
| `run_staleness_control.sh` | freshness control: unbounded vs `--staleness-drop 1` | `trainer/oldest_staleness` max, `groups_dropped_stale` | 0 (measurement) |
| `run_weight_store_p2p.sh` | the weight plane: publish per update, per-replica pulls, retention GC | `store/*` (publishes, pulls, bytes, evictions, per-consumer versions) | 0; kimi/mooncake without a cluster exit with wiring instructions |

## Usage

```bash
# defaults (48 prompts × 8 responses, 4 replicas)
bash run_repack_ab.sh

# tune like any example script
REPLICAS=8 BATCH_PER_REPLICA=6 DECODE_RATE=1200 bash run_repack_ab.sh

# the weight path through the real engines (cluster only)
P2P_BACKEND=mooncake bash run_weight_store_p2p.sh
P2P_BACKEND=kimi bash run_weight_store_p2p.sh
```

The scripts are CI-able as-is: the two invariants that must never break
(delivery granularity and repack must not change what is trained) make
the demo exit non-zero when violated, so `set -e` fails the script.

## Where the active-migration algorithm lives

* **plan**: `best_fit_consolidation()` in
  `verl/experimental/trajectory_async/repack.py` — Laminar Algorithm 1:
  idleness detection (KVCache ramp-down / no refill pressure, running
  count below the roofline bound `B`), sources ascending by KVCache
  footprint, Best-Fit destination (fullest viable bin), `CanFit` =
  projected KV ≤ `C_max` and projected requests ≤ `B`;
* **trigger**: `RepackManager.run()` — periodic check **plus** an
  immediate wakeup when the trainer publishes new weights
  (`notify_update`);
* **execute**: `MultiReplicaEngine.migrate()` — CanFit re-check at
  execution time, moves running+waiting requests between same-version
  replicas, returns the round's `MigrationResult` (requests, KV tokens,
  sources actually emptied).

Unit coverage: `tests/experimental/trajectory_async/test_laminar_scheduling.py`.
