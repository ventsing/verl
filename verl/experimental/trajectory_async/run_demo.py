# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only end-to-end demo / A-B benchmark for trajectory-level async RL.

Runs the full pipeline — rollout engine → per-trajectory delivery →
trainer-side group aggregation → mini-batch updates — twice over an
*identical* workload (same per-request seeds, hence same response lengths
and same failures) and compares:

* ``group``       — fully_async_policy-style sample-level streaming: a
  prompt group (1 prompt × n responses) is delivered as one unit after all
  n responses settled;
* ``trajectory``  — this package: each response is delivered the moment it
  finishes.

Because rewards are already computed per-trajectory inside both modes
(mirroring ``AgentLoopWorker``), the measurable difference comes from (a)
the trainer-side per-trajectory preprocessing pipeline starting on arrived
responses instead of on whole groups, and (b) per-trajectory failure
retries when ``--group-retry none`` makes the baseline all-or-nothing.

Examples:

    # quick look, single mode
    python -m verl.experimental.trajectory_async.run_demo --mode trajectory

    # honest A/B with identical workload + data-equivalence verification
    python -m verl.experimental.trajectory_async.run_demo --compare

    # stress the failure-granularity benefit (baseline drops whole groups)
    python -m verl.experimental.trajectory_async.run_demo --compare \
        --failure-rate 0.08 --group-retry none

    # stress the preprocessing-pipeline benefit
    python -m verl.experimental.trajectory_async.run_demo --compare \
        --preprocess-latency-s 0.15 --reward-latency-s 0.4
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.mock_rollout import MockEngineConfig, MockRolloutEngine
from verl.experimental.trajectory_async.multi_replica_engine import (
    MultiReplicaEngine,
    MultiReplicaEngineConfig,
)
from verl.experimental.trajectory_async.repack import RepackConfig, RepackManager
from verl.experimental.trajectory_async.rollouter import PromptRecord, RollouterConfig, TrajectoryRollouter
from verl.experimental.trajectory_async.trainer import TrainerConfig, TrajectoryTrainer
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import GroupRecord, TrajectorySample
from verl.experimental.trajectory_async.versioned_weight_store import (
    VersionedStoreRelayAdapter,
    VersionedWeightStore,
    make_p2p_backend,
)
from verl.experimental.trajectory_async.weight_relay import RelayConfig, WeightRelayService


# --------------------------------------------------------------------- args


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["trajectory", "group"], default="trajectory", help="delivery mode to run")
    p.add_argument("--compare", action="store_true", help="run both modes over the same workload and diff")
    p.add_argument(
        "--compare-vs-sync",
        action="store_true",
        help="the paper's headline A/B: synchronous RL (wait for the whole batch, then "
        "train) vs the configured async pipeline over the same workload",
    )
    p.add_argument("--num-prompts", type=int, default=24)
    p.add_argument("--n", type=int, default=8, help="rollout.n — responses per prompt")
    p.add_argument("--mini-batch-groups", type=int, default=4, help="groups per policy update")
    # engine
    p.add_argument("--tokens-per-s", type=float, default=2500.0, help="decode speed of one engine slot")
    p.add_argument("--concurrency", type=int, default=64, help="engine continuous-batching slots")
    p.add_argument("--length-mean", type=float, default=1500.0, help="mean response length (tokens)")
    p.add_argument("--length-sigma", type=float, default=1.0, help="lognormal sigma (long-tail knob)")
    p.add_argument("--max-response-tokens", type=int, default=8192, help="response length cap (max_response_length)")
    p.add_argument("--failure-rate", type=float, default=0.0, help="per-attempt failure probability")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    # pipeline
    p.add_argument("--max-inflight", type=int, default=512)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--reward-latency-s", type=float, default=0.2)
    p.add_argument("--reward-pass-rate", type=float, default=0.35, help="bernoulli reward for the mock verifier")
    p.add_argument("--preprocess-latency-s", type=float, default=0.0, help="per-trajectory trainer-side work (e.g. logprob)")
    p.add_argument("--preprocess-policy", choices=["per-trajectory", "on-group-complete"], default="per-trajectory",
                   help="when trainer-side preprocessing runs: on arrival (speculative, "
                        "better wall time) or on group completion (no speculation, "
                        "earliest first mini-batch)")
    p.add_argument("--update-time-s", type=float, default=0.5, help="simulated update_actor duration")
    p.add_argument("--group-retry", choices=["per-trajectory", "none"], default="per-trajectory",
                   help="group-mode failure policy; 'none' = all-or-nothing (fully_async sample granularity)")
    p.add_argument("--staleness-drop", type=int, default=None, help="drop groups staler than this many versions")
    # --- multi-replica (Laminar-style) mode
    p.add_argument("--replicas", type=int, default=1,
                   help="number of rollout replicas; >1 switches to the multi-replica engine "
                        "with KVCache lifecycle, relay weight service and (optional) repack")
    p.add_argument("--batch-per-replica", type=int, default=24, help="assignment quota per replica activation")
    p.add_argument("--max-running", type=int, default=24, help="B: roofline decode batch bound per replica")
    p.add_argument("--kv-capacity-tokens", type=int, default=32768, help="C_max KVCache tokens per replica")
    p.add_argument("--decode-rate-tok-s", type=float, default=800.0, help="per-request decode rate")
    p.add_argument("--decode-tick-s", type=float, default=0.02, help="simulation step")
    p.add_argument("--repack", choices=["on", "off"], default="on", help="Laminar trajectory repack (active scheduling)")
    p.add_argument("--repack-interval-s", type=float, default=1.0, help="periodic repack check cadence")
    p.add_argument("--repack-overhead-s", type=float, default=0.5, help="cost of one repack round")
    p.add_argument("--actor-stall-s", type=float, default=0.5, help="actor→master relay transfer (the only actor stall)")
    p.add_argument("--relay-hop-s", type=float, default=0.1, help="per-hop chain broadcast latency")
    p.add_argument("--pcie-pull-s", type=float, default=0.2, help="relay→rollout GPU weight load")
    p.add_argument("--weight-store", choices=["relay", "p2p"], default="relay",
                   help="'relay': timing mock (chain broadcast); 'p2p': the real multi-version "
                        "VersionedWeightStore orchestration over a P2P backend (see --p2p-backend)")
    p.add_argument("--p2p-backend", choices=["fake", "kimi", "mooncake"], default="fake",
                   help="P2P transport for --weight-store p2p: 'fake' runs the full store "
                        "orchestration on CPU (owner-memory + direct-read semantics); 'kimi'/"
                        "'mooncake' select the real checkpoint-engine backends and require a "
                        "cluster (torch + RDMA + the engine package) — on a bare machine the "
                        "demo exits with the wiring instructions instead of failing deep in "
                        "the transport")
    p.add_argument("--keep-last-versions", type=int, default=2,
                   help="p2p store: how many recent weight versions stay staged (retention)")
    p.add_argument("--weight-mb", type=float, default=8.0, help="p2p store: fake payload size per version")
    p.add_argument("--compare-repack", action="store_true",
                   help="run repack=off vs repack=on over the same workload and compare")
    p.add_argument("--quiet", action="store_true", help="suppress per-event timeline prints")
    return p


# ------------------------------------------------------------------ reward


async def make_reward_fn(args: argparse.Namespace):
    """A mock verifier: latency + deterministic bernoulli correctness.

    Deterministic in the trajectory's request seed, so both modes compute
    identical rewards for identical (uid, traj_index, attempt).
    """

    async def reward_fn(traj: TrajectorySample) -> float:
        await asyncio.sleep(args.reward_latency_s)
        rng = random.Random(f"reward:{traj.payload['seed']}")
        return 1.0 if rng.random() < args.reward_pass_rate else 0.0

    return reward_fn


async def make_preprocess_fn(args: argparse.Namespace):
    """Mock trainer-side per-trajectory work (e.g. old-log-prob compute)."""

    async def preprocess_fn(traj: TrajectorySample) -> TrajectorySample:
        if args.preprocess_latency_s > 0:
            await asyncio.sleep(args.preprocess_latency_s)
        return traj

    return preprocess_fn


# ------------------------------------------------------------------- run


@dataclass
class RunResult:
    mode: str
    wall_time_s: float
    rollouter_stats: dict[str, Any]
    trainer_stats: dict[str, Any]
    queue_stats: dict[str, Any]
    aggregator_stats: dict[str, Any]
    engine_stats: dict[str, Any]
    relay_stats: dict[str, Any] = field(default_factory=dict)
    repack_stats: dict[str, Any] = field(default_factory=dict)
    trained_groups: dict[str, list[tuple]] = field(default_factory=dict)
    timeline: list[tuple] = field(default_factory=list)


async def run_one(
    args: argparse.Namespace,
    mode: str,
    repack: str | None = None,
    label: str | None = None,
    sync: bool = False,
) -> RunResult:
    """One full pipeline run.

    ``mode`` is the delivery granularity (group | trajectory); ``repack``
    overrides ``args.repack`` for A/B runs; ``label`` names the result in
    reports (defaults to ``mode``); ``sync`` selects the synchronous-RL
    baseline (no update until the whole batch finished generating).
    """
    repack_mode = args.repack if repack is None else repack
    run_label = label if label is not None else mode
    relay: WeightRelayService | None = None
    manager: RepackManager | None = None

    if args.replicas > 1:
        # --- Laminar-style deployment: several rollout replicas, each with
        # its own KVCache budget and per-activation batch quota; weights
        # flow through a hierarchical relay service (timing mock) OR the
        # real multi-version pull-based store (over the fake P2P
        # transport — same orchestration as the kimi/mooncake backends);
        # optional active scheduling via trajectory repack.
        if args.weight_store == "p2p":
            try:
                backend = make_p2p_backend(
                    args.p2p_backend,
                    stage_latency_s=args.actor_stall_s,  # offload+register ≈ the actor stall
                    read_latency_s=args.pcie_pull_s,     # direct p2p read
                )
            except ValueError as e:
                # kimi/mooncake selected on a machine without a cluster: exit
                # with the exact wiring instead of failing deep in the transport
                raise SystemExit(
                    f"--p2p-backend {args.p2p_backend!r}: {e}\n"
                    "\nOn a real cluster, build the stock engine the way "
                    "CheckpointEngineManager does, then wrap it:\n"
                    "  engine = CheckpointEngineRegistry.new(<backend>, bucket_size=...)\n"
                    "  #   ... prepare() / build_topology() / init_process_group() ...\n"
                    "  backend = make_p2p_backend(<backend>, engine=engine)\n"
                    "  store = VersionedWeightStore(backend, keep_last=2)\n"
                    "For the CPU demo use --p2p-backend fake."
                ) from e
            store = VersionedWeightStore(backend, keep_last=args.keep_last_versions)
            relay = VersionedStoreRelayAdapter(store, weight_mb=args.weight_mb)
        else:
            relay = WeightRelayService(
                RelayConfig(
                    num_relays=args.replicas,
                    actor_to_master_s=args.actor_stall_s,
                    hop_latency_s=args.relay_hop_s,
                    pcie_pull_s=args.pcie_pull_s,
                )
            )
        engine = MultiReplicaEngine(
            config=MultiReplicaEngineConfig(
                seed=args.seed,
                length_mean_tokens=args.length_mean,
                length_sigma=args.length_sigma,
                max_tokens=args.max_response_tokens,
                failure_rate=args.failure_rate,
                num_replicas=args.replicas,
                batch_per_replica=args.batch_per_replica,
                max_running_requests=args.max_running,
                kv_capacity_tokens=args.kv_capacity_tokens,
                decode_rate_tok_s=args.decode_rate_tok_s,
                decode_tick_s=args.decode_tick_s,
                repack_overhead_s=args.repack_overhead_s,
            ),
            relay=relay,
        )
        if repack_mode == "on":
            manager = RepackManager(engine=engine, config=RepackConfig(check_interval_s=args.repack_interval_s))
    else:
        engine = MockRolloutEngine(
            MockEngineConfig(
                tokens_per_s=args.tokens_per_s,
                max_concurrency=args.concurrency,
                failure_rate=args.failure_rate,
                length_mean_tokens=args.length_mean,
                length_sigma=args.length_sigma,
                max_tokens=args.max_response_tokens,
                seed=args.seed,
            )
        )
    queue = InProcessTrajectoryQueue()
    aggregator = GroupAggregator()
    mini_batcher = MiniBatcher(mini_batch_groups=args.mini_batch_groups)
    trainer = TrajectoryTrainer(
        queue=queue,
        aggregator=aggregator,
        mini_batcher=mini_batcher,
        config=TrainerConfig(
            mini_batch_groups=args.mini_batch_groups,
            update_time_s=args.update_time_s,
            max_staleness_drop=args.staleness_drop,
            preprocess_policy=args.preprocess_policy,
            sync_wait_full_batch=sync,
        ),
        preprocess_fn=await make_preprocess_fn(args),
        update_fn=None,  # default: simulated sleep + version advance
    )

    # weight-sync simulation. Single-replica: instant global sync after the
    # update. Multi-replica: Laminar §4 — after the update compute the
    # trainer pushes to the master relay only (that push is the whole actor
    # stall), the chain broadcast proceeds in the background, and each
    # replica pulls on its own when its batch drains. The post-update repack
    # trigger fires here too.
    async def update_fn(batch) -> None:
        await asyncio.sleep(args.update_time_s)
        if relay is not None:
            await relay.publish(trainer.current_version + 1)
            if manager is not None:
                manager.notify_update()
        else:
            engine.set_version(trainer.current_version + 1)

    trainer.update_fn = update_fn

    rollouter = TrajectoryRollouter(
        engine=engine,
        queue=queue,
        config=RollouterConfig(
            n=args.n,
            mode=mode,
            max_inflight_trajectories=args.max_inflight,
            max_retries=args.max_retries,
            group_retry=args.group_retry,
        ),
        reward_fn=await make_reward_fn(args),
    )

    timeline: list[tuple] = []
    trained_groups: dict[str, list[tuple]] = {}

    def on_trajectory(traj: TrajectorySample) -> None:
        timeline.append((time.monotonic(), "traj_delivered", traj.uid, traj.traj_index))
        _maybe_print(args, f"[{mode:10s}] traj {traj.uid}#{traj.traj_index} delivered "
                           f"(v{traj.model_version}, {traj.num_tokens} tok, r={traj.reward:.2f})")

    def on_group_settled(uid: str, ok: bool) -> None:
        timeline.append((time.monotonic(), "group_settled", uid, ok))
        if not ok:
            _maybe_print(args, f"[{mode:10s}] group {uid} DROPPED on rollout side")

    def on_group(group: GroupRecord) -> None:
        timeline.append((time.monotonic(), "group_completed", group.uid, group.version_span))
        _maybe_print(args, f"[{mode:10s}] group {group.uid} complete at trainer "
                           f"(span={group.version_span}, first-wait={group.train_ready_latency_s:.2f}s)")

    def on_batch(batch, metrics: dict) -> None:
        timeline.append((time.monotonic(), "batch_trained", metrics["batch_uid"], len(metrics["groups"])))
        for g in batch.groups:
            trained_groups[g.uid] = [
                (t.traj_index, t.num_tokens, t.reward, t.attempts) for t in g.trajectories
            ]
        _maybe_print(
            args,
            f"[{mode:10s}] UPDATE {metrics['batch_uid']} trained on {len(metrics['groups'])} groups "
            f"(|A|={metrics['advantage_abs_mean']:.3f})",
        )

    rollouter.on_trajectory = on_trajectory
    rollouter.on_group_settled = on_group_settled
    trainer.on_group = on_group
    trainer.on_batch = on_batch

    prompts = [PromptRecord(uid=f"p{i:04d}", prompt_tokens=args.prompt_tokens) for i in range(args.num_prompts)]

    if manager is not None:
        manager.start()

    start = time.monotonic()
    try:
        await asyncio.gather(rollouter.run(prompts, num_consumers=1), trainer.run())
    finally:
        wall = time.monotonic() - start
        if manager is not None:
            await manager.stop()
        if isinstance(engine, MultiReplicaEngine):
            await engine.stop()

    relay_stats = relay.stats.snapshot() if relay is not None else {}
    if isinstance(relay, VersionedStoreRelayAdapter):
        relay_stats.update(relay.store.snapshot())

    return RunResult(
        mode=run_label,
        wall_time_s=wall,
        rollouter_stats=rollouter.stats.snapshot(),
        trainer_stats=trainer.stats.snapshot(),
        queue_stats=queue.snapshot(),
        aggregator_stats=aggregator.snapshot(),
        engine_stats=engine.stats.snapshot(),
        relay_stats=relay_stats,
        repack_stats=manager.stats.snapshot() if manager is not None else {},
        trained_groups=trained_groups,
        timeline=timeline,
    )


def _maybe_print(args: argparse.Namespace, msg: str) -> None:
    if not args.quiet:
        print(msg, flush=True)


# ------------------------------------------------------------------ report


def _fmt_snapshot(d: dict[str, Any], indent: int = 4) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict) and v:
            inner = ", ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in v.items())
            lines.append(f"{' ' * indent}{k}: {inner}")
        elif isinstance(v, float):
            lines.append(f"{' ' * indent}{k}: {v:.4f}")
        else:
            lines.append(f"{' ' * indent}{k}: {v}")
    return "\n".join(lines)


def print_report(result: RunResult) -> None:
    print(f"\n===== mode={result.mode} =====")
    print(f"wall_time_s: {result.wall_time_s:.3f}")
    sections = ["rollouter_stats", "trainer_stats", "queue_stats", "aggregator_stats", "engine_stats"]
    if result.relay_stats:
        sections.append("relay_stats")
    if result.repack_stats:
        sections.append("repack_stats")
    for section in sections:
        print(section.replace("_stats", "") + ":")
        print(_fmt_snapshot(getattr(result, section)))


def verify_equivalence(a: RunResult, b: RunResult) -> list[str]:
    """Both modes must train on identical data (same groups, same responses,
    same rewards/attempts) — only timing may differ. Version labels are
    exempt because update timing legitimately shifts them."""
    problems = []
    ua, ub = set(a.trained_groups), set(b.trained_groups)
    if ua != ub:
        problems.append(f"trained group sets differ: only-{a.mode}={sorted(ua - ub)[:5]} only-{b.mode}={sorted(ub - ua)[:5]}")
    for uid in ua & ub:
        if a.trained_groups[uid] != b.trained_groups[uid]:
            problems.append(f"group {uid} trained on different data:\n  {a.mode}: {a.trained_groups[uid]}\n  {b.mode}: {b.trained_groups[uid]}")
    return problems


def print_compare(group: RunResult, traj: RunResult, group_retry: str) -> bool:
    """Print the A/B table; return True iff data equivalence held."""
    print("\n" + "=" * 72)
    print("A/B COMPARISON (identical workload: same lengths, same failures)")
    print("=" * 72)

    def row(name: str, ga, tb, fmt="{:>{w}}", better: str = "lower") -> None:
        w = 14
        gs = fmt.format(ga if ga is not None else "-", w=w)
        ts = fmt.format(tb if tb is not None else "-", w=w)
        mark = ""
        if ga is not None and tb is not None:
            if better == "lower":
                mark = "  <- trajectory wins" if tb < ga else ("  <- group wins" if ga < tb else "")
            else:
                mark = "  <- trajectory wins" if tb > ga else ("  <- group wins" if ga > tb else "")
        print(f"{name:<42} {gs} {ts}{mark}")

    ts, gs = traj.trainer_stats, group.trainer_stats
    ro_t, ro_g = traj.rollouter_stats, group.rollouter_stats
    print(f"{'metric':<42} {'group':>14} {'trajectory':>14}")
    row("wall time (s)", group.wall_time_s, traj.wall_time_s, "{:>{w}.3f}")
    row("time to first mini-batch (s)", gs.get("trainer/time_to_first_batch_s"), ts.get("trainer/time_to_first_batch_s"), "{:>{w}.3f}")
    row("trainer wait-for-queue (s)", gs.get("trainer/wait_for_queue_s"), ts.get("trainer/wait_for_queue_s"), "{:>{w}.3f}")
    row("groups trained", gs.get("trainer/groups_trained"), ts.get("trainer/groups_trained"), "{:>{w}d}", "higher")
    row("groups dropped on rollout side", ro_g.get("rollouter/groups_dropped"), ro_t.get("rollouter/groups_dropped"), "{:>{w}d}")
    row("wasted trajectory attempts (dropped)", ro_g.get("rollouter/wasted_trajectory_attempts"), ro_t.get("rollouter/wasted_trajectory_attempts"), "{:>{w}d}")
    row("engine busy time (s)", group.engine_stats.get("engine/busy_time_s"), traj.engine_stats.get("engine/busy_time_s"), "{:>{w}.3f}")
    row("reward busy time (s)", ro_g.get("rollouter/reward_busy_time_s"), ro_t.get("rollouter/reward_busy_time_s"), "{:>{w}.3f}")
    row("preprocess busy time (s)", gs.get("trainer/preprocess_busy_time_s"), ts.get("trainer/preprocess_busy_time_s"), "{:>{w}.3f}")
    stal_g, stal_t = gs.get("trainer/staleness", {}), ts.get("trainer/staleness", {})
    row("mean group staleness (versions)", stal_g.get("mean"), stal_t.get("mean"), "{:>{w}.3f}")
    span_g, span_t = gs.get("trainer/version_span", {}), ts.get("trainer/version_span", {})
    row("max intra-group version span", span_g.get("max"), span_t.get("max"), "{:>{w}.3f}")

    problems = verify_equivalence(group, traj)
    saved_by_trajectory = set(traj.trained_groups) - set(group.trained_groups)
    content_problems = [p for p in problems if not p.startswith("trained group sets differ")]
    lost_by_trajectory = set(group.trained_groups) - set(traj.trained_groups)
    ok = not content_problems and not lost_by_trajectory
    if content_problems:
        print("\nDATA EQUIVALENCE: FAILED — same group trained on different data")
        for p in content_problems[:10]:
            print("  - " + p)
    elif saved_by_trajectory:
        # expected under --group-retry none: the all-or-nothing baseline
        # loses whole groups to single-trajectory failures
        print(f"\nDATA EQUIVALENCE: content OK — shared groups trained on identical data; "
              f"trajectory mode additionally SAVED {len(saved_by_trajectory)} groups that "
              f"the all-or-nothing baseline dropped: {sorted(saved_by_trajectory)[:8]}")
    elif lost_by_trajectory:
        print("\nDATA EQUIVALENCE: FAILED — trajectory mode lost groups the baseline trained")
    else:
        n = len(set(group.trained_groups) & set(traj.trained_groups))
        print(f"\nDATA EQUIVALENCE: OK — both modes trained the same {n} groups on "
              f"identical responses (lengths, rewards, attempts)")
    if group_retry == "none":
        print("\n(baseline is all-or-nothing: one dead trajectory drops its whole group; "
              "run without --group-retry none for the pure delivery-granularity comparison)")
    print(
        "\nNOTE: with zero trainer-side preprocessing and no failures, both modes are\n"
        "timing-equivalent by design — rewards already overlap generation inside both\n"
        "(mirroring AgentLoopWorker's per-row tasks). The trajectory-level wins show up\n"
        "when per-trajectory trainer-side work exists or failures happen:\n"
        "  # per-trajectory preprocessing pipeline (e.g. old-log-prob compute)\n"
        "  python -m verl.experimental.trajectory_async.run_demo --compare \\\n"
        "      --preprocess-latency-s 0.5 --num-prompts 48 --mini-batch-groups 6\n"
        "  # failure granularity: all-or-nothing baseline loses whole groups\n"
        "  python -m verl.experimental.trajectory_async.run_demo --compare \\\n"
        "      --failure-rate 0.08 --group-retry none\n"
        "The preprocess policy itself is a knob: 'per-trajectory' (default) overlaps\n"
        "preprocessing with remaining generation (better wall time) but head-of-line-\n"
        "blocks fast groups behind slow groups' early trajectories (later first batch);\n"
        "'on-group-complete' avoids the speculation entirely."
    )
    return ok


def print_repack_compare(off: RunResult, on: RunResult) -> bool:
    """Laminar-style A/B: same multi-replica workload, repack off vs on.

    Returns True iff data equivalence held (repack must not change what
    is trained — only where in-flight trajectories finish).
    """
    print("\n" + "=" * 72)
    print("REPACK A/B (identical workload: same lengths, same failures, same replicas)")
    print("=" * 72)

    def row(name: str, a, b, fmt="{:>{w}}", better: str = "lower") -> None:
        w = 14
        as_ = fmt.format(a if a is not None else "-", w=w)
        bs = fmt.format(b if b is not None else "-", w=w)
        mark = ""
        if a is not None and b is not None:
            if better == "lower":
                mark = "  <- repack wins" if b < a else ("  <- no-repack wins" if a < b else "")
            else:
                mark = "  <- repack wins" if b > a else ("  <- no-repack wins" if a > b else "")
        print(f"{name:<42} {as_} {bs}{mark}")

    ts_off, ts_on = off.trainer_stats, on.trainer_stats
    es_off, es_on = off.engine_stats, on.engine_stats
    print(f"{'metric':<42} {'repack off':>14} {'repack on':>14}")
    row("wall time (s)", off.wall_time_s, on.wall_time_s, "{:>{w}.3f}")
    row("mean throughput (tok/s)", ts_off.get("trainer/tokens_per_s", {}).get("mean"), ts_on.get("trainer/tokens_per_s", {}).get("mean"), "{:>{w}.1f}", "higher")
    row("avg KVCache utilization", es_off.get("engine/kv_util_avg"), es_on.get("engine/kv_util_avg"), "{:>{w}.4f}", "higher")
    row("peak KVCache utilization", es_off.get("engine/kv_util_peak"), es_on.get("engine/kv_util_peak"), "{:>{w}.4f}", "higher")
    row("mean inherent staleness (versions)", ts_off.get("trainer/inherent_staleness", {}).get("mean"), ts_on.get("trainer/inherent_staleness", {}).get("mean"), "{:>{w}.3f}")
    row("mean group staleness (versions)", ts_off.get("trainer/staleness", {}).get("mean"), ts_on.get("trainer/staleness", {}).get("mean"), "{:>{w}.3f}")
    row("replica drains (batches completed)", es_off.get("engine/replica_drains"), es_on.get("engine/replica_drains"), "{:>{w}d}")
    rs = on.repack_stats
    print(f"\nrepack activity: checks={rs.get('repack/checks', 0)}, post-update triggers="
          f"{rs.get('repack/update_triggers', 0)}, plans={rs.get('repack/plans', 0)}, "
          f"sources released={rs.get('repack/sources_released', 0)}, "
          f"sources actually emptied={rs.get('repack/sources_emptied', 0)}, "
          f"trajectories moved={rs.get('repack/requests_moved', 0)}, "
          f"KV tokens migrated={rs.get('repack/kv_tokens_moved', 0)}, "
          f"overhead={rs.get('repack/overhead_total_s', 0):.2f}s")
    rounds = rs.get("repack/rounds", [])
    if rounds:
        deltas = [r["kv_util_after"] - r["kv_util_before"] for r in rounds]
        print("per-round KV effect of active migration (fleet KVCache utilization at "
              "trigger -> after the round):")
        for r in rounds[-6:]:
            print(f"  round {r['round']}: plan={r['plan']} moved={r['requests_moved']} reqs "
                  f"({r['kv_tokens_moved']} KV tokens), emptied {r['sources_emptied']} source(s), "
                  f"KV util {r['kv_util_before']:.3f} -> {r['kv_util_after']:.3f}, "
                  f"idle replicas {r['idle_replicas_before']} -> {r['idle_replicas_after']}")
        print(f"  mean per-round KV util delta: {sum(deltas) / len(deltas):+.4f}")
    rly = on.relay_stats
    if rly:
        print(f"relay service: publishes={rly.get('relay/publishes', 0)}, "
              f"actor stall total={rly.get('relay/actor_stall_total_s', 0):.2f}s, "
              f"pulls={rly.get('relay/pulls', 0)}, chain-wait={rly.get('relay/pull_wait_total_s', 0):.2f}s, "
              f"pcie={rly.get('relay/pcie_total_s', 0):.2f}s")

    problems = verify_equivalence(off, on)
    if problems:
        print("\nDATA EQUIVALENCE: FAILED — repack must not change trained data")
        for p in problems[:10]:
            print("  - " + p)
    else:
        n = len(set(off.trained_groups) & set(on.trained_groups))
        print(f"\nDATA EQUIVALENCE: OK — both runs trained the same {n} groups on identical data "
              f"(repack only moves in-flight work between same-version replicas)")
    print(
        "\nNOTE: repack wins when replicas fall into long-tail straggling (KVCache in\n"
        "ramp-down, running count below the roofline bound B). Consolidating stragglers\n"
        "frees their replicas to pull fresh weights and start on-policy batches — at the\n"
        "cost of one migration round and slightly older trajectories finishing elsewhere."
    )
    return not problems


def print_sync_compare(sync: RunResult, asy: RunResult) -> bool:
    """The paper's headline A/B: synchronous RL vs trajectory-level async.

    Same workload, same replicas. Sync (Laminar Figure 3(a)): the trainer
    waits for the slowest trajectory before its first update; async:
    updates start as soon as the first mini-batch of complete groups
    assembled (updates overlap remaining generation). Data equivalence is
    the hard invariant.
    """
    print("\n" + "=" * 72)
    print("SYNC vs TRAJECTORY-ASYNC (identical workload: same lengths, same failures)")
    print("=" * 72)

    def row(name, a, b, fmt="{:>{w}}", better: str = "lower") -> None:
        w = 16
        as_ = fmt.format(a if a is not None else "-", w=w)
        bs = fmt.format(b if b is not None else "-", w=w)
        mark = ""
        if a is not None and b is not None:
            if better == "lower":
                mark = "  <- async wins" if b < a else ("  <- sync wins" if a < b else "")
            else:
                mark = "  <- async wins" if b > a else ("  <- sync wins" if a > b else "")
        print(f"{name:<40} {as_} {bs}{mark}")

    ts_s, ts_a = sync.trainer_stats, asy.trainer_stats
    print(f"{'metric':<40} {'sync':>16} {'async':>16}")
    row("wall time (s)", sync.wall_time_s, asy.wall_time_s, "{:>{w}.3f}")
    row("time to first update (s)", ts_s.get("trainer/time_to_first_batch_s"), ts_a.get("trainer/time_to_first_batch_s"), "{:>{w}.3f}")
    row("groups trained", ts_s.get("trainer/groups_trained"), ts_a.get("trainer/groups_trained"), "{:>{w}d}", "higher")
    # end-to-end throughput (the paper's headline metric): trained tokens
    # over wall time — NOT tokens-per-update-interval, whose denominator
    # degenerates in sync mode (updates run back-to-back after generation)
    def _e2e(r: RunResult) -> float:
        return r.trainer_stats.get("trainer/total_trained_tokens", 0) / max(r.wall_time_s, 1e-9)

    row("end-to-end throughput (tok/s)", _e2e(sync), _e2e(asy), "{:>{w}.1f}", "higher")
    row("mean inherent staleness (versions)", ts_s.get("trainer/inherent_staleness", {}).get("mean"), ts_a.get("trainer/inherent_staleness", {}).get("mean"), "{:>{w}.3f}")
    row("max intra-group version span", ts_s.get("trainer/version_span", {}).get("max"), ts_a.get("trainer/version_span", {}).get("max"), "{:>{w}.1f}")

    problems = verify_equivalence(sync, asy)
    if problems:
        print("\nDATA EQUIVALENCE: FAILED — sync and async must train on the same data")
        for p in problems[:10]:
            print("  - " + p)
        return False
    n = len(set(sync.trained_groups) & set(asy.trained_groups))
    print(f"\nDATA EQUIVALENCE: OK — both pipelines trained the same {n} groups on identical data")
    print(
        "\nNOTE: sync waits for the slowest trajectory of the batch before its first\n"
        "update (Laminar Fig. 3(a)); trajectory-async starts updating on complete\n"
        "groups while the long tail is still generating (Fig. 3(e)) — the paper's\n"
        "headline speedup mechanism, at the cost of per-trajectory staleness.\n"
        "(Sync here = one fully-generated batch, then sequential mini-batch updates;\n"
        "the paper's 5.48x is against full multi-iteration sync training.)"
    )
    return True


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.compare_vs_sync:
        # sync baseline: group delivery, no repack, updates held until the
        # whole batch finished generating (Laminar Figure 3(a))
        sync = asyncio.run(run_one(args, "group", repack="off", label="sync", sync=True))
        asy = asyncio.run(run_one(args, args.mode, label="async"))
        if not args.quiet:
            print_report(sync)
            print_report(asy)
        if not print_sync_compare(sync, asy):
            raise SystemExit(1)  # sync vs async must never change trained data
    elif args.compare_repack:
        if args.replicas < 2:
            print("--compare-repack needs --replicas >= 2 (single replica has nothing to consolidate)")
            raise SystemExit(2)
        off = asyncio.run(run_one(args, args.mode, repack="off", label="repoff"))
        on = asyncio.run(run_one(args, args.mode, repack="on", label="repon"))
        if not args.quiet:
            print_report(off)
            print_report(on)
        if not print_repack_compare(off, on):
            raise SystemExit(1)  # repack must never change what is trained
    elif args.compare:
        group = asyncio.run(run_one(args, "group"))
        traj = asyncio.run(run_one(args, "trajectory"))
        if not args.quiet:
            print_report(group)
            print_report(traj)
        if not print_compare(group, traj, group_retry=args.group_retry):
            raise SystemExit(1)  # delivery granularity must never change data
    else:
        result = asyncio.run(run_one(args, args.mode))
        print_report(result)


if __name__ == "__main__":
    main()
