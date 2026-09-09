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
"""Active scheduling: Laminar's trajectory repack (§5), Algorithm 1.

The repack mechanism is the "主动调度" layer — it observes rollout replicas
and *acts* on them, in contrast to the passive FIFO pipeline underneath:

* **Trigger** (§5.1): a periodic check (paper: e.g. every 5s) *plus* an
  immediate trigger right after the trainer publishes new weights — the
  moment fresh versions exist is the best moment to free straggler
  replicas so they pull the new version and produce on-policy data.
* **Grouping** (§5.1 step ①): replicas are grouped by their current model
  weight version; consolidation happens *within* a group, so a migrated
  in-flight trajectory keeps generating under the policy that started it
  (no intra-trajectory version mixing from repack itself).
* **Idleness detection** (§5.2): a replica is a candidate source iff its
  KVCache is in ramp-down (``C_used < min(C_max, C_prev)``, i.e. below
  capacity and non-increasing) AND its remaining request count is below
  the roofline batch bound ``B``. No workload-specific threshold tuning.
* **Best-Fit consolidation** (Algorithm 1): sources sorted by KVCache
  footprint ascending (smallest workload is easiest to release); each
  source is packed onto the candidate destination that ends up *most
  densely packed* after the merge, subject to ``CanFit`` (projected
  KVCache load ≤ C_max and projected request count ≤ B).

What the freed sources do next is the actual payoff: with no in-flight
requests left they hit their batch boundary, pull the latest weights from
their colocated relay, and become routable for fresh prompts — restoring
decode parallelism and lowering system staleness (paper: +26% generation
throughput, +14.8% average KVCache utilization, 0.69s repack overhead).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from verl.experimental.trajectory_async.multi_replica_engine import MultiReplicaEngine, ReplicaState
from verl.experimental.trajectory_async.relay_tier import RepackExecutor

logger = logging.getLogger(__name__)


@dataclass
class RepackConfig:
    # periodic trigger cadence; the paper suggests e.g. 5s — the mock runs
    # faster than a real cluster, so default tighter
    check_interval_s: float = 1.0
    # a version group needs at least this many idle candidates to bother
    min_group_candidates: int = 2


@dataclass
class RepackStats:
    checks: int = 0
    update_triggers: int = 0
    plans: int = 0
    sources_released: int = 0  # sources in executed plans (planned)
    sources_emptied: int = 0  # sources that actually became workless
    requests_moved: int = 0
    kv_tokens_moved: int = 0  # KVCache footprint that traveled with them
    overhead_total_s: float = 0.0
    # per-round KV effect of active migration: fleet utilization before
    # and after each executed round, plus what moved (bounded history)
    rounds: list[dict] = field(default_factory=list)

    def snapshot(self) -> dict[str, float]:
        delta = [r["kv_util_after"] - r["kv_util_before"] for r in self.rounds]
        return {
            "repack/checks": self.checks,
            "repack/update_triggers": self.update_triggers,
            "repack/plans": self.plans,
            "repack/sources_released": self.sources_released,
            "repack/sources_emptied": self.sources_emptied,
            "repack/requests_moved": self.requests_moved,
            "repack/kv_tokens_moved": self.kv_tokens_moved,
            "repack/overhead_total_s": round(self.overhead_total_s, 4),
            "repack/kv_util_delta_mean": round(sum(delta) / len(delta), 4) if delta else 0.0,
            "repack/rounds": self.rounds[-64:],
        }


def best_fit_consolidation(states: list[ReplicaState]) -> list[tuple[int, int]]:
    """Algorithm 1: Best-Fit Trajectory Consolidation (pure function).

    Operates on ONE weight-version group's snapshot. Returns the plan as
    ``(source_replica_id, destination_replica_id)`` pairs.

    CanFit (paper line 9): the destination's projected load after absorbing
    everything already planned onto it plus the source stays within
    ``C_max`` (KVCache) and ``B`` (roofline batch bound). Request counts
    include waiting requests — they move too, even though they hold no
    KVCache yet.

    Destination choice (line 11): the candidate that ends up with the
    highest projected KVCache load — Best-Fit packs the fullest viable bin,
    maximizing the number of released sources.
    """
    # line 3: idle candidates — KVCache ramp-down + remaining requests < B;
    # skip empty replicas (nothing to consolidate) and mid-pull replicas
    candidates = [r for r in states if r.idle_candidate and r.has_work and not r.pulling]
    # line 4: smallest KVCache footprint first (easiest to replace)
    candidates.sort(key=lambda r: r.kv_used)

    by_id = {r.replica_id: r for r in states}
    plan: list[tuple[int, int]] = []
    emptied: set[int] = set()

    def projected_load(dst_id: int) -> tuple[int, int]:
        """(kv_load, request_load) already on dst from the current plan."""
        kv = by_id[dst_id].kv_used
        reqs = by_id[dst_id].num_running + by_id[dst_id].num_waiting
        for src_id, d_id in plan:
            if d_id == dst_id:
                kv += by_id[src_id].kv_used
                reqs += by_id[src_id].num_running + by_id[src_id].num_waiting
        return kv, reqs

    # line 6: for each source, ascending footprint
    for src in candidates:
        if src.replica_id in emptied:  # line 7
            continue
        best_dst: int | None = None
        best_load = -1
        for dst in candidates:  # line 9: destinations also come from S
            if dst.replica_id in emptied or dst.replica_id == src.replica_id or dst.pulling:
                continue
            dst_kv, dst_reqs = projected_load(dst.replica_id)
            src_kv = src.kv_used
            src_reqs = src.num_running + src.num_waiting
            if dst_kv + src_kv > dst.kv_capacity:  # C_load ≤ C_max
                continue
            if dst_reqs + src_reqs > dst.max_running:  # N_load ≤ B
                continue
            # line 11: argmax projected load — pack the fullest viable bin
            if dst_kv + src_kv > best_load:
                best_load = dst_kv + src_kv
                best_dst = dst.replica_id
        if best_dst is not None:  # line 12
            plan.append((src.replica_id, best_dst))
            emptied.add(src.replica_id)

    return plan


class RepackManager:
    """Rollout manager loop: monitor replicas, plan, execute migrations.

    One instance drives one **repack executor** — anything implementing
    :class:`~verl.experimental.trajectory_async.relay_tier.RepackExecutor`:
    the demo's :class:`MultiReplicaEngine` (mock transport) or
    :class:`~verl.experimental.trajectory_async.relay_tier.RolloutRepackExecutor`
    over real rollout replicas. The algorithm
    (:func:`best_fit_consolidation` + :class:`ReplicaState` idleness) is
    executor-agnostic. Run the manager as a task alongside the rollouter
    and trainer; call :meth:`notify_update` from the trainer's
    weight-publish path for the immediate post-update trigger.
    """

    def __init__(
        self,
        engine: MultiReplicaEngine | RepackExecutor,
        config: RepackConfig | None = None,
        on_plan=None,
    ) -> None:
        self.engine = engine
        self.config = config or RepackConfig()
        self.stats = RepackStats()
        self.on_plan = on_plan
        self._wakeup = asyncio.Event()
        self._stopped = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="repack-manager")

    async def stop(self) -> None:
        self._stopped = True
        self._wakeup.set()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def notify_update(self) -> None:
        """Immediate trigger: the trainer just published new weights — the
        best moment to free straggler replicas onto the fresh version."""
        self.stats.update_triggers += 1
        self._wakeup.set()

    async def run(self) -> None:
        """Periodic check loop (§5.1): wait for the interval OR an update
        notification, then repack once."""
        while not self._stopped:
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self.config.check_interval_s)
            except TimeoutError:
                pass
            self._wakeup.clear()
            if self._stopped:
                break
            try:
                await self.repack_once()
            except Exception:  # noqa: BLE001 — the manager must survive
                logger.exception("repack round failed; continuing")

    # ----------------------------------------------------------------- core

    async def repack_once(self) -> list[tuple[int, int]]:
        """One monitor → group → plan → execute cycle (Figure 8 steps ①-③).

        Records the round's KV-denominated effect: fleet KVCache
        utilization at trigger time vs. after the migration, KV tokens
        moved, and sources actually emptied (freed to pull fresh weights).
        """
        self.stats.checks += 1
        states = self.engine.snapshot()
        kv_util_before = self.engine.fleet_kv_util()
        idle_before = sum(1 for r in states if not r.has_work)

        # step ①: group replicas by their current weight version
        groups: dict[int, list[ReplicaState]] = {}
        for state in states:
            groups.setdefault(state.version, []).append(state)

        plan: list[tuple[int, int]] = []
        for version, group in groups.items():
            idle = sum(1 for r in group if r.idle_candidate and r.has_work and not r.pulling)
            if idle < self.config.min_group_candidates:
                continue
            group_plan = best_fit_consolidation(group)
            if group_plan:
                logger.info(
                    "repack plan (version %d): %s", version, group_plan
                )
            plan.extend(group_plan)

        if not plan:
            return []

        # step ③: transfer unfinished trajectories of sources to destinations
        result = await self.engine.migrate(plan)
        after = self.engine.snapshot()
        self.stats.plans += 1
        self.stats.sources_released += result.sources_planned
        self.stats.sources_emptied += result.sources_emptied
        self.stats.requests_moved += result.requests_moved
        self.stats.kv_tokens_moved += result.kv_tokens_moved
        self.stats.overhead_total_s += self.engine.repack_overhead_s
        self.stats.rounds.append(
            {
                "round": self.stats.plans,
                "plan": [list(pair) for pair in plan],
                "requests_moved": result.requests_moved,
                "kv_tokens_moved": result.kv_tokens_moved,
                "sources_emptied": result.sources_emptied,
                "kv_util_before": round(kv_util_before, 4),
                "kv_util_after": round(self.engine.fleet_kv_util(), 4),
                "idle_replicas_before": idle_before,
                "idle_replicas_after": sum(1 for r in after if not r.has_work),
            }
        )
        logger.info(
            "repack round %d: moved %d requests (%d KV tokens), emptied %d/%d sources, "
            "fleet KV util %.3f -> %.3f",
            self.stats.plans,
            result.requests_moved,
            result.kv_tokens_moved,
            result.sources_emptied,
            result.sources_planned,
            kv_util_before,
            self.engine.fleet_kv_util(),
        )
        if self.on_plan is not None:
            self.on_plan(plan, result.requests_moved)
        return plan
