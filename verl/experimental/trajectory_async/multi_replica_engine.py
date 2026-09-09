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
"""Multi-replica rollout engine with KVCache lifecycle simulation.

Models the rollout-side dynamics that Laminar's active scheduling (§5)
depends on:

* **several rollout replicas**, each with its own admission queue, KVCache
  capacity ``C_max`` (tokens) and roofline decode-batch bound ``B``;
* **per-replica assignment quota** — a replica accepts at most
  ``batch_per_replica`` requests per activation, mirroring "each rollout
  generates its own batch sampled from the prompt pool"; once the quota is
  reached the replica is not routable until the batch drains (this is what
  creates the long-tail straggler phase the repack attacks);
* **KVCache lifecycle** (paper Figure 9): utilization ramps up to C_max,
  plateaus while waiting requests refill freed space, and ramps down only
  when the remaining running requests finish — the ramp-down phase is the
  idleness signal;
* **per-replica weight versions**: a replica pulls the latest weights from
  its colocated relay (:class:`WeightRelayService`) when its batch drains
  (or when a repack empties it), so different replicas run different
  versions concurrently — no lockstep;
* **migration API** for the repack manager: move in-flight requests
  (running, with progress, or waiting) between same-version replicas.

Per-request decode rate is constant while the running count stays below
``B`` (the roofline observation: decode-step latency is stable in the
memory-bound regime), so replica throughput ≈ running count × rate.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field

from verl.experimental.trajectory_async.mock_rollout import MockGenResult, MockRolloutError
from verl.experimental.trajectory_async.weight_relay import WeightRelayService

logger = logging.getLogger(__name__)


def draw_request(
    seed: int,
    length_mean_tokens: float,
    length_sigma: float,
    min_tokens: int,
    max_tokens: int | None,
    failure_rate: float,
) -> tuple[int, bool]:
    """Deterministic (num_tokens, failed) draw for a request seed.

    Same draw order as :meth:`MockRolloutEngine.generate`, so A/B runs over
    the same seeds produce identical workloads on both engines.
    """
    import math
    import random

    rng = random.Random(seed)
    num_tokens = max(
        min_tokens,
        int(round(math.exp(rng.gauss(math.log(length_mean_tokens), length_sigma)))),
    )
    if max_tokens is not None:
        num_tokens = min(num_tokens, max_tokens)
    failed = rng.random() < failure_rate
    return num_tokens, failed


@dataclass
class MultiReplicaEngineConfig:
    # --- request draw parameters (same semantics as MockEngineConfig)
    seed: int = 7
    length_mean_tokens: float = 800.0
    length_sigma: float = 1.0
    min_tokens: int = 16
    max_tokens: int | None = 4096
    failure_rate: float = 0.0
    base_latency_s: float = 0.02  # fixed per-attempt overhead
    # --- replica / scheduling parameters
    num_replicas: int = 4
    batch_per_replica: int = 24  # assignment quota per activation (the "batch")
    max_running_requests: int = 24  # B: roofline decode batch bound
    kv_capacity_tokens: int = 32768  # C_max per replica
    decode_rate_tok_s: float = 800.0  # per-request decode rate (constant under B)
    decode_tick_s: float = 0.02  # simulation step
    repack_overhead_s: float = 0.5  # cost of one repack execution round


@dataclass
class ReplicaState:
    """Point-in-time snapshot of a replica, for the repack manager."""

    replica_id: int
    version: int
    kv_used: int
    kv_capacity: int  # C_max
    kv_prev: int  # previous tick's kv_used (ramp-down detection)
    num_running: int
    num_waiting: int
    batch_quota: int  # assignment quota per activation
    max_running: int  # B: roofline decode batch bound
    pulling: bool
    routable: bool

    @property
    def kv_util(self) -> float:
        return self.kv_used / self.kv_capacity

    @property
    def idle_candidate(self) -> bool:
        """Laminar §5.2 idleness: the KVCache is below capacity, no refill
        pressure remains, and the running count is below the roofline batch
        bound ``B`` — Algorithm 1 line 3.

        The paper detects "no refill" as a declining KVCache (``C_used <
        C_prev``); because this mock's per-request KV grows with decode
        progress, a flat-low single-straggler replica would never show a
        strict decline, so we also accept the equivalent direct signal:
        an empty waiting queue (completions are not being refilled).
        """
        below_cap = self.kv_used < self.kv_capacity
        no_refill = self.num_waiting == 0 or self.kv_used < self.kv_prev
        return below_cap and no_refill and self.num_running < self.max_running

    @property
    def has_work(self) -> bool:
        return self.num_running + self.num_waiting > 0


@dataclass
class _Request:
    seed: int
    prompt_tokens: int
    num_tokens: int
    failed: bool
    version: int  # replica version at routing time (the generating policy)
    replica_id: int
    submit_time: float
    progress: int = 0  # generated tokens so far
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    @property
    def kv(self) -> int:
        return self.prompt_tokens + self.progress


@dataclass
class MigrationResult:
    """Outcome of one repack execution round (the KV-denominated report).

    ``requests_moved`` counts trajectories migrated; ``kv_tokens_moved``
    is the KVCache footprint that traveled with them (running requests
    only — waiting ones hold no KV yet); ``sources_emptied`` is how many
    source replicas actually became workless (free to pull fresh weights
    and re-enter routing), which can be less than planned when the
    execution-time CanFit re-check rejects part of the work.
    """

    plan: list[tuple[int, int]]
    requests_moved: int = 0
    kv_tokens_moved: int = 0
    sources_emptied: int = 0

    @property
    def sources_planned(self) -> int:
        return len({src for src, _ in self.plan})


@dataclass
class EngineStats:
    requests_submitted: int = 0
    requests_finished: int = 0
    requests_failed: int = 0
    total_tokens_generated: int = 0
    migrations_executed: int = 0
    migration_rounds: int = 0
    kv_tokens_migrated: int = 0
    replica_drains: int = 0
    weight_pulls: int = 0
    kv_util_sum: float = 0.0
    kv_util_samples: int = 0
    kv_util_peak: float = 0.0

    def snapshot(self) -> dict[str, float]:
        avg_util = self.kv_util_sum / self.kv_util_samples if self.kv_util_samples else 0.0
        return {
            "engine/requests_submitted": self.requests_submitted,
            "engine/requests_finished": self.requests_finished,
            "engine/requests_failed": self.requests_failed,
            "engine/total_tokens_generated": self.total_tokens_generated,
            "engine/migration_rounds": self.migration_rounds,
            "engine/migrations_executed": self.migrations_executed,
            "engine/kv_tokens_migrated": self.kv_tokens_migrated,
            "engine/replica_drains": self.replica_drains,
            "engine/weight_pulls": self.weight_pulls,
            "engine/kv_util_avg": round(avg_util, 4),
            "engine/kv_util_peak": round(self.kv_util_peak, 4),
        }


class MultiReplicaEngine:
    """A pool of rollout replicas with KVCache-bounded decoding."""

    def __init__(
        self,
        config: MultiReplicaEngineConfig | None = None,
        relay: WeightRelayService | None = None,
    ) -> None:
        self.config = config or MultiReplicaEngineConfig()
        if self.config.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        self.relay = relay
        self.stats = EngineStats()

        self._replicas: list[dict] = []
        for i in range(self.config.num_replicas):
            self._replicas.append(
                {
                    "id": i,
                    "version": 0,
                    "running": [],
                    "waiting": deque(),
                    "kv_used": 0,
                    "kv_prev": 0,
                    "assigned": 0,
                    "pulling": False,
                }
            )
        self._tick_tasks: list[asyncio.Task] = []
        self._router_index = 0
        self._available_event = asyncio.Event()
        self._stopped = False

    # ------------------------------------------------------------- lifecycle

    def _ensure_started(self) -> None:
        if self._tick_tasks or self._stopped:
            return
        for rep in self._replicas:
            self._tick_tasks.append(asyncio.create_task(self._tick_loop(rep), name=f"replica-tick-{rep['id']}"))

    async def stop(self) -> None:
        """Cancel tick loops; call after the workload is done."""
        self._stopped = True
        for task in self._tick_tasks:
            task.cancel()
        await asyncio.gather(*self._tick_tasks, return_exceptions=True)
        self._tick_tasks.clear()

    # --------------------------------------------------------------- submit

    async def generate(self, seed: int, prompt_tokens: int = 256) -> MockGenResult:
        """Generate one trajectory on some routable replica.

        Blocks while no replica is routable (assignment quotas exhausted /
        weights pulling) — the multi-replica analogue of continuous
        batching backpressure. Records the replica's version at routing
        time, which is the policy version that generates this trajectory.
        """
        self._ensure_started()
        num_tokens, failed = draw_request(
            seed,
            self.config.length_mean_tokens,
            self.config.length_sigma,
            self.config.min_tokens,
            self.config.max_tokens,
            self.config.failure_rate,
        )
        self.stats.requests_submitted += 1
        while True:
            self._available_event.clear()
            rep = self._route()
            if rep is not None:
                break
            await self._available_event.wait()

        request = _Request(
            seed=seed,
            prompt_tokens=prompt_tokens,
            num_tokens=num_tokens,
            failed=failed,
            version=rep["version"],
            replica_id=rep["id"],
            submit_time=asyncio.get_running_loop().time(),
        )
        rep["waiting"].append(request)
        rep["assigned"] += 1
        if not self._routable(rep):
            self._notify_availability()  # wake other submitters to re-route
        result = await request.future
        if isinstance(result, Exception):
            raise result
        return result

    def _route(self) -> dict | None:
        """Round-robin over routable replicas (quota not exhausted, not
        pulling weights)."""
        for _ in range(self.config.num_replicas):
            rep = self._replicas[self._router_index]
            self._router_index = (self._router_index + 1) % self.config.num_replicas
            if self._routable(rep):
                return rep
        return None

    def _routable(self, rep: dict) -> bool:
        return not rep["pulling"] and rep["assigned"] < self.config.batch_per_replica

    # ------------------------------------------------------------ tick loop

    async def _tick_loop(self, rep: dict) -> None:
        """Per-replica simulation step: advance decode, complete requests,
        admit waiting ones, and run the drain → pull cycle."""
        interval = self.config.decode_tick_s
        try:
            while not self._stopped:
                self._step(rep)
                if self._drained(rep) and not rep["pulling"]:
                    await self._drain_cycle(rep)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            # a dead tick loop would silently stall the replica forever
            logger.exception("replica %d tick loop crashed", rep["id"])
            raise

    def _step(self, rep: dict) -> None:
        cfg = self.config
        rate = cfg.decode_rate_tok_s * cfg.decode_tick_s

        # advance running requests
        for req in list(rep["running"]):
            if req.failed:
                # failed attempts complete after the base latency
                if asyncio.get_running_loop().time() - req.submit_time >= cfg.base_latency_s:
                    self._finish(req, error=MockRolloutError(f"request seed={req.seed} failed"))
                continue
            req.progress = min(req.num_tokens, req.progress + rate)
            if req.progress >= req.num_tokens:
                self._finish(req)

        # admit waiting requests while KVCache headroom and batch bound hold;
        # admission reserves prompt + the mean expected growth (a real engine
        # would allocate blocks incrementally and preempt on exhaustion — no
        # preemption is modeled here, so lognormal overshoot can transiently
        # push utilization past C_max)
        reserve = int(self.config.length_mean_tokens)
        while rep["waiting"]:
            req = rep["waiting"][0]
            if len(rep["running"]) >= cfg.max_running_requests:
                break
            if rep["kv_used"] + req.prompt_tokens + reserve > cfg.kv_capacity_tokens:
                break
            rep["waiting"].popleft()
            rep["running"].append(req)
            rep["kv_used"] += req.prompt_tokens

        # kv bookkeeping for the idleness metric (ramp-down detection)
        rep["kv_prev"] = rep["kv_used"]
        rep["kv_used"] = sum(r.kv for r in rep["running"])
        if rep["running"]:
            util = rep["kv_used"] / cfg.kv_capacity_tokens
            self.stats.kv_util_sum += util
            self.stats.kv_util_samples += 1
            self.stats.kv_util_peak = max(self.stats.kv_util_peak, util)

    def _finish(self, req: _Request, error: Exception | None = None) -> None:
        rep = self._replicas[req.replica_id]
        if req in rep["running"]:
            rep["running"].remove(req)
            rep["kv_used"] -= req.kv
        if error is not None:
            self.stats.requests_failed += 1
            if not req.future.done():
                req.future.set_result(error)
        else:
            self.stats.requests_finished += 1
            self.stats.total_tokens_generated += req.num_tokens
            if not req.future.done():
                req.future.set_result(
                    MockGenResult(
                        num_tokens=req.num_tokens,
                        latency_s=asyncio.get_running_loop().time() - req.submit_time,
                        model_version=req.version,
                    )
                )

    # ---------------------------------------------------------- drain & pull

    @staticmethod
    def _drained(rep: dict) -> bool:
        return not rep["running"] and not rep["waiting"]

    async def _drain_cycle(self, rep: dict) -> None:
        """Batch boundary: reset the assignment quota, pull the latest
        weights from the colocated relay if a newer version exists, become
        routable again (Laminar §3.2 step ⑦ / §5.1)."""
        if rep["assigned"] > 0:
            self.stats.replica_drains += 1
        rep["assigned"] = 0
        self._notify_availability()
        if self.relay is None:
            return
        if self.relay.latest_published_version() > rep["version"]:
            rep["pulling"] = True
            self._notify_availability()
            try:
                version = await self.relay.pull(rep["id"])
                self.stats.weight_pulls += 1
                rep["version"] = version
            finally:
                rep["pulling"] = False
                self._notify_availability()

    # -------------------------------------------------------------- migration

    async def migrate(self, plan: list[tuple[int, int]]) -> MigrationResult:
        """Execute a repack plan: move every in-flight request of each
        source replica to its destination replica.

        Re-checks CanFit at execution time (destination KVCache headroom
        and roofline batch bound); the same-version constraint is the
        caller's (the manager groups replicas by version). Returns the
        round's :class:`MigrationResult` (requests moved, KV tokens
        moved, sources actually emptied). Costs ``repack_overhead_s``.
        """
        if not plan:
            return MigrationResult(plan=[])
        await asyncio.sleep(self.config.repack_overhead_s)
        result = MigrationResult(plan=list(plan))
        self.stats.migration_rounds += 1
        for src_id, dst_id in plan:
            src, dst = self._replicas[src_id], self._replicas[dst_id]
            for req in list(src["running"]) + list(src["waiting"]):
                was_running = req in src["running"]
                if not self._can_fit(dst, req):
                    continue
                self._remove(src, req)
                req.replica_id = dst_id
                if was_running:
                    dst["running"].append(req)
                    dst["kv_used"] += req.kv
                    result.kv_tokens_moved += req.kv
                else:
                    dst["waiting"].append(req)
                dst["assigned"] += 1
                result.requests_moved += 1
                self.stats.migrations_executed += 1
            if not src["running"] and not src["waiting"]:
                result.sources_emptied += 1
        self.stats.kv_tokens_migrated += result.kv_tokens_moved
        self._notify_availability()
        return result

    def _can_fit(self, dst: dict, req: _Request) -> bool:
        return (
            dst["kv_used"] + req.kv <= self.config.kv_capacity_tokens
            and len(dst["running"]) < self.config.max_running_requests
        )

    def _remove(self, rep: dict, req: _Request) -> None:
        if req in rep["running"]:
            rep["running"].remove(req)
            rep["kv_used"] -= req.kv
        elif req in rep["waiting"]:
            rep["waiting"].remove(req)

    # --------------------------------------------------------------- snapshot

    def snapshot(self) -> list[ReplicaState]:
        cfg = self.config
        return [
            ReplicaState(
                replica_id=rep["id"],
                version=rep["version"],
                kv_used=int(rep["kv_used"]),
                kv_capacity=cfg.kv_capacity_tokens,
                kv_prev=int(rep["kv_prev"]),
                num_running=len(rep["running"]),
                num_waiting=len(rep["waiting"]),
                batch_quota=cfg.batch_per_replica,
                max_running=cfg.max_running_requests,
                pulling=rep["pulling"],
                routable=self._routable(rep),
            )
            for rep in self._replicas
        ]

    @property
    def num_replicas(self) -> int:
        return self.config.num_replicas

    @property
    def repack_overhead_s(self) -> float:
        """Per-round migration cost (protocol attribute for RepackManager)."""
        return self.config.repack_overhead_s

    def fleet_kv_util(self) -> float:
        """Fleet-wide KVCache utilization: occupied tokens over the whole
        pool (all replicas × C_max) — the paper's "average KVCache
        utilization" denominator. Read by the repack manager before/after
        each round to measure what active migration actually did."""
        capacity = self.config.num_replicas * self.config.kv_capacity_tokens
        if capacity <= 0:
            return 0.0
        return sum(rep["kv_used"] for rep in self._replicas) / capacity

    def version_mix(self) -> int:
        """Number of distinct versions currently in use across replicas."""
        return len({rep["version"] for rep in self._replicas if rep["running"] or rep["waiting"]})

    def _notify_availability(self) -> None:
        self._available_event.set()
