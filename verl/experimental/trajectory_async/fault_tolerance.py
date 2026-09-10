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
"""Fleet + relay fault tolerance (paper §3.3 + §4.3).

Three mechanisms, each a small pure core (CPU-testable) plus a thin
async driver:

* :class:`ReplicaHealthMonitor` (§3.3 heartbeat detection): consecutive
  probe failures per server; ``failure_threshold`` strikes in a row =
  dead. Hysteresis: one success revives. A dead replica is RETIRED —
  removed from the load balancer's routing pool (the committed
  ``remove_servers`` socket) — so in-flight rows that fail on it are
  retried by the row-retry loop onto healthy replicas: the §3.3
  "redirect to a healthy replica" step, with recompute-prefill resume
  (the partial response pool, ``partial_pool.py``, turns that into a
  partial resume wherever a writer seam exists).
* :class:`RelaySupervisor` (§4.3 master failover): heartbeats the
  relay-controller actor; on actor death it recreates the controller
  via a factory, RECOVERS its state (the registry is derived data —
  the trainer knows the published version; per-replica versions
  re-sync on the next pull), and re-attaches every consumer (producer
  batch-boundary pulls, repack controller). Also drives the health
  monitor's probes and the retire/revive lifecycle.
* :func:`rebuild_chain` (§4.3 relay-chain rebuild): pure O(dead)
  topology splice — exclude dead ranks, neighbors reconnect; the live
  chain never re-broadcasts. Engine-side re-registration (mooncake
  RDMA buckets) is cluster-validated work; this is the planning core.

Checkpoint recovery (§3.3): the trainer's standard checkpoint path
already restores model state; the async-specific state (controller
registry) is exactly what :meth:`RelaySupervisor` recovery rebuilds —
no separate async checkpoint exists by design.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- §3.3


class ReplicaHealthMonitor:
    """Per-server liveness from consecutive probe outcomes.

    ``record(server_id, ok)`` per probe tick; ``dead_servers()`` returns
    servers with >= ``failure_threshold`` consecutive failures that have
    not yet been retired (the caller retires them and passes the list to
    :meth:`retired`). Revival is immediate on any success — a restarted
    replica is re-added to routing by the caller (:meth:`revive`).
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = max(1, int(failure_threshold))
        self._consecutive_failures: dict[str, int] = {}
        self._dead: set[str] = set()
        self._retired: set[str] = set()

        self.probe_ticks = 0
        self.probe_failures = 0
        self.deaths = 0
        self.revivals = 0

    def record(self, server_id: str, ok: bool) -> None:
        """One probe outcome for one server."""
        if ok:
            was_dead = server_id in self._dead
            self._consecutive_failures.pop(server_id, None)
            self._dead.discard(server_id)
            if was_dead:
                self.revivals += 1
        else:
            self.probe_failures += 1
            failures = self._consecutive_failures.get(server_id, 0) + 1
            self._consecutive_failures[server_id] = failures
            if failures >= self.failure_threshold and server_id not in self._dead:
                self._dead.add(server_id)
                self.deaths += 1

    def record_probe(self, results: dict[str, bool]) -> None:
        """One probe tick: server_id -> alive."""
        self.probe_ticks += 1
        for server_id, ok in results.items():
            self.record(server_id, ok)

    def dead_servers(self) -> list[str]:
        """Dead and not yet retired (sorted for deterministic ordering)."""
        return sorted(s for s in self._dead if s not in self._retired)

    def retired(self, server_ids: Sequence[str]) -> None:
        """Mark servers as retired (the caller removed them from routing)."""
        self._retired.update(server_ids)

    def revive(self, server_id: str) -> None:
        """A retired server came back (re-added to routing)."""
        self._retired.discard(server_id)
        self._dead.discard(server_id)
        self._consecutive_failures.pop(server_id, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "health/probe_ticks": self.probe_ticks,
            "health/probe_failures": self.probe_failures,
            "health/dead_replicas": len(self._dead),
            "health/retired_replicas": len(self._retired),
            "health/deaths": self.deaths,
            "health/revivals": self.revivals,
        }


# ----------------------------------------------------------------- §4.3


def rebuild_chain(order: Sequence[int], dead: set[int]) -> list[int]:
    """Rebuild a relay chain excluding dead ranks (paper §4.3, O(dead)).

    The paper's relay tier streams weights along a rank chain; a dead
    member would stall the stream. Rebuild = splice it out: neighbors
    reconnect directly, live ranks NEVER re-receive what already
    passed them. Order preservation is required (a chain's upstream/
    downstream direction is load-bearing).

    Args:
        order: the current chain order (rank ids, upstream first).
        dead: ranks to exclude.

    Returns:
        The new chain order (empty if nothing survives — the caller
        must then treat the whole path as failed, not silently loop).
    """
    survivors = [r for r in order if r not in dead]
    if not survivors:
        return []
    if len(survivors) == len(order):
        return list(survivors)  # nothing died — same chain
    logger.info(
        "relay chain rebuilt: %d -> %d ranks (removed %s)",
        len(order),
        len(survivors),
        sorted(set(order) & set(dead)),
    )
    return survivors


class RelaySupervisor:
    """Heartbeat + failover driver for the relay tier and the fleet.

    Runs one loop with two cadences:

    * every ``heartbeat_s``: ping the relay controller; on failure
      (RPC error = actor dead) run the failover: ``factory()`` recreates
      the controller, ``recover_fn(new_controller)`` re-attaches every
      consumer and recovers state, and the loop continues on the new
      handle. Consecutive failures between recoveries are counted, not
      fatal — the trainer keeps training on its current weights either
      way (only NEW version distribution is stalled).
    * every ``probe_s`` (may equal ``heartbeat_s``): probe every
      replica's liveness via ``probe_fn()`` (server_id -> alive); dead
      ones are retired via ``retire_fn`` (LB removal + repack executor
      retirement); retired ones that come back are revived via
      ``revive_fn``.

    All seams are injected callables — the whole supervisor is
    CPU-testable with fakes; the trainer wires the real ray handles.
    """

    def __init__(
        self,
        *,
        ping_fn: Callable[[], Awaitable[bool]],
        factory: Callable[[], Any],
        recover_fn: Callable[[Any], Awaitable[None]],
        probe_fn: Callable[[], Awaitable[dict[str, bool]]] | None = None,
        retire_fn: Callable[[list[str]], Awaitable[None]] | None = None,
        revive_fn: Callable[[list[str]], Awaitable[None]] | None = None,
        heartbeat_s: float = 5.0,
        probe_s: float | None = None,
        failure_threshold: int = 3,
    ):
        self.ping_fn = ping_fn
        self.factory = factory
        self.recover_fn = recover_fn
        self.probe_fn = probe_fn
        self.retire_fn = retire_fn
        self.revive_fn = revive_fn
        self.heartbeat_s = heartbeat_s
        self.probe_s = probe_s if probe_s is not None else heartbeat_s
        self.health = ReplicaHealthMonitor(failure_threshold=failure_threshold)

        self.controller: Any = None
        self.controller_alive = False
        self.failovers = 0
        self.heartbeat_failures = 0
        self._stopped = False
        self._task: asyncio.Task | None = None

    # --------------------------------------------------------- lifecycle

    def start(self, controller: Any = None) -> None:
        """Start the loop (optionally with the current controller)."""
        self.controller = controller
        self.controller_alive = controller is not None
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="relay-supervisor")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def run(self) -> None:
        """The heartbeat + probe loop. Never raises: every tick's failure
        is counted and survived — a supervisor crash would defeat its
        own purpose."""
        next_heartbeat = 0.0
        next_probe = 0.0
        while not self._stopped:
            now = asyncio.get_running_loop().time()
            wait = min(
                [w for w in (self.heartbeat_s - (now - next_heartbeat),
                             self.probe_s - (now - next_probe)) if w > 0] or [0.0],
            )
            if wait > 0:
                await asyncio.sleep(wait)
            if self._stopped:
                break
            now = asyncio.get_running_loop().time()
            if now >= next_heartbeat:
                next_heartbeat = now + self.heartbeat_s
                await self._tick_heartbeat()
            if now >= next_probe and self.probe_fn is not None:
                next_probe = now + self.probe_s
                await self._tick_probe()

    # ------------------------------------------------------------- ticks

    async def _tick_heartbeat(self) -> None:
        if self.controller is None:
            return
        try:
            alive = bool(await self.ping_fn())
        except Exception:  # noqa: BLE001 — a dead actor raises; that IS the signal
            alive = False
        if alive:
            self.controller_alive = True
            return
        self.controller_alive = False
        self.heartbeat_failures += 1
        # master failover (§4.3): recreate + recover + re-attach
        try:
            new_controller = self.factory()
            await self.recover_fn(new_controller)
            self.controller = new_controller
            self.controller_alive = True
            self.failovers += 1
            logger.warning(
                "relay controller failed over (attempt %d after %d failed heartbeat(s))",
                self.failovers,
                self.heartbeat_failures,
            )
        except Exception:  # noqa: BLE001 — the supervisor must survive
            logger.exception("relay controller failover failed; retrying next heartbeat")

    async def _tick_probe(self) -> None:
        try:
            results = await self.probe_fn()
        except Exception:  # noqa: BLE001
            logger.exception("replica liveness probe failed; skipping tick")
            return
        if not isinstance(results, dict):
            return
        self.health.record_probe(results)

        dead = self.health.dead_servers()
        if dead and self.retire_fn is not None:
            try:
                await self.retire_fn(dead)
                self.health.retired(dead)
                logger.warning("retired dead replicas %s (removed from routing)", dead)
            except Exception:  # noqa: BLE001
                logger.exception("retiring dead replicas %s failed; retrying next probe", dead)

        # revival: retired servers whose probes now succeed
        revived = [
            sid
            for sid in self.health._retired
            if results.get(sid, False)
        ]
        if revived and self.revive_fn is not None:
            try:
                await self.revive_fn(revived)
                for sid in revived:
                    self.health.revive(sid)
                logger.info("revived replicas %s (re-added to routing)", revived)
            except Exception:  # noqa: BLE001
                logger.exception("reviving replicas %s failed; retrying next probe", revived)

    # ---------------------------------------------------------- metrics

    def snapshot(self) -> dict[str, Any]:
        out = self.health.snapshot()
        out.update(
            {
                "relay_supervisor/controller_alive": self.controller_alive,
                "relay_supervisor/failovers": self.failovers,
                "relay_supervisor/heartbeat_failures": self.heartbeat_failures,
            }
        )
        return out
