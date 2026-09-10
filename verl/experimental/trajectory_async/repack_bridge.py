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
"""Repack closed loop over the real rollout fleet (paper §5 wiring).

What runs for real on this stack today — and what deliberately does not:

* **Idle-replica refresh** (real): right after a publish, replicas with no
  in-flight requests that lag the latest version pull it over their OWN
  engine subgroup (``RelayController.pull_replica``). This is the paper's
  repack payoff — "freed sources pull fresh weights" — for the common case
  of replicas draining naturally between generations; no migration needed.
* **KV-stats-driven idleness** (partial): per-replica in-flight counts come
  from the rollout side's load balancer; per-token KV usage introspection
  needs engine-side metrics RPCs (cluster TODO), so a replica counts as
  idle only with ZERO in-flight requests (conservative — a replica with
  one straggler stays untouched).
* **Cross-replica request migration** (NOT real yet):
  ``RolloutReplicaHandle.remove_request`` / ``admit_request`` require
  per-request control RPCs on the rollout servers (abort-one + redirect +
  re-admit). Until those exist, views report ``migration_supported=False``
  and the executor SKIPS migration plans (never half-migrates: no request
  is dropped, no group is lost) while still counting what it declined —
  the metric ``repack/migrations_declined`` shows the untapped headroom.

The :class:`RolloutReplicaView` implements the handle protocol from
``relay_tier.py`` over injected async callables, so the whole bridge is
CPU-testable without ray; :func:`build_repack_controller` wires the real
relay controller + rollouter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from verl.experimental.trajectory_async.repack import RepackConfig, RepackManager
from verl.experimental.trajectory_async.relay_tier import (
    MigrationResult,
    ReplicaState,
    RolloutRepackExecutor,
    RunningRequest,
)

logger = logging.getLogger(__name__)


class MigrationUnsupported(RuntimeError):
    """Raised by views whose rollout servers lack per-request control RPCs."""


# injected seams (all awaitable from the bridge actor):
InflightFn = Callable[[], Awaitable[dict[str, int] | None]]  # server_id -> inflight
ReplicaVersionFn = Callable[[int], Awaitable[int | None]]
PullReplicaFn = Callable[[int, "int | None"], Awaitable[int | None]]  # (replica_id, version|None=latest)


@dataclass
class RolloutReplicaView:
    """One rollout replica as seen by the repack bridge.

    Args:
        replica_id: engine replica index (matches the relay controller's
            replica partition order).
        server_id: the load-balancer server id of this replica (identity
            mapping documented in :func:`build_repack_controller`).
        version_fn: async -> the version this replica currently runs.
        pull_fn: async (version) -> pulls that version into this replica
            over its own engine subgroup (idleness is the CALLER's gate).
        inflight_fn: async -> in-flight request count, or None when
            unknown (treated as BUSY — never refresh a maybe-busy replica).
        migration_supported: whether request migration RPCs exist.
    """

    replica_id: int
    server_id: str
    version_fn: ReplicaVersionFn
    pull_fn: PullReplicaFn
    inflight_fn: InflightFn
    migration_supported: bool = False

    # --- RolloutReplicaHandle protocol (relay_tier) ---

    @property
    def weight_version(self) -> int:
        return -1  # async-only on this bridge; snapshot() fills the real value

    async def current_version(self) -> int | None:
        return await self.version_fn(self.replica_id)

    async def inflight(self) -> int | None:
        return await self.inflight_fn()

    async def running_count(self) -> int | None:
        inflight = await self.inflight_fn()
        return None if inflight is None else inflight.get(self.server_id)

    def kv_used_tokens(self) -> int:
        return 0  # no per-token KV introspection RPC yet (cluster TODO)

    def kv_capacity_tokens(self) -> int:
        return 0

    def batch_limit(self) -> int:
        return 0

    def running_requests(self) -> list[RunningRequest]:
        return []  # per-request listing needs engine RPCs (see module doc)

    def remove_request(self, request_id: str) -> RunningRequest:
        raise MigrationUnsupported(
            f"replica {self.replica_id}: per-request migration needs rollout-server "
            "abort/redirect RPCs (cluster TODO); refusing to half-migrate"
        )

    async def admit_request(self, request: RunningRequest, *, prefill: str = "recompute") -> None:
        raise MigrationUnsupported(
            f"replica {self.replica_id}: per-request migration needs rollout-server "
            "admit RPCs (cluster TODO); refusing to half-migrate"
        )

    async def pull_weights(self, version: int | None = None) -> int | None:
        return await self.pull_fn(self.replica_id, version)


class FleetRepackExecutor(RolloutRepackExecutor):
    """Repack executor over :class:`RolloutReplicaView` replicas.

    Inherits the migration machinery from ``RolloutRepackExecutor`` but
    overrides :meth:`migrate` to DECLINE plans when views cannot migrate
    (instead of crashing or half-migrating), and adds :meth:`refresh_idle`
    — the real, executable repack payoff on this stack: idle replicas
    lagging the latest version pull it right after a publish.
    """

    def __init__(self, handles: Sequence[RolloutReplicaView], *, latest_version_fn: Callable[[], Awaitable[int | None]]):
        # bypass RolloutRepackExecutor.__init__: its handle dict expects
        # synchronous handle methods; this bridge is async-seamed
        self.handles: dict[int, RolloutReplicaView] = {h.replica_id: h for h in handles}
        self.kv_transfer_fn = None
        self.repack_overhead_s = 0.0
        self.latest_version_fn = latest_version_fn
        self.refreshes = 0
        self.refresh_failures = 0
        self.migrations_declined = 0

    # ------------------------------------------------------------ probes

    async def _snapshot_states(self) -> list[tuple[RolloutReplicaView, int | None, int | None]]:
        """Per view: (view, current_version, running_count)."""
        async def one(view: RolloutReplicaView):
            version, running = await asyncio.gather(view.current_version(), view.running_count())
            return view, version, running

        return await asyncio.gather(*[one(v) for v in self.handles.values()])

    async def refresh_idle(self) -> int:
        """Pull the latest version into every IDLE replica that lags it.

        A replica is idle iff its in-flight count is known and zero —
        unknown counts are treated as busy (conservative). Returns the
        number of replicas refreshed.
        """
        latest = await self.latest_version_fn()
        if latest is None:
            return 0
        refreshed = 0
        for view, version, running in await self._snapshot_states():
            if version is None or running is None or running > 0:
                continue  # unknown version, unknown load, or busy -> skip
            if version >= latest:
                continue  # already fresh
            try:
                await view.pull_fn(view.replica_id, latest)
                refreshed += 1
                logger.info("repack refresh: replica %d %s -> v%d", view.replica_id, view.server_id, latest)
            except Exception:  # noqa: BLE001 — one replica must not stop the rest
                self.refresh_failures += 1
                logger.exception("repack refresh failed for replica %d", view.replica_id)
        self.refreshes += refreshed
        return refreshed

    # -------------------------------------------- RepackExecutor protocol

    def snapshot(self) -> list[ReplicaState]:
        """Synchronous snapshot DEGRADED on this bridge (async probes only):
        returns empty — the async path (:meth:`refresh_idle`) is the live
        one; migration planning needs per-request views anyway (TODO)."""
        return []

    def fleet_kv_util(self) -> float:
        return 0.0

    async def migrate(self, plan: list[tuple[int, int]]) -> MigrationResult:
        """Decline migrations when the views cannot migrate (the honest
        P0 state); count what was declined so the headroom stays visible."""
        if not plan:
            return MigrationResult(plan=[], requests_moved=0, kv_tokens_moved=0, sources_emptied=0)
        self.migrations_declined += len(plan)
        logger.warning(
            "repack declined %d migration(s) %s: rollout servers lack per-request "
            "control RPCs (cluster TODO); requests keep running where they are",
            len(plan),
            plan,
        )
        return MigrationResult(
            plan=[],
            requests_moved=0,
            kv_tokens_moved=0,
            sources_emptied=0,
        )

    def bridge_metrics(self) -> dict[str, Any]:
        return {
            "repack/idle_refreshes": self.refreshes,
            "repack/refresh_failures": self.refresh_failures,
            "repack/migrations_declined": self.migrations_declined,
        }


# ------------------------------------------------------------------ wiring


async def _default_inflight() -> dict[str, int] | None:
    return None


async def _rollouter_inflight(rollouter) -> dict[str, int] | None:
    """Per-server in-flight counts from the rollouter's load balancer."""
    import ray

    try:
        return await rollouter.replica_inflight.remote()
    except AttributeError:
        return None
    except Exception:  # noqa: BLE001 — probing must never kill the loop
        logger.exception("replica_inflight probe failed")
        return None


def build_repack_controller(
    relay_controller,
    rollouter=None,
    server_ids: Sequence[str] | None = None,
    config: RepackConfig | None = None,
) -> tuple[RepackManager, FleetRepackExecutor]:
    """Build the repack manager + fleet executor over the real controller.

    Args:
        relay_controller: the :class:`RelayControllerActor` handle (version
            probes + per-replica pulls).
        rollouter: optional rollouter exposing ``replica_inflight()`` (the
            producer's load-balancer probe); without it, inflight counts
            are unknown and refresh stays conservative (nothing refreshes).
        server_ids: the load-balancer server id per replica index. Identity
            convention: replica i (engine partition order) ↔ the i-th entry.
            Defaults to ["0", "1", ...] — pass the real ids when they
            diverge (cluster TODO: unify replica identity engine ↔ LB).
        config: :class:`RepackConfig` for the manager loop.

    Returns:
        (manager, executor) — start the manager loop with ``manager.start()``.
    """
    async def replica_version_fn(replica_id: int) -> int | None:
        try:
            return await relay_controller.replica_version.remote(replica_id)
        except Exception:  # noqa: BLE001
            logger.exception("replica_version probe failed for %d", replica_id)
            return None

    async def latest_version_fn() -> int | None:
        try:
            return await relay_controller.latest_version.remote()
        except Exception:  # noqa: BLE001
            logger.exception("latest_version probe failed")
            return None

    async def pull_replica_fn(replica_id: int, version: int | None) -> int | None:
        if version is None:
            return await relay_controller.pull_replica.remote(replica_id)
        return await relay_controller.pull_replica.remote(replica_id, version)

    if server_ids is None:
        # without an explicit server-id list we cannot map LB servers to
        # engine replicas — stay conservative (unknown inflight -> never
        # refresh); the launcher passes the real mapping
        ids: list[str] = []
        inflight_fn = _default_inflight
    else:
        ids = list(server_ids)
        inflight_fn = _rollouter_inflight(rollouter) if rollouter is not None else _default_inflight

    views = [
        RolloutReplicaView(
            replica_id=i,
            server_id=sid,
            version_fn=replica_version_fn,
            pull_fn=pull_replica_fn,
            inflight_fn=inflight_fn,
        )
        for i, sid in enumerate(ids)
    ]
    executor = FleetRepackExecutor(views, latest_version_fn=latest_version_fn)
    manager = RepackManager(engine=executor, config=config)
    return manager, executor


def make_repack_controller_actor():
    """Wrap the repack manager in a Ray actor: the trainer notifies on
    publish (fire-and-forget), the actor's event loop runs the manager."""
    import ray

    @ray.remote(num_cpus=1, max_concurrency=10)
    class RepackControllerActor:
        """Repack closed loop: manager task + fleet bridge over the relay
        controller. The trainer's ``_publish_versioned_weights`` tail calls
        ``notify_update``; the manager also checks periodically."""

        def __init__(self, relay_controller, rollouter=None, server_ids=None, config=None):
            self._manager, self._executor = build_repack_controller(
                relay_controller, rollouter=rollouter, server_ids=server_ids, config=config
            )
            self._manager.start()

        def notify_update(self) -> None:
            self._manager.notify_update()

        def snapshot(self) -> dict:
            snap = dict(self._manager.stats.snapshot())
            snap.update(self._executor.bridge_metrics())
            return snap

        async def stop(self) -> None:
            await self._manager.stop()

    return RepackControllerActor
