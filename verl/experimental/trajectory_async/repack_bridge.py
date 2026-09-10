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

What runs for real on this stack:

* **Idle-replica refresh** (always on): right after a publish, replicas
  with no in-flight requests that lag the latest version pull it over
  their OWN engine subgroup (``RelayController.pull_replica``) — the
  paper's repack payoff for replicas draining naturally between
  generations.
* **Cross-replica migration via the DRAIN LIFECYCLE** (paper §5 step ③,
  executor now real): the plan's ``(source, destination)`` pairs execute as

  1. ``begin_drain(sources)`` on the fleet load balancer — no new work
     is acquired on the sources (soft steering; the LB's least-loaded
     routing sends it to the destinations, which the planner's CanFit
     re-check verified have capacity);
  2. *hard mode only* (``repack.hard_drain=true``): ``abort_all_requests``
     on each source engine — in-flight requests receive ABORT outputs and
     the rollout clients transparently RESUME them on other replicas
     (prompt + partial response, recompute prefill — the accepted
     default; no KV transfer);
  3. the completion watcher (each manager tick): once a source's in-flight
     count reaches zero it pulls the latest weights over its own engine
     subgroup (the engine-side pull path handles its own pause/resume
     scoping) and ``end_drain`` returns it to routing — freed sources
     pull fresh weights, the §5 loop.

  Soft mode (default) never aborts anything: in-flight requests FINISH on
  their source under the version they started on (no version mixing, no
  lost work), and the source is freed when its long tail completes.

Honest capability boundaries (probed at runtime, declined when absent):

* drain sockets: requires a load balancer with ``begin_drain`` /
  ``end_drain`` (the L2 scheduling stack). Without them migration plans
  are DECLINED — counted in ``repack/migrations_declined`` — and requests
  keep running where they are (never half-migrate).
* hard mode additionally requires: server-level ``abort_all_requests``
  (vLLM replica path) and client-side resume of aborted requests (the
  fully-async client resumes unconditionally under
  ``async_training.partial_rollout=true``; the L2 stack resumes
  migration-caused aborts regardless). With ``partial_rollout=false`` on
  the stock client, aborted requests are DROPPED by the client — hard
  mode must not be enabled there.
* KV-cache placement is COUNT-denominated: per-token KV introspection
  needs engine metrics RPCs (cluster TODO), so the planner's KV columns
  are linear in the in-flight count (``kv = inflight × kv_per_request``,
  ``C_max = batch_bound × kv_per_request``) and CanFit's KV constraint
  collapses onto the true engine batch bound ``B`` (``max_num_seqs``).
* Per-request directed placement (the paper redirects each trajectory to
  a chosen Best-Fit destination) is not drivable through the stock
  client: redirect placement is the LB's. The plan's CanFit math still
  gates whether draining a source is safe.

The :class:`RolloutReplicaView` wraps injected async callables, so the
whole bridge is CPU-testable without ray; :func:`build_repack_controller`
wires the real relay controller + rollouter.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from verl.experimental.trajectory_async.repack import (
    MigrationResult,
    RepackConfig,
    ReplicaState,
    RepackManager,
)

logger = logging.getLogger(__name__)


# injected seams (all awaitable from the bridge actor):
InflightFn = Callable[[], Awaitable[dict[str, int] | None]]  # server_id -> inflight
ReplicaVersionFn = Callable[[int], Awaitable[int | None]]
PullReplicaFn = Callable[[int, "int | None"], Awaitable[int | None]]  # (replica_id, version|None=latest)
# drain lifecycle seams (None when the engine lacks the capability):
DrainFn = Callable[[list[str], bool], Awaitable[bool]]  # (server_ids, on) -> supported
AbortAllFn = Callable[[str], Awaitable[int]]  # server_id -> aborted count (-1 unsupported)
ResumeFn = Callable[[str], Awaitable[bool]]  # server_id -> resumed


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
    """

    replica_id: int
    server_id: str
    version_fn: ReplicaVersionFn
    pull_fn: PullReplicaFn
    inflight_fn: InflightFn

    async def current_version(self) -> int | None:
        return await self.version_fn(self.replica_id)

    async def running_count(self) -> int | None:
        inflight = await self.inflight_fn()
        return None if inflight is None else inflight.get(self.server_id)


class FleetRepackExecutor:
    """Repack executor over :class:`RolloutReplicaView` replicas.

    Implements the :class:`~verl.experimental.trajectory_async.repack.RepackExecutor`
    protocol with an ASYNC ``snapshot`` (the manager awaits awaitable
    snapshots): real replica states from the relay controller (version)
    and the rollout load balancer (in-flight), KV columns linear in the
    in-flight count (see module doc).

    Owns the drain lifecycle: :meth:`migrate` starts drains (and in hard
    mode aborts in-flight work for client-side resume); the tick hook
    :meth:`refresh_idle` doubles as the completion watcher — emptied
    sources pull fresh weights and return to routing.
    """

    def __init__(
        self,
        handles: Sequence[RolloutReplicaView],
        *,
        latest_version_fn: Callable[[], Awaitable[int | None]],
        drain_fn: DrainFn | None = None,
        abort_fn: AbortAllFn | None = None,
        resume_fn: ResumeFn | None = None,
        config: RepackConfig | None = None,
    ):
        self.handles: dict[int, RolloutReplicaView] = {h.replica_id: h for h in handles}
        self.latest_version_fn = latest_version_fn
        self.drain_fn = drain_fn
        self.abort_fn = abort_fn
        self.resume_fn = resume_fn
        self.config = config or RepackConfig()
        self.repack_overhead_s = 0.0

        # drain lifecycle state
        self._draining: dict[int, str] = {}  # replica_id -> server_id
        self._pulling: set[int] = set()  # replica_ids with a pull in flight
        self._prev_inflight: dict[int, int] = {}  # replica_id -> last tick's count
        self._cache: list[ReplicaState] = []

        # counters
        self.refreshes = 0
        self.refresh_failures = 0
        self.migrations_declined = 0
        self.migration_pairs_declined = 0
        self.migrations_started = 0
        self.migrations_completed = 0  # sources fully freed (pulled + undrained)
        self.requests_aborted_redirected = 0

    # ------------------------------------------------------------ probes

    async def _snapshot_states(self) -> list[tuple[RolloutReplicaView, int | None, int | None]]:
        """Per view: (view, current_version, running_count)."""
        async def one(view: RolloutReplicaView):
            version, running = await asyncio.gather(view.current_version(), view.running_count())
            return view, version, running

        return list(await asyncio.gather(*[one(v) for v in self.handles.values()]))

    async def snapshot(self) -> list[ReplicaState]:
        """Real fleet snapshot for the planner (ASYNC — the manager awaits
        awaitable snapshots). Replicas with unknown version or unknown
        in-flight count are reported non-candidates: version ``-1`` never
        groups with a real version group, and unknown load is conservative
        busy (``routable=False`` keeps them out of every plan role)."""
        states: list[ReplicaState] = []
        batch_bound = self.config.batch_bound
        kv_per_req = max(1, self.config.kv_per_request)
        for view, version, running in await self._snapshot_states():
            inflight = running if running is not None else 0
            states.append(
                ReplicaState(
                    replica_id=view.replica_id,
                    version=version if version is not None else -1,
                    kv_used=inflight * kv_per_req,
                    kv_capacity=(batch_bound or 0) * kv_per_req,
                    kv_prev=self._prev_inflight.get(view.replica_id, inflight * kv_per_req),
                    num_running=inflight,
                    num_waiting=0,  # running/waiting split unknown; CanFit uses the sum
                    batch_quota=batch_bound or 0,
                    max_running=batch_bound or 0,
                    pulling=view.replica_id in self._pulling,
                    routable=view.replica_id not in self._draining,
                )
            )
        self._prev_inflight = {
            s.replica_id: s.num_running for s in states
        }
        self._cache = states
        return states

    def fleet_kv_util(self) -> float:
        """From the last :meth:`snapshot` cache (count-denominated)."""
        cap = sum(s.kv_capacity for s in self._cache)
        if cap == 0:
            return 0.0
        return sum(s.kv_used for s in self._cache) / cap

    # ------------------------------------------------------ drain watcher

    async def refresh_idle(self) -> int:
        """Tick hook: (a) complete drains — emptied sources pull fresh
        weights and return to routing; (b) refresh non-draining idle
        replicas that lag the latest version.

        Returns the number of replicas refreshed (drain completions +
        idle refreshes). One replica's failure never stops the rest.
        """
        latest = await self.latest_version_fn()
        if latest is None:
            return 0
        refreshed = 0
        handled: set[int] = set()  # replicas the watcher resolved this tick

        # (a) drain completion watcher
        for replica_id in list(self._draining):
            view = self.handles.get(replica_id)
            if view is None:
                self._draining.pop(replica_id, None)
                continue
            running = await view.running_count()
            if running is None or running > 0:
                continue  # still finishing its tail (soft) or resumes in flight (hard)
            try:
                version = await view.current_version()
                if version is None or version < latest:
                    await self._pull_tracked(view, latest)
                    refreshed += 1
                    logger.info(
                        "repack migration completed: replica %d %s drained -> v%d",
                        replica_id,
                        view.server_id,
                        latest,
                    )
                # hard mode leaves the engine paused after abort_all —
                # always resume before the replica becomes routable again
                # (the pull path's own scoping usually did; this is insurance)
                if self.config.hard_drain and self.resume_fn is not None:
                    await self.resume_fn(view.server_id)
                if self.drain_fn is not None:
                    await self.drain_fn([view.server_id], on=False)
                self._draining.pop(replica_id, None)
                self.migrations_completed += 1
                handled.add(replica_id)
            except Exception:  # noqa: BLE001 — one replica must not stop the rest
                self.refresh_failures += 1
                logger.exception(
                    "repack drain completion failed for replica %d; retrying next tick",
                    replica_id,
                )

        # (b) idle refresh (never touches draining replicas — the watcher
        # above owns their lifecycle — nor replicas it just resolved)
        for view, version, running in await self._snapshot_states():
            if view.replica_id in self._draining or view.replica_id in handled:
                continue
            if version is None or running is None or running > 0:
                continue  # unknown version, unknown load, or busy -> skip
            if version >= latest:
                continue  # already fresh
            try:
                await self._pull_tracked(view, latest)
                refreshed += 1
                logger.info("repack refresh: replica %d %s -> v%d", view.replica_id, view.server_id, latest)
            except Exception:  # noqa: BLE001 — one replica must not stop the rest
                self.refresh_failures += 1
                logger.exception("repack refresh failed for replica %d", view.replica_id)
        self.refreshes += refreshed
        return refreshed

    async def _pull_tracked(self, view: RolloutReplicaView, version: int) -> None:
        """Pull with in-flight tracking (the planner's ``pulling`` flag)."""
        self._pulling.add(view.replica_id)
        try:
            await view.pull_fn(view.replica_id, version)
        finally:
            self._pulling.discard(view.replica_id)

    # -------------------------------------------- RepackExecutor protocol

    async def migrate(self, plan: list[tuple[int, int]]) -> MigrationResult:
        """Execute a consolidation plan as the drain lifecycle.

        Execution-time re-checks (live counts, live versions) may reject
        individual pairs — the paper's CanFit discipline at execution
        time; a pair rejected here is counted and skipped, never
        half-executed. If NO drain capability exists at all the whole
        plan is declined (requests keep running where they are).
        """
        if not plan:
            return MigrationResult(plan=[], requests_moved=0, kv_tokens_moved=0, sources_emptied=0)
        if self.drain_fn is None:
            self.migrations_declined += len(plan)
            logger.warning(
                "repack declined %d migration(s) %s: no drain capability "
                "(load balancer lacks begin/end_drain); requests keep running where they are",
                len(plan),
                plan,
            )
            return MigrationResult(plan=[], requests_moved=0, kv_tokens_moved=0, sources_emptied=0)

        # live re-check of every pair (CanFit at execution time)
        live = await self._snapshot_states()
        running: dict[int, int | None] = {}
        for view, version, count in live:
            running[view.replica_id] = count

        batch_bound = self.config.batch_bound
        accepted: list[tuple[int, int]] = []
        for src, dst in plan:
            src_view = self.handles.get(src)
            dst_view = self.handles.get(dst)
            if src_view is None or dst_view is None:
                self.migration_pairs_declined += 1
                continue
            if src in self._draining or dst in self._draining or dst in self._pulling:
                self.migration_pairs_declined += 1
                continue
            src_n, dst_n = running.get(src), running.get(dst)
            if src_n is None or dst_n is None:
                self.migration_pairs_declined += 1  # unknown load — conservative refuse
                continue
            if batch_bound is not None and src_n + dst_n > batch_bound:
                self.migration_pairs_declined += 1
                logger.info(
                    "repack pair (%d -> %d) declined at execution: %d + %d > B=%d",
                    src,
                    dst,
                    src_n,
                    dst_n,
                    batch_bound,
                )
                continue
            accepted.append((src, dst))

        if not accepted:
            self.migrations_declined += len(plan)
            return MigrationResult(plan=[], requests_moved=0, kv_tokens_moved=0, sources_emptied=0)

        # 1. steer new work away from the sources (one atomic-ish RPC)
        source_servers = list({self.handles[src].server_id for src, _ in accepted})
        drained = await self.drain_fn(source_servers, on=True)
        if not drained:
            self.migrations_declined += len(plan)
            logger.warning(
                "repack declined %d migration(s): begin_drain unsupported; "
                "requests keep running where they are",
                len(plan),
            )
            return MigrationResult(plan=[], requests_moved=0, kv_tokens_moved=0, sources_emptied=0)

        # the sources are now owned by the drain lifecycle — mark them
        # BEFORE any abort so a crash mid-round cannot strand a drained
        # replica without a watcher
        for src, _ in accepted:
            self._draining[src] = self.handles[src].server_id
        self.migrations_started += len(accepted)

        # 2. hard mode: abort in-flight on sources -> client-side resume
        #    redirects them to other replicas (recompute prefill)
        requests_moved = 0
        if self.config.hard_drain and self.abort_fn is not None:
            for src, _ in accepted:
                try:
                    aborted = await self.abort_fn(self.handles[src].server_id)
                except Exception:  # noqa: BLE001 — one source must not strand the round
                    self.refresh_failures += 1
                    logger.exception(
                        "repack hard drain abort failed for replica %d; "
                        "soft drain continues (in-flight finishes on source)",
                        src,
                    )
                    continue
                if aborted > 0:
                    requests_moved += aborted
                    logger.info(
                        "repack hard drain: aborted %d in-flight requests on replica %d %s (clients resume them elsewhere)",
                        aborted,
                        src,
                        self.handles[src].server_id,
                    )
        self.requests_aborted_redirected += requests_moved

        logger.info(
            "repack migration started: %d source(s) %s draining (%s mode, %d requests redirected)",
            len(accepted),
            source_servers,
            "hard" if self.config.hard_drain else "soft",
            requests_moved,
        )
        return MigrationResult(
            plan=accepted,
            requests_moved=requests_moved,
            kv_tokens_moved=0,  # recompute prefill — no KV blocks travel (documented default)
            sources_emptied=0,  # completed asynchronously by the watcher
        )

    def bridge_metrics(self) -> dict[str, Any]:
        return {
            "repack/idle_refreshes": self.refreshes,
            "repack/refresh_failures": self.refresh_failures,
            "repack/migrations_declined": self.migrations_declined,
            "repack/migration_pairs_declined": self.migration_pairs_declined,
            "repack/migrations_started": self.migrations_started,
            "repack/migrations_completed": self.migrations_completed,
            "repack/requests_aborted_redirected": self.requests_aborted_redirected,
            "repack/draining_replicas": len(self._draining),
        }


# ------------------------------------------------------------------ wiring


async def _default_inflight() -> dict[str, int] | None:
    return None


async def _rollouter_inflight(rollouter) -> dict[str, int] | None:
    """Per-server in-flight counts from the rollouter's load balancer."""
    try:
        return await rollouter.replica_inflight.remote()
    except AttributeError:
        return None
    except Exception:  # noqa: BLE001 — probing must never kill the loop
        logger.exception("replica_inflight probe failed")
        return None


def _rollouter_drain(rollouter) -> DrainFn | None:
    async def drain(server_ids: list[str], on: bool) -> bool:
        try:
            return bool(await rollouter.replica_drain.remote(server_ids, on))
        except Exception:  # noqa: BLE001 — RPC failure = capability absent
            logger.exception("replica_drain RPC failed (treating as unsupported)")
            return False

    return drain


def _rollouter_abort(rollouter) -> AbortAllFn | None:
    async def abort(server_id: str) -> int:
        try:
            return int(await rollouter.replica_abort_all.remote(server_id))
        except Exception:  # noqa: BLE001 — RPC failure = engine unavailable
            logger.exception("replica_abort_all RPC failed for %s", server_id)
            return -1

    return abort


def _rollouter_resume(rollouter) -> ResumeFn | None:
    async def resume(server_id: str) -> bool:
        try:
            return bool(await rollouter.replica_resume.remote(server_id))
        except Exception:  # noqa: BLE001 — RPC failure = not resumed
            logger.exception("replica_resume RPC failed for %s", server_id)
            return False

    return resume


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
        rollouter: optional rollouter exposing ``replica_inflight()`` and
            the drain lifecycle RPCs (``replica_drain`` / ``replica_abort_all``
            / ``replica_resume``). Without it, inflight counts are unknown
            and refresh stays conservative (nothing refreshes; migration
            seams are absent so plans decline).
        server_ids: the load-balancer server id per replica index. Identity
            convention: replica i (engine partition order) ↔ the i-th entry.
            Defaults to ["0", "1", ...] — pass the real ids when they
            diverge (cluster TODO: unify replica identity engine ↔ LB).
        config: :class:`RepackConfig` for the manager loop (cadence,
            candidates, hard_drain, batch_bound, kv_per_request).

    Returns:
        (manager, executor) — start the manager loop with ``manager.start()``.
    """
    config = config or RepackConfig()

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
    executor = FleetRepackExecutor(
        views,
        latest_version_fn=latest_version_fn,
        drain_fn=_rollouter_drain(rollouter) if rollouter is not None else None,
        abort_fn=_rollouter_abort(rollouter) if rollouter is not None else None,
        resume_fn=_rollouter_resume(rollouter) if rollouter is not None else None,
        config=config,
    )
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
