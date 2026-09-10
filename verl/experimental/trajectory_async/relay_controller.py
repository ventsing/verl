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
"""Ray-native control plane for the multi-version, pull-based weight path.

The CPU-side tier (:mod:`verl.experimental.trajectory_async.relay_tier`)
orchestrates relay backends that hold the tensors themselves. On a real
cluster the tensors never leave the worker processes: the actor ranks'
checkpoint engines hold the registered CPU shards (kimi P2P store), and
"the relay" is *where those engines live plus this controller's
bookkeeping*. This module is that driver-side piece — the Ray twin of
``RelayService``:

* ``publish(version)`` — the ONE synchronized phase: actor ranks
  ``stage_weights_version`` (offload + register + gather metas, no tensor
  push) while rollout ranks ``gather_version_metas`` concurrently. This
  single metadata gather is the whole trainer-side stall; the version then
  stays pullable until retention retires it.
* ``pull(version=None)`` — rollout-driven: abort in-flight requests, free
  kv_cache, ``pull_weights_version`` into every replica, resume. The
  rollout side calls this at ITS batch boundaries (see
  ``rollout_producer.py``), so weight sync is decoupled from training
  progress — the Laminar no-lockstep property.
* ``pull_replica(replica_id, version=None)`` — the per-replica variant:
  with a replica partition installed in the engine (one process-group
  subgroup per replica), only that replica aborts / pulls / resumes over
  its subgroup; every OTHER replica keeps generating. The repack path
  uses this to refresh idle replicas to the fresh version right after a
  publish (paper §5 "freed sources pull fresh weights").

Locking: a fleet pull or publish acquires ALL per-replica locks (in
replica-id order — deadlock-free); a per-replica pull acquires only its
own lock, so pulls of distinct replicas proceed concurrently. A
per-replica pull never re-enters fleet collectives (its subgroup
barriers touch only its own ranks).

Host-memory quota (actor pinned shards): each staged version pins a full
CPU copy of the actor shards on every actor rank; ``publish`` records the
staged bytes (the engine returns per-rank sizes) and retires the oldest
live versions — beyond ``keep_last`` AND beyond ``max_staged_bytes`` —
never the latest.

The pure-logic class (:class:`RelayController`) takes injected async
callables, so the versioning/retention/metrics behavior is CPU-testable
without ray; :func:`build_relay_controller` wires the real worker groups.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# injected remote-operation signatures:
PublishFn = Callable[[int], Awaitable[dict | None]]  # -> {"staged_bytes": int} | None
PullFn = Callable[[int], Awaitable[None]]
PullReplicaFn = Callable[[int, int], Awaitable[None]]  # (replica_id, version)
UnstageFn = Callable[[int], Awaitable[None]]


def derive_replica_partition(actor_world_size: int, replicas: list) -> list[list[int]] | None:
    """Derive the engine replica partition from the replica list (pure).

    Rollout engine ranks are assigned in the worker flatten order
    ``[r.workers for r in replicas]`` (the same order
    ``build_replica_worker_group`` uses), so each replica's block is
    contiguous in global engine coordinates starting at
    ``actor_world_size``. Returns None when fewer than 2 worker-bearing
    replicas exist (one block == the fleet — no subgroups needed).
    """
    parts: list[list[int]] = []
    next_rank = actor_world_size
    for replica in replicas:
        workers = list(getattr(replica, "workers", None) or [])
        if not workers:
            continue
        parts.append(list(range(next_rank, next_rank + len(workers))))
        next_rank += len(workers)
    return parts if len(parts) >= 2 else None


class RelayController:
    """Version registry + retention + pull driver for the versioned weight path.

    Args:
        publish_fn: stage ``version`` on the actor ranks (concurrent rollout
            metadata gather happens inside the wired implementation).
        pull_fn: load ``version`` into every rollout replica (with the stock
            abort → release kv → pull → resume sequencing).
        unstage_fn: retire ``version`` (actor unregister + rollout snapshot drop).
        keep_last: retention — how many recent versions stay pullable. Each
            staged version pins a full CPU copy of the actor shards on every
            actor rank, so this bounds that memory.
    """

    def __init__(
        self,
        publish_fn: PublishFn,
        pull_fn: PullFn,
        unstage_fn: UnstageFn,
        keep_last: int = 2,
        pull_replica_fn: PullReplicaFn | None = None,
        num_replicas: int = 0,
        max_staged_bytes: int | None = None,
    ) -> None:
        self._publish_fn = publish_fn
        self._pull_fn = pull_fn
        self._unstage_fn = unstage_fn
        self.keep_last = max(1, int(keep_last))
        self._pull_replica_fn = pull_replica_fn
        self._num_replicas = num_replicas
        self.max_staged_bytes = max_staged_bytes

        # version -> {"published_s": float, "retired": bool, "staged_bytes": int}
        self._versions: dict[int, dict[str, Any]] = {}
        self._last_pulled: int | None = None
        self._replica_versions: dict[int, int] = {}
        self._publishes = 0
        self._pulls = 0
        self._replica_pulls = 0
        self._quota_retires = 0
        self._publish_seconds = 0.0
        self._pull_seconds = 0.0
        self._last_publish_at = 0.0
        self._last_pull_at = 0.0
        # per-replica locks (acquired in replica-id order when multiple are
        # needed — deadlock-free); fleet ops take them ALL, per-replica ops
        # take their own, so distinct replicas pull concurrently. On the
        # flat path (no replica partition) a single global lock keeps the
        # historical strict publish/pull serialization.
        self._replica_locks: dict[int, asyncio.Lock] = {i: asyncio.Lock() for i in range(num_replicas)}
        self._flat_lock = asyncio.Lock()

    def _lock_all(self) -> list[asyncio.Lock]:
        """All replica locks in stable id order (fleet operations); the
        single global lock on the flat path."""
        if not self._replica_locks:
            return [self._flat_lock]
        return [self._replica_locks[i] for i in sorted(self._replica_locks)]

    async def _acquire_all(self) -> list[asyncio.Lock]:
        locks = self._lock_all()
        for lock in locks:
            await lock.acquire()
        return locks

    def _release_all(self, locks: list[asyncio.Lock]) -> None:
        for lock in locks:
            lock.release()

    # ------------------------------------------------------------- publish

    async def publish(self, version: int) -> dict[str, Any]:
        """Stage ``version``; the only trainer-side stall on the weight path.

        Idempotent per version (re-publishing a version is a no-op) and
        retains only the last ``keep_last`` versions within the
        ``max_staged_bytes`` host-memory quota (actor pinned shards).
        """
        locks = await self._acquire_all()
        try:
            if version in self._versions and not self._versions[version]["retired"]:
                logger.warning("version %s already staged; ignoring duplicate publish", version)
                return self.snapshot()

            start = time.monotonic()
            result = await self._publish_fn(version)
            elapsed = time.monotonic() - start
            staged_bytes = int((result or {}).get("staged_bytes", 0))
            self._versions[version] = {
                "published_s": elapsed,
                "retired": False,
                "staged_bytes": staged_bytes,
            }
            self._publishes += 1
            self._publish_seconds += elapsed
            self._last_publish_at = time.time()

            await self._retire_old_versions()

            logger.info(
                "relay publish v%d: staged in %.3fs, %.2f GiB pinned (latest pullable: %s)",
                version,
                elapsed,
                staged_bytes / (1 << 30),
                self.latest_version,
            )
            return self.snapshot()
        finally:
            self._release_all(locks)

    async def _retire_old_versions(self) -> None:
        """Retire staged versions beyond ``keep_last`` AND beyond the
        ``max_staged_bytes`` host-memory quota (each staged version pins a
        full CPU copy of the actor shards on every actor rank). Never
        retires the latest, whatever the quota says."""
        async def retire(version: int, why: str, *, quota: bool = False) -> None:
            await self._unstage_fn(version)
            self._versions[version]["retired"] = True
            if quota:
                self._quota_retires += 1
            logger.info("relay retire v%d (%s)", version, why)

        live = sorted(v for v, info in self._versions.items() if not info["retired"])
        for version in live[: max(0, len(live) - self.keep_last)]:
            await retire(version, f"keep_last={self.keep_last}")

        if self.max_staged_bytes is None:
            return
        # byte-quota enforcement: retire oldest-first while over quota and
        # more than one live version remains (the latest always survives)
        while len([v for v, i in self._versions.items() if not i["retired"]]) > 1:
            live = sorted(v for v, info in self._versions.items() if not info["retired"])
            total = sum(self._versions[v]["staged_bytes"] for v in live)
            if total <= self.max_staged_bytes:
                break
            await retire(live[0], f"max_staged_bytes={self.max_staged_bytes}", quota=True)

    # ---------------------------------------------------------------- pull

    def recover(self, latest_version: int, replica_versions: dict[int, int] | None = None) -> None:
        """Master-failover state recovery (§4.3): a resurrected controller
        rebuilds its registry from the authority that survives it — the
        trainer knows the published version; engines keep their staged
        shards. The synthetic ``_versions`` entry carries ``recovered``
        (staged_bytes=0 — quota retirement under-counts until the next
        publish re-stages, the conservative direction: over-retention is
        bounded by ``keep_last``). Per-replica versions default unknown:
        the next ``pull_replica`` re-syncs each one from engine truth."""
        self._versions = {
            latest_version: {
                "staged_bytes": 0,
                "staged_params": 0,
                "retired": False,
                "recovered": True,
            }
        }
        self._replica_versions = dict(replica_versions) if replica_versions else {}
        self.recoveries = getattr(self, "recoveries", 0) + 1
        logger.warning(
            "relay controller recovered: latest=v%s, %d/%d replica versions known",
            latest_version,
            len(self._replica_versions),
            self._num_replicas,
        )

    @property
    def latest_version(self) -> int | None:
        live = [v for v, info in self._versions.items() if not info["retired"]]
        return max(live) if live else None

    async def pull(self, version: int | None = None) -> int | None:
        """Load weights into the rollout fleet; rollout-side driven.

        Args:
            version: the version to pull; ``None`` = the latest complete
                (pulling an older live version is allowed — e.g. a replica
                class that deliberately trains a step behind).

        Returns:
            The version actually pulled, or ``None`` if nothing is staged yet.
        """
        locks = await self._acquire_all()
        try:
            target = version if version is not None else self.latest_version
            if target is None:
                return None
            info = self._versions.get(target)
            if info is None or info["retired"]:
                raise LookupError(
                    f"version {target} is not pullable (staged: "
                    f"{sorted(v for v, i in self._versions.items() if not i['retired'])})"
                )

            start = time.monotonic()
            await self._pull_fn(target)
            elapsed = time.monotonic() - start
            self._last_pulled = target
            self._replica_versions = {i: target for i in range(self._num_replicas)}
            self._pulls += 1
            self._pull_seconds += elapsed
            self._last_pull_at = time.time()
            logger.info("relay pull v%d: loaded in %.3fs", target, elapsed)
            return target
        finally:
            self._release_all(locks)

    @property
    def supports_pull_replica(self) -> bool:
        """Whether per-replica pulls are wired (engine replica partition
        installed — needs >=2 worker-bearing rollout replicas; a
        single-replica topology is its own fleet, so scoped pulls are
        meaningless there and the wiring is absent by design)."""
        return self._pull_replica_fn is not None

    async def pull_replica(self, replica_id: int, version: int | None = None) -> int | None:
        """Load weights into ONE replica (per-replica, any-time pull).

        Requires the per-replica wiring (engine replica partition +
        ``pull_replica_fn``); only that replica aborts / pulls over its own
        subgroup / resumes — every other replica keeps generating. The
        CALLER decides idleness (e.g. the repack path refreshes replicas
        with no in-flight requests right after a publish).
        """
        if self._pull_replica_fn is None:
            raise NotImplementedError(
                "per-replica pulls need the engine replica partition "
                "(multiple rollout replicas) and pull_replica_fn wiring"
            )
        if replica_id not in self._replica_locks:
            raise ValueError(f"replica_id {replica_id} out of range (0..{self._num_replicas - 1})")

        async with self._replica_locks[replica_id]:
            target = version if version is not None else self.latest_version
            if target is None:
                return None
            info = self._versions.get(target)
            if info is None or info["retired"]:
                raise LookupError(
                    f"version {target} is not pullable (staged: "
                    f"{sorted(v for v, i in self._versions.items() if not i['retired'])})"
                )
            if self._replica_versions.get(replica_id) == target:
                return target  # already there — idempotent no-op

            start = time.monotonic()
            await self._pull_replica_fn(replica_id, target)
            elapsed = time.monotonic() - start
            self._replica_versions[replica_id] = target
            self._replica_pulls += 1
            logger.info(
                "relay pull_replica %d v%d: loaded in %.3fs", replica_id, target, elapsed
            )
            return target

    def replica_version(self, replica_id: int) -> int | None:
        """The version a replica currently runs (fleet pulls set all
        replicas; per-replica pulls set just one)."""
        return self._replica_versions.get(replica_id)

    def behind_latest(self, current: int) -> bool:
        """Whether a replica running ``current`` should pull (fleet-level P0
        check: pulls are synchronized, so one number decides for all)."""
        latest = self.latest_version
        return latest is not None and current < latest

    # ------------------------------------------------------------ metrics

    def snapshot(self) -> dict[str, Any]:
        live = sorted(v for v, info in self._versions.items() if not info["retired"])
        latest = live[-1] if live else None
        snap = {
            "relay/versions_live": len(live),
            "relay/latest_version": latest,
            "relay/last_pulled_version": self._last_pulled,
            "relay/publishes": self._publishes,
            "relay/pulls": self._pulls,
            "relay/replica_pulls": self._replica_pulls,
            "relay/recoveries": getattr(self, "recoveries", 0),
            "relay/quota_retires": self._quota_retires,
            "relay/staged_bytes": sum(self._versions[v]["staged_bytes"] for v in live),
            "relay/publish_seconds_total": self._publish_seconds,
            "relay/pull_seconds_total": self._pull_seconds,
            "relay/publish_seconds_last": self._versions[latest]["published_s"] if latest is not None else 0.0,
        }
        if self.max_staged_bytes is not None:
            snap["relay/staged_bytes_limit"] = self.max_staged_bytes
        if self._last_pulled is not None and latest is not None:
            snap["relay/fleet_lag_versions"] = latest - self._last_pulled
        if self._replica_versions and latest is not None:
            lags = [latest - v for v in self._replica_versions.values()]
            snap["relay/replica_lag_versions_max"] = max(lags)
            snap["relay/replica_lag_versions_mean"] = sum(lags) / len(lags)
        return snap


# ------------------------------------------------------------------ wiring


def build_relay_controller(
    checkpoint_manager,
    keep_last: int = 2,
    max_staged_bytes: int | None = None,
) -> "RelayController":
    """Wire a :class:`RelayController` over a real ``CheckpointEngineManager``.

    Runs on the trainer (has the actor worker group and the rollout replica
    handles). Builds the combined rollout worker group and the engine process
    group ONCE — the versioned path reuses it for every publish/pull (the
    stock manager rebuilds per update; kimi's ``init_process_group`` is
    idempotent, guarded by ``initialized``).
    """
    import ray

    from verl.checkpoint_engine.base import build_replica_worker_group

    actor_wg = checkpoint_manager.actor_wg
    replicas = checkpoint_manager.replicas
    rollout_wg = build_replica_worker_group(replicas)
    # per-replica subgroups: with >=2 worker-bearing replicas, install one
    # engine process-group subgroup per replica so pulls can be scoped to a
    # single replica (rollout ranks keep generating on the others)
    replica_partition = derive_replica_partition(actor_wg.world_size, replicas)
    # mirrors CheckpointEngineManager.update_weights step 4, minus finalize:
    # prepare + topology + init_process_group once for the controller's life
    checkpoint_manager.build_process_group(rollout_wg, replica_partition=replica_partition)
    # one checkpoint-engine worker group PER replica for scoped dispatches
    replica_wgs = [build_replica_worker_group([replica]) for replica in replicas]

    async def publish_fn(version: int) -> dict:
        # fire BOTH sides before gathering: gather_metas is collective, the
        # actor stage and the rollout gather must run concurrently
        refs = actor_wg.stage_weights_version(version) + rollout_wg.gather_version_metas(version)
        results = ray.get(refs)
        # sum per-rank staged sizes for the host-memory quota accounting
        staged_bytes = 0
        for result in results:
            if isinstance(result, dict):
                staged_bytes += int(result.get("staged_bytes", 0))
        return {"staged_bytes": staged_bytes}

    async def pull_fn(version: int) -> None:
        # stock sequencing (CheckpointEngineManager.update_weights steps 1/3/5):
        # abort in-flight requests, free kv_cache, load, resume — but timed by
        # the ROLLOUT side, not the trainer
        await asyncio.gather(*[replica.abort_all_requests() for replica in replicas])
        await asyncio.gather(*[replica.release_kv_cache() for replica in replicas])
        ray.get(rollout_wg.pull_weights_version(version))
        await asyncio.gather(*[replica.resume_kv_cache() for replica in replicas])
        await asyncio.gather(*[replica.resume_generation() for replica in replicas])

    async def pull_replica_fn(replica_id: int, version: int) -> None:
        # scoped twin of pull_fn: only this replica aborts / pulls over its
        # OWN engine subgroup / resumes; the other replicas never stall
        if replica_partition is None:
            raise NotImplementedError(
                "per-replica pulls require >=2 worker-bearing rollout replicas "
                "(no replica partition was installed)"
            )
        replica = replicas[replica_id]
        await replica.abort_all_requests()
        await replica.release_kv_cache()
        ray.get(replica_wgs[replica_id].pull_weights_version(version, replica_id=replica_id))
        await replica.resume_kv_cache()
        await replica.resume_generation()

    async def unstage_fn(version: int) -> None:
        ray.get(
            actor_wg.execute_checkpoint_engine(
                method=["unstage_version"] * actor_wg.world_size, version=[version] * actor_wg.world_size
            )
            + rollout_wg.execute_checkpoint_engine(
                method=["drop_version"] * rollout_wg.world_size, version=[version] * rollout_wg.world_size
            )
        )

    return RelayController(
        publish_fn=publish_fn,
        pull_fn=pull_fn,
        unstage_fn=unstage_fn,
        keep_last=keep_last,
        pull_replica_fn=pull_replica_fn if replica_partition is not None else None,
        num_replicas=len(replicas),
        max_staged_bytes=max_staged_bytes,
    )


def make_relay_controller_actor():
    """Wrap :class:`RelayController` in a Ray actor class (import-time ray
    dependency isolated here so the controller itself stays CPU-testable)."""
    import ray

    @ray.remote(num_cpus=1, max_concurrency=10)
    class RelayControllerActor:
        """Ray-exposed relay controller; the trainer publishes, the rollout
        side pulls (see ``rollout_producer.py``), the repack path refreshes
        idle replicas per-replica right after a publish."""

        def __init__(self, checkpoint_manager, keep_last: int = 2, max_staged_bytes: int | None = None):
            self._controller = build_relay_controller(
                checkpoint_manager, keep_last=keep_last, max_staged_bytes=max_staged_bytes
            )

        async def publish(self, version: int) -> dict:
            return await self._controller.publish(version)

        async def pull(self, version: int | None = None) -> int | None:
            return await self._controller.pull(version)

        async def pull_replica(self, replica_id: int, version: int | None = None) -> int | None:
            return await self._controller.pull_replica(replica_id, version)

        def replica_version(self, replica_id: int) -> int | None:
            return self._controller.replica_version(replica_id)

        def supports_pull_replica(self) -> bool:
            return self._controller.supports_pull_replica

        def behind_latest(self, current: int) -> bool:
            return self._controller.behind_latest(current)

        def latest_version(self) -> int | None:
            return self._controller.latest_version

        def snapshot(self) -> dict:
            return self._controller.snapshot()

        def ping(self) -> bool:
            return True

        def recover(self, latest_version: int, replica_versions=None) -> None:
            self._controller.recover(latest_version, replica_versions)

    return RelayControllerActor
