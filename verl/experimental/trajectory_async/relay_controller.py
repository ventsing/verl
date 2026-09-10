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

P0 topology (documented, not hidden): the stock kimi engine group spans
actor + ALL rollout ranks and ``receive_tensor`` barriers on it, so pulls
are fleet-synchronized — every replica pulls the same version together,
at a moment the rollout side chooses. With one rollout replica this IS
per-replica anytime pulling; per-replica process groups (true
per-replica, any-version pulls) are the remaining cluster item
(README TODO). All engine-group traffic is serialized behind one lock
until then — concurrent collectives on the same group would interleave
barriers and corrupt.

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

# injected remote-operation signatures (all awaitable, return None):
PublishFn = Callable[[int], Awaitable[None]]
PullFn = Callable[[int], Awaitable[None]]
UnstageFn = Callable[[int], Awaitable[None]]


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
    ) -> None:
        self._publish_fn = publish_fn
        self._pull_fn = pull_fn
        self._unstage_fn = unstage_fn
        self.keep_last = max(1, int(keep_last))

        # version -> {"published_s": float, "retired": bool}
        self._versions: dict[int, dict[str, Any]] = {}
        self._last_pulled: int | None = None
        self._publishes = 0
        self._pulls = 0
        self._publish_seconds = 0.0
        self._pull_seconds = 0.0
        self._last_publish_at = 0.0
        self._last_pull_at = 0.0
        # one engine process group behind everything: serialize all traffic
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- publish

    async def publish(self, version: int) -> dict[str, Any]:
        """Stage ``version``; the only trainer-side stall on the weight path.

        Idempotent per version (re-publishing a version is a no-op) and
        retains only the last ``keep_last`` versions.
        """
        async with self._lock:
            if version in self._versions and not self._versions[version]["retired"]:
                logger.warning("version %s already staged; ignoring duplicate publish", version)
                return self.snapshot()

            start = time.monotonic()
            await self._publish_fn(version)
            elapsed = time.monotonic() - start
            self._versions[version] = {"published_s": elapsed, "retired": False}
            self._publishes += 1
            self._publish_seconds += elapsed
            self._last_publish_at = time.time()

            await self._retire_old_versions()

            logger.info(
                "relay publish v%d: staged in %.3fs (latest pullable: %s)",
                version,
                elapsed,
                self.latest_version,
            )
            return self.snapshot()

    async def _retire_old_versions(self) -> None:
        """Retire staged versions beyond ``keep_last`` (frees the actors'
        pinned per-version CPU shard copies). Never retires the latest."""
        live = sorted(v for v, info in self._versions.items() if not info["retired"])
        for version in live[: max(0, len(live) - self.keep_last)]:
            await self._unstage_fn(version)
            self._versions[version]["retired"] = True
            logger.info("relay retire v%d (keep_last=%d)", version, self.keep_last)

    # ---------------------------------------------------------------- pull

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
        async with self._lock:
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
            self._pulls += 1
            self._pull_seconds += elapsed
            self._last_pull_at = time.time()
            logger.info("relay pull v%d: loaded in %.3fs", target, elapsed)
            return target

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
            "relay/publish_seconds_total": self._publish_seconds,
            "relay/pull_seconds_total": self._pull_seconds,
            "relay/publish_seconds_last": self._versions[latest]["published_s"] if latest is not None else 0.0,
        }
        if self._last_pulled is not None and latest is not None:
            snap["relay/fleet_lag_versions"] = latest - self._last_pulled
        return snap


# ------------------------------------------------------------------ wiring


def build_relay_controller(
    checkpoint_manager,
    keep_last: int = 2,
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
    # mirrors CheckpointEngineManager.update_weights step 4, minus finalize:
    # prepare + topology + init_process_group once for the controller's life
    checkpoint_manager.build_process_group(rollout_wg)

    async def publish_fn(version: int) -> None:
        # fire BOTH sides before gathering: gather_metas is collective, the
        # actor stage and the rollout gather must run concurrently
        refs = actor_wg.stage_weights_version(version) + rollout_wg.gather_version_metas(version)
        ray.get(refs)

    async def pull_fn(version: int) -> None:
        # stock sequencing (CheckpointEngineManager.update_weights steps 1/3/5):
        # abort in-flight requests, free kv_cache, load, resume — but timed by
        # the ROLLOUT side, not the trainer
        await asyncio.gather(*[replica.abort_all_requests() for replica in replicas])
        await asyncio.gather(*[replica.release_kv_cache() for replica in replicas])
        ray.get(rollout_wg.pull_weights_version(version))
        await asyncio.gather(*[replica.resume_kv_cache() for replica in replicas])
        await asyncio.gather(*[replica.resume_generation() for replica in replicas])

    async def unstage_fn(version: int) -> None:
        ray.get(
            actor_wg.execute_checkpoint_engine(
                method=["unstage_version"] * actor_wg.world_size, version=[version] * actor_wg.world_size
            )
            + rollout_wg.execute_checkpoint_engine(
                method=["drop_version"] * rollout_wg.world_size, version=[version] * rollout_wg.world_size
            )
        )

    return RelayController(publish_fn=publish_fn, pull_fn=pull_fn, unstage_fn=unstage_fn, keep_last=keep_last)


def make_relay_controller_actor():
    """Wrap :class:`RelayController` in a Ray actor class (import-time ray
    dependency isolated here so the controller itself stays CPU-testable)."""
    import ray

    @ray.remote(num_cpus=1, max_concurrency=10)
    class RelayControllerActor:
        """Ray-exposed relay controller; the trainer publishes, the rollout
        side pulls (see ``rollout_producer.py``)."""

        def __init__(self, checkpoint_manager, keep_last: int = 2):
            self._controller = build_relay_controller(checkpoint_manager, keep_last=keep_last)

        async def publish(self, version: int) -> dict:
            return await self._controller.publish(version)

        async def pull(self, version: int | None = None) -> int | None:
            return await self._controller.pull(version)

        def behind_latest(self, current: int) -> bool:
            return self._controller.behind_latest(current)

        def latest_version(self) -> int | None:
            return self._controller.latest_version

        def snapshot(self) -> dict:
            return self._controller.snapshot()

    return RelayControllerActor
