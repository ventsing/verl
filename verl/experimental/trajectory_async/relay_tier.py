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
"""Hierarchical relay tier (Laminar §4) over the P2P weight backends.

The relay tier is a *distributed parameter service*: one relay process per
rollout machine, each hosting the latest actor weights in local (CPU)
P2P-readable memory. Weight flow (paper Fig. 6):

1. the actor stages a new version into the **master relay** (node 0) and
   immediately resumes training — the actor stall is this single hop;
2. the master-to-relays distribution runs **in the background**, chunk
   pipelined along a relay chain: relay ``i`` starts copying chunk ``k``
   as soon as relay ``i-1`` has it, so chain hops overlap and the
   broadcast time stays near-constant in chain length;
3. each rollout pulls from its **colocated** relay at any time (PCIe,
   sub-second) — a pull never waits for the broadcast to finish; it gets
   the newest *fully staged* version on its local relay.

Each relay node wraps one :class:`P2PWeightBackend` (kimi / mooncake /
fake) — on a real cluster that backend wraps the engine instance living
on that rollout machine, so relay memory is machine-local and consumers
read it over PCIe, not the actor's NIC.

Trainer-side hooks:

* ``format_fn(name, tensor) -> (name, tensor)`` — applied once at
  publish, before the master stages anything. On a real deployment this
  is where actor-internal params convert to **HF format** (verl's
  ``actor_wg.get_per_tensor_param`` iterator + the HF name mapping, the
  same export the checkpoint engines produce), so every replica can load
  weights with a stock ``from_pretrained``-style path.
* ``reshard_fn(name, tensor) -> (name, tensor) | None`` — master-side
  **resharding** (Laminar §4.2): convert trainer-coordinate tensors into
  the rollout's TP-sharded loadable layout, the same transform for every
  relay. Each machine's rollout loads its local TP ranks from the
  colocated relay. Returning ``None`` drops a tensor (never broadcast).

Everything here is orchestration: no engine imports, no cluster needed.
The :class:`FakeP2PBackend` exercises the full topology on CPU; the
kimi/mooncake adapters are the real transports, one backend per node.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from verl.experimental.trajectory_async.repack import (
    MigrationResult,
    RepackExecutor,
    ReplicaState,
)
from verl.experimental.trajectory_async.versioned_weight_store import (
    ConsumerState,
    P2PWeightBackend,
    WeightManifest,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RelayTierConfig",
    "RelayTierStats",
    "RelayNode",
    "RelayService",
    "RelayTierAdapter",
    "RolloutReplicaHandle",
    "RunningRequest",
    "RolloutRepackExecutor",
]


# ------------------------------------------------------------- relay tier


@dataclass
class RelayTierConfig:
    """Topology + retention knobs of the relay tier.

    ``hop_read_latency_s`` / ``local_read_latency_s`` are fake-transport
    hints (chain hop ≈ network, local pull ≈ PCIe); real backends ignore
    them — their latency is physical.
    """

    num_relays: int = 1  # one relay per rollout machine (node 0 = master)
    chunks: int = 4  # model split for pipelined chain broadcast
    keep_last: int = 2  # complete versions retained per relay node
    hop_read_latency_s: float = 0.15
    local_read_latency_s: float = 0.02

    def __post_init__(self) -> None:
        if self.num_relays < 1:
            raise ValueError("num_relays must be >= 1")
        if self.chunks < 1:
            raise ValueError("chunks must be >= 1")
        if self.keep_last < 1:
            raise ValueError("keep_last must be >= 1")


@dataclass
class RelayTierStats:
    publishes: int = 0
    local_pulls: int = 0
    hop_reads: int = 0
    hop_read_s: float = 0.0
    local_read_s: float = 0.0
    hop_bytes: int = 0
    local_bytes: int = 0
    actor_stall_total_s: float = 0.0
    pull_wait_s: float = 0.0
    chain_completion_s: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        chain = self.chain_completion_s
        return {
            "relay/publishes": self.publishes,
            "relay/actor_stall_total_s": round(self.actor_stall_total_s, 4),
            "relay/pulls": self.local_pulls,
            # a pull reads the local relay and normally never waits for the
            # chain broadcast (it takes the newest complete local version);
            # nonzero only at startup, before the first version finished
            # arriving on this node
            "relay/pull_wait_total_s": round(self.pull_wait_s, 4),
            "relay/pcie_total_s": round(self.local_read_s, 4),
            "relay/hop_total_s": round(self.hop_read_s, 4),
            "relay/hop_reads": self.hop_reads,
            "relay/hop_bytes": self.hop_bytes,
            "relay/chain_completion_s": {
                "rounds": len(chain),
                "mean": round(sum(chain) / len(chain), 4) if chain else None,
                "max": round(max(chain), 4) if chain else None,
            },
        }


def _as_aiter(items: list[tuple[str, Any]]):
    async def gen():
        for item in items:
            yield item

    return gen()


class RelayNode:
    """One relay process: its own backend + the staged chunk grid.

    A version ``v`` is a grid of ``chunks`` manifests named
    ``{base}:v{v}:c{k}``. A chunk that resharding filtered to empty is
    tracked as a zero-byte manifest (never hits the backend) so version
    completeness is uniform across nodes.
    """

    def __init__(
        self,
        node_id: int,
        backend: P2PWeightBackend,
        *,
        base_name: str = "actor",
        keep_last: int = 2,
    ) -> None:
        self.node_id = node_id
        self.backend = backend
        self.base_name = base_name
        self.keep_last = keep_last
        self._chunks: dict[int, dict[int, WeightManifest]] = {}
        self._expected: dict[int, int] = {}
        self._waiters: dict[tuple[int, int], list[asyncio.Future]] = {}
        self._version_done: dict[int, asyncio.Event] = {}

    # ------------------------------------------------------------ naming

    def chunk_name(self, version: int, chunk: int) -> str:
        return f"{self.base_name}:v{version}:c{chunk}"

    # --------------------------------------------------------- bookkeeping

    def expect(self, version: int, num_chunks: int) -> None:
        """Register the chunk grid of a version being distributed."""
        self._expected.setdefault(version, num_chunks)
        self._chunks.setdefault(version, {})

    def expected_chunks(self, version: int) -> int | None:
        return self._expected.get(version)

    def manifest_for(self, version: int, chunk: int) -> WeightManifest | None:
        return self._chunks.get(version, {}).get(chunk)

    def chunk_ready(self, version: int, chunk: int) -> bool:
        return chunk in self._chunks.get(version, {})

    def version_complete(self, version: int) -> bool:
        expected = self._expected.get(version)
        if expected is None:
            return False
        return len(self._chunks.get(version, {})) == expected

    def complete_versions(self) -> list[int]:
        return sorted(v for v in self._expected if self.version_complete(v))

    def latest_complete(self) -> int | None:
        complete = self.complete_versions()
        return complete[-1] if complete else None

    def _mark_ready(self, version: int, chunk: int, manifest: WeightManifest) -> None:
        self._chunks.setdefault(version, {})[chunk] = manifest
        for fut in self._waiters.pop((version, chunk), []):
            if not fut.done():
                fut.set_result(manifest)
        if self.version_complete(version):
            self._version_done.setdefault(version, asyncio.Event()).set()

    async def wait_chunk(self, version: int, chunk: int) -> WeightManifest:
        """Resolve when this chunk is staged on this node (chain step)."""
        manifest = self.manifest_for(version, chunk)
        if manifest is not None:
            return manifest
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.setdefault((version, chunk), []).append(fut)
        return await fut

    async def wait_version_complete(self, version: int) -> None:
        """Block until ``version`` is fully staged on this node."""
        if self.version_complete(version):
            return
        if version not in self._expected:
            raise LookupError(
                f"version {version} unknown to relay node {self.node_id} "
                "(never distributed, or evicted by retention)"
            )
        await self._version_done.setdefault(version, asyncio.Event()).wait()

    # ------------------------------------------------------------ staging

    async def stage_chunk(
        self, version: int, chunk: int, tensors: list[tuple[str, Any]]
    ) -> WeightManifest:
        manifest = WeightManifest(
            version=version, checkpoint_name=self.chunk_name(version, chunk)
        )
        await self.backend.stage(manifest, _as_aiter(tensors))
        self._mark_ready(version, chunk, manifest)
        return manifest

    def mark_empty(self, version: int, chunk: int) -> WeightManifest:
        """Reshard filtered everything out — bookkeeping-only chunk."""
        manifest = WeightManifest(
            version=version,
            checkpoint_name=self.chunk_name(version, chunk),
            nbytes=0,
            tensor_count=0,
        )
        self._mark_ready(version, chunk, manifest)
        return manifest

    # ----------------------------------------------------------- retention

    async def evict_old_versions(self) -> list[int]:
        """Drop complete versions beyond ``keep_last`` (unstage memory)."""
        complete = self.complete_versions()
        if len(complete) <= self.keep_last:
            return []
        evict = complete[: len(complete) - self.keep_last]
        for version in evict:
            for manifest in self._chunks.pop(version, {}).values():
                if manifest.nbytes > 0:
                    await self.backend.unstage(manifest)
            self._expected.pop(version, None)
            self._version_done.pop(version, None)
        return evict

    def max_lag(self, latest_published: int) -> int:
        """How far this node trails the master (broadcast backlog)."""
        local = self.latest_complete()
        return max(0, latest_published - (local if local is not None else 0))


class RelayService:
    """The hierarchical relay: publish once, distribute in background,
    pull anytime from the local relay.

    ``publish`` returns after the master staged the full version (the
    actor stall — one hop); the chain distribution to the remaining
    relays proceeds as a background task, and ``pull`` never waits for
    it: a replica always gets its local relay's newest *complete*
    version.
    """

    def __init__(
        self,
        backends: Sequence[P2PWeightBackend],
        config: RelayTierConfig | None = None,
        *,
        base_name: str = "actor",
        format_fn: Callable[[str, Any], tuple[str, Any]] | None = None,
        reshard_fn: Callable[[int, str], bool] | None = None,
        node_for_replica: Callable[[int], int] | None = None,
    ) -> None:
        config = config or RelayTierConfig(num_relays=len(backends))
        if len(backends) != config.num_relays:
            raise ValueError(
                f"num_relays={config.num_relays} but {len(backends)} backends given "
                "(one backend per relay node: each wraps the engine instance "
                "on that rollout machine)"
            )
        self.config = config
        self.nodes = [
            RelayNode(i, backend, base_name=base_name, keep_last=config.keep_last)
            for i, backend in enumerate(backends)
        ]
        # trainer-side HF-format conversion, applied once at publish
        self.format_fn = format_fn
        # master-side resharding: trainer coords -> rollout TP layout
        self.reshard_fn = reshard_fn
        self._node_for_replica = node_for_replica or (
            lambda replica_id: replica_id % config.num_relays
        )
        self.stats = RelayTierStats()
        self._latest_published: int | None = None
        self._consumers: dict[str, ConsumerState] = {}
        self._lock = asyncio.Lock()
        self._dist_tasks: set[asyncio.Task] = set()

    # ----------------------------------------------------------- topology

    @property
    def master(self) -> RelayNode:
        return self.nodes[0]

    def node_for_replica(self, replica_id: int) -> RelayNode:
        return self.nodes[self._node_for_replica(replica_id)]

    def latest_published_version(self) -> int:
        """Master-side hint: the newest version the actor published."""
        return self._latest_published if self._latest_published is not None else 0

    # ------------------------------------------------------------ publish

    async def publish(
        self,
        version: int,
        weights: Iterable[tuple[str, Any]],
        format_fn: Callable[[str, Any], tuple[str, Any]] | None = None,
        reshard_fn: Callable[[str, Any], tuple[str, Any] | None] | None = None,
    ) -> float:
        """Stage a version at the master relay; return the actor stall.

        Applies ``format_fn`` (trainer-side HF-format conversion) and
        ``reshard_fn`` (trainer coords -> rollout TP layout) once, per
        tensor, at the master — then chunks the result, stages every
        chunk at the master, and kicks off the background chain
        distribution. Never waits for any consumer. Both hooks default
        to the constructor level if not given per call.
        """
        format_fn = format_fn if format_fn is not None else self.format_fn
        reshard_fn = reshard_fn if reshard_fn is not None else self.reshard_fn
        async with self._lock:
            if self._latest_published is not None and version <= self._latest_published:
                raise ValueError(
                    f"versions must increase: latest={self._latest_published}, got {version}"
                )

        items = list(weights)
        if format_fn is not None:
            items = [format_fn(name, tensor) for name, tensor in items]
        if reshard_fn is not None:
            items = [
                reshaped
                for reshaped in (reshard_fn(name, tensor) for name, tensor in items)
                if reshaped is not None
            ]

        num = self.config.chunks
        per = max(1, math.ceil(len(items) / num)) if items else 1
        chunk_grid: list[list[tuple[str, Any]]] = [
            items[i * per : (i + 1) * per] for i in range(num)
        ]

        t0 = time.monotonic()
        for node in self.nodes:
            node.expect(version, num)
        for k, chunk in enumerate(chunk_grid):
            if chunk:
                await self.master.stage_chunk(version, k, chunk)
            else:
                self.master.mark_empty(version, k)
        stall = time.monotonic() - t0

        async with self._lock:
            self._latest_published = version
        self.stats.publishes += 1
        self.stats.actor_stall_total_s += stall
        evicted = await self.master.evict_old_versions()
        if evicted:
            logger.info("relay master: evicted versions %s (keep_last=%d)", evicted, self.config.keep_last)

        # background chain distribution — the actor does NOT wait for it
        if len(self.nodes) > 1:
            task = asyncio.create_task(self._distribute(version))
            self._dist_tasks.add(task)
            task.add_done_callback(self._dist_tasks.discard)
        return stall

    async def wait_for_distribution(self) -> None:
        """Drain all in-flight chain distributions (shutdown/tests)."""
        if self._dist_tasks:
            await asyncio.gather(*self._dist_tasks, return_exceptions=True)

    async def _distribute(self, version: int) -> None:
        t0 = time.monotonic()
        hops = [
            asyncio.create_task(self._relay_hop(i, version))
            for i in range(1, len(self.nodes))
        ]
        await asyncio.gather(*hops)
        self.stats.chain_completion_s.append(time.monotonic() - t0)

    async def _relay_hop(self, node_idx: int, version: int) -> None:
        """Relay ``node_idx`` pulls version ``version`` from its upstream.

        Chunk-pipelined: chunk ``k`` is read from upstream as soon as it
        is staged there — relay ``i`` and relay ``i+1`` overlap, which is
        what keeps broadcast time near-constant in chain length.
        """
        upstream = self.nodes[node_idx - 1]
        node = self.nodes[node_idx]
        for k in range(self.config.chunks):
            manifest = await upstream.wait_chunk(version, k)
            if manifest.nbytes == 0:
                node.mark_empty(version, k)
                continue
            payload: list[tuple[str, Any]] = []

            def sink(name: str, tensor: Any, payload=payload) -> None:
                payload.append((name, tensor))

            read_stats = await upstream.backend.read_into(
                f"relay-{node.node_id}",
                {"read_latency_s": self.config.hop_read_latency_s, "role": "chain"},
                manifest,
                sink,
            )
            self.stats.hop_reads += 1
            self.stats.hop_read_s += read_stats.seconds
            self.stats.hop_bytes += read_stats.nbytes
            await node.stage_chunk(version, k, payload)
        await node.evict_old_versions()

    # --------------------------------------------------------------- pull

    def local_latest(self, replica_id: int) -> int:
        """Newest fully staged version on this replica's local relay."""
        latest = self.node_for_replica(replica_id).latest_complete()
        return latest if latest is not None else 0

    async def pull(
        self,
        replica_id: int,
        version: int | None = None,
        sink: Callable[[str, Any], None] | None = None,
    ) -> int:
        """Pull a version from the replica's colocated relay (PCIe).

        ``version=None`` pulls the local relay's latest complete version —
        never a partially distributed one, and normally without waiting
        for the broadcast (an in-flight newer version is simply not taken
        yet). The one wait is the startup edge: nothing complete on the
        local node yet, so the pull blocks until the newest published
        version finishes arriving. An explicit version pins (must be
        retained locally).
        """
        node = self.node_for_replica(replica_id)
        wait_t0: float | None = None
        if version is None:
            target = node.latest_complete()
            if target is None:
                if self.latest_published_version() == 0:
                    raise LookupError("no version published yet")
                wait_t0 = time.monotonic()
                await node.wait_version_complete(self.latest_published_version())
                target = node.latest_complete()
        else:
            target = version
            if not node.version_complete(target):
                wait_t0 = time.monotonic()
                await node.wait_version_complete(target)
        if target is None:
            raise LookupError("no complete version on the local relay")
        if wait_t0 is not None:
            self.stats.pull_wait_s += time.monotonic() - wait_t0
        expected = node.expected_chunks(target)
        if expected is None:
            raise LookupError(
                f"version {target} unknown to relay node {node.node_id}"
            )
        nbytes = 0
        for k in range(expected):
            manifest = node.manifest_for(target, k)
            if manifest is None:
                raise LookupError(
                    f"version {target} incomplete on relay node {node.node_id}"
                )
            if manifest.nbytes > 0:
                read_stats = await node.backend.read_into(
                    f"replica-{replica_id}",
                    {"read_latency_s": self.config.local_read_latency_s, "role": "local"},
                    manifest,
                    sink or (lambda name, tensor: None),
                )
                nbytes += read_stats.nbytes
                self.stats.local_read_s += read_stats.seconds
        consumer_id = f"replica-{replica_id}"
        state = self._consumers.setdefault(consumer_id, ConsumerState(consumer_id=consumer_id))
        state.version = target
        state.pulls += 1
        state.bytes_read += nbytes
        state.last_pull_at = time.monotonic()
        self.stats.local_pulls += 1
        self.stats.local_bytes += nbytes
        return target

    # ------------------------------------------------------------- report

    def consumer_versions(self) -> dict[str, int]:
        return {cid: s.version for cid, s in sorted(self._consumers.items())}

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = self.stats.snapshot()
        out.update(
            {
                "relay/num_relays": len(self.nodes),
                "relay/chunks": self.config.chunks,
                "relay/backend": self.nodes[0].backend.name if self.nodes else None,
                "relay/local_versions": {
                    node.node_id: node.latest_complete() for node in self.nodes
                },
                "relay/max_node_lag": max(
                    (node.max_lag(self.latest_published_version()) for node in self.nodes),
                    default=0,
                ),
                "relay/consumers": self.consumer_versions(),
            }
        )
        return out


# ------------------------------------------------------- demo adapter


class _RelayTierStatsView:
    """Expose tier stats under the demo's ``relay/*`` keys."""

    def __init__(self, service: RelayService) -> None:
        self._service = service

    def snapshot(self) -> dict[str, Any]:
        return self._service.snapshot()


class RelayTierAdapter:
    """Make a :class:`RelayService` quack like the demo's relay.

    The multi-replica demo engine needs ``publish(version)``,
    ``latest_published_version()`` and ``pull(replica_id)`` — this adapter
    feeds fake byte payloads through the *real* tier topology (master
    stage, background chunk-pipelined chain, local anytime pull), so the
    CPU demo exercises exactly the code a cluster deployment runs.
    """

    def __init__(self, service: RelayService, weight_mb: float = 8.0) -> None:
        self.service = service
        self._bytes = int(weight_mb * (1 << 20))
        self.stats = _RelayTierStatsView(service)

    async def publish(self, version: int) -> float:
        shards = self.service.config.chunks
        per = self._bytes // max(shards, 1)

        def weights():
            for i in range(shards):
                yield f"shard.{i}", b"w" * per

        return await self.service.publish(version, weights())

    def latest_published_version(self) -> int:
        return self.service.latest_published_version()

    async def pull(self, replica_id: int) -> int:
        return await self.service.pull(replica_id)


# ------------------------------------------------- repack executor seam
#
# The repack ALGORITHM (repack.best_fit_consolidation + ReplicaState
# idleness) is engine-agnostic; the executor seam below binds it to real
# rollout replicas (see RolloutRepackExecutor below).


class RepackExecutor(Protocol):
    """Execution seam of the repacker (Laminar §5.1 step ③).

    Anything providing a point-in-time replica snapshot, fleet KV
    utilization, per-round migration cost, and plan execution can be
    driven by
    :class:`~verl.experimental.trajectory_async.repack.RepackManager`.
    """

    repack_overhead_s: float

    def snapshot(self) -> list:
        ...

    def fleet_kv_util(self) -> float:
        ...

    async def migrate(self, plan: list[tuple[int, int]]) -> MigrationResult:
        ...


class RunningRequest:
    """One in-flight rollout request, engine-agnostic.

    ``payload`` carries whatever the transfer path needs: prompt +
    partial response text (recompute prefill), or a KV-cache handle
    (KV-transfer prefill).
    """

    def __init__(self, request_id: str, kv_tokens: int, payload: Any = None) -> None:
        self.request_id = request_id
        self.kv_tokens = kv_tokens
        self.payload = payload


class RolloutReplicaHandle(Protocol):
    """The real-rollout-facing interface one replica must expose.

    Mapping for common stacks (vLLM flavor): ``kv_used_tokens`` /
    ``kv_capacity_tokens`` from the scheduler metrics (e.g.
    ``kv_cache_usage`` × token capacity), ``batch_limit`` is the roofline
    decode-batch bound ``B``, ``running_requests`` / ``remove_request`` /
    ``admit_request`` wrap the engine's request states, and
    ``pull_weights`` loads the newest version from the colocated relay
    (a collective_rpc broadcast of the relay pull).
    """

    replica_id: int
    weight_version: int

    def kv_used_tokens(self) -> int: ...
    def kv_capacity_tokens(self) -> int: ...
    def batch_limit(self) -> int: ...
    def running_requests(self) -> list[RunningRequest]: ...
    def remove_request(self, request_id: str) -> RunningRequest: ...
    async def admit_request(
        self, request: RunningRequest, *, prefill: str = "recompute"
    ) -> None: ...
    async def pull_weights(self, version: int | None = None) -> int: ...


class RolloutRepackExecutor:
    """Repack executor over :class:`RolloutReplicaHandle` replicas.

    Migration semantics (paper §5.1 step ③): move every running request
    from each source replica to its destination — either by an explicit
    KV-cache transfer (``kv_transfer_fn``: a callable doing the actual
    block/tensor movement between the two engines) or by the portable
    fallback, recompute prefill (resend prompt + partial response; the
    destination regenerates the KV, trading a re-prefill for the freed
    source). Sources that end up empty immediately pull the latest
    weights — the whole point of the repack.
    """

    def __init__(
        self,
        handles: Sequence[RolloutReplicaHandle],
        *,
        kv_transfer_fn: Callable[[int, int, RunningRequest], Any] | None = None,
        repack_overhead_s: float = 0.0,
    ) -> None:
        self.handles: dict[int, RolloutReplicaHandle] = {
            handle.replica_id: handle for handle in handles
        }
        self.kv_transfer_fn = kv_transfer_fn
        # per-round migration cost (KV transfer / re-prefill time); a real
        # deployment measures it — the manager books it as round overhead
        self.repack_overhead_s = repack_overhead_s

    # ------------------------------------------------------------ probe

    def snapshot(self) -> list[ReplicaState]:
        """Build the algorithm's :class:`ReplicaState` view from handles."""
        states = []
        for handle in self.handles.values():
            running = handle.running_requests()
            kv_used = handle.kv_used_tokens()
            states.append(
                ReplicaState(
                    replica_id=handle.replica_id,
                    version=handle.weight_version,
                    kv_used=kv_used,
                    kv_capacity=handle.kv_capacity_tokens(),
                    kv_prev=kv_used,  # no tick history: ramp-down signal off
                    num_running=len(running),
                    num_waiting=0,  # waiting work is admission-side, not replica-side
                    batch_quota=handle.batch_limit(),
                    max_running=handle.batch_limit(),
                    pulling=False,
                    routable=True,
                )
            )
        return states

    def fleet_kv_util(self) -> float:
        total_used = sum(h.kv_used_tokens() for h in self.handles.values())
        total_cap = sum(h.kv_capacity_tokens() for h in self.handles.values())
        return total_used / total_cap if total_cap else 0.0

    # ---------------------------------------------------------- migrate

    async def migrate(self, plan: list[tuple[int, int]]) -> MigrationResult:
        requests_moved = 0
        kv_tokens_moved = 0
        sources = {src for src, _ in plan}
        for src_id, dst_id in plan:
            src = self.handles[src_id]
            dst = self.handles[dst_id]
            for request in list(src.running_requests()):
                moved = src.remove_request(request.request_id)
                prefill = "recompute"
                if self.kv_transfer_fn is not None:
                    await self.kv_transfer_fn(src_id, dst_id, moved)
                    prefill = "kv_transfer"
                await dst.admit_request(moved, prefill=prefill)
                requests_moved += 1
                kv_tokens_moved += moved.kv_tokens
        sources_emptied = 0
        for src_id in sources:
            src = self.handles[src_id]
            if not src.running_requests():
                # freed replica: pull the latest weights and re-enter
                # routing as an on-policy generator (the repack payoff)
                await src.pull_weights()
                sources_emptied += 1
        return MigrationResult(
            plan=plan,
            requests_moved=requests_moved,
            kv_tokens_moved=kv_tokens_moved,
            sources_emptied=sources_emptied,
        )
