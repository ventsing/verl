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
"""Multi-version, pull-based weight management over P2P checkpoint engines.

This replaces the timing mock (:mod:`weight_relay`) with a deployable
design built on verl's checkpoint engines — specifically the two engines
that already speak peer-to-peer:

* ``kimi_ckpt_engine`` (:class:`KIMICheckpointEngine`,
  ``verl/checkpoint_engine/kimi_checkpoint_engine.py``) — actor ranks
  register their CPU-offloaded shards in a P2P store
  (``register_checkpoint`` / ``register_named_tensors``); receivers read
  the owner's memory directly (``receive_tensor`` with ``ranks`` /
  ``ranks_group``), then spread within their own group.
* ``mooncake`` (:class:`MooncakeCheckpointEngine`,
  ``verl/checkpoint_engine/mooncake_checkpoint_engine.py``) — RDMA
  ``TransferEngine`` with registered buffers and direct
  ``transfer_sync_read``/``transfer_sync_write`` between sessions.

What this module adds on top of the stock engines (Laminar,
https://arxiv.org/abs/2510.12633, §4):

1. **Versioned retention** — each published version gets a
   version-scoped checkpoint name (``{base}:v{version}``) and *stays
   registered* after publishing. The stock kimi ``send_weights``
   unregisters right after a barrier (one-shot transfer); here the
   registered memory IS the relay memory, and old versions remain
   readable until the retention policy evicts them.
2. **Pull, not broadcast** — ``publish`` returns as soon as the version
   is staged (the actor stall is just the offload + register); every
   consumer pulls *when it decides to* (batch boundary, or when a repack
   releases it). Different replicas run different versions concurrently.
3. **Per-consumer accounting** — current version, pull count, bytes and
   lag (``latest − consumer_version``) per replica: the observability
   Laminar gets from its rollout manager, derived here from the store.

Deployment shape (who owns what):

* the trainer/actor process holds one backend (kimi parameter server
  rank or mooncake master engine) and calls :meth:`VersionedWeightStore.publish`
  with the same ``per_tensor_param`` generator the stock
  ``CheckpointEngineManager.update_weights`` uses — version identity is
  the ``global_steps`` that already flows through that stack;
* each rollout replica holds its own consumer-side engine context and
  calls :meth:`VersionedWeightStore.pull` at its own pace, feeding the
  yielded tensors to ``server_adapter.update_weights``;
* the store itself is the control plane: in one process it is a plain
  object; across processes, wrap it in a Ray actor whose methods are
  exactly ``publish``/``latest_version``/``pull``/``release`` (the
  method surface here is kept Ray-actor-friendly for that reason).

The orchestration (registry, retention, per-consumer state) is pure
asyncio and fully unit-tested here with :class:`FakeP2PBackend`, which
mimics the owner-memory + remote-read semantics with bytes payloads.
The kimi/mooncake adapters isolate every real engine call behind lazy
imports — they need a cluster (torch / RDMA) to run and are written
against the engine code as it stands in this repository.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ protocol


@dataclass
class WeightManifest:
    """Control-plane record of one published weight version.

    ``descriptor`` is backend-opaque: the kimi backend stores its
    checkpoint name, the mooncake backend stores (session_id, ptr, len,
    bucket metadata) of the registered staging buffer.
    """

    version: int
    checkpoint_name: str
    nbytes: int = 0
    tensor_count: int = 0
    created_at: float = field(default_factory=time.monotonic)
    descriptor: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadStats:
    """Per-pull accounting returned by backends."""

    nbytes: int = 0
    tensors: int = 0
    seconds: float = 0.0


Sink = Callable[[str, Any], None]
"""Consumer of pulled tensors — in deployment, ``server_adapter.update_weights``."""


class P2PWeightBackend(ABC):
    """Transport seam: stage versioned weights into P2P-readable memory.

    Contract (matching the two engines this builds on):

    * ``stage`` copies the actor's tensors into owner-side memory that
      peers can read directly (kimi: registered checkpoint in the P2P
      store; mooncake: RDMA-registered staging buffer), fills
      ``manifest.nbytes``/``tensor_count``/``descriptor``, and returns —
      the actor does NOT wait for any consumer;
    * ``read_into`` performs the actual peer-to-peer read for one
      consumer and feeds every tensor to ``sink``;
    * ``unstage`` unregisters and frees a version's memory (retention
      eviction). After ``unstage``, reads of that version must fail.
    """

    name: str = "abstract"

    @abstractmethod
    async def stage(self, manifest: WeightManifest, weights: AsyncIterator[tuple[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def read_into(
        self,
        consumer_id: str,
        consumer_ctx: dict[str, Any],
        manifest: WeightManifest,
        sink: Sink,
    ) -> ReadStats:
        raise NotImplementedError

    @abstractmethod
    async def unstage(self, manifest: WeightManifest) -> None:
        raise NotImplementedError


async def _aiter(weights: Iterable | AsyncIterator | Iterator) -> AsyncIterator[tuple[str, Any]]:
    """Accept a sync or async iterator of ``(name, tensor)``."""
    if hasattr(weights, "__anext__"):
        async for item in weights:
            yield item
    else:
        for item in weights:
            yield item


def _noop_sink(name: str, tensor: Any) -> None:
    pass


# -------------------------------------------------------------------- store


@dataclass
class ConsumerState:
    """Per-replica pull state — the no-lockstop bookkeeping."""

    consumer_id: str
    version: int = 0
    pulls: int = 0
    bytes_read: int = 0
    last_pull_at: float | None = None

    def lag(self, latest: int) -> int:
        return max(0, latest - self.version)


@dataclass
class StoreStats:
    publishes: int = 0
    evictions: int = 0
    pulls: int = 0
    bytes_staged: int = 0
    bytes_read: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "store/publishes": self.publishes,
            "store/evictions": self.evictions,
            "store/pulls": self.pulls,
            "store/bytes_staged": self.bytes_staged,
            "store/bytes_read": self.bytes_read,
        }


class VersionedWeightStore:
    """Multi-version weight store: publish once, pull per consumer, evict by age.

    Laminar's relay service as a concrete data structure: versions are
    retained in P2P-readable memory, consumers (rollout replicas) pull
    whenever they reach a batch boundary, and the store tracks who is on
    which version — the input to version-group repacking and to staleness
    metrics.

    Args:
        backend: the transport (kimi / mooncake adapter, or the fake).
        base_name: checkpoint-name prefix; version ``v`` becomes
            ``f"{base_name}:v{v}"``.
        keep_last: retention — how many most-recent versions stay staged.
            Steady state only ever needs the current version (freed
            replicas pull the *latest*; repack migrates work between
            same-version replicas instead of restoring old ones), so the
            default of 2 covers one in-flight publication; raise it only
            for explicit pinned pulls or fault-recovery windows.
    """

    def __init__(
        self,
        backend: P2PWeightBackend,
        base_name: str = "actor",
        keep_last: int = 2,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.backend = backend
        self.base_name = base_name
        self.keep_last = keep_last
        self.stats = StoreStats()
        self._manifests: dict[int, WeightManifest] = {}
        self._latest: int | None = None
        self._consumers: dict[str, ConsumerState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ publish

    def checkpoint_name(self, version: int) -> str:
        return f"{self.base_name}:v{version}"

    async def publish(self, version: int, weights: Iterable | AsyncIterator) -> WeightManifest:
        """Stage a new version; returns when peers can read it.

        Blocks only for the backend's staging cost (offload + register) —
        the analogue of Laminar's actor→master-relay hop. Never waits for
        consumers.
        """
        async with self._lock:
            if version in self._manifests:
                raise ValueError(f"version {version} already published")
            if self._latest is not None and version <= self._latest:
                raise ValueError(f"versions must increase: latest={self._latest}, got {version}")
        manifest = WeightManifest(version=version, checkpoint_name=self.checkpoint_name(version))
        await self.backend.stage(manifest, _aiter(weights))
        async with self._lock:
            self._manifests[version] = manifest
            self._latest = version
            self.stats.publishes += 1
            self.stats.bytes_staged += manifest.nbytes
        evicted = await self.release(keep_last=self.keep_last)
        if evicted:
            logger.info("weight store: evicted versions %s (keep_last=%d)", evicted, self.keep_last)
        return manifest

    # ------------------------------------------------------------- inspect

    def latest_version(self) -> int:
        return self._latest if self._latest is not None else 0

    def manifest(self, version: int) -> WeightManifest | None:
        return self._manifests.get(version)

    @property
    def retained_versions(self) -> list[int]:
        return sorted(self._manifests)

    def consumer_version(self, consumer_id: str) -> int:
        state = self._consumers.get(consumer_id)
        return state.version if state else 0

    def consumer_lag(self, consumer_id: str) -> int:
        state = self._consumers.get(consumer_id)
        return state.lag(self.latest_version()) if state else self.latest_version()

    # ---------------------------------------------------------------- pull

    async def pull(
        self,
        consumer_id: str,
        version: int | None = None,
        consumer_ctx: dict[str, Any] | None = None,
        sink: Sink | None = None,
    ) -> int:
        """Pull a version into a consumer; returns the version pulled.

        ``version=None`` pulls the latest — the steady-state path (batch
        boundary or repack release). An explicit version pins the read
        (must still be retained). Consumers are expected to serialize
        their own pulls (a replica pulls at its batch boundary, one at a
        time); concurrent pulls from *different* consumers are the norm.
        """
        async with self._lock:
            target = self._latest if version is None else version
            if target is None:
                raise LookupError("no version published yet")
            manifest = self._manifests.get(target)
            if manifest is None:
                raise LookupError(
                    f"version {version} is not retained (evicted?); retained={self.retained_versions}"
                )
        read_stats = await self.backend.read_into(consumer_id, consumer_ctx or {}, manifest, sink or _noop_sink)
        async with self._lock:
            state = self._consumers.setdefault(consumer_id, ConsumerState(consumer_id=consumer_id))
            state.version = target
            state.pulls += 1
            state.bytes_read += read_stats.nbytes
            state.last_pull_at = time.monotonic()
            self.stats.pulls += 1
            self.stats.bytes_read += read_stats.nbytes
        return target

    # ----------------------------------------------------------------- gc

    async def release(self, keep_last: int | None = None) -> list[int]:
        """Evict all but the newest ``keep_last`` versions; returns evicted."""
        keep = self.keep_last if keep_last is None else keep_last
        async with self._lock:
            versions = sorted(self._manifests)
            if len(versions) <= keep:
                return []
            evict = versions[:-keep] if keep > 0 else versions
            manifests = [self._manifests.pop(v) for v in evict]
        for manifest in manifests:
            await self.backend.unstage(manifest)
            self.stats.evictions += 1
        return evict

    # --------------------------------------------------------------- report

    def snapshot(self) -> dict[str, Any]:
        consumers = {
            cid: {"version": s.version, "lag": s.lag(self.latest_version()), "pulls": s.pulls, "bytes": s.bytes_read}
            for cid, s in sorted(self._consumers.items())
        }
        out = self.stats.snapshot()
        out.update(
            {
                "store/latest_version": self.latest_version(),
                "store/retained_versions": self.retained_versions,
                "store/consumers": consumers,
                "store/backend": self.backend.name,
            }
        )
        return out


# ------------------------------------------------------------ fake backend


@dataclass
class FakeP2PStats:
    stages: int = 0
    reads: int = 0
    unstages: int = 0
    stage_total_s: float = 0.0
    read_total_s: float = 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "p2p/stages": self.stages,
            "p2p/reads": self.reads,
            "p2p/unstages": self.unstages,
            "p2p/stage_total_s": round(self.stage_total_s, 4),
            "p2p/read_total_s": round(self.read_total_s, 4),
        }


class FakeP2PBackend(P2PWeightBackend):
    """Bytes-level stand-in with real P2P semantics.

    ``stage`` copies payloads into an owner-side "registered memory" dict
    (after a configurable staging latency); ``read_into`` reads it back
    for any consumer (after a configurable read latency); ``unstage``
    removes the memory, after which reads fail — the same
    lifecycle a registered RDMA buffer or kimi checkpoint has. Payloads
    are ``bytes``; sizes and counts flow into the manifest exactly as
    the real backends would report them.
    """

    name = "fake"

    def __init__(self, stage_latency_s: float = 0.3, read_latency_s: float = 0.15) -> None:
        self.stage_latency_s = stage_latency_s
        self.read_latency_s = read_latency_s
        self.stats = FakeP2PStats()
        self._memory: dict[str, dict[str, bytes]] = {}

    async def stage(self, manifest: WeightManifest, weights: AsyncIterator[tuple[str, Any]]) -> None:
        payload: dict[str, bytes] = {}
        nbytes = 0
        async for name, tensor in weights:
            data = bytes(tensor)
            payload[name] = data
            nbytes += len(data)
        await asyncio.sleep(self.stage_latency_s)
        self._memory[manifest.checkpoint_name] = payload
        self.stats.stages += 1
        self.stats.stage_total_s += self.stage_latency_s
        manifest.nbytes = nbytes
        manifest.tensor_count = len(payload)
        manifest.descriptor = {"kind": "fake", "checkpoint_name": manifest.checkpoint_name}

    async def read_into(
        self,
        consumer_id: str,
        consumer_ctx: dict[str, Any],
        manifest: WeightManifest,
        sink: Sink,
    ) -> ReadStats:
        payload = self._memory.get(manifest.checkpoint_name)
        if payload is None:
            raise LookupError(
                f"{manifest.checkpoint_name} is not staged — registered memory was released"
            )
        await asyncio.sleep(self.read_latency_s)
        for name, data in payload.items():
            sink(name, data)
        self.stats.reads += 1
        self.stats.read_total_s += self.read_latency_s
        return ReadStats(nbytes=manifest.nbytes, tensors=manifest.tensor_count, seconds=self.read_latency_s)

    async def unstage(self, manifest: WeightManifest) -> None:
        self._memory.pop(manifest.checkpoint_name, None)
        self.stats.unstages += 1


# ------------------------------------------------------- demo relay adapter


class _RelayStatsView:
    """Expose store/backend stats under the demo's ``relay/*`` keys."""

    def __init__(self, store: VersionedWeightStore, backend: FakeP2PBackend) -> None:
        self._store = store
        self._backend = backend

    def snapshot(self) -> dict[str, Any]:
        return {
            "relay/publishes": self._store.stats.publishes,
            "relay/actor_stall_total_s": round(self._backend.stats.stage_total_s, 4),
            "relay/pulls": self._store.stats.pulls,
            # direct P2P read: no chain propagation wait
            "relay/pull_wait_total_s": 0.0,
            "relay/pcie_total_s": round(self._backend.stats.read_total_s, 4),
        }


class VersionedStoreRelayAdapter:
    """Make a :class:`VersionedWeightStore` quack like the demo's relay.

    The multi-replica demo engine only needs ``publish(version)``,
    ``latest_published_version()`` and ``pull(replica_id)`` — this adapter
    routes them through the *real* store orchestration (retention,
    per-consumer state, staleness) over the fake transport, so the CPU
    demo exercises exactly the code a deployment would run.
    """

    def __init__(self, store: VersionedWeightStore, weight_mb: float = 8.0) -> None:
        self.store = store
        self._bytes = int(weight_mb * (1 << 20))
        self.stats = _RelayStatsView(store, store.backend)  # type: ignore[arg-type]

    async def publish(self, version: int) -> float:
        shards = 4

        def weights():
            per = self._bytes // shards
            for i in range(shards):
                yield f"shard.{i}", b"w" * per

        await self.store.publish(version, weights())
        return time.monotonic()

    def latest_published_version(self) -> int:
        return self.store.latest_version()

    async def pull(self, replica_id: int) -> int:
        return await self.store.pull(f"replica-{replica_id}")


# ---------------------------------------------------------- real backends
#
# The two adapters below isolate every call into verl's checkpoint
# engines. They cannot run without a cluster (torch + RDMA + the engine
# packages), so they are written against the engine code as checked in
# and clearly mark which calls are stock API and which are the
# multi-version extensions. Everything above this line is orchestration
# and is fully covered by the stdlib test suite.


class KimiP2PBackend(P2PWeightBackend):
    """Multi-version extension of ``KIMICheckpointEngine`` (backend ``kimi_ckpt_engine``).

    Stock behavior (``kimi_checkpoint_engine.py``): ``send_weights``
    offloads the actor's shard to CPU, ``register_checkpoint``s it in the
    P2P store, ``gather_metas`` advertises it, then *unregisters* after a
    barrier — a one-shot collective. ``receive_tensor`` (the patch
    installed in ``init_process_group``) lets a set of ranks pull those
    registered tensors directly from their owners, then broadcast within
    the receiver group.

    This backend turns that into Laminar's relay:

    * ``stage`` = offload + ``register_checkpoint(f"{base}:v{version}")``
      + ``gather_metas`` — and **no unregister**: the actor's registered
      CPU shards ARE the relay memory, and they stay readable until the
      retention policy evicts the version (``unstage`` unregisters).
    * ``read_into`` = ``receive_tensor`` scoped to ONE consumer group —
      the per-replica pull. Where the stock manager builds a single
      process group over actor + all rollout workers, the versioned
      deployment builds one group per replica (see README); the call
      signature is unchanged, only ``ranks``/``ranks_group`` differ.

    Args:
        engine: an *initialized* ``KIMICheckpointEngine`` living on the
            actor side (``init_process_group`` already run, so
            ``parameter_server`` exists and carries the ``receive_tensor``
            patch).
        rollout_dtype: dtype cast before CPU offload (stock default bf16).
    """

    name = "kimi"

    def __init__(self, engine, rollout_dtype=None) -> None:
        self.engine = engine
        self.rollout_dtype = rollout_dtype

    async def stage(self, manifest: WeightManifest, weights: AsyncIterator[tuple[str, Any]]) -> None:
        import torch  # lazy: cluster-only path

        named_tensors: dict[str, Any] = {}
        async for name, tensor in weights:
            if self.rollout_dtype is not None:
                tensor = tensor.to(self.rollout_dtype)
            named_tensors[name] = tensor.to("cpu", non_blocking=True)
        # non_blocking CPU offload must complete before peers can read
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        ps = self.engine.parameter_server
        # stock API — the extension is that we do NOT unregister here
        ps.register_checkpoint(manifest.checkpoint_name, named_tensors=named_tensors)
        ps.gather_metas(manifest.checkpoint_name)

        manifest.nbytes = sum(t.numel() * t.element_size() for t in named_tensors.values())
        manifest.tensor_count = len(named_tensors)
        manifest.descriptor = {"kind": "kimi", "checkpoint_name": manifest.checkpoint_name}

    async def read_into(
        self,
        consumer_id: str,
        consumer_ctx: dict[str, Any],
        manifest: WeightManifest,
        sink: Sink,
    ) -> ReadStats:
        ps = self.engine.parameter_server
        ranks_group = consumer_ctx["ranks_group"]  # this replica's own process group
        ranks = consumer_ctx["ranks"]
        bucket_size = consumer_ctx.get("bucket_size", self.engine.bucket_size)

        start = time.monotonic()
        nbytes = tensors = 0
        # stock patched receive_tensor, scoped to one consumer group
        async for name, tensor in ps.receive_tensor(
            manifest.checkpoint_name, ranks_group, ranks, bucket_size
        ):
            sink(name, tensor)
            nbytes += tensor.element_size() * tensor.numel()
            tensors += 1
        return ReadStats(nbytes=nbytes, tensors=tensors, seconds=time.monotonic() - start)

    async def unstage(self, manifest: WeightManifest) -> None:
        self.engine.parameter_server.unregister_checkpoint(manifest.checkpoint_name)


class MooncakeP2PBackend(P2PWeightBackend):
    """Multi-version extension of ``MooncakeCheckpointEngine`` (backend ``mooncake``).

    Stock behavior (``mooncake_checkpoint_engine.py``): the actor (rank 0)
    packs buckets into one double-buffered registered ``buf`` and streams
    them along a chain (rank 0 → 1 → … → N) using
    ``transfer_sync_read`` between adjacent sessions — a one-shot
    broadcast, with the buffer immediately reused.

    This backend switches to direct P2P multi-version reads:

    * ``stage`` packs the version into its own RDMA-registered staging
      buffer (per-version allocation instead of reusing the transient
      ``buf``) and advertises ``(session_id, ptr, nbytes, buckets)`` in
      the manifest — memory that stays readable until eviction;
    * ``read_into`` performs ``transfer_sync_read`` directly from the
      actor's registered session into the consumer's local registered
      buffer — no chain, no barrier, no coordination with other
      replicas; each chunk is sliced into tensors and handed to ``sink``;
    * ``unstage`` unregisters and releases the version's buffer.

    Args:
        engine: an *initialized* ``MooncakeCheckpointEngine`` on the
            actor side (its ``TransferEngine`` session is the read
            source).
        staging_device: where staging buffers live — "cpu" (pinned host
            memory, the Laminar relay placement) or "cuda".
        rollout_dtype: dtype cast before staging (stock default bf16).
        chunk_bytes: read granularity for ``read_into`` (defaults to the
            engine's bucket size).
    """

    name = "mooncake"

    def __init__(self, engine, staging_device: str = "cpu", rollout_dtype=None, chunk_bytes: int | None = None) -> None:
        self.engine = engine
        self.staging_device = staging_device
        self.rollout_dtype = rollout_dtype
        self.chunk_bytes = chunk_bytes
        self._buffers: dict[str, Any] = {}  # keep staging buffers alive

    async def stage(self, manifest: WeightManifest, weights: AsyncIterator[tuple[str, Any]]) -> None:
        import torch  # lazy: cluster-only path

        tensors: list[tuple[str, Any]] = []
        async for name, tensor in weights:
            if self.rollout_dtype is not None:
                tensor = tensor.to(self.rollout_dtype)
            tensors.append((name, tensor))
        nbytes = sum(t.numel() * t.element_size() for _, t in tensors)
        if nbytes == 0:
            raise ValueError("refusing to stage an empty weight set")

        buf = torch.empty(nbytes, dtype=torch.uint8, device=self.staging_device)
        view = buf.view(-1)
        buckets: list[dict[str, Any]] = []  # (offset, size, {name: (offset, shape, dtype)})
        offset = 0
        per_tensor: dict[str, tuple[int, Any, Any]] = {}
        for name, tensor in tensors:
            flat = tensor.detach().contiguous().view(-1).view(torch.uint8)
            view[offset : offset + flat.numel()].copy_(flat, non_blocking=True)
            per_tensor[name] = (offset, tensor.shape, tensor.dtype)
            offset += flat.numel()
        if self.staging_device.startswith("cuda"):
            torch.cuda.synchronize()

        chunk = self.chunk_bytes or self.engine.bucket_size
        for start in range(0, max(nbytes, 1), chunk):
            buckets.append({"offset": start, "size": min(chunk, nbytes - start)})

        # stock TransferEngine API: register the buffer for remote reads
        ret = self.engine.engine.batch_register_memory([buf.data_ptr()], [nbytes])
        assert ret == 0, f"batch_register_memory failed ret={ret} for {manifest.checkpoint_name}"
        self._buffers[manifest.checkpoint_name] = buf

        manifest.nbytes = nbytes
        manifest.tensor_count = len(tensors)
        manifest.descriptor = {
            "kind": "mooncake",
            "session_id": self.engine.session_id,
            "ptr": buf.data_ptr(),
            "nbytes": nbytes,
            "chunk_bytes": chunk,
            "buckets": buckets,
            "tensors": per_tensor,
        }

    async def read_into(
        self,
        consumer_id: str,
        consumer_ctx: dict[str, Any],
        manifest: WeightManifest,
        sink: Sink,
    ) -> ReadStats:
        import torch  # lazy: cluster-only path

        consumer_engine = consumer_ctx["engine"]  # the replica's own MooncakeCheckpointEngine
        te = consumer_engine.engine
        desc = manifest.descriptor
        chunk = desc["chunk_bytes"]

        start = time.monotonic()
        # the consumer's already-registered double buffer (engine.buf is
        # 2*bucket_size and registered in __init__)
        for bucket in desc["buckets"]:
            size = bucket["size"]
            if size <= 0:
                continue
            assert size <= consumer_engine.bucket_size, "chunk larger than consumer buffer"
            ret = te.transfer_sync_read(
                desc["session_id"],  # source: the actor's registered staging buffer
                consumer_engine.buf.data_ptr(),
                desc["ptr"] + bucket["offset"],
                size,
            )
            assert ret == 0, f"transfer_sync_read failed ret={ret} for {manifest.checkpoint_name}"
            await asyncio.sleep(0)  # keep the loop responsive between chunks

        # hand the reassembled tensors to the rollout server adapter
        flat = consumer_engine.buf.view(-1).view(torch.uint8)
        for name, (offset, shape, dtype) in desc["tensors"].items():
            size = dtype.itemsize * shape.numel()
            sink(name, flat[offset : offset + size].view(dtype=dtype).view(shape))
        return ReadStats(
            nbytes=desc["nbytes"], tensors=len(desc["tensors"]), seconds=time.monotonic() - start
        )

    async def unstage(self, manifest: WeightManifest) -> None:
        desc = manifest.descriptor
        # TransferEngine unregister (counterpart of batch_register_memory)
        ret = self.engine.engine.unregister_memory(desc["ptr"])
        assert ret == 0, f"unregister_memory failed ret={ret} for {manifest.checkpoint_name}"
        self._buffers.pop(manifest.checkpoint_name, None)


# ----------------------------------------------------------------- factory


def make_p2p_backend(
    name: str,
    *,
    engine=None,
    stage_latency_s: float = 0.3,
    read_latency_s: float = 0.15,
    staging_device: str = "cpu",
    rollout_dtype=None,
    chunk_bytes: int | None = None,
) -> P2PWeightBackend:
    """Construct a P2P weight backend by name — the single switch point.

    Backend selection:

    * ``"fake"`` — :class:`FakeP2PBackend` (CPU-testable, no cluster):
      latencies model offload+register (``stage_latency_s`` ≈ the actor
      stall) and the direct peer read (``read_latency_s``).
    * ``"kimi"`` — :class:`KimiP2PBackend` over an **initialized**
      ``KIMICheckpointEngine`` living on the actor side. Requires the
      ``checkpoint_engine`` package and a cluster; pass the engine via
      ``engine=``. Optional ``rollout_dtype`` (stock default bf16).
    * ``"mooncake"`` — :class:`MooncakeP2PBackend` over an **initialized**
      ``MooncakeCheckpointEngine``. Requires the ``mooncake`` package and
      RDMA; pass ``engine=``. Optional ``staging_device`` ("cpu" = pinned
      host memory, the Laminar relay placement, or "cuda"),
      ``rollout_dtype``, ``chunk_bytes`` (defaults to the engine's bucket
      size).

    The kimi/mooncake engines themselves are created exactly as the stock
    stack creates them (``CheckpointEngineRegistry.new(backend, bucket_size=...)``
    followed by ``prepare``/``init_process_group`` — see
    ``CheckpointEngineManager.build_process_group``); the kimi topology
    needs **one process group per replica** for per-consumer pulls (the
    stock manager builds a single group over actor + all rollout workers).

    Raises:
        ValueError: unknown backend name, or a cluster backend selected
            without its engine.
    """
    name = name.lower().strip()
    if name == "fake":
        return FakeP2PBackend(stage_latency_s=stage_latency_s, read_latency_s=read_latency_s)
    if name == "kimi":
        if engine is None:
            raise ValueError(
                "backend 'kimi' needs engine=<initialized KIMICheckpointEngine> "
                "(CheckpointEngineRegistry.new('kimi_ckpt_engine', ...) after "
                "init_process_group; per-replica process groups for pulls)"
            )
        return KimiP2PBackend(engine, rollout_dtype=rollout_dtype)
    if name == "mooncake":
        if engine is None:
            raise ValueError(
                "backend 'mooncake' needs engine=<initialized MooncakeCheckpointEngine> "
                "(CheckpointEngineRegistry.new('mooncake', ...) with an RDMA-capable "
                "TransferEngine session)"
            )
        return MooncakeP2PBackend(
            engine, staging_device=staging_device, rollout_dtype=rollout_dtype, chunk_bytes=chunk_bytes
        )
    raise ValueError(f"unknown p2p backend {name!r}; available: fake, kimi, mooncake")


#: backend name → class; ``make_p2p_backend`` is the switching entry point,
#: this table is for introspection (e.g. CLI help, config validation)
P2P_BACKENDS: dict[str, type[P2PWeightBackend]] = {
    "fake": FakeP2PBackend,
    "kimi": KimiP2PBackend,
    "mooncake": MooncakeP2PBackend,
}
