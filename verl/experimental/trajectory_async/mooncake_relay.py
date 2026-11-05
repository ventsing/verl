# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Node-local CPU snapshots, connected by a Mooncake chunk pipeline.

Ray carries control/metadata. Weight bytes cross nodes only through Mooncake.
The master is colocated with actor rank zero; each consumer reads its own
node's Ray shared-memory snapshot. No global rollout collective is created.
"""

import asyncio
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.experimental.trajectory_async.cpu_relay import CPUSharedMemoryCheckpointEngine, CPUWeightRelay
from verl.workers.rollout.utils import ensure_async_iterator


def worker_node_info():
    return {"node_id": ray.get_runtime_context().get_node_id(), "host": ray.util.get_node_ip_address().strip("[]")}


def validate_schema(schema):
    if not schema or len({x["name"] for x in schema}) != len(schema):
        raise ValueError("A version requires nonempty, uniquely named tensors")
    for entry in schema:
        dtype = getattr(torch, entry["dtype"], None)
        shape = entry["shape"]
        if not isinstance(dtype, torch.dtype) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid tensor shape or dtype")
        if entry["nbytes"] != math.prod(shape) * torch.empty((), dtype=dtype).element_size():
            raise ValueError("Tensor byte count does not match shape/dtype")


class MooncakeTransport:
    """Use the installed TransferEngine API; never silently switch protocols."""

    def __init__(self, host, protocol="rdma", device_name="", engine=None):
        if protocol not in {"rdma", "tcp"}:
            raise ValueError("CPU relays support explicit rdma or tcp transport")
        if engine is None:
            # Classic Mooncake auto-discovery may otherwise ignore the Python
            # protocol argument. These affect only this dedicated relay process.
            force, other = ("MC_FORCE_TCP", "MC_FORCE_HCA") if protocol == "tcp" else ("MC_FORCE_HCA", "MC_FORCE_TCP")
            os.environ[force] = "1"
            os.environ.pop(other, None)
            try:
                from mooncake.engine import TransferEngine
            except ImportError as error:
                raise RuntimeError(
                    "Mooncake CPU TransferEngine import failed; provide a CPU-compatible build in the relay runtime_env"
                ) from error
            engine = TransferEngine()
        self.engine = engine
        self.protocol = protocol
        ret = engine.initialize(host, "P2PHANDSHAKE", protocol, device_name)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake CPU {protocol} initialization failed: {ret}. Use a CPU-compatible Mooncake build; "
                "some Ascend builds unconditionally initialize device transport. No protocol fallback was applied."
            )
        self.session = f"{host}:{engine.get_rpc_port()}"

    def register(self, buffer):
        if buffer.nbytes:
            ret = self.engine.register_memory(buffer.ctypes.data, buffer.nbytes)
            if ret != 0:
                raise RuntimeError(f"Mooncake register_memory failed: {ret}")

    def unregister(self, buffer):
        if buffer.nbytes:
            ret = self.engine.unregister_memory(buffer.ctypes.data)
            if ret != 0:
                raise RuntimeError(f"Mooncake unregister_memory failed: {ret}")

    def write(self, session, buffer, source_offset, destination, size):
        ret = self.engine.transfer_sync_write(session, buffer.ctypes.data + source_offset, destination, size)
        if ret != 0:
            raise RuntimeError(f"Mooncake transfer_sync_write failed: {ret}")


@dataclass
class _Transfer:
    version: int
    schema: list
    buffers: list
    registered: list = field(default_factory=list)
    received: list = field(default_factory=list)
    tail: object = None
    prepare: object = None
    task: object = None
    sealed: bool = False
    shared_source: bool = False
    replay: bool = False
    lock: object = field(default_factory=asyncio.Lock)


class MooncakeRelayNode(CPUWeightRelay):
    """One CPU relay per node; independent local commits and background fanout.

    Transfers are ordered across versions and pipelined across chain hops.
    Native buffers remain registered until downstream transfer acknowledgement;
    visible snapshots have separate immutable Ray ownership and read leases.
    This is fail-closed transport, not automatic chain/master failover.
    """

    def __init__(
        self,
        keep_last=2,
        chunk_bytes=64 << 20,
        protocol="rdma",
        device_name="",
        verify_checksums=True,
        acquire_timeout_s=300,
        max_pending_versions=2,
        max_versions=0,
        max_bytes=0,
        capacity_timeout_s=600,
        node_id=None,
        host=None,
        engine=None,
    ):
        super().__init__(keep_last, max_versions, max_bytes, capacity_timeout_s)
        if chunk_bytes < 1 or acquire_timeout_s <= 0 or max_pending_versions < 1:
            raise ValueError("Chunk size, timeout and pending-version bound must be positive")
        info = worker_node_info() if node_id is None or host is None else {"node_id": node_id, "host": host}
        self.node_id, self.host = info["node_id"], info["host"]
        self.transport = MooncakeTransport(self.host, protocol, device_name, engine)
        self.chunk_bytes, self.verify_checksums = chunk_bytes, verify_checksums
        self.timeout, self.max_pending = acquire_timeout_s, max_pending_versions
        self.successor = None
        self.incarnation = uuid4().hex
        self.epoch = None
        self.revision = -1
        self.cache = None
        self.transfers = {}
        self.last_broadcast = None
        self.broadcast_completed = -1
        self.failure = None
        self.closed = False
        self.condition = asyncio.Condition()
        self.publish_lock = asyncio.Lock()
        self.quarantined = []
        self.stats = {"sent_bytes": 0, "received_bytes": 0, "sent_chunks": 0}

    def configure_successor(self, successor):
        if self.transfers or self.latest is not None:
            raise RuntimeError("Relay topology must be configured before publication")
        self.successor = successor

    def node_info(self):
        return {
            "node_id": self.node_id,
            "host": self.host,
            "session": self.transport.session,
            "incarnation": self.incarnation,
            "epoch": self.epoch,
        }

    def _check_epoch(self, epoch):
        if self.epoch is not None and epoch != self.epoch:
            raise ValueError("Stale Mooncake topology epoch")

    def _epoch_kwargs(self):
        return {"epoch": self.epoch} if self.epoch is not None else {}

    def configure_recovery(self, epoch, successor, cache, manifests, retained, revision=0):
        if self.epoch is not None or self.transfers or self.latest is not None:
            raise RuntimeError("Only a fresh native engine can join a recovery epoch")
        self.epoch, self.successor, self.cache = epoch, successor, cache
        self.revision = revision
        self.versions = dict(manifests)
        self.latest = max(self.versions, default=None)
        self.owners = {f"catalog:{v}": v for v in retained}

    def set_retained(self, epoch, retained, revision=0):
        self._check_epoch(epoch)
        if self.epoch is None:
            raise RuntimeError("A fresh native engine must first join the topology epoch")
        if revision < self.revision:
            return False
        self.revision = revision
        self.owners = {f"catalog:{v}": v for v in retained}
        for version in list(self.versions):
            if version not in retained and version not in self.transfers:
                del self.versions[version]

    async def _cached_transfer(self, version):
        manifest = self.versions[version]
        schema = [
            {**{k: entry[k] for k in ("name", "dtype", "nbytes", "sha256")}, "shape": tuple(entry["shape"])}
            for entry in manifest["tensors"]
        ]
        buffers = [
            await entry["ref"] if isinstance(entry["ref"], ray.ObjectRef) else entry["ref"]
            for entry in manifest["tensors"]
        ]
        state = await self._new_transfer(version, schema, buffers, replay=True, reuse=True)
        state.received = [entry["nbytes"] for entry in schema]
        state.sealed = True
        return state

    async def replay(self, version, epoch):
        self._check_epoch(epoch)
        self.health()
        state = await self._cached_transfer(version)
        state.task = asyncio.create_task(self._finish(state, source=True))
        state.task.add_done_callback(self._consume_exception)
        self.last_broadcast = state.task
        await state.task
        return {"version": version, "replayed": True}

    def health(self):
        if self.closed or self.failure:
            raise RuntimeError(f"Mooncake relay unavailable: {self.failure or 'closed'}")
        return True

    async def _failed(self, error):
        self.failure = self.failure or repr(error)
        async with self.condition:
            self.condition.notify_all()

    @staticmethod
    async def _native_call(method, *args):
        # Cancellation cannot release memory while a blocking native call still
        # owns its pointer. Join the thread before propagating cancellation.
        task = asyncio.create_task(asyncio.to_thread(method, *args))
        try:
            return await asyncio.shield(task)
        finally:
            if not task.done():
                await task

    def resident_versions(self):
        return set(self.versions) | set(self.transfers)

    def resident_bytes(self):
        amount = super().resident_bytes() + sum(b.nbytes for b in self.quarantined)
        for state in self.transfers.values():
            size = sum(e["nbytes"] for e in state.schema)
            if state.shared_source:
                amount += 0 if state.version in self.versions else size
            else:
                # Reserve the imminent Ray copy as well as the native receive
                # buffers, even while the incoming version is still incomplete.
                amount += size * (1 if state.sealed else 2)
        return amount

    async def _new_transfer(self, version, schema, buffers=None, replay=False, reuse=False):
        self.health()
        validate_schema(schema)
        if type(version) is not int or version < 0 or version in self.transfers:
            raise ValueError("Invalid or duplicate transfer version")
        if not replay and self.latest is not None and version <= self.latest:
            raise ValueError("Relay versions must increase monotonically")
        shared_source = buffers is not None
        if not reuse:
            await self.wait_for_capacity(
                version, sum(e["nbytes"] for e in schema), reserve_copies=1 if shared_source else 2
            )
        buffers = buffers if buffers is not None else [np.empty(x["nbytes"], dtype=np.uint8) for x in schema]
        state = _Transfer(
            version, schema, buffers, received=[0] * len(schema), shared_source=shared_source, replay=replay
        )
        self.transfers[version] = state
        try:
            for buffer in buffers:
                registration = asyncio.create_task(asyncio.to_thread(self.transport.register, buffer))
                try:
                    await asyncio.shield(registration)
                finally:
                    if not registration.done():
                        await registration
                    if not registration.cancelled() and registration.exception() is None:
                        state.registered.append(buffer)
            self.health()
        except BaseException as error:
            await self._failed(error)
            await self._free_buffers(state)
            self.transfers.pop(version, None)
            raise
        return state

    async def begin(self, version, schema, epoch=None, replay=False):
        """Allocate/register destinations; this RPC contains no weight bytes."""
        self._check_epoch(epoch)
        async with self.publish_lock:
            if replay and version in self.versions:
                state = await self._cached_transfer(version)
                if state.schema != schema:
                    raise ValueError("Cached recovery schema does not match source")
                state.task = asyncio.create_task(self._finish(state, source=True))
                state.task.add_done_callback(self._consume_exception)
                return {"session": self.transport.session, "addresses": [], "cached": True}
            state = await self._new_transfer(version, schema, replay=replay)
            if self.successor is not None:
                state.prepare = asyncio.create_task(self._prepare_next(state))
                state.prepare.add_done_callback(self._consume_exception)
            return {"session": self.transport.session, "addresses": [b.ctypes.data for b in state.buffers]}

    async def _prepare_next(self, state):
        return await self.successor.begin.remote(
            state.version,
            state.schema,
            **({"epoch": self.epoch, "replay": state.replay} if self.epoch is not None else {}),
        )

    @staticmethod
    def _consume_exception(task):
        if not task.cancelled():
            task.exception()

    async def commit(self, manifest, epoch=None):
        """Master accepts a local immutable snapshot, then returns before fanout."""
        self._check_epoch(epoch)
        async with self.publish_lock:
            self.health()
            if manifest.get("node_id") != self.node_id:
                raise ValueError("Actor publisher must share the master relay node")
            while len(self.transfers) >= self.max_pending:
                pending = next(iter(self.transfers.values()))
                if pending.task is None:
                    raise RuntimeError("Master has an unfinished incoming transfer")
                await asyncio.shield(pending.task)
                self.health()
            resolving = time.monotonic()
            schema, buffers, refs = [], [], []
            for entry in manifest["tensors"]:
                ref = entry["ref"]
                buffer = await ref if isinstance(ref, ray.ObjectRef) else ref
                if not isinstance(buffer, np.ndarray) or buffer.dtype != np.uint8 or not buffer.flags.c_contiguous:
                    raise ValueError("Publication requires contiguous uint8 CPU snapshots")
                schema.append(
                    {
                        "name": entry["name"],
                        "shape": tuple(entry["shape"]),
                        "dtype": entry["dtype"],
                        "nbytes": buffer.nbytes,
                        **({"sha256": entry["sha256"]} if "sha256" in entry else {}),
                    }
                )
                buffers.append(buffer.reshape(-1))
                refs.append(ref)
            if sum(e["nbytes"] for e in schema) != manifest["nbytes"]:
                raise ValueError("Manifest byte count mismatch")
            resolve_seconds = time.monotonic() - resolving
            preparing = time.monotonic()
            state = await self._new_transfer(manifest["version"], schema, buffers)
            prepare_seconds = time.monotonic() - preparing
            state.received = [e["nbytes"] for e in schema]
            installing = time.monotonic()
            cache_reply = await self._local_commit(state, refs)
            install_seconds = time.monotonic() - installing
            previous = self.last_broadcast
            state.task = asyncio.create_task(self._finish(state, previous, source=True))
            state.task.add_done_callback(self._consume_exception)
            self.last_broadcast = state.task
            return {
                "version": state.version,
                "nbytes": manifest["nbytes"],
                "timings": {
                    "native_resolve_seconds": resolve_seconds,
                    "native_prepare_seconds": prepare_seconds,
                    "endpoint_install_seconds": install_seconds,
                    "endpoint_hash_seconds": (cache_reply or {}).get("hash_seconds", 0.0),
                    "endpoint_checksums_reused": float((cache_reply or {}).get("checksums_reused", False)),
                },
            }

    async def _local_commit(self, state, refs=None):
        if refs is None:
            # A node-local object-store copy makes consumer ownership independent
            # of registered native transfer buffers and their reclamation.
            refs = [
                ray.put(buffer, _owner=self.cache) if self.cache is not None else ray.put(buffer)
                for buffer in state.buffers
            ]
        tensors = [{**entry, "ref": ref} for entry, ref in zip(state.schema, refs, strict=True)]
        CPUWeightRelay.commit(
            self,
            {
                "version": state.version,
                "node_id": self.node_id,
                "tensors": tensors,
                "nbytes": sum(x["nbytes"] for x in state.schema),
            },
        )
        cache_reply = {}
        if self.cache is not None:
            cache_reply = await self.cache.install.remote(self.versions[state.version], self.epoch)
        state.sealed = True
        async with self.condition:
            self.condition.notify_all()
        return cache_reply

    async def chunk_ready(self, version, index, offset, size, digest=None, epoch=None):
        """Acknowledge a completed native write, then queue next-hop transfer."""
        self._check_epoch(epoch)
        self.health()
        state = self.transfers[version]
        async with state.lock:
            self.health()
            if state.sealed or not 0 <= index < len(state.schema):
                raise ValueError("Chunk targets a sealed version or invalid tensor")
            expected = min(self.chunk_bytes, state.schema[index]["nbytes"] - state.received[index])
            if offset != state.received[index] or size != expected or size <= 0:
                raise ValueError("Duplicate, overlapping or incorrectly sized Mooncake chunk")
            if any(state.received[i] != state.schema[i]["nbytes"] for i in range(index)):
                raise ValueError("Chunks must arrive in tensor order")
            if self.verify_checksums:
                actual = await asyncio.to_thread(self._digest, state.buffers[index], offset, size)
                if digest != actual:
                    error = RuntimeError("Mooncake chunk checksum mismatch")
                    await self._failed(error)
                    raise error
            state.received[index] += size
            self.stats["received_bytes"] += size
            if self.successor is not None:
                previous = state.tail
                state.tail = asyncio.create_task(self._forward_chunk(state, previous, index, offset, size, digest))
                state.tail.add_done_callback(self._consume_exception)

    @staticmethod
    def _digest(buffer, offset, size):
        return hashlib.sha256(memoryview(buffer)[offset : offset + size]).hexdigest()

    async def _forward_chunk(self, state, previous, index, offset, size, digest):
        try:
            if previous is not None:
                await previous
            destination = await state.prepare
            if destination.get("cached"):
                return
            await self._native_call(
                self.transport.write,
                destination["session"],
                state.buffers[index],
                offset,
                destination["addresses"][index] + offset,
                size,
            )
            self.stats["sent_bytes"] += size
            self.stats["sent_chunks"] += 1
            await self.successor.chunk_ready.remote(state.version, index, offset, size, digest, **self._epoch_kwargs())
        except BaseException as error:
            await self._failed(error)
            raise

    async def seal(self, version, epoch=None):
        self._check_epoch(epoch)
        self.health()
        if self.epoch is not None and (version not in self.transfers or self.transfers[version].sealed):
            if version in self.versions:
                return {"version": version, "existing": True}
        state = self.transfers[version]
        async with state.lock:
            self.health()
            if state.sealed or any(n != e["nbytes"] for n, e in zip(state.received, state.schema, strict=True)):
                raise ValueError("Cannot commit incomplete or already sealed transfer")
            await self._local_commit(state)
            state.task = asyncio.create_task(self._finish(state))
            state.task.add_done_callback(self._consume_exception)
            async with self.condition:
                self.condition.notify_all()
        return {"version": version}

    async def _finish(self, state, previous=None, source=False):
        try:
            if previous is not None:
                await previous
            if source and self.successor is not None:
                state.prepare = asyncio.create_task(self._prepare_next(state))
                destination = await state.prepare
                for index, entry in enumerate(state.schema):
                    if destination.get("cached"):
                        break
                    for offset in range(0, entry["nbytes"], self.chunk_bytes):
                        size = min(self.chunk_bytes, entry["nbytes"] - offset)
                        digest = (
                            await asyncio.to_thread(self._digest, state.buffers[index], offset, size)
                            if self.verify_checksums
                            else None
                        )
                        await self._forward_chunk(state, None, index, offset, size, digest)
            elif state.tail is not None:
                await state.tail
            if self.successor is not None:
                await state.prepare
                # Local commit and downstream completion are deliberately distinct.
                await self.successor.seal.remote(state.version, **self._epoch_kwargs())
                await self.successor.wait_distributed.remote(state.version, **self._epoch_kwargs())
            self.broadcast_completed = max(self.broadcast_completed, state.version)
        except BaseException as error:
            await self._failed(error)
            raise
        finally:
            # Never cancel a to_thread native transfer and free its source early.
            await self._free_buffers(state)
            self.transfers.pop(state.version, None)
            async with self.condition:
                self.condition.notify_all()

    async def _free_buffers(self, state):
        for buffer in state.registered:
            try:
                await self._native_call(self.transport.unregister, buffer)
            except Exception as error:
                self.quarantined.append(buffer)  # Keep failed registrations alive until process teardown.
                await self._failed(error)
        state.registered.clear()
        state.buffers.clear()

    async def wait_distributed(self, version, epoch=None):
        self._check_epoch(epoch)
        async with asyncio.timeout(self.timeout):
            while self.broadcast_completed < version:
                self.health()
                async with self.condition:
                    state = self.transfers.get(version)
                    if state is None or state.task is None:
                        await self.condition.wait()
                        continue
                await asyncio.shield(state.task)
        self.health()
        return {"version": version, "distributed": True}

    async def acquire(self, version=None):
        async with asyncio.timeout(self.timeout):
            async with self.condition:
                while True:
                    self.health()
                    target = self.latest if version is None else version
                    if target in self.versions:
                        return CPUWeightRelay.acquire(self, target)
                    if target is not None and self.latest is not None and target < self.latest:
                        raise KeyError(f"Requested weight version {target} has been evicted")
                    await self.condition.wait()

    def status(self):
        return {
            **super().status(),
            **self.stats,
            "node_id": self.node_id,
            "broadcast_completed": self.broadcast_completed,
            "pending_versions": sorted(self.transfers),
            "failure": self.failure,
            "protocol": self.transport.protocol,
        }

    async def close(self):
        self.closed = True
        async with self.condition:
            self.condition.notify_all()
        async with self.publish_lock:
            for state in list(self.transfers.values()):
                async with state.lock:
                    # Teardown is upstream-first. Wait also for registration and
                    # checksum RPCs before joining transfers and releasing memory.
                    tasks = [t for t in (state.task, state.tail, state.prepare) if t is not None]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await self._free_buffers(state)
        self.transfers.clear()
        self.versions.clear()
        return {"quarantined_buffers": len(self.quarantined), "failure": self.failure}


@CheckpointEngineRegistry.register("mooncake_relay")
class MooncakeRelayCheckpointEngine(CPUSharedMemoryCheckpointEngine):
    """Actor-to-local-master publication and local-only rollout consumption."""

    def node_info(self):
        return worker_node_info()

    def configure(self, relay, root_node_id):
        if self.is_master and self.node_info()["node_id"] != root_node_id:
            raise ValueError("Master relay must be colocated with actor rank zero")
        self.relay = relay

    def set_manifest(self, manifest):
        if manifest.get("node_id") != self.node_info()["node_id"]:
            raise ValueError("Rollout must consume its own node's relay snapshot")
        super().set_manifest(manifest)

    async def send_weights(self, weights, global_steps=None):
        if self.relay is None:
            raise RuntimeError("Configure the Mooncake relay before publication")
        started, tensors, nbytes = time.monotonic(), [], 0
        capacity_seconds = cpu_seconds = put_seconds = iterator_seconds = 0.0
        waiting = time.monotonic()
        # Every rank must wait before advancing the collective FSDP iterator;
        # a master-only CPU wait leaves its peers inside HCCL ALLGATHER.
        reservation = await self.relay.wait_for_capacity.remote(global_steps)
        capacity_seconds += time.monotonic() - waiting
        exporting = previous = time.monotonic()
        export_capacity = 0.0
        async for name, tensor in ensure_async_iterator(weights):
            iterator_seconds += time.monotonic() - previous
            estimate = tensor.numel() * (2 if tensor.is_floating_point() else tensor.element_size())
            if nbytes + estimate > reservation:
                waiting = time.monotonic()
                reservation = await self.relay.wait_for_capacity.remote(global_steps, nbytes + estimate)
                duration = time.monotonic() - waiting
                capacity_seconds += duration
                export_capacity += duration
            if self.is_master:
                dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                copying = time.monotonic()
                snapshot = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
                payload = snapshot.reshape(-1).view(torch.uint8).numpy()
                cpu_seconds += time.monotonic() - copying
                putting = time.monotonic()
                reference = ray.put(payload)
                put_seconds += time.monotonic() - putting
                tensors.append(
                    {
                        "name": name,
                        "shape": tuple(snapshot.shape),
                        "dtype": str(dtype).removeprefix("torch."),
                        "ref": reference,
                    }
                )
            nbytes += estimate
            previous = time.monotonic()
        if not self.is_master:
            return {}
        export_seconds = time.monotonic() - exporting - export_capacity
        committing = time.monotonic()
        response = await self.relay.commit.remote(
            {"version": global_steps, "tensors": tensors, "nbytes": nbytes, "node_id": self.node_info()["node_id"]}
        )
        metrics = {
            "laminar/publish_seconds": time.monotonic() - started,
            "laminar/publish_bytes": nbytes,
            "laminar/publish_capacity_seconds": capacity_seconds,
            "laminar/publish_export_seconds": export_seconds,
            "laminar/publish_iterator_seconds": iterator_seconds,
            "laminar/publish_cpu_copy_seconds": cpu_seconds,
            "laminar/publish_ray_put_seconds": put_seconds,
            "laminar/publish_relay_seconds": time.monotonic() - committing,
            "laminar/publish_tensors": len(tensors),
        }
        metrics.update(
            {f"laminar/publish_{name}": value for name, value in (response or {}).get("timings", {}).items()}
        )
        return metrics


def create_relay_hierarchy(actor_node, replica_nodes, options):
    """Hard-pin relays to actual worker nodes; return chain and per-replica routes."""
    nodes = {actor_node["node_id"]: actor_node}
    for node in replica_nodes:
        nodes.setdefault(node["node_id"], node)
    options = dict(options)
    runtime_env = options.pop("runtime_env", {})
    runtime_env = (
        OmegaConf.to_container(runtime_env, resolve=True) if OmegaConf.is_config(runtime_env) else dict(runtime_env)
    )
    relays = []
    try:
        for node in nodes.values():
            relay = (
                ray.remote(MooncakeRelayNode)
                .options(
                    num_cpus=1,
                    max_concurrency=64,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["node_id"], soft=False),
                    runtime_env=runtime_env,
                )
                .remote(**options, node_id=node["node_id"], host=node["host"])
            )
            relays.append(relay)
        ray.get([relay.node_info.remote() for relay in relays])  # Fail initialization before configuring workers.
        ray.get(
            [
                relay.configure_successor.remote(relays[i + 1] if i + 1 < len(relays) else None)
                for i, relay in enumerate(relays)
            ]
        )
    except Exception:
        for relay in relays:
            ray.kill(relay)
        raise
    by_node = dict(zip(nodes, relays, strict=True))
    return relays, [by_node[n["node_id"]] for n in replica_nodes]
