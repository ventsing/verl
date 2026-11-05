# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Single-node CPU parameter relay for independently updating rollout replicas.

Immutable numpy buffers live in Ray's shared-memory object store. This is a
portable single-node transport, not Laminar's multi-node UCX/RDMA broadcast.
Publication never calls a rollout or waits for its requests. A pinned manifest
keeps a version alive until all TP ranks have finished loading it.
"""

import asyncio
import time
from uuid import uuid4

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.workers.rollout.utils import ensure_async_iterator


class CPUWeightRelay:
    """Atomic publication and lease-based retention; wrap in a CPU Ray actor."""

    def __init__(self, keep_last=2, max_versions=0, max_bytes=0, capacity_timeout_s=600):
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        if max_versions and max_versions < keep_last + 1:
            raise ValueError("Weight budget must allow keep_last plus one incoming version")
        if max_bytes < 0 or max_versions < 0 or capacity_timeout_s <= 0:
            raise ValueError("Invalid relay capacity limits")
        self.max_versions, self.max_bytes = max_versions, max_bytes
        self.capacity_timeout_s = capacity_timeout_s
        self.capacity_waiters = 0
        self.capacity_wait_seconds = 0.0
        self.keep_last = keep_last
        self.versions = {}
        self.leases = {}
        self.owners = {}
        self.latest = None

    def resident_bytes(self):
        return sum(m["nbytes"] for m in self.versions.values())

    def resident_versions(self):
        return set(self.versions)

    async def wait_for_capacity(self, version, nbytes=0, reserve_copies=1):
        # The fixed model layout lets later exports wait before full_tensor()
        # and before creating new Ray objects. The first export checks each tensor.
        estimate = max(nbytes, self.versions[self.latest]["nbytes"] if self.latest in self.versions else 0)
        incoming = estimate * reserve_copies
        minimum = sum(self.versions[v]["nbytes"] for v in sorted(self.versions)[-self.keep_last :]) + incoming
        if self.max_bytes and minimum > self.max_bytes:
            raise ValueError("Weight byte budget cannot hold keep_last plus incoming/staging copies")
        started = time.monotonic()
        self.capacity_waiters += 1
        try:
            async with asyncio.timeout(self.capacity_timeout_s):
                while True:
                    if hasattr(self, "health"):
                        self.health()
                    self._evict()
                    count = len(self.resident_versions() | {version})
                    if (not self.max_versions or count <= self.max_versions) and (
                        not self.max_bytes or self.resident_bytes() + incoming <= self.max_bytes
                    ):
                        return estimate
                    await asyncio.sleep(0.05)
        finally:
            self.capacity_waiters -= 1
            self.capacity_wait_seconds += time.monotonic() - started

    def commit(self, manifest):
        version = manifest["version"]
        if self.latest is not None and version <= self.latest:
            raise ValueError(f"Publication must increase version: {version} <= {self.latest}")
        if self.max_versions and len(self.resident_versions() | {version}) > self.max_versions:
            raise RuntimeError("Relay version budget exceeded; publisher must await capacity")
        if self.max_bytes and sum(m["nbytes"] for m in self.versions.values()) + manifest["nbytes"] > self.max_bytes:
            raise RuntimeError("Relay byte budget exceeded; publisher must await capacity")
        if not manifest["tensors"]:
            raise ValueError("Cannot publish an empty model")
        self.versions[version] = manifest
        self.latest = version
        self._evict()
        return {"version": version, "nbytes": manifest["nbytes"]}

    def acquire(self, version=None):
        version = self.latest if version is None else version
        if version not in self.versions:
            raise KeyError(f"Weight version {version} is not retained")
        lease = uuid4().hex
        self.leases[lease] = version
        return {**self.versions[version], "lease": lease}

    def release(self, lease):
        self.leases.pop(lease, None)
        self._evict()

    def retain(self, version, owner):
        if owner in self.owners and self.owners[owner] != version:
            raise ValueError("Retention owner cannot change weight version")
        self.owners[owner] = version

    def release_owner(self, owner):
        self.owners.pop(owner, None)
        self._evict()

    def _evict(self):
        retained = set(sorted(self.versions)[-self.keep_last :]) | set(self.leases.values()) | set(self.owners.values())
        for version in list(self.versions):
            if version not in retained:
                del self.versions[version]

    def status(self):
        return {
            "latest": self.latest,
            "max_versions": self.max_versions,
            "max_bytes": self.max_bytes,
            "accounted_bytes": self.resident_bytes(),
            "capacity_waiters": self.capacity_waiters,
            "capacity_wait_seconds": self.capacity_wait_seconds,
            "versions": sorted(self.versions),
            "leases": len(self.leases),
            "retention_owners": len(self.owners),
            "nbytes": sum(m["nbytes"] for m in self.versions.values()),
        }


@CheckpointEngineRegistry.register("cpu_relay")
class CPUSharedMemoryCheckpointEngine(CheckpointEngine):
    """Export full FSDP tensors once; consumers load without a collective."""

    def __init__(self, bucket_size, is_master=False, **kwargs):
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.relay = None
        self.manifest = None

    def configure(self, relay):
        self.relay = relay

    def set_manifest(self, manifest):
        self.manifest = manifest

    def prepare(self):
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        raise RuntimeError("cpu_relay uses independent pulls, not collective topology")

    def init_process_group(self, **kwargs):
        raise RuntimeError("cpu_relay does not create a weight-transfer process group")

    def finalize(self):
        self.manifest = None

    async def send_weights(self, weights, global_steps=None):
        if self.relay is None:
            raise RuntimeError("Configure the CPU relay before publication")
        started = time.monotonic()
        tensors = []
        nbytes = 0
        # full_tensor() is collective. Backpressure must hold every export rank
        # on the CPU, otherwise peers enqueue ALLGATHER while the master waits
        # for rollout owners to release old snapshots and hit the HCCL timeout.
        reservation = await self.relay.wait_for_capacity.remote(global_steps)
        # All FSDP ranks must exhaust the iterator: full_tensor() is collective.
        async for name, tensor in ensure_async_iterator(weights):
            estimate = tensor.numel() * (2 if tensor.is_floating_point() else tensor.element_size())
            if nbytes + estimate > reservation:
                reservation = await self.relay.wait_for_capacity.remote(global_steps, nbytes + estimate)
            if self.is_master:
                dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
                snapshot = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
                payload = snapshot.reshape(-1).view(torch.uint8).numpy()
                tensors.append(
                    {
                        "name": name,
                        "shape": tuple(snapshot.shape),
                        "dtype": str(dtype).removeprefix("torch."),
                        "ref": ray.put(payload),
                    }
                )
            nbytes += estimate
        if not self.is_master:
            return {}
        await self.relay.commit.remote({"version": global_steps, "tensors": tensors, "nbytes": nbytes})
        return {"laminar/publish_seconds": time.monotonic() - started, "laminar/publish_bytes": nbytes}

    async def receive_weights(self, global_steps=None):
        if self.manifest is None or self.manifest["version"] != global_steps:
            raise RuntimeError("Consumer must pin the exact weight manifest before loading")
        for index, entry in enumerate(self.manifest["tensors"]):
            for attempt in range(4):
                try:
                    payload = await entry["ref"]
                    break
                except (ray.exceptions.RayActorError, ray.exceptions.ObjectLostError, ray.exceptions.RaySystemError):
                    endpoint = self.manifest.get("recovery_endpoint")
                    if endpoint is None or attempt == 3:
                        raise
                    refreshed = await endpoint.renew_manifest.remote(global_steps, self.manifest["lease"])
                    replacement = refreshed["tensors"][index]
                    fields = ("name", "shape", "dtype", "sha256")
                    if any(
                        (tuple(replacement[k]) != tuple(entry[k]))
                        if k == "shape"
                        else replacement.get(k) != entry.get(k)
                        for k in fields
                    ):
                        raise ValueError("Recovered manifest changed exact policy content") from None
                    self.manifest = refreshed
                    entry = replacement
            # Ray numpy buffers are immutable. tensor() deliberately makes a
            # writable copy before handing the storage to device/IPC adapters.
            weight = torch.tensor(payload).view(getattr(torch, entry["dtype"])).reshape(entry["shape"])
            if weight.nbytes > self.bucket_size:
                raise ValueError(f"CPU relay tensor {entry['name']} exceeds the configured weight-transfer bucket")
            # The vLLM adapter copies into its own CPU-SHM or device-IPC
            # bucket. Staging here on NPU would add a needless CPU→NPU→CPU
            # round trip on Ascend installations that use shared memory.
            yield entry["name"], weight
