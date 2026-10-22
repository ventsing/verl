# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Process-fault recovery for the Mooncake relay data plane.

Stable node endpoints own snapshots independently of native TransferEngine
workers. A bounded in-memory catalog owns publication/retention/leases by default. A single
Ray-serialized coordinator fences epochs, elects an ingress standby and rebuilds
the chain. This is process failover, not consensus across a failed Ray cluster.
"""

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.experimental.trajectory_async.cpu_relay import CPUWeightRelay
from verl.experimental.trajectory_async.mooncake_relay import MooncakeRelayNode, validate_schema
from verl.experimental.trajectory_async.weight_snapshot import checksum_arrays, validate_hash_workers


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}")
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != data:
            raise OSError("Incomplete relay metadata write")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class RelaySnapshotStore:
    """One writer, immutable version directories and atomic catalog metadata."""

    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError("Relay recovery catalog already has a live writer") from error
        self.identity = identity
        self.metadata_path = self.path / "catalog.json"
        if self.metadata_path.exists():
            self.state = json.loads(self.metadata_path.read_text())
            if self.state["identity"] != identity or self.state["format"] != 1:
                self.close()
                raise ValueError("Relay recovery catalog belongs to another model/run")
        else:
            self.state = {
                "format": 1,
                "identity": identity,
                "epoch": 0,
                "latest": None,
                "versions": {},
                "owners": {},
                "leases": {},
                "master": 0,
            }
            self.save()
        # Staging directories were never acknowledged. No committed directory
        # is removed here; an unacknowledged full version can be retried safely.
        for entry in self.path.glob(".staging-*"):
            shutil.rmtree(entry)
        for entry in self.path.glob("version-*"):
            if entry.name.removeprefix("version-") not in self.state["versions"]:
                shutil.rmtree(entry)  # No catalog commit/ACK ever referenced it.
        self.state.setdefault("revision", 0)

    def save(self):
        atomic_json(self.metadata_path, self.state)

    def write_version(self, version, entries, arrays):
        if type(version) is not int or version < 0:
            raise ValueError("Weight version must be a nonnegative integer")
        schema = [{k: e[k] for k in ("name", "shape", "dtype", "nbytes")} for e in entries]
        validate_schema(schema)
        destination = self.path / f"version-{version}"
        staging = self.path / f".staging-{uuid4().hex}"
        staging.mkdir()
        try:
            result = []
            for index, (entry, array) in enumerate(zip(schema, arrays, strict=True)):
                if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or not array.flags.c_contiguous:
                    raise ValueError("Relay backup requires contiguous uint8 snapshots")
                if array.nbytes != entry["nbytes"]:
                    raise ValueError("Relay backup byte count mismatch")
                payload = memoryview(array.reshape(-1))
                digest = hashlib.sha256()
                filename = f"tensor-{index:06d}.bin"
                with (staging / filename).open("xb") as stream:
                    for offset in range(0, array.nbytes, 64 << 20):
                        chunk = payload[offset : offset + (64 << 20)]
                        if stream.write(chunk) != len(chunk):
                            raise OSError("Short relay snapshot write")
                        digest.update(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                result.append({**entry, "file": filename, "sha256": digest.hexdigest()})
            record = {"version": version, "tensors": result, "nbytes": sum(e["nbytes"] for e in result)}
            record["digest"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
            record = json.loads(json.dumps(record))
            atomic_json(staging / "manifest.json", record)
            if destination.exists():
                original = json.loads((destination / "manifest.json").read_text())
                if original != record:
                    raise ValueError("Conflicting publication for an existing weight version")
            else:
                os.replace(staging, destination)
                directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            return record
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def read_arrays(self, record):
        arrays = []
        for entry in record["tensors"]:
            path = self.path / f"version-{record['version']}" / entry["file"]
            if path.stat().st_size != entry["nbytes"]:
                raise ValueError("Truncated relay recovery snapshot")
            array = np.memmap(path, dtype=np.uint8, mode="r") if entry["nbytes"] else np.empty(0, dtype=np.uint8)
            if hashlib.sha256(memoryview(array)).hexdigest() != entry["sha256"]:
                raise ValueError("Corrupt relay recovery snapshot")
            arrays.append(array)
        return arrays

    def remove_version(self, version):
        path = self.path / f"version-{version}"
        if path.exists():
            shutil.rmtree(path)

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()


class MemorySnapshotStore:
    """Bounded by the coordinator's version/byte budget; never writes recovery data.

    Arrays are read-only Ray object-store views in production, not another full
    model copy. Their pins survive native/endpoint death while this actor lives.
    Loss of this catalog is fail-stop: no restart from an empty version history.
    """

    def __init__(self, path, identity, hash_workers=4):
        self.hash_workers = validate_hash_workers(hash_workers)
        self.path = Path(path)
        self.arrays = {}
        self.manifests = {}
        self.state = {
            "format": 1,
            "identity": identity,
            "epoch": 0,
            "latest": None,
            "versions": {},
            "owners": {},
            "leases": {},
            "master": 0,
            "revision": 0,
        }

    def write_version(self, version, entries, arrays):
        if type(version) is not int or version < 0:
            raise ValueError("Weight version must be a nonnegative integer")
        schema = [{k: e[k] for k in ("name", "shape", "dtype", "nbytes")} for e in entries]
        validate_schema(schema)
        result = []
        for entry, array in zip(schema, arrays, strict=True):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype != np.uint8
                or not array.flags.c_contiguous
                or array.nbytes != entry["nbytes"]
            ):
                raise ValueError("Relay snapshots require contiguous uint8 payloads with matching sizes")
        hashes = checksum_arrays(arrays, self.hash_workers)
        result = [{**entry, "sha256": digest} for entry, digest in zip(schema, hashes, strict=True)]
        record = {"version": version, "tensors": result, "nbytes": sum(e["nbytes"] for e in result)}
        record["digest"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        record = json.loads(json.dumps(record))
        old = self.state["versions"].get(str(version))
        if old is not None and old["digest"] != record["digest"]:
            raise ValueError("Conflicting publication for a committed version")
        # The coordinator pins arrays only inside the metadata commit lock.
        # Cancellation during hashing must not leak uncommitted model versions.
        return record

    def read_arrays(self, record):
        arrays = self.arrays.get(record["version"])
        if arrays is None:
            raise RuntimeError("Exact weight version is unavailable in memory; durable recovery is deferred")
        hashes = checksum_arrays(arrays, self.hash_workers)
        for entry, digest in zip(record["tensors"], hashes, strict=True):
            if digest != entry["sha256"]:
                raise ValueError("Corrupt in-memory relay snapshot")
        return arrays

    def remove_version(self, version):
        self.arrays.pop(version, None)
        self.manifests.pop(version, None)

    def close(self):
        self.arrays.clear()
        self.manifests.clear()


def task_lease_id():
    if ray.is_initialized():
        task_id = ray.get_runtime_context().get_task_id()
        return task_id.hex() if hasattr(task_id, "hex") else str(task_id)
    return uuid4().hex


class RelayEndpoint(CPUWeightRelay):
    """Stable, node-local object owner; never owns native engine allocations."""

    def __init__(self, node, coordinator_name, namespace, budget, keep_last, hash_workers=4):
        super().__init__(keep_last=keep_last, **budget)
        self.node = node
        self.hash_workers = validate_hash_workers(hash_workers)
        self.coordinator_name, self.namespace = coordinator_name, namespace
        self.coordinator = None
        self.incarnation = uuid4().hex
        self.epoch = None
        self.revision = -1
        self.accepted = set()

    def bind_coordinator(self, coordinator):
        self.coordinator = coordinator

    def _coordinator(self):
        if self.coordinator is None:
            self.coordinator = ray.get_actor(self.coordinator_name, namespace=self.namespace)
        return self.coordinator

    def node_info(self):
        return {**self.node, "incarnation": self.incarnation, "epoch": self.epoch}

    def sync(self, epoch, retained, revision=0):
        if self.epoch is not None and (epoch < self.epoch or epoch == self.epoch and revision < self.revision):
            return False
        self.epoch, self.revision = epoch, revision
        self.accepted = set(retained)
        self.owners = {f"catalog:{v}": v for v in retained}
        for version in list(self.versions):
            if version not in self.accepted:
                del self.versions[version]
        self.latest = max(self.versions, default=None)

    async def install(self, manifest, epoch):
        started = time.monotonic()
        if epoch != self.epoch or manifest["version"] not in self.accepted:
            raise ValueError("Stale relay epoch or uncommitted weight version")
        if manifest["node_id"] != self.node["node_id"]:
            raise ValueError("Relay snapshots must remain node local")
        version = manifest["version"]
        if version in self.versions:
            old = self.versions[version]

            def signature(entry):
                return (entry["name"], tuple(entry["shape"]), entry["dtype"], entry["nbytes"], entry["sha256"])

            if [signature(e) for e in old["tensors"]] != [signature(e) for e in manifest["tensors"]]:
                raise ValueError("Conflicting recovered weight version")
            return {"version": version, "existing": True}
        validate_schema(manifest["tensors"])
        # Ray ObjectRefs identify immutable objects. Only the catalog that
        # actually hashed these exact refs can certify them. New refs, mutable
        # arrays, changed schema/digests and network destinations still hash.
        reused = await self._coordinator().source_is_verified.remote(manifest, epoch)
        hash_seconds = 0.0
        if not reused:
            arrays = []
            for entry in manifest["tensors"]:
                payload = await entry["ref"] if isinstance(entry["ref"], ray.ObjectRef) else entry["ref"]
                if (
                    not isinstance(payload, np.ndarray)
                    or payload.dtype != np.uint8
                    or payload.nbytes != entry["nbytes"]
                ):
                    raise ValueError("Invalid recovered payload")
                arrays.append(payload)
            hashing = time.monotonic()
            hashes = await asyncio.to_thread(checksum_arrays, arrays, self.hash_workers)
            hash_seconds = time.monotonic() - hashing
            if any(digest != entry["sha256"] for digest, entry in zip(hashes, manifest["tensors"], strict=True)):
                raise ValueError("Recovered weight checksum mismatch")
        if epoch != self.epoch or version not in self.accepted:
            raise ValueError("Relay epoch changed during snapshot verification")
        if self.max_versions and len(self.versions) + 1 > self.max_versions:
            raise RuntimeError("Recovered cache exceeds version budget")
        if self.max_bytes and self.resident_bytes() + manifest["nbytes"] > self.max_bytes:
            raise RuntimeError("Recovered cache exceeds byte budget")
        self.versions[version] = manifest
        self.latest = max(self.versions)
        return {
            "version": version,
            "existing": False,
            "checksums_reused": bool(reused),
            "verify_seconds": time.monotonic() - started,
            "hash_seconds": hash_seconds,
        }

    def manifests(self):
        return dict(self.versions)

    def local_acquire(self, version):
        if version not in self.versions:
            raise KeyError(version)
        return self.versions[version]

    async def commit(self, manifest):
        return await self._coordinator().publish.remote(manifest)

    async def wait_for_capacity(self, version, nbytes=0):
        return await self._coordinator().wait_for_capacity.remote(version, nbytes)

    async def acquire(self, version=None):
        lease = f"{self.node['key']}:{task_lease_id()}"
        return await self._coordinator().acquire_for.remote(self.node["key"], version, lease)

    async def release(self, lease):
        return await self._coordinator().release.remote(lease)

    async def renew_manifest(self, version, lease):
        return await self._coordinator().acquire_for.remote(self.node["key"], version, lease)

    async def retain(self, version, owner):
        return await self._coordinator().retain.remote(version, owner)

    async def release_owner(self, owner):
        return await self._coordinator().release_owner.remote(owner)

    async def health(self):
        return await self._coordinator().health.remote()

    def status(self):
        return {**super().status(), **self.node_info(), "resilient": True}

    async def recovery_status(self):
        return await self._coordinator().status.remote()

    async def topology(self):
        return {**await self._coordinator().topology.remote(), "coordinator": self._coordinator()}

    async def close(self):
        return await self._coordinator().close.remote()


class RelayRecoveryCoordinator:
    """Single-writer publication control, surviving native/endpoint death.

    Memory-only by default; disk catalog reconstruction is an explicit opt-in.
    """

    def __init__(self, nodes, endpoints, pools, options, recovery, directory, identity):
        self.nodes, self.endpoints, self.pools = nodes, endpoints, pools
        self.options, self.recovery = dict(options), dict(recovery)
        self.persist_to_disk = self.recovery.get("persist_to_disk", False)
        self.hash_workers = validate_hash_workers(self.recovery.get("hash_workers", 4))
        self.reuse_verified_source = self.recovery.get("reuse_verified_source", True)
        self.store = (
            RelaySnapshotStore(directory, identity)
            if self.persist_to_disk
            else MemorySnapshotStore(directory, identity, self.hash_workers)
        )
        self.state = self.store.state
        self.lock = asyncio.Lock()
        self.operations = asyncio.Lock()
        self.ready = asyncio.Event()
        self.closed = False
        self.failure = None
        self.task = None
        self.repair = None
        self.active = {}
        self.infos = {}
        self.protected = set()
        self.distributing = set()
        self.broadcasts = set()
        self.waiters = 0
        self.started = time.monotonic()
        self.stats = {"recoveries": 0, "master_switches": 0}
        self.event_path = Path(directory) / "relay_events.jsonl"
        self.event_path.parent.mkdir(parents=True, exist_ok=True)

    def _event(self, event, **fields):
        with self.event_path.open("a") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), **fields}) + "\n")

    def _check(self):
        if self.closed or self.failure:
            raise RuntimeError(f"Relay recovery unavailable: {self.failure or 'closed'}")

    def _start(self):
        self._check()
        if self.task is None:
            self.task = asyncio.create_task(self._monitor())
            self._schedule_repair("startup or coordinator reconstruction")

    def _schedule_repair(self, reason):
        if self.repair is None or self.repair.done():
            self.ready.clear()
            self.repair = asyncio.create_task(self._recover(reason))

    async def _rpc(self, reference, timeout=None):
        return await asyncio.wait_for(reference, timeout or self.recovery.get("rpc_timeout_s", 10.0))

    async def _await_ready(self):
        self._start()
        async with asyncio.timeout(self.recovery.get("recovery_timeout_s", 900.0)):
            while not self.ready.is_set():
                self._check()
                try:
                    await asyncio.wait_for(self.ready.wait(), 0.1)
                except TimeoutError:
                    pass
        self._check()

    def _retained(self):
        versions = sorted(map(int, self.state["versions"]))
        return (
            set(versions[-self.options["keep_last"] :])
            | set(self.state["owners"].values())
            | set(self.state["leases"].values())
            | self.protected
            | self.distributing
        )

    async def _save(self):
        # Call only under the metadata lock. Write a detached copy so subsequent
        # coroutine mutations cannot race the background serializer.
        if self.persist_to_disk:
            state = json.loads(json.dumps(self.state))
            await asyncio.to_thread(atomic_json, self.store.metadata_path, state)

    async def _sync(self):
        async with self.lock:
            retained = list(map(int, self.state["versions"]))
            epoch, revision = self.state["epoch"], self.state["revision"]
        for key, endpoint in self.endpoints.items():
            if key not in self.active:
                continue
            await self._rpc(endpoint.sync.remote(epoch, retained, revision))
            await self._rpc(self.active[key].set_retained.remote(epoch, retained, revision))

    async def _gc(self):
        discarded = []
        async with self.lock:
            retained = self._retained()
            for key in list(self.state["versions"]):
                if int(key) not in retained:
                    discarded.append(int(key))
                    del self.state["versions"][key]
            if discarded:
                self.state["revision"] += 1
                await self._save()
        if discarded and self.ready.is_set():
            try:
                await self._sync()
            except Exception as error:
                self._schedule_repair(repr(error))
        for version in discarded:
            await asyncio.to_thread(self.store.remove_version, version)

    async def retain(self, version, owner):
        self._start()
        async with self.lock:
            if owner in self.state["owners"]:
                if self.state["owners"][owner] != version:
                    raise ValueError("Retention owner cannot change version")
                return
            self.state["owners"][owner] = version
            await self._save()
        if self.ready.is_set():
            try:
                await self._sync()
            except Exception as error:
                self._schedule_repair(repr(error))

    async def release_owner(self, owner):
        self._start()
        async with self.lock:
            if self.state["owners"].pop(owner, None) is not None:
                await self._save()
        await self._gc()

    async def release(self, lease):
        self._start()
        async with self.lock:
            if self.state["leases"].pop(lease, None) is not None:
                await self._save()
        await self._gc()

    async def wait_for_capacity(self, version, nbytes=0):
        await self._await_ready()
        latest = self.state["latest"]
        estimate = max(nbytes, self.state["versions"].get(str(latest), {}).get("nbytes", 0))
        self.waiters += 1
        try:
            async with asyncio.timeout(self.options.get("capacity_timeout_s", 600)):
                while True:
                    self._check()
                    await self._gc()
                    records = self.state["versions"]
                    if str(version) in records:
                        return estimate
                    count = len(set(map(int, records)) | {version})
                    size = sum(r["nbytes"] for r in records.values())
                    minimum = sum(
                        records[str(v)]["nbytes"] for v in sorted(map(int, records))[-self.options["keep_last"] :]
                    )
                    copies = 2 if len(self.nodes) > 1 else 1
                    cap = self.options.get("max_bytes", 0)
                    if cap and minimum + copies * estimate > cap:
                        raise ValueError("Relay budget cannot hold retained snapshots and receive staging")
                    if (not self.options.get("max_versions") or count <= self.options["max_versions"]) and (
                        not cap or size + copies * estimate <= cap
                    ):
                        return estimate
                    await asyncio.sleep(0.05)
        finally:
            self.waiters -= 1

    async def _arrays(self, manifest):
        arrays = []
        entries = []
        for entry in manifest["tensors"]:
            payload = await entry["ref"] if isinstance(entry["ref"], ray.ObjectRef) else entry["ref"]
            arrays.append(payload)
            entries.append({**entry, "nbytes": payload.nbytes})
        if sum(e["nbytes"] for e in entries) != manifest["nbytes"]:
            raise ValueError("Publication byte count mismatch")
        return entries, arrays

    def _verified_source(self, manifest, require_hash=True):
        if self.persist_to_disk or not self.reuse_verified_source:
            return False
        version = manifest.get("version")
        if type(version) is not int or manifest.get("node_id") != self.nodes[0]["node_id"]:
            return False
        record = self.state["versions"].get(str(version))
        original = self.store.manifests.get(version)
        if record is None or original is None or manifest.get("nbytes") != record["nbytes"]:
            return False
        current, saved, expected = manifest.get("tensors", []), original["tensors"], record["tensors"]
        if len(current) != len(saved) or len(current) != len(expected):
            return False
        for entry, source, schema in zip(current, saved, expected, strict=True):
            if (
                not isinstance(entry.get("ref"), ray.ObjectRef)
                or not isinstance(source.get("ref"), ray.ObjectRef)
                or entry["ref"] != source["ref"]
            ):
                return False
            if (
                entry.get("name") != schema["name"]
                or tuple(entry.get("shape", ())) != tuple(schema["shape"])
                or entry.get("dtype") != schema["dtype"]
                or entry.get("nbytes", schema["nbytes"]) != schema["nbytes"]
                or require_hash
                and entry.get("sha256") != schema["sha256"]
            ):
                return False
        return True

    def source_is_verified(self, manifest, epoch):
        # This method deliberately takes neither publication nor metadata locks:
        # the source core awaits endpoint install while publish owns operations.
        # Committed state is read atomically between this actor's await points.
        return epoch == self.state["epoch"] and self._verified_source(manifest)

    async def publish(self, manifest):
        if manifest.get("node_id") != self.nodes[0]["node_id"]:
            raise ValueError("Publisher must remain colocated with the ingress relay")
        version = manifest["version"]
        if type(version) is not int or version < 0:
            raise ValueError("Weight version must be a nonnegative integer")
        called_at = time.monotonic()
        await self.wait_for_capacity(version, manifest["nbytes"])
        for retry in range(self.recovery.get("max_retries", 3) + 1):
            await self._await_ready()
            try:
                async with self.operations:
                    self._check()
                    if (
                        self.state["latest"] is not None
                        and version <= self.state["latest"]
                        and str(version) not in self.state["versions"]
                    ):
                        raise ValueError("Weight versions must increase monotonically")
                    began = time.monotonic()
                    fetch_started = began
                    cached_record = self._verified_source(manifest, require_hash=False)
                    if cached_record:
                        arrays = self.store.arrays[version]
                        record = self.state["versions"][str(version)]
                        entries = None
                    else:
                        entries, arrays = await self._arrays(manifest)
                    fetch_seconds = time.monotonic() - fetch_started
                    self._event(
                        "backup_begin" if self.persist_to_disk else "snapshot_begin",
                        version=version,
                        nbytes=manifest["nbytes"],
                    )
                    hash_started = time.monotonic()
                    if not cached_record:
                        record = await asyncio.to_thread(self.store.write_version, version, entries, arrays)
                    hash_seconds = time.monotonic() - hash_started
                    self._event(
                        "backup_committed" if self.persist_to_disk else "snapshot_committed",
                        version=version,
                        nbytes=record["nbytes"],
                    )
                    control_started = time.monotonic()
                    async with self.lock:
                        original = self.state["versions"].get(str(version))
                        if original is not None and original["digest"] != record["digest"]:
                            raise ValueError("Conflicting publication for a committed version")
                        if not self.persist_to_disk and original is None:
                            self.store.arrays[version] = arrays
                            self.store.manifests[version] = manifest
                        if original is None:
                            if self.state["latest"] is not None and version <= self.state["latest"]:
                                raise ValueError("Weight versions must increase monotonically")
                            self.state["versions"][str(version)] = record
                            self.state["latest"] = version
                            self.state["revision"] += 1
                            self.distributing.add(version)
                            await self._save()  # Publication linearizes here, before the reply.
                    await self._sync()
                    root = self.nodes[0]["key"]
                    existing = await self._rpc(self.endpoints[root].manifests.remote())
                    control_seconds = time.monotonic() - control_started
                    commit_started = time.monotonic()
                    source_reply = {}
                    if version not in existing:
                        normalized = {
                            **manifest,
                            "tensors": [
                                {**e, "ref": source["ref"]}
                                for e, source in zip(record["tensors"], manifest["tensors"], strict=True)
                            ],
                        }
                        source_reply = await self._rpc(
                            self.active[root].commit.remote(normalized, epoch=self.state["epoch"]),
                            self.recovery.get("recovery_timeout_s", 900),
                        )
                    self.distributing.add(version)
                    broadcast = asyncio.create_task(
                        self._watch_broadcast(version, self.active[root], self.state["epoch"])
                    )
                    self.broadcasts.add(broadcast)
                    broadcast.add_done_callback(self.broadcasts.discard)
                    timings = {
                        "catalog_fetch_seconds": fetch_seconds,
                        "catalog_hash_seconds": hash_seconds,
                        "catalog_control_seconds": control_seconds,
                        "native_source_seconds": time.monotonic() - commit_started,
                        "catalog_wait_seconds": began - called_at,
                        "catalog_seconds": time.monotonic() - began,
                        "catalog_hash_reused": float(cached_record),
                        **(source_reply or {}).get("timings", {}),
                    }
                    self._event("published", version=version, epoch=self.state["epoch"], timings=timings)
                    return {"version": version, "nbytes": record["nbytes"], "timings": timings}
            except ValueError:
                raise  # Corrupt snapshots / bad identity are not network failures.
            except Exception as error:
                if isinstance(error, OSError) and not isinstance(error, (ConnectionError, TimeoutError)):
                    raise
                if retry >= self.recovery.get("max_retries", 3):
                    raise
                self._schedule_repair(repr(error))
                await self._await_ready()
        raise RuntimeError("Unreachable relay publication state")

    async def _watch_broadcast(self, version, core, epoch):
        try:
            await self._rpc(
                core.wait_distributed.remote(version, epoch=epoch), self.recovery.get("recovery_timeout_s", 900)
            )
        except Exception as error:
            if not self.closed:
                self._schedule_repair(repr(error))
        finally:
            self.distributing.discard(version)
            if not self.closed:
                await self._gc()

    async def acquire_for(self, key, version, lease):
        self._start()
        if key not in self.endpoints:
            raise KeyError(f"Unknown relay endpoint {key}")
        target = self.state["latest"] if version is None else version
        async with self.lock:
            if str(target) not in self.state["versions"]:
                raise KeyError(f"Exact weight version {target} is not retained")
            previous = self.state["leases"].get(lease)
            if previous is not None and previous != target:
                raise ValueError("A lease cannot change weight version")
            self.state["leases"][lease] = target
            await self._save()
        try:
            async with asyncio.timeout(self.recovery.get("recovery_timeout_s", 900)):
                while True:
                    self._check()
                    try:
                        result = await self._rpc(self.endpoints[key].local_acquire.remote(target))
                        return {**result, "lease": lease, "recovery_endpoint": self.endpoints[key]}
                    except Exception:
                        if self.ready.is_set() and target not in self.distributing:
                            self._schedule_repair("consumer cache unavailable")
                        if not self.ready.is_set():
                            await self._await_ready()
                        else:
                            await asyncio.sleep(0.05)
        except BaseException:
            # No lease ACK was delivered. Explicit releases from consumers are
            # idempotent, including after coordinator reconstruction.
            await self.release(lease)
            raise

    async def _restore_source(self, epoch):
        root = self.nodes[0]["key"]
        cached = await self._rpc(self.endpoints[root].manifests.remote())
        for version in sorted(map(int, self.state["versions"])):
            if version in cached:
                continue
            record = self.state["versions"][str(version)]
            arrays = await asyncio.to_thread(self.store.read_arrays, record)
            if not self.persist_to_disk:
                # Original refs remain owned by the live trainer publisher.
                # Pin them in the catalog and reattach on root endpoint loss:
                # ray.put here would duplicate every retained model in RAM.
                original = self.store.manifests[version]
                tensors = [
                    {**entry, "ref": source["ref"]}
                    for entry, source in zip(record["tensors"], original["tensors"], strict=True)
                ]
            else:
                tensors = [
                    {**entry, "ref": ray.put(array, _owner=self.endpoints[root])}
                    for entry, array in zip(record["tensors"], arrays, strict=True)
                ]
            await self._rpc(
                self.endpoints[root].install.remote(
                    {
                        "version": version,
                        "node_id": self.nodes[0]["node_id"],
                        "nbytes": record["nbytes"],
                        "tensors": tensors,
                    },
                    epoch,
                ),
                self.recovery.get("recovery_timeout_s", 900),
            )

    async def _recover(self, reason):
        started = time.monotonic()
        try:
            for attempt in range(self.recovery.get("max_retries", 3) + 1):
                try:
                    async with self.operations:
                        async with self.lock:
                            self.protected = set(map(int, self.state["versions"]))
                            self.state["epoch"] += 1
                            epoch = self.state["epoch"]
                            old_master = self.state["master"]
                            if self.active:
                                self.state["master"] = (old_master + 1) % len(self.pools[self.nodes[0]["key"]])
                            await self._save()
                        self._event("recovery_begin", reason=reason, epoch=epoch, attempt=attempt)
                        retained = list(self.protected)
                        # Fence snapshot installation before tearing down engines.
                        await asyncio.gather(
                            *[
                                self._rpc(e.sync.remote(epoch, retained, self.state["revision"]))
                                for e in self.endpoints.values()
                            ]
                        )
                        previous = {}
                        for key, pool in self.pools.items():
                            for index, core in enumerate(pool):
                                try:
                                    previous[(key, index)] = (await self._rpc(core.node_info.remote()))["incarnation"]
                                except Exception:
                                    previous[(key, index)] = self.infos.get((key, index), {}).get("incarnation")
                                ray.kill(core, no_restart=False)
                        # Every old writer must be dead before allocating any new
                        # destination buffers. Fresh sessions also fence old native QPs.
                        fresh = {}
                        async with asyncio.timeout(self.recovery.get("recovery_timeout_s", 900)):
                            while len(fresh) < sum(map(len, self.pools.values())):
                                for key, pool in self.pools.items():
                                    for index, core in enumerate(pool):
                                        pair = (key, index)
                                        if pair in fresh:
                                            continue
                                        try:
                                            info = await self._rpc(core.node_info.remote())
                                            if info["incarnation"] != previous[pair]:
                                                fresh[pair] = info
                                        except Exception:
                                            pass
                                await asyncio.sleep(0.05)
                        self.infos = fresh
                        await self._restore_source(epoch)
                        keys = [n["key"] for n in self.nodes]
                        self.active = {
                            key: self.pools[key][self.state["master"] if i == 0 else 0] for i, key in enumerate(keys)
                        }
                        for index, key in enumerate(keys):
                            manifests = await self._rpc(self.endpoints[key].manifests.remote())
                            successor = self.active[keys[index + 1]] if index + 1 < len(keys) else None
                            await self._rpc(
                                self.active[key].configure_recovery.remote(
                                    epoch, successor, self.endpoints[key], manifests, retained, self.state["revision"]
                                )
                            )
                        root = keys[0]
                        for version in sorted(self.protected):
                            await self._rpc(
                                self.active[root].replay.remote(version, epoch),
                                self.recovery.get("recovery_timeout_s", 900),
                            )
                        if old_master != self.state["master"]:
                            self.stats["master_switches"] += 1
                        self.stats["recoveries"] += 1
                        self.distributing.difference_update(self.protected)
                        self.protected.clear()
                        self.ready.set()
                        self._event(
                            "recovery_complete",
                            epoch=epoch,
                            master=self.state["master"],
                            versions=list(map(int, self.state["versions"])),
                            seconds=time.monotonic() - started,
                        )
                    await self._gc()
                    return
                except Exception as error:
                    self._event("recovery_retry", error=repr(error), attempt=attempt)
                    if attempt >= self.recovery.get("max_retries", 3):
                        raise
                    await asyncio.sleep(self.recovery.get("retry_interval_s", 0.2))
        except BaseException as error:
            self.failure = repr(error)
            self._event("recovery_failed", error=self.failure)
            self.ready.set()  # Wake waiters so they fail with the original reason.

    async def _monitor(self):
        try:
            while not self.closed:
                await asyncio.sleep(self.recovery.get("heartbeat_interval_s", 5))
                if self.closed or not self.ready.is_set() or self.failure:
                    continue
                try:
                    for key, core in self.active.items():
                        info = await self._rpc(core.node_info.remote())
                        if info["epoch"] != self.state["epoch"]:
                            raise RuntimeError("Native relay incarnation changed")
                        await self._rpc(core.health.remote())
                        endpoint = await self._rpc(self.endpoints[key].node_info.remote())
                        if endpoint["epoch"] != self.state["epoch"]:
                            raise RuntimeError("Relay endpoint was reconstructed")
                except Exception as error:
                    self._schedule_repair(repr(error))
        except asyncio.CancelledError:
            pass

    def health(self):
        self._start()
        return {"recovering": not self.ready.is_set(), "epoch": self.state["epoch"]}

    def topology(self):
        return {"active": self.active, "pools": self.pools, "endpoints": self.endpoints}

    async def status(self):
        self._start()
        return {
            "recovering": not self.ready.is_set(),
            "epoch": self.state["epoch"],
            "master": self.state["master"],
            "latest": self.state["latest"],
            "versions": sorted(map(int, self.state["versions"])),
            "leases": len(self.state["leases"]),
            "owners": len(self.state["owners"]),
            "disk_bytes": sum(r["nbytes"] for r in self.state["versions"].values()) if self.persist_to_disk else 0,
            "snapshot_bytes": sum(r["nbytes"] for r in self.state["versions"].values()),
            "storage": "disk" if self.persist_to_disk else "memory",
            "catalog_restart_supported": self.persist_to_disk,
            "capacity_waiters": self.waiters,
            "failure": self.failure,
            **self.stats,
        }

    async def close(self):
        if self.closed:
            return {"closed": True}
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.repair and not self.repair.done():
            # All native work lives in killable isolated processes. Do not free
            # their buffers in a different incarnation while writes are alive.
            self.repair.cancel()
            await asyncio.gather(self.repair, return_exceptions=True)
        for pool in self.pools.values():
            for core in pool:
                ray.kill(core, no_restart=True)
        for broadcast in list(self.broadcasts):
            broadcast.cancel()
        await asyncio.gather(*self.broadcasts, return_exceptions=True)
        async with self.operations:
            self.store.close()
        return {"closed": True}


class RecoverableRelayHandle:
    """Keep the independent actors referenced across any one process restart.

    A named non-detached Ray actor still needs a live handle. Keeping only its
    string name in endpoints would allow Ray to collect the coordinator before
    the first publish, or while the last endpoint is being reconstructed.
    """

    def __init__(self, endpoint, coordinator, resources, arguments):
        self.endpoint, self.coordinator, self.resources = endpoint, coordinator, resources
        self.arguments = arguments  # Pin Ray actor-creation arguments across restarts.

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "endpoint"), name)

    def __hash__(self):
        return hash(self.endpoint._actor_id)

    def __eq__(self, other):
        return isinstance(other, RecoverableRelayHandle) and self.endpoint._actor_id == other.endpoint._actor_id


def create_resilient_relay_hierarchy(actor_node, replica_nodes, options, recovery, directory, identity):
    """Return stable endpoint actors; every native worker remains replaceable."""
    nodes = {actor_node.get("key", actor_node["node_id"]): actor_node}
    for node in replica_nodes:
        nodes.setdefault(node.get("key", node["node_id"]), node)
    nodes = [{**node, "key": key} for key, node in nodes.items()]
    options = dict(options)
    environment = options.pop("runtime_env", {})
    namespace = ray.get_runtime_context().namespace
    name = f"laminar-relay-catalog-{uuid4().hex}"
    hash_workers = validate_hash_workers(recovery.get("hash_workers", 4))
    budget = {key: options[key] for key in ("max_versions", "max_bytes", "capacity_timeout_s")}
    pools, endpoints, owned = {}, {}, []
    coordinator = None
    try:
        for index, node in enumerate(nodes):
            affinity = NodeAffinitySchedulingStrategy(node["node_id"], soft=False)
            endpoint = (
                ray.remote(RelayEndpoint)
                .options(
                    num_cpus=hash_workers,
                    max_concurrency=128,
                    max_restarts=3,
                    max_task_retries=3,
                    scheduling_strategy=affinity,
                )
                .remote(node, name, namespace, budget, options["keep_last"], hash_workers)
            )
            endpoints[node["key"]] = endpoint
            owned.append(endpoint)
            pool = []
            for _ in range(2 if index == 0 else 1):
                core = (
                    ray.remote(MooncakeRelayNode)
                    .options(
                        num_cpus=1,
                        max_concurrency=128,
                        max_restarts=-1,
                        max_task_retries=0,
                        scheduling_strategy=affinity,
                        runtime_env=environment,
                    )
                    .remote(**options, node_id=node["node_id"], host=node["host"])
                )
                pool.append(core)
                owned.append(core)
            pools[node["key"]] = pool
        ray.get([core.node_info.remote() for pool in pools.values() for core in pool])
        arguments = [ray.put(value) for value in (nodes, endpoints, pools, options, recovery, str(directory), identity)]
        coordinator = (
            ray.remote(RelayRecoveryCoordinator)
            .options(
                name=name,
                namespace=namespace,
                num_cpus=hash_workers,
                max_concurrency=128,
                max_restarts=3 if recovery.get("persist_to_disk", False) else 0,
                max_task_retries=3 if recovery.get("persist_to_disk", False) else 0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(actor_node["node_id"], soft=False),
            )
            .remote(*arguments)
        )
        owned.append(coordinator)
        ray.get([endpoint.bind_coordinator.remote(coordinator) for endpoint in endpoints.values()])
        ray.get(coordinator.health.remote())
        wrappers = {
            key: RecoverableRelayHandle(endpoint, coordinator, list(owned), arguments)
            for key, endpoint in endpoints.items()
        }
        routes = [wrappers[node.get("key", node["node_id"])] for node in replica_nodes]
        return list(wrappers.values()), routes
    except Exception:
        for actor in reversed(owned):
            ray.kill(actor, no_restart=True)
        raise
