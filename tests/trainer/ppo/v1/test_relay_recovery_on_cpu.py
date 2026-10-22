# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Disk durability, exact-version chain reconstruction and stable leases."""

import asyncio

import numpy as np
import pytest
from test_mooncake_relay_on_cpu import NativeCopy, Remote, manifest

from verl.experimental.trajectory_async import relay_recovery as module
from verl.experimental.trajectory_async.mooncake_relay import MooncakeRelayNode


def budget():
    return {"max_versions": 4, "max_bytes": 1024, "capacity_timeout_s": 1}


def entries(m):
    return [{**e, "nbytes": e["ref"].nbytes} for e in m["tensors"]]


def test_snapshot_catalog_commits_all_bytes_and_detects_corruption(tmp_path):
    store = module.RelaySnapshotStore(tmp_path, {"model": "qwen"})
    m = manifest(0)
    record = store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    assert store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]]) == record
    for expected, actual in zip(m["tensors"], store.read_arrays(record), strict=True):
        np.testing.assert_array_equal(expected["ref"], actual)
    m["tensors"][0]["ref"][0] ^= 1
    with pytest.raises(ValueError, match="Conflicting"):
        store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    target = tmp_path / "version-0/tensor-000000.bin"
    data = bytearray(target.read_bytes())
    data[0] ^= 1
    target.write_bytes(data)
    with pytest.raises(ValueError, match="Corrupt"):
        store.read_arrays(record)
    store.close()


def test_catalog_rejects_second_writer_and_other_run(tmp_path):
    store = module.RelaySnapshotStore(tmp_path, "run-a")
    with pytest.raises(RuntimeError, match="live writer"):
        module.RelaySnapshotStore(tmp_path, "run-a")
    store.close()
    with pytest.raises(ValueError, match="another model/run"):
        module.RelaySnapshotStore(tmp_path, "run-b")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(module.ray, "put", lambda payload, **kwargs: payload.copy())
    NativeCopy.engines = {}
    NativeCopy.log = []
    nodes = [{"key": k, "node_id": k, "host": "127.0.0.1"} for k in "abc"]
    faces = {n["key"]: Remote(module.RelayEndpoint(n, "unused", "test", budget(), 1)) for n in nodes}
    options = {"keep_last": 1, "protocol": "tcp", "chunk_bytes": 8, **budget()}
    pools = {}
    for n in nodes:
        pools[n["key"]] = [
            Remote(MooncakeRelayNode(node_id=n["node_id"], host=n["host"], engine=NativeCopy(), **options))
            for _ in range(2 if n["key"] == "a" else 1)
        ]
    recovery = {
        "persist_to_disk": True,
        "rpc_timeout_s": 1,
        "recovery_timeout_s": 3,
        "heartbeat_interval_s": 0.01,
        "max_retries": 1,
        "retry_interval_s": 0.001,
    }
    coordinator = Remote(
        module.RelayRecoveryCoordinator(nodes, faces, pools, options, recovery, tmp_path / "catalog", "test-model")
    )
    for f in faces.values():
        f.obj.coordinator = coordinator
    killed = []

    def kill(actor, no_restart=True):
        killed.append(actor)
        if no_restart:
            return
        old = actor.obj
        actor.obj = MooncakeRelayNode(node_id=old.node_id, host=old.host, engine=NativeCopy(), **options)

    monkeypatch.setattr(module.ray, "kill", kill)
    return coordinator, faces, pools, killed


async def finish_broadcast(c):
    root = c.nodes[0]["key"]
    version = c.state["latest"]
    await c.active[root].wait_distributed.remote(version, epoch=c.state["epoch"])
    while version in c.distributing:
        await asyncio.sleep(0.001)


async def until(check):
    async with asyncio.timeout(5):
        while not await check():
            await asyncio.sleep(0.005)


def test_primary_death_switches_master_and_reuses_healthy_node_snapshots(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            await c.retain(0, "unfinished")
            lease = await c.acquire_for("c", 0, "consumer-lease")
            before = len(NativeCopy.log)
            epoch = c.state["epoch"]
            # A native engine is reconstructed, preserving its stable handle.
            module.ray.kill(c.active["a"], no_restart=False)

            async def recovered():
                return c.state["epoch"] > epoch and c.ready.is_set()

            await until(recovered)
            assert c.state["master"] == 1
            assert len(NativeCopy.log) == before, "complete local snapshots should not be retransmitted"
            renewed = await c.acquire_for("c", 0, "consumer-lease")
            assert renewed["version"] == lease["version"] == 0
            assert len(c.state["leases"]) == 1
            for a, b in zip(lease["tensors"], renewed["tensors"], strict=True):
                np.testing.assert_array_equal(a["ref"], b["ref"])
            await c.release("consumer-lease")
            await c.publish(manifest(1))
            await finish_broadcast(c)
            assert all(1 in f.obj.versions for f in faces.values())
        finally:
            await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("node", ["a", "b", "c"])
def test_cache_endpoint_restart_restores_exact_old_versions(rig, node):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            await c.retain(0, "unfinished")
            await c.publish(manifest(1))
            await finish_broadcast(c)
            await c.acquire_for(node, 0, "old-lease")
            endpoint = faces[node].obj
            faces[node].obj = module.RelayEndpoint(endpoint.node, "unused", "test", budget(), 1)
            faces[node].obj.coordinator = handle
            epoch = c.state["epoch"]

            async def recovered():
                return c.state["epoch"] > epoch and c.ready.is_set()

            await until(recovered)
            assert set(faces[node].obj.versions) == {0, 1}
            result = await c.acquire_for(node, 0, "old-lease")
            assert result["version"] == 0
            assert c.state["leases"] == {"old-lease": 0}
        finally:
            await c.close()

    asyncio.run(scenario())


def test_coordinator_reconstruction_keeps_committed_versions_and_leases(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        old = handle.obj
        c = old
        try:
            await old.publish(manifest(0))
            await finish_broadcast(old)
            await old.acquire_for("b", 0, "live-lease")
            epoch = old.state["epoch"]
            old.task.cancel()
            await asyncio.gather(old.task, return_exceptions=True)
            old.store.close()  # Simulate only the coordinator process exiting.
            c = module.RelayRecoveryCoordinator(
                old.nodes, faces, pools, old.options, old.recovery, old.store.path, "test-model"
            )
            handle.obj = c
            await c._await_ready()
            assert c.state["epoch"] > epoch
            assert c.state["leases"] == {"live-lease": 0}
            assert (await c.acquire_for("b", 0, "live-lease"))["version"] == 0
            await c.release("live-lease")
            assert c.state["leases"] == {}
        finally:
            await c.close()

    asyncio.run(scenario())


def test_publication_survives_master_failure_after_disk_commit_before_reply(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        await c._await_ready()
        root = c.active["a"].obj
        original = root.commit

        async def interrupted(manifest, epoch=None):
            raise ConnectionError("master died after durable catalog commit")

        root.commit = interrupted
        try:
            result = await c.publish(manifest(0))
            assert result["version"] == 0
            assert c.state["latest"] == 0
            assert all(0 in e.obj.versions for e in faces.values())
        finally:
            root.commit = original
            await c.close()

    asyncio.run(scenario())


def test_old_epochs_cannot_install_or_write_into_reconstructed_chain(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            epoch = c.state["epoch"]
            c._schedule_repair("test fence")
            await c._await_ready()
            with pytest.raises(ValueError, match="Stale"):
                await c.active["b"].chunk_ready.remote(0, 0, 0, 8, "invalid", epoch=epoch)
            with pytest.raises(ValueError, match="Stale"):
                await faces["b"].install.remote(faces["b"].obj.versions[0], epoch)
        finally:
            await c.close()

    asyncio.run(scenario())


def test_unused_version_disk_and_cache_are_reclaimed_but_lease_prevents_it(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            await c.acquire_for("b", 0, "held")
            await c.publish(manifest(1))
            await finish_broadcast(c)
            assert (c.store.path / "version-0").exists()
            await c.release("held")
            assert not (c.store.path / "version-0").exists()
            assert all(0 not in e.obj.versions for e in faces.values())
            with pytest.raises(KeyError):
                await c.acquire_for("b", 0, "late")
        finally:
            await c.close()

    asyncio.run(scenario())


def test_unacknowledged_snapshot_directory_is_collected_after_restart(tmp_path):
    store = module.RelaySnapshotStore(tmp_path, "same-run")
    m = manifest(0)
    store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    store.close()  # No catalog publication was committed.
    replacement = module.RelaySnapshotStore(tmp_path, "same-run")
    assert not (tmp_path / "version-0").exists()
    replacement.close()


def test_short_snapshot_write_never_publishes_a_version(tmp_path, monkeypatch):
    store = module.RelaySnapshotStore(tmp_path, "same-run")
    m = manifest(0)

    def no_space(fd):
        raise OSError("injected disk full")

    monkeypatch.setattr(module.os, "fsync", no_space)
    with pytest.raises(OSError, match="disk full"):
        store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    assert not (tmp_path / "version-0").exists()
    assert list(tmp_path.glob(".staging-*")) == []
    store.close()


@pytest.mark.parametrize("version", [-1, True, "0"])
def test_invalid_versions_are_rejected_before_disk_io(tmp_path, version):
    store = module.RelaySnapshotStore(tmp_path, "same-run")
    with pytest.raises(ValueError, match="nonnegative integer"):
        store.write_version(version, [], [])
    assert list(tmp_path.glob("version-*")) == []
    store.close()


def test_stale_retention_updates_cannot_undo_a_new_epoch_or_revision(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            epoch, revision = c.state["epoch"], c.state["revision"]
            assert faces["b"].obj.sync(epoch - 1, [], revision + 99) is False
            assert faces["b"].obj.sync(epoch, [], revision - 1) is False
            assert 0 in faces["b"].obj.versions
            with pytest.raises(ValueError, match="Stale"):
                await c.active["b"].set_retained.remote(epoch - 1, [])
            assert await c.active["b"].set_retained.remote(epoch, [], revision - 1) is False
        finally:
            await c.close()

    asyncio.run(scenario())


def test_consumer_waiting_for_healthy_inflight_broadcast_does_not_rebuild_chain(rig):
    import threading

    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        gate = threading.Event()
        try:
            await c._await_ready()
            c.active["a"].obj.transport.engine.gate = gate
            epoch = c.state["epoch"]
            await c.publish(manifest(0))
            pending = asyncio.create_task(c.acquire_for("c", 0, "waiting-consumer"))
            await asyncio.sleep(0.05)
            assert not pending.done()
            assert c.state["epoch"] == epoch and c.ready.is_set()
            gate.set()
            assert (await pending)["version"] == 0
            await c.release("waiting-consumer")
        finally:
            gate.set()
            await c.close()

    asyncio.run(scenario())


def test_corrupted_network_chunk_rebuilds_chain_from_clean_snapshot(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c._await_ready()
            epoch = c.state["epoch"]
            c.active["a"].obj.transport.engine.corrupt = True
            await c.publish(manifest(0))
            result = await c.acquire_for("c", 0, "after-corruption")
            assert result["version"] == 0
            assert c.state["epoch"] > epoch
            expected = manifest(0)
            for a, b in zip(result["tensors"], expected["tensors"], strict=True):
                np.testing.assert_array_equal(a["ref"], b["ref"])
            await c.release("after-corruption")
        finally:
            await c.close()

    asyncio.run(scenario())


def test_corrupt_disk_and_missing_source_cache_fail_without_new_policy_fallback(rig):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            path = c.store.path / "version-0/tensor-000000.bin"
            path.write_bytes(b"bad")
            face = faces["a"].obj
            faces["a"].obj = module.RelayEndpoint(face.node, "unused", "test", budget(), 1)
            faces["a"].obj.coordinator = handle

            async def failed():
                return c.failure is not None

            await until(failed)
            assert c.state["latest"] == 0
            assert "Truncated" in c.failure
            with pytest.raises(RuntimeError, match="Truncated"):
                await c.acquire_for("a", 0, "must-fail")
        finally:
            await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["versions", "bytes"])
def test_durable_publication_budget_waits_until_old_lease_releases(rig, limit):
    async def scenario():
        handle, faces, pools, killed = rig
        c = handle.obj
        c.options["max_versions"] = 2 if limit == "versions" else 4
        c.options["max_bytes"] = 216 if limit == "bytes" else 1024
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            await c.acquire_for("b", 0, "held-old-version")
            await c.publish(manifest(1))
            await finish_broadcast(c)
            waiting = asyncio.create_task(c.wait_for_capacity(2, 72))
            await asyncio.sleep(0.02)
            assert not waiting.done()
            assert (c.store.path / "version-0").exists()
            await c.release("held-old-version")
            assert await waiting == 72
            await c.publish(manifest(2))
            await finish_broadcast(c)
            assert sum(r["nbytes"] for r in c.state["versions"].values()) <= c.options["max_bytes"]
            assert not (c.store.path / "version-0").exists()
        finally:
            await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("close_fails", [False, True])
def test_trainer_kills_only_owned_relay_actors_even_if_close_raises(monkeypatch, close_fails):
    from types import SimpleNamespace

    from verl.trainer.ppo.v1 import trainer_laminar

    killed = []

    def close():
        if close_fails:
            raise RuntimeError("injected close failure")
        return None

    monkeypatch.setattr(trainer_laminar.ray, "get", lambda x: x)
    monkeypatch.setattr(trainer_laminar.ray, "kill", lambda actor, no_restart: killed.append((actor, no_restart)))
    trainer = object.__new__(trainer_laminar.PPOTrainerLaminar)
    trainer.relay_nodes = [
        SimpleNamespace(resources=["cache", "catalog", "master", "standby"], close=SimpleNamespace(remote=close))
    ]
    if close_fails:
        with pytest.raises(RuntimeError, match="close failure"):
            trainer._close_relay_hierarchy()
    else:
        trainer._close_relay_hierarchy()
    assert killed == [(name, True) for name in ("cache", "catalog", "master", "standby")]


def test_memory_catalog_never_writes_snapshots_and_restores_root_endpoint(rig, tmp_path, monkeypatch):
    async def scenario():
        handle, faces, pools, killed = rig
        old = handle.obj
        old.store.close()
        recovery = {**old.recovery, "persist_to_disk": False}
        path = tmp_path / "memory-only"

        def no_disk(*args, **kwargs):
            raise AssertionError("Recovery data must not be written in memory mode")

        monkeypatch.setattr(module, "atomic_json", no_disk)
        c = module.RelayRecoveryCoordinator(old.nodes, faces, pools, old.options, recovery, path, "test-model")
        handle.obj = c
        try:
            await c.publish(manifest(0))
            await finish_broadcast(c)
            await c.acquire_for("b", 0, "held")
            assert c.store.arrays[0]
            root = faces["a"].obj
            faces["a"].obj = module.RelayEndpoint(root.node, "unused", "test", budget(), 1)
            faces["a"].obj.coordinator = handle
            epoch = c.state["epoch"]
            original_refs = [e["ref"] for e in c.store.manifests[0]["tensors"]]

            async def recovered():
                return c.state["epoch"] > epoch and c.ready.is_set()

            await until(recovered)
            acquired = await c.acquire_for("a", 0, "root-consumer")
            assert all(a["ref"] is b for a, b in zip(acquired["tensors"], original_refs, strict=True))
            for expected, actual in zip(manifest(0)["tensors"], acquired["tensors"], strict=True):
                np.testing.assert_array_equal(expected["ref"], actual["ref"])
            await c.release("root-consumer")
            await c.publish(manifest(1))
            await finish_broadcast(c)
            assert set(c.store.arrays) == {0, 1}
            await c.release("held")
            assert set(c.store.arrays) == {1}
            assert set(c.store.manifests) == {1}
            status = await c.status()
            assert status["storage"] == "memory" and status["disk_bytes"] == 0
            assert not status["catalog_restart_supported"]
            assert set(p.name for p in path.iterdir()) == {"relay_events.jsonl"}
        finally:
            await c.close()

    asyncio.run(scenario())


def test_memory_snapshot_conflicts_and_missing_exact_versions_fail_closed(tmp_path):
    store = module.MemorySnapshotStore(tmp_path / "must-not-exist", "run")
    m = manifest(0)
    record = store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    assert store.arrays == {}, "Hashing alone must not pin uncommitted arrays"
    store.arrays[0] = [e["ref"] for e in m["tensors"]]
    store.state["versions"]["0"] = record
    assert store.read_arrays(record)
    m["tensors"][0]["ref"][0] ^= 1
    with pytest.raises(ValueError, match="Conflicting"):
        store.write_version(0, entries(m), [e["ref"] for e in m["tensors"]])
    with pytest.raises(ValueError, match="Corrupt"):
        store.read_arrays(record)
    store.remove_version(0)
    with pytest.raises(RuntimeError, match="Exact weight version"):
        store.read_arrays(record)
    assert not store.path.exists()


def test_partial_memory_store_keeps_fencing_and_token_logprob_identity_without_files(tmp_path, monkeypatch):
    from verl.experimental.trajectory_async.partial_store import PartialTrajectoryStore

    monkeypatch.chdir(tmp_path)
    store = PartialTrajectoryStore(":memory:", identity="memory-run")
    try:
        store.create("r", [1, 2], {}, 8, 5)
        attempt = store.begin_attempt("r", "a")
        store.append("r", attempt["epoch"], 5, 0, [3, 4], [-0.5, -0.25])
        saved = store.fence("r")
        assert saved["token_ids"] == [3, 4] and saved["log_probs"] == [-0.5, -0.25]
        assert saved["version"] == 5 and saved["epoch"] > attempt["epoch"]
        with pytest.raises(ValueError, match="Stale"):
            store.append("r", attempt["epoch"], 5, 2, [5], [-0.1])
        assert not store.persistent and store.writer_lock is None
        assert list(tmp_path.iterdir()) == []
    finally:
        store.close()
    replacement = PartialTrajectoryStore(":memory:", identity="memory-run")
    assert replacement.lookup("r") is None
    replacement.close()


class ImmutableReference:
    """No payload read is possible; only an already verified identity may pass."""

    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return isinstance(other, ImmutableReference) and self.key == other.key

    def __await__(self):
        raise AssertionError("Already verified immutable payload was read again")
        yield


def verified_source_fixture(rig, tmp_path, monkeypatch):
    handle, faces, pools, killed = rig
    old = handle.obj
    old.store.close()
    recovery = {**old.recovery, "persist_to_disk": False}
    c = module.RelayRecoveryCoordinator(old.nodes, faces, pools, old.options, recovery, tmp_path / "verified", "model")
    handle.obj = c
    original = manifest(0)
    arrays = [e["ref"] for e in original["tensors"]]
    record = c.store.write_version(0, entries(original), arrays)
    source = {**original, "tensors": [{**e, "ref": ImmutableReference(i)} for i, e in enumerate(original["tensors"])]}
    c.state["versions"]["0"] = record
    c.state["latest"] = 0
    c.state["epoch"] = 7
    c.store.arrays[0] = arrays
    c.store.manifests[0] = source
    normalized = {
        **source,
        "tensors": [
            {**schema, "ref": ref["ref"]} for schema, ref in zip(record["tensors"], source["tensors"], strict=True)
        ],
    }
    monkeypatch.setattr(module.ray, "ObjectRef", ImmutableReference)
    return c, faces["a"].obj, source, normalized


def test_verified_source_install_checks_identity_and_reuses_checksum_without_rereading(rig, tmp_path, monkeypatch):
    async def scenario():
        c, endpoint, source, normalized = verified_source_fixture(rig, tmp_path, monkeypatch)
        try:
            endpoint.sync(7, [0])
            assert c._verified_source(source, require_hash=False)
            result = await endpoint.install(normalized, 7)
            assert result["checksums_reused"] and result["hash_seconds"] == 0
            assert endpoint.versions[0]["tensors"][0]["ref"] == source["tensors"][0]["ref"]
        finally:
            await c.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "change", ["reference", "shape", "dtype", "name", "sha256", "nbytes", "version", "node", "epoch"]
)
def test_checksum_reuse_rejects_changed_reference_version_schema_or_digest(rig, tmp_path, monkeypatch, change):
    async def scenario():
        c, endpoint, source, normalized = verified_source_fixture(rig, tmp_path, monkeypatch)
        try:
            modified = {**normalized, "tensors": [dict(e) for e in normalized["tensors"]]}
            epoch = 7
            first = modified["tensors"][0]
            if change == "reference":
                first["ref"] = ImmutableReference("new-object")
            elif change == "shape":
                first["shape"] = [first["nbytes"]]
            elif change == "dtype":
                first["dtype"] = "uint8"
            elif change == "name":
                first["name"] += "changed"
            elif change == "sha256":
                first["sha256"] = "0" * 64
            elif change == "nbytes":
                first["nbytes"] += 1
            elif change == "version":
                modified["version"] = 1
            elif change == "node":
                modified["node_id"] = "other-node"
            else:
                epoch = 6
            assert not c.source_is_verified(modified, epoch)
        finally:
            await c.close()

    asyncio.run(scenario())


def test_mutable_arrays_never_get_reference_checksum_reuse(rig, tmp_path, monkeypatch):
    async def scenario():
        c, endpoint, source, normalized = verified_source_fixture(rig, tmp_path, monkeypatch)
        try:
            original = manifest(0)
            c.store.manifests[0] = original
            raw = {
                **original,
                "tensors": [
                    {**schema, "ref": ref["ref"]}
                    for schema, ref in zip(c.state["versions"]["0"]["tensors"], original["tensors"], strict=True)
                ],
            }
            assert not c.source_is_verified(raw, 7)
            c.reuse_verified_source = False
            c.store.manifests[0] = source
            assert not c.source_is_verified(normalized, 7)
        finally:
            await c.close()

    asyncio.run(scenario())
