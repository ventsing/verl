# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Exercise actual byte copying, chunk ordering, atomicity and native lifetimes."""

import asyncio
import ctypes
import hashlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.trajectory_async import mooncake_relay as module
from verl.experimental.trajectory_async.laminar_controller import LaminarController


class Remote:
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            result = getattr(self.obj, name)(*args, **kwargs)
            return await result if asyncio.iscoroutine(result) else result

        return SimpleNamespace(remote=call)


class NativeCopy:
    engines = {}
    counter = 0
    log = []

    def __init__(self):
        type(self).counter += 1
        self.port = type(self).counter
        self.memory = {}
        self.gate = None
        self.corrupt = False
        self.fail_register = False

    def initialize(self, host, metadata, protocol, device):
        self.engines[f"{host}:{self.port}"] = self
        return 0

    def get_rpc_port(self):
        return self.port

    def register_memory(self, ptr, size):
        if self.fail_register:
            return -1
        self.memory[ptr] = size
        return 0

    def unregister_memory(self, ptr):
        assert ptr in self.memory, "double or premature unregister"
        self.memory.pop(ptr)
        return 0

    def transfer_sync_write(self, session, source, destination, size):
        if self.gate is not None:
            assert self.gate.wait(5), "test transfer was never released"
        peer = self.engines[session]
        assert any(p <= source and source + size <= p + n for p, n in self.memory.items())
        assert any(p <= destination and destination + size <= p + n for p, n in peer.memory.items())
        time.sleep(0.003)
        ctypes.memmove(destination, source, size)
        if self.corrupt:
            ctypes.c_uint8.from_address(destination).value ^= 1
        self.log.append((self.port, peer.port, size))
        return 0


@pytest.fixture(autouse=True)
def local_object_store(monkeypatch):
    # Keep native buffers distinct from committed consumer snapshots.
    monkeypatch.setattr(module.ray, "put", lambda a: a.copy())
    NativeCopy.engines = {}
    NativeCopy.log = []


def manifest(version, node="a"):
    values = [
        ("x", torch.arange(32, dtype=torch.bfloat16)),
        ("n", torch.tensor(17, dtype=torch.int64)),
        ("empty", torch.empty(0, dtype=torch.float32)),
    ]
    tensors = []
    for name, t in values:
        raw = t.reshape(-1).view(torch.uint8).numpy().copy()
        tensors.append(
            {"name": name, "shape": tuple(t.shape), "dtype": str(t.dtype).removeprefix("torch."), "ref": raw}
        )
    return {"version": version, "node_id": node, "tensors": tensors, "nbytes": sum(e["ref"].nbytes for e in tensors)}


def chain(count=3, **kwargs):
    engines = [NativeCopy() for _ in range(count)]
    nodes = [
        module.MooncakeRelayNode(
            node_id=chr(97 + i), host="127.0.0.1", engine=e, protocol="tcp", chunk_bytes=8, **kwargs
        )
        for i, e in enumerate(engines)
    ]
    for a, b in zip(nodes, nodes[1:], strict=False):
        a.configure_successor(Remote(b))
    return nodes, engines


async def close(nodes, engines):
    for node in nodes:
        await node.close()
    assert all(not e.memory for e in engines)


def test_chain_is_pipelined_and_snapshots_match():
    async def scenario():
        nodes, engines = chain()
        m = manifest(0)
        try:
            await nodes[0].commit(m)
            await nodes[0].wait_distributed(0)
            for node in nodes:
                view = await node.acquire(0)
                assert view["node_id"] == node.node_id
                for expected, actual in zip(m["tensors"], view["tensors"], strict=True):
                    np.testing.assert_array_equal(expected["ref"], actual["ref"])
                node.release(view["lease"])
            assert engines[1].port in [a for a, _, _ in NativeCopy.log[:-1]]
            root_writes = [i for i, (a, _, _) in enumerate(NativeCopy.log) if a == engines[0].port]
            middle_writes = [i for i, (a, _, _) in enumerate(NativeCopy.log) if a == engines[1].port]
            assert min(middle_writes) < max(root_writes), "next hop must start before first hop finishes"
            assert nodes[0].stats["sent_bytes"] == nodes[1].stats["received_bytes"] == m["nbytes"]
            assert all(not e.memory for e in engines), "transfer registrations should be reclaimed"
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_publish_returns_before_broadcast_and_exact_version_waits():
    async def scenario():
        nodes, engines = chain(2)
        gate = threading.Event()
        try:
            await nodes[0].commit(manifest(0))
            await nodes[0].wait_distributed(0)
            engines[0].gate = gate
            await asyncio.wait_for(nodes[0].commit(manifest(1)), 0.5)
            waiter = asyncio.create_task(nodes[1].acquire(1))
            await asyncio.sleep(0.04)
            assert not waiter.done() and nodes[1].latest == 0
            old = await nodes[1].acquire(0)
            nodes[1].release(old["lease"])
            assert engines[0].memory
            gate.set()
            new = await waiter
            assert new["version"] == 1
            nodes[1].release(new["lease"])
            await nodes[0].wait_distributed(1)
        finally:
            gate.set()
            await close(nodes, engines)

    asyncio.run(scenario())


def test_retention_respects_local_read_lease():
    async def scenario():
        nodes, engines = chain(2, keep_last=1)
        try:
            await nodes[0].commit(manifest(0))
            await nodes[0].wait_distributed(0)
            held = await nodes[1].acquire(0)
            await nodes[0].commit(manifest(1))
            await nodes[0].wait_distributed(1)
            assert sorted(nodes[1].versions) == [0, 1]
            nodes[1].release(held["lease"])
            assert sorted(nodes[1].versions) == [1]
            with pytest.raises(KeyError):
                await nodes[1].acquire(0)
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_corruption_never_publishes_partial_version():
    async def scenario():
        nodes, engines = chain(3)
        engines[0].corrupt = True
        try:
            await nodes[0].commit(manifest(0))
            with pytest.raises(RuntimeError, match="checksum"):
                await nodes[0].wait_distributed(0)
            assert nodes[1].latest is None and not nodes[1].versions
            with pytest.raises(RuntimeError):
                nodes[0].health()
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_incomplete_duplicate_and_overlap_are_rejected():
    async def scenario():
        nodes, engines = chain(1)
        node = nodes[0]
        schema = [{"name": "x", "shape": (4,), "dtype": "int32", "nbytes": 16}]
        try:
            info = await node.begin(0, schema)
            raw = bytes(range(16))
            ctypes.memmove(info["addresses"][0], raw, 16)
            await node.chunk_ready(0, 0, 0, 8, hashlib.sha256(raw[:8]).hexdigest())
            with pytest.raises(ValueError, match="incomplete"):
                await node.seal(0)
            assert not node.versions
            with pytest.raises(ValueError, match="Duplicate"):
                await node.chunk_ready(0, 0, 0, 8, "x")
            await node.chunk_ready(0, 0, 8, 8, hashlib.sha256(raw[8:]).hexdigest())
            await node.seal(0)
            await node.wait_distributed(0)
            assert node.latest == 0
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_pending_versions_apply_backpressure():
    async def scenario():
        nodes, engines = chain(2, max_pending_versions=2)
        gate = threading.Event()
        engines[0].gate = gate
        try:
            await nodes[0].commit(manifest(0))
            await nodes[0].commit(manifest(1))
            third = asyncio.create_task(nodes[0].commit(manifest(2)))
            await asyncio.sleep(0.04)
            assert not third.done() and len(nodes[0].transfers) == 2
            gate.set()
            await third
            await nodes[0].wait_distributed(2)
            assert nodes[1].latest == 2
        finally:
            gate.set()
            await close(nodes, engines)

    asyncio.run(scenario())


def test_wait_timeout_does_not_cancel_native_transfer():
    async def scenario():
        nodes, engines = chain(2, acquire_timeout_s=0.03)
        gate = threading.Event()
        engines[0].gate = gate
        try:
            await nodes[0].commit(manifest(0))
            with pytest.raises(TimeoutError):
                await nodes[0].wait_distributed(0)
            assert nodes[0].transfers[0].task.cancelled() is False
            assert engines[0].memory
            gate.set()
            nodes[0].timeout = 3
            await nodes[0].wait_distributed(0)
        finally:
            gate.set()
            await close(nodes, engines)

    asyncio.run(scenario())


def test_registration_failure_is_fail_closed():
    async def scenario():
        nodes, engines = chain(1)
        engines[0].fail_register = True
        try:
            with pytest.raises(RuntimeError, match="register_memory"):
                await nodes[0].commit(manifest(0))
            assert not nodes[0].versions
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_wrong_master_and_consumer_nodes_are_rejected(monkeypatch):
    async def scenario():
        nodes, engines = chain(1)
        try:
            with pytest.raises(ValueError, match="publisher"):
                await nodes[0].commit(manifest(0, "wrong"))
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())
    engine = module.MooncakeRelayCheckpointEngine(bucket_size=1024, is_master=True)
    monkeypatch.setattr(engine, "node_info", lambda: {"node_id": "a"})
    with pytest.raises(ValueError, match="colocated"):
        engine.configure(None, "b")
    with pytest.raises(ValueError, match="own node"):
        engine.set_manifest({"node_id": "b", "version": 0})


def test_controller_pulls_and_releases_replica_local_relay(tmp_path):
    async def scenario():
        calls = []

        class Local:
            def acquire(self, version):
                calls.append(("acquire", version))
                return {"version": version, "lease": "local"}

            def release(self, lease):
                calls.append(("release", lease))

        class Root:
            def acquire(self):
                raise AssertionError("must not read remote master snapshot")

        class Worker:
            def execute_checkpoint_engine(self, method, *args):
                calls.append((method, args))

            def update_weights(self, global_steps):
                calls.append(("load", global_steps))

        class Server:
            async def wait_for_requests_to_drain(self):
                pass

        c = LaminarController(
            [{"server": Remote(Server()), "workers": [Remote(Worker())], "relay": Remote(Local())}],
            Remote(Root()),
            event_path=str(tmp_path / "scheduler.jsonl"),
        )
        c.latest = 7
        await c._pull(0)
        assert ("acquire", 7) in calls and ("release", "local") in calls and ("load", 7) in calls
        assert c.replicas[0].version == 7

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "schema",
    [
        [{"name": "x", "shape": (-1,), "dtype": "float32", "nbytes": 4}],
        [{"name": "x", "shape": (2,), "dtype": "float32", "nbytes": 4}],
        [{"name": "x", "shape": (1,), "dtype": "invalid", "nbytes": 4}],
    ],
)
def test_bad_tensor_metadata_is_rejected(schema):
    with pytest.raises(ValueError):
        module.validate_schema(schema)


@pytest.mark.parametrize("structured_config", [False, True])
def test_hierarchy_pins_each_node_and_routes_local_consumers(monkeypatch, structured_config):
    placed = []

    class ActorClass:
        def options(self, **kwargs):
            self.kwargs = kwargs
            return self

        def remote(self, **kwargs):
            actor = SimpleNamespace(node_id=kwargs["node_id"], successor=None)

            def successor(target):
                actor.successor = target

            actor.node_info = SimpleNamespace(remote=lambda: {"node_id": actor.node_id})
            actor.configure_successor = SimpleNamespace(remote=successor)
            placed.append((actor, self.kwargs, kwargs))
            return actor

    monkeypatch.setattr(module.ray, "remote", lambda cls: ActorClass())
    monkeypatch.setattr(module.ray, "get", lambda values: values)
    nodes = [{"node_id": n, "host": "host-" + n} for n in ["a", "b", "b", "c"]]
    env = {"env_vars": {"PYTHONPATH": "/isolated/mooncake"}}
    config_env = module.OmegaConf.create(env) if structured_config else env
    relays, routes = module.create_relay_hierarchy(nodes[0], nodes[1:], {"protocol": "tcp", "runtime_env": config_env})
    assert [r.node_id for r in relays] == ["a", "b", "c"]
    assert routes[0] is routes[1] is relays[1]
    assert routes[2] is relays[2]
    assert relays[0].successor is relays[1] and relays[1].successor is relays[2]
    for actor, options, kwargs in placed:
        assert options["scheduling_strategy"].node_id == actor.node_id
        assert options["scheduling_strategy"].soft is False
        assert options["runtime_env"] == env
        assert isinstance(options["runtime_env"]["env_vars"], dict)
        assert "runtime_env" not in kwargs


async def wait_for_thread(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


def test_cancelled_publication_joins_native_registration():
    async def scenario():
        entered, gate = threading.Event(), threading.Event()

        class Registration(NativeCopy):
            def register_memory(self, ptr, size):
                ret = super().register_memory(ptr, size)
                entered.set()
                assert gate.wait(5)
                return ret

        engine = Registration()
        node = module.MooncakeRelayNode(node_id="a", host="127.0.0.1", engine=engine, protocol="tcp", chunk_bytes=8)
        task = asyncio.create_task(node.commit(manifest(0)))
        try:
            await wait_for_thread(entered)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done() and engine.memory
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not engine.memory and not node.versions
        finally:
            gate.set()
            await node.close()

    asyncio.run(scenario())


def test_cancelled_broadcast_joins_native_write_before_unregister():
    async def scenario():
        nodes, engines = chain(2)
        entered, gate = threading.Event(), threading.Event()
        original = engines[0].transfer_sync_write

        def blocked(*args):
            entered.set()
            assert gate.wait(5)
            return original(*args)

        engines[0].transfer_sync_write = blocked
        try:
            await nodes[0].commit(manifest(0))
            await wait_for_thread(entered)
            task = nodes[0].transfers[0].task
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done() and engines[0].memory
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not engines[0].memory
        finally:
            gate.set()
            await close(nodes, engines)

    asyncio.run(scenario())


def test_receive_budget_reserves_native_buffer_and_imminent_ray_copy():
    async def scenario():
        nodes, engines = chain(1, keep_last=1, max_versions=3, max_bytes=216, capacity_timeout_s=1)
        node = nodes[0]
        m = manifest(0)
        size = m["nbytes"]
        assert size == 72
        try:
            await node.commit(m)
            await node.wait_distributed(0)
            node.retain(0, "unfinished-v0")
            await node.commit(manifest(1))
            await node.wait_distributed(1)
            assert node.resident_bytes() == 2 * size
            schema = [{"name": "x", "shape": (36,), "dtype": "bfloat16", "nbytes": size}]
            waiting = asyncio.create_task(node.begin(2, schema))
            await asyncio.sleep(0.03)
            assert not waiting.done()  # 144 cache + 144 receive/copy exceeds 216.
            assert 2 not in node.transfers, "do not allocate a new native buffer before capacity exists"
            node.release_owner("unfinished-v0")
            await waiting
            assert node.resident_bytes() == 3 * size
            assert node.resident_bytes() <= node.max_bytes
            state = node.transfers[2]
            await node._free_buffers(state)
            node.transfers.pop(2)
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_ascend_transport_pins_sources_and_receivers_and_binds_threads(monkeypatch):
    bindings, allocations = [], []
    real_empty = torch.empty

    def bind(self):
        if self.protocol == "ascend":
            bindings.append(threading.get_ident())

    def pinned_empty(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            allocations.append(args[0])
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(module.MooncakeTransport, "_bind_device", bind)
    monkeypatch.setattr(module.torch, "empty", pinned_empty)

    async def scenario():
        engines = [NativeCopy(), NativeCopy()]
        nodes = [
            module.MooncakeRelayNode(node_id=node, host="127.0.0.1", engine=engine, protocol="ascend", chunk_bytes=8)
            for node, engine in zip(["a", "b"], engines, strict=True)
        ]
        nodes[0].configure_successor(Remote(nodes[1]))
        source = manifest(0)
        try:
            await nodes[0].commit(source)
            await nodes[0].wait_distributed(0)
            for node in nodes:
                view = await node.acquire(0)
                for expected, actual in zip(source["tensors"], view["tensors"], strict=True):
                    np.testing.assert_array_equal(expected["ref"], actual["ref"])
                node.release(view["lease"])
            assert len(allocations) == 2 * len(source["tensors"])
            assert any(thread != threading.get_ident() for thread in bindings)
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_export_snapshots_survive_generator_storage_reuse_and_preserve_types():
    exporter = module._PinnedWeightExport(32)
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
    expected = source.to(torch.bfloat16).clone()
    exporter.append("strided", source)
    source.fill_(-99)  # A producer may reuse its storage after yielding.
    exporter.append("scalar", torch.tensor(17, dtype=torch.int64))
    exporter.append("empty", torch.empty(0))
    exporter.flush()
    exporter.close()
    restored = [
        torch.tensor(e["ref"]).view(getattr(torch, e["dtype"])).reshape(e["shape"]) for e in exporter.tensors[:2]
    ]
    torch.testing.assert_close(restored[0], expected, rtol=0, atol=0)
    assert restored[1].item() == 17 and restored[1].dtype == torch.int64
    assert exporter.tensors[2]["shape"] == (0,)
    assert exporter.tensors[2]["ref"].nbytes == 0
    assert exporter.tensors[2]["dtype"] == "bfloat16"
    assert exporter.peak_bytes <= 32


def test_failed_export_never_commits_partial_weight_version():
    class Relay:
        def __init__(self):
            self.commits = []

        def wait_for_capacity(self, *args):
            return 1024

        def commit(self, manifest):
            self.commits.append(manifest)

    def broken_iterator():
        yield "first", torch.arange(16, dtype=torch.float32)
        raise RuntimeError("producer failed")

    async def scenario():
        relay = Relay()
        engine = module.MooncakeRelayCheckpointEngine(bucket_size=32, is_master=True)
        engine.relay = Remote(relay)
        with pytest.raises(RuntimeError, match="producer failed"):
            await engine.send_weights(broken_iterator(), global_steps=4)
        assert not relay.commits

    asyncio.run(scenario())


def test_oversized_export_is_rejected_before_host_staging():
    exporter = module._PinnedWeightExport(8)
    with pytest.raises(ValueError, match="exceeds the configured"):
        exporter.append("too_large", torch.ones(16))
    assert not exporter.pending and exporter.peak_bytes == 0


def test_native_batches_combine_small_tensors_and_keep_exact_snapshots():
    async def scenario():
        nodes, engines = chain(3)
        for node in nodes:
            node.chunk_bytes = 128
        batch_calls = []
        for engine in engines:

            def batch_write(session, sources, destinations, sizes, engine=engine):
                batch_calls.append(len(sizes))
                for source, destination, size in zip(sources, destinations, sizes, strict=True):
                    engine.transfer_sync_write(session, source, destination, size)
                return 0

            engine.batch_transfer_sync_write = batch_write
        tensors = []
        for i in range(14):
            raw = np.array([i], dtype=np.int32).view(np.uint8)
            tensors.append({"name": f"n{i}", "shape": (1,), "dtype": "int32", "ref": raw})
        source = {"version": 0, "node_id": "a", "tensors": tensors, "nbytes": 56}
        try:
            await nodes[0].commit(source)
            await nodes[0].wait_distributed(0)
            assert batch_calls == [14, 14]
            for node in nodes:
                view = await node.acquire(0)
                for actual, expected in zip(view["tensors"], tensors, strict=True):
                    np.testing.assert_array_equal(actual["ref"], expected["ref"])
                node.release(view["lease"])
            assert nodes[0].stats["sent_batches"] == nodes[1].stats["sent_batches"] == 1
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_corrupt_batch_does_not_acknowledge_earlier_valid_chunks():
    async def scenario():
        nodes, engines = chain(1)
        node = nodes[0]
        schema = [{"name": name, "shape": (1,), "dtype": "int32", "nbytes": 4} for name in ["a", "b"]]
        try:
            await node.begin(0, schema)
            state = node.transfers[0]
            digests = [node._digest(buffer, 0, 4) for buffer in state.buffers]
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                await node.chunks_ready(0, [(0, 0, 4), (1, 0, 4)], [digests[0], "corrupt"])
            assert state.received == [0, 0]
            assert node.latest is None
        finally:
            await close(nodes, engines)

    asyncio.run(scenario())


def test_host_preparation_keeps_heartbeat_live_and_joins_before_cancel(monkeypatch):
    async def scenario():
        nodes, engines = chain(1, max_bytes=128)
        node = nodes[0]
        entered, gate = threading.Event(), threading.Event()
        original = node._allocate_buffers

        def blocked(*args):
            buffers = original(*args)
            entered.set()
            assert gate.wait(5)
            return buffers

        monkeypatch.setattr(node, "_allocate_buffers", blocked)
        schema = [{"name": "x", "shape": (16,), "dtype": "float32", "nbytes": 64}]
        task = asyncio.create_task(node.begin(0, schema))
        try:
            await wait_for_thread(entered)
            assert node.health()  # Event loop runs while host preparation is blocked.
            assert node.resident_bytes() == 128  # Includes the future Ray snapshot.
            assert not engines[0].memory
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done(), "cancellation must join the allocation thread"
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not node.transfers and not engines[0].memory
        finally:
            gate.set()
            await asyncio.gather(task, return_exceptions=True)
            await close(nodes, engines)

    asyncio.run(scenario())
