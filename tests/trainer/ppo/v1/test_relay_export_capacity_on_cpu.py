# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Capacity backpressure must keep every FSDP rank out of the next collective."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.trajectory_async.cpu_relay import CPUSharedMemoryCheckpointEngine
from verl.experimental.trajectory_async.mooncake_relay import MooncakeRelayCheckpointEngine


@pytest.mark.parametrize("engine_cls", [CPUSharedMemoryCheckpointEngine, MooncakeRelayCheckpointEngine])
@pytest.mark.parametrize("blocked_bytes", [0, 2, 4])
def test_all_ranks_wait_before_export_collective(engine_cls, blocked_bytes, monkeypatch):
    async def scenario():
        gate = asyncio.Event()
        waiting = asyncio.Event()
        entered = [[], []]
        calls = []
        commits = []

        async def capacity(version, nbytes=0):
            assert version == 74
            calls.append(nbytes)
            if nbytes == blocked_bytes:
                waiting.set()
                await gate.wait()
            return nbytes

        async def commit(manifest):
            commits.append(manifest)
            return {}

        relay = SimpleNamespace(
            wait_for_capacity=SimpleNamespace(remote=capacity), commit=SimpleNamespace(remote=commit)
        )
        engines = [engine_cls(bucket_size=1024, is_master=rank == 0) for rank in range(2)]
        for engine in engines:
            engine.relay = relay
            monkeypatch.setattr(engine, "node_info", lambda: {"node_id": "test"}, raising=False)

        async def weights(rank):
            for index in range(3):
                # Entering this iterator represents starting full_tensor()/ALLGATHER.
                entered[rank].append(index)
                yield str(index), torch.tensor([index], dtype=torch.float32)

        tasks = [asyncio.create_task(engine.send_weights(weights(rank), 74)) for rank, engine in enumerate(engines)]
        try:
            await asyncio.wait_for(waiting.wait(), 1)
            # Let all ranks reach either the CPU wait or the next collective.
            for _ in range(5):
                await asyncio.sleep(0)
            allowed = 0 if blocked_bytes == 0 else blocked_bytes // 2
            assert all(len(items) <= allowed for items in entered), (
                f"A peer entered an export collective while another rank waited for relay capacity: {entered}"
            )
            gate.set()
            result = await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert entered == [[0, 1, 2], [0, 1, 2]]
            assert len(commits) == 1 and commits[0]["nbytes"] == 6
            assert result[1] == {}
            assert [entry["name"] for entry in commits[0]["tensors"]] == ["0", "1", "2"]
            assert all(entry["dtype"] == "bfloat16" for entry in commits[0]["tensors"])
        finally:
            gate.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    monkeypatch.setattr("ray.put", lambda payload: payload.copy())
    asyncio.run(scenario())
