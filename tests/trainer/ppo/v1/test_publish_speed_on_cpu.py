# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Checksum ordering, bounded parallelism and publication phase accounting."""

import asyncio
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.trajectory_async import weight_snapshot
from verl.experimental.trajectory_async.mooncake_relay import MooncakeRelayCheckpointEngine


def test_parallel_snapshot_hashes_match_serial_in_tensor_order(monkeypatch):
    monkeypatch.setattr(weight_snapshot, "PARALLEL_MIN_BYTES", 0)
    arrays = [np.arange(n, dtype=np.uint8) for n in (0, 15, 8192, 3, 100000, 2000)]
    expected = [hashlib.sha256(memoryview(a)).hexdigest() for a in arrays]
    assert weight_snapshot.checksum_arrays(arrays, 4) == expected
    assert weight_snapshot.checksum_arrays(arrays, 1) == expected


@pytest.mark.parametrize("workers", [0, 33, True, 1.5, None])
def test_invalid_digest_pool_sizes_fail_before_launch(workers):
    with pytest.raises(ValueError, match="hash_workers"):
        weight_snapshot.checksum_arrays([], workers)


def test_noncontiguous_or_nonbyte_payloads_cannot_be_certified():
    for arrays in ([np.arange(16, dtype=np.uint8)[::2]], [np.ones(3, dtype=np.float32)]):
        with pytest.raises(ValueError, match="contiguous uint8"):
            weight_snapshot.checksum_arrays(arrays)


def test_export_timings_cover_capacity_copy_put_and_relay_without_changing_snapshot(monkeypatch):
    from test_mooncake_relay_on_cpu import Remote

    from verl.experimental.trajectory_async import mooncake_relay

    class Relay:
        manifest = None

        def wait_for_capacity(self, version, nbytes=0):
            return max(64, nbytes)

        def commit(self, manifest):
            self.manifest = manifest
            return {"timings": {"catalog_hash_seconds": 0.001, "endpoint_checksums_reused": 1.0}}

    async def scenario():
        relay = Relay()
        engine = MooncakeRelayCheckpointEngine(bucket_size=64, is_master=True)
        engine.configure(Remote(relay), "node")
        floating = torch.tensor([1.0, 2.0], dtype=torch.float32)
        integer = torch.tensor(7, dtype=torch.int64)
        metrics = await engine.send_weights(iter([("w", floating), ("n", integer)]), 5)
        assert relay.manifest["version"] == 5 and relay.manifest["nbytes"] == 12
        floating.add_(10)
        first = relay.manifest["tensors"][0]
        assert first["dtype"] == "bfloat16"
        np.testing.assert_array_equal(
            first["ref"], torch.tensor([1.0, 2.0], dtype=torch.bfloat16).view(torch.uint8).numpy()
        )
        assert metrics["laminar/publish_tensors"] == 2
        assert metrics["laminar/publish_endpoint_checksums_reused"] == 1
        assert metrics["laminar/publish_catalog_hash_seconds"] == 0.001
        assert all(value >= 0 for key, value in metrics.items() if key.endswith("_seconds"))
        accounted = sum(metrics[f"laminar/publish_{name}_seconds"] for name in ("capacity", "export", "relay"))
        assert accounted <= metrics["laminar/publish_seconds"] + 0.01

    monkeypatch.setattr(MooncakeRelayCheckpointEngine, "node_info", lambda self: {"node_id": "node"})
    monkeypatch.setattr(mooncake_relay.ray, "put", lambda array: array.copy())
    asyncio.run(scenario())


def test_nonpublisher_still_consumes_all_collective_tensors_without_cpu_publication(monkeypatch):
    from verl.experimental.trajectory_async import mooncake_relay

    consumed = []

    def weights():
        for index in range(3):
            consumed.append(index)
            yield str(index), torch.tensor([index])

    async def scenario():
        engine = MooncakeRelayCheckpointEngine(bucket_size=64, is_master=False)
        engine.relay = SimpleNamespace()
        assert await engine.send_weights(weights(), 2) == {}

    monkeypatch.setattr(mooncake_relay.ray, "put", lambda _: pytest.fail("Nonpublisher cannot export CPU snapshots"))
    asyncio.run(scenario())
