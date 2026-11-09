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
"""Regression coverage for trajectory delivery and weight lifecycle races."""

import asyncio
import ctypes
import unittest
from types import SimpleNamespace

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.types import TrajectorySample, TrajectoryStatus
from verl.experimental.trajectory_async.versioned_weight_store import (
    FakeP2PBackend,
    MooncakeP2PBackend,
    VersionedWeightStore,
)


def sample(uid, version=0, index=0, n=1):
    return TrajectorySample(uid=uid, traj_index=index, group_size=n, model_version=version, reward=1.0)


class TestCompletedGroupReplay(unittest.TestCase):
    def test_completed_group_cannot_be_replayed_or_evicted(self):
        aggregator = GroupAggregator(max_dead_uids=1)
        self.assertIsNone(aggregator.add_trajectory(sample("done", index=0, n=2)))
        self.assertIsNotNone(aggregator.add_trajectory(sample("done", index=1, n=2)))
        # Evicted-id churn must not expire completed-group deduplication.
        for uid in ("failed-1", "failed-2"):
            failed = sample(uid)
            failed.status = TrajectoryStatus.FAILED
            aggregator.add_trajectory(failed)
        for index in (1, 0, 0, 1):
            self.assertIsNone(aggregator.add_trajectory(sample("done", index=index, n=2)))
        failed = sample("done", n=2)
        failed.status = TrajectoryStatus.FAILED
        self.assertIsNone(aggregator.add_trajectory(failed))
        self.assertEqual(aggregator.total_groups_completed, 1)
        self.assertEqual(aggregator.total_groups_evicted, 2)
        self.assertEqual(aggregator.num_partial_groups, 0)


class GatedReadBackend(FakeP2PBackend):
    def __init__(self):
        super().__init__(stage_latency_s=0, read_latency_s=0)
        self.started = asyncio.Queue()
        self.resume = asyncio.Event()

    async def read_into(self, consumer_id, *args):
        await self.started.put(consumer_id)
        await self.resume.wait()
        return await super().read_into(consumer_id, *args)


class TestWeightLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_eviction_waits_for_all_readers_without_blocking_publication(self):
        backend = GatedReadBackend()
        store = VersionedWeightStore(backend, keep_last=1)
        await store.publish(1, [("w", b"old")])
        got = {}
        first = asyncio.create_task(store.pull("r0", sink=lambda n, t: got.setdefault(n, t)))
        second = asyncio.create_task(store.pull("r1"))
        for _ in range(2):
            await asyncio.wait_for(backend.started.get(), timeout=1)
        await asyncio.wait_for(store.publish(2, [("w", b"new")]), timeout=1)
        self.assertIsNone(store.manifest(1))
        self.assertIn("actor:v1", backend._memory)
        with self.assertRaises(LookupError):
            await store.pull("late", version=1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertIn("actor:v1", backend._memory)
        backend.resume.set()
        self.assertEqual(await second, 1)
        self.assertNotIn("actor:v1", backend._memory)
        self.assertEqual(store.consumer_version("r1"), 1)
        self.assertEqual(store.stats.pulls, 1)
        self.assertEqual(store.stats.evictions, 1)

    async def test_failed_sink_releases_retired_memory(self):
        backend = GatedReadBackend()
        store = VersionedWeightStore(backend, keep_last=1)
        await store.publish(1, [("w", b"old")])

        def fail(name, tensor):
            raise RuntimeError("sink failed")

        pulling = asyncio.create_task(store.pull("r", sink=fail))
        await asyncio.wait_for(backend.started.get(), timeout=1)
        await store.release(keep_last=0)
        self.assertIn("actor:v1", backend._memory)
        backend.resume.set()
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            await pulling
        self.assertNotIn("actor:v1", backend._memory)
        self.assertEqual(store.stats.pulls, 0)

    async def test_concurrent_publications_and_duplicates_are_serialized(self):
        class GatedStageBackend(FakeP2PBackend):
            def __init__(self):
                super().__init__(0, 0)
                self.started, self.resume = asyncio.Event(), asyncio.Event()

            async def stage(self, manifest, weights):
                if manifest.version == 1:
                    self.started.set()
                    await self.resume.wait()
                await super().stage(manifest, weights)

        backend = GatedStageBackend()
        store = VersionedWeightStore(backend, keep_last=3)
        first = asyncio.create_task(store.publish(1, [("w", b"1")]))
        await asyncio.wait_for(backend.started.wait(), timeout=1)
        duplicate = asyncio.create_task(store.publish(1, [("w", b"wrong")]))
        second = asyncio.create_task(store.publish(2, [("w", b"2")]))
        backend.resume.set()
        results = await asyncio.gather(first, duplicate, second, return_exceptions=True)
        self.assertEqual(results[0].version, 1)
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual(results[2].version, 2)
        self.assertEqual(store.latest_version(), 2)
        self.assertEqual(backend.stats.stages, 2)
        got = {}
        await store.pull("r", version=1, sink=lambda n, t: got.setdefault(n, t))
        self.assertEqual(got, {"w": b"1"})

    async def test_slow_publication_cannot_overwrite_a_newer_version(self):
        class SlowFirstBackend(FakeP2PBackend):
            async def stage(self, manifest, weights):
                if manifest.version == 1:
                    await asyncio.sleep(0.01)
                await super().stage(manifest, weights)

        store = VersionedWeightStore(SlowFirstBackend(0, 0), keep_last=2)
        await asyncio.gather(
            store.publish(1, [("w", b"1")]),
            store.publish(2, [("w", b"2")]),
        )
        self.assertEqual(store.latest_version(), 2)
        self.assertEqual(store.retained_versions, [1, 2])


class TestMooncakeTensorBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_tensor_reassembly_with_small_receiver_and_unaligned_chunks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch required for adapter test")

        class CopyTransport:
            def batch_register_memory(self, ptrs, sizes):
                return 0

            def transfer_sync_read(self, session, dst, src, size):
                ctypes.memmove(dst, src, size)
                return 0

        transport = CopyTransport()
        actor = SimpleNamespace(engine=transport, session_id="actor", bucket_size=64)
        receiver = SimpleNamespace(
            engine=transport,
            bucket_size=7,
            buf=torch.zeros(14, dtype=torch.uint8),
        )
        backend = MooncakeP2PBackend(actor)
        store = VersionedWeightStore(backend)
        expected = {
            "matrix": torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
            "bf16": torch.arange(3, dtype=torch.bfloat16),
            "scalar": torch.tensor(42, dtype=torch.int64),
            "empty": torch.empty((0, 2), dtype=torch.float32),
        }
        await store.publish(1, expected.items())
        got = {}
        # Retain returned tensors directly: later reads must not alias them.
        await store.pull("r", consumer_ctx={"engine": receiver}, sink=lambda n, t: got.setdefault(n, t))
        self.assertEqual(set(got), set(expected))
        for name, tensor in expected.items():
            self.assertTrue(torch.equal(got[name], tensor), name)
            self.assertEqual(got[name].dtype, tensor.dtype)
            self.assertEqual(got[name].shape, tensor.shape)


if __name__ == "__main__":
    unittest.main()
