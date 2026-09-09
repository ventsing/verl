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
"""Tests for the multi-version, pull-based weight store.

The orchestration (registry, retention, per-consumer state, GC) is
exercised end-to-end over :class:`FakeP2PBackend`, which reproduces the
owner-memory + remote-read lifecycle of the real P2P engines. The kimi /
mooncake adapters themselves need a cluster and are covered by design
review against ``verl/checkpoint_engine/*`` instead (see module docstring).
"""

import asyncio
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.versioned_weight_store import (  # noqa: E402
    FakeP2PBackend,
    KimiP2PBackend,
    MooncakeP2PBackend,
    P2P_BACKENDS,
    VersionedStoreRelayAdapter,
    VersionedWeightStore,
    make_p2p_backend,
)


def _weights(nbytes: int, shards: int = 2):
    per = nbytes // shards
    for i in range(shards):
        yield f"shard.{i}", bytes([i]) * per


class TestVersionedWeightStore(unittest.TestCase):
    def test_publish_manifest_latest(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0))
            m = await store.publish(3, _weights(1000))
            self.assertEqual(m.version, 3)
            self.assertEqual(m.checkpoint_name, "actor:v3")
            self.assertEqual(m.nbytes, 1000)
            self.assertEqual(m.tensor_count, 2)
            self.assertEqual(store.latest_version(), 3)
            self.assertEqual(store.manifest(3).nbytes, 1000)
            self.assertEqual(store.retained_versions, [3])

        asyncio.run(scenario())

    def test_versions_must_increase_and_be_unique(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0))
            await store.publish(1, _weights(10))
            with self.assertRaises(ValueError):
                await store.publish(1, _weights(10))  # duplicate
            with self.assertRaises(ValueError):
                await store.publish(0, _weights(10))  # not increasing

        asyncio.run(scenario())

    def test_retention_evicts_oldest_and_unstages(self):
        async def scenario():
            backend = FakeP2PBackend(stage_latency_s=0, read_latency_s=0)
            store = VersionedWeightStore(backend, keep_last=2)
            await store.publish(1, _weights(10))
            await store.publish(2, _weights(10))
            self.assertEqual(store.retained_versions, [1, 2])
            await store.publish(3, _weights(10))
            # v1 evicted, v2/v3 retained
            self.assertEqual(store.retained_versions, [2, 3])
            self.assertIsNone(store.manifest(1))
            self.assertEqual(store.stats.evictions, 1)
            self.assertEqual(backend.stats.unstages, 1)
            # evicted memory is gone: pinned pull must fail
            with self.assertRaises(LookupError):
                await store.pull("r0", version=1)

        asyncio.run(scenario())

    def test_release_explicit(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0), keep_last=5)
            for v in (1, 2, 3):
                await store.publish(v, _weights(10))
            evicted = await store.release(keep_last=1)
            self.assertEqual(evicted, [1, 2])
            self.assertEqual(store.retained_versions, [3])

        asyncio.run(scenario())

    def test_pull_latest_and_pinned(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0), keep_last=5)
            await store.publish(1, _weights(10))
            await store.publish(2, _weights(10))
            got = {}
            v = await store.pull("r0", sink=lambda n, t: got.setdefault(n, t))
            self.assertEqual(v, 2)
            self.assertEqual(got, {"shard.0": bytes([0]) * 5, "shard.1": bytes([1]) * 5})
            # pinned pull of a retained older version
            v = await store.pull("r0", version=1)
            self.assertEqual(v, 1)
            self.assertEqual(store.consumer_version("r0"), 1)

        asyncio.run(scenario())

    def test_consumers_pull_independently_no_lockstep(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0), keep_last=5)
            await store.publish(1, _weights(10))
            await store.publish(2, _weights(10))
            await store.publish(3, _weights(10))
            # r0 pulls late (v3), r1 stays on v1 — the Laminar no-lockstep property
            self.assertEqual(await store.pull("r0"), 3)
            self.assertEqual(await store.pull("r1", version=1), 1)
            self.assertEqual(store.consumer_version("r0"), 3)
            self.assertEqual(store.consumer_version("r1"), 1)
            self.assertEqual(store.consumer_lag("r1"), 2)
            self.assertEqual(store.consumer_lag("r0"), 0)
            # unknown consumer: lag = latest
            self.assertEqual(store.consumer_lag("rX"), 3)

        asyncio.run(scenario())

    def test_fake_backend_read_after_unstage_fails(self):
        async def scenario():
            backend = FakeP2PBackend(stage_latency_s=0, read_latency_s=0)
            store = VersionedWeightStore(backend, keep_last=1)
            await store.publish(1, _weights(10))
            await store.publish(2, _weights(10))  # v1 unstaged by retention
            with self.assertRaises(LookupError):
                await store.pull("r0", version=1)

        asyncio.run(scenario())

    def test_concurrent_pulls_and_publish(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0.01, read_latency_s=0.01), keep_last=4)
            await store.publish(1, _weights(100))
            results = await asyncio.gather(
                store.pull("r0"),
                store.pull("r1"),
                store.publish(2, _weights(100)),
                store.pull("r2"),
            )
            versions = sorted([results[0], results[1], results[3]])
            # pulls that started before/after the publish resolve to v1 or v2
            self.assertTrue(set(versions) <= {1, 2})
            self.assertEqual(results[2].version, 2)
            self.assertEqual(store.latest_version(), 2)
            self.assertEqual(store.stats.pulls, 3)
            # each consumer is consistent with exactly what it pulled
            for cid in ("r0", "r1", "r2"):
                self.assertIn(store.consumer_version(cid), (1, 2))

        asyncio.run(scenario())

    def test_bytes_and_snapshot_accounting(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0, read_latency_s=0), keep_last=3)
            await store.publish(1, _weights(400))
            await store.publish(2, _weights(600))
            await store.pull("r0")
            await store.pull("r0")
            snap = store.snapshot()
            self.assertEqual(snap["store/publishes"], 2)
            self.assertEqual(snap["store/pulls"], 2)
            self.assertEqual(snap["store/bytes_staged"], 1000)
            self.assertEqual(snap["store/bytes_read"], 1200)  # v2 twice
            self.assertEqual(snap["store/latest_version"], 2)
            self.assertEqual(snap["store/retained_versions"], [1, 2])
            self.assertEqual(snap["store/consumers"]["r0"]["pulls"], 2)
            self.assertEqual(snap["store/consumers"]["r0"]["version"], 2)
            self.assertEqual(snap["store/backend"], "fake")

        asyncio.run(scenario())

    def test_keep_last_validation(self):
        with self.assertRaises(ValueError):
            VersionedWeightStore(FakeP2PBackend(), keep_last=0)

    def test_pull_before_any_publish(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend())
            with self.assertRaises(LookupError):
                await store.pull("r0")

        asyncio.run(scenario())


class TestRelayAdapterWithEngine(unittest.TestCase):
    def test_adapter_contract_with_multi_replica_engine(self):
        """The store adapter satisfies the relay interface the engine uses,
        and a real engine run pulls versions through the store."""
        from verl.experimental.trajectory_async.multi_replica_engine import (
            MultiReplicaEngine,
            MultiReplicaEngineConfig,
        )

        async def scenario():
            backend = FakeP2PBackend(stage_latency_s=0.02, read_latency_s=0.02)
            store = VersionedWeightStore(backend, keep_last=2)
            relay = VersionedStoreRelayAdapter(store, weight_mb=1.0 / 1024)  # 1 KB payloads
            engine = MultiReplicaEngine(
                config=MultiReplicaEngineConfig(
                    seed=3,
                    length_mean_tokens=400.0,
                    length_sigma=0.3,
                    max_tokens=1500,
                    num_replicas=2,
                    batch_per_replica=4,
                    max_running_requests=4,
                    kv_capacity_tokens=8192,
                    decode_rate_tok_s=4000.0,
                    decode_tick_s=0.01,
                ),
                relay=relay,
            )
            tasks = [asyncio.create_task(engine.generate(900 + i)) for i in range(8)]
            await asyncio.sleep(0.03)
            # trainer publishes two versions while generation is in flight
            await relay.publish(1)
            await relay.publish(2)
            self.assertEqual(relay.latest_published_version(), 2)
            results = await asyncio.gather(*tasks)
            await engine.stop()

            self.assertEqual(len(results), 8)
            self.assertGreaterEqual(engine.stats.weight_pulls, 1)
            # requests routed before any publish ran under v0
            for r in results:
                self.assertIn(r.model_version, (0, 1, 2))
            # the store saw per-replica consumers
            snap = store.snapshot()
            self.assertEqual(snap["store/publishes"], 2)
            self.assertTrue(any(c.startswith("replica-") for c in snap["store/consumers"]))
            self.assertEqual(snap["store/backend"], "fake")

        asyncio.run(scenario())

    def test_adapter_stats_shape(self):
        async def scenario():
            store = VersionedWeightStore(FakeP2PBackend(stage_latency_s=0.01, read_latency_s=0.01))
            relay = VersionedStoreRelayAdapter(store, weight_mb=1.0 / 1024)
            await relay.publish(1)
            await relay.pull(0)
            snap = relay.stats.snapshot()
            self.assertEqual(snap["relay/publishes"], 1)
            self.assertEqual(snap["relay/pulls"], 1)
            self.assertAlmostEqual(snap["relay/actor_stall_total_s"], 0.01, places=3)
            self.assertAlmostEqual(snap["relay/pcie_total_s"], 0.01, places=3)
            self.assertEqual(snap["relay/pull_wait_total_s"], 0.0)

        asyncio.run(scenario())


class TestBackendSelection(unittest.TestCase):
    """The single switch point: make_p2p_backend / --p2p-backend."""

    def test_fake_backend_with_latencies(self):
        backend = make_p2p_backend("fake", stage_latency_s=0.5, read_latency_s=0.2)
        self.assertIsInstance(backend, FakeP2PBackend)
        self.assertEqual(backend.stage_latency_s, 0.5)
        self.assertEqual(backend.read_latency_s, 0.2)

    def test_kimi_requires_engine(self):
        with self.assertRaises(ValueError) as ctx:
            make_p2p_backend("kimi")
        self.assertIn("engine", str(ctx.exception))

    def test_mooncake_requires_engine(self):
        with self.assertRaises(ValueError) as ctx:
            make_p2p_backend("mooncake")
        self.assertIn("engine", str(ctx.exception))

    def test_unknown_backend_lists_options(self):
        with self.assertRaises(ValueError) as ctx:
            make_p2p_backend("nccl")
        self.assertIn("fake", str(ctx.exception))
        self.assertIn("kimi", str(ctx.exception))
        self.assertIn("mooncake", str(ctx.exception))

    def test_kimi_with_stub_engine(self):
        # engine objects are opaque to the factory — a bare stub is enough
        # to verify dispatch without importing the cluster-only package
        backend = make_p2p_backend("kimi", engine=object())
        self.assertIsInstance(backend, KimiP2PBackend)

    def test_mooncake_with_stub_engine(self):
        backend = make_p2p_backend("mooncake", engine=object(), staging_device="cpu")
        self.assertIsInstance(backend, MooncakeP2PBackend)

    def test_name_normalization(self):
        # case and surrounding whitespace are normalized
        self.assertIsInstance(make_p2p_backend("FAKE"), FakeP2PBackend)
        self.assertIsInstance(make_p2p_backend("  fake "), FakeP2PBackend)
        self.assertIsInstance(make_p2p_backend("KIMI", engine=object()), KimiP2PBackend)

    def test_registry_table(self):
        self.assertEqual(set(P2P_BACKENDS), {"fake", "kimi", "mooncake"})
        self.assertIs(P2P_BACKENDS["fake"], FakeP2PBackend)
        self.assertIs(P2P_BACKENDS["kimi"], KimiP2PBackend)
        self.assertIs(P2P_BACKENDS["mooncake"], MooncakeP2PBackend)


if __name__ == "__main__":
    unittest.main()
