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
"""CPU tests for the relay controller (the Ray-native control plane of the
versioned pull-based weight path). The controller takes injected async
callables, so versioning / retention / pull policy / metrics run without ray;
the wired dispatch (build_relay_controller) is cluster-gated."""

import asyncio
import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.relay_controller import RelayController, derive_replica_partition


class FakeOps:
    """Records publish/pull/unstage calls; optionally fails them."""

    def __init__(self):
        self.published = []
        self.pulled = []
        self.unstaged = []
        self.fail_publish = False

    async def publish_fn(self, version):
        if self.fail_publish:
            raise RuntimeError("publish boom")
        await asyncio.sleep(0)
        self.published.append(version)

    async def pull_fn(self, version):
        await asyncio.sleep(0)
        self.pulled.append(version)

    async def unstage_fn(self, version):
        self.unstaged.append(version)

    def controller(self, **kwargs):
        return RelayController(
            publish_fn=self.publish_fn, pull_fn=self.pull_fn, unstage_fn=self.unstage_fn, **kwargs
        )


class TestRelayController(unittest.TestCase):
    def test_publish_stages_and_tracks_latest(self):
        ops = FakeOps()
        c = ops.controller(keep_last=3)

        async def scenario():
            self.assertIsNone(c.latest_version)
            await c.publish(1)
            await c.publish(2)
            self.assertEqual(ops.published, [1, 2])
            self.assertEqual(c.latest_version, 2)
            snap = c.snapshot()
            self.assertEqual(snap["relay/latest_version"], 2)
            self.assertEqual(snap["relay/versions_live"], 2)
            self.assertEqual(snap["relay/publishes"], 2)

        asyncio.run(scenario())

    def test_publish_is_idempotent_per_version(self):
        ops = FakeOps()
        c = ops.controller()

        async def scenario():
            await c.publish(5)
            await c.publish(5)  # duplicate: no-op
            self.assertEqual(ops.published, [5])
            self.assertEqual(c.snapshot()["relay/versions_live"], 1)

        asyncio.run(scenario())

    def test_retention_retires_beyond_keep_last(self):
        ops = FakeOps()
        c = ops.controller(keep_last=2)

        async def scenario():
            await c.publish(1)
            await c.publish(2)
            self.assertEqual(ops.unstaged, [])  # latest two stay live
            await c.publish(3)
            self.assertEqual(ops.unstaged, [1])  # oldest retired
            self.assertEqual(c.latest_version, 3)
            # pulling a retired version raises
            with self.assertRaises(LookupError):
                await c.pull(version=1)

        asyncio.run(scenario())

    def test_retention_never_retires_the_latest(self):
        ops = FakeOps()
        c = ops.controller(keep_last=1)

        async def scenario():
            await c.publish(1)
            await c.publish(2)
            await c.publish(3)
            self.assertEqual(ops.unstaged, [1, 2])
            self.assertEqual(c.latest_version, 3)
            self.assertIsNotNone(await c.pull())  # latest still pullable

        asyncio.run(scenario())

    def test_pull_defaults_to_latest_and_tracks_lag(self):
        ops = FakeOps()
        c = ops.controller(keep_last=3)

        async def scenario():
            self.assertIsNone(await c.pull())  # nothing staged yet
            await c.publish(1)
            await c.publish(2)
            self.assertTrue(c.behind_latest(1))
            self.assertFalse(c.behind_latest(2))
            pulled = await c.pull()
            self.assertEqual(pulled, 2)
            self.assertEqual(ops.pulled, [2])
            self.assertEqual(c.snapshot()["relay/last_pulled_version"], 2)
            self.assertEqual(c.snapshot()["relay/fleet_lag_versions"], 0)
            # explicit older live version is allowed
            self.assertEqual(await c.pull(version=1), 1)
            self.assertEqual(c.snapshot()["relay/fleet_lag_versions"], 1)
            self.assertEqual(c.snapshot()["relay/pulls"], 2)

        asyncio.run(scenario())

    def test_publish_failure_leaves_no_version(self):
        ops = FakeOps()
        c = ops.controller()

        async def scenario():
            ops.fail_publish = True
            with self.assertRaises(RuntimeError):
                await c.publish(1)
            self.assertIsNone(c.latest_version)
            self.assertEqual(c.snapshot()["relay/versions_live"], 0)

        asyncio.run(scenario())

    def test_traffic_is_serialized(self):
        """One engine process group behind everything: concurrent publish/pull
        must not interleave (the lock enforces it)."""
        ops = FakeOps()
        c = ops.controller(keep_last=3)
        order = []

        async def slow_publish(version):
            order.append(f"pub-start:{version}")
            await asyncio.sleep(0.02)
            order.append(f"pub-end:{version}")
            ops.published.append(version)

        async def slow_pull(version):
            order.append(f"pull-start:{version}")
            await asyncio.sleep(0.02)
            order.append(f"pull-end:{version}")
            ops.pulled.append(version)

        c._publish_fn = slow_publish
        c._pull_fn = slow_pull

        async def scenario():
            await c.publish(1)
            # gather order: publish(2) reaches the lock first, pull waits
            await asyncio.gather(c.publish(2), c.pull(version=1))
            self.assertEqual(
                order,
                [
                    "pub-start:1",
                    "pub-end:1",
                    "pub-start:2",
                    "pub-end:2",
                    "pull-start:1",
                    "pull-end:1",
                ],
            )

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])


# ---------------------------------------------------------------- new: partition / per-replica / quota


class _FakeReplica:
    def __init__(self, n_workers):
        self.workers = [f"w{i}" for i in range(n_workers)]


class TestDeriveReplicaPartition(unittest.TestCase):
    def test_contiguous_blocks_in_flatten_order(self):
        parts = derive_replica_partition(4, [_FakeReplica(2), _FakeReplica(3), _FakeReplica(1)])
        self.assertEqual(parts, [[4, 5], [6, 7, 8], [9]])

    def test_skips_workerless_replicas(self):
        parts = derive_replica_partition(2, [_FakeReplica(0), _FakeReplica(2), _FakeReplica(0), _FakeReplica(2)])
        self.assertEqual(parts, [[2, 3], [4, 5]])

    def test_single_replica_is_none(self):
        self.assertIsNone(derive_replica_partition(4, [_FakeReplica(8)]))

    def test_no_replicas_is_none(self):
        self.assertIsNone(derive_replica_partition(4, []))


class TestPullCapabilityProbe(unittest.TestCase):
    """supports_pull_replica tells the bridge whether scoped pulls exist
    (single-replica topologies install no subgroups BY DESIGN)."""

    def test_probe_false_without_wiring(self):
        from verl.experimental.trajectory_async.relay_controller import RelayController

        async def publish(v):
            return {}

        async def pull(v):
            return None

        async def unstage(v):
            return None

        c = RelayController(publish_fn=publish, pull_fn=pull, unstage_fn=unstage)
        self.assertFalse(c.supports_pull_replica)

    def test_probe_true_with_wiring(self):
        from verl.experimental.trajectory_async.relay_controller import RelayController

        async def publish(v):
            return {}

        async def pull(v):
            return None

        async def unstage(v):
            return None

        async def pull_replica(rid, v):
            return None

        c = RelayController(
            publish_fn=publish,
            pull_fn=pull,
            unstage_fn=unstage,
            pull_replica_fn=pull_replica,
            num_replicas=2,
        )
        self.assertTrue(c.supports_pull_replica)
        # and pull_replica itself stays functional
        c._versions[3] = {"published_s": 0.0, "retired": False, "staged_bytes": 0}
        import asyncio

        self.assertEqual(asyncio.run(c.pull_replica(1)), 3)


class TestPerReplicaPull(unittest.TestCase):
    def _controller(self, num_replicas=3, **kwargs):
        async def publish_fn(version):
            return None

        async def pull_fn(version):
            return None

        async def unstage_fn(version):
            return None

        return RelayController(
            publish_fn=publish_fn,
            pull_fn=pull_fn,
            unstage_fn=unstage_fn,
            pull_replica_fn=kwargs.pop("pull_replica_fn", None),
            num_replicas=num_replicas,
            **kwargs,
        )

    def test_pull_replica_updates_one_version(self):
        pulled = []

        async def pull_replica_fn(replica_id, version):
            pulled.append((replica_id, version))

        ctrl = self._controller(pull_replica_fn=pull_replica_fn)

        async def run():
            await ctrl.publish(1)
            await ctrl.publish(2)
            self.assertEqual(await ctrl.pull_replica(1), 2)
            self.assertEqual(ctrl.replica_version(1), 2)
            self.assertIsNone(ctrl.replica_version(0))  # untouched
            self.assertEqual(pulled, [(1, 2)])

        asyncio.run(run())

    def test_pull_replica_idempotent_when_current(self):
        calls = []

        async def pull_replica_fn(replica_id, version):
            calls.append(replica_id)

        async def run():
            ctrl = self._controller(pull_replica_fn=pull_replica_fn)
            await ctrl.publish(3)
            self.assertEqual(await ctrl.pull_replica(2), 3)
            self.assertEqual(await ctrl.pull_replica(2), 3)  # no-op
            self.assertEqual(calls, [2])

        asyncio.run(run())

    def test_pull_replica_requires_wiring(self):
        async def run():
            ctrl = self._controller()
            with self.assertRaises(NotImplementedError):
                await ctrl.pull_replica(0)

        asyncio.run(run())

    def test_pull_replica_out_of_range(self):
        async def pull_replica_fn(replica_id, version):
            return None

        async def run():
            ctrl = self._controller(num_replicas=2, pull_replica_fn=pull_replica_fn)
            with self.assertRaises(ValueError):
                await ctrl.pull_replica(5)

        asyncio.run(run())

    def test_fleet_pull_sets_all_replica_versions(self):
        async def run():
            ctrl = self._controller(num_replicas=3)
            await ctrl.publish(7)
            await ctrl.pull()
            self.assertEqual([ctrl.replica_version(i) for i in range(3)], [7, 7, 7])
            snap = ctrl.snapshot()
            self.assertEqual(snap["relay/replica_lag_versions_max"], 0)

        asyncio.run(run())

    def test_distinct_replicas_pull_concurrently(self):
        """Per-replica pulls must overlap in time (each takes only ITS lock)."""
        entered = []
        overlap_seen = []

        async def pull_replica_fn(replica_id, version):
            entered.append(replica_id)
            await asyncio.sleep(0.02)
            if len(entered) > 1:
                overlap_seen.append(len(entered))
            entered.remove(replica_id)

        async def run():
            ctrl = self._controller(num_replicas=2, pull_replica_fn=pull_replica_fn)
            await ctrl.publish(1)
            await ctrl.publish(2)
            await asyncio.gather(ctrl.pull_replica(0), ctrl.pull_replica(1))
            self.assertTrue(overlap_seen, "per-replica pulls did not overlap")

        asyncio.run(run())

    def test_pull_replica_retired_version_raises(self):
        async def pull_replica_fn(replica_id, version):
            return None

        async def run():
            ctrl = self._controller(num_replicas=2, pull_replica_fn=pull_replica_fn, keep_last=1)
            await ctrl.publish(1)
            await ctrl.publish(2)  # retires v1 (keep_last=1)
            with self.assertRaises(LookupError):
                await ctrl.pull_replica(0, version=1)

        asyncio.run(run())


class TestStagedBytesQuota(unittest.TestCase):
    def _make(self, keep_last=2, max_staged_bytes=None):
        async def pull_fn(version):
            return None

        async def unstage_fn(version):
            return None

        ctrl = RelayController(
            publish_fn=self._publish_fn,
            pull_fn=pull_fn,
            unstage_fn=unstage_fn,
            keep_last=keep_last,
            max_staged_bytes=max_staged_bytes,
        )
        return ctrl

    async def _publish_fn(self, version):
        return {"staged_bytes": version * 100}

    def test_bytes_recorded_and_retired_by_keep_last(self):
        async def run():
            ctrl = self._make(keep_last=2)
            await ctrl.publish(1)
            await ctrl.publish(2)
            await ctrl.publish(3)  # retires v1
            snap = ctrl.snapshot()
            self.assertEqual(snap["relay/staged_bytes"], 500)  # v2 (200) + v3 (300)
            self.assertEqual(snap["relay/quota_retires"], 0)

        asyncio.run(run())

    def test_byte_quota_retires_oldest_beyond_keep_last(self):
        async def run():
            # keep_last=3 would keep v1..v3, but 150-byte quota only fits
            # the latest (100) + nothing else -> retire down to v3 only
            ctrl = self._make(keep_last=3, max_staged_bytes=150)
            await ctrl.publish(1)
            await ctrl.publish(2)
            await ctrl.publish(3)
            snap = ctrl.snapshot()
            self.assertEqual(snap["relay/versions_live"], 1)
            self.assertEqual(snap["relay/staged_bytes"], 300)
            self.assertEqual(snap["relay/staged_bytes_limit"], 150)
            self.assertEqual(snap["relay/quota_retires"], 2)

        asyncio.run(run())

    def test_quota_never_retires_latest(self):
        async def run():
            # even one version overflows the quota: keep it anyway
            ctrl = self._make(keep_last=2, max_staged_bytes=1)
            await ctrl.publish(1)
            snap = ctrl.snapshot()
            self.assertEqual(snap["relay/versions_live"], 1)
            self.assertEqual(ctrl.latest_version, 1)

        asyncio.run(run())

    def test_quota_rolling_retirement(self):
        async def run():
            # publish sizes grow with the version number (100, 200, ...);
            # the 500-byte quota keeps only the versions that fit, newest
            # first — a rolling window, never the latest
            ctrl = self._make(keep_last=5, max_staged_bytes=500)
            for v in range(1, 5):
                await ctrl.publish(v)
            snap = ctrl.snapshot()
            # after v4: [v2,v3,v4]=900 -> retire v2; [v3,v4]=700 -> retire
            # v3; [v4]=400 fits
            self.assertEqual(snap["relay/versions_live"], 1)
            self.assertEqual(snap["relay/staged_bytes"], 400)
            self.assertEqual(snap["relay/quota_retires"], 3)
            self.assertEqual(ctrl.latest_version, 4)

        asyncio.run(run())

    def test_quota_keeps_fitting_pair(self):
        async def run():
            # v1 (100) + v2 (200) fit in 350; v3 (300) would overflow
            ctrl = self._make(keep_last=5, max_staged_bytes=350)
            for v in range(1, 3):
                await ctrl.publish(v)
            snap = ctrl.snapshot()
            self.assertEqual(snap["relay/versions_live"], 2)
            self.assertEqual(snap["relay/staged_bytes"], 300)
            self.assertEqual(snap["relay/quota_retires"], 0)

        asyncio.run(run())
