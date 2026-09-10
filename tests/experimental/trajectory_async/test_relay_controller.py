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

from verl.experimental.trajectory_async.relay_controller import RelayController


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
