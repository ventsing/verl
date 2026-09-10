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
"""CPU tests for the repack bridge: idle-replica refresh (the real §5
payoff on this stack), migration declining, and the manager hook."""

import asyncio
import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.repack import RepackConfig, RepackManager
from verl.experimental.trajectory_async.repack_bridge import (
    FleetRepackExecutor,
    MigrationUnsupported,
    RolloutReplicaView,
    build_repack_controller,
)


def _view(replica_id, versions, inflight, pulls, server_id=None, fail_pull=False):
    """A view over scriptable fakes: per-replica current version, per-server
    inflight map, recorded pulls."""

    async def version_fn(rid):
        return versions.get(rid)

    async def pull_fn(rid, version):
        if fail_pull:
            raise RuntimeError("engine said no")
        pulled = version if version is not None else 999
        pulls.append((rid, pulled))

    async def inflight_fn():
        return inflight

    return RolloutReplicaView(
        replica_id=replica_id,
        server_id=server_id or f"srv-{replica_id}",
        version_fn=version_fn,
        pull_fn=pull_fn,
        inflight_fn=inflight_fn,
    )


def _executor(views, latest):
    async def latest_version_fn():
        return latest

    return FleetRepackExecutor(views, latest_version_fn=latest_version_fn)


class TestRefreshIdle(unittest.TestCase):
    def test_idle_lagging_replica_pulls_latest(self):
        pulls = []
        views = [
            _view(0, {0: 3}, {"srv-0": 0}, pulls),  # idle + lagging -> pull
            _view(1, {1: 5}, {"srv-1": 3}, pulls),  # busy -> skip
        ]
        ex = _executor(views, latest=5)
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(refreshed, 1)
        self.assertEqual(pulls, [(0, 5)])

    def test_fresh_replica_not_pulled(self):
        pulls = []
        views = [_view(0, {0: 5}, {"srv-0": 0}, pulls)]  # already at latest
        ex = _executor(views, latest=5)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_unknown_inflight_is_conservative(self):
        """Unknown load must NEVER trigger a pull (could abort live work)."""
        pulls = []
        views = [_view(0, {0: 1}, None, pulls)]  # inflight unknown
        ex = _executor(views, latest=5)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_unknown_version_skipped(self):
        pulls = []
        views = [_view(0, {}, {"srv-0": 0}, pulls)]  # version unknown
        ex = _executor(views, latest=5)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_no_latest_no_refresh(self):
        pulls = []
        views = [_view(0, {0: 1}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=None)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_one_failure_does_not_stop_others(self):
        pulls = []
        views = [
            _view(0, {0: 1}, {"srv-0": 0}, pulls, fail_pull=True),
            _view(1, {1: 1}, {"srv-1": 0}, pulls),
        ]
        ex = _executor(views, latest=4)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 1)
        self.assertEqual(pulls, [(1, 4)])
        self.assertEqual(ex.refresh_failures, 1)
        self.assertEqual(ex.bridge_metrics()["repack/idle_refreshes"], 1)
        self.assertEqual(ex.bridge_metrics()["repack/refresh_failures"], 1)


class TestMigrationDeclined(unittest.TestCase):
    def test_views_cannot_migrate(self):
        view = _view(0, {0: 1}, {}, [])
        with self.assertRaises(MigrationUnsupported):
            view.remove_request("req-1")

        async def admit(request, *, prefill="recompute"):
            return None

        with self.assertRaises(MigrationUnsupported):
            asyncio.run(view.admit_request(None))

    def test_executor_declines_plans_without_half_migrating(self):
        pulls = []
        views = [_view(0, {0: 1}, {}, pulls), _view(1, {1: 1}, {}, pulls)]
        ex = _executor(views, latest=1)
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [])  # declined — nothing moved
        self.assertEqual(result.requests_moved, 0)
        self.assertEqual(result.sources_emptied, 0)
        self.assertEqual(ex.bridge_metrics()["repack/migrations_declined"], 1)

    def test_empty_plan_is_a_noop(self):
        ex = _executor([], latest=1)
        result = asyncio.run(ex.migrate([]))
        self.assertEqual(result.plan, [])
        self.assertEqual(ex.bridge_metrics()["repack/migrations_declined"], 0)

    def test_degraded_sync_snapshot(self):
        pulls = []
        views = [_view(0, {0: 1}, {}, pulls)]
        ex = _executor(views, latest=1)
        self.assertEqual(ex.snapshot(), [])
        self.assertEqual(ex.fleet_kv_util(), 0.0)


class TestManagerHook(unittest.TestCase):
    def test_repack_once_runs_refresh_idle_and_survives(self):
        """The manager loop must call the executor's refresh hook and not
        crash on the degraded (empty) snapshot."""
        pulls = []
        views = [
            _view(0, {0: 1}, {"srv-0": 0}, pulls),
            _view(1, {1: 2}, {"srv-1": 0}, pulls),
        ]
        ex = _executor(views, latest=3)
        manager = RepackManager(engine=ex, config=RepackConfig(check_interval_s=0.01))

        async def run():
            plan = await manager.repack_once()
            self.assertEqual(plan, [])  # nothing to migrate (snapshot degraded)
            self.assertEqual(pulls, [(0, 3), (1, 3)])  # but refreshes ran
            self.assertEqual(manager.stats.checks, 1)

        asyncio.run(run())
        self.assertEqual(ex.bridge_metrics()["repack/idle_refreshes"], 2)

    def test_notify_update_wakes_the_loop(self):
        pulls = []
        views = [_view(0, {0: 1}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=2)
        manager = RepackManager(engine=ex, config=RepackConfig(check_interval_s=60.0))

        async def run():
            manager.start()
            await asyncio.sleep(0.01)
            self.assertEqual(pulls, [])  # interval is 60s — still waiting
            manager.notify_update()  # the publish trigger
            for _ in range(100):
                if pulls:
                    break
                await asyncio.sleep(0.01)
            await manager.stop()
            self.assertEqual(pulls, [(0, 2)])

        asyncio.run(run())


class TestBuildWiring(unittest.TestCase):
    """The ray-free parts of build_repack_controller (identity mapping)."""

    def test_no_server_ids_means_conservative(self):
        manager, executor = build_repack_controller(relay_controller=None, rollouter=None, server_ids=None)
        self.assertEqual(executor.handles, {})  # no views without a mapping
        # and refresh on no views is a safe no-op
        self.assertEqual(asyncio.run(executor.refresh_idle()), 0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
