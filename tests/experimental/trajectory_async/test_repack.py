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
"""Tests for the repack algorithm (Algorithm 1, pure function) and the
RepackManager loop driving a real :class:`RolloutRepackExecutor`."""

import asyncio
import unittest

from tests.experimental.trajectory_async import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.repack import (
    RepackConfig,
    RepackManager,
    ReplicaState,
    best_fit_consolidation,
)
from verl.experimental.trajectory_async.relay_tier import (
    RolloutRepackExecutor,
    RunningRequest,
)


def _state(replica_id, version=3, kv_used=1000, kv_capacity=4096, kv_prev=2000,
           running=2, waiting=0, **kw):
    return ReplicaState(
        replica_id=replica_id,
        version=version,
        kv_used=kv_used,
        kv_capacity=kv_capacity,
        kv_prev=kv_prev,
        num_running=running,
        num_waiting=waiting,
        batch_quota=16,
        max_running=kw.pop("max_running", 16),
        pulling=kw.pop("pulling", False),
        routable=True,
    )


class TestBestFitConsolidation(unittest.TestCase):
    def test_packs_smallest_onto_fullest_and_releases_them(self):
        # three idle stragglers with footprints 100, 200, 3000; cap 4096, B=16
        states = [_state(0, kv_used=3000, kv_prev=3500, running=3),
                  _state(1, kv_used=100, kv_prev=500, running=1),
                  _state(2, kv_used=200, kv_prev=900, running=2)]
        plan = best_fit_consolidation(states)
        # ascending order: r1(100) packs onto the fullest viable destination
        # (r0 with 3000); then r2(200) onto r0 too (3300 ≤ 4096; requests
        # 3+1+2 = 6 ≤ 16). r0 is a destination and is never released.
        self.assertIn((1, 0), plan)
        self.assertIn((2, 0), plan)
        self.assertNotIn((0, 1), plan)
        self.assertNotIn((0, 2), plan)
        # both small sources released
        sources = {s for s, _ in plan}
        self.assertEqual(sources, {1, 2})

    def test_respects_roofline_batch_bound(self):
        # B=5: dst has 4 reqs, src has 2 -> 6 > 5, no plan possible
        states = [_state(0, kv_used=1000, running=4),
                  _state(1, kv_used=100, running=2, max_running=5)]
        states[0].max_running = 5
        plan = best_fit_consolidation(states)
        self.assertEqual(plan, [])

    def test_respects_kv_capacity(self):
        states = [_state(0, kv_used=3500, kv_prev=3800, running=2),
                  _state(1, kv_used=1000, kv_prev=1200, running=2)]
        states[1].kv_capacity = 4096
        plan = best_fit_consolidation(states)
        # 1000 + 3500 > 4096 and 3500 + 1000 > 4096: neither direction fits
        self.assertEqual(plan, [])

    def test_skips_busy_replicas_and_empty_replicas(self):
        # replica 2 is saturated (kv at cap, rising) -> not even a candidate;
        # replica 3 is empty -> has_work False; r0 packs onto r1 (the only
        # viable destination), and the DESTINATION itself is never released
        states = [
            _state(0, kv_used=100, kv_prev=200, running=1),
            _state(1, kv_used=200, kv_prev=300, running=1),
            _state(2, kv_used=4000, kv_prev=3900, running=4),  # at cap, growing
            _state(3, kv_used=0, kv_prev=0, running=0),  # idle but empty
        ]
        states[2].kv_capacity = 4096
        plan = best_fit_consolidation(states)
        self.assertEqual(plan, [(0, 1)])
        sources = {s for s, _ in plan}
        self.assertEqual(sources, {0})
        self.assertNotIn(2, sources)
        self.assertNotIn(3, sources)

    def test_version_groups_are_caller_scoped(self):
        """Consolidation within one version group only — the manager groups
        by version; the algorithm itself is group-scoped."""
        states = [_state(0, version=3, kv_used=100, kv_prev=200),
                  _state(1, version=4, kv_used=200, kv_prev=300)]
        plan = best_fit_consolidation(states)  # called on the MIXED set
        for src, dst in plan:
            self.assertNotEqual(src, dst)


# --------------------------------------------------- manager over executor


class _FakeHandle:
    """Duck-typed rollout replica (see RolloutReplicaHandle protocol)."""

    def __init__(self, replica_id, version=3, kv_capacity=4096, requests=None,
                 fail_migrate=False):
        self.replica_id = replica_id
        self.weight_version = version
        self.kv_capacity = kv_capacity
        self._batch_limit = 16
        self.fail_migrate = fail_migrate
        self._requests = {r.request_id: r for r in (requests or [])}
        self.pull_count = 0

    def kv_used_tokens(self):
        return sum(r.kv_tokens for r in self._requests.values())

    def kv_capacity_tokens(self):
        return self.kv_capacity

    def batch_limit(self):
        return self._batch_limit

    def running_requests(self):
        return list(self._requests.values())

    def remove_request(self, request_id):
        if self.fail_migrate:
            raise RuntimeError("engine migration API failed")
        return self._requests.pop(request_id)

    async def admit_request(self, request, *, prefill="recompute"):
        self._requests[request.request_id] = request

    async def pull_weights(self, version=None):
        self.pull_count += 1
        self.weight_version = version if version is not None else self.weight_version + 1
        return self.weight_version


class TestRepackManager(unittest.TestCase):
    def test_periodic_and_update_triggers(self):
        """The manager wakes on the check interval AND on notify_update
        (the post-weight-publish trigger, §5.1)."""

        async def run():
            handles = [_FakeHandle(i) for i in range(3)]  # no work: no plans
            manager = RepackManager(
                engine=RolloutRepackExecutor(handles),
                config=RepackConfig(check_interval_s=0.01),
            )
            manager.start()
            await asyncio.sleep(0.05)
            manager.notify_update()
            await asyncio.sleep(0.05)
            await manager.stop()
            self.assertGreaterEqual(manager.stats.checks, 2)
            self.assertEqual(manager.stats.update_triggers, 1)
            self.assertEqual(manager.stats.plans, 0)  # nothing to consolidate

        asyncio.run(run())

    def test_manager_consolidates_stragglers(self):
        async def run():
            # three stragglers on v3: smallest two pack onto the fullest
            handles = [
                _FakeHandle(0, requests=[RunningRequest("a", 300)]),
                _FakeHandle(1, requests=[RunningRequest("b", 100)]),
                _FakeHandle(2, requests=[RunningRequest("c", 200)]),
            ]
            manager = RepackManager(
                engine=RolloutRepackExecutor(handles),
                config=RepackConfig(check_interval_s=0.01),
            )
            try:
                plan = await manager.repack_once()
                self.assertTrue(plan)
                # destination absorbed everything; sources freed + pulled
                busiest = max(handles, key=lambda h: len(h.running_requests()))
                self.assertEqual(len(busiest.running_requests()), 3)
                for h in handles:
                    if h is not busiest:
                        self.assertEqual(h.pull_count, 1)
                self.assertEqual(busiest.pull_count, 0)
                snap = manager.stats.snapshot()
                self.assertEqual(snap["repack/plans"], 1)
                self.assertGreater(snap["repack/kv_tokens_moved"], 0)
                self.assertIn("kv_util_before", snap["repack/rounds"][0])
            finally:
                await manager.stop()

        asyncio.run(run())

    def test_manager_survives_exceptions(self):
        async def run():
            handles = [
                _FakeHandle(0, requests=[RunningRequest("a", 300)]),
                _FakeHandle(1, requests=[RunningRequest("b", 100)], fail_migrate=True),
                _FakeHandle(2, requests=[RunningRequest("c", 200)]),
            ]
            manager = RepackManager(
                engine=RolloutRepackExecutor(handles),
                config=RepackConfig(check_interval_s=0.01),
            )
            manager.start()
            await asyncio.sleep(0.05)
            await manager.stop()  # must not raise despite executor failures
            self.assertGreaterEqual(manager.stats.checks, 1)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
