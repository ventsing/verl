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
"""CPU tests for the repack bridge: idle-replica refresh, the migration
DRAIN LIFECYCLE (soft + hard), the completion watcher, real snapshots,
and the manager hook."""

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
    RolloutReplicaView,
    build_repack_controller,
)


def _view(replica_id, versions, inflight, pulls, server_id=None, fail_pull=False):
    """A view over scriptable fakes: per-replica current version, per-server
    inflight map (mutable — the watcher sees live updates), recorded pulls."""

    async def version_fn(rid):
        return versions.get(rid)

    async def pull_fn(rid, version):
        if fail_pull:
            raise RuntimeError("engine said no")
        pulled = version if version is not None else 999
        versions[rid] = pulled  # the replica now runs this version (like the real engine)
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


def _executor(views, latest, drain=None, abort=None, resume=None, config=None):
    async def latest_version_fn():
        return latest

    return FleetRepackExecutor(
        views,
        latest_version_fn=latest_version_fn,
        drain_fn=drain,
        abort_fn=abort,
        resume_fn=resume,
        config=config,
    )


class _FakeDrain:
    """Records drain calls; can be scripted to report unsupported."""

    def __init__(self, supported=True):
        self.calls = []
        self.supported = supported

    async def __call__(self, server_ids, on):
        self.calls.append((tuple(sorted(server_ids)), on))
        return self.supported


class _FakeAbort:
    def __init__(self, counts):
        self.counts = dict(counts)
        self.calls = []

    async def __call__(self, server_id):
        self.calls.append(server_id)
        return self.counts.get(server_id, 0)


class _FakeResume:
    def __init__(self):
        self.calls = []

    async def __call__(self, server_id):
        self.calls.append(server_id)
        return True


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
        pulls = []
        views = [_view(0, {0: 3}, None, pulls)]  # unknown load -> busy
        ex = _executor(views, latest=5)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_unknown_version_skipped(self):
        pulls = []
        views = [_view(0, {}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=5)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_no_latest_no_refresh(self):
        pulls = []
        views = [_view(0, {0: 3}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=None)
        self.assertEqual(asyncio.run(ex.refresh_idle()), 0)
        self.assertEqual(pulls, [])

    def test_one_failure_does_not_stop_others(self):
        pulls = []
        views = [
            _view(0, {0: 3}, {"srv-0": 0}, pulls, fail_pull=True),
            _view(1, {1: 3}, {"srv-1": 0}, pulls),
        ]
        ex = _executor(views, latest=5)
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(refreshed, 1)
        self.assertEqual(pulls, [(1, 5)])
        self.assertEqual(ex.refresh_failures, 1)


class TestMigrationDrainLifecycle(unittest.TestCase):
    def _cfg(self, **kw):
        return RepackConfig(batch_bound=32, **kw)

    def test_soft_drain_executes_plan(self):
        """Accepted plan: sources drained, nothing aborted (soft default),
        emptiness deferred to the watcher."""
        pulls, inflight = [], {"srv-0": 3, "srv-1": 4}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [(0, 1)])
        self.assertEqual(result.requests_moved, 0)  # soft: in-flight finishes
        self.assertEqual(drain.calls, [(("srv-0",), True)])  # begin_drain only
        self.assertEqual(ex.migrations_started, 1)
        self.assertEqual(ex.migrations_completed, 0)  # deferred
        self.assertEqual(ex._draining, {0: "srv-0"})

    def test_no_drain_capability_declines(self):
        pulls = []
        inflight = {"srv-0": 3, "srv-1": 4}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        ex = _executor(views, latest=5, drain=None, config=self._cfg())
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [])
        self.assertEqual(ex.migrations_declined, 1)
        self.assertEqual(ex._draining, {})

    def test_lb_without_drain_socket_declines(self):
        """The LB probe returns False (stock balancer) — decline, never
        half-migrate."""
        pulls = []
        inflight = {"srv-0": 3, "srv-1": 4}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain(supported=False)
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [])
        self.assertEqual(ex.migrations_declined, 1)
        self.assertEqual(ex._draining, {})

    def test_hard_drain_aborts_and_counts(self):
        pulls = []
        inflight = {"srv-0": 3, "srv-1": 4}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        abort = _FakeAbort({"srv-0": 3})
        ex = _executor(
            views, latest=5, drain=drain, abort=abort, config=self._cfg(hard_drain=True)
        )
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.requests_moved, 3)
        self.assertEqual(abort.calls, ["srv-0"])
        self.assertEqual(ex.requests_aborted_redirected, 3)
        # drain comes BEFORE abort (steering must precede redirection)
        self.assertEqual(drain.calls, [(("srv-0",), True)])

    def test_execution_time_canfit_rejects_overflow_pair(self):
        """Live re-check: src+dst over the batch bound -> pair declined,
        the rest of the plan still executes."""
        pulls = []
        inflight = {"srv-0": 20, "srv-1": 20, "srv-2": 2}
        views = [
            _view(i, {i: 5}, inflight, pulls) for i in range(3)
        ]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=RepackConfig(batch_bound=32))
        result = asyncio.run(ex.migrate([(0, 1), (0, 2)]))
        # (0,1): 20+20 > 32 -> declined; (0,2): 20+2 <= 32 -> accepted
        self.assertEqual(result.plan, [(0, 2)])
        self.assertEqual(ex.migration_pairs_declined, 1)
        self.assertEqual(drain.calls, [(("srv-0",), True)])

    def test_unknown_inflight_pair_declined(self):
        pulls = []
        views = [_view(0, {0: 5}, None, pulls), _view(1, {1: 5}, {"srv-1": 1}, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [])
        self.assertEqual(ex.migration_pairs_declined, 1)

    def test_abort_failure_degrades_to_soft_not_stranded(self):
        """A crashing abort RPC must not kill the round or strand the
        source: the drain marking precedes aborts, the source keeps
        draining (soft), the watcher still owns it."""
        pulls = []
        inflight = {"srv-0": 3, "srv-1": 4}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()

        async def bad_abort(server_id):
            raise RuntimeError("RPC exploded")

        ex = _executor(
            views, latest=5, drain=drain, abort=bad_abort, config=self._cfg(hard_drain=True)
        )
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [(0, 1)])  # round survived
        self.assertEqual(result.requests_moved, 0)  # degraded to soft
        self.assertEqual(ex._draining, {0: "srv-0"})  # watcher owns the source
        self.assertEqual(ex.refresh_failures, 1)

    def test_empty_plan_is_a_noop(self):
        ex = _executor([], latest=5, drain=_FakeDrain(), config=self._cfg())
        result = asyncio.run(ex.migrate([]))
        self.assertEqual(result.plan, [])
        self.assertEqual(ex.migrations_declined, 0)


class TestDrainWatcher(unittest.TestCase):
    def _cfg(self, **kw):
        return RepackConfig(batch_bound=32, **kw)

    def test_emptied_source_pulls_and_undrains(self):
        pulls = []
        inflight = {"srv-0": 0, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex._draining[0] = "srv-0"  # mid-migration: source drained, tail gone
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [(0, 5)])  # lagging source pulled
        self.assertIn((("srv-0",), False), drain.calls)  # end_drain
        self.assertEqual(ex.migrations_completed, 1)
        self.assertNotIn(0, ex._draining)

    def test_busy_draining_source_waits(self):
        pulls = []
        inflight = {"srv-0": 3}
        views = [_view(0, {0: 3}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex._draining[0] = "srv-0"
        asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [])
        self.assertEqual(drain.calls, [])
        self.assertIn(0, ex._draining)

    def test_hard_mode_completion_resumes_before_undrain(self):
        pulls = []
        inflight = {"srv-0": 0}
        views = [_view(0, {0: 5}, inflight, pulls)]  # already fresh
        drain = _FakeDrain()
        resume = _FakeResume()
        ex = _executor(
            views, latest=5, drain=drain, resume=resume, config=self._cfg(hard_drain=True)
        )
        ex._draining[0] = "srv-0"
        asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [])  # fresh: no pull needed
        self.assertEqual(resume.calls, ["srv-0"])  # engine un-paused
        self.assertIn((("srv-0",), False), drain.calls)

    def test_draining_replica_not_double_refreshed(self):
        """Path (b) idle refresh never touches a draining replica — the
        watcher owns its lifecycle."""
        pulls = []
        inflight = {"srv-0": 2, "srv-1": 0}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 3}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex._draining[0] = "srv-0"
        refreshed = asyncio.run(ex.refresh_idle())
        # replica 1 (idle, lagging, NOT draining) refreshed; replica 0 untouched
        self.assertEqual(pulls, [(1, 5)])
        self.assertEqual(refreshed, 1)

    def test_completion_failure_is_retried(self):
        pulls = []
        inflight = {"srv-0": 0}
        views = [_view(0, {0: 3}, inflight, pulls, fail_pull=True)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex._draining[0] = "srv-0"
        asyncio.run(ex.refresh_idle())
        self.assertEqual(ex.refresh_failures, 1)
        self.assertIn(0, ex._draining)  # still draining -> retried next tick
        self.assertEqual(drain.calls, [])  # end_drain NOT called


class TestDrainEscalation(unittest.TestCase):
    """Soft-drain deadline escalation (v1-integration posture): drain
    first, abort as the bounded fallback — one long tail must not pin a
    migration."""

    def _cfg(self, deadline, **kw):
        return RepackConfig(batch_bound=32, drain_deadline_s=deadline, **kw)

    def test_soft_drain_escalates_past_deadline(self):
        """One long-tail generation on the source: soft first, abort
        past the deadline, natural completion takes over."""
        pulls, inflight = [], {"srv-0": 1, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        abort = _FakeAbort({"srv-0": 1})
        ex = _executor(views, latest=5, drain=drain, abort=abort, config=self._cfg(0.0))

        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [(0, 1)])  # migration accepted (soft)
        self.assertEqual(abort.calls, [])  # not yet: soft mode first

        # tick: busy past a 0s deadline -> escalate (abort once)
        asyncio.run(ex.refresh_idle())
        self.assertEqual(abort.calls, ["srv-0"])
        self.assertEqual(ex.drains_escalated, 1)

        # the aborted request resumes elsewhere -> source empties -> done
        inflight["srv-0"] = 0
        asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [(0, 5)])
        self.assertEqual(ex.migrations_completed, 1)
        self.assertIn((("srv-0",), False), drain.calls)  # undrained
        self.assertNotIn(0, ex._drain_escalated)  # bookkeeping cleaned

    def test_no_deadline_never_escalates(self):
        """Static modes keep today's semantics: soft waits, no abort."""
        pulls, inflight = [], {"srv-0": 1, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        abort = _FakeAbort({"srv-0": 1})
        ex = _executor(views, latest=5, drain=drain, abort=abort, config=self._cfg(None))

        asyncio.run(ex.migrate([(0, 1)]))
        asyncio.run(ex.refresh_idle())
        asyncio.run(ex.refresh_idle())
        self.assertEqual(abort.calls, [])  # soft drain waits indefinitely
        self.assertEqual(ex.drains_escalated, 0)

    def test_deadline_not_reached_stays_soft(self):
        """Inside the deadline window: no abort; natural completion wins."""
        pulls, inflight = [], {"srv-0": 1, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        abort = _FakeAbort({"srv-0": 1})
        ex = _executor(views, latest=5, drain=drain, abort=abort, config=self._cfg(3600.0))

        asyncio.run(ex.migrate([(0, 1)]))
        asyncio.run(ex.refresh_idle())
        self.assertEqual(abort.calls, [])  # deadline far away
        self.assertEqual(ex.drains_escalated, 0)
        # the tail finishes naturally: migration completes without abort
        inflight["srv-0"] = 0
        asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [(0, 5)])
        self.assertEqual(abort.calls, [])
        self.assertEqual(ex.drains_escalated, 0)

    def test_escalation_fires_once_per_source(self):
        """A still-busy source is aborted once, not once per tick."""
        pulls, inflight = [], {"srv-0": 2, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        abort = _FakeAbort({"srv-0": 1})  # aborts 1; one request remains
        ex = _executor(views, latest=5, drain=drain, abort=abort, config=self._cfg(0.0))

        asyncio.run(ex.migrate([(0, 1)]))
        asyncio.run(ex.refresh_idle())
        asyncio.run(ex.refresh_idle())
        asyncio.run(ex.refresh_idle())
        self.assertEqual(abort.calls, ["srv-0"])  # once, despite 3 ticks
        self.assertEqual(ex.requests_aborted_redirected, 1)

    def test_abort_failure_stays_soft(self):
        """An escalation abort that fails degrades to soft (retry next
        tick) instead of stranding the drain."""
        pulls, inflight = [], {"srv-0": 1, "srv-1": 2}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()

        class _ExplodingAbort:
            async def __call__(self, server_id):
                raise RuntimeError("engine unreachable")

        ex = _executor(views, latest=5, drain=drain, abort=_ExplodingAbort(), config=self._cfg(0.0))
        asyncio.run(ex.migrate([(0, 1)]))
        asyncio.run(ex.refresh_idle())
        self.assertEqual(ex.drains_escalated, 0)  # not marked
        self.assertIn(0, ex._draining)  # drain lifecycle intact
        self.assertEqual(ex.refresh_failures, 1)


class TestPullCapabilityDegradation(unittest.TestCase):
    """Single-replica topologies: per-replica pulls are unavailable BY
    DESIGN (one block == the fleet); refresh must skip such views
    cleanly instead of raising NotImplementedError every tick."""

    def test_unrefreshable_view_is_skipped_not_failed(self):
        pulls, inflight = [], {"srv-0": 0}
        views = [_view(0, {0: 3}, inflight, pulls)]
        views[0].pull_fn = None  # wiring disabled the capability
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=32))
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [])  # no pull attempted
        self.assertEqual(refreshed, 0)
        self.assertEqual(ex.refresh_failures, 0)  # NOT counted as failures

    def test_capability_metric_reflects_wiring(self):
        pulls, inflight = [], {"srv-0": 0, "srv-1": 0}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 3}, inflight, pulls)]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=32))
        snap = ex.bridge_metrics()
        self.assertEqual(snap["repack/per_replica_pulls"], 1)
        views[1].pull_fn = None  # still >=1 capable
        self.assertEqual(ex.bridge_metrics()["repack/per_replica_pulls"], 1)
        views[0].pull_fn = None  # none capable
        self.assertEqual(ex.bridge_metrics()["repack/per_replica_pulls"], 0)

    def test_first_notimplementederror_latches_off(self):
        """The real single-replica deployment: pull_fn is wired but the
        controller raises NotImplementedError (no subgroups BY DESIGN).
        First refresh latches the view off and counts NOTHING; the second
        does not even try."""
        versions = {0: 3}
        inflight = {"srv-0": 0}
        attempts = []

        async def version_fn(rid):
            return versions.get(rid)

        async def unavailable_pull(rid, version):
            attempts.append((rid, version))
            raise NotImplementedError("no per-replica subgroups")

        async def inflight_fn():
            return inflight

        view = RolloutReplicaView(
            replica_id=0, server_id="srv-0", version_fn=version_fn,
            pull_fn=unavailable_pull, inflight_fn=inflight_fn,
        )
        ex = _executor([view], latest=5, config=RepackConfig(batch_bound=32))
        asyncio.run(ex.refresh_idle())
        self.assertEqual(attempts, [(0, 5)])  # tried once
        self.assertIsNone(view.pull_fn)  # latched off
        self.assertEqual(ex.refresh_failures, 0)  # unavailability, not failure
        asyncio.run(ex.refresh_idle())
        self.assertEqual(attempts, [(0, 5)])  # never retried
        self.assertEqual(ex.refresh_failures, 0)

    def test_watcher_unavailable_pull_completes_drain(self):
        """A drained source whose pull turns out unavailable: the drain
        COMPLETES (end_drain, back to routing) instead of stranding, and
        the source keeps its old version (the producer's fleet pull
        brings it current)."""
        versions = {0: 3}
        inflight = {"srv-0": 0}

        async def version_fn(rid):
            return versions.get(rid)

        async def unavailable_pull(rid, version):
            raise NotImplementedError("no per-replica subgroups")

        async def inflight_fn():
            return inflight

        view = RolloutReplicaView(
            replica_id=0, server_id="srv-0", version_fn=version_fn,
            pull_fn=unavailable_pull, inflight_fn=inflight_fn,
        )
        drain = _FakeDrain()
        ex = _executor([view], latest=5, drain=drain, config=RepackConfig(batch_bound=32))
        ex._draining[0] = "srv-0"
        asyncio.run(ex.refresh_idle())
        self.assertNotIn(0, ex._draining)  # drain completed, not stranded
        self.assertEqual(ex.migrations_completed, 1)
        self.assertEqual(ex.refresh_failures, 0)
        self.assertEqual(versions[0], 3)  # kept its old version
        self.assertIn((("srv-0",), False), drain.calls)  # back to routing

    def test_drained_unrefreshable_source_completes_without_pull(self):
        """Defensive watcher path: a drained source with no pull wiring
        returns to routing at its current version (the producer's fleet
        pull brings it current) instead of stranding the drain."""
        pulls, inflight = [], {"srv-0": 0}
        views = [_view(0, {0: 3}, inflight, pulls)]
        views[0].pull_fn = None
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=RepackConfig(batch_bound=32))
        ex._draining[0] = "srv-0"
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [])  # no pull possible
        self.assertNotIn(0, ex._draining)  # drain released
        self.assertIn((("srv-0",), False), drain.calls)  # back to routing
        self.assertEqual(ex.migrations_completed, 1)
        self.assertEqual(ex.refresh_failures, 0)


class TestRetireRevive(unittest.TestCase):
    """Fault tolerance (§3.3): dead replicas are excluded from every
    lifecycle path; revived ones return."""

    def _cfg(self, **kw):
        return RepackConfig(batch_bound=32, **kw)

    def test_retired_replica_never_refreshed_or_routable(self):
        pulls = []
        inflight = {"srv-0": 0, "srv-1": 0}
        views = [_view(0, {0: 3}, inflight, pulls), _view(1, {1: 3}, inflight, pulls)]
        ex = _executor(views, latest=5, config=self._cfg())
        self.assertEqual(ex.retire(["srv-0"]), 1)
        refreshed = asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [(1, 5)])  # only the live replica refreshes
        self.assertEqual(refreshed, 1)
        states = {s.replica_id: s for s in asyncio.run(ex.snapshot())}
        self.assertFalse(states[0].routable)  # dead: not a plan party
        self.assertTrue(states[1].routable)

    def test_retired_released_from_drain(self):
        """A replica that dies mid-migration cannot complete its drain —
        release it so the counters stay honest."""
        pulls = []
        inflight = {"srv-0": 2}
        views = [_view(0, {0: 3}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex._draining[0] = "srv-0"
        ex.retire(["srv-0"])
        self.assertNotIn(0, ex._draining)  # released

    def test_migrate_pairs_with_retired_party_declined(self):
        pulls = []
        inflight = {"srv-0": 1, "srv-1": 1}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        drain = _FakeDrain()
        ex = _executor(views, latest=5, drain=drain, config=self._cfg())
        ex.retire(["srv-1"])
        result = asyncio.run(ex.migrate([(0, 1)]))
        self.assertEqual(result.plan, [])  # destination dead -> declined
        self.assertEqual(ex.migration_pairs_declined, 1)

    def test_revive_returns_replica_to_lifecycle(self):
        pulls = []
        inflight = {"srv-0": 0}
        views = [_view(0, {0: 3}, inflight, pulls)]
        ex = _executor(views, latest=5, config=self._cfg())
        ex.retire(["srv-0"])
        self.assertEqual(ex.revive(["srv-0"]), 1)
        asyncio.run(ex.refresh_idle())
        self.assertEqual(pulls, [(0, 5)])  # refreshable again

    def test_retire_is_idempotent(self):
        pulls = []
        views = [_view(0, {0: 3}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=5, config=self._cfg())
        self.assertEqual(ex.retire(["srv-0"]), 1)
        self.assertEqual(ex.retire(["srv-0"]), 0)  # already retired
        self.assertEqual(ex.retired_replicas, 1)


class TestSnapshot(unittest.TestCase):
    def test_real_states_from_probes(self):
        pulls = []
        inflight = {"srv-0": 3, "srv-1": 10}
        views = [
            _view(0, {0: 5}, inflight, pulls),
            _view(1, {1: 5}, inflight, pulls),
            _view(2, {}, None, pulls),  # unknown everything
        ]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=32, kv_per_request=100))
        states = asyncio.run(ex.snapshot())
        by_id = {s.replica_id: s for s in states}
        self.assertEqual(by_id[0].num_running, 3)
        self.assertEqual(by_id[0].kv_used, 300)
        self.assertEqual(by_id[0].kv_capacity, 3200)
        self.assertEqual(by_id[0].max_running, 32)
        self.assertTrue(by_id[0].routable)
        self.assertFalse(by_id[0].pulling)
        # unknown: version -1 (never groups); capacity still counts (the
        # replica exists — util is understated, the conservative direction)
        self.assertEqual(by_id[2].version, -1)
        self.assertEqual(by_id[2].kv_capacity, 3200)
        # draining replica is not routable
        ex._draining[1] = "srv-1"
        states = asyncio.run(ex.snapshot())
        self.assertFalse({s.replica_id: s for s in states}[1].routable)

    def test_kv_prev_tracks_decline(self):
        pulls = []
        inflight = {"srv-0": 5}
        views = [_view(0, {0: 5}, inflight, pulls)]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=32))
        s1 = asyncio.run(ex.snapshot())[0]
        self.assertEqual(s1.kv_prev, s1.kv_used)  # first tick: no history
        inflight["srv-0"] = 2  # ramp-down
        s2 = asyncio.run(ex.snapshot())[0]
        self.assertEqual(s2.kv_prev, s1.kv_used)  # previous tick's value
        self.assertTrue(s2.kv_used < s2.kv_prev)

    def test_fleet_kv_util_from_cache(self):
        pulls = []
        inflight = {"srv-0": 8, "srv-1": 0}
        views = [_view(0, {0: 5}, inflight, pulls), _view(1, {1: 5}, inflight, pulls)]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=16))
        asyncio.run(ex.snapshot())
        self.assertAlmostEqual(ex.fleet_kv_util(), 8 / 32)

    def test_no_batch_bound_no_capacity(self):
        """batch_bound None (no honest capacity) -> zero capacity -> the
        planner finds no candidates -> no plans (graceful degradation)."""
        pulls = []
        inflight = {"srv-0": 3}
        views = [_view(0, {0: 5}, inflight, pulls)]
        ex = _executor(views, latest=5, config=RepackConfig())  # batch_bound None
        states = asyncio.run(ex.snapshot())
        self.assertEqual(states[0].kv_capacity, 0)
        self.assertFalse(states[0].idle_candidate)


class TestManagerHook(unittest.TestCase):
    def test_repack_once_runs_refresh_idle_and_survives(self):
        """repack_once with the async-snapshot bridge executor: refresh
        runs, the awaitable snapshot is awaited, no batch_bound -> no plan."""
        pulls = []
        inflight = {"srv-0": 0, "srv-1": 3}
        views = [
            _view(0, {0: 3}, inflight, pulls),
            _view(1, {1: 5}, inflight, pulls),
        ]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=8))
        manager = RepackManager(engine=ex, config=RepackConfig(check_interval_s=60.0))
        asyncio.run(manager.repack_once())
        self.assertEqual(pulls, [(0, 5)])  # refresh ran inside the round
        self.assertEqual(manager.stats.checks, 1)

    def test_planning_over_real_snapshot_produces_and_executes_plan(self):
        """End-to-end: two same-version tail replicas, CanFit passes ->
        the manager plans and the executor drains the source."""
        pulls = []
        # two replicas on v5 with declining tails, one busy keeper on v5
        inflight = {"srv-0": 2, "srv-1": 6, "srv-2": 0}
        views = [
            _view(i, {i: 5}, inflight, pulls) for i in range(3)
        ]
        drain = _FakeDrain()
        ex = _executor(views, latest=6, drain=drain, config=RepackConfig(batch_bound=8))
        # seed kv_prev history so replica 0 shows a decline (idle_candidate)
        asyncio.run(ex.snapshot())
        inflight["srv-0"] = 1
        manager = RepackManager(engine=ex, config=RepackConfig(check_interval_s=60.0, batch_bound=8))
        asyncio.run(manager.repack_once())
        self.assertEqual(manager.stats.plans, 1)
        self.assertEqual(manager.stats.sources_released, 1)
        self.assertEqual(ex.migrations_started, 1)
        self.assertTrue(any(on for _, on in drain.calls))  # begin_drain happened
        # the emptied source is completed by the watcher on the next tick;
        # replica 2 was already refreshed by (b) during repack_once's tick
        # (dynamic versions: it now runs v6 and is never re-pulled)
        inflight["srv-0"] = 0
        asyncio.run(ex.refresh_idle())
        self.assertEqual(ex.migrations_completed, 1)
        self.assertEqual(pulls, [(2, 6), (0, 6)])

    def test_notify_update_wakes_the_loop(self):
        pulls = []
        views = [_view(0, {0: 3}, {"srv-0": 0}, pulls)]
        ex = _executor(views, latest=5, config=RepackConfig(batch_bound=8))
        manager = RepackManager(engine=ex, config=RepackConfig(check_interval_s=60.0))

        async def drive():
            manager.start()
            await asyncio.sleep(0.05)
            manager.notify_update()
            await asyncio.sleep(0.05)
            await manager.stop()

        asyncio.run(drive())
        self.assertEqual(pulls, [(0, 5)])
        self.assertEqual(manager.stats.update_triggers, 1)


class TestBuildWiring(unittest.TestCase):
    def test_no_server_ids_means_conservative(self):
        class _FakeRelay:
            def __getattr__(self, name):
                async def _probe(*args):
                    return None

                return _probe

        manager, executor = build_repack_controller(_FakeRelay(), rollouter=None)
        self.assertEqual(len(executor.handles), 0)
        self.assertIsNone(executor.drain_fn)  # no rollouter -> no drain seam

    def test_config_flows_to_executor(self):
        class _FakeRelay:
            def __getattr__(self, name):
                async def _probe(*args):
                    return None

                return _probe

        cfg = RepackConfig(hard_drain=True, batch_bound=16, kv_per_request=512)
        manager, executor = build_repack_controller(_FakeRelay(), rollouter=None, config=cfg)
        self.assertTrue(executor.config.hard_drain)
        self.assertEqual(executor.config.batch_bound, 16)
        self.assertEqual(executor.config.kv_per_request, 512)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
