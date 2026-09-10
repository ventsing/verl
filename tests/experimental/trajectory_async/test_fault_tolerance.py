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
"""CPU tests for the partial response pool (§3.1 substrate) and the
fault-tolerance cores (§3.3 health/retire, §4.3 failover + chain rebuild)."""

import asyncio
import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.fault_tolerance import (
    RelaySupervisor,
    ReplicaHealthMonitor,
    rebuild_chain,
)
from verl.experimental.trajectory_async.partial_pool import PartialResponsePool


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestPartialPool(unittest.TestCase):
    def test_put_get_same_version(self):
        pool = PartialResponsePool()
        pool.put("g", 1, model_version=5, tokens=[1, 2, 3])
        entry = pool.get("g", 1, current_version=5)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.tokens, [1, 2, 3])
        self.assertEqual(pool.hits, 1)

    def test_version_mismatch_refused_and_dropped(self):
        """Same-version redirect rule: a partial from another version is
        never resumed from — and is dropped (it can only get staler)."""
        pool = PartialResponsePool()
        pool.put("g", 1, model_version=5, tokens=[1, 2])
        self.assertIsNone(pool.get("g", 1, current_version=6))
        self.assertEqual(pool.version_mismatches, 1)
        # dropped: a second look is a plain miss now
        self.assertIsNone(pool.get("g", 1, current_version=6))
        self.assertEqual(pool.misses, 1)
        self.assertEqual(pool.version_mismatches, 1)

    def test_complete_entries_are_not_resumable(self):
        pool = PartialResponsePool()
        pool.put("g", 1, model_version=5, tokens=[1], complete=True)
        self.assertIsNone(pool.get("g", 1, current_version=5))

    def test_no_version_regression_on_overwrite(self):
        pool = PartialResponsePool()
        pool.put("g", 1, model_version=6, tokens=[9, 9])
        pool.put("g", 1, model_version=5, tokens=[1])  # older — refused
        entry = pool.get("g", 1, current_version=6)
        self.assertEqual(entry.tokens, [9, 9])

    def test_ttl_expiry(self):
        clock = _Clock()
        pool = PartialResponsePool(ttl_s=10.0, clock=clock)
        pool.put("g", 1, model_version=5, tokens=[1])
        clock.advance(11.0)
        self.assertIsNone(pool.get("g", 1, current_version=5))
        self.assertEqual(pool.expired, 1)

    def test_lru_eviction_never_the_just_put(self):
        pool = PartialResponsePool(max_entries=2)
        pool.put("a", 0, 5, [1])
        pool.put("b", 0, 5, [1])
        pool.put("c", 0, 5, [1])  # evicts the oldest ("a")
        self.assertIsNone(pool.get("a", 0, 5))
        self.assertIsNotNone(pool.get("c", 0, 5))
        self.assertEqual(pool.evicted_lru, 1)

    def test_byte_quota_eviction(self):
        pool = PartialResponsePool(max_bytes=8 * 2)  # two 1-token entries
        pool.put("a", 0, 5, [1])
        pool.put("b", 0, 5, [1, 2, 3])  # 24 bytes -> evicts "a"
        self.assertIsNone(pool.get("a", 0, 5))
        self.assertIsNotNone(pool.get("b", 0, 5))
        self.assertGreater(pool.evicted_bytes, 0)

    def test_discard_and_snapshot(self):
        pool = PartialResponsePool()
        pool.put("g", 1, 5, [1, 2, 3])
        pool.discard("g", 1)
        self.assertIsNone(pool.get("g", 1, 5))
        snap = pool.snapshot()
        self.assertEqual(snap["partial_pool/discarded"], 1)
        self.assertEqual(snap["partial_pool/entries"], 0)


class TestReplicaHealthMonitor(unittest.TestCase):
    def test_threshold_strike_declares_dead(self):
        mon = ReplicaHealthMonitor(failure_threshold=3)
        for _ in range(2):
            mon.record("s0", False)
        self.assertEqual(mon.dead_servers(), [])  # below threshold
        mon.record("s0", False)
        self.assertEqual(mon.dead_servers(), ["s0"])
        self.assertEqual(mon.deaths, 1)

    def test_success_resets_streak(self):
        mon = ReplicaHealthMonitor(failure_threshold=3)
        mon.record("s0", False)
        mon.record("s0", False)
        mon.record("s0", True)  # reset
        mon.record("s0", False)
        self.assertEqual(mon.dead_servers(), [])

    def test_retired_servers_not_re_reported(self):
        mon = ReplicaHealthMonitor(failure_threshold=1)
        mon.record("s0", False)
        self.assertEqual(mon.dead_servers(), ["s0"])
        mon.retired(["s0"])
        self.assertEqual(mon.dead_servers(), [])  # already handled
        mon.record("s1", False)
        self.assertEqual(mon.dead_servers(), ["s1"])

    def test_revive_clears_state(self):
        mon = ReplicaHealthMonitor(failure_threshold=1)
        mon.record("s0", False)
        mon.retired(["s0"])
        mon.revive("s0")
        self.assertNotIn("s0", mon._retired)

    def test_probe_tick_batch(self):
        mon = ReplicaHealthMonitor(failure_threshold=2)
        mon.record_probe({"a": False, "b": True})
        mon.record_probe({"a": False, "b": True})
        self.assertEqual(mon.dead_servers(), ["a"])
        self.assertEqual(mon.probe_ticks, 2)


class TestRelaySupervisor(unittest.TestCase):
    def test_heartbeat_failover_recovers(self):
        state = {"alive": True, "recovered": []}

        async def ping():
            if not state["alive"]:
                raise RuntimeError("actor dead")
            return True

        def factory():
            state["alive"] = True  # the new actor answers pings
            return "c1"

        async def recover(new_controller):
            state["recovered"].append(new_controller)

        sup = RelaySupervisor(
            ping_fn=ping,
            factory=factory,
            recover_fn=recover,
            heartbeat_s=0.01,
        )

        async def scenario():
            sup.start("c0")
            await asyncio.sleep(0.05)
            self.assertTrue(sup.controller_alive)
            state["alive"] = False  # the controller dies
            await asyncio.sleep(0.08)
            self.assertTrue(sup.controller_alive)  # failed over
            self.assertEqual(sup.failovers, 1)
            self.assertEqual(state["recovered"], ["c1"])  # consumers re-attached
            await sup.stop()

        asyncio.run(scenario())

    def test_failover_failure_is_survived(self):
        async def ping():
            raise RuntimeError("actor dead")

        def factory():
            raise RuntimeError("resource exhausted")  # resurrection fails

        async def recover(new_controller):
            pass

        sup = RelaySupervisor(ping_fn=ping, factory=factory, recover_fn=recover, heartbeat_s=0.01)

        async def scenario():
            sup.start("c0")
            await asyncio.sleep(0.05)
            self.assertFalse(sup.controller_alive)
            self.assertGreaterEqual(sup.heartbeat_failures, 1)  # counted, survived
            self.assertEqual(sup.failovers, 0)
            await sup.stop()

        asyncio.run(scenario())

    def test_probe_retires_and_revives(self):
        events = {"retired": [], "revived": [], "alive": {"s0": False, "s1": True}}

        async def probe():
            return dict(events["alive"])

        async def retire(ids):
            events["retired"].extend(ids)
            for sid in ids:
                events["alive"][sid] = False

        async def revive(ids):
            events["revived"].extend(ids)
            for sid in ids:
                events["alive"][sid] = True

        async def ping():
            return True

        async def recover(c):
            pass

        sup = RelaySupervisor(
            ping_fn=ping,
            factory=lambda: "c1",
            recover_fn=recover,
            probe_fn=probe,
            retire_fn=retire,
            revive_fn=revive,
            heartbeat_s=0.5,
            probe_s=0.01,
            failure_threshold=2,
        )

        async def scenario():
            sup.start("c0")
            await asyncio.sleep(0.08)  # several probe ticks: s0 strikes out
            self.assertIn("s0", events["retired"])
            self.assertEqual(events["retired"].count("s0"), 1)  # retire fires once
            events["alive"]["s0"] = True  # s0 restarts
            await asyncio.sleep(0.08)
            self.assertIn("s0", events["revived"])
            await sup.stop()

        asyncio.run(scenario())


class TestRebuildChain(unittest.TestCase):
    def test_no_deads_is_identity(self):
        self.assertEqual(rebuild_chain([0, 1, 2, 3], set()), [0, 1, 2, 3])

    def test_middle_death_splices(self):
        """O(dead) splice: neighbors reconnect, order preserved."""
        self.assertEqual(rebuild_chain([0, 1, 2, 3], {1}), [0, 2, 3])

    def test_multiple_deaths(self):
        self.assertEqual(rebuild_chain([0, 1, 2, 3, 4], {1, 3}), [0, 2, 4])

    def test_all_dead_returns_empty(self):
        """An empty chain is a FAILED path — never a silent loop."""
        self.assertEqual(rebuild_chain([0, 1], {0, 1}), [])

    def test_unknown_deads_ignored(self):
        self.assertEqual(rebuild_chain([0, 1], {9}), [0, 1])


class TestRowRetryPoolHook(unittest.TestCase):
    """The pool consult hook: retries receive hints, first attempts don't."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_hint_flows_to_retry_attempt(self):
        from verl.experimental.trajectory_async.row_retry import generate_row_with_retry

        seen_hints = []

        async def gen(row, hint):
            seen_hints.append(hint)
            if len(seen_hints) < 2:
                raise RuntimeError("fail once")
            return "ok"

        async def deliver(result, attempts, version):
            pass

        async def consult(attempt, version):
            return ["tok", "tok2"] if version == 9 else None

        ok = self._run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=deliver,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 9,
                pool_consult_fn=consult,
                max_attempts=2,
            )
        )
        self.assertTrue(ok)
        self.assertEqual(seen_hints, [None, ["tok", "tok2"]])  # retry resumes

    def test_consult_error_never_breaks_retry(self):
        from verl.experimental.trajectory_async.row_retry import generate_row_with_retry

        calls = []

        async def gen(row, hint):
            calls.append(hint)
            if len(calls) < 2:
                raise RuntimeError("fail once")
            return "ok"

        async def deliver(result, attempts, version):
            pass

        async def broken_consult(attempt, version):
            raise RuntimeError("pool exploded")

        ok = self._run(
            generate_row_with_retry(
                "row",
                generate_fn=gen,
                deliver_fn=deliver,
                failed_row_fn=lambda: "FAILED",
                version_for_attempt=lambda a: 1,
                pool_consult_fn=broken_consult,
                max_attempts=2,
            )
        )
        self.assertTrue(ok)  # the pool must never break retries
        self.assertEqual(calls, [None, None])  # clean restart


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
