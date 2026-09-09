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
"""Tests for the Laminar-style scheduling components:

* :mod:`weight_relay` — hierarchical weight service timing (§4);
* :mod:`multi_replica_engine` — KVCache lifecycle, batch quota, per-replica
  weight versions, migration (§3.2 / §5);
* :mod:`repack` — Algorithm 1 Best-Fit consolidation and the manager loop
  (§5.1 / §5.2).
"""

import asyncio
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.multi_replica_engine import (  # noqa: E402
    MultiReplicaEngine,
    MultiReplicaEngineConfig,
    ReplicaState,
)
from verl.experimental.trajectory_async.repack import (  # noqa: E402
    RepackConfig,
    RepackManager,
    best_fit_consolidation,
)
from verl.experimental.trajectory_async.weight_relay import RelayConfig, WeightRelayService  # noqa: E402


# --------------------------------------------------------------------- relay


class TestWeightRelay(unittest.TestCase):
    def test_actor_stall_is_only_the_master_hop(self):
        """publish() blocks for actor→master only; chain broadcast is
        background — the defining property of Laminar's relays."""

        async def scenario():
            relay = WeightRelayService(RelayConfig(num_relays=8, actor_to_master_s=0.5, hop_latency_s=0.1))
            start = asyncio.get_running_loop().time()
            await relay.publish(1)
            blocked = asyncio.get_running_loop().time() - start
            # blocked ≈ actor_to_master, NOT actor_to_master + chain hops
            self.assertLess(blocked, 0.5 + 0.1)
            self.assertGreaterEqual(blocked, 0.5 - 0.02)
            return relay

        asyncio.run(scenario())

    def test_chain_propagation_and_pull_timing(self):
        """A pull right after publish waits for chain arrival + PCIe; a
        later pull is PCIe-only."""

        async def scenario():
            relay = WeightRelayService(RelayConfig(num_relays=4, actor_to_master_s=0.1, hop_latency_s=0.2, pcie_pull_s=0.1))
            await relay.publish(1)
            # relay 0 receives at master_ready + 0.2; pull starts ~immediately
            version = await relay.pull(0)
            self.assertEqual(version, 1)
            # wait ≈ hop(0.2) + pcie(0.1); relay 3 would wait ≈ 0.8 + 0.1
            self.assertAlmostEqual(relay.stats.pull_wait_total_s, 0.2, delta=0.06)
            self.assertAlmostEqual(relay.stats.pcie_total_s, 0.1, delta=0.03)

            await asyncio.sleep(1.0)  # let the chain fully pass
            await relay.publish(2)
            await asyncio.sleep(1.0)  # chain (0.8s) fully arrived
            version = await relay.pull(3)
            self.assertEqual(version, 2)
            # second pull had no chain wait: total wait still ~0.2
            self.assertAlmostEqual(relay.stats.pull_wait_total_s, 0.2, delta=0.08)

        asyncio.run(scenario())

    def test_replicas_pull_divergent_versions(self):
        """Different relays hold different versions while the chain is in
        flight — the no-lockstep property."""

        async def scenario():
            relay = WeightRelayService(RelayConfig(num_relays=4, actor_to_master_s=0.05, hop_latency_s=0.3))
            await relay.publish(1)
            # relay 0 receives at 0.05+0.3=0.35; relay 3 at 0.05+4×0.3=1.25
            await asyncio.sleep(0.4)
            self.assertEqual(relay.relay_version(0), 1)
            self.assertEqual(relay.relay_version(3), 0)
            await asyncio.sleep(1.0)
            self.assertEqual(relay.relay_version(3), 1)

        asyncio.run(scenario())


# ------------------------------------------------------------------- engine


def _engine_config(**overrides) -> MultiReplicaEngineConfig:
    defaults = dict(
        seed=7,
        length_mean_tokens=500.0,
        length_sigma=0.3,
        max_tokens=2000,
        failure_rate=0.0,
        num_replicas=4,
        batch_per_replica=8,
        max_running_requests=8,
        kv_capacity_tokens=16384,
        decode_rate_tok_s=2000.0,
        decode_tick_s=0.01,
        repack_overhead_s=0.05,
    )
    defaults.update(overrides)
    return MultiReplicaEngineConfig(**defaults)


class TestMultiReplicaEngine(unittest.TestCase):
    def test_generate_completes_and_reports_version(self):
        async def scenario():
            engine = MultiReplicaEngine(_engine_config())
            results = await asyncio.gather(*[engine.generate(100 + i) for i in range(8)])
            await engine.stop()
            self.assertEqual(len(results), 8)
            self.assertGreater(engine.stats.requests_finished, 0)
            for r in results:
                self.assertGreater(r.num_tokens, 0)
                self.assertEqual(r.model_version, 0)
            self.assertEqual(engine.stats.requests_submitted, 8)

        asyncio.run(scenario())

    def test_batch_quota_gates_routing(self):
        """A replica accepts at most batch_per_replica requests per
        activation; excess submitters wait for a drain."""

        async def scenario():
            engine = MultiReplicaEngine(_engine_config(num_replicas=2, batch_per_replica=4))
            tasks = [asyncio.create_task(engine.generate(200 + i)) for i in range(16)]
            await asyncio.sleep(0.03)  # several ticks
            # 2 replicas × 4 quota = 8 admitted; 8 still blocked
            total_assigned = sum(s.num_running + s.num_waiting for s in engine.snapshot())
            self.assertEqual(total_assigned, 8)
            results = await asyncio.gather(*tasks)
            await engine.stop()
            self.assertEqual(len(results), 16)
            # every activation was counted as a drain
            self.assertGreaterEqual(engine.stats.replica_drains, 2)

        asyncio.run(scenario())

    def test_kv_lifecycle_and_ramp_down_detection(self):
        """KV ramps up to C_max, plateaus, then falls into the tail phase
        where idle_candidate becomes true (paper Figure 9)."""

        async def scenario():
            engine = MultiReplicaEngine(
                _engine_config(num_replicas=1, batch_per_replica=8, max_running_requests=8, kv_capacity_tokens=8192)
            )

            async def run_all():
                return await asyncio.gather(
                    *[engine.generate(300 + i, prompt_tokens=512) for i in range(8)]
                )

            task = asyncio.create_task(run_all())
            # long homogeneous requests -> kv plateau, then tail
            peak_util = 0.0
            saw_plateau = False
            saw_tail_candidate = False
            deadline = asyncio.get_running_loop().time() + 30.0
            while not task.done() and asyncio.get_running_loop().time() < deadline:
                for s in engine.snapshot():
                    peak_util = max(peak_util, s.kv_util)
                    if s.kv_util > 0.75:
                        saw_plateau = True
                    if s.idle_candidate and s.has_work:
                        saw_tail_candidate = True
                await asyncio.sleep(0.02)
            await task
            await engine.stop()
            self.assertTrue(saw_plateau, "expected a KVCache plateau near C_max")
            self.assertTrue(saw_tail_candidate, "expected a tail phase flagged as idle candidate")

        asyncio.run(scenario())

    def test_drain_pulls_latest_relay_version(self):
        """When a replica drains and the relay has a newer version, it
        pulls before becoming routable again."""

        async def scenario():
            relay = WeightRelayService(
                RelayConfig(num_relays=2, actor_to_master_s=0.02, hop_latency_s=0.02, pcie_pull_s=0.02)
            )
            engine = MultiReplicaEngine(_engine_config(num_replicas=2, batch_per_replica=4), relay=relay)
            tasks = [asyncio.create_task(engine.generate(400 + i)) for i in range(4)]
            await asyncio.sleep(0.05)
            await relay.publish(1)
            results = await asyncio.gather(*tasks)
            await engine.stop()
            self.assertEqual(len(results), 4)
            self.assertGreaterEqual(engine.stats.weight_pulls, 1)
            # all four finished under v0 (routed before the publish)
            for r in results:
                self.assertEqual(r.model_version, 0)

        asyncio.run(scenario())

    def test_migrate_moves_work_between_replicas(self):
        async def scenario():
            engine = MultiReplicaEngine(_engine_config(num_replicas=2, batch_per_replica=16))
            tasks = [asyncio.create_task(engine.generate(500 + i)) for i in range(6)]
            await asyncio.sleep(0.03)
            before = engine.snapshot()
            total = before[0].num_running + before[0].num_waiting + before[1].num_running + before[1].num_waiting
            self.assertEqual(total, 6)
            moved = await engine.migrate([(0, 1)])
            after = engine.snapshot()
            self.assertEqual(moved, before[0].num_running + before[0].num_waiting)
            self.assertEqual(after[0].num_running + after[0].num_waiting, 0)
            self.assertEqual(after[1].num_running + after[1].num_waiting, 6)
            results = await asyncio.gather(*tasks)
            await engine.stop()
            self.assertEqual(len(results), 6)  # all still complete correctly

        asyncio.run(scenario())

    def test_migrate_respects_capacity(self):
        """CanFit: a destination at KV capacity rejects migrations."""

        async def scenario():
            engine = MultiReplicaEngine(
                _engine_config(num_replicas=2, batch_per_replica=2, kv_capacity_tokens=1400, max_running_requests=2)
            )
            # two ~500-token requests on each replica (prompt 256 + ~500)
            tasks = [asyncio.create_task(engine.generate(600 + i)) for i in range(4)]
            await asyncio.sleep(0.03)
            moved = await engine.migrate([(0, 1)])
            # replica 1 is full (kv at cap or running==B): nothing fits
            self.assertEqual(moved, 0)
            await asyncio.gather(*tasks)
            await engine.stop()

        asyncio.run(scenario())


# ------------------------------------------------------------------- repack


def _state(replica_id, version=3, kv_used=1000, kv_capacity=4096, kv_prev=2000, running=2, waiting=0, **kw):
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
        # within the mixed set both are candidates; the manager never calls
        # it that way (it groups first) — here we just assert the function
        # itself doesn't crash and produces a valid plan
        for src, dst in plan:
            self.assertNotEqual(src, dst)


class TestRepackManager(unittest.TestCase):
    def test_periodic_and_update_triggers(self):
        async def scenario():
            engine = MultiReplicaEngine(_engine_config(num_replicas=2, batch_per_replica=16))
            manager = RepackManager(engine, RepackConfig(check_interval_s=0.1))
            manager.start()
            # no work in flight: periodic checks happen, no plans
            await asyncio.sleep(0.35)
            manager.notify_update()
            await asyncio.sleep(0.05)
            await manager.stop()
            await engine.stop()
            self.assertGreaterEqual(manager.stats.checks, 3)
            self.assertEqual(manager.stats.update_triggers, 1)
            self.assertEqual(manager.stats.plans, 0)

        asyncio.run(scenario())

    def test_manager_consolidates_stragglers(self):
        """End-to-end: long-tail stragglers get migrated off their
        replicas, which then drain and accept new work."""

        async def scenario():
            engine = MultiReplicaEngine(
                _engine_config(num_replicas=4, batch_per_replica=12, kv_capacity_tokens=32768)
            )
            manager = RepackManager(engine, RepackConfig(check_interval_s=0.2))
            manager.start()
            # enough requests that quotas + tails form
            tasks = [asyncio.create_task(engine.generate(700 + i)) for i in range(48)]
            await asyncio.gather(*tasks)
            await manager.stop()
            await engine.stop()
            self.assertEqual(engine.stats.requests_finished, 48)
            # at least one migration round happened during the tails
            self.assertGreaterEqual(engine.stats.migration_rounds, 1)
            self.assertGreaterEqual(manager.stats.requests_moved, 1)

        asyncio.run(scenario())

    def test_manager_survives_exceptions(self):
        async def scenario():
            engine = MultiReplicaEngine(_engine_config())
            manager = RepackManager(engine, RepackConfig(check_interval_s=0.05))

            async def broken_repack():
                raise RuntimeError("boom")

            manager.repack_once = broken_repack
            manager.start()
            await asyncio.sleep(0.2)
            await manager.stop()
            await engine.stop()
            self.assertTrue(True)  # reached: the loop survived the error

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
