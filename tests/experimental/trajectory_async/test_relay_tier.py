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
"""Tests for the hierarchical relay tier (Laminar §4) and the rollout
repack executor — both run entirely on the fake P2P transport."""

import asyncio
import unittest

from tests.experimental.trajectory_async import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.relay_tier import (
    RelayService,
    RelayTierConfig,
    RelayTierAdapter,
    RolloutRepackExecutor,
    RunningRequest,
)
from verl.experimental.trajectory_async.versioned_weight_store import FakeP2PBackend


def _weights(n: int = 4, size: int = 64) -> list[tuple[str, bytes]]:
    return [(f"t.{i}", b"x" * size) for i in range(n)]


def _service(num_relays: int = 3, chunks: int = 4, keep_last: int = 2, **kw):
    config = RelayTierConfig(num_relays=num_relays, chunks=chunks, keep_last=keep_last, **kw)
    backends = [FakeP2PBackend(stage_latency_s=0.0, read_latency_s=0.0) for _ in range(num_relays)]
    return RelayService(backends, config)


class TestChainDistribution(unittest.TestCase):
    def test_publish_distributes_to_every_node(self):
        async def run():
            svc = _service(num_relays=3)
            await svc.publish(1, _weights())
            await svc.wait_for_distribution()
            self.assertEqual(svc.latest_published_version(), 1)
            for node in svc.nodes:
                self.assertTrue(node.version_complete(1))
                self.assertEqual(node.latest_complete(), 1)
            # every replica pulls its local relay and sees v1
            for replica in range(6):
                pulled = await svc.pull(replica)
                self.assertEqual(pulled, 1)
            snap = svc.snapshot()
            self.assertEqual(snap["relay/publishes"], 1)
            self.assertEqual(snap["relay/local_versions"], {0: 1, 1: 1, 2: 1})
            self.assertEqual(snap["relay/max_node_lag"], 0)

        asyncio.run(run())

    def test_actor_stall_is_master_stage_only(self):
        """publish() returns after the master staged — before the chain
        finished distributing to the other relays (paper §4.2 goal 1)."""

        async def run():
            config = RelayTierConfig(
                num_relays=3,
                chunks=4,
                hop_read_latency_s=0.2,
                local_read_latency_s=0.0,
            )
            backends = [
                FakeP2PBackend(stage_latency_s=0.05, read_latency_s=0.0)
                for _ in range(3)
            ]
            # hop latency lives in consumer_ctx; the fake backend honors it
            svc = RelayService(backends, config)
            await svc.publish(1, _weights())
            # master complete, far relays not yet (distribution in flight)
            self.assertTrue(svc.nodes[0].version_complete(1))
            self.assertFalse(svc.nodes[2].version_complete(1))
            await svc.wait_for_distribution()
            self.assertTrue(svc.nodes[2].version_complete(1))

        asyncio.run(run())

    def test_chain_pipelining_faster_than_sequential(self):
        """Relay i starts chunk k as soon as relay i-1 staged it — total
        time ~ (chunks + hops - 1) * chunk_time, not hops * chunks * it."""

        async def run():
            chunks, relays = 4, 3
            config = RelayTierConfig(
                num_relays=relays,
                chunks=chunks,
                hop_read_latency_s=0.05,  # network read per chunk
                local_read_latency_s=0.0,
            )
            backends = [
                FakeP2PBackend(stage_latency_s=0.1, read_latency_s=0.0)
                for _ in range(relays)
            ]
            svc = RelayService(backends, config)
            await svc.publish(1, _weights())
            await svc.wait_for_distribution()
            # pipelined bound: 0.15 * (4 + 2 - 1) = 0.75s; sequential would
            # be 2 hops * 4 chunks * 0.15 = 1.2s — assert under 0.95 with
            # margin
            self.assertLess(svc.stats.chain_completion_s[0], 0.95)

        asyncio.run(run())

    def test_local_pull_anytime_takes_previous_version(self):
        """A pull during an in-flight broadcast gets the local relay's
        latest COMPLETE version — never a partial one, never blocking."""

        async def run():
            config = RelayTierConfig(
                num_relays=2, chunks=2, hop_read_latency_s=0.1, local_read_latency_s=0.0
            )
            backends = [FakeP2PBackend(0.0, 0.0) for _ in range(2)]
            svc = RelayService(backends, config)
            await svc.publish(1, _weights(2))
            await svc.wait_for_distribution()
            # v2 distribution in flight; replica on the DOWNSTREAM relay
            # (node 1) pulls and gets v1 — v2 is not complete locally yet
            await svc.publish(2, _weights(2))
            self.assertEqual(await svc.pull(1), 1)
            # the master's local replica already sees v2
            self.assertEqual(await svc.pull(0), 2)
            await svc.wait_for_distribution()
            self.assertEqual(await svc.pull(1), 2)

        asyncio.run(run())

    def test_pull_waits_at_startup_edge(self):
        """Before ANY complete version exists on the local relay, the pull
        waits for the broadcast instead of failing."""

        async def run():
            config = RelayTierConfig(
                num_relays=2, chunks=2, hop_read_latency_s=0.05, local_read_latency_s=0.0
            )
            backends = [FakeP2PBackend(0.0, 0.0) for _ in range(2)]
            svc = RelayService(backends, config)
            await svc.publish(1, _weights(2))  # distribution still running
            pulled = await svc.pull(1)  # node 1 has nothing complete yet
            self.assertEqual(pulled, 1)
            self.assertGreater(svc.stats.pull_wait_s, 0.0)

        asyncio.run(run())

    def test_versions_must_increase(self):
        async def run():
            svc = _service()
            await svc.publish(1, _weights())
            await svc.publish(2, _weights())
            with self.assertRaises(ValueError):
                await svc.publish(2, _weights())
            with self.assertRaises(ValueError):
                await svc.publish(1, _weights())

        asyncio.run(run())

    def test_empty_chunks_kept_in_bookkeeping(self):
        """Fewer tensors than chunks: empty chunks complete the grid."""

        async def run():
            svc = _service(chunks=4)
            await svc.publish(1, [("only.tensor", b"x" * 16)])
            await svc.wait_for_distribution()
            for node in svc.nodes:
                self.assertTrue(node.version_complete(1))
            self.assertEqual(await svc.pull(3), 1)

        asyncio.run(run())


class TestRelayHooks(unittest.TestCase):
    def test_format_fn_applied_once_at_master(self):
        """Trainer-side HF-format conversion: every consumer sees the
        transformed names on every node."""

        async def run():
            svc = _service(num_relays=2)
            await svc.publish(
                1,
                [("layer.0.weight", b"x" * 8), ("layer.1.weight", b"x" * 8)],
                format_fn=lambda name, tensor: (f"hf:{name}", tensor),
            )
            await svc.wait_for_distribution()
            for replica in range(2):
                seen: list[str] = []
                await svc.pull(replica, sink=lambda n, t: seen.append(n))
                self.assertEqual(seen, ["hf:layer.0.weight", "hf:layer.1.weight"])

        asyncio.run(run())

    def test_reshard_fn_master_side_layout_transform(self):
        """Master-side resharding (paper §4.2): trainer-coordinate names
        convert to the rollout TP layout ONCE at the master; every relay
        (and thus every replica) sees the same converted set — the chain
        broadcasts the full converted model, so a downstream relay never
        depends on another relay's shard selection."""

        async def run():
            svc = _service(num_relays=3)
            weights = [
                ("layer.0.weight", b"a" * 8),
                ("layer.1.weight", b"b" * 8),
                ("optimizer.state", b"c" * 8),  # not rollout-loadable: dropped
            ]

            def reshard(name, tensor):
                if name.startswith("optimizer."):
                    return None  # never broadcast
                tp = 0 if name.endswith("0.weight") else 1
                return f"tp.{tp}/{name}", tensor

            await svc.publish(1, weights, reshard_fn=reshard)
            await svc.wait_for_distribution()
            for node in svc.nodes:
                seen: list[str] = []
                await svc.pull(node.node_id, sink=lambda n, t: seen.append(n))
                self.assertEqual(
                    seen, ["tp.0/layer.0.weight", "tp.1/layer.1.weight"]
                )

        asyncio.run(run())

    def test_retention_per_node(self):
        async def run():
            svc = _service(num_relays=2, keep_last=1)
            for v in (1, 2, 3):
                await svc.publish(v, _weights(2))
                await svc.wait_for_distribution()
            for node in svc.nodes:
                self.assertEqual(node.latest_complete(), 3)
                self.assertEqual(node.complete_versions(), [3])
            # v1 evicted everywhere; pinned pull fails
            with self.assertRaises(LookupError):
                await svc.pull(0, version=1)
            self.assertEqual(await svc.pull(0), 3)

        asyncio.run(run())

    def test_node_for_replica_mapping(self):
        async def run():
            svc = _service(num_relays=2)
            svc._node_for_replica = lambda r: 0 if r < 4 else 1
            await svc.publish(1, _weights(2))
            await svc.wait_for_distribution()
            await svc.pull(5)
            self.assertIn("replica-5", svc.consumer_versions())
            self.assertEqual(svc.consumer_versions()["replica-5"], 1)

        asyncio.run(run())


class TestRelayTierAdapter(unittest.TestCase):
    def test_demo_contract(self):
        async def run():
            svc = _service(num_relays=2, chunks=4)
            adapter = RelayTierAdapter(svc, weight_mb=1.0 / 1024)  # 1 KB
            stall = await adapter.publish(1)
            self.assertGreaterEqual(stall, 0.0)
            self.assertEqual(adapter.latest_published_version(), 1)
            await svc.wait_for_distribution()
            self.assertEqual(await adapter.pull(1), 1)
            snap = adapter.stats.snapshot()
            self.assertEqual(snap["relay/publishes"], 1)
            self.assertEqual(snap["relay/pulls"], 1)
            self.assertIn("relay/chain_completion_s", snap)

        asyncio.run(run())


# ------------------------------------------------- rollout repack executor


class _FakeHandle:
    """Duck-typed rollout replica for the executor tests."""

    def __init__(self, replica_id: int, version: int = 0, kv_capacity: int = 1000,
                 batch_limit: int = 64, requests=None) -> None:
        self.replica_id = replica_id
        self.weight_version = version
        self.kv_capacity = kv_capacity
        self._batch_limit = batch_limit
        self._requests: dict[str, RunningRequest] = {r.request_id: r for r in (requests or [])}
        self.pulled_versions: list[int | None] = []
        self.admitted: list[tuple[str, str]] = []  # (request_id, prefill)

    def kv_used_tokens(self) -> int:
        return sum(r.kv_tokens for r in self._requests.values())

    def kv_capacity_tokens(self) -> int:
        return self.kv_capacity

    def batch_limit(self) -> int:
        return self._batch_limit

    def running_requests(self) -> list[RunningRequest]:
        return list(self._requests.values())

    def remove_request(self, request_id: str) -> RunningRequest:
        return self._requests.pop(request_id)

    async def admit_request(self, request: RunningRequest, *, prefill: str = "recompute") -> None:
        self._requests[request.request_id] = request
        self.admitted.append((request.request_id, prefill))

    async def pull_weights(self, version: int | None = None) -> int:
        self.pulled_versions.append(version)
        self.weight_version = version if version is not None else self.weight_version + 1
        return self.weight_version


class TestRolloutRepackExecutor(unittest.TestCase):
    def _handles(self):
        src = _FakeHandle(
            0, version=2, kv_capacity=1000, batch_limit=64,
            requests=[RunningRequest("r1", 300), RunningRequest("r2", 100)],
        )
        dst = _FakeHandle(1, version=2, kv_capacity=1000, batch_limit=64)
        return src, dst

    def test_snapshot_builds_algorithm_states(self):
        src, dst = self._handles()
        executor = RolloutRepackExecutor([src, dst])
        states = executor.snapshot()
        by_id = {s.replica_id: s for s in states}
        self.assertEqual(by_id[0].kv_used, 400)
        self.assertEqual(by_id[0].num_running, 2)
        self.assertEqual(by_id[0].version, 2)
        self.assertEqual(by_id[0].max_running, 64)
        # fleet utilization over handles
        self.assertAlmostEqual(executor.fleet_kv_util(), 400 / 2000)

    def test_migrate_moves_requests_and_frees_source(self):
        async def run():
            src, dst = self._handles()
            executor = RolloutRepackExecutor([src, dst], repack_overhead_s=0.25)
            self.assertEqual(executor.repack_overhead_s, 0.25)
            result = await executor.migrate([(0, 1)])
            self.assertEqual(result.requests_moved, 2)
            self.assertEqual(result.kv_tokens_moved, 400)
            self.assertEqual(result.sources_emptied, 1)
            # source emptied and pulled fresh weights (the repack payoff)
            self.assertEqual(len(src.pulled_versions), 1)
            self.assertEqual(dst.weight_version, 2)  # destination untouched
            # default prefill: recompute (portable across stacks)
            self.assertEqual(dst.admitted, [("r1", "recompute"), ("r2", "recompute")])

        asyncio.run(run())

    def test_migrate_with_kv_transfer(self):
        async def run():
            src, dst = self._handles()
            transferred: list[tuple[int, int, str]] = []

            async def kv_transfer(src_id, dst_id, request):
                transferred.append((src_id, dst_id, request.request_id))

            executor = RolloutRepackExecutor([src, dst], kv_transfer_fn=kv_transfer)
            await executor.migrate([(0, 1)])
            self.assertEqual(transferred, [(0, 1, "r1"), (0, 1, "r2")])
            self.assertEqual(dst.admitted, [("r1", "kv_transfer"), ("r2", "kv_transfer")])

        asyncio.run(run())

    def test_manager_drives_executor_end_to_end(self):
        """RepackManager + RolloutRepackExecutor: the real-rollout binding
        works through the same manager loop as the mock engine."""

        async def run():
            from verl.experimental.trajectory_async.repack import (
                RepackConfig,
                RepackManager,
                best_fit_consolidation,
            )

            # three stragglers on v2 (Algorithm 1: destinations come from
            # the straggler set — pack onto the fullest viable one)
            a = _FakeHandle(0, version=2, kv_capacity=1000, requests=[RunningRequest("a", 100)])
            b = _FakeHandle(1, version=2, kv_capacity=1000, requests=[RunningRequest("b", 120)])
            c = _FakeHandle(2, version=2, kv_capacity=1000, requests=[RunningRequest("c", 80)])
            executor = RolloutRepackExecutor([a, b, c])
            manager = RepackManager(engine=executor, config=RepackConfig(check_interval_s=0.01))
            try:
                plan = await manager.repack_once()
                self.assertTrue(plan)  # consolidation happened
                # b (fullest viable) absorbed a and c; both were freed and
                # pulled fresh weights (the repack payoff)
                self.assertEqual(len(b.running_requests()), 3)
                self.assertEqual(len(a.pulled_versions), 1)
                self.assertEqual(len(c.pulled_versions), 1)
                self.assertEqual(len(b.pulled_versions), 0)
            finally:
                await manager.stop()

        asyncio.run(run())

    def test_algorithm_runs_on_executor_states(self):
        """best_fit_consolidation accepts executor snapshots directly;
        destinations come from the straggler set itself (Algorithm 1)."""
        from verl.experimental.trajectory_async.repack import best_fit_consolidation

        # two stragglers, each with a little work: the smaller one is
        # consolidated onto the fuller one and gets released
        a = _FakeHandle(0, version=2, kv_capacity=1000, requests=[RunningRequest("a", 300)])
        b = _FakeHandle(1, version=2, kv_capacity=1000, requests=[RunningRequest("b", 100)])
        executor = RolloutRepackExecutor([a, b])
        states = executor.snapshot()
        self.assertTrue(all(s.idle_candidate for s in states))
        plan = best_fit_consolidation(states)
        self.assertEqual(plan, [(1, 0)])  # b (smaller) packs onto a (fuller)


if __name__ == "__main__":
    unittest.main()
