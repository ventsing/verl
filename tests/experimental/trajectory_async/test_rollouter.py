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
"""Unit tests for the trajectory-level rollouter.

Uses a scripted fake engine (exact per-(uid, traj_index) latencies) for
timing assertions, and the real mock engine for data-equivalence and
failure-semantics checks.
"""

from __future__ import annotations

import asyncio
import random
import types
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.mock_rollout import (
    MockEngineConfig,
    MockGenResult,
    MockRolloutEngine,
    request_seed,
)
from verl.experimental.trajectory_async.rollouter import (
    PromptRecord,
    RollouterConfig,
    TrajectoryRollouter,
)
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import TrajectorySample, TrajectoryStatus

BASE_SEED = 7


class ScriptedEngine:
    """Engine with fixed per-(uid, traj_index) latencies (first attempt only)."""

    def __init__(self, prompts: list[PromptRecord], n: int, fast_s: float, slow_s: float) -> None:
        self.config = types.SimpleNamespace(seed=BASE_SEED)
        self.version = 0
        self._seed_map = {
            request_seed(p.uid, i, 1, base_seed=BASE_SEED): (p.uid, i) for p in prompts for i in range(n)
        }
        self.fast_s = fast_s
        self.slow_s = slow_s

    async def generate(self, seed: int, prompt_tokens: int = 256) -> MockGenResult:
        uid, idx = self._seed_map[seed]
        latency = self.fast_s if idx == 0 else self.slow_s
        await asyncio.sleep(latency)
        return MockGenResult(num_tokens=100 + idx, latency_s=latency, model_version=self.version)


def make_prompts(count: int, prompt_tokens: int = 64) -> list[PromptRecord]:
    return [PromptRecord(uid=f"p{i:03d}", prompt_tokens=prompt_tokens) for i in range(count)]


async def deterministic_reward(traj: TrajectorySample) -> float:
    rng = random.Random(f"reward:{traj.payload['seed']}")
    return 1.0 if rng.random() < 0.5 else 0.0


async def drain_queue(queue: InProcessTrajectoryQueue) -> list:
    """Consume until the first end-of-stream sentinel."""
    items = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=10)
        if item is None:
            return items
        items.append(item)


class TestDeliveryTiming(unittest.TestCase):
    def _run_with_probe(self, mode: str):
        """Run one prompt (n=2: fast traj #0, slow traj #1) and return
        (first_dequeued, elapsed_until_first, remaining_items)."""
        prompts = make_prompts(1)
        engine = ScriptedEngine(prompts, n=2, fast_s=0.02, slow_s=0.5)
        queue = InProcessTrajectoryQueue()

        async def run():
            rollouter = TrajectoryRollouter(
                engine,
                queue,
                RollouterConfig(n=2, mode=mode),
                reward_fn=deterministic_reward,
            )
            run_task = asyncio.create_task(rollouter.run(prompts))
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            first = await asyncio.wait_for(queue.get(), timeout=5)
            elapsed = loop.time() - t0
            remaining = await drain_queue(queue)
            await run_task
            return first, elapsed, remaining

        return asyncio.run(run())

    def test_trajectory_mode_delivers_fast_sibling_before_slow_one_finishes(self):
        first, elapsed, remaining = self._run_with_probe("trajectory")
        self.assertEqual(first.traj_index, 0)
        self.assertLess(
            elapsed, 0.35, "fast trajectory must cross the queue before the slow sibling finishes"
        )
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].traj_index, 1)

    def test_group_mode_holds_delivery_until_whole_group_settled(self):
        first, elapsed, remaining = self._run_with_probe("group")
        self.assertGreaterEqual(elapsed, 0.45, "group delivery must wait for the slowest response")
        self.assertEqual(first.traj_index, 0)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].traj_index, 1)


class TestModeEquivalence(unittest.TestCase):
    def test_both_modes_generate_identical_data(self):
        """Same seeds ⇒ same lengths, same rewards, same groups — the A/B
        comparison only differs in timing, never in content."""
        prompts = make_prompts(6)
        n = 4

        def collect(mode: str) -> dict[tuple[str, int], tuple]:
            async def run() -> list:
                engine = MockRolloutEngine(
                    MockEngineConfig(tokens_per_s=1e6, base_latency_s=0.001, seed=BASE_SEED)
                )
                queue = InProcessTrajectoryQueue()
                rollouter = TrajectoryRollouter(
                    engine,
                    queue,
                    RollouterConfig(n=n, mode=mode),
                    reward_fn=deterministic_reward,
                )
                run_task = asyncio.create_task(rollouter.run(prompts))
                items = await drain_queue(queue)
                await run_task
                return items

            items = asyncio.run(run())
            return {(t.uid, t.traj_index): (t.num_tokens, t.reward, t.attempts) for t in items}

        traj_data = collect("trajectory")
        group_data = collect("group")
        self.assertEqual(traj_data, group_data)
        self.assertEqual(len(traj_data), 6 * n)


class TestFailureSemantics(unittest.TestCase):
    def _run_mode(self, mode: str, group_retry: str, prompts, n, failure_rate, max_retries) -> list:
        async def run() -> list:
            engine = MockRolloutEngine(
                MockEngineConfig(
                    tokens_per_s=1e6,
                    base_latency_s=0.001,
                    failure_rate=failure_rate,
                    seed=BASE_SEED,
                )
            )
            queue = InProcessTrajectoryQueue()
            rollouter = TrajectoryRollouter(
                engine,
                queue,
                RollouterConfig(n=n, mode=mode, max_retries=max_retries, group_retry=group_retry),
                reward_fn=deterministic_reward,
            )
            run_task = asyncio.create_task(rollouter.run(prompts))
            items = await drain_queue(queue)
            await run_task
            return items

        return asyncio.run(run())

    def test_per_trajectory_retry_saves_groups_that_all_or_nothing_drops(self):
        prompts = make_prompts(8)
        n = 4
        saved = self._run_mode("trajectory", "per-trajectory", prompts, n, 0.3, 5)
        baseline = self._run_mode("group", "none", prompts, n, 0.3, 5)

        saved_groups = {t.uid for t in saved if t.status == TrajectoryStatus.COMPLETED}
        baseline_groups = {t.uid for t in baseline}
        # all-or-nothing delivered strictly less than a full run
        self.assertLess(len(baseline), len(prompts) * n)
        # per-trajectory retry can only recover groups, never lose extra ones
        self.assertGreaterEqual(len(saved_groups), len(baseline_groups))
        self.assertTrue(baseline_groups.issubset(saved_groups))

    def test_failed_sentinels_arrive_and_retries_happen(self):
        prompts = make_prompts(8)
        n = 4
        # p(dead after retries) = 0.5^3 ≈ 12.5% per trajectory ⇒ ~4 dead of 32
        items = self._run_mode("trajectory", "per-trajectory", prompts, n, 0.5, 2)
        statuses = [t.status for t in items]
        self.assertIn(TrajectoryStatus.FAILED, statuses)
        attempts = [t.attempts for t in items if t.status == TrajectoryStatus.COMPLETED]
        self.assertTrue(any(a > 1 for a in attempts), "some trajectory must have succeeded via retry")

    def test_sentinel_count_matches_consumers(self):
        prompts = make_prompts(2)

        async def run():
            engine = MockRolloutEngine(
                MockEngineConfig(tokens_per_s=1e6, base_latency_s=0.001, seed=BASE_SEED)
            )
            queue = InProcessTrajectoryQueue()
            rollouter = TrajectoryRollouter(engine, queue, RollouterConfig(n=2, mode="trajectory"))
            run_task = asyncio.create_task(rollouter.run(prompts, num_consumers=3))
            items = await drain_queue(queue)
            extra = [await asyncio.wait_for(queue.get(), timeout=5) for _ in range(2)]
            await run_task
            return items, extra

        items, extra = asyncio.run(run())
        self.assertEqual(len(items), 4)  # 2 prompts × n=2, no failures
        self.assertEqual(extra, [None, None])


if __name__ == "__main__":
    unittest.main()
