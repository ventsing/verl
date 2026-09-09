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
"""End-to-end pipeline tests (mock engine, CPU-only): both delivery modes
over the same workload must train on identical data; failure and staleness
paths must never leak partial groups into training."""

from __future__ import annotations

import asyncio
import random
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.mock_rollout import MockEngineConfig, MockRolloutEngine
from verl.experimental.trajectory_async.rollouter import (
    PromptRecord,
    RollouterConfig,
    TrajectoryRollouter,
)
from verl.experimental.trajectory_async.trainer import TrainerConfig, TrajectoryTrainer
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import TrajectorySample


class PipelineHarness:
    """Builds and runs the full pipeline once; records trained groups."""

    def __init__(
        self,
        mode: str,
        num_prompts: int = 8,
        n: int = 4,
        mini_batch_groups: int = 2,
        update_time_s: float = 0.01,
        reward_latency_s: float = 0.0,
        preprocess_latency_s: float = 0.0,
        failure_rate: float = 0.0,
        max_retries: int = 3,
        staleness_drop: int | None = None,
        tokens_per_s: float = 1e6,
        seed: int = 7,
        preprocess_policy: str = "per-trajectory",
    ) -> None:
        self.trained: dict[str, list[tuple]] = {}
        self.batch_versions: list[int] = []
        engine = MockRolloutEngine(
            MockEngineConfig(
                tokens_per_s=tokens_per_s,
                base_latency_s=0.001,
                failure_rate=failure_rate,
                seed=seed,
            )
        )
        self.engine = engine
        queue = InProcessTrajectoryQueue()
        aggregator = GroupAggregator()
        mini_batcher = MiniBatcher(mini_batch_groups=mini_batch_groups)
        trainer = TrajectoryTrainer(
            queue=queue,
            aggregator=aggregator,
            mini_batcher=mini_batcher,
            config=TrainerConfig(
                mini_batch_groups=mini_batch_groups,
                update_time_s=update_time_s,
                max_staleness_drop=staleness_drop,
                preprocess_policy=preprocess_policy,
            ),
            preprocess_fn=self._make_preprocess(preprocess_latency_s),
        )

        async def update_fn(batch) -> None:
            await asyncio.sleep(update_time_s)
            engine.set_version(trainer.current_version + 1)

        trainer.update_fn = update_fn

        async def reward_fn(traj: TrajectorySample) -> float:
            if reward_latency_s:
                await asyncio.sleep(reward_latency_s)
            rng = random.Random(f"reward:{traj.payload['seed']}")
            return 1.0 if rng.random() < 0.5 else 0.0

        self.rollouter = TrajectoryRollouter(
            engine=engine,
            queue=queue,
            config=RollouterConfig(
                n=n,
                mode=mode,
                max_retries=max_retries,
            ),
            reward_fn=reward_fn,
        )
        self.trainer = trainer
        self.prompts = [PromptRecord(uid=f"p{i:03d}", prompt_tokens=64) for i in range(num_prompts)]

        def on_batch(batch, metrics):
            self.batch_versions.append(batch.policy_version_before)
            for g in batch.groups:
                self.trained[g.uid] = [
                    (t.traj_index, t.num_tokens, t.reward, t.attempts) for t in g.trajectories
                ]

        trainer.on_batch = on_batch

    @staticmethod
    def _make_preprocess(latency_s: float):
        async def preprocess_fn(traj: TrajectorySample) -> TrajectorySample:
            if latency_s:
                await asyncio.sleep(latency_s)
            return traj

        return preprocess_fn

    def run(self):
        async def run_async():
            await asyncio.gather(
                self.rollouter.run(self.prompts, num_consumers=1),
                self.trainer.run(),
            )

        asyncio.run(run_async())
        return self


class TestEndToEndEquivalence(unittest.TestCase):
    def test_both_modes_train_identical_data(self):
        num_prompts, n, mbg = 8, 4, 2
        group = PipelineHarness("group", num_prompts=num_prompts, n=n, mini_batch_groups=mbg).run()
        traj = PipelineHarness("trajectory", num_prompts=num_prompts, n=n, mini_batch_groups=mbg).run()

        self.assertEqual(set(group.trained), set(traj.trained))
        for uid in group.trained:
            self.assertEqual(
                group.trained[uid],
                traj.trained[uid],
                f"group {uid} trained on different data across modes",
            )
        # every non-dropped group trained exactly once, full mini-batches only
        self.assertEqual(group.trainer.stats.groups_trained, len(group.trained))
        self.assertEqual(
            group.trainer.stats.mini_batches, len(group.trained) // mbg
        )
        self.assertEqual(
            traj.trainer.stats.mini_batches, len(traj.trained) // mbg
        )

    def test_accounting_is_consistent(self):
        h = PipelineHarness("trajectory", num_prompts=8, n=4, mini_batch_groups=2).run()
        stats = h.trainer.stats
        total_groups = 8
        self.assertEqual(stats.groups_trained + stats.groups_dropped_stale, total_groups)
        self.assertEqual(stats.updates, stats.mini_batches)
        self.assertEqual(stats.trajectories_consumed, 8 * 4)
        self.assertEqual(h.rollouter.stats.trajectories_pushed, 8 * 4)
        self.assertIsNotNone(stats.time_to_first_batch_s)
        # the trainer main loop only waits on the queue; updates and
        # preprocessing run in concurrent tasks, so their busy times may
        # legitimately overlap with each other and with waiting — they are
        # individually bounded by the wall time, never additive to it.
        self.assertLessEqual(stats.update_busy_time_s, stats.wall_time_s)
        self.assertLessEqual(stats.wait_for_queue_s, stats.wall_time_s)
        # queue fully drained
        self.assertEqual(h.rollouter.queue.total_put, h.trainer.stats.trajectories_consumed)

    def test_leftover_groups_are_counted_not_lost(self):
        """Complete groups that never fill a mini-batch must be accounted:
        trained + dropped-stale + leftover == prompted groups."""
        # 8 prompts, 3 groups per mini-batch -> 2 batches + 2 leftover
        h = PipelineHarness("trajectory", num_prompts=8, n=4, mini_batch_groups=3).run()
        stats = h.trainer.stats
        self.assertEqual(stats.groups_trained, 6)
        self.assertEqual(stats.groups_leftover, 2)
        self.assertEqual(
            stats.groups_trained + stats.groups_dropped_stale + stats.groups_leftover, 8
        )
        self.assertEqual(stats.mini_batches, 2)


class TestPreprocessPolicies(unittest.TestCase):
    def test_on_group_complete_policy_trains_identical_data(self):
        num_prompts, n, mbg = 8, 4, 2
        group = PipelineHarness(
            "group", num_prompts=num_prompts, n=n, mini_batch_groups=mbg,
            preprocess_latency_s=0.005, preprocess_policy="on-group-complete",
        ).run()
        traj = PipelineHarness(
            "trajectory", num_prompts=num_prompts, n=n, mini_batch_groups=mbg,
            preprocess_latency_s=0.005, preprocess_policy="on-group-complete",
        ).run()
        self.assertEqual(set(group.trained), set(traj.trained))
        for uid in group.trained:
            self.assertEqual(group.trained[uid], traj.trained[uid])
        self.assertEqual(group.trainer.stats.groups_trained, num_prompts)
        self.assertEqual(traj.trainer.stats.groups_trained, num_prompts)

    def test_on_group_complete_avoids_wasted_preprocess_on_failure(self):
        """With failures, per-trajectory preprocessing speculatively works
        on trajectories whose group later dies; on-group-complete does
        not — its preprocess busy time only covers trainable groups."""
        h = PipelineHarness(
            "trajectory",
            num_prompts=16,
            n=4,
            mini_batch_groups=2,
            failure_rate=0.3,
            max_retries=1,
            preprocess_latency_s=0.01,
            preprocess_policy="on-group-complete",
        ).run()
        trained_trajs = 4 * len(h.trained)
        # busy time accounts only for trajectories of groups that trained
        expected = trained_trajs * 0.01
        self.assertAlmostEqual(
            h.trainer.stats.preprocess_busy_time_s, expected, delta=expected * 0.5 + 0.05
        )


class TestStalenessAndFailures(unittest.TestCase):
    def test_stale_groups_are_dropped_not_trained(self):
        # slow generation + fast updates ⇒ versions advance while old
        # groups wait; staleness_drop=0 must evict them before training
        h = PipelineHarness(
            "trajectory",
            num_prompts=16,
            n=4,
            mini_batch_groups=2,
            update_time_s=0.002,
            tokens_per_s=2e4,  # ~0.2s per 4k-token trajectory
            staleness_drop=0,
        ).run()
        stats = h.trainer.stats
        self.assertGreater(stats.groups_dropped_stale, 0)
        self.assertLess(stats.groups_trained, 16)
        self.assertEqual(stats.groups_trained, len(h.trained))
        for uid, rows in h.trained.items():
            self.assertEqual(len(rows), 4, "only complete groups may be trained")

    def test_failed_groups_are_never_trained(self):
        h = PipelineHarness(
            "trajectory",
            num_prompts=12,
            n=4,
            mini_batch_groups=2,
            failure_rate=0.3,
            max_retries=2,
        ).run()
        agg = h.trainer.aggregator
        stats = h.trainer.stats
        self.assertGreater(agg.total_groups_evicted, 0)
        self.assertGreaterEqual(agg.total_groups_evicted, h.rollouter.stats.groups_dropped)
        # every group either fully trained or evicted exactly once
        self.assertEqual(stats.groups_trained + agg.total_groups_evicted, 12)
        # trained groups are complete
        for uid, rows in h.trained.items():
            self.assertEqual(len(rows), 4)
        # everything the rollouter put (completed + FAILED sentinels) was consumed
        self.assertEqual(h.rollouter.queue.total_put, stats.trajectories_consumed)


if __name__ == "__main__":
    unittest.main()
