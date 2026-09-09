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
"""Unit tests for the trainer-side group aggregator (stdlib-only)."""

from __future__ import annotations

import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.types import (
    GroupRecord,
    TrajectorySample,
    TrajectoryStatus,
)


def make_traj(uid: str, idx: int, n: int = 4, version: int = 0, reward: float | None = None) -> TrajectorySample:
    return TrajectorySample(
        uid=uid,
        traj_index=idx,
        group_size=n,
        model_version=version,
        reward=reward if reward is not None else float(idx),
        num_tokens=10 * (idx + 1),
    )


class TestGroupCompletion(unittest.TestCase):
    def test_group_completes_only_when_all_arrive(self):
        agg = GroupAggregator()
        self.assertIsNone(agg.add_trajectory(make_traj("a", 0)))
        self.assertIsNone(agg.add_trajectory(make_traj("a", 3)))
        self.assertIsNone(agg.add_trajectory(make_traj("a", 1)))
        self.assertEqual(agg.num_partial_groups, 1)
        self.assertEqual(agg.num_buffered_trajectories, 3)
        group = agg.add_trajectory(make_traj("a", 2))
        self.assertIsNotNone(group)
        self.assertEqual(group.uid, "a")
        self.assertEqual(group.group_size, 4)
        self.assertEqual([t.traj_index for t in group.trajectories], [0, 1, 2, 3])
        self.assertEqual(agg.num_partial_groups, 0)
        self.assertEqual(agg.total_groups_completed, 1)

    def test_interleaved_groups_emit_in_completion_order(self):
        agg = GroupAggregator()
        agg.add_trajectory(make_traj("a", 0, n=2))
        agg.add_trajectory(make_traj("b", 0, n=2))
        first = agg.add_trajectory(make_traj("b", 1, n=2))
        self.assertEqual(first.uid, "b")  # b completed before a
        second = agg.add_trajectory(make_traj("a", 1, n=2))
        self.assertEqual(second.uid, "a")
        self.assertEqual(agg.total_groups_completed, 2)

    def test_duplicate_arrival_keeps_first(self):
        agg = GroupAggregator()
        first = make_traj("a", 0, n=2, reward=1.0)
        agg.add_trajectory(first)
        dup = make_traj("a", 0, n=2, reward=99.0)
        self.assertIsNone(agg.add_trajectory(dup))
        group = agg.add_trajectory(make_traj("a", 1, n=2))
        self.assertEqual(group.trajectories[0].reward, 1.0)  # first arrival won
        self.assertEqual(agg.total_duplicates_dropped, 1)

    def test_group_size_mismatch_rejected(self):
        agg = GroupAggregator()
        agg.add_trajectory(make_traj("a", 0, n=4))
        with self.assertRaises(ValueError):
            agg.add_trajectory(make_traj("a", 1, n=8))

    def test_out_of_range_index_rejected(self):
        agg = GroupAggregator()
        with self.assertRaises(ValueError):
            agg.add_trajectory(make_traj("a", 7, n=4))
        with self.assertRaises(ValueError):
            agg.add_trajectory(make_traj("a", -1, n=4))


class TestEviction(unittest.TestCase):
    def test_failed_trajectory_evicts_group(self):
        agg = GroupAggregator()
        agg.add_trajectory(make_traj("a", 0))
        agg.add_trajectory(make_traj("a", 1))
        failed = TrajectorySample(uid="a", traj_index=2, group_size=4, status=TrajectoryStatus.FAILED, attempts=3)
        self.assertIsNone(agg.add_trajectory(failed))
        self.assertEqual(agg.num_partial_groups, 0)
        self.assertEqual(agg.total_groups_evicted, 1)
        self.assertEqual(agg.total_trajectories_evicted, 3)  # 2 buffered + 1 failed

    def test_failed_first_arrival_still_evicts(self):
        agg = GroupAggregator()
        failed = TrajectorySample(uid="a", traj_index=0, group_size=4, status=TrajectoryStatus.FAILED)
        self.assertIsNone(agg.add_trajectory(failed))
        self.assertEqual(agg.total_groups_evicted, 1)
        self.assertEqual(agg.num_partial_groups, 0)

    def test_late_sibling_after_eviction_is_dropped(self):
        agg = GroupAggregator()
        failed = TrajectorySample(uid="a", traj_index=0, group_size=2, status=TrajectoryStatus.FAILED)
        agg.add_trajectory(failed)
        # protocol violation: sibling arrives after eviction
        self.assertIsNone(agg.add_trajectory(make_traj("a", 1, n=2)))
        self.assertEqual(agg.total_late_dropped, 1)
        self.assertEqual(agg.num_partial_groups, 0)  # did not resurrect the group

    def test_explicit_evict_returns_partial_record(self):
        agg = GroupAggregator()
        agg.add_trajectory(make_traj("a", 0, n=4))
        agg.add_trajectory(make_traj("a", 2, n=4))
        record = agg.evict_group("a", reason="staleness")
        self.assertIsNotNone(record)
        self.assertTrue(record.evicted)
        self.assertEqual(record.group_size, 4)
        self.assertEqual(len(record.trajectories), 2)
        self.assertEqual(agg.num_partial_groups, 0)
        # evicted records must not pass GroupRecord's trainable invariant
        self.assertIsNone(agg.evict_group("a"))

    def test_buffer_limit_drops_oldest_partial(self):
        agg = GroupAggregator(max_buffered_groups=2)
        agg.add_trajectory(make_traj("a", 0, n=4))
        agg.add_trajectory(make_traj("b", 0, n=4))
        agg.add_trajectory(make_traj("c", 0, n=4))  # over the soft limit
        self.assertEqual(agg.num_partial_groups, 2)
        self.assertNotIn("a", agg.partial_uids())  # oldest dropped


class TestGroupRecordStats(unittest.TestCase):
    def test_version_span_and_staleness(self):
        group = GroupRecord(
            uid="a",
            group_size=3,
            trajectories=[
                make_traj("a", 0, version=5),
                make_traj("a", 1, version=7),
                make_traj("a", 2, version=6),
            ],
        )
        self.assertEqual(group.version_span, 2)
        self.assertEqual(group.staleness(current_version=10), 3)  # 10 - max(7)
        self.assertEqual(group.oldest_staleness(current_version=10), 5)  # 10 - min(5)

    def test_trainable_record_must_be_complete(self):
        with self.assertRaises(ValueError):
            GroupRecord(uid="a", group_size=3, trajectories=[make_traj("a", 0)])
        # partial is fine when marked as eviction
        record = GroupRecord(uid="a", group_size=3, trajectories=[make_traj("a", 0)], evicted=True)
        self.assertTrue(record.evicted)

    def test_rewards_ordered_by_traj_index(self):
        group = GroupRecord(
            uid="a",
            group_size=2,
            trajectories=[make_traj("a", 0, reward=3.0), make_traj("a", 1, reward=1.0)],
        )
        self.assertEqual(group.rewards, [3.0, 1.0])


if __name__ == "__main__":
    unittest.main()
