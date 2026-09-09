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
"""Tests for TrajectoryBatchCollector — the consumption core of the REAL
(separate-deployment) trainer, kept stdlib-only so it runs anywhere.

The collector must:
* accept BOTH producer granularities (group-level samples and
  trajectory-level single rows) through one code path;
* only ever emit mini-batches of COMPLETE groups (GRPO semantics);
* account every group exactly once (trained / evicted /
  dropped-stale / leftover / incomplete);
* respect the staleness bound without shrinking a batch.
"""

import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.group_collector import (  # noqa: E402
    TrajectoryBatchCollector,
    row_from_sample_batch,
)


def _group_rows(uid: str, n: int, version: int = 0, reward: float = 1.0):
    return [
        {
            "uid": uid,
            "traj_index": i,
            "group_size": n,
            "model_version": version,
            "reward": reward,
            "num_tokens": 10,
            "failed": False,
            "payload": f"{uid}#{i}",
        }
        for i in range(n)
    ]


class TestGroupLevelProducer(unittest.TestCase):
    def test_group_samples_form_exact_mini_batches(self):
        c = TrajectoryBatchCollector(mini_batch_groups=2, group_size=4)
        for g in range(5):
            c.add_sample(f"uid-{g}", _group_rows(f"uid-{g}", 4))
        batch = c.take_mini_batch(current_version=0)
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch), 2)
        for group in batch:
            self.assertEqual(group.group_size, 4)
            self.assertEqual([t.traj_index for t in group.trajectories], [0, 1, 2, 3])
        # one more batch from two of the remaining three groups
        self.assertEqual(c.take_mini_batch(0) is not None, True)
        self.assertIsNone(c.take_mini_batch(0))  # only 1 group pending
        c.finalize()
        snap = c.snapshot()
        self.assertEqual(snap["trajectory_async/groups_trained"], 4)
        self.assertEqual(snap["trajectory_async/groups_leftover"], 1)


class TestTrajectoryLevelProducer(unittest.TestCase):
    def test_rows_arriving_separately_reassemble_into_groups(self):
        c = TrajectoryBatchCollector(mini_batch_groups=2, group_size=3)
        # interleave rows of two groups, out of order — like two async
        # rollouts streaming single responses
        order = [(0, 2), (1, 0), (0, 0), (1, 2), (0, 1), (1, 1)]
        for uid, idx in order:
            c.add_trajectory(
                uid=f"uid-{uid}", traj_index=idx, group_size=3, model_version=2, payload=f"p{uid}{idx}"
            )
        batch = c.take_mini_batch(current_version=2)
        self.assertIsNotNone(batch)
        self.assertEqual({g.uid for g in batch}, {"uid-0", "uid-1"})
        for group in batch:
            self.assertEqual([t.payload for t in group.trajectories], [f"p{group.uid[-1]}{i}" for i in range(3)])
        self.assertEqual(c.stats.groups_trained, 2)

    def test_mixed_granularity_streams_merge(self):
        """A group-level producer and a trajectory-level producer feeding
        the same trainer — the collector is granularity-agnostic."""
        c = TrajectoryBatchCollector(mini_batch_groups=2, group_size=2)
        c.add_sample("uid-a", _group_rows("uid-a", 2))
        c.add_trajectory("uid-b", traj_index=1, group_size=2, payload="b1")
        self.assertIsNone(c.take_mini_batch(0))  # uid-b still missing a row
        c.add_trajectory("uid-b", traj_index=0, group_size=2, payload="b0")
        batch = c.take_mini_batch(0)
        self.assertEqual(len(batch), 2)


class TestFailureAndStaleness(unittest.TestCase):
    def test_failed_row_evicts_group_and_it_is_never_trained(self):
        c = TrajectoryBatchCollector(mini_batch_groups=1, group_size=3)
        c.add_trajectory("uid-a", traj_index=0, group_size=3, payload="a0")
        c.add_trajectory("uid-a", traj_index=1, group_size=3, failed=True)
        c.add_trajectory("uid-a", traj_index=2, group_size=3, payload="a2")
        self.assertEqual(c.stats.groups_evicted, 1)
        # wasted = the completed sibling row (a0) + the failed attempt (a1)
        self.assertEqual(c.stats.trajectories_wasted_evicted, 2)
        self.assertIsNone(c.take_mini_batch(0))
        # reconciliation: no group trained, one evicted
        c.finalize()
        self.assertEqual(c.stats.groups_trained, 0)
        self.assertEqual(c.stats.groups_evicted, 1)

    def test_staleness_drop_refuses_old_groups_keeps_batch_exact(self):
        c = TrajectoryBatchCollector(mini_batch_groups=2, group_size=2, max_staleness_drop=1)
        # group-0 generated under version 0, group-1 under version 5;
        # consuming at version 6: group-0 is 6 versions stale (refused),
        # group-1 is 1 stale (fine)
        c.add_sample("old", _group_rows("old", 2, version=0))
        c.add_sample("new", _group_rows("new", 2, version=5))
        batch = c.take_mini_batch(current_version=6)
        self.assertIsNone(batch)  # only ONE fresh group available
        self.assertEqual(c.stats.groups_dropped_stale, 1)
        # the fresh group stays pending; a second fresh group completes a batch
        c.add_sample("new2", _group_rows("new2", 2, version=6))
        batch = c.take_mini_batch(current_version=6)
        self.assertEqual([g.uid for g in batch], ["new", "new2"])
        self.assertEqual(c.stats.groups_dropped_stale, 1)  # old refused exactly once

    def test_reconciliation_identity_holds(self):
        """trained + evicted + dropped_stale + leftover + incomplete ==
        groups ever started."""
        c = TrajectoryBatchCollector(mini_batch_groups=2, group_size=2, max_staleness_drop=0)
        # g0, g1 generated at v0, consumed at v0 -> trained while fresh
        c.add_sample("g0", _group_rows("g0", 2, version=0))
        c.add_sample("g1", _group_rows("g1", 2, version=0))
        batch = c.take_mini_batch(current_version=0)
        self.assertEqual([g.uid for g in batch], ["g0", "g1"])
        # then the trainer advances to v1
        c.add_sample("g2", _group_rows("g2", 2, version=0))  # stale at v1 -> refused
        c.add_sample("g3", _group_rows("g3", 2, version=1))  # fresh, but never enough
        c.add_trajectory("g4", traj_index=0, group_size=2)  # incomplete
        c.add_trajectory("g5", traj_index=0, group_size=2, failed=True)  # evicted
        self.assertIsNone(c.take_mini_batch(current_version=1))
        c.finalize()
        s = c.stats
        self.assertEqual(s.groups_trained, 2)
        self.assertEqual(s.groups_dropped_stale, 1)
        self.assertEqual(s.groups_leftover, 1)
        self.assertEqual(s.groups_evicted, 1)
        self.assertEqual(s.groups_incomplete, 1)
        total = s.groups_trained + s.groups_dropped_stale + s.groups_leftover + s.groups_evicted + s.groups_incomplete
        self.assertEqual(total, 6)


class _FakeDataProtoRow:
    def __init__(self, tag):
        self.tag = tag


class _FakeDataProto:
    """Minimal DataProto lookalike: len(), non_tensor_batch, union(i)."""

    def __init__(self, n, ntb):
        self.n = n
        self.non_tensor_batch = ntb

    def __len__(self):
        return self.n

    def union(self, position):
        return _FakeDataProtoRow((self.non_tensor_batch["uid"][position], position))


class TestRowAdapter(unittest.TestCase):
    def test_reads_trajectory_fields_and_slices_payload(self):
        ntb = {
            "uid": ["uid_x", "uid_x"],
            "traj_index": [1, 0],
            "model_version": [3, 2],
            "rollout_failed": [False, False],
        }
        batch = _FakeDataProto(2, ntb)
        row = row_from_sample_batch("fallback", batch, 0)
        self.assertEqual(row["uid"], "uid_x")
        self.assertEqual(row["traj_index"], 1)
        self.assertEqual(row["model_version"], 3)
        self.assertFalse(row["failed"])
        self.assertEqual(row["payload"].tag, ("uid_x", 0))

    def test_falls_back_to_positional_split_without_fields(self):
        batch = _FakeDataProto(3, {"uid": ["uid_y"] * 3})
        row = row_from_sample_batch("uid_y", batch, 2)
        self.assertEqual(row["traj_index"], 2)
        self.assertEqual(row["group_size"], 3)
        self.assertEqual(row["model_version"], 0)
        self.assertEqual(row["payload"].tag, ("uid_y", 2))


if __name__ == "__main__":
    unittest.main()
