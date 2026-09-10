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
"""Unit tests for mini-batch formation and the GRPO advantage helper."""

from __future__ import annotations

import math
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.group_collector import grpo_group_advantages
from verl.experimental.trajectory_async.types import GroupRecord, TrajectorySample


def make_group(uid: str, n: int = 2) -> GroupRecord:
    return GroupRecord(
        uid=uid,
        group_size=n,
        trajectories=[
            TrajectorySample(uid=uid, traj_index=i, group_size=n, reward=float(i)) for i in range(n)
        ],
    )


class TestMiniBatcher(unittest.TestCase):
    def test_emits_batch_exactly_when_full(self):
        mb = MiniBatcher(mini_batch_groups=3)
        self.assertIsNone(mb.add_group(make_group("a")))
        self.assertIsNone(mb.add_group(make_group("b")))
        batch = mb.add_group(make_group("c"))
        self.assertEqual([g.uid for g in batch], ["a", "b", "c"])
        self.assertEqual(mb.total_mini_batches, 1)
        self.assertEqual(mb.num_pending, 0)

    def test_batches_flow_in_completion_order(self):
        mb = MiniBatcher(mini_batch_groups=2)
        mb.add_group(make_group("a"))
        b1 = mb.add_group(make_group("b"))
        mb.add_group(make_group("c"))
        b2 = mb.add_group(make_group("d"))
        self.assertEqual([g.uid for g in b1], ["a", "b"])
        self.assertEqual([g.uid for g in b2], ["c", "d"])

    def test_evicted_records_do_not_consume_slots(self):
        mb = MiniBatcher(mini_batch_groups=2)
        evicted = GroupRecord(
            uid="dead",
            group_size=2,
            trajectories=[TrajectorySample(uid="dead", traj_index=0, group_size=2)],
            evicted=True,
        )
        self.assertIsNone(mb.add_group(evicted))
        self.assertEqual(mb.total_groups_seen, 0)
        self.assertIsNone(mb.add_group(make_group("a")))
        batch = mb.add_group(make_group("b"))
        self.assertEqual(len(batch), 2)

    def test_drain_returns_leftover(self):
        mb = MiniBatcher(mini_batch_groups=2)
        mb.add_group(make_group("a"))
        leftover = mb.drain()
        self.assertEqual([g.uid for g in leftover], ["a"])
        self.assertEqual(mb.drain(), [])

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            MiniBatcher(mini_batch_groups=0)


class TestGrpoAdvantage(unittest.TestCase):
    def test_zero_mean_unit_std(self):
        adv = grpo_group_advantages([1.0, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(sum(adv) / len(adv), 0.0, places=12)
        std = math.sqrt(sum(a * a for a in adv) / len(adv))
        self.assertAlmostEqual(std, 1.0, places=12)

    def test_degenerate_group_gets_zero_advantage(self):
        self.assertEqual(grpo_group_advantages([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_empty(self):
        self.assertEqual(grpo_group_advantages([]), [])


if __name__ == "__main__":
    unittest.main()
