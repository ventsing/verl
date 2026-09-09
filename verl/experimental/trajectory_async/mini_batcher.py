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
"""Mini-batch formation from completed groups.

Mirrors ``fully_async_policy``'s consumption rule — the trainer trains once
it has collected ``ppo_mini_batch_size`` *samples* (prompt groups) — with one
decoupling: which groups end up in a mini-batch is decided by **completion
order** on the trainer side, not by submission order on the rollout side.
"""

from __future__ import annotations

from collections import deque

from verl.experimental.trajectory_async.types import GroupRecord


class MiniBatcher:
    """Collects complete groups and emits mini-batches of exactly
    ``mini_batch_groups`` groups (``ppo_mini_batch_size`` in verl terms,
    where one "sample" is one prompt group of ``rollout.n`` rows)."""

    def __init__(self, mini_batch_groups: int) -> None:
        if mini_batch_groups < 1:
            raise ValueError(f"mini_batch_groups must be >= 1, got {mini_batch_groups}")
        self.mini_batch_groups = mini_batch_groups
        self._pending: deque[GroupRecord] = deque()

        self.total_groups_seen = 0
        self.total_mini_batches = 0

    def add_group(self, group: GroupRecord) -> list[GroupRecord] | None:
        """Add one completed group; return a full mini-batch (list of groups
        in completion order) when the batch is complete, else ``None``."""
        if group.evicted:
            # Eviction records are accounting-only; they never consume a
            # mini-batch slot.
            return None
        self._pending.append(group)
        self.total_groups_seen += 1
        if len(self._pending) >= self.mini_batch_groups:
            batch = [self._pending.popleft() for _ in range(self.mini_batch_groups)]
            self.total_mini_batches += 1
            return batch
        return None

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    def drain(self) -> list[GroupRecord]:
        """Flush any pending groups (end-of-training accounting; normally the
        caller ensures the totals divide evenly)."""
        leftover = list(self._pending)
        self._pending.clear()
        return leftover

    def snapshot(self) -> dict[str, int]:
        return {
            "mini_batcher/pending_groups": self.num_pending,
            "mini_batcher/total_groups": self.total_groups_seen,
            "mini_batcher/total_mini_batches": self.total_mini_batches,
        }
