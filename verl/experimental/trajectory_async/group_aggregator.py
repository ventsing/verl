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
"""Trainer-side group aggregation: reassemble prompt groups from
individually-delivered trajectories.

This is the component that makes trajectory-level delivery compatible with
group-based advantage estimators (GRPO/DAPO/RLOO...): trajectories stream in
in completion order — interleaved across prompts — and each prompt's group
becomes *trainable* only once all ``rollout.n`` of its responses arrived.
Completed groups are emitted in completion order, which decouples the
mini-batch composition from the prompt submission order.

Delivery protocol (guaranteed by the rollouter):

* trajectories of a group may arrive in any order, interleaved with other
  groups' trajectories;
* ``FAILED`` sentinels for a group are emitted only after **all** sibling
  tasks of that group have settled, so no trajectory of an evicted group can
  arrive after its eviction (the aggregator still keeps a bounded dead-uid
  guard against protocol violations).

Semantics:

* a group is emitted **exactly once**, with all ``group_size`` trajectories
  ordered by ``traj_index``;
* duplicate arrivals for the same ``(uid, traj_index)`` are tolerated: the
  first wins, later copies are counted and dropped (retry races);
* a terminally-failed trajectory (``status == FAILED``) evicts the group: the
  partial rows are returned for accounting and never trained on;
* the buffer never blocks the producer; memory is bounded by
  ``max_buffered_groups`` (soft limit, exceeded only by in-flight groups).
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from verl.experimental.trajectory_async.types import (
    GroupRecord,
    TrajectorySample,
    TrajectoryStatus,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_DEAD_UIDS = 4096


class GroupAggregator:
    """Buffers individual trajectories keyed by group uid.

    Call :meth:`add_trajectory` for every trajectory pulled from the queue.
    It returns a :class:`GroupRecord` when this arrival completed the group
    (``None`` otherwise). Evictions are returned as partial
    ``GroupRecord(evicted=True)`` for accounting; callers must not train on
    them.
    """

    def __init__(
        self,
        max_buffered_groups: int | None = None,
        max_dead_uids: int = _DEFAULT_MAX_DEAD_UIDS,
    ) -> None:
        # uid -> {traj_index: TrajectorySample}
        self._partial: OrderedDict[str, dict[int, TrajectorySample]] = OrderedDict()
        # uid -> expected group size (recorded from the first arrival)
        self._group_sizes: dict[str, int] = {}
        # uids whose group was dropped; late arrivals are counted and dropped
        self._dead_uids: OrderedDict[str, None] = OrderedDict()
        self.max_buffered_groups = max_buffered_groups
        self.max_dead_uids = max_dead_uids

        # counters
        self.total_added = 0
        self.total_duplicates_dropped = 0
        self.total_late_dropped = 0
        self.total_groups_completed = 0
        self.total_groups_evicted = 0
        self.total_trajectories_evicted = 0

    # ------------------------------------------------------------------ core

    def add_trajectory(self, traj: TrajectorySample) -> GroupRecord | None:
        """Add one trajectory; return the completed group if this arrival
        finishes it, else ``None``.

        A ``FAILED`` trajectory evicts its group immediately (partial
        eviction record available via the return value of :meth:`evict_group`
        or counted internally — see module docstring for the protocol).
        """
        if traj.uid in self._dead_uids:
            # Sibling arrived after the group was evicted — protocol
            # violation, but we fail soft: count and drop.
            self.total_late_dropped += 1
            logger.warning("Late trajectory %s#%d after group eviction; dropped", traj.uid, traj.traj_index)
            return None

        if traj.status == TrajectoryStatus.FAILED:
            return self._evict_on_failure(traj)

        self.total_added += 1

        expected_size = self._group_sizes.get(traj.uid)
        if expected_size is None:
            self._group_sizes[traj.uid] = traj.group_size
            expected_size = traj.group_size
        elif expected_size != traj.group_size:
            raise ValueError(
                f"Group {traj.uid} declared group_size={expected_size} on first "
                f"arrival but trajectory {traj.traj_index} claims {traj.group_size}"
            )

        if traj.traj_index < 0 or traj.traj_index >= expected_size:
            raise ValueError(
                f"Group {traj.uid} received traj_index={traj.traj_index} "
                f"outside [0, {expected_size})"
            )

        rows = self._partial.setdefault(traj.uid, {})
        if traj.traj_index in rows:
            # Retry race: the same slot already arrived. Keep the first
            # arrival, count the duplicate.
            self.total_duplicates_dropped += 1
            logger.warning(
                "Group %s traj_index=%d arrived twice; dropping duplicate (attempts=%d)",
                traj.uid,
                traj.traj_index,
                traj.attempts,
            )
            return None

        rows[traj.traj_index] = traj

        if len(rows) == expected_size:
            del self._partial[traj.uid]
            del self._group_sizes[traj.uid]
            self.total_groups_completed += 1
            return GroupRecord(
                uid=traj.uid,
                group_size=expected_size,
                trajectories=[rows[i] for i in range(expected_size)],
                complete_time=max(t.complete_time for t in rows.values()),
            )

        self._enforce_buffer_limit()
        return None

    def evict_group(self, uid: str, reason: str = "manual") -> GroupRecord | None:
        """Explicitly drop a partial group (e.g. staleness timeout). Returns
        the eviction record if the group was buffered, else ``None``."""
        if uid not in self._partial:
            return None
        return self._evict(uid, reason=reason)

    # ------------------------------------------------------------- inspection

    @property
    def num_partial_groups(self) -> int:
        return len(self._partial)

    @property
    def num_buffered_trajectories(self) -> int:
        return sum(len(rows) for rows in self._partial.values())

    def partial_uids(self) -> list[str]:
        return list(self._partial.keys())

    def snapshot(self) -> dict[str, int]:
        """Point-in-time counters for metrics/logging."""
        return {
            "aggregator/partial_groups": self.num_partial_groups,
            "aggregator/buffered_trajectories": self.num_buffered_trajectories,
            "aggregator/total_added": self.total_added,
            "aggregator/groups_completed": self.total_groups_completed,
            "aggregator/groups_evicted": self.total_groups_evicted,
            "aggregator/trajectories_evicted": self.total_trajectories_evicted,
            "aggregator/duplicates_dropped": self.total_duplicates_dropped,
            "aggregator/late_dropped": self.total_late_dropped,
        }

    # ---------------------------------------------------------------- private

    def _evict_on_failure(self, failed: TrajectorySample) -> None:
        """Evict the group owning a terminally-failed trajectory.

        Under the delivery protocol the failed sentinel arrives after all
        sibling tasks settled, so whatever is buffered for the uid is all
        there will ever be. The eviction is accounted internally (the partial
        rows are not returned to the caller — they are never trainable).
        """
        uid = failed.uid
        record = self._evict(uid, reason=f"trajectory #{failed.traj_index} failed after {failed.attempts} attempts")
        if record is not None:
            # include the failed trajectory itself in the accounting record
            record.trajectories.append(failed)
            self.total_trajectories_evicted += 1
        else:
            # No buffered rows: the failed trajectory was the group's only
            # arrival (or the group was already dead).
            if uid not in self._dead_uids:
                self.total_groups_evicted += 1
                self.total_trajectories_evicted += 1
        self._mark_dead(uid)

    def _evict(self, uid: str, reason: str) -> GroupRecord | None:
        rows = self._partial.pop(uid, None)
        expected_size = self._group_sizes.pop(uid, None)
        if rows is None:
            return None
        if expected_size is None:
            expected_size = next(iter(rows.values())).group_size
        trajectories = [rows[i] for i in sorted(rows.keys())]
        self.total_groups_evicted += 1
        self.total_trajectories_evicted += len(trajectories)
        self._mark_dead(uid)
        record = GroupRecord(
            uid=uid,
            group_size=expected_size,
            trajectories=trajectories,
            evicted=True,
        )
        logger.warning(
            "Evicted partial group %s (%d/%d trajectories) — reason: %s",
            uid,
            len(trajectories),
            expected_size,
            reason,
        )
        return record

    def _mark_dead(self, uid: str) -> None:
        self._dead_uids[uid] = None
        while len(self._dead_uids) > self.max_dead_uids:
            self._dead_uids.popitem(last=False)

    def _enforce_buffer_limit(self) -> None:
        if self.max_buffered_groups is None or len(self._partial) <= self.max_buffered_groups:
            return
        # Soft limit: drop the oldest partial group. In steady state this
        # only triggers when prompts systematically fail to complete, which
        # is a rollout-side bug, not a scheduling outcome.
        overflow = len(self._partial) - self.max_buffered_groups
        for _ in range(overflow):
            uid, _ = self._partial.popitem(last=False)
            self._group_sizes.pop(uid, None)
            self.total_groups_evicted += 1
            self._mark_dead(uid)
            logger.warning("Aggregator buffer over limit, dropped oldest partial group %s", uid)
