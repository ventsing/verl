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
"""Trainer-side batch collection for the real (separate-deployment) trainer.

``TrajectoryAsyncTrainer`` (``async_trainer.py``) consumes rollout messages
from the fully-async message queue and must form mini-batches of *complete
GRPO groups*. This module is that consumption logic, factored out so it is
stdlib-only and unit-testable on a bare CPU machine — the same contract the
mock trainer simulates.

It accepts BOTH producer granularities without configuration:

* **group-level** — the stock ``fully_async_policy`` producer pushes one
  ``RolloutSample`` per prompt with all ``rollout.n`` response rows
  (``uid`` shared). The collector splits it into per-trajectory rows and
  re-aggregates them through :class:`GroupAggregator`, adding per-group
  staleness / version-span accounting on top of the stock behaviour.
* **trajectory-level** — a trajectory-mode producer pushes each response
  as its own message. Rows carry ``traj_index`` (and ideally
  ``model_version``); the aggregator re-assembles the group on the
  trainer side — the whole point of trajectory-level delivery.

A failed row (``rollout_failed`` / terminally failed trajectory) evicts
its group through the aggregator: eviction records are accounted, never
trained. Complete groups that age beyond ``max_staleness_drop`` versions
are refused at mini-batch formation (freshness control).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.types import TrajectorySample, TrajectoryStatus

logger = logging.getLogger(__name__)


@dataclass
class CollectorStats:
    samples_consumed: int = 0
    rows_consumed: int = 0
    groups_trained: int = 0
    groups_evicted: int = 0
    groups_partial: int = 0
    rows_recovered_partial: int = 0
    rows_wasted_failed: int = 0
    groups_dropped_stale: int = 0
    groups_leftover: int = 0  # complete groups flushed at end-of-stream
    groups_incomplete: int = 0  # partial groups still buffered at the end
    trajectories_wasted_evicted: int = 0
    staleness_history: list[int] = field(default_factory=list)
    oldest_staleness_history: list[int] = field(default_factory=list)
    version_span_history: list[int] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        def _summary(values: list[int]) -> dict[str, float]:
            if not values:
                return {}
            return {"mean": sum(values) / len(values), "max": max(values), "min": min(values)}

        return {
            "trajectory_async/samples_consumed": self.samples_consumed,
            "trajectory_async/rows_consumed": self.rows_consumed,
            "trajectory_async/groups_trained": self.groups_trained,
            "trajectory_async/groups_evicted": self.groups_evicted,
            "trajectory_async/groups_partial": self.groups_partial,
            "trajectory_async/rows_recovered_partial": self.rows_recovered_partial,
            "trajectory_async/rows_wasted_failed": self.rows_wasted_failed,
            "trajectory_async/groups_dropped_stale": self.groups_dropped_stale,
            "trajectory_async/groups_leftover": self.groups_leftover,
            "trajectory_async/groups_incomplete": self.groups_incomplete,
            "trajectory_async/trajectories_wasted_evicted": self.trajectories_wasted_evicted,
            "trajectory_async/staleness": _summary(self.staleness_history),
            "trajectory_async/oldest_staleness": _summary(self.oldest_staleness_history),
            "trajectory_async/version_span": _summary(self.version_span_history),
        }


class TrajectoryBatchCollector:
    """Group-aware consumption: messages in, complete-group mini-batches out.

    Args:
        mini_batch_groups: groups per mini-batch (``ppo_mini_batch_size``,
            where one "sample" is one prompt group of ``rollout.n`` rows).
        max_staleness_drop: refuse groups whose ``oldest_staleness`` against
            the consuming version exceeds this bound (None = train everything).
        group_size: default ``rollout.n`` for rows that do not carry one.
        on_group_complete: optional callback ``(GroupRecord) -> None`` fired
            when a group completes (used for logging/timeline events).
    """

    def __init__(
        self,
        mini_batch_groups: int,
        max_staleness_drop: int | None = None,
        group_size: int | None = None,
        on_group_complete: Callable | None = None,
        **aggregator_kwargs,
    ) -> None:
        if mini_batch_groups < 1:
            raise ValueError(f"mini_batch_groups must be >= 1, got {mini_batch_groups}")
        self.mini_batch_groups = mini_batch_groups
        self.max_staleness_drop = max_staleness_drop
        self.default_group_size = group_size
        self.on_group_complete = on_group_complete
        self.aggregator = GroupAggregator(**aggregator_kwargs)
        # complete groups awaiting a mini-batch. Kept here (not in
        # MiniBatcher) because emission is staleness-aware: a batch is
        # emitted only when mini_batch_groups FRESH groups are available,
        # so ppo_mini_batch_size stays exact under freshness control.
        self._pending: deque = deque()
        self.stats = CollectorStats()

    # ------------------------------------------------------------- intake

    def add_sample(self, uid: str, rows: list[dict[str, Any]]) -> None:
        """Group-level producer message: one prompt's ``n`` response rows.

        Each row dict may carry ``traj_index``, ``group_size``,
        ``model_version``, ``reward``, ``num_tokens``, ``failed`` and an
        opaque ``payload``; missing fields are defaulted (``traj_index``
        falls back to the row position inside the sample).
        """
        self.stats.samples_consumed += 1
        n = rows[0].get("group_size") or len(rows) or self.default_group_size
        for pos, row in enumerate(rows):
            self.add_trajectory(
                uid=uid,
                traj_index=row.get("traj_index", pos),
                group_size=row.get("group_size") or n,
                attempts=int(row.get("attempts") or 1),
                model_version=row.get("model_version", 0),
                reward=row.get("reward"),
                num_tokens=row.get("num_tokens", 0),
                failed=row.get("failed", False),
                payload=row.get("payload"),
            )

    def add_trajectory(
        self,
        uid: str,
        traj_index: int,
        group_size: int | None = None,
        model_version: int = 0,
        reward: float | None = None,
        num_tokens: int = 0,
        failed: bool = False,
        attempts: int = 1,
        payload: Any = None,
    ) -> None:
        """Trajectory-level producer message: one response row."""
        self.stats.rows_consumed += 1
        n = group_size or self.default_group_size
        if n is None:
            raise ValueError(
                "group_size unknown: pass group_size (or construct with "
                "group_size=rollout.n) so the aggregator knows when a group is complete"
            )
        traj = TrajectorySample(
            uid=uid,
            traj_index=traj_index,
            group_size=n,
            payload=payload,
            model_version=model_version,
            reward=reward,
            num_tokens=num_tokens,
            attempts=attempts,
            status=TrajectoryStatus.FAILED if failed else TrajectoryStatus.COMPLETED,
        )
        evicted_before = self.aggregator.total_groups_evicted
        wasted_before = self.aggregator.total_trajectories_evicted
        group = self.aggregator.add_trajectory(traj)
        # the aggregator accounts waste internally (evicted groups AND the
        # failed rows of partially-delivered groups); a FAILED row may still
        # return a PARTIAL survivor record when the long-tail mitigation is
        # enabled — surface the evicted delta in the collector's metric family
        self.stats.groups_evicted += self.aggregator.total_groups_evicted - evicted_before
        self.stats.trajectories_wasted_evicted += self.aggregator.total_trajectories_evicted - wasted_before
        if group is not None:
            self._on_group_record(group)

    # ------------------------------------------------------------- output

    def take_mini_batch(self, current_version: int) -> list | None:
        """Pop one mini-batch of exactly ``mini_batch_groups`` complete,
        fresh-enough groups — or ``None`` if not enough fresh groups are
        pending yet. Stale groups are refused permanently at first
        examination (counted in ``groups_dropped_stale``, never re-queued,
        never trained)."""
        fresh: list = []
        stale: list = []
        while self._pending and len(fresh) < self.mini_batch_groups:
            group = self._pending.popleft()
            if self.max_staleness_drop is not None and group.oldest_staleness(current_version) > self.max_staleness_drop:
                stale.append(group)
            else:
                fresh.append(group)
        self.stats.groups_dropped_stale += len(stale)
        for group in stale:
            logger.info(
                "Dropped stale group %s (oldest version gap %d > %d)",
                group.uid,
                group.oldest_staleness(current_version),
                self.max_staleness_drop,
            )
        if len(fresh) < self.mini_batch_groups:
            # not enough fresh groups: the survivors return to pending
            # (FIFO order preserved). Stale ones are dropped for good —
            # they will not get fresher by waiting.
            for group in reversed(fresh):
                self._pending.appendleft(group)
            return None
        self.stats.groups_trained += len(fresh)
        for group in fresh:
            self.stats.staleness_history.append(group.staleness(current_version))
            self.stats.oldest_staleness_history.append(group.oldest_staleness(current_version))
            self.stats.version_span_history.append(group.version_span)
        return fresh

    @property
    def pending_groups(self) -> int:
        return len(self._pending)

    def finalize(self) -> dict[str, int]:
        """End-of-stream accounting: leftover complete groups and partials.

        Leftover complete groups are NOT trained (they never filled a
        mini-batch); partial groups are simply what never assembled. Both
        are counted so the reconciliation identity holds:
        trained + evicted + dropped_stale + leftover + incomplete == groups
        ever started.
        """
        self.stats.groups_leftover += len(self._pending)
        self._pending.clear()
        self.stats.groups_incomplete += self.aggregator.num_partial_groups
        return {"leftover": self.stats.groups_leftover, "incomplete": self.stats.groups_incomplete}

    def snapshot(self) -> dict[str, Any]:
        out = self.stats.snapshot()
        out.update(self.aggregator.snapshot())
        out["trajectory_async/pending_groups"] = len(self._pending)
        return out

    # ------------------------------------------------------------- private

    def _on_group_record(self, group) -> None:
        if group.evicted:
            # manual evict_group() records — accounted, never trained
            self.stats.groups_evicted += 1
            self.stats.trajectories_wasted_evicted += len(group.trajectories)
            return
        if group.partial:
            # long-tail mitigation: survivors are trainable; the failed or
            # deadline-expired members were already accounted by the
            # aggregator (rows_wasted_failed mirrors the row-level waste)
            self.stats.groups_partial += 1
            self.stats.rows_recovered_partial += len(group.trajectories)
            self.stats.rows_wasted_failed += group.group_size - len(group.trajectories)
        self._pending.append(group)
        if self.on_group_complete is not None:
            self.on_group_complete(group)


def row_from_sample_batch(
    uid: str,
    batch: Any,
    position: int,
    version_getter: Callable[[Any, int], int] | None = None,
) -> dict[str, Any]:
    """Adapt one row of a real ``RolloutSample.full_batch`` (DataProto) to a
    collector row dict — the glue used by ``TrajectoryAsyncTrainer``.

    Reads ``non_tensor_batch`` fields when present (``traj_index``,
    ``model_version``, ``rollout_failed``, ``uid``) and slices the row's
    payload out of the batch for the trainer to re-assemble later. Kept
    here (not in async_trainer) so the mapping is testable without ray.
    """
    ntb = getattr(batch, "non_tensor_batch", None) or {}
    n = len(batch)

    def _scalar(name, default=None):
        if name in ntb:
            value = ntb[name]
            try:
                return value[position].item() if hasattr(value[position], "item") else value[position]
            except (TypeError, IndexError):
                return default
        return default

    traj_index = _scalar("traj_index", position)
    group_size = _scalar("group_size", n)
    model_version = _scalar("model_version", 0)
    attempts = _scalar("attempts", 1)
    failed = bool(_scalar("rollout_failed", False))
    row_uid = _scalar("uid", uid) or uid
    reward = _scalar("reward", None)
    num_tokens = _scalar("num_tokens", 0) or 0

    payload = None
    if hasattr(batch, "union"):
        payload = batch.union(position)  # DataProto row slice
    elif isinstance(batch, list):
        payload = batch[position]

    return {
        "uid": row_uid,
        "traj_index": int(traj_index),
        "group_size": int(group_size) if group_size else None,
        "model_version": int(model_version or 0),
        "attempts": int(attempts or 1),
        "failed": failed,
        "reward": float(reward) if reward is not None else None,
        "num_tokens": int(num_tokens),
        "payload": payload,
        "_time": time.monotonic(),
    }


def grpo_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-normalized advantages (GRPO/DAPO style): zero mean, unit std
    within each prompt group. Degenerate groups (all-equal rewards) get
    zero advantages. The group-preserving contract the collector serves:
    a group is trainable once all its trajectories are present — or, with
    the long-tail mitigation, once its settled survivors are (partial
    records; normalization then runs over the survivors present). This is
    what group reassembly protects (the real training path computes this
    inside the actor workers; this stdlib twin is for metrics and tests)."""
    if not rewards:
        return []
    import statistics

    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]
