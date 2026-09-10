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
"""Core data types for trajectory-level asynchronous RL.

This module is deliberately dependency-free (stdlib only) so that the
trajectory-level data plane can be unit-tested and demoed on a bare CPU
machine without ray/torch/numpy. The verl-specific glue (DataProto rows,
Ray actors) attaches to :class:`TrajectorySample` through the opaque
``payload`` field; see ``README.md`` for the real-engine wiring guide.

Two granularities matter here:

* **sample / prompt-group level** — what ``verl.experimental.fully_async_policy``
  already streams: one prompt expanded to ``rollout.n`` rows, delivered as a
  unit once *all* ``n`` responses finished. The slowest response of the group
  gates the whole unit.
* **trajectory level** (this package) — a *single response* of a single prompt.
  Each of the ``n`` responses is an independent request; the moment it
  finishes it is pushed downstream (reward scoring included), without waiting
  for its siblings. Groups are re-assembled on the trainer side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrajectoryStatus(Enum):
    """Lifecycle of a single trajectory (one response of one prompt)."""

    COMPLETED = "completed"
    # The trajectory exhausted its retry budget; the owning group must be
    # evicted (GRPO needs the full group to compute advantages).
    FAILED = "failed"


def _now() -> float:
    """Monotonic clock, so timings survive wall-clock adjustments."""
    return time.monotonic()


@dataclass
class TrajectorySample:
    """One completed (or terminally failed) rollout trajectory.

    This is the *minimum transmission unit* of the trajectory-level data
    plane — the analogue of ``RolloutSample`` in ``fully_async_policy``,
    shrunk from a prompt group to a single response.

    Attributes:
        uid: group (prompt) identifier shared by all ``group_size`` siblings.
        traj_index: position of this response inside its group, ``0..n-1``.
        group_size: the group's ``rollout.n``.
        payload: opaque per-row data. In mock mode a plain dict; in real mode
            a 1-row ``DataProto`` (see README wiring guide).
        model_version: policy version the request *started* under. A real
            engine may sync weights mid-generation, producing intra-trajectory
            version mixing; that span is the engine's to report (payload-
            dependent), while this field is the submission-time version used
            for staleness accounting.
        reward: per-response reward, filled by the reward stage before the
            trajectory reaches the aggregator (mirrors agent-loop scoring).
        gen_time_s: engine-side generation wall time of the final attempt.
        num_tokens: number of generated tokens in the final attempt.
        submit_time: monotonic time the request was submitted to the engine
            (first attempt).
        complete_time: monotonic time the final attempt returned.
        attempts: number of engine attempts consumed (1 + retries).
        status: completion status.
    """

    uid: str
    traj_index: int
    group_size: int
    payload: Any = None
    model_version: int = 0
    reward: float | None = None
    gen_time_s: float = 0.0
    num_tokens: int = 0
    submit_time: float = field(default_factory=_now)
    complete_time: float = field(default_factory=_now)
    attempts: int = 1
    status: TrajectoryStatus = TrajectoryStatus.COMPLETED

    @property
    def latency_s(self) -> float:
        """End-to-end latency from submission to (final) completion."""
        return self.complete_time - self.submit_time


@dataclass
class GroupRecord:
    """A prompt group re-assembled from ``group_size`` trajectory samples.

    Produced by :class:`~verl.experimental.trajectory_async.group_aggregator.GroupAggregator`
    once all ``n`` responses of a prompt arrived. Semantically equivalent to
    one ``RolloutSample`` of ``fully_async_policy`` (one prompt × n rows),
    but assembled on the trainer side instead of the rollout side.

    Attributes:
        uid: group (prompt) identifier.
        group_size: number of trajectories in the group (``rollout.n``).
        trajectories: the ``n`` trajectories, ordered by ``traj_index``.
        complete_time: monotonic time the last trajectory arrived at the
            aggregator. This is the earliest moment the group is trainable.
        evicted: True if the group was dropped because one trajectory
            terminally failed (only set on eviction records, which are
            reported for accounting but never trained on).
        partial: True if the group is TRAINABLE despite missing
            trajectories — the long-tail mitigation (one failed or
            deadline-expired member must not poison its n−1 healthy
            siblings): the record carries only the survivors, with
            ``group_size`` keeping the ORIGINAL rollout.n so staleness/
            version_span stay meaningful. Requires
            ``min_group_survivors`` tolerance on the aggregator (else
            eviction, the strict default).
    """

    uid: str
    group_size: int
    trajectories: list[TrajectorySample] = field(default_factory=list)
    complete_time: float = field(default_factory=_now)
    evicted: bool = False
    partial: bool = False

    @property
    def survivors(self) -> int:
        """Trajectories actually present (== group_size unless partial)."""
        return len(self.trajectories)

    def __post_init__(self) -> None:
        if len(self.trajectories) > self.group_size:
            raise ValueError(
                f"GroupRecord({self.uid}) built with {len(self.trajectories)} "
                f"trajectories but group_size={self.group_size}"
            )
        if not self.evicted and not self.partial and len(self.trajectories) != self.group_size:
            raise ValueError(
                f"GroupRecord({self.uid}) is marked trainable but has "
                f"{len(self.trajectories)}/{self.group_size} trajectories; only "
                f"eviction or partial-survivor records may be incomplete"
            )

    # ------------------------------------------------------------------ stats

    @property
    def rewards(self) -> list[float]:
        """Per-trajectory rewards in ``traj_index`` order."""
        return [t.reward for t in self.trajectories]

    @property
    def model_versions(self) -> list[int]:
        return [t.model_version for t in self.trajectories]

    @property
    def version_span(self) -> int:
        """Max minus min model version within the group.

        Zero means the whole group was generated under one policy version —
        the property that group-level delivery guarantees for free and that
        trajectory-level delivery trades away for earlier streaming.
        """
        versions = self.model_versions
        return max(versions) - min(versions)

    @property
    def max_staleness(self) -> int:
        """Gap between the newest version in the group and the oldest."""
        return self.version_span

    def staleness(self, current_version: int) -> int:
        """Gap between the current trainer version and the newest version
        used by this group (>= 0 once versions only move forward)."""
        return max(0, current_version - max(self.model_versions))

    def oldest_staleness(self, current_version: int) -> int:
        """Gap between the current trainer version and the oldest version
        used by this group — the worst-case off-policy distance."""
        return max(0, current_version - min(self.model_versions))

    @property
    def train_ready_latency_s(self) -> float:
        """From the group's first submission to last-arrival: how long the
        prompt occupied the pipeline before becoming trainable."""
        first_submit = min(t.submit_time for t in self.trajectories)
        return self.complete_time - first_submit

    @property
    def total_attempts(self) -> int:
        return sum(t.attempts for t in self.trajectories)
