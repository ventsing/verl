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
"""Trainer-side consumption pipeline: queue → per-trajectory preprocessing →
group aggregation → mini-batch formation → policy update.

The trainer never blocks the rollouter: it pulls trajectories as they
arrive, pipelines per-trajectory preprocessing (the stage that benefits
most from trajectory-level delivery — e.g. old-log-prob computation can
start on arrived responses while their siblings still generate), and only
the *update* waits for complete groups (GRPO semantics).

One mini-batch = ``mini_batch_groups`` complete prompt groups = one policy
update, mirroring ``fully_async_policy``'s
``ppo_mini_batch_size × require_batches`` consumption rule at group
granularity. After every update the policy version advances by one
(``current_version``), which the demo feeds back to the engine as a
simulated parameter sync — that is what makes staleness measurable.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import GroupRecord, TrajectorySample

logger = logging.getLogger(__name__)

PreprocessFn = Callable[[TrajectorySample], Awaitable[TrajectorySample]]
UpdateFn = Callable[["TrainerBatch"], Awaitable[None]]


@dataclass
class TrainerBatch:
    """One trainable mini-batch: complete groups in completion order."""

    groups: list[GroupRecord]
    policy_version_before: int  # trainer version at update time
    created_at: float  # monotonic time the batch became complete

    @property
    def uid(self) -> str:
        return f"mb-{self.policy_version_before}"


@dataclass
class TrainerConfig:
    mini_batch_groups: int = 8  # groups (prompt samples) per policy update
    update_time_s: float = 2.0  # simulated update_actor duration
    preprocess_concurrency: int = 16  # concurrent per-trajectory preprocessing
    # When to run trainer-side per-trajectory preprocessing:
    #  * "per-trajectory" — start on arrival. Speculative: work on responses
    #    whose siblings (and thus group) may complete much later or never;
    #    early trajectories of slow groups can head-of-line-block fast
    #    groups in the preprocess queue, but total wall time improves
    #    because work overlaps remaining generation.
    #  * "on-group-complete" — start only when the group completed at the
    #    aggregator. No speculation, no cross-group head-of-line blocking,
    #    earliest first mini-batch; identical preprocess timing to
    #    group-level delivery while keeping trajectory-level delivery for
    #    memory/failure granularity.
    preprocess_policy: str = "per-trajectory"
    # drop groups whose *oldest* trajectory is staler than this many
    # versions (None disables staleness-based dropping)
    max_staleness_drop: int | None = None


@dataclass
class TrainerStats:
    trajectories_consumed: int = 0
    groups_trained: int = 0
    groups_dropped_stale: int = 0
    groups_leftover: int = 0  # complete groups that never filled a mini-batch
    mini_batches: int = 0
    updates: int = 0
    update_busy_time_s: float = 0.0
    preprocess_busy_time_s: float = 0.0
    wait_for_queue_s: float = 0.0
    time_to_first_batch_s: float | None = None
    wall_time_s: float = 0.0
    staleness_history: list[int] = field(default_factory=list)
    oldest_staleness_history: list[int] = field(default_factory=list)
    version_span_history: list[int] = field(default_factory=list)
    advantage_abs_mean_history: list[float] = field(default_factory=list)
    # Laminar-style metrics: inherent staleness = trainer version when the
    # trajectory FINISHED generating minus the version that generated it
    # (paper §4.4: typically < 3 in their deployment); throughput = tokens
    # trained per RL iteration (one actor-update interval), the paper's main
    # performance metric
    inherent_staleness_history: list[int] = field(default_factory=list)
    tokens_per_s_history: list[float] = field(default_factory=list)
    total_trained_tokens: int = 0

    def snapshot(self) -> dict[str, Any]:
        def _summary(values: list[int | float]) -> dict[str, float]:
            if not values:
                return {}
            return {"mean": sum(values) / len(values), "max": max(values), "min": min(values)}

        return {
            "trainer/trajectories_consumed": self.trajectories_consumed,
            "trainer/groups_trained": self.groups_trained,
            "trainer/groups_dropped_stale": self.groups_dropped_stale,
            "trainer/groups_leftover": self.groups_leftover,
            "trainer/mini_batches": self.mini_batches,
            "trainer/updates": self.updates,
            "trainer/update_busy_time_s": round(self.update_busy_time_s, 4),
            "trainer/preprocess_busy_time_s": round(self.preprocess_busy_time_s, 4),
            "trainer/wait_for_queue_s": round(self.wait_for_queue_s, 4),
            "trainer/time_to_first_batch_s": (
                None if self.time_to_first_batch_s is None else round(self.time_to_first_batch_s, 4)
            ),
            "trainer/wall_time_s": round(self.wall_time_s, 4),
            "trainer/staleness": _summary(self.staleness_history),
            "trainer/oldest_staleness": _summary(self.oldest_staleness_history),
            "trainer/version_span": _summary(self.version_span_history),
            "trainer/advantage_abs_mean": _summary(self.advantage_abs_mean_history),
            "trainer/inherent_staleness": _summary(self.inherent_staleness_history),
            "trainer/tokens_per_s": _summary(self.tokens_per_s_history),
            "trainer/total_trained_tokens": self.total_trained_tokens,
        }


def grpo_group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Group-normalized advantages (GRPO/DAPO style): zero mean, unit std
    within each prompt group. Degenerate groups (all-equal rewards) get
    zero advantages."""
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]


class TrajectoryTrainer:
    """Consumes trajectories from the queue and trains on complete groups."""

    def __init__(
        self,
        queue: InProcessTrajectoryQueue,
        aggregator: GroupAggregator,
        mini_batcher: MiniBatcher,
        config: TrainerConfig | None = None,
        preprocess_fn: PreprocessFn | None = None,
        update_fn: UpdateFn | None = None,
        on_batch: Callable[[TrainerBatch, dict[str, Any]], None] | None = None,
        on_group: Callable[[GroupRecord], None] | None = None,
    ) -> None:
        if config and config.preprocess_policy not in ("per-trajectory", "on-group-complete"):
            raise ValueError(f"unknown preprocess_policy: {config.preprocess_policy}")
        self.queue = queue
        self.aggregator = aggregator
        self.mini_batcher = mini_batcher
        self.config = config or TrainerConfig()
        self.preprocess_fn = preprocess_fn
        self.update_fn = update_fn
        self.on_batch = on_batch
        self.on_group = on_group
        self.stats = TrainerStats()
        self.current_version = 0
        # (monotonic_time, version_after) at each update completion — used
        # to compute per-trajectory inherent staleness at the trajectory's
        # own finish time (Laminar §4.4), not at training time
        self._version_timeline: list[tuple[float, int]] = []
        self._last_update_completion: float | None = None
        self._preprocess_slots = asyncio.Semaphore(self.config.preprocess_concurrency)
        self._preprocess_tasks: set[asyncio.Task] = set()
        # updates are serialized (one policy update at a time), while
        # per-trajectory preprocessing keeps flowing during an update —
        # exactly the pipeline trajectory-level delivery enables
        self._update_lock = asyncio.Lock()
        self._start_time: float | None = None

    # ------------------------------------------------------------------- run

    async def run(self) -> TrainerStats:
        """Consume the queue until the end-of-stream sentinel.

        Per trajectory: pipeline preprocessing concurrently, then hand to
        the aggregator. Per completed group: enqueue into the mini-batcher.
        Per full mini-batch: drop stale groups, compute GRPO advantages,
        run the policy update, advance the version.
        """
        self._start_time = time.monotonic()
        while True:
            t0 = time.monotonic()
            item = await self.queue.get()
            self.stats.wait_for_queue_s += time.monotonic() - t0
            if item is None:
                break
            self.stats.trajectories_consumed += 1
            self._spawn_preprocess(item)

        # drain in-flight preprocessing before final aggregation state
        if self._preprocess_tasks:
            await asyncio.gather(*self._preprocess_tasks, return_exceptions=True)

        self._finalize()
        self.stats.wall_time_s = time.monotonic() - self._start_time
        return self.stats

    # ------------------------------------------------------------ preprocess

    def _spawn_preprocess(self, traj: TrajectorySample) -> None:
        """Preprocess the trajectory concurrently and feed the aggregator.

        Preprocessing order is decoupled from arrival order; the aggregator
        only cares about (uid, traj_index), so reordering is harmless. This
        is the pipeline stage trajectory-level delivery enables: work starts
        on *arrived* responses instead of waiting for whole groups.
        """
        task = asyncio.create_task(self._preprocess_and_aggregate(traj))
        self._preprocess_tasks.add(task)
        task.add_done_callback(self._preprocess_tasks.discard)

    async def _preprocess_and_aggregate(self, traj: TrajectorySample) -> None:
        if self.config.preprocess_policy == "on-group-complete":
            # aggregate raw, preprocess only complete groups — no speculation
            group = self.aggregator.add_trajectory(traj)
            if group is None or group.evicted:
                return
            await self._preprocess_group(group)
            await self._on_group_and_maybe_train(group)
        else:
            # per-trajectory: preprocess on arrival, aggregate afterwards —
            # preprocessing order is decoupled from arrival order; the
            # aggregator only cares about (uid, traj_index)
            loop = asyncio.get_running_loop()
            async with self._preprocess_slots:
                if self.preprocess_fn is not None:
                    t0 = loop.time()
                    traj = await self.preprocess_fn(traj)
                    self.stats.preprocess_busy_time_s += loop.time() - t0
            group = self.aggregator.add_trajectory(traj)
            if group is None or group.evicted:
                return
            await self._on_group_and_maybe_train(group)

    async def _preprocess_group(self, group: GroupRecord) -> None:
        """Preprocess every trajectory of a completed group (concurrently,
        still bounded by preprocess_concurrency)."""
        loop = asyncio.get_running_loop()

        async def one(traj: TrajectorySample) -> TrajectorySample:
            async with self._preprocess_slots:
                if self.preprocess_fn is not None:
                    t0 = loop.time()
                    traj = await self.preprocess_fn(traj)
                    self.stats.preprocess_busy_time_s += loop.time() - t0
            return traj

        group.trajectories = list(await asyncio.gather(*[one(t) for t in group.trajectories]))

    async def _on_group_and_maybe_train(self, group: GroupRecord) -> None:
        if self.on_group:
            self.on_group(group)
        batch = self.mini_batcher.add_group(group)
        if batch is not None:
            await self._train_batch(batch)

    # ----------------------------------------------------------------- train

    async def _train_batch(self, groups: list[GroupRecord]) -> None:
        """One policy update over ``mini_batch_groups`` complete groups.

        Serialized by ``_update_lock``: concurrent completions queue up here
        while their preprocessing tasks have already returned.
        """
        async with self._update_lock:
            await self._train_batch_locked(groups)

    async def _train_batch_locked(self, groups: list[GroupRecord]) -> None:
        batch = TrainerBatch(
            groups=groups,
            policy_version_before=self.current_version,
            created_at=time.monotonic(),
        )
        self.stats.mini_batches += 1
        if self.stats.time_to_first_batch_s is None and self._start_time is not None:
            self.stats.time_to_first_batch_s = batch.created_at - self._start_time

        # staleness-based group dropping (freshness control)
        trainable: list[GroupRecord] = groups
        if self.config.max_staleness_drop is not None:
            trainable = []
            for group in groups:
                if group.oldest_staleness(self.current_version) > self.config.max_staleness_drop:
                    self.stats.groups_dropped_stale += 1
                    logger.info(
                        "Dropped stale group %s (oldest version gap %d > %d)",
                        group.uid,
                        group.oldest_staleness(self.current_version),
                        self.config.max_staleness_drop,
                    )
                else:
                    trainable.append(group)
            if not trainable:
                return

        # GRPO advantages (computed for accounting; a real trainer would feed
        # them into update_actor alongside the payloads)
        advantages: list[float] = []
        for group in trainable:
            advantages.extend(grpo_group_advantages(group.rewards))
        self.stats.advantage_abs_mean_history.append(
            sum(abs(a) for a in advantages) / max(1, len(advantages))
        )

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        if self.update_fn is not None:
            await self.update_fn(batch)
        else:
            await asyncio.sleep(self.config.update_time_s)
        self.stats.update_busy_time_s += loop.time() - t0

        self.current_version += 1
        self.stats.updates += 1
        self.stats.groups_trained += len(trainable)
        now = time.monotonic()
        self._version_timeline.append((now, self.current_version))

        # off-policy distance of the update: how far the consuming version
        # (before the increment) was from the versions that generated the data
        for group in trainable:
            self.stats.staleness_history.append(group.staleness(batch.policy_version_before))
            self.stats.oldest_staleness_history.append(group.oldest_staleness(batch.policy_version_before))
            self.stats.version_span_history.append(group.version_span)
            # Laminar inherent staleness: version when the trajectory finished
            # generating minus the version that generated it (clamped at 0:
            # clock boundaries can transiently read one version behind)
            for traj in group.trajectories:
                self.stats.inherent_staleness_history.append(
                    max(0, self.version_at(traj.complete_time) - traj.model_version)
                )

        # tokens/s per RL iteration — the paper's main throughput metric:
        # tokens in the trained batch over the interval between consecutive
        # actor update completions
        batch_tokens = sum(
            traj.num_tokens + self._prompt_tokens_of(traj)
            for group in trainable
            for traj in group.trajectories
        )
        self.stats.total_trained_tokens += batch_tokens
        if self._last_update_completion is not None:
            dt = now - self._last_update_completion
            if dt > 0:
                self.stats.tokens_per_s_history.append(batch_tokens / dt)
        self._last_update_completion = now

        if self.on_batch is not None:
            metrics = {
                "batch_uid": batch.uid,
                "groups": [g.uid for g in trainable],
                "advantage_abs_mean": self.stats.advantage_abs_mean_history[-1],
                "versions": [g.model_versions for g in trainable],
            }
            self.on_batch(batch, metrics)

    # --------------------------------------------------------------- helpers

    def version_at(self, t: float) -> int:
        """Trainer version at monotonic time ``t`` (0 before the first update).

        Used for inherent-staleness accounting: the trajectory finished at
        ``t`` — which policy version had the actor reached by then?
        """
        version = 0
        for ts, v in self._version_timeline:
            if ts <= t:
                version = v
            else:
                break
        return version

    @staticmethod
    def _prompt_tokens_of(traj: TrajectorySample) -> int:
        payload = traj.payload
        if isinstance(payload, dict):
            return int(payload.get("prompt_tokens", 0))
        return 0

    # --------------------------------------------------------------- finalize

    def _finalize(self) -> None:
        leftover = self.mini_batcher.drain()
        self.stats.groups_leftover = len(leftover)
        if leftover:
            logger.warning(
                "Training finished with %d complete group(s) not forming a full "
                "mini-batch (choose num_prompts × n divisible by mini_batch_groups); "
                "they are NOT trained",
                len(leftover),
            )
        if self.aggregator.num_partial_groups:
            logger.warning(
                "Training finished with %d partial group(s) in the aggregator: %s",
                self.aggregator.num_partial_groups,
                self.aggregator.partial_uids()[:8],
            )
