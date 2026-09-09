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
"""Trajectory-level rollouter.

Two delivery modes over an identical generation backend, so the effect of
the delivery granularity can be measured in isolation:

* ``mode="trajectory"`` — each of a prompt's ``n`` responses is an
  independent request. The moment a response finishes, its reward is scored
  and the trajectory is pushed downstream; siblings are not waited for.
  This is the trajectory-level data plane this package implements.
* ``mode="group"`` — the baseline that mirrors ``fully_async_policy``'s
  sample-level streaming: the ``n`` responses run as independent
  per-trajectory tasks *inside* the group (exactly like
  ``AgentLoopWorker.generate_sequences``, which runs one asyncio task per
  row and gathers), but delivery to the queue happens only after the whole
  group settled — the unit the trainer sees is the group.

Both modes share the same generation, retry, and reward logic; the only
difference is *when results cross the queue*.

Failure semantics (``group_retry``):

* ``"per-trajectory"`` — both modes retry a transiently-failed trajectory
  individually. This isolates the pure delivery-granularity effect.
* ``"none"`` — in group mode a terminally-failed trajectory loses the whole
  group (all-or-nothing, the granularity ``fully_async_policy`` operates
  at: one failed sample drops n responses of work). Trajectory mode always
  retries per-trajectory; comparing against ``group_retry="none"`` shows the
  failure-granularity benefit.

Delivery protocol towards the trainer (see ``group_aggregator``): in
trajectory mode, ``FAILED`` sentinels for a group are emitted only after
every sibling task of that group settled, so no trajectory can arrive after
its group's eviction.

Reward scoring is rollouter-side, mirroring verl's agent-loop scoring: it
runs as soon as its trajectory is available, bounded by
``reward_concurrency`` — in both modes (this matches the real
``AgentLoopWorker``, whose per-row tasks score while siblings still
generate).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from verl.experimental.trajectory_async.mock_rollout import (
    MockGenResult,
    MockRolloutEngine,
    MockRolloutError,
    request_seed,
)
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import (
    TrajectorySample,
    TrajectoryStatus,
)

logger = logging.getLogger(__name__)

RewardFn = Callable[[TrajectorySample], Awaitable[float]]

GroupRetryPolicy = Literal["per-trajectory", "none"]


@dataclass
class PromptRecord:
    """One prompt fed into the rollouter.

    ``payload`` is opaque — a dict in mock mode, a 1-row DataProto slice in
    real mode (see README wiring guide).
    """

    uid: str
    payload: Any = None
    prompt_tokens: int = 256


@dataclass
class RollouterConfig:
    n: int = 8  # rollout.n — trajectories per prompt group
    mode: str = "trajectory"  # "trajectory" | "group"
    max_inflight_trajectories: int = 512  # engine-side admission control
    max_retries: int = 2  # retries per trajectory beyond the first attempt
    reward_concurrency: int = 32  # concurrent reward scorings
    # failure policy for group mode (see module docstring); trajectory mode
    # always retries per-trajectory
    group_retry: GroupRetryPolicy = "per-trajectory"


@dataclass
class RollouterStats:
    prompts_fed: int = 0
    trajectories_generated: int = 0
    trajectory_attempts: int = 0
    trajectory_failures: int = 0
    trajectories_pushed: int = 0
    groups_dropped: int = 0
    wasted_trajectory_attempts: int = 0  # work thrown away with dropped groups
    reward_busy_time_s: float = 0.0
    wall_time_s: float = 0.0

    def snapshot(self) -> dict[str, float]:
        return {f"rollouter/{k}": v for k, v in vars(self).items()}


class TrajectoryRollouter:
    """Feeds prompts to the engine and streams results to the queue."""

    def __init__(
        self,
        engine: MockRolloutEngine,
        queue: InProcessTrajectoryQueue,
        config: RollouterConfig | None = None,
        reward_fn: RewardFn | None = None,
        on_trajectory: Callable[[TrajectorySample], None] | None = None,
        on_group_settled: Callable[[str, bool], None] | None = None,
    ) -> None:
        if config and config.mode not in ("trajectory", "group"):
            raise ValueError(f"unknown mode: {config.mode}")
        if config and config.group_retry not in ("per-trajectory", "none"):
            raise ValueError(f"unknown group_retry policy: {config.group_retry}")
        self.engine = engine
        self.queue = queue
        self.config = config or RollouterConfig()
        self.reward_fn = reward_fn
        self.on_trajectory = on_trajectory
        self.on_group_settled = on_group_settled
        self.stats = RollouterStats()
        self._gen_slots = asyncio.Semaphore(self.config.max_inflight_trajectories)
        self._reward_slots = asyncio.Semaphore(self.config.reward_concurrency)

    # ------------------------------------------------------------------- run

    async def run(
        self,
        prompts: AsyncIterator[PromptRecord] | list[PromptRecord],
        num_consumers: int = 1,
    ) -> None:
        """Generate for every prompt, then close the queue (one sentinel per
        consumer). Returns when all trajectories have been pushed."""
        start = time.monotonic()
        supervisors: set[asyncio.Task] = set()

        prompts_iter: AsyncIterator[PromptRecord] = (
            _list_iterator(prompts) if isinstance(prompts, list) else prompts
        )

        async for prompt in prompts_iter:
            self.stats.prompts_fed += 1
            task = asyncio.create_task(self._run_group(prompt), name=f"group-{prompt.uid}"[:255])
            supervisors.add(task)
            task.add_done_callback(supervisors.discard)

        if supervisors:
            await asyncio.gather(*supervisors)

        for _ in range(num_consumers):
            await self.queue.put(None)
        self.stats.wall_time_s = time.monotonic() - start

    # --------------------------------------------------------------- groups

    async def _run_group(self, prompt: PromptRecord) -> None:
        """Supervisor for one prompt group."""
        if self.config.mode == "trajectory":
            await self._run_group_trajectory_mode(prompt)
        else:
            await self._run_group_group_mode(prompt)

    async def _run_group_trajectory_mode(self, prompt: PromptRecord) -> None:
        """trajectory mode: n independent tasks, each streaming on
        completion; FAILED sentinels after the group settles."""
        tasks = [
            asyncio.create_task(self._run_trajectory(prompt, i), name=f"traj-{prompt.uid}-{i}"[:255])
            for i in range(self.config.n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dead_indexes = [
            i
            for i, r in enumerate(results)
            if isinstance(r, BaseException) or r is None
        ]
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.error("Trajectory %s#%d crashed: %r", prompt.uid, i, r)
        if dead_indexes:
            # every sibling has settled — safe to emit failure sentinels now
            self.stats.groups_dropped += 1
            for i in dead_indexes:
                await self.queue.put(
                    TrajectorySample(
                        uid=prompt.uid,
                        traj_index=i,
                        group_size=self.config.n,
                        status=TrajectoryStatus.FAILED,
                    )
                )
        if self.on_group_settled:
            self.on_group_settled(prompt.uid, not dead_indexes)

    async def _run_group_group_mode(self, prompt: PromptRecord) -> None:
        """group mode: per-trajectory (gen → reward) tasks gathered — the
        same structure as ``AgentLoopWorker.generate_sequences`` — then the
        n rows are pushed together as one delivery unit."""
        # all-or-nothing policy loses the whole group on any failure;
        # per-trajectory policy retries each response individually.
        max_attempts = 1 if self.config.group_retry == "none" else None
        tasks = [
            asyncio.create_task(self._gen_and_score(prompt, i, max_attempts), name=f"gtraj-{prompt.uid}-{i}"[:255])
            for i in range(self.config.n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples: list[TrajectorySample] = []
        dead = 0
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.error("Trajectory %s#%d crashed: %r", prompt.uid, i, r)
                dead += 1
            elif r is None:
                dead += 1
            else:
                samples.append(r)
        if dead:
            self.stats.groups_dropped += 1
            self.stats.wasted_trajectory_attempts += len(samples)
            if self.on_group_settled:
                self.on_group_settled(prompt.uid, False)
            return
        for traj in samples:
            await self.queue.put(traj)
            self.stats.trajectories_pushed += 1
            if self.on_trajectory:
                self.on_trajectory(traj)
        if self.on_group_settled:
            self.on_group_settled(prompt.uid, True)

    # ----------------------------------------------------------------- tasks

    async def _run_trajectory(self, prompt: PromptRecord, traj_index: int) -> TrajectorySample | None:
        """One independent trajectory task: generate → reward → push."""
        traj = await self._gen_and_score(prompt, traj_index)
        if traj is None:
            return None  # supervisor emits the FAILED sentinel after settlement
        self.stats.trajectories_pushed += 1
        await self.queue.put(traj)
        if self.on_trajectory:
            self.on_trajectory(traj)
        return traj

    async def _gen_and_score(
        self, prompt: PromptRecord, traj_index: int, max_attempts: int | None = None
    ) -> TrajectorySample | None:
        """Generate one trajectory (with retries) and score it. Returns
        ``None`` on terminal failure."""
        traj = await self._generate_with_retries(prompt, traj_index, max_attempts)
        if traj is None:
            return None
        traj.reward = await self._score(traj)
        return traj

    async def _generate_with_retries(
        self, prompt: PromptRecord, traj_index: int, max_attempts: int | None = None
    ) -> TrajectorySample | None:
        """Generate one trajectory, retrying transient failures.

        Returns ``None`` when the retry budget is exhausted (terminal
        failure).
        """
        submit_time = time.monotonic()
        attempts_allowed = max_attempts if max_attempts is not None else 1 + self.config.max_retries
        last_error: Exception | None = None
        for attempt in range(1, attempts_allowed + 1):
            self.stats.trajectory_attempts += 1
            seed = request_seed(prompt.uid, traj_index, attempt, base_seed=self.engine.config.seed)
            try:
                async with self._gen_slots:
                    result: MockGenResult = await self.engine.generate(seed, prompt.prompt_tokens)
                self.stats.trajectories_generated += 1
                return TrajectorySample(
                    uid=prompt.uid,
                    traj_index=traj_index,
                    group_size=self.config.n,
                    payload={
                        "prompt_tokens": prompt.prompt_tokens,
                        "num_tokens": result.num_tokens,
                        "seed": seed,
                    },
                    model_version=result.model_version,
                    gen_time_s=result.latency_s,
                    num_tokens=result.num_tokens,
                    submit_time=submit_time,
                    complete_time=time.monotonic(),
                    attempts=attempt,
                    status=TrajectoryStatus.COMPLETED,
                )
            except MockRolloutError as e:  # transient — retry the single trajectory
                self.stats.trajectory_failures += 1
                last_error = e
                continue
        logger.warning(
            "Trajectory %s#%d dead after %d attempts: %s",
            prompt.uid,
            traj_index,
            attempts_allowed,
            last_error,
        )
        return None

    async def _score(self, traj: TrajectorySample) -> float:
        if self.reward_fn is None:
            return 0.0
        loop = asyncio.get_running_loop()
        async with self._reward_slots:
            t0 = loop.time()
            reward = await self.reward_fn(traj)
            self.stats.reward_busy_time_s += loop.time() - t0
        return float(reward)


async def _list_iterator(items: list) -> AsyncIterator:
    for item in items:
        yield item
