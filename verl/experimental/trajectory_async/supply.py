# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Just-in-time training supply and exclusive, drain-before-transfer replica loans."""

import asyncio
import json
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SupplyConfig:
    enabled: bool = False
    elastic_evaluation: bool = True
    min_training_replicas: int = 2
    low_ready_batches: float = 0.5
    target_ready_batches: float = 1.0
    high_ready_batches: float = 1.5
    max_pending_batches: float = 2.0
    refill_groups: int = 0  # 0 inherits the training batch size.
    target_staleness: float = 3.0
    staleness_window: int = 5
    healthy_polls: int = 3
    loan_cooldown_s: float = 60.0
    poll_interval_s: float = 2.0
    feedback_timeout_s: float = 15.0
    starvation_wait_s: float = 5.0

    def __post_init__(self):
        if not 0 <= self.low_ready_batches < self.target_ready_batches <= self.high_ready_batches:
            raise ValueError("Supply watermarks must satisfy 0 <= low < target <= high")
        if self.max_pending_batches < self.high_ready_batches or self.min_training_replicas < 1:
            raise ValueError("Supply pending capacity must cover the high watermark and reserve at least one replica")
        if type(self.refill_groups) is not int or self.refill_groups < 0:
            raise ValueError("Supply refill_groups must be a nonnegative integer")
        if self.staleness_window < 1 or self.healthy_polls < 1 or self.target_staleness < 0:
            raise ValueError("Invalid supply staleness window, healthy polls or target")
        if min(self.poll_interval_s, self.feedback_timeout_s, self.starvation_wait_s) <= 0 or self.loan_cooldown_s < 0:
            raise ValueError("Supply polling/timeouts must be positive and cooldown nonnegative")
        if self.feedback_timeout_s <= self.poll_interval_s:
            raise ValueError("Supply feedback timeout must exceed the polling interval")


def queue_supply_snapshot(partition, rollout_n, batch_size):
    """Count whole, terminal groups; sampled tensor rows are deliberately ignored."""
    prompts = {key: tag for key, tag in partition.items() if tag.get("is_prompt")}
    completed = defaultdict(set)
    for key, tag in partition.items():
        if tag.get("is_prompt") or tag.get("status") != "success":
            continue
        parts = key.rsplit("_", 2)
        if len(parts) == 3 and parts[0] in prompts and parts[2] == "0":
            completed[parts[0]].add(int(parts[1]))
    ready = sum(
        tag.get("status") == "finished" and completed[uid] == set(range(rollout_n)) for uid, tag in prompts.items()
    )
    successes = sum(len(completed[uid] & set(range(rollout_n))) for uid in prompts)
    return {
        "ready_groups": ready,
        "pending_groups": len(prompts),
        "ready_batches": ready / batch_size,
        "unfinished_trajectories": len(prompts) * rollout_n - successes,
    }


def should_refill(config, queue, runtime, batch_size, now):
    """Refill a ready-group deficit while there is engine work capacity.

    Completed members of an incomplete group must not occupy production slots.
    The separate prompt-count bound still caps total unsampled experience.
    """
    if queue["ready_groups"] >= math.ceil(config.target_ready_batches * batch_size):
        return False
    if not runtime or now - runtime["observed_at"] > config.feedback_timeout_s:
        return True  # Fail open within the hard experience bound.
    return queue["unfinished_trajectories"] < runtime["admission_total"]


class SupplyPolicy:
    def __init__(self, config):
        self.config = config
        self.paused = False
        self.healthy = 0
        self.last_loan = -math.inf

    def decide(self, feedback, borrowed, total, evaluation_demand, now):
        c = self.config
        ready = feedback["ready_batches"]
        age = feedback.get("window_staleness", 0.0)
        if feedback.get("training_finished"):
            self.paused = True
            return True, total if evaluation_demand else 0, "training_finished"
        starving = ready <= c.low_ready_batches or feedback.get("sample_wait_s", 0.0) >= c.starvation_wait_s
        if starving:
            self.paused = False
            self.healthy = 0
            return False, 0, "refill"
        if ready >= c.high_ready_batches or (ready >= c.target_ready_batches and age > c.target_staleness):
            self.paused = True
        elif ready <= c.low_ready_batches:
            self.paused = False
        self.healthy = self.healthy + 1 if ready >= c.target_ready_batches else 0
        limit = max(0, total - c.min_training_replicas)
        if not evaluation_demand or not c.elastic_evaluation:
            return self.paused, 0, "no_evaluation_demand"
        if borrowed > limit:
            return self.paused, limit, "training_reserve"
        if self.healthy >= c.healthy_polls and borrowed < limit and now - self.last_loan >= c.loan_cooldown_s:
            self.last_loan = now
            self.healthy = 0
            return self.paused, borrowed + 1, "surplus"
        return self.paused, borrowed, "hold"


class SupplyCoordinator:
    """Only this actor transfers ownership; controllers never share admission.

    Every shared engine has a stable slot in each controller. The evaluation
    slot starts unowned. A loan first fences the source, waits for active work,
    pull tasks and migration reservations, then disables the source before
    enabling the destination. Returning engines reload the latest train policy.
    """

    def __init__(self, training, evaluation, training_replicas, evaluation_offset, options, event_path=None):
        self.training = training
        self.evaluation = evaluation
        self.total = training_replicas
        self.offset = evaluation_offset
        self.config = SupplyConfig(**options)
        self.policy = SupplyPolicy(self.config)
        self.borrowed = set()
        self.transitions = {}  # index -> destination pool
        self.feedback = None
        self.feedback_at = -math.inf
        self.failure = None
        self.task = None
        self.stopped = False
        self.last_decision = {}
        self.transfers = 0
        self.event_path = Path(event_path) if event_path else None
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)

    def _event(self, event, **fields):
        if self.event_path:
            with self.event_path.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")

    async def start(self):
        self.task = asyncio.create_task(self._run())

    def update(self, feedback):
        self.feedback = dict(feedback)
        self.feedback_at = time.monotonic()

    async def tick(self, now=None):
        now = time.monotonic() if now is None else now
        fresh = self.feedback is not None and now - self.feedback_at <= self.config.feedback_timeout_s
        evaluation_status = await self.evaluation.status.remote() if self.evaluation is not None else {}
        demand = bool(evaluation_status.get("evaluation_requests"))
        if fresh:
            committed = len(self.borrowed) + sum(dest == "evaluation" for dest in self.transitions.values())
            paused, desired, reason = self.policy.decide(self.feedback, committed, self.total, demand, now)
        else:
            paused, desired, reason = False, 0, "feedback_timeout"
            self.policy.paused = False
            self.policy.healthy = 0
        await self.training.set_supply_paused.remote(paused)
        # Cancel a not-yet-released loan when the consumer needs supply again.
        for index, destination in list(self.transitions.items()):
            if destination == "evaluation" and desired <= len(self.borrowed):
                await self.training.cancel_release.remote(index)
                del self.transitions[index]
                self._event("loan_cancelled", replica=index)
        returning = [i for i, dest in self.transitions.items() if dest == "training"]
        if len(self.borrowed) - len(returning) > desired:
            for index in sorted(self.borrowed - set(returning), reverse=True)[
                : len(self.borrowed) - len(returning) - desired
            ]:
                self.transitions[index] = "training"
                self._event("replica_draining", replica=index, destination="training")
        planned = len(self.borrowed) + sum(dest == "evaluation" for dest in self.transitions.values())
        if planned < desired:
            status = await self.training.status.remote()
            candidates = [
                i
                for i, r in enumerate(status["replicas"])
                if i not in self.borrowed and i not in self.transitions and r["healthy"]
            ]
            candidates.sort(key=lambda i: status["replicas"][i]["active"] + status["replicas"][i]["reserved"])
            reserve = 0 if fresh and self.feedback.get("training_finished") else self.config.min_training_replicas
            owned_healthy = sum(r["owned"] and r["healthy"] and not r["releasing"] for r in status["replicas"])
            for index in candidates[: min(desired - planned, max(0, owned_healthy - reserve))]:
                self.transitions[index] = "evaluation"
                self._event("replica_draining", replica=index, destination="evaluation")
        for index, destination in list(self.transitions.items()):
            if destination == "evaluation":
                if await self.training.release_replica.remote(index):
                    status = await self.training.status.remote()
                    await self.evaluation.acquire_replica.remote(
                        self.offset + index, admission_ceiling=status["replicas"][index]["admission_limit"]
                    )
                    self.borrowed.add(index)
                else:
                    continue
            else:
                if await self.evaluation.release_replica.remote(self.offset + index):
                    await self.training.acquire_replica.remote(index)
                    self.borrowed.remove(index)
                else:
                    continue
            del self.transitions[index]
            self.transfers += 1
            self._event("ownership_transferred", replica=index, destination=destination)
        decision = {"paused": paused, "desired_loans": desired, "reason": reason, "feedback_fresh": fresh}
        if decision != self.last_decision:
            self._event("supply_decision", **decision, feedback=self.feedback)
        self.last_decision = decision
        return self.status()

    async def _run(self):
        try:
            while not self.stopped:
                await self.tick()
                await asyncio.sleep(self.config.poll_interval_s)
        except Exception as error:
            self.failure = repr(error)  # Fail the run rather than risk ambiguous ownership.

    def status(self):
        if self.failure:
            raise RuntimeError(f"Supply coordinator failed: {self.failure}")
        return {
            **self.last_decision,
            "feedback": dict(self.feedback or {}),
            "borrowed": sorted(self.borrowed),
            "transitions": dict(self.transitions),
            "transfers": self.transfers,
        }

    async def stop(self):
        self.stopped = True
        if self.task:
            await self.task


class SupplyMonitor:
    """Trainer-side metadata polling continues while NPU collectives are busy."""

    def __init__(self, poll, interval):
        self.poll = poll
        self.interval = interval
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="laminar-supply-monitor", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self.poll()
                self.stop_event.wait(self.interval)
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("Training supply monitor failed") from self.error

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Training supply monitor failed to stop")
