# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded prompt production independent of synchronous actor updates."""

import math
import threading


class PromptFeeder:
    def __init__(self, count_pending, submit_batch, batch_size, capacity, lock, poll_interval=0.2, can_submit=None):
        if batch_size < 1 or capacity < batch_size or poll_interval <= 0:
            raise ValueError("Prompt capacity must hold a positive batch and polling must be positive")
        self.count_pending = count_pending
        self.submit_batch = submit_batch
        self.batch_size = batch_size
        self.capacity = capacity
        self.lock = lock
        self.poll_interval = poll_interval
        self.can_submit = can_submit
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.batches = 0

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="laminar-prompt-feeder", daemon=True)
            self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    if self.count_pending() + self.batch_size <= self.capacity and (
                        self.can_submit is None or self.can_submit()
                    ):
                        self.submit_batch()
                        self.batches += 1
                        continue
                self.stop_event.wait(self.poll_interval)
        except Exception as error:
            self.error = error

    def check(self):
        if self.error is not None:
            raise RuntimeError("Laminar prompt producer failed") from self.error

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Laminar prompt producer did not stop within 10 seconds")


class AdaptivePromptBuffer:
    """Bound unsampled experience without cancelling already submitted groups."""

    def __init__(self, batch_size, hard_capacity, options=None):
        opts = dict(options or {})
        self.minimum = math.ceil(opts.get("min_batches", 2.0) * batch_size)
        self.maximum = min(hard_capacity, math.floor(opts.get("max_batches", 4.0) * batch_size))
        self.current = min(self.maximum, math.ceil(opts.get("initial_batches", 4.0) * batch_size))
        self.step = opts.get("step_groups", 16)
        self.refill_groups = opts.get("refill_groups", 16)
        self.interval = opts.get("adjust_interval_s", 60.0)
        self.healthy_polls = opts.get("healthy_polls", 3)
        self.target_staleness = opts.get("target_staleness", 1.0)
        self.feedback_timeout = opts.get("feedback_timeout_s", 15.0)
        self.batch_size = batch_size
        if not batch_size <= self.minimum <= self.current <= self.maximum:
            raise ValueError("Adaptive prompt buffer requires batch <= min <= initial <= max <= hard capacity")
        if any(type(x) is not int or x < 1 for x in [self.step, self.refill_groups, self.healthy_polls]):
            raise ValueError("Adaptive prompt buffer step, refill and healthy polls must be positive integers")
        if self.refill_groups > self.minimum or not all(
            math.isfinite(x) and x > 0 for x in [self.interval, self.feedback_timeout]
        ):
            raise ValueError("Invalid adaptive prompt buffer refill size or timing")
        if not math.isfinite(self.target_staleness) or self.target_staleness < 0:
            raise ValueError("Adaptive prompt buffer staleness must be finite and nonnegative")
        self.last_adjusted = -math.inf
        self.direction = 0
        self.streak = 0
        self.reason = "initial"
        self.adjustments = 0

    def observe(self, queue, runtime, staleness, sample_wait_s, now):
        # Missing/stale capacity information never authorizes expansion/shrink.
        if not runtime or not 0 <= now - runtime["observed_at"] <= self.feedback_timeout:
            self.direction, self.streak, self.reason = 0, 0, "stale_feedback"
            return self.current
        ready = queue["ready_groups"] / self.batch_size
        admission = runtime["admission_total"]
        direction, reason = 0, "holding"
        if ready >= 1.5 or (ready >= 0.5 and staleness > self.target_staleness):
            direction, reason = -1, "ready_surplus_or_staleness"
        elif (
            ready >= 0.5
            and admission > 0
            and runtime["active_total"] >= 0.8 * admission
            and queue["unfinished_trajectories"] > 2 * admission
        ):
            direction, reason = -1, "excess_queued_work"
        elif (
            ready < 0.5
            and sample_wait_s >= 5
            and admission > 0
            and runtime["active_total"] < 0.8 * admission
            and queue["unfinished_trajectories"] < admission
            and queue["pending_groups"] >= self.current - self.refill_groups
        ):
            direction, reason = 1, "starved_with_engine_headroom"
        self.streak = self.streak + 1 if direction and direction == self.direction else int(bool(direction))
        self.direction, self.reason = direction, reason
        if self.streak >= self.healthy_polls and now - self.last_adjusted >= self.interval:
            proposed = max(self.minimum, min(self.maximum, self.current + direction * self.step))
            if proposed != self.current:
                self.current = proposed
                self.last_adjusted = now
                self.adjustments += 1
            self.streak = 0
        return self.current
