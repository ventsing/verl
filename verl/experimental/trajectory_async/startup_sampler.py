# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in reservation of one startup batch independently of response completion.

Select StartupCohortReplayBuffer explicitly and set sampler_kwargs.startup_batch_groups
to the training prompt batch size. The first training batch waits for these prompt
groups; generation and admission can continue concurrently. Later batches use the
unchanged TrajectoryReplayBuffer selection policy. No reward or response length
is consulted to reserve the cohort, and no new version-age limit is introduced.

Ordering is first observation of prompt admission metadata, including pending
and running prompts. It does not certify the original dataloader order. Native
terminal eviction/refill policies still apply, with replacements recorded. A new
sampler instance, including one after a driver restart, reserves a fresh cohort.
This is a candidate startup intervention, not a claim of improved RL accuracy.
"""

import time

from verl.experimental.trajectory_async.tq_sampler import TrajectoryReplayBuffer


class StartupCohortReplayBuffer(TrajectoryReplayBuffer):
    """Reserve the first observed prompt groups before selecting a training batch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        size = self.sampler_kwargs.get("startup_batch_groups")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("sampler_kwargs.startup_batch_groups must be a positive integer")
        self.startup_batch_groups = size
        self._startup_observed = {}
        self._startup_cohort = ()
        self._startup_complete = False
        self._startup_materializing = False
        self._startup_started_at = None
        self._startup_wait_seconds = 0.0
        self._startup_evicted_groups = 0

    def _sync_metadata_from_transfer_queue(self):
        super()._sync_metadata_from_transfer_queue()
        if self._startup_complete:
            return
        prompts = self.prompt_global_steps["train"]
        missing = set(self._startup_cohort) - prompts.keys()
        if missing:
            raise RuntimeError(f"Reserved startup prompts disappeared without terminal eviction: {sorted(missing)}")
        for uid in prompts:
            self._startup_observed.setdefault(uid, None)

    def _reserve_startup_cohort(self):
        old = self._startup_cohort
        current = self.prompt_global_steps["train"]
        additions = [uid for uid in self._startup_observed if uid in current and uid not in old]
        needed = self.startup_batch_groups - len(old)
        if needed and len(additions) >= needed:
            self._startup_cohort = old + tuple(additions[:needed])
            if self._startup_started_at is None:
                self._startup_started_at = time.monotonic()
            self._record_trajectories(
                "train",
                [{"event": "startup_cohort_reserved", "uids": list(self._startup_cohort),
                  "added_uids": additions[:needed], "reserved_at": time.time(),
                  "ordering": "first_observed_prompt_metadata"}],
            )

    def _sampleable_terminal_keys(self, partition_id, eviction_reasons):
        eligible = super()._sampleable_terminal_keys(partition_id, eviction_reasons)
        if partition_id != "train" or self._startup_complete:
            return eligible
        self._reserve_startup_cohort()
        return eligible & set(self._startup_cohort)

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        if partition_id != "train" or self._startup_complete:
            return super()._select_prompt_uids(partition_id, sampleable_keys, batch_size)
        if batch_size != self.startup_batch_groups:
            raise ValueError("The startup cohort must equal the training prompt batch size")
        if len(self._startup_cohort) != batch_size or not set(self._startup_cohort) <= sampleable_keys:
            raise RuntimeError("The reserved startup cohort is not complete and eligible")
        return (list(self._startup_cohort), dict(self.partitions[partition_id]),
                dict(self.prompt_global_steps[partition_id]))

    def _forget_groups(self, partition_id, uids):
        super()._forget_groups(partition_id, uids)
        if partition_id != "train" or self._startup_complete or self._startup_materializing:
            return
        removed = [uid for uid in self._startup_cohort if uid in uids]
        self._startup_cohort = tuple(uid for uid in self._startup_cohort if uid not in uids)
        for uid in uids:
            self._startup_observed.pop(uid, None)
        if removed:
            self._startup_evicted_groups += len(removed)
            self._record_trajectories(
                "train", [{"event": "startup_cohort_terminal_eviction", "uids": removed,
                           "observed_at": time.time()}],
            )

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        if partition_id != "train" or self._startup_complete:
            return super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
        if tuple(selected_prompt_uids) != self._startup_cohort:
            raise RuntimeError("Startup selection changed the reserved prompt cohort")
        self._startup_materializing = True
        try:
            batch = super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
        finally:
            self._startup_materializing = False
        if len(batch.keys) != self.startup_batch_groups * self.rollout_n:
            raise RuntimeError("Startup selection must preserve complete trajectory groups")
        self._startup_complete = True
        self._startup_wait_seconds = time.monotonic() - self._startup_started_at
        self._startup_observed.clear()
        self._record_trajectories(
            "train", [{"event": "startup_cohort_consumed", "uids": list(self._startup_cohort),
                       "wait_seconds": self._startup_wait_seconds, "selected_at": time.time()}],
        )
        return batch

    def sample(self, global_steps, partition_id, batch_size):
        startup = partition_id == "train" and not self._startup_complete
        if startup and batch_size != self.startup_batch_groups:
            raise ValueError("The startup cohort must equal the training prompt batch size")
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        if partition_id == "train":
            metrics["trajectory/train/startup/cohort_batch"] = float(startup)
            metrics["trajectory/train/startup/wait_s"] = self._startup_wait_seconds if startup else 0.0
            metrics["trajectory/train/startup/evicted_groups"] = self._startup_evicted_groups if startup else 0
        return batch, metrics
