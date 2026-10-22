# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Scheduler metrics consumed by the real Laminar controller."""

import time

from vllm.v1.metrics.loggers import StatLoggerBase


class TrajectoryStatsLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_index=0, sink=None):
        self.sink = sink

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if iteration_stats is not None:
            self.sink["preemptions"] = self.sink.get("preemptions", 0) + int(
                getattr(iteration_stats, "num_preempted_reqs", 0)
            )
        if scheduler_stats is not None:
            self.sink.update(
                kv_usage=float(scheduler_stats.kv_cache_usage),
                running=int(scheduler_stats.num_running_reqs),
                waiting=int(scheduler_stats.num_waiting_reqs),
                observed_at=time.time(),
            )

    def log_engine_initialized(self):
        pass
