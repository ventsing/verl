# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Per-replica online admission using profiled pools and bounded exploration.

Pool pages come from the loaded engine, including recurrent-state groups.
Configuration supplies a starting batch, safety margin and an immutable engine
ceiling. Only fresh, sufficiently loaded observations permit gradual growth.
"""

import math
from dataclasses import dataclass


@dataclass
class CapacityConfig:
    enabled: bool = True
    kv_usage_limit: float = 0.9
    healthy_polls: int = 3
    growth_factor: float = 1.25
    backoff_factor: float = 0.75
    load_ratio: float = 0.75
    context_bucket_tokens: int = 4096
    max_observation_age_s: float = 15.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("Capacity enabled must be boolean")
        for name in ("healthy_polls", "context_bucket_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid capacity {name}")
        for name in ("kv_usage_limit", "backoff_factor", "load_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"Invalid capacity {name}")
        if not math.isfinite(self.growth_factor) or not 1 < self.growth_factor <= 2:
            raise ValueError("Capacity growth_factor must be in (1, 2]")
        if not math.isfinite(self.max_observation_age_s) or self.max_observation_age_s <= 0:
            raise ValueError("Invalid capacity observation age")


def engine_capacity(worker):
    """Read only CPU configuration; no device synchronization or tensor copy.

    The scheduler shares its block pool across cache groups. A group's maximum
    resident page demand includes Mamba none/align/all and speculative states.
    Use the engine's max_memory_usage_bytes contract rather than inferring SSM
    capacity from attention token counts. Reserve each request's full model
    context, so growing tails cannot invalidate a short-context extrapolation.
    """
    try:
        runner = worker.model_runner
        config = runner.kv_cache_config
        vllm_config = runner.vllm_config
        groups = []
        for group in config.kv_cache_groups:
            spec = group.kv_cache_spec
            page = int(spec.page_size_bytes)
            maximum_bytes = int(spec.max_memory_usage_bytes(vllm_config))
            if page <= 0 or maximum_bytes <= 0:
                raise ValueError("Nonpositive cache-group capacity")
            groups.append(
                {
                    "type": type(spec).__name__,
                    "layers": len(group.layer_names),
                    "page_bytes": page,
                    "max_pages_per_request": math.ceil(maximum_bytes / page),
                    "mamba_cache_mode": getattr(spec, "mamba_cache_mode", None),
                }
            )
        pages = sum(g["max_pages_per_request"] for g in groups)
        blocks = int(config.num_blocks) - 1  # Engine reserves a null block.
        if blocks <= 0 or pages <= 0:
            raise ValueError("Empty cache pool")
        return {
            "available": True,
            "rank": int(getattr(worker, "rank", getattr(worker, "local_rank", 0))),
            "total_blocks": blocks,
            "max_pages_per_request": pages,
            "max_model_len": int(vllm_config.model_config.max_model_len),
            "engine_limit": int(vllm_config.scheduler_config.max_num_seqs),
            "groups": groups,
        }
    except (AttributeError, TypeError, ValueError, NotImplementedError) as error:
        return {"available": False, "error": repr(error)}


def join_engine_capacity(packets, tp_size):
    """Take the tightest TP rank, never average independent device capacities."""
    values = [p.get("capacity", {}) for p in packets]
    if (
        len(values) != tp_size
        or not all(v.get("available") for v in values)
        or len({v.get("rank") for v in values}) != tp_size
    ):
        return {"available": False, "reason": "missing_tp_capacity", "workers": values}

    def signature(value):
        return value["max_model_len"], value["engine_limit"], repr(value["groups"])

    if len({signature(v) for v in values}) != 1:
        return {"available": False, "reason": "inconsistent_tp_layout", "workers": values}
    worst = max(v["max_pages_per_request"] / v["total_blocks"] for v in values)
    return {
        "available": True,
        "request_fraction": worst,
        "engine_limit": min(v["engine_limit"] for v in values),
        "max_model_len": values[0]["max_model_len"],
        "workers": values,
    }


class OnlineCapacity:
    def __init__(self, initial, ceiling, options=None):
        self.config = CapacityConfig(**dict(options or {}))
        self.ceiling = ceiling
        self.initial = min(initial, ceiling)
        self.current = self.initial
        self.healthy = 0
        self.last_observed = -math.inf
        self.preemptions = None
        self.profile = None
        self.pool = {}
        self.reason = "configured_initial"
        self.updates = 0

    def memory_limit(self):
        if not self.pool.get("available"):
            return self.initial
        return min(
            self.ceiling,
            self.pool["engine_limit"],
            math.floor(self.config.kv_usage_limit / self.pool["request_fraction"]),
        )

    def update(self, telemetry, roofline, now):
        old = self.status()
        observed = telemetry.get("observed_at", 0)
        usage = telemetry.get("kv_usage", math.nan)
        if (
            not isinstance(observed, (int, float))
            or not math.isfinite(observed)
            or observed <= self.last_observed
            or observed > now
            or now - observed > self.config.max_observation_age_s
            or not isinstance(usage, (int, float))
            or not math.isfinite(usage)
            or not 0 <= usage <= 1
        ):
            self.healthy = 0
            self.reason = "stale_or_invalid_telemetry"
            self.current = min(self.current, self.initial)
            return old != self.status()
        self.last_observed = observed
        data = telemetry.get("roofline", {})
        self.pool = data.get("capacity", {"available": False})
        if not self.pool.get("available"):
            self.current, self.healthy, self.reason = min(self.current, self.initial), 0, "capacity_unavailable"
            return old != self.status()
        memory_limit = self.memory_limit()
        if memory_limit < 1:
            raise RuntimeError("No request fits the profiled KV/SSM safety budget at max_model_len")
        maximum = data.get("max_context_tokens", 0)
        mode = data.get("mode", "unknown")
        profile = (mode, int((maximum - 1) // self.config.context_bucket_tokens)) if maximum > 0 else None
        if profile != self.profile:
            self.profile, self.healthy = profile, 0
            self.current = min(self.current, self.initial)
        measured = roofline.calibrated_limit(mode, maximum) if roofline else None
        hard = min(memory_limit, measured if measured is not None else self.ceiling)
        total_preemptions = int(telemetry.get("preemptions", 0))
        pressure = (
            usage >= self.config.kv_usage_limit
            or telemetry.get("waiting", 0) > 0
            or total_preemptions > (self.preemptions or 0)
        )
        self.preemptions = total_preemptions
        if pressure:
            self.current = min(hard, max(1, math.floor(self.current * self.config.backoff_factor)))
            self.healthy, self.reason = 0, "memory_or_scheduler_pressure"
        elif self.current > hard:
            self.current, self.healthy, self.reason = hard, 0, "calibrated_bound"
        elif self.current >= hard:
            self.current, self.healthy, self.reason = hard, 0, "at_safety_bound"
        elif (
            data.get("available")
            and (data.get("collecting") or measured is not None)
            and mode != "unknown"
            and telemetry.get("running", 0) >= math.ceil(self.current * self.config.load_ratio)
        ):
            self.healthy += 1
            self.reason = "observing"
            if self.healthy >= self.config.healthy_polls:
                self.current = min(hard, max(self.current + 1, math.ceil(self.current * self.config.growth_factor)))
                self.healthy = 0
                self.reason = "bounded_exploration" if self.current < hard else "at_safety_bound"
                self.updates += 1
        else:
            self.healthy, self.reason = 0, "holding"
        return old != self.status()

    def limit(self):
        return min(self.current, self.memory_limit())

    def projected_usage(self, current_usage, arrivals):
        if not self.pool.get("available"):
            return math.inf  # Missing layout cannot authorize migration.
        return current_usage + arrivals * self.pool["request_fraction"]

    def status(self):
        return {
            "initial": self.initial,
            "limit": self.limit(),
            "engine_ceiling": self.ceiling,
            "kv_batch_limit": self.memory_limit(),
            "kv_usage_limit": self.config.kv_usage_limit,
            "reason": self.reason,
            "healthy_polls": self.healthy,
            "updates": self.updates,
            "profile": self.profile,
            "pool": self.pool,
        }
