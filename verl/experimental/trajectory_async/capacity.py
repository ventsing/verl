# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Per-replica online admission using profiled pools and bounded exploration.

Pool pages come from the loaded engine, including recurrent-state groups.
Configuration supplies a starting batch, safety margin and an immutable engine
ceiling. Only fresh, sufficiently loaded observations permit gradual growth.
"""

import json
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path


@dataclass
class CapacityConfig:
    enabled: bool = True
    token_budget_admission: bool = False
    rolling_kv_enabled: bool = False
    rolling_kv_horizon_s: float = 30.0
    rolling_kv_rate_margin: float = 1.25
    rolling_kv_rate_ttl_s: float = 60.0
    rolling_kv_pressure_threshold: float = 0.90
    rolling_kv_resume_threshold: float = 0.80
    rolling_kv_max_moves: int = 4
    rolling_kv_migration_cooldown_s: float = 60.0
    response_prior_tokens: int = 8192
    response_history_window: int = 1024
    response_min_samples: int = 8
    response_quantile: float = 0.75
    response_ewma_alpha: float = 0.1
    response_step_calibration: bool = True
    response_calibration_online_weight: float = 0.2
    response_calibration_versions: int = 8
    response_safety_factor: float = 1.15
    telemetry_guard_tokens: int = 256
    response_history_path: str | None = None
    kv_usage_limit: float = 0.9
    healthy_polls: int = 3
    growth_factor: float = 1.25
    backoff_factor: float = 0.75
    load_ratio: float = 0.75
    context_bucket_tokens: int = 4096
    max_observation_age_s: float = 15.0

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.token_budget_admission) is not bool:
            raise ValueError("Capacity enabled must be boolean")
        if type(self.rolling_kv_enabled) is not bool or (self.rolling_kv_enabled and not self.token_budget_admission):
            raise ValueError("Rolling KV requires boolean enablement and token-budget admission")
        for name in ("rolling_kv_horizon_s", "rolling_kv_rate_ttl_s", "rolling_kv_migration_cooldown_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid capacity {name}")
        if not math.isfinite(self.rolling_kv_rate_margin) or self.rolling_kv_rate_margin < 1:
            raise ValueError("Rolling KV rate margin must be at least one")
        if not 0 < self.rolling_kv_resume_threshold < self.rolling_kv_pressure_threshold < self.kv_usage_limit:
            if self.rolling_kv_enabled:
                raise ValueError("Rolling KV thresholds must satisfy resume < pressure < KV limit")
        if type(self.rolling_kv_max_moves) is not int or self.rolling_kv_max_moves < 1:
            raise ValueError("Rolling KV migration bound must be positive")
        for name in (
            "healthy_polls",
            "context_bucket_tokens",
            "response_prior_tokens",
            "response_history_window",
            "response_min_samples",
            "telemetry_guard_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid capacity {name}")
        for name in ("kv_usage_limit", "backoff_factor", "load_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"Invalid capacity {name}")
        if type(self.response_step_calibration) is not bool:
            raise ValueError("Response step calibration must be boolean")
        if (
            not math.isfinite(self.response_calibration_online_weight)
            or not 0 <= self.response_calibration_online_weight <= 1
        ):
            raise ValueError("Response calibration online weight must be in [0, 1]")
        if type(self.response_calibration_versions) is not int or self.response_calibration_versions < 1:
            raise ValueError("Response calibration version count must be positive")
        if not 0 < self.response_quantile <= 1 or not 0 < self.response_ewma_alpha <= 1:
            raise ValueError("Response quantile and EWMA alpha must be in (0, 1]")
        if not math.isfinite(self.response_safety_factor) or self.response_safety_factor < 1:
            raise ValueError("Response safety factor must be finite and at least one")
        if self.response_min_samples > self.response_history_window:
            raise ValueError("Response sample minimum exceeds the history window")
        if not math.isfinite(self.growth_factor) or not 1 < self.growth_factor <= 2:
            raise ValueError("Capacity growth_factor must be in (1, 2]")
        if not math.isfinite(self.max_observation_age_s) or self.max_observation_age_s <= 0:
            raise ValueError("Invalid capacity observation age")


def engine_capacity(worker):
    """Read only CPU configuration; no device synchronization or tensor copy.

    The scheduler shares its block pool across cache groups. A group's maximum
    resident page demand includes Mamba none/align/all and speculative states.
    Use the engine's max_memory_usage_bytes contract rather than inferring SSM
    capacity from attention token counts. The full-context profile remains the
    bootstrap/fallback bound. Full-attention layouts also expose their block
    size for opt-in request-aware token reservations.
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
                    "block_size": int(getattr(spec, "block_size", 0)),
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


class ResponseLengthEstimator:
    """Bounded completion history, including capped responses as long samples."""

    def __init__(self, config):
        self.config = config
        self.values = deque(maxlen=config.response_history_window)
        self.ewma = float(config.response_prior_tokens)
        self.seeded = 0
        self.revision = 0
        self.ordered = None
        self.prefix_sum = None

    def observe(self, tokens):
        if type(tokens) is not int or tokens < 1:
            return
        self.ewma = (
            tokens
            if not self.values
            else self.config.response_ewma_alpha * tokens + (1 - self.config.response_ewma_alpha) * self.ewma
        )
        self.values.append(tokens)
        self.ordered = None
        self.revision += 1

    def seed(self, path):
        # Read append-only trajectory diagnostics, never model data or KV tensors.
        with Path(path).open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # A live writer may not have finished the last line.
                if isinstance(row, dict) and row.get("event") == "arrived":
                    self.observe(row.get("response_tokens"))
        self.seeded = len(self.values)
        if self.values:
            self.ewma = sum(self.values) / len(self.values)
            self.revision += 1

    def calibrate(self, step, records):
        """Anchor predictions to consumed batches without re-observing completions."""
        if type(step) is not int or step < 1:
            raise ValueError("Calibration step must be positive")
        if step <= getattr(self, "last_calibration_step", 0):
            return False
        grouped = {}
        for version, tokens in records:
            if type(version) is not int or not 0 <= version < step or type(tokens) is not int or tokens < 1:
                raise ValueError("Invalid behavior version or response length in calibration batch")
            grouped.setdefault(version, []).append(tokens)
        if not grouped:
            return False
        updated = dict(getattr(self, "calibrations", {}))
        for version, tokens in grouped.items():
            values = (list(updated.get(version, {}).get("values", ())) + tokens)[-self.config.response_history_window :]
            ordered = sorted(values)
            updated[version] = {
                "values": tuple(values),
                "ordered": ordered,
                "prefix_sum": list(accumulate(ordered, initial=0)),
            }
        keep = self.config.response_calibration_versions
        self.calibrations = {v: updated[v] for v in sorted(updated)[-keep:]}
        self.last_calibration_step = step
        self.revision += 1
        return True

    def _batch_estimate(self, generated, version):
        calibrations = getattr(self, "calibrations", {})
        candidates = [
            v
            for v in calibrations
            if (version is None or v <= version) and len(calibrations[v]["ordered"]) >= self.config.response_min_samples
        ]
        if not candidates:
            return None
        data = calibrations[max(candidates)]
        ordered = data["ordered"]
        if len(ordered) < self.config.response_min_samples:
            return None
        start = bisect_right(ordered, generated)
        count = len(ordered) - start
        if count:
            quantile = ordered[start + math.ceil(count * self.config.response_quantile) - 1]
            mean = (data["prefix_sum"][-1] - data["prefix_sum"][start]) / count
            return max(mean, quantile)
        mean = data["prefix_sum"][-1] / len(ordered)
        return generated + max(self.config.telemetry_guard_tokens, mean / 4)

    def remaining(self, generated, budget, version=None):
        if generated >= budget:
            return 0
        if len(self.values) < self.config.response_min_samples:
            estimate = max(self.config.response_prior_tokens, self.ewma)
        else:
            # Conditional survivors avoid treating an already-long request as
            # an average fresh request with almost no remaining KV demand.
            if self.ordered is None:
                self.ordered = sorted(self.values)
                self.prefix_sum = list(accumulate(self.ordered, initial=0))
            start = bisect_right(self.ordered, generated)
            count = len(self.ordered) - start
            if count:
                quantile = self.ordered[start + math.ceil(count * self.config.response_quantile) - 1]
                mean = (self.prefix_sum[-1] - self.prefix_sum[start]) / count
                estimate = max(self.ewma, mean, quantile)
            else:
                estimate = generated + max(self.config.telemetry_guard_tokens, self.ewma / 4)
        if getattr(self.config, "response_step_calibration", False):
            baseline = self._batch_estimate(generated, version)
            if baseline is not None:
                weight = self.config.response_calibration_online_weight
                estimate = (1 - weight) * baseline + weight * estimate
        if estimate <= generated:
            estimate = generated + max(self.config.telemetry_guard_tokens, self.ewma / 4)
        remaining = max(self.config.telemetry_guard_tokens, estimate - generated)
        return min(budget - generated, math.ceil(remaining * self.config.response_safety_factor))

    def status(self):
        return {
            "samples": len(self.values),
            "seeded": self.seeded,
            "ewma_tokens": self.ewma,
            "mean_tokens": sum(self.values) / len(self.values) if self.values else self.config.response_prior_tokens,
            "estimated_response_tokens": self.remaining(0, 1 << 30),
            "calibration_step": getattr(self, "last_calibration_step", 0),
            "calibration_versions": sorted(getattr(self, "calibrations", {})),
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
        self.profile_available = True
        self.responses = ResponseLengthEstimator(self.config)
        self.observed_usage = 0.0
        self.backend_ids = set()
        self.generated_tokens = {}
        self.decode_rates = deque(maxlen=128)
        self.rate_previous = None
        self.rate_batch = 0
        self.rate_context = 0
        self.rate_revision = 0
        self.decode_rate_cache = {}
        self.pressure_paused = False
        self.pressure_preempted = False
        self.reason = "configured_initial"
        self.updates = 0

    def token_budget_enabled(self):
        workers = self.pool.get("workers", [])
        return bool(
            self.config.token_budget_admission
            and self.pool.get("available")
            and workers
            and all(
                w.get("groups")
                and all(g.get("type") == "FullAttentionSpec" and g.get("block_size", 0) > 0 for g in w["groups"])
                for w in workers
            )
        )

    def fraction_for_tokens(self, tokens):
        if not self.token_budget_enabled():
            return self.pool.get("request_fraction", math.inf)
        return max(
            sum(math.ceil(max(1, tokens) / g["block_size"]) for g in w["groups"]) / w["total_blocks"]
            for w in self.pool["workers"]
        )

    def fresh(self, now):
        return (
            self.profile_available
            and math.isfinite(self.last_observed)
            and 0 <= now - self.last_observed <= self.config.max_observation_age_s
        )

    def reset_observation(self, capacity_profile=None):
        """Bootstrap a drained engine; revalidate only an unchanged KV layout.

        An idle backend may stop emitting usage ticks after a transient missing
        profile. Its current status can still confirm the profiled layout,
        without treating stale usage as a fresh measurement.
        """
        if capacity_profile == self.pool and self.token_budget_enabled():
            self.profile_available = True
        self.last_observed = -math.inf
        self.observed_usage = 0.0
        self.backend_ids.clear()
        self.generated_tokens.clear()
        self.decode_rates.clear()
        self.rate_previous = None
        self.rate_revision += 1
        self.decode_rate_cache.clear()
        self.pressure_paused = False
        self.pressure_preempted = False
        self.healthy = 0
        self.reason = "drained_engine_bootstrap"

    def _observe_decode_rate(self, telemetry, now):
        """Measure visible per-sequence progress, conditional on batch and context.

        Exclude completed/new/restarted requests from deltas. Kernel timings are
        an upper speed bound when available, including for a newly loaded pool.
        """
        sampled = telemetry.get("sampled_at", now)
        tokens = telemetry.get("generated_tokens", {})
        batch = max(1, telemetry.get("running", 0), len(tokens))
        roofline = telemetry.get("roofline", {})
        context = int(roofline.get("max_context_tokens", 0) // self.config.context_bucket_tokens)
        self.rate_batch, self.rate_context = batch, context
        if self.rate_previous is not None:
            previous_at, previous = self.rate_previous
            dt = sampled - previous_at
            if 0 < dt <= self.config.max_observation_age_s * 2:
                rates = sorted((n - previous[k]) / dt for k, n in tokens.items() if k in previous and n > previous[k])
                if rates:
                    self.decode_rates.append(
                        (self.last_observed, batch, context, rates[math.ceil(len(rates) * 0.9) - 1])
                    )
        self.rate_previous = (sampled, dict(tokens))
        for sample in roofline.get("samples", []):
            ms = sample.get("elapsed_ms", 0)
            if sample.get("mode") in ("eager", "graph") and math.isfinite(ms) and ms > 0:
                self.decode_rates.append(
                    (
                        self.last_observed,
                        max(1, sample["batch"]),
                        int(sample.get("max_context_tokens", 0) // self.config.context_bucket_tokens),
                        1000 / ms,
                    )
                )
        self.rate_revision += 1
        self.decode_rate_cache.clear()

    def decode_rate(self, now, batch=None):
        if not self.config.rolling_kv_enabled or not math.isfinite(now):
            return None
        batch = self.rate_batch if batch is None else max(1, batch)
        key = (self.rate_revision, batch, self.rate_context)
        cached = self.decode_rate_cache.get(key)
        if cached is not None and cached[0] <= now <= cached[1]:
            return cached[2]
        values = [row for row in self.decode_rates if 0 <= now - row[0] <= self.config.rolling_kv_rate_ttl_s]
        if not values:
            return None

        # Never extrapolate a slower rate for unobserved larger batches. Select
        # nearby measured shapes, then use an upper quantile and speed margin.
        def distance(row):
            return abs(row[2] - self.rate_context), abs(math.log2(row[1] / max(1, batch)))

        nearest = sorted(values, key=distance)[:32]
        rates = sorted(row[3] for row in nearest)
        result = rates[math.ceil(len(rates) * 0.9) - 1] * self.config.rolling_kv_rate_margin
        if len(self.decode_rate_cache) >= 128:
            self.decode_rate_cache.clear()
        self.decode_rate_cache[key] = (now, min(row[0] + self.config.rolling_kv_rate_ttl_s for row in values), result)
        return result

    def window_tokens(self, remaining, now):
        rate = self.decode_rate(now)
        if rate is None or not self.fresh(now):
            return remaining
        # Include observation age; no speculative credit for future completions.
        horizon = self.config.rolling_kv_horizon_s + max(0, now - self.last_observed)
        return min(remaining, math.ceil(rate * horizon))

    def update_pressure(self, projected, now):
        if not self.config.rolling_kv_enabled or not self.fresh(now):
            return self.pressure_paused
        if (
            self.observed_usage >= self.config.rolling_kv_pressure_threshold
            or self.pressure_preempted
            or (projected is not None and projected >= self.config.kv_usage_limit)
        ):
            self.pressure_paused = True
        elif (
            self.observed_usage <= self.config.rolling_kv_resume_threshold
            and projected is not None
            and projected < self.config.rolling_kv_pressure_threshold
        ):
            self.pressure_paused = False
        return self.pressure_paused

    def request_reserve(self, request, now, incoming=False, version=None):
        if not self.fresh(now) or request.response_budget <= 0:
            return self.pool.get("request_fraction", math.inf)
        budget = request.response_budget or self.pool["max_model_len"]
        observed = self.generated_tokens.get(request.backend_id) if request.backend_id in self.backend_ids else None
        generated = max(request.generated_tokens, request.generated_base + (observed or 0))
        remaining = self.responses.remaining(generated, budget, version=request.version if version is None else version)
        remaining = self.window_tokens(remaining, now)
        # Unknown/pre-engine arrivals, and migration destinations, need their
        # full prefix reserved as well. Fresh visible decode KV is already in usage.
        prefix = request.prompt_tokens + generated if incoming or observed is None or observed == 0 else 0
        return self.fraction_for_tokens(prefix + remaining + self.config.telemetry_guard_tokens)

    def reserved_usage(self, requests, reserved, now):
        if not self.token_budget_enabled():
            return None
        if not self.fresh(now):
            if math.isfinite(self.last_observed) or not self.profile_available:
                return math.inf  # Never expand from stale telemetry.
            return (len(requests) + len(reserved)) * self.pool["request_fraction"]
        return (
            self.observed_usage
            + sum(self.request_reserve(q, now) for q in requests)
            + sum(self.request_reserve(q, now, incoming=True) for q in reserved)
        )

    def projected_admission(self, requests, reserved, request, now):
        base = self.reserved_usage(requests, reserved, now)
        return None if base is None else base + self.request_reserve(request, now, incoming=True)

    def memory_limit(self):
        if not self.pool.get("available"):
            return self.initial
        fraction = (
            self.fraction_for_tokens(
                256 + self.window_tokens(self.responses.remaining(0, self.pool["max_model_len"]), self.last_observed)
            )
            if self.token_budget_enabled()
            else self.pool["request_fraction"]
        )
        return min(
            self.ceiling,
            self.pool["engine_limit"],
            math.floor(self.config.kv_usage_limit / fraction),
        )

    def update(self, telemetry, roofline, now):
        old = self.status()
        observed = telemetry.get("observed_at", 0)
        usage = telemetry.get("kv_usage", math.nan)
        if (
            not isinstance(observed, int | float)
            or not math.isfinite(observed)
            or observed <= self.last_observed
            or observed > now
            or now - observed > self.config.max_observation_age_s
            or not isinstance(usage, int | float)
            or not math.isfinite(usage)
            or not 0 <= usage <= 1
        ):
            self.healthy = 0
            self.reason = "stale_or_invalid_telemetry"
            self.current = min(self.current, self.initial)
            return old != self.status()
        self.last_observed = observed
        self.observed_usage = usage
        self.backend_ids = set(telemetry.get("request_ids", []))
        self.generated_tokens = dict(telemetry.get("generated_tokens", {}))
        if self.config.rolling_kv_enabled:
            self._observe_decode_rate(telemetry, now)
        data = telemetry.get("roofline", {})
        new_pool = data.get("capacity", {"available": False})
        if self.config.token_budget_admission and not new_pool.get("available") and self.pool.get("available"):
            self.profile_available = False
            self.current, self.healthy, self.reason = min(self.current, self.initial), 0, "capacity_unavailable"
            return old != self.status()
        self.pool = new_pool
        self.profile_available = bool(self.pool.get("available"))
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
        self.pressure_preempted = total_preemptions > (self.preemptions or 0)
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
            "token_budget_admission": self.token_budget_enabled(),
            "rolling_kv_enabled": self.config.rolling_kv_enabled,
            "rolling_kv_horizon_s": self.config.rolling_kv_horizon_s,
            "decode_tokens_per_sequence_s": self.decode_rate(self.last_observed),
            "pressure_paused": self.pressure_paused,
            "kv_headroom_fraction": 1 - self.config.kv_usage_limit,
            "response_history": self.responses.status(),
        }
