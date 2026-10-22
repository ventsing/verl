# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded online decode calibration; a missing knee is a valid result.

Fits are conditional on execution mode and maximum-context interval. This is an
empirical service-time knee, not proof of a hardware FLOP/HBM transition. No
training/model weights or calibration samples are persisted by this module.
"""

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class RooflineConfig:
    enabled: bool = True
    calibration_rounds: int = 3
    sample_every: int = 8
    coverage_sample_every: int = 1
    warmup_per_shape: int = 2
    max_samples: int = 16384
    max_samples_per_bin: int = 128
    max_pending_events: int = 64
    context_bucket_tokens: int = 4096
    min_samples_per_bin: int = 16
    min_batch_sizes: int = 6
    min_side_sizes: int = 2
    block_steps: int = 128
    min_blocks: int = 8
    bootstrap_samples: int = 64
    min_rss_improvement: float = 0.35
    min_slope_ratio: float = 2.0
    max_ci_fraction: float = 0.25
    kneedle_sensitivity: float = 1.0
    latency_tolerance: float = 0.25
    stable_fits: int = 2
    fit_interval_s: float = 10.0

    def __post_init__(self):
        positive = (
            "calibration_rounds",
            "sample_every",
            "coverage_sample_every",
            "max_samples",
            "max_samples_per_bin",
            "max_pending_events",
            "context_bucket_tokens",
            "min_samples_per_bin",
            "min_batch_sizes",
            "min_side_sizes",
            "block_steps",
            "min_blocks",
            "bootstrap_samples",
            "stable_fits",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in positive):
            raise ValueError("Roofline counts and capacities must be positive integers")
        if not 0 <= self.warmup_per_shape or self.min_batch_sizes < 2 * self.min_side_sizes + 1:
            raise ValueError("Roofline needs warmup and samples on both sides of an interior knee")
        if self.coverage_sample_every > self.sample_every:
            raise ValueError("Roofline coverage sampling cannot be sparser than periodic sampling")
        if self.max_samples_per_bin < self.min_samples_per_bin:
            raise ValueError("Roofline bin capacity is below its minimum sample count")
        for name in ("fit_interval_s", "kneedle_sensitivity", "max_ci_fraction", "latency_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid roofline {name}")
        if not 0 < self.min_rss_improvement < 1 or self.min_slope_ratio <= 1:
            raise ValueError("Roofline fit thresholds must identify a slope increase")

    @classmethod
    def from_dict(cls, values=None):
        return cls(**dict(values or {}))


def kneedle(x, y, sensitivity=1.0):
    """Concave increasing throughput: normalized difference and a crossing.

    Endpoints, a line, and a curve that has not dropped below the local maximum's
    threshold produce no knee. This is distinct from a marginal-gain cutoff.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.any(np.diff(x) <= 0) or np.ptp(y) <= 1e-12:
        return None
    if np.any(np.diff(y) < -0.05 * np.ptp(y)):
        return None
    x = (x - x[0]) / np.ptp(x)
    y = (np.maximum.accumulate(y) - y[0]) / np.ptp(y)
    difference = y - x
    threshold_drop = sensitivity * float(np.mean(np.diff(x)))
    candidates = [
        i
        for i in range(1, len(x) - 1)
        if difference[i] > 1e-6
        and difference[i] >= difference[i - 1]
        and difference[i] >= difference[i + 1]
        and np.any(difference[i + 1 :] < difference[i] - threshold_drop)
    ]
    return max(candidates, key=lambda i: difference[i]) if candidates else None


def _nonnegative_slopes(matrix, targets):
    # Tiny active-set NNLS (intercept free, all slopes nonnegative). Enumerating
    # <=8 active sets avoids scipy/sklearn on worker/trainer environments.
    best = None
    for mask in itertools.product((False, True), repeat=matrix.shape[1] - 1):
        columns = [0, *[i + 1 for i, active in enumerate(mask) if active]]
        coefficients = np.linalg.lstsq(matrix[:, columns], targets, rcond=None)[0]
        if coefficients[0] < 0 or np.any(coefficients[1:] < -1e-10):
            continue
        full = np.zeros(matrix.shape[1])
        full[columns] = np.maximum(coefficients, 0)
        residual = targets - matrix @ full
        rss = float(residual @ residual)
        if best is None or rss < best[0]:
            best = rss, full
    return best


def _segmented(points, config):
    b = np.array([p["batch"] for p in points], dtype=float)
    c = np.array([p["context_tokens"] for p in points], dtype=float) / config.context_bucket_tokens
    t = np.array([p["median_ms"] for p in points], dtype=float)
    design = np.column_stack((np.ones(len(b)), b, c))
    null = _nonnegative_slopes(design, t)
    if null is None or null[0] < 1e-10:
        return None
    candidates = []
    for index in range(config.min_side_sizes, len(b) - config.min_side_sizes):
        hinge = np.column_stack((design, np.maximum(b - b[index], 0)))
        fit = _nonnegative_slopes(hinge, t)
        if fit is not None:
            candidates.append((fit[0], int(b[index]), fit[1]))
    if not candidates:
        return None
    rss, knee, coefficients = min(candidates, key=lambda f: f[0])
    improvement = 1 - rss / null[0]
    # Context cost grows with batch even in the low-slope region.
    low_slope = coefficients[1] + coefficients[2] * float(np.median(c / b))
    slope_ratio = (low_slope + coefficients[3]) / max(low_slope, 1e-8)
    bic_gain = len(b) * math.log((null[0] + 1e-12) / (rss + 1e-12)) - 2 * math.log(len(b))
    if improvement < config.min_rss_improvement or slope_ratio < config.min_slope_ratio or bic_gain < 2:
        return None
    return {"knee": knee, "rss_improvement": improvement, "slope_ratio": float(slope_ratio), "bic_gain": bic_gain}


def _summaries(samples, minimum):
    bins = defaultdict(list)
    for sample in samples:
        bins[sample["batch"]].append(sample)
    return [
        {
            "batch": batch,
            "median_ms": float(np.median([s["elapsed_ms"] for s in values])),
            "p95_ms": float(np.quantile([s["elapsed_ms"] for s in values], 0.95)),
            "context_tokens": float(np.median([s["context_tokens"] for s in values])),
            "samples": len(values),
        }
        for batch, values in sorted(bins.items())
        if len(values) >= minimum
    ]


class EmpiricalLatencyGuard:
    """A P95 engineering bound independent of the existence of a batch knee.

    Two fresh stable estimates establish a bound. Once ready within this short
    calibration window it can tighten immediately, but never loosen silently.
    This is empirical evidence, not a distribution-free future latency SLA.
    """

    def __init__(self, config):
        self.config = config
        self.candidates = {}
        self.ready = {}

    def update(self, key, profile, revision):
        points = profile.get("bins", [])
        coverage = profile.get("coverage", {})
        blocks = coverage.get("qualified_independent_blocks", coverage.get("independent_blocks", 0))
        if len(points) < self.config.min_batch_sizes or blocks < self.config.min_blocks:
            return self.ready.get(key, {"status": "insufficient_coverage"})
        baseline = points[0]["p95_ms"]
        # This is a measured operating-point limit, not a monotonicity
        # assertion about every smaller shape. Retain irregular unsafe shapes
        # explicitly instead of letting one transient shape erase throughput.
        safe = [p["batch"] for p in points if p["p95_ms"] <= baseline * (1 + self.config.latency_tolerance)]
        limit = max(safe)
        unsafe = [p["batch"] for p in points if p["batch"] not in safe]
        previous = self.candidates.get(key)
        count = 1
        if previous and previous["limit"] == limit:
            count = previous["count"] + int(revision > previous["revision"])
        self.candidates[key] = {"limit": limit, "count": count, "revision": revision}
        result = {
            "status": "candidate",
            "limit": limit,
            "baseline_batch": points[0]["batch"],
            "baseline_p95_ms": baseline,
            "tolerance": self.config.latency_tolerance,
            "fresh_stable_fits": count,
            "unsafe_observed_batches": unsafe,
            "blocks": blocks,
        }
        retained = self.ready.get(key)
        if retained is not None:
            # A retained numeric cap can itself become an unsafe shape. Keep
            # an actually passing observed point below the old cap instead.
            supported = [batch for batch in safe if batch <= retained["limit"]]
            result.update(status="ready", limit=max(supported) if supported else points[0]["batch"])
        elif count >= self.config.stable_fits:
            result["status"] = "ready"
        if result["status"] == "ready":
            self.ready[key] = result
        return result


class OnlineRoofline:
    """One homogeneous TP replica's samples; no pooling across device layouts."""

    def __init__(self, options=None):
        self.config = RooflineConfig.from_dict(options)
        self.bins = defaultdict(list)
        self.seen = defaultdict(int)
        self.rng = np.random.default_rng(0)
        self.models = {}
        self.candidates = {}
        self.latency_guard = EmpiricalLatencyGuard(self.config)
        self.accepted = 0
        self.rejected = 0
        self.total = 0

    def add(self, samples):
        for sample in samples:
            try:
                b, c, maximum, elapsed = (
                    sample[k] for k in ("batch", "context_tokens", "max_context_tokens", "elapsed_ms")
                )
                mode = sample["mode"]
                if (
                    type(b) is not int
                    or b < 1
                    or not isinstance(mode, str)
                    or mode == "unknown"
                    or not all(math.isfinite(v) and v > 0 for v in (c, maximum, elapsed))
                    or c < maximum
                    or maximum < c / b
                    or sample.get("pure_decode") is not True
                ):
                    raise ValueError("Invalid pure-decode sample")
                key = (mode, int((maximum - 1) // self.config.context_bucket_tokens), b)
                value = dict(sample)
                value["block"] = (sample.get("version", 0), sample["step"] // self.config.block_steps)
                if key not in self.bins and self.total >= self.config.max_samples:
                    continue
                self.seen[key] += 1
                values = self.bins[key]
                if len(values) < self.config.max_samples_per_bin:
                    if self.total >= self.config.max_samples:
                        continue
                    values.append(value)
                    self.total += 1
                else:
                    # Reservoir sampling keeps early/late blocks represented.
                    at = int(self.rng.integers(self.seen[key]))
                    if at < len(values):
                        values[at] = value
                self.accepted += 1
            except (KeyError, TypeError, ValueError):
                self.rejected += 1

    def _estimate(self, samples):
        config = self.config
        points = _summaries(samples, config.min_samples_per_bin)
        counts = defaultdict(int)
        for sample in samples:
            counts[sample["batch"]] += 1
        blocks = {sample["block"] for sample in samples}
        qualified_batches = {point["batch"] for point in points}
        result = {
            "status": "insufficient_coverage",
            "batch_sizes": [p["batch"] for p in points],
            "bins": points,
            "coverage": {
                "observed_batch_sizes": sorted(counts),
                "qualified_batch_sizes": len(points),
                "required_batch_sizes": config.min_batch_sizes,
                "required_samples_per_bin": config.min_samples_per_bin,
                "underfilled_bins": [
                    {"batch": batch, "samples": count, "missing": config.min_samples_per_bin - count}
                    for batch, count in sorted(counts.items())
                    if count < config.min_samples_per_bin
                ],
                "independent_blocks": len(blocks),
                "qualified_independent_blocks": len(
                    {sample["block"] for sample in samples if sample["batch"] in qualified_batches}
                ),
                "required_blocks": config.min_blocks,
            },
        }
        if len(points) < config.min_batch_sizes:
            return result
        fit = _segmented(points, config)
        if fit is None:
            return {**result, "status": "no_observed_knee"}
        position = kneedle(
            [p["batch"] for p in points], [p["batch"] / p["median_ms"] for p in points], config.kneedle_sensitivity
        )
        if position is None:
            return {**result, **fit, "status": "kneedle_unconfirmed"}
        second = points[position]["batch"]
        if abs(second - fit["knee"]) > max(2, config.max_ci_fraction * fit["knee"]):
            return {**result, **fit, "kneedle": second, "status": "estimators_disagree"}
        blocks = defaultdict(list)
        for sample in samples:
            blocks[sample["block"]].append(sample)
        if len(blocks) < config.min_blocks:
            return {**result, **fit, "status": "insufficient_independent_blocks", "blocks": len(blocks)}
        values = list(blocks.values())
        bootstrap = []
        rng = np.random.default_rng(0)
        for _ in range(config.bootstrap_samples):
            draw = [s for i in rng.integers(len(values), size=len(values)) for s in values[i]]
            summaries = _summaries(draw, 1)
            estimate = _segmented(summaries, config) if len(summaries) >= config.min_batch_sizes else None
            if estimate is not None:
                bootstrap.append(estimate["knee"])
        if len(bootstrap) < 0.8 * config.bootstrap_samples:
            return {**result, **fit, "status": "unstable_bootstrap"}
        lower, upper = map(float, np.quantile(bootstrap, [0.025, 0.975]))
        result.update(fit, kneedle=second, ci95=[lower, upper], blocks=len(blocks))
        if upper - lower > max(2, config.max_ci_fraction * fit["knee"]):
            return {**result, "status": "wide_confidence_interval"}
        baseline = points[0]["p95_ms"]
        latency_limit = points[0]["batch"]
        for point in points:
            if point["p95_ms"] > baseline * (1 + config.latency_tolerance):
                break
            latency_limit = point["batch"]
        limit = min(math.floor(lower), latency_limit)
        supported = [p["batch"] for p in points if p["batch"] <= limit]
        if not supported:
            return {**result, "status": "no_latency_safe_batch"}
        return {
            **result,
            "status": "candidate",
            "limit": max(supported),
            "latency_limit": latency_limit,
            "baseline_batch": points[0]["batch"],
            "baseline_p95_ms": baseline,
        }

    def fit(self):
        groups = defaultdict(list)
        for (mode, bucket, _), values in self.bins.items():
            groups[(mode, bucket)].extend(values)
        for key, samples in groups.items():
            if self.models.get(key, {}).get("status") == "ready":
                continue  # Freeze a confident calibration within this window.
            revision = sum(self.seen[k] for k in self.seen if k[:2] == key)
            result = self._estimate(samples)
            result["latency_guard"] = self.latency_guard.update(key, result, revision)
            if result["status"] == "candidate":
                previous = self.candidates.get(key)
                count = previous["count"] + 1 if previous and abs(previous["limit"] - result["limit"]) <= 1 else 1
                self.candidates[key] = {"count": count, "limit": result["limit"], "samples": revision}
                # A second fit must include new observations, not the same points.
                if previous and revision <= previous["samples"]:
                    count = previous["count"]
                    self.candidates[key]["count"] = count
                if count >= self.config.stable_fits:
                    result["status"] = "ready"
            else:
                # An intervening rejected fit invalidates a stability streak.
                self.candidates.pop(key, None)
            self.models[key] = result
        return self.status()

    def calibrated_limit(self, mode, max_context_tokens):
        """Return a measured bound for this exact context stratum, if ready."""
        if max_context_tokens <= 0:
            return None
        key = (mode, int((max_context_tokens - 1) // self.config.context_bucket_tokens))
        model = self.models.get(key, {})
        limits = []
        if model.get("status") == "ready":
            limits.append(model["limit"])
        guard = model.get("latency_guard", {})
        if guard.get("status") == "ready":
            limits.append(guard["limit"])
        return min(limits) if limits else None

    def limit(self, mode, max_context_tokens, fallback):
        measured = self.calibrated_limit(mode, max_context_tokens)
        return min(fallback, measured) if measured is not None else fallback

    def status(self):
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "retained_samples": self.total,
            "profiles": [
                {"mode": key[0], "context_bucket": key[1], **model} for key, model in sorted(self.models.items())
            ],
        }
