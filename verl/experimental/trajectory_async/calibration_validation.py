# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Acceptance summaries distinguish coverage, knee evidence and uncertain overhead."""

import math

import numpy as np


def paired_overhead(pairs, tolerance=0.05, draws=2000):
    """Bootstrap entire matched on/off runs, not correlated individual steps."""
    if len(pairs) < 4:
        return {"status": "insufficient_pairs", "pairs": len(pairs)}
    ratios = []
    for pair in pairs:
        off, on = pair["off"], pair["on"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (off, on)):
            raise ValueError("A/B latencies must be positive and finite")
        ratios.append(on / off - 1)
    values = np.asarray(ratios)
    rng = np.random.default_rng(0)
    boot = np.median(values[rng.integers(len(values), size=(draws, len(values)))], axis=1)
    low, high = map(float, np.quantile(boot, [0.025, 0.975]))
    status = "passed" if high <= tolerance else "failed" if low > tolerance else "inconclusive"
    return {
        "status": status,
        "pairs": len(pairs),
        "median_overhead": float(np.median(values)),
        "ci95": [low, high],
        "tolerance": tolerance,
        "paired_overheads": ratios,
    }


def decode_wall_latency(rows, batch):
    """Full-batch pure-decode output intervals, excluding prefill boundaries.

    This is client-side service spacing, not device-event elapsed_ms. The
    identical CPU statistics recorder remains enabled in both A/B arms.
    """
    values = []
    previous = None
    for row in rows:
        pure = row["prompt_tokens"] == 0 and row["generation_tokens"] == batch and row["running"] == batch
        if pure and previous is not None:
            elapsed = (row["clock"] - previous) * 1000
            if elapsed > 0:
                values.append(elapsed)
        previous = row["clock"] if pure else None
    if len(values) < 16:
        return {"available": False, "samples": len(values)}
    return {
        "available": True,
        "samples": len(values),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def profile_acceptance(status, context_bucket, required_batches):
    profiles = [p for p in status["profiles"] if p["context_bucket"] == context_bucket]
    if len(profiles) != 1:
        return {"coverage": "incomplete", "reason": "missing_or_mixed_mode", "profiles": profiles}
    profile = profiles[0]
    missing = sorted(set(required_batches) - set(profile.get("batch_sizes", [])))
    blocks = profile.get("coverage", {})
    coverage = not missing and blocks.get("independent_blocks", 0) >= blocks.get("required_blocks", 8)
    return {
        "coverage": "passed" if coverage else "incomplete",
        "missing_batches": missing,
        "knee_outcome": profile["status"],
        "knee_confirmed": profile["status"] == "ready",
        "profile": profile,
    }
