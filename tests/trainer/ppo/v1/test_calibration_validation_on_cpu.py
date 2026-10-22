# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Acceptance cannot turn missing coverage or inconclusive overhead into a pass."""

from types import SimpleNamespace

import pytest

from verl.experimental.trajectory_async.calibration_validation import (
    decode_wall_latency,
    paired_overhead,
    profile_acceptance,
)
from verl.experimental.trajectory_async.roofline_worker import set_decode_probe_sampling


def test_paired_overhead_uses_whole_runs_and_rejects_excessive_cost():
    assert paired_overhead([{"off": 10, "on": 10.1}] * 6)["status"] == "passed"
    assert paired_overhead([{"off": 10, "on": 11}] * 6)["status"] == "failed"
    assert paired_overhead([{"off": 10, "on": value} for value in (9, 9, 9, 12, 12, 12)])["status"] == "inconclusive"
    assert paired_overhead([{"off": 10, "on": 11}])["status"] == "insufficient_pairs"
    with pytest.raises(ValueError):
        paired_overhead([{"off": 0, "on": 1}] * 4)


def test_decode_spacing_excludes_prefill_and_boundary_intervals():
    rows = [{"clock": i * 0.01, "prompt_tokens": 0, "generation_tokens": 4, "running": 4} for i in range(50)]
    rows[10]["prompt_tokens"] = 1000
    rows[20]["generation_tokens"] = 2
    result = decode_wall_latency(rows, 4)
    assert result["available"] and result["samples"] == 45
    assert result["median_ms"] == pytest.approx(10)
    assert not decode_wall_latency(rows[:5], 4)["available"]


def test_complete_coverage_does_not_claim_a_knee_when_none_was_observed():
    profile = {
        "context_bucket": 5,
        "status": "no_observed_knee",
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "coverage": {"independent_blocks": 12, "required_blocks": 8},
    }
    result = profile_acceptance({"profiles": [profile]}, 5, [1, 2, 4, 8, 16, 32])
    assert result["coverage"] == "passed" and not result["knee_confirmed"]
    assert profile_acceptance({"profiles": [profile]}, 5, [1, 2, 4, 8, 16, 32, 45])["coverage"] == "incomplete"
    assert profile_acceptance({"profiles": [profile]}, 4, [1, 2])["coverage"] == "incomplete"


def test_sampling_switch_never_reopens_an_expired_version_window_or_resets_state():
    probe = SimpleNamespace(enabled=False, suspended=False, current=None, error=None, step=99)
    worker = SimpleNamespace(_trajectory_decode_probe=probe)
    assert not set_decode_probe_sampling(worker, True)["enabled"]
    assert probe.step == 99 and not probe.enabled
    probe.enabled = True
    assert set_decode_probe_sampling(worker, False)["suspended"]
    assert set_decode_probe_sampling(worker, True)["enabled"]
    with pytest.raises(ValueError):
        set_decode_probe_sampling(worker, 1)
    probe.current = {}
    with pytest.raises(RuntimeError):
        set_decode_probe_sampling(worker, False)


def test_fresh_test_window_preserves_version_steps_and_refuses_pending_events():
    from collections import Counter, deque

    from probe_online_calibration import fresh_measurement_window

    probe = SimpleNamespace(
        current=None,
        pending=deque(),
        samples=deque(),
        error=None,
        versions={0},
        config=SimpleNamespace(calibration_rounds=3, enabled=True),
        bins=Counter({("eager", 0, 16): 128}),
        coverage_attempts=Counter({(0, 16): 34}),
        counts=Counter(queued=128),
        step=2048,
        version=0,
        enabled=False,
    )
    worker = SimpleNamespace(_trajectory_decode_probe=probe)
    assert fresh_measurement_window(worker) == {"step": 2048, "version": 0}
    assert probe.enabled and not probe.bins and not probe.counts
    probe.pending.append(object())
    with pytest.raises(RuntimeError):
        fresh_measurement_window(worker)
    probe.pending.clear()
    probe.versions = {0, 1, 2, 3}
    with pytest.raises(RuntimeError):
        fresh_measurement_window(worker)
