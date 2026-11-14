# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Calibration identifiability, bounded telemetry and projected repack limits."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from verl.experimental.trajectory_async.laminar_controller import LaminarController, consolidation_plan
from verl.experimental.trajectory_async.roofline import OnlineRoofline, RooflineConfig, kneedle
from verl.experimental.trajectory_async.roofline_worker import DecodeStepProbe, TPRooflineJoin


def observations(blocks=20, maximum=24, knee=8, context=2048, noise=0.015):
    rng = np.random.default_rng(42)
    result = []
    for block in range(blocks):
        for batch in range(1, maximum + 1):
            for repeat in range(3):
                mean_context = context + int(rng.integers(0, 10))
                elapsed = max(10 + 0.005 * batch, (10 / knee) * batch) if knee else 10 + 0.005 * batch
                result.append(
                    {
                        "batch": batch,
                        "context_tokens": batch * mean_context,
                        "max_context_tokens": mean_context,
                        "elapsed_ms": elapsed + float(rng.normal(0, noise)),
                        "step": block * 128 + batch * 3 + repeat,
                        "version": 0,
                        "mode": "eager",
                        "pure_decode": True,
                        "host_enqueue_ms": 0.05,
                    }
                )
    return result


def test_known_knee_requires_fresh_observations_and_never_extrapolates():
    estimator = OnlineRoofline({"bootstrap_samples": 16})
    estimator.add(observations())
    first = estimator.fit()["profiles"][0]
    assert first["status"] == "candidate", first
    assert first["knee"] == 8
    assert estimator.limit("eager", 2050, 32) == 32
    assert estimator.fit()["profiles"][0]["status"] == "candidate"
    extra = observations(blocks=2)
    for sample in extra:
        sample["step"] += 4096
    estimator.add(extra)
    ready = estimator.fit()["profiles"][0]
    assert ready["status"] == "ready", ready
    assert ready["limit"] == 8
    assert estimator.limit("eager", 2050, 32) == 8
    assert estimator.limit("eager", 2050, 4) == 4
    assert estimator.limit("full", 2050, 32) == 32
    assert estimator.limit("eager", 16000, 32) == 32


@pytest.mark.parametrize("maximum,knee", [(24, None), (8, 32), (4, 2)])
def test_no_knee_or_no_coverage_retains_configured_capacity(maximum, knee):
    estimator = OnlineRoofline({"bootstrap_samples": 8, "stable_fits": 1})
    estimator.add(observations(maximum=maximum, knee=knee))
    profiles = estimator.fit()["profiles"]
    assert profiles and all(p["status"] != "ready" for p in profiles)
    assert estimator.limit("eager", 2050, 8) == 8


def test_context_cost_is_not_misreported_as_a_batch_knee():
    samples = observations(noise=0)
    for sample in samples:
        b = sample["batch"]
        context = 1000 + 60 * b
        sample.update(
            context_tokens=b * context, max_context_tokens=context, elapsed_ms=5 + 0.003 * b + 0.0003 * b * context
        )
    estimator = OnlineRoofline({"stable_fits": 1})
    estimator.add(samples)
    assert all(p["status"] != "ready" for p in estimator.fit()["profiles"])


def test_adjacent_steps_are_not_counted_as_independent_blocks():
    samples = observations()
    for sample in samples:
        sample["step"] = 1
    estimator = OnlineRoofline({"stable_fits": 1})
    estimator.add(samples)
    profile = estimator.fit()["profiles"][0]
    assert profile["status"] == "insufficient_independent_blocks", profile


def test_medians_do_not_hide_a_tail_latency_violation():
    samples = observations()
    for sample in samples:
        if sample["batch"] >= 4 and sample["step"] % 5 == 0:
            sample["elapsed_ms"] *= 3
    estimator = OnlineRoofline({"stable_fits": 1, "bootstrap_samples": 8})
    estimator.add(samples)
    profile = estimator.fit()["profiles"][0]
    assert profile["status"] == "ready", profile
    assert profile["limit"] < 8


def test_sample_budget_and_invalid_samples_are_bounded():
    estimator = OnlineRoofline({"max_samples": 32, "max_samples_per_bin": 16})
    estimator.add(observations(blocks=2))
    for index in range(1000):
        sample = observations(blocks=1, maximum=1)[0]
        sample.update(max_context_tokens=4096 * (index + 1), context_tokens=4096 * (index + 1))
        estimator.add([sample])
    assert estimator.total <= 32
    assert len(estimator.bins) <= 32
    before = estimator.rejected
    for bad in ({}, {**observations(blocks=1, maximum=1)[0], "elapsed_ms": math.nan}):
        estimator.add([bad])
    assert estimator.rejected == before + 2


def test_kneedle_does_not_invent_an_endpoint_or_linear_knee():
    assert kneedle([1, 2, 3, 4], [1, 2, 3, 4]) is None
    assert kneedle([1, 2, 3], [5, 5, 5]) is None
    assert kneedle([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 8, 8]) == 3


@pytest.mark.parametrize("options", [{"sample_every": 0}, {"min_batch_sizes": 2}, {"latency_tolerance": math.nan}])
def test_invalid_calibration_options_fail_early(options):
    with pytest.raises(ValueError):
        RooflineConfig.from_dict(options)


class Event:
    def __init__(self, device):
        self.device = device
        self.timestamp = None

    def record(self):
        self.timestamp = self.device.clock

    def query(self):
        return self.device.ready

    def elapsed_time(self, end):
        assert self.query() and end.query()
        return end.timestamp - self.timestamp


class Device:
    def __init__(self):
        self.clock = 0
        self.ready = False

    def Event(self, enable_timing):
        assert enable_timing
        return Event(self)


def probe_rig(options=None):
    device = Device()
    runner = SimpleNamespace(
        requests={"r": SimpleNamespace(num_prompt_tokens=10)}, model_config=SimpleNamespace(enforce_eager=True)
    )
    worker = SimpleNamespace(model_runner=runner, rank=0)
    context = SimpleNamespace(
        cudagraph_runtime_mode=SimpleNamespace(name="NONE"), batch_descriptor=SimpleNamespace(num_tokens=1)
    )
    counters = SimpleNamespace(num_cudagraph_captured=0, num_backend_compilations=0)
    probe = DecodeStepProbe(
        worker,
        {"sample_every": 1, "warmup_per_shape": 0, **(options or {})},
        device_api=device,
        context_reader=lambda: context,
        counters=counters,
    )
    probe.set_version(0)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"r": 1},
        scheduled_new_reqs=[],
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], num_computed_tokens=[20]),
    )
    return probe, device, scheduled, counters


def timed_step(probe, device, scheduled, duration=6):
    probe.begin(scheduled)
    probe.annotate()
    device.clock += duration
    probe.finish()


def test_device_clock_is_polled_later_without_synchronizing():
    probe, device, scheduled, _ = probe_rig()
    timed_step(probe, device, scheduled)
    assert probe.drain()["samples"] == []
    assert len(probe.pending) == 1
    device.ready = True
    samples = probe.drain()["samples"]
    assert len(samples) == 1 and samples[0]["elapsed_ms"] == 6
    assert samples[0]["context_tokens"] == 21
    assert probe.drain()["samples"] == []


@pytest.mark.parametrize("kind", ["new", "chunked", "last_prefill_token", "speculative", "encoder"])
def test_only_unambiguous_pure_decode_is_timed(kind):
    probe, device, scheduled, _ = probe_rig()
    if kind == "new":
        scheduled.scheduled_new_reqs = [object()]
    elif kind == "chunked":
        scheduled.num_scheduled_tokens = {"r": 8}
    elif kind == "last_prefill_token":
        scheduled.scheduled_cached_reqs.num_computed_tokens = [9]
    elif kind == "speculative":
        scheduled.scheduled_spec_decode_tokens = {"r": [4]}
    else:
        scheduled.scheduled_encoder_inputs = {"r": [0]}
    timed_step(probe, device, scheduled)
    assert probe.current is None and not probe.pending


def test_capture_and_first_shape_samples_are_discarded():
    probe, device, scheduled, counters = probe_rig({"warmup_per_shape": 1})
    timed_step(probe, device, scheduled)
    assert not probe.pending
    probe.begin(scheduled)
    probe.annotate()
    counters.num_cudagraph_captured += 1
    probe.finish()
    assert not probe.pending
    timed_step(probe, device, scheduled)
    device.ready = True
    assert len(probe.drain()["samples"]) == 1


def test_probe_stops_after_local_policy_window_and_pending_events_are_bounded():
    probe, device, scheduled, _ = probe_rig({"max_pending_events": 2})
    for _ in range(10):
        timed_step(probe, device, scheduled)
    assert len(probe.pending) == 2
    for version in (1, 2, 3):
        probe.set_version(version)
    assert not probe.enabled


def packet(rank, elapsed=10, context=2048):
    sample = observations(blocks=1, maximum=1)[0]
    sample.update(elapsed_ms=elapsed, context_tokens=context, max_context_tokens=context)
    return {"rank": rank, "samples": [sample]}


def test_tp_uses_slowest_matching_rank_and_waits_for_incomplete_steps():
    join = TPRooflineJoin(2)
    assert join.add([packet(0, 10)]) == []
    result = join.add([packet(1, 15)])
    assert len(result) == 1 and result[0]["elapsed_ms"] == 15
    assert not join.pending
    assert join.add([packet(0), packet(1, context=3000)]) == []
    assert join.rejected == 1


def test_repack_projects_context_and_applies_destination_limit():
    snapshots = [
        {"id": 0, "usage": 0.1, "count": 2, "context_tokens": 20000},
        {"id": 1, "usage": 0.2, "count": 2, "context_tokens": 2000},
    ]
    seen = []

    def limit(destination, context):
        seen.append(context)
        return 3 if context >= 16000 else 8

    assert consolidation_plan(snapshots, 8, request_limit=limit) == []
    assert seen and all(context == 20000 for context in seen)
    assert consolidation_plan(snapshots, 8, request_limit=lambda *_: 4) == [(0, 1)]


def test_controller_keeps_engine_admission_separate_from_repack_limit():
    controller = LaminarController([], None, max_running=16, repack_max_running=4)
    assert controller.max_running == 16 and controller._repack_limit(0, {}) == 4
    assert controller.status()["repack_max_running"] == 4


def test_new_samples_in_another_context_do_not_validate_a_stale_candidate():
    estimator = OnlineRoofline({"bootstrap_samples": 8})
    estimator.add(observations())
    assert estimator.fit()["profiles"][0]["status"] == "candidate"
    estimator.add(observations(context=16000))
    profiles = estimator.fit()["profiles"]
    assert all(p["status"] == "candidate" for p in profiles)


def test_repack_off_still_drains_calibration_without_reserving_or_aborting():
    import asyncio
    import time

    from test_laminar_on_cpu import Handle

    class Server:
        def trajectory_status(self):
            return {"kv_usage": 0.1, "waiting": 0, "request_ids": [], "observed_at": time.time()}

        def trajectory_roofline(self):
            return {"available": True, "samples": observations(), "mode": "eager", "max_context_tokens": 2050}

    async def scenario():
        controller = LaminarController(
            [{"server": Handle(Server()), "workers": []}],
            None,
            enable_repack=False,
            roofline_options={"enabled": True, "bootstrap_samples": 8},
        )
        assert await controller.repack_once(perform_moves=False) == 0
        assert controller.roofline[0].accepted > 0
        assert controller.metrics["repack_checks"] == 0
        assert controller.metrics["roofline_polls"] == 1
        assert not controller.replicas[0].reserved

    asyncio.run(scenario())


def test_short_lived_batch_bins_are_covered_before_periodic_thinning():
    probe, device, scheduled, _ = probe_rig({"sample_every": 8})
    device.ready = True
    # This shape exists for only 24 steps; periodic sampling would collect 3.
    for _ in range(24):
        timed_step(probe, device, scheduled)
    samples = probe.drain()["samples"]
    assert len(samples) == 24
    assert probe.counts["coverage_attempts"] == 24
    # A covered bin returns to the configured sparse rate.
    for _ in range(16):
        timed_step(probe, device, scheduled)
    assert probe.counts["coverage_attempts"] == 32
    assert probe.counts["periodic_attempts"] == 1
    # Entering another context interval activates a separate bounded phase.
    scheduled.scheduled_cached_reqs.num_computed_tokens = [8192]
    before = probe.counts["coverage_attempts"]
    for _ in range(4):
        timed_step(probe, device, scheduled)
    assert probe.counts["coverage_attempts"] == before + 4


def test_tp_coverage_schedule_does_not_depend_on_local_device_progress():
    fast, fast_device, scheduled, _ = probe_rig({"sample_every": 8, "max_pending_events": 2})
    slow, slow_device, _, _ = probe_rig({"sample_every": 8, "max_pending_events": 2})
    fast_device.ready = True
    slow_device.ready = False
    for _ in range(80):
        timed_step(fast, fast_device, scheduled)
        timed_step(slow, slow_device, scheduled)
    assert fast.coverage_attempts == slow.coverage_attempts
    assert fast.counts["coverage_attempts"] == slow.counts["coverage_attempts"] == 32
    assert fast.counts["periodic_attempts"] == slow.counts["periodic_attempts"]
    assert len(slow.pending) == 2
    assert slow.counts["pending_limit"] > 0


@pytest.mark.parametrize("options", [{"coverage_sample_every": 0}, {"coverage_sample_every": 9}])
def test_coverage_sampling_configuration_is_bounded(options):
    with pytest.raises(ValueError):
        RooflineConfig.from_dict(options)


def test_sparse_coverage_diagnostics_explain_missing_bins_and_blocks():
    samples = observations(blocks=1, maximum=3)
    estimator = OnlineRoofline()
    estimator.add(samples)
    profile = estimator.fit()["profiles"][0]
    assert profile["status"] == "insufficient_coverage"
    assert profile["coverage"]["observed_batch_sizes"] == [1, 2, 3]
    assert profile["coverage"]["qualified_batch_sizes"] == 0
    assert all(b["missing"] == 13 for b in profile["coverage"]["underfilled_bins"])
    assert profile["coverage"]["independent_blocks"] == 1
    assert estimator.limit("eager", 2050, 32) == 32


def test_rejected_fit_breaks_consecutive_stability_streak(monkeypatch):
    estimator = OnlineRoofline()
    estimator.add(observations())
    outputs = iter(
        [
            {"status": "candidate", "limit": 8},
            {"status": "unstable_bootstrap"},
            {"status": "candidate", "limit": 8},
        ]
    )
    monkeypatch.setattr(estimator, "_estimate", lambda _: next(outputs))
    assert estimator.fit()["profiles"][0]["status"] == "candidate"
    assert estimator.fit()["profiles"][0]["status"] == "unstable_bootstrap"
    estimator.add(observations(blocks=1))
    assert estimator.fit()["profiles"][0]["status"] == "candidate"
    assert estimator.limit("eager", 2050, 32) == 32


def test_pending_samples_are_fitted_after_throttle_even_when_worker_stops(monkeypatch):
    import asyncio

    from test_laminar_on_cpu import Handle

    from verl.experimental.trajectory_async import laminar_controller

    clock = [10.0]

    class Server:
        samples = []

        def trajectory_roofline(self):
            samples, self.samples = self.samples, []
            return {"available": True, "samples": samples}

    async def scenario():
        server = Server()
        controller = LaminarController(
            [{"server": Handle(server), "workers": []}],
            None,
            roofline_options={"enabled": True, "fit_interval_s": 10, "bootstrap_samples": 8},
        )
        estimator = controller.roofline[0]
        estimator.add(observations())
        assert estimator.fit()["profiles"][0]["status"] == "candidate"
        controller.roofline_last_fit[0] = 9.0
        server.samples = observations(blocks=2)
        telemetry = [{}]
        await controller._ingest_roofline(telemetry)
        assert controller.roofline_dirty == {0}
        assert estimator.status()["profiles"][0]["status"] == "candidate"
        # No further worker samples arrive. The queued revision must still fit.
        clock[0] = 21.0
        await controller._ingest_roofline(telemetry)
        assert controller.roofline_dirty == set()
        assert estimator.status()["profiles"][0]["status"] == "ready"
        assert (
            controller._repack_limit(
                0,
                {
                    0: {
                        "roofline": {
                            "available": True,
                            "mode": "eager",
                            "max_context_tokens": 2050,
                        }
                    }
                },
            )
            == 8
        )

    # Replace only this module's time object; event-loop monotonic time is intact.
    monkeypatch.setattr(laminar_controller, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    asyncio.run(scenario())


def linear_latency_samples(blocks=20, context=2048):
    samples = observations(blocks=blocks, context=context, noise=0)
    for sample in samples:
        sample["elapsed_ms"] = 10 + 0.4 * sample["batch"]
    return samples


def test_p95_bound_applies_without_inventing_a_knee_and_requires_fresh_fits():
    estimator = OnlineRoofline({"bootstrap_samples": 8})
    estimator.add(linear_latency_samples())
    profile = estimator.fit()["profiles"][0]
    assert profile["status"] == "no_observed_knee"
    assert profile["latency_guard"]["status"] == "candidate"
    assert estimator.limit("eager", 2050, 32) == 32
    assert estimator.fit()["profiles"][0]["latency_guard"]["status"] == "candidate"
    extra = linear_latency_samples(blocks=2)
    for sample in extra:
        sample["step"] += 8192
    estimator.add(extra)
    profile = estimator.fit()["profiles"][0]
    assert profile["status"] == "no_observed_knee"
    assert profile["latency_guard"]["status"] == "ready"
    assert profile["latency_guard"]["limit"] == 7
    assert estimator.limit("eager", 2050, 32) == 7


def test_latency_guard_tightens_immediately_after_new_tail_spikes():
    estimator = OnlineRoofline({"stable_fits": 1, "bootstrap_samples": 8})
    estimator.add(linear_latency_samples())
    estimator.fit()
    assert estimator.limit("eager", 2050, 32) == 7
    extra = linear_latency_samples(blocks=3)
    for sample in extra:
        sample["step"] += 8192
        if sample["batch"] >= 4:
            sample["elapsed_ms"] *= 3
    estimator.add(extra)
    profile = estimator.fit()["profiles"][0]
    assert profile["latency_guard"]["status"] == "ready"
    assert estimator.limit("eager", 2050, 32) == 3


def test_latency_samples_in_another_context_cannot_validate_an_old_bound():
    estimator = OnlineRoofline({"bootstrap_samples": 8})
    estimator.add(linear_latency_samples())
    estimator.fit()
    estimator.add(linear_latency_samples(context=16000))
    profiles = estimator.fit()["profiles"]
    assert all(p["latency_guard"]["status"] == "candidate" for p in profiles)
    assert estimator.limit("eager", 2050, 32) == 32


def test_latency_guard_reports_irregular_shapes_without_claiming_monotonic_safety():
    from verl.experimental.trajectory_async.roofline import EmpiricalLatencyGuard

    guard = EmpiricalLatencyGuard(RooflineConfig(stable_fits=1))
    profile = {
        "bins": [{"batch": b, "p95_ms": t} for b, t in [(1, 10), (2, 11), (4, 30), (8, 12), (16, 12), (24, 12)]],
        "coverage": {"independent_blocks": 8},
    }
    result = guard.update(("eager", 0), profile, 100)
    assert result["status"] == "ready" and result["limit"] == 24
    assert result["unsafe_observed_batches"] == [4]


def test_retained_latency_cap_must_still_be_a_currently_passing_shape():
    from verl.experimental.trajectory_async.roofline import EmpiricalLatencyGuard

    guard = EmpiricalLatencyGuard(RooflineConfig(stable_fits=1))
    profile = {
        "bins": [{"batch": b, "p95_ms": t} for b, t in [(1, 10), (2, 11), (4, 12), (8, 12), (16, 12), (24, 30)]],
        "coverage": {"independent_blocks": 8},
    }
    assert guard.update(("eager", 0), profile, 100)["limit"] == 16
    profile["bins"][-2]["p95_ms"] = 30
    profile["bins"][-1]["p95_ms"] = 12
    result = guard.update(("eager", 0), profile, 200)
    assert result["status"] == "ready" and result["limit"] == 8
    assert 16 in result["unsafe_observed_batches"]


def test_continuous_sampling_survives_policy_updates_without_unbounding_events():
    probe, device, scheduled, _ = probe_rig({"continuous_sampling": True, "max_pending_events": 2})
    for version in range(10):
        probe.set_version(version)
        timed_step(probe, device, scheduled)
    assert probe.enabled
    assert len(probe.pending) == 2
