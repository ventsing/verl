# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Actual pool budgets, bounded online growth and destination-specific safety."""

import asyncio
import math
from types import SimpleNamespace

import pytest

from verl.experimental.trajectory_async.capacity import (
    CapacityConfig,
    OnlineCapacity,
    engine_capacity,
    join_engine_capacity,
)
from verl.experimental.trajectory_async.laminar_controller import LaminarController, consolidation_plan
from verl.experimental.trajectory_async.roofline import OnlineRoofline


def test_ready_long_context_recovers_admission_after_sampling_stops():
    governor = OnlineCapacity(initial=8, ceiling=64)
    estimator = OnlineRoofline({})
    estimator.models[("eager", 6)] = {"status": "no_observed_knee", "latency_guard": {"status": "ready", "limit": 28}}
    for i in range(24):
        value = telemetry()
        value["observed_at"] = i + 1.0
        value["running"] = governor.limit()
        value["roofline"].update(collecting=False, max_context_tokens=25000)
        governor.update(value, estimator, now=i + 1.0)
    assert governor.limit() == 28


def test_unmeasured_context_cannot_explore_after_sampling_stops():
    governor = OnlineCapacity(initial=8, ceiling=64)
    estimator = OnlineRoofline({})
    estimator.models[("eager", 0)] = {"status": "ready", "limit": 32}
    for i in range(12):
        value = telemetry()
        value["observed_at"] = i + 1.0
        value["running"] = 8
        value["roofline"].update(collecting=False, max_context_tokens=25000)
        governor.update(value, estimator, now=i + 1.0)
    assert governor.limit() == 8


def pool(fraction=0.02, ceiling=64):
    return {
        "available": True,
        "request_fraction": fraction,
        "engine_limit": ceiling,
        "max_model_len": 26624,
        "workers": [],
    }


def telemetry(at=1.0, **changes):
    value = {
        "observed_at": at,
        "kv_usage": 0.2,
        "waiting": 0,
        "running": 64,
        "preemptions": 0,
        "roofline": {
            "available": True,
            "collecting": True,
            "mode": "eager",
            "max_context_tokens": 2048,
            "capacity": pool(),
        },
    }
    value.update(changes)
    return value


def test_capacity_grows_from_configured_seed_and_never_exceeds_pool_or_engine():
    governor = OnlineCapacity(8, 64)
    for at in range(1, 50):
        governor.update(telemetry(at), None, at)
        assert 1 <= governor.limit() <= 45
    assert governor.limit() == 45
    assert governor.updates > 1
    engine_limited = OnlineCapacity(8, 16)
    for at in range(1, 50):
        engine_limited.update(telemetry(at), None, at)
    assert engine_limited.limit() == 16


@pytest.mark.parametrize("pressure", [{"kv_usage": 0.95}, {"waiting": 1}, {"preemptions": 1}])
def test_scheduler_or_memory_pressure_causes_immediate_backoff(pressure):
    governor = OnlineCapacity(8, 64)
    governor.update(telemetry(**pressure), None, 1)
    assert governor.limit() == 6
    assert governor.reason == "memory_or_scheduler_pressure"


def test_old_or_repeated_telemetry_cannot_authorize_growth():
    governor = OnlineCapacity(8, 64, {"healthy_polls": 1})
    governor.update(telemetry(), None, 1)
    assert governor.limit() == 10
    for _ in range(5):
        governor.update(telemetry(), None, 1)
    assert governor.limit() == 8
    governor.update(telemetry(2), None, 100)
    assert governor.limit() == 8


def test_missing_pool_or_decode_probe_never_drives_exploration():
    governor = OnlineCapacity(8, 64, {"healthy_polls": 1})
    value = telemetry()
    value["roofline"]["capacity"] = {"available": False}
    governor.update(value, None, 1)
    assert governor.limit() == 8
    assert math.isinf(governor.projected_usage(0.1, 2))
    value = telemetry(2)
    value["roofline"]["collecting"] = False
    governor.update(value, None, 2)
    assert governor.limit() == 8


def test_new_context_starts_conservatively_and_confident_roofline_shrinks_capacity():
    governor = OnlineCapacity(8, 64, {"healthy_polls": 1})
    governor.update(telemetry(), None, 1)
    assert governor.limit() == 10
    value = telemetry(2)
    value["roofline"].update(max_context_tokens=8192, collecting=False)
    governor.update(value, None, 2)
    assert governor.limit() == 8
    estimator = OnlineRoofline()
    estimator.models[("eager", 1)] = {"status": "ready", "limit": 4}
    value["observed_at"] = 3
    governor.update(value, estimator, 3)
    assert governor.limit() == 4


class AttentionSpec:
    page_size_bytes = 128

    def max_memory_usage_bytes(self, config):
        return 128 * 10


class MambaSpec:
    page_size_bytes = 128
    mamba_cache_mode = "align"

    def max_memory_usage_bytes(self, config):
        return 128 * 2


def worker(blocks=121, rank=0):
    groups = [
        SimpleNamespace(layer_names=["attn"], kv_cache_spec=AttentionSpec()),
        SimpleNamespace(layer_names=["ssm"], kv_cache_spec=MambaSpec()),
    ]
    return SimpleNamespace(
        rank=rank,
        model_runner=SimpleNamespace(
            kv_cache_config=SimpleNamespace(num_blocks=blocks, kv_cache_groups=groups),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(max_model_len=26624), scheduler_config=SimpleNamespace(max_num_seqs=64)
            ),
        ),
    )


def test_profiled_capacity_counts_attention_and_ssm_in_shared_pool():
    result = engine_capacity(worker())
    assert result["available"]
    assert result["total_blocks"] == 120
    assert result["max_pages_per_request"] == 12
    assert result["groups"][1]["mamba_cache_mode"] == "align"
    governor = OnlineCapacity(8, 64)
    value = telemetry()
    value["roofline"]["capacity"] = join_engine_capacity(
        [
            {"capacity": result},
            {"capacity": engine_capacity(worker(rank=1))},
        ],
        2,
    )
    governor.update(value, None, 1)
    assert governor.memory_limit() == 9  # 90% of 120 shared pages / 12 per request.


def test_tp_pool_join_uses_tightest_rank_and_rejects_missing_or_duplicate_ranks():
    packets = [{"capacity": engine_capacity(worker(rank=0))}, {"capacity": engine_capacity(worker(blocks=61, rank=1))}]
    result = join_engine_capacity(packets, 2)
    assert result["available"] and result["request_fraction"] == pytest.approx(0.2)
    assert not join_engine_capacity(packets[:1], 2)["available"]
    assert not join_engine_capacity([packets[0], packets[0]], 2)["available"]
    packets[1]["capacity"]["groups"][0]["page_bytes"] = 256
    assert not join_engine_capacity(packets, 2)["available"]


def test_unknown_cache_layout_stays_explicitly_unavailable():
    assert not engine_capacity(SimpleNamespace())["available"]


def test_destination_projection_does_not_sum_incompatible_usage_percentages():
    snapshots = [{"id": 0, "usage": 0.1, "count": 2}, {"id": 1, "usage": 0.2, "count": 2}]

    # The source is lightly occupied on a large device, but its requests cannot
    # fit on the smaller destination. Raw 10% + 20% would falsely permit it.
    def projection(target, usage, count):
        return usage + count * (0.5 if target == 1 else 0.01)

    plan = consolidation_plan(snapshots, 8, usage_projection=projection)
    assert (0, 1) not in plan
    assert plan == [(1, 0)]


def test_initial_pool_caps_admission_before_the_first_generated_token():
    controller = LaminarController(
        [{"server": None, "workers": []}],
        None,
        batch_quota=8,
        max_running=64,
        roofline_options={"enabled": True},
        capacity_options={"enabled": True},
        capacity_profiles=[pool(fraction=0.25)],
    )
    controller.replicas[0].version = 0
    assert controller._admission_limit(0) == 3

    async def assign():
        for request_id in ("a", "b", "c"):
            await controller._assign(request_id)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await controller._assign("fourth")
        assert controller.replicas[0].load == 3

    asyncio.run(assign())


def test_disabling_roofline_preserves_configured_legacy_admission():
    controller = LaminarController(
        [], None, batch_quota=8, max_running=64, roofline_options={"enabled": False}, capacity_options={"enabled": True}
    )
    assert not controller.capacity
    assert controller._admission_limit(0) == 8


@pytest.mark.parametrize(
    "options",
    [
        {"healthy_polls": 0},
        {"kv_usage_limit": 1.0},
        {"growth_factor": 1.0},
        {"backoff_factor": math.nan},
        {"load_ratio": 0},
        {"max_observation_age_s": 0},
    ],
)
def test_invalid_capacity_configuration_fails_early(options):
    with pytest.raises(ValueError):
        CapacityConfig(**options)


@pytest.mark.parametrize("boundary", ["growing", "new_version", "migrating", "reserved", "ramp_down", "newly_draining"])
def test_exploration_only_reopens_the_current_healthy_nondraining_batch(boundary):
    from test_laminar_on_cpu import Handle

    from verl.experimental.trajectory_async.laminar_controller import Request

    class Server:
        def trajectory_status(self):
            import time

            return {
                "kv_usage": 0.2,
                "waiting": 0,
                "running": 8,
                "observed_at": time.time(),
                "request_ids": [str(i) for i in range(8)],
            }

        def trajectory_roofline(self):
            return {**telemetry()["roofline"], "samples": []}

    async def scenario():
        controller = LaminarController(
            [{"server": Handle(Server()), "workers": []}],
            None,
            max_running=64,
            roofline_options={"enabled": True},
            capacity_options={"enabled": True, "healthy_polls": 1},
            capacity_profiles=[pool()],
        )
        replica = controller.replicas[0]
        controller.latest = replica.version = 0
        replica.closed, replica.admitted = True, 8
        replica.active = {str(i): Request(str(i), 0, 0, backend_id=str(i)) for i in range(8)}
        if boundary == "new_version":
            controller.latest = 1
        elif boundary == "migrating":
            replica.active["0"].destination = 1
        elif boundary == "reserved":
            replica.reserved.add("incoming")
        elif boundary == "ramp_down":
            replica.ramp_down = True
        elif boundary == "newly_draining":
            replica.previous_usage = 0.3
        await controller.repack_once(perform_moves=False)
        assert controller._admission_limit(0) == 10
        assert replica.closed == (boundary != "growing")

    asyncio.run(scenario())


def test_empirical_latency_limit_clips_capacity_without_a_ready_knee():
    governor = OnlineCapacity(8, 64)
    governor.pool = pool()
    governor.current = 45
    governor.profile = ("eager", 3)
    estimator = OnlineRoofline()
    estimator.models[("eager", 3)] = {
        "status": "no_observed_knee",
        "latency_guard": {"status": "ready", "limit": 40},
    }
    value = telemetry()
    value["roofline"]["max_context_tokens"] = 13312
    governor.update(value, estimator, 1)
    assert governor.memory_limit() == 45
    assert governor.limit() == 40
    assert governor.reason == "calibrated_bound"
