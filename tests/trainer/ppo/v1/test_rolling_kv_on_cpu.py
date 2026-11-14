# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Windowed KV reservations and pressure evacuation preserve version/capacity fences."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest

from verl.experimental.trajectory_async.capacity import CapacityConfig, OnlineCapacity
from verl.experimental.trajectory_async.laminar_controller import LaminarController, Request


def pool():
    worker = {
        "rank": 0,
        "total_blocks": 4096,
        "max_pages_per_request": 256,
        "max_model_len": 32768,
        "engine_limit": 256,
        "groups": [
            {"type": "FullAttentionSpec", "block_size": 128, "page_bytes": 262144, "max_pages_per_request": 256}
        ],
    }
    return {
        "available": True,
        "request_fraction": 1 / 16,
        "engine_limit": 256,
        "max_model_len": 32768,
        "workers": [worker],
    }


def options(**kwargs):
    return {
        "enabled": True,
        "token_budget_admission": True,
        "rolling_kv_enabled": True,
        "kv_usage_limit": 0.95,
        **kwargs,
    }


def observe(g, at, generated, usage=0.1, **kwargs):
    data = {
        "kv_usage": usage,
        "observed_at": at,
        "sampled_at": at,
        "running": len(generated),
        "waiting": 0,
        "preemptions": 0,
        "generated_tokens": generated,
        "request_ids": list(generated),
        "roofline": {
            "available": True,
            "collecting": True,
            "mode": "eager",
            "max_context_tokens": 4096,
            "capacity": pool(),
        },
        **kwargs,
    }
    g.update(data, None, at)
    return data


def governor(**kwargs):
    return OnlineCapacity(64, 256, options(**kwargs))


@pytest.mark.parametrize(
    "field,value",
    [
        ("rolling_kv_enabled", 1),
        ("rolling_kv_horizon_s", 0),
        ("rolling_kv_horizon_s", float("nan")),
        ("rolling_kv_rate_margin", 0.9),
        ("rolling_kv_rate_ttl_s", 0),
        ("rolling_kv_pressure_threshold", 0.96),
        ("rolling_kv_resume_threshold", 0.91),
        ("rolling_kv_max_moves", 0),
        ("rolling_kv_migration_cooldown_s", -1),
        ("token_budget_admission", False),
    ],
)
def test_invalid_rolling_options(field, value):
    with pytest.raises(ValueError):
        CapacityConfig(**options(**{field: value}))


def test_window_reserves_measured_growth_and_keeps_the_entire_incoming_prefix():
    g = governor()
    observe(g, 10, {"a": 1000})
    q = Request("q", 0, 0, backend_id="a", prompt_tokens=256, response_budget=32768, generated_tokens=1050)
    fallback = g.request_reserve(q, 10)
    observe(g, 15, {"a": 1050})  # 10 tokens/s, with a 25% speed margin.
    assert g.decode_rate(15) == 12.5
    assert g.window_tokens(8192, 15) == 375
    assert g.request_reserve(q, 15) == g.fraction_for_tokens(375 + 256)
    assert g.request_reserve(q, 15, incoming=True) == g.fraction_for_tokens(256 + 1050 + 375 + 256)
    assert g.request_reserve(q, 15) < fallback
    assert g.reserved_usage([q], [], 15) == 0.1 + g.request_reserve(q, 15)
    assert g.window_tokens(8192, 20) == 438  # Telemetry age extends the horizon.
    assert g.window_tokens(8192, 100) == 8192  # Stale rates revert to full-tail reservation.
    assert math.isinf(g.reserved_usage([q], [], 100))


def test_new_and_finished_requests_do_not_corrupt_decode_speed():
    g = governor()
    observe(g, 10, {"a": 1000, "finished": 9000})
    observe(g, 15, {"a": 1050, "new": 8000})
    assert g.decode_rate(15) == 12.5
    observe(g, 20, {"a": 0})  # Prefix was restarted: never interpret a reset as speed.
    assert g.decode_rate(20) == 12.5


def test_speed_estimate_depends_on_batch_and_context_without_slow_extrapolation():
    g = governor()
    g.rate_context = 1
    g.decode_rates.extend([(10, 16, 1, 20)] * 32 + [(10, 64, 1, 8)] * 32)
    assert g.decode_rate(10, 16) == 25
    assert g.decode_rate(10, 64) == 10
    assert g.decode_rate(10, 128) == 10  # Use measured speed, not extrapolated slowdown.
    assert g.decode_rate(100) is None


def test_pressure_hysteresis_and_preemption_pause_admission():
    g = governor()
    observe(g, 10, {"a": 1000}, usage=0.91)
    assert g.update_pressure(0.92, 10)
    observe(g, 15, {"a": 1050}, usage=0.85)
    assert g.update_pressure(0.86, 15)  # Do not reopen on tiny fluctuations.
    observe(g, 20, {"a": 1100}, usage=0.79)
    assert not g.update_pressure(0.85, 20)
    observe(g, 25, {"a": 1150}, usage=0.4, preemptions=1)
    assert g.update_pressure(0.5, 25)
    observe(g, 30, {"a": 1200}, usage=0.4, preemptions=1)
    assert not g.update_pressure(0.5, 30)


def pressure_controller(count=3, **kwargs):
    c = LaminarController(
        [{"server": None, "workers": []} for _ in range(count)],
        None,
        batch_quota=64,
        max_running=256,
        check_interval_s=1,
        capacity_options=options(**kwargs),
        capacity_profiles=[pool()] * count,
    )
    c.latest = 0
    packets = []
    now = time.time()
    for index, g in c.capacity.items():
        c.replicas[index].version = 0
        q = Request(
            f"q{index}",
            index,
            0,
            backend_id=f"b{index}",
            prompt_tokens=1024,
            response_budget=32768,
            generated_tokens=8000,
        )
        c.replicas[index].active[q.request_id] = q
        observe(g, now - 5, {q.backend_id: 7950}, usage=0.92 if index == 0 else 0.1)
        packets.append(observe(g, now, {q.backend_id: 8000}, usage=0.92 if index == 0 else 0.1))
        g.update_pressure(c._admission_projection(index), now)
    return c, packets


def test_pressure_migration_reserves_target_before_abort_without_ramp_down():
    c, packets = pressure_controller()
    q = c.replicas[0].active["q0"]
    assert not c.replicas[0].ramp_down
    moves = c._reserve_pressure_moves(packets)
    assert len(moves) == 1
    target = q.destination
    assert target != 0 and q.request_id in c.replicas[target].reserved
    assert c.replicas[target].version == q.version
    assert c._admission_projection(target) < 0.90
    calls = []

    async def abort(backend):
        assert q.request_id in c.replicas[target].reserved
        calls.append(backend)
        return {"aborted": True}

    c.replicas[0].server = SimpleNamespace(abort_trajectory=SimpleNamespace(remote=abort))
    assert asyncio.run(c._execute_moves(moves)) == 1
    assert calls == ["b0"]


@pytest.mark.parametrize("unavailable", ["wrong_version", "full", "stale", "pulling", "releasing", "recovering"])
def test_no_abort_without_safe_same_version_target(unavailable):
    c, packets = pressure_controller(2)
    dst = c.replicas[1]
    g = c.capacity[1]
    if unavailable == "wrong_version":
        dst.version = 1
    elif unavailable == "full":
        g.observed_usage = 0.89
    elif unavailable == "stale":
        g.last_observed -= 100
    else:
        setattr(dst, unavailable, True)
    assert c._reserve_pressure_moves(packets) == []
    assert not dst.reserved and c.replicas[0].active["q0"].destination is None
    assert c.capacity[0].pressure_paused
    assert c._select_admission([0], Request("new", -1, -1, prompt_tokens=1, response_budget=100)) is None


def test_pressure_moves_are_bounded_and_cooldown_prevents_bouncing():
    c, packets = pressure_controller(2, rolling_kv_max_moves=1)
    first = c.replicas[0].active["q0"]
    first.last_pressure_migration = time.time()
    assert c._reserve_pressure_moves(packets) == []
    first.last_pressure_migration = -math.inf
    for i in range(4):
        q = Request(
            f"extra{i}", 0, 0, backend_id=f"extra{i}", prompt_tokens=128, response_budget=32768, generated_tokens=100
        )
        c.replicas[0].active[q.request_id] = q
        c.capacity[0].backend_ids.add(q.backend_id)
        c.capacity[0].generated_tokens[q.backend_id] = 100
    assert len(c._reserve_pressure_moves(packets)) == 1


def test_reset_clears_speed_and_pressure_for_new_weight_version():
    g = governor()
    observe(g, 10, {"a": 100})
    observe(g, 15, {"a": 150}, usage=0.91)
    g.update_pressure(0.95, 15)
    assert g.decode_rate(15) and g.pressure_paused
    g.reset_observation(g.pool)
    assert not g.decode_rates and g.rate_previous is None and not g.pressure_paused


def test_pressure_reservations_account_for_every_arrival_without_overbooking():
    c, packets = pressure_controller(2, rolling_kv_max_moves=4)
    dst = c.capacity[1]
    dst.observed_usage = 0.86
    src = c.replicas[0]
    src.active.clear()
    c.capacity[0].backend_ids.clear()
    c.capacity[0].generated_tokens.clear()
    for i in range(8):
        q = Request(
            f"tail{i}", 0, 0, backend_id=f"tail{i}", prompt_tokens=1024, response_budget=32768, generated_tokens=8000
        )
        src.active[q.request_id] = q
        c.capacity[0].backend_ids.add(q.backend_id)
        c.capacity[0].generated_tokens[q.backend_id] = 8000
    moves = c._reserve_pressure_moves(packets)
    assert 0 < len(moves) < 4
    assert c._admission_projection(1) < 0.90
    assert len(c.replicas[1].reserved) == len(moves)
    assert all(q.destination == 1 and q.version == c.replicas[1].version for _, q, _ in moves)


def test_unobserved_source_request_cannot_be_aborted_for_pressure_relief():
    c, packets = pressure_controller(2)
    c.capacity[0].backend_ids.clear()
    assert c._reserve_pressure_moves(packets) == []


def test_horizon_must_cover_telemetry_and_control_delay():
    with pytest.raises(ValueError, match="polling intervals"):
        LaminarController(
            [{"server": None, "workers": []}],
            None,
            batch_quota=64,
            max_running=256,
            check_interval_s=20,
            capacity_options=options(),
            capacity_profiles=[pool()],
        )


def test_rate_cache_expires_at_the_earliest_sample_and_invalidates_on_progress():
    g = governor()
    observe(g, 10, {"a": 100})
    observe(g, 15, {"a": 150})
    assert g.decode_rate(15) == 12.5
    assert g.decode_rate_cache
    assert g.decode_rate(75) == 12.5
    assert g.decode_rate(75.01) is None
    observe(g, 20, {"a": 250})
    assert g.decode_rate(20) == 25
