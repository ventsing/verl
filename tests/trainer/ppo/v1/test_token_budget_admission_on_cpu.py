# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Request-aware KV reservations, continuous refill and exact-version fences."""

import asyncio
import math
import time

import pytest

from verl.experimental.trajectory_async.capacity import CapacityConfig, OnlineCapacity, ResponseLengthEstimator
from verl.experimental.trajectory_async.laminar_controller import LaminarController, Request
from verl.experimental.trajectory_async.roofline import OnlineRoofline


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
    return {"enabled": True, "token_budget_admission": True, "kv_usage_limit": 0.8, **kwargs}


def observation(usage=0.1, generated=None, observed_at=None):
    return {
        "kv_usage": usage,
        "observed_at": observed_at or time.time(),
        "running": 16,
        "waiting": 0,
        "preemptions": 0,
        "generated_tokens": generated or {},
        "request_ids": list(generated or {}),
        "roofline": {
            "available": True,
            "collecting": True,
            "mode": "eager",
            "max_context_tokens": 4096,
            "capacity": pool(),
        },
    }


def controller(count=1):
    c = LaminarController(
        [{"server": None, "workers": []} for _ in range(count)],
        None,
        batch_quota=64,
        max_running=256,
        roofline_options={"enabled": True},
        capacity_options=options(),
        capacity_profiles=[pool() for _ in range(count)],
    )
    c.latest = 0
    for r in c.replicas:
        r.version = 0
    return c


def test_history_conditions_on_long_running_tails_and_keeps_capped_samples():
    predictor = ResponseLengthEstimator(CapacityConfig())
    for _ in range(8):
        predictor.observe(4096)
    short = predictor.remaining(0, 32768)
    assert 4096 < short < 8192
    predictor.observe(32768)
    assert predictor.remaining(16000, 32768) == 16768
    assert predictor.remaining(0, 256) == 256
    assert predictor.remaining(32768, 32768) == 0
    assert predictor.remaining(0, 32768) > short


def test_page_rounding_and_unknown_cache_layout_keep_conservative_bounds():
    g = OnlineCapacity(64, 256, options())
    g.pool = pool()
    assert g.fraction_for_tokens(129) == 2 / 4096
    assert 12 < g.memory_limit() < 64
    g.pool["workers"][0]["groups"][0]["type"] = "MambaSpec"
    assert not g.token_budget_enabled()
    assert g.memory_limit() == 12


def test_bootstrap_uses_full_context_and_stale_observations_cannot_expand():
    c = controller()

    async def run():
        for i in range(12):
            await c._assign(str(i), prompt_tokens=128, response_budget=32768)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await c._assign("blocked", prompt_tokens=128, response_budget=32768)
        g = c.capacity[0]
        g.update(observation(), c.roofline[0], time.time())
        assert await c._assign("after_telemetry", prompt_tokens=128, response_budget=32768)
        g.last_observed = time.time() - 100
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await c._assign("stale", prompt_tokens=128, response_budget=32768)

    asyncio.run(run())


def test_existing_future_growth_and_unseen_arrivals_consume_headroom():
    g = OnlineCapacity(64, 256, options())
    g.pool = pool()
    g.update(observation(usage=0.1, generated={"a": 4096}), None, time.time())
    visible = Request("a", 0, 0, backend_id="a", prompt_tokens=128, response_budget=32768)
    unseen = Request("u", 0, 0, prompt_tokens=1024, response_budget=32768)
    new = Request("new", 0, 0, prompt_tokens=128, response_budget=32768)
    base = g.projected_admission([visible], [], new, time.time())
    assert base > 0.1
    assert g.projected_admission([visible, unseen], [], new, time.time()) > base
    assert g.projected_admission([visible], [unseen], new, time.time()) > base
    g.observed_usage = 0.79
    assert g.projected_admission([visible], [], new, time.time()) > 0.8
    g.last_observed -= 100
    assert math.isinf(g.projected_admission([], [], new, time.time()))


def test_dispatch_prefers_projected_kv_instead_of_smallest_request_count():
    c = controller(2)
    for g, usage in zip(c.capacity.values(), [0.65, 0.1], strict=True):
        g.update(observation(usage), None, time.time())
    c.replicas[1].active["existing"] = Request("existing", 1, 0, prompt_tokens=128, response_budget=32768)
    request = asyncio.run(c._assign("new", prompt_tokens=128, response_budget=32768))
    assert request.slot == 1  # More requests, but much more projected free KV.


def test_completions_refill_without_waiting_for_a_batch_drain_and_versions_stay_fenced():
    c = controller()
    c.capacity[0].update(observation(), c.roofline[0], time.time())

    async def run():
        old = await c._assign("old", prompt_tokens=128, response_budget=32768)
        c.replicas[0].admitted = 1000  # Cumulative admissions no longer close a latest-version stream.
        new = await c._assign("new", prompt_tokens=128, response_budget=32768)
        assert new.slot == old.slot and not c.replicas[0].closed
        c.replicas[0].active.pop("new")
        assert await c._assign("refill", prompt_tokens=128, response_budget=32768)
        await c.notify_weights(1)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await c._assign("must_wait_for_new_weights", prompt_tokens=128, response_budget=32768)
        assert old.version == 0

    asyncio.run(run())


def test_measured_roofline_bounds_predicted_memory_capacity():
    g = OnlineCapacity(64, 256, options())
    g.pool = pool()
    estimator = OnlineRoofline()
    estimator.models[("eager", 0)] = {"status": "ready", "limit": 20}
    g.update(observation(), estimator, time.time())
    assert g.limit() == 20 < g.memory_limit()


def test_history_is_shared_bounded_and_seeded_from_real_trajectory_diagnostics(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(f'{{"event":"arrived","response_tokens":{n}}}' for n in range(1000, 1010)) + "\n{")
    c = LaminarController(
        [{"server": None, "workers": []}] * 2,
        None,
        batch_quota=64,
        max_running=256,
        roofline_options={"enabled": True},
        capacity_options=options(response_history_path=str(path), response_history_window=8),
        capacity_profiles=[pool(), pool()],
    )
    assert c.capacity[0].responses is c.capacity[1].responses
    assert c.capacity[0].responses.seeded == 8
    c._observe_response(32000)
    assert list(c.capacity[1].responses.values)[-1] == 32000
    assert len(c.capacity[0].responses.values) == 8


@pytest.mark.parametrize("new_version", [False, True])
def test_continuous_latest_stream_avoids_repacking_but_stale_versions_can_drain(new_version):
    from test_laminar_on_cpu import Handle

    c = controller(2)

    class Server:
        def __init__(self, request_id):
            self.request_id = request_id

        def trajectory_status(self):
            return observation(usage=0.1, generated={self.request_id: 2048})

        def trajectory_roofline(self):
            return {**observation()["roofline"], "samples": []}

    for i, r in enumerate(c.replicas):
        name = str(i)
        r.server = Handle(Server(name))
        r.active[name] = Request(name, i, 0, backend_id=name, prompt_tokens=128, response_budget=32768)
        r.closed = True
        r.admitted = 1000
        r.previous_usage = 0.2
    if new_version:
        c.latest = 1
    planned = []

    async def execute(moves):
        planned.extend(moves)
        return len(moves)

    c._execute_moves = execute
    asyncio.run(c.repack_once())
    assert bool(planned) == new_version
    assert all(q.version == 0 for r in c.replicas for q in r.active.values())
    if not new_version:
        assert all(not r.closed for r in c.replicas)


def test_headroom_pressure_backs_off_and_unknown_reservations_block_new_admission():
    c = controller()
    g = c.capacity[0]
    before = g.limit()
    g.update(observation(usage=0.81), c.roofline[0], time.time())
    assert g.limit() < before
    c.replicas[0].reserved.add("untracked_migration")
    request = Request("new", -1, -1, prompt_tokens=128, response_budget=32768)
    assert math.isinf(c._admission_projection(0, request))


def test_drained_weight_load_can_bootstrap_but_missing_profiles_never_expand():
    g = OnlineCapacity(64, 256, options())
    g.pool = pool()
    q = Request("new", 0, 0, prompt_tokens=128, response_budget=32768)
    g.update(observation(), None, time.time())
    g.last_observed -= 100
    assert math.isinf(g.projected_admission([], [], q, time.time()))
    g.reset_observation()
    assert g.projected_admission([], [], q, time.time()) == 1 / 16
    bad = observation()
    bad["roofline"]["capacity"] = {"available": False}
    g.update(bad, None, time.time())
    assert math.isinf(g.projected_admission([], [], q, time.time()))
    g.reset_observation()
    assert math.isinf(g.projected_admission([], [], q, time.time()))
    g.update(observation(), None, time.time())
    assert g.projected_admission([], [], q, time.time()) < 0.8


def test_token_budget_does_not_start_with_unknown_pool():
    with pytest.raises(ValueError, match="profiled KV pool"):
        LaminarController(
            [{"server": None, "workers": []}],
            None,
            batch_quota=64,
            max_running=256,
            roofline_options={"enabled": True},
            capacity_options=options(),
            capacity_profiles=[{"available": False}],
        )


def test_reservation_cache_invalidates_on_identity_progress_history_and_expiry(monkeypatch):
    c = controller()
    g = c.capacity[0]
    g.update(observation(), None, time.time())
    original = g.reserved_usage
    calls = []

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(g, "reserved_usage", counted)
    c.replicas[0].active["a"] = Request("a", 0, 0, prompt_tokens=128, response_budget=32768)
    new = Request("new", -1, -1, prompt_tokens=128, response_budget=32768)
    first = c._admission_projection(0, new)
    assert c._admission_projection(0, new) == first and len(calls) == 1
    # The candidate is deliberately outside the cached existing ledger.
    new.prompt_tokens = 4096
    assert c._admission_projection(0, new) > first and len(calls) == 1
    c.replicas[0].active.pop("a")
    c.replicas[0].active["b"] = Request("b", 0, 0, prompt_tokens=4096, response_budget=32768)
    c._admission_projection(0, new)
    assert len(calls) == 2  # Same count, different identity and prefix.
    c.replicas[0].active["b"].generated_base = 4096
    c._admission_projection(0, new)
    assert len(calls) == 3
    c._observe_response(32768)
    c._admission_projection(0, new)
    assert len(calls) == 4
    g.last_observed -= 100
    assert math.isinf(c._admission_projection(0, new)) and len(calls) == 5


def test_idle_engine_revalidates_unchanged_layout_after_missing_profile():
    g = OnlineCapacity(64, 256, options())
    g.update(observation(), None, time.time())
    bad = observation()
    bad["roofline"]["capacity"] = {"available": False}
    g.update(bad, None, time.time())
    q = Request("new", 0, 0, prompt_tokens=128, response_budget=32768)
    g.reset_observation({"available": False})
    assert math.isinf(g.projected_admission([], [], q, time.time()))
    changed = pool()
    changed["workers"][0]["total_blocks"] = 2048
    g.reset_observation(changed)
    assert math.isinf(g.projected_admission([], [], q, time.time()))
    g.reset_observation(pool())
    assert g.profile_available
    assert g.projected_admission([], [], q, time.time()) == 1 / 16
    # Bootstrap continues reserving a full context per arrival until fresh stats.
    requests = [Request(str(i), 0, 0, response_budget=32768) for i in range(12)]
    assert g.projected_admission(requests, [], q, time.time()) > 0.8


def test_idle_replica_refills_after_profile_loss_without_new_stats_tick():
    from test_laminar_on_cpu import Handle

    c = controller()
    g = c.capacity[0]
    old = observation(observed_at=time.time() - 100)
    old.update(request_ids=[], running=0, waiting=0, version=0, kv_usage=0.0)

    class Server:
        async def trajectory_status(self):
            return old

        async def trajectory_roofline(self):
            return old["roofline"]

    c.replicas[0].server = Handle(Server())
    g.profile_available = False
    g.last_observed = old["observed_at"]

    async def run():
        await c.repack_once(perform_moves=False)
        assert g.profile_available
        async with asyncio.timeout(0.1):
            request = await c._assign("refill", prompt_tokens=128, response_budget=32768)
        assert request.slot == 0 and request.version == c.latest
        assert not c.replicas[0].closed

    asyncio.run(run())


def migration_fixture(counts=(2, 2, 2)):
    c = controller(len(counts))
    c.latest = 1
    telemetry = []
    group = []
    for i, count in enumerate(counts):
        g = c.capacity[i]
        g.current = 3
        for _ in range(32):
            g.responses.observe(4096)
        generated = {}
        for j in range(count):
            name = f"{i}-{j}"
            c.replicas[i].active[name] = Request(
                name, i, 0, backend_id=name, prompt_tokens=128, response_budget=32768, generated_tokens=2048
            )
            generated[name] = 2048
        data = observation(usage=0.1 if i == 0 else 0.7, generated=generated)
        g.update(data, c.roofline[i], time.time())
        g.current = 3
        c.replicas[i].closed = True
        c.replicas[i].previous_usage = 0.8
        telemetry.append(data)
        group.append({"id": i, "usage": data["kv_usage"], "count": count, "context_tokens": 4096})
    return c, group, telemetry


def test_migration_splits_complete_source_across_targets_with_request_budgets():
    c, group, telemetry = migration_fixture()
    # Whole-source consolidation cannot fit either target by request count or
    # by the old full-context KV reservation (0.7 + 2/16 > 0.8).
    plan = c._plan_request_consolidation(group, telemetry)
    assert len(plan) == 2
    assert {s for s, _, _ in plan} == {0}
    assert {t for _, t, _ in plan} == {1, 2}
    assert len({q.request_id for _, _, q in plan}) == 2
    for _, target, q in plan:
        g = c.capacity[target]
        assert c._admission_projection(target) + g.request_reserve(q, time.time(), incoming=True) <= 0.8
    assert not any(r.reserved for r in c.replicas)  # Planning is transactional.
    assert all(q.destination is None for r in c.replicas for q in r.active.values())


@pytest.mark.parametrize("blocked", ["count", "stale", "version", "reservation", "prefix"])
def test_migration_rejects_incomplete_drain_without_partial_reservations(blocked):
    c, group, telemetry = migration_fixture()
    if blocked == "count":
        c.capacity[2].current = 2
    elif blocked == "stale":
        c.capacity[2].last_observed -= 100
    elif blocked == "version":
        c.replicas[2].version = 1
    elif blocked == "reservation":
        c.replicas[2].reserved.add("unknown-arrival")
    else:
        for q in c.replicas[0].active.values():
            q.prompt_tokens = 100000
    plan = c._plan_request_consolidation(group, telemetry)
    assert not any(source == 0 for source, _, _ in plan)
    assert all(q.destination is None for r in c.replicas for q in r.active.values())


def test_automatic_repack_commits_split_reservations_before_backend_abort():
    from test_laminar_on_cpu import Handle

    c, group, telemetry = migration_fixture()

    class Server:
        def __init__(self, index):
            self.index = index

        def trajectory_status(self):
            return telemetry[self.index]

        def trajectory_roofline(self):
            return {**telemetry[self.index]["roofline"], "samples": []}

    for i, replica in enumerate(c.replicas):
        replica.server = Handle(Server(i))
    executed = []

    async def execute(moves):
        assert len(moves) == 2
        assert len(c.replicas[1].reserved) == len(c.replicas[2].reserved) == 1
        assert {q.destination for _, q, _ in moves} == {1, 2}
        executed.extend(moves)
        return len(moves)

    c._execute_moves = execute
    assert asyncio.run(c.repack_once()) == 2
    assert len(executed) == 2


def test_multiple_sources_share_destination_budget_without_chained_moves():
    c, group, telemetry = migration_fixture((1, 1, 1))
    plan = c._plan_request_consolidation(group, telemetry)
    assert len(plan) == 2
    sources = {source for source, _, _ in plan}
    targets = {target for _, target, _ in plan}
    assert not sources & targets
    assert len(targets) == 1
    target = next(iter(targets))
    arrivals = [q for _, t, q in plan if t == target]
    g = c.capacity[target]
    assert c.replicas[target].load + len(arrivals) <= c._repack_limit(target, telemetry)
    assert (
        c._admission_projection(target) + sum(g.request_reserve(q, time.time(), incoming=True) for q in arrivals)
        <= g.config.kv_usage_limit
    )


def test_step_calibration_reduces_completion_order_bias_without_duplicate_history():
    p = ResponseLengthEstimator(CapacityConfig())
    for _ in range(512):
        p.observe(32768)
    before = p.remaining(0, 32768)
    history = list(p.values)
    assert p.calibrate(1, [(0, 4096)] * 512)
    assert p.remaining(0, 32768, version=0) < before / 2
    assert list(p.values) == history
    revision = p.revision
    assert not p.calibrate(1, [(0, 8192)] * 512)
    assert p.revision == revision
    assert p.status()["samples"] == 512


def test_step_calibration_versions_fallback_tail_conditioning_and_online_updates():
    p = ResponseLengthEstimator(CapacityConfig(response_calibration_online_weight=0))
    p.calibrate(1, [(0, 4096)] * 32)
    p.calibrate(2, [(1, 16384)] * 32)
    assert p.remaining(0, 32768, version=0) == math.ceil(4096 * 1.15)
    assert p.remaining(0, 32768, version=1) == math.ceil(16384 * 1.15)
    assert p.remaining(0, 32768, version=2) == p.remaining(0, 32768, version=1)
    assert p.remaining(20000, 32768, version=0) >= 256
    assert p.remaining(32768, 32768, version=0) == 0
    q = ResponseLengthEstimator(CapacityConfig())
    q.calibrate(1, [(0, 4096)] * 32)
    before = q.remaining(0, 32768, version=0)
    for _ in range(32):
        q.observe(16384)
    assert q.remaining(0, 32768, version=0) > before


def test_calibration_batch_validation_is_atomic_and_storage_is_bounded():
    p = ResponseLengthEstimator(CapacityConfig(response_history_window=8, response_calibration_versions=2))
    with pytest.raises(ValueError):
        p.calibrate(1, [(0, 4096), (1, 4096)])
    assert not getattr(p, "calibrations", {})
    assert not p.calibrate(1, [])
    for step in range(1, 5):
        p.calibrate(step, [(step - 1, 4096)] * 20)
    assert set(p.calibrations) == {2, 3}
    assert all(len(v["values"]) == 8 for v in p.calibrations.values())


def test_controller_calibration_invalidates_budget_cache_once_per_step():
    c = controller()
    c._budget_cache[0] = ("old", 0.7)
    status = c.calibrate_response_lengths(1, [(0, 4096)] * 32)
    assert status["updated"] and status["calibration_step"] == 1
    assert not c._budget_cache
    assert status["samples"] == 0
    assert not c.calibrate_response_lengths(1, [(0, 8192)] * 32)["updated"]


def test_fresh_request_uses_target_replica_version_for_batch_prediction():
    c = controller()
    g = c.capacity[0]
    g.update(observation(usage=0, generated={}), None, time.time())
    g.config.response_calibration_online_weight = 0
    g.responses.calibrate(1, [(0, 4096)] * 32)
    q = Request("unassigned", -1, -1, prompt_tokens=128, response_budget=32768)
    assert c._admission_projection(0, q) == g.request_reserve(q, time.time(), incoming=True, version=0)
    assert c._admission_projection(0, q) < g.request_reserve(q, time.time(), incoming=True)


def test_trainer_step_calibrates_behavior_versions_and_excludes_padding_and_mixed_rows(monkeypatch):
    from types import SimpleNamespace

    import verl.trainer.ppo.v1.trainer_laminar as module
    from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode

    trainer = module.PPOTrainerLaminar.__new__(module.PPOTrainerLaminar)
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer._length_calibration_enabled = True
    trainer._step_sample_wait_seconds = 0.0
    trainer._sample_start = time.perf_counter()
    trainer.global_steps = 2
    trainer.replay_buffer = SimpleNamespace(
        last_selected_tags={
            "train": [
                {"global_steps": 0, "response_len": 4096},
                {"global_steps": 1, "response_len": 8192},
                {"global_steps": 0, "min_global_steps": 0, "max_global_steps": 1, "response_len": 2000},
                {"global_steps": 0, "response_len": 1000, "is_padding": True},
            ]
        }
    )
    calls = []
    trainer.laminar_controller = SimpleNamespace(
        calibrate_response_lengths=SimpleNamespace(remote=lambda step, records: calls.append((step, records)))
    )
    monkeypatch.setattr(module.ray, "get", lambda value, **kwargs: value)
    trainer.on_sample_end()
    assert calls == [(2, [(0, 4096), (1, 8192)])]
    assert trainer._sample_started_at is None
