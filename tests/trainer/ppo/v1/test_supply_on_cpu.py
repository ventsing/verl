# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Demand feedback and real-controller, exact-version resource handoffs."""

import asyncio
import time

import pytest

from tests.trainer.ppo.v1.test_evaluation_controller_on_cpu import Handle, Relay, Server, until
from verl.experimental.trajectory_async.evaluation_controller import EvaluationController
from verl.experimental.trajectory_async.laminar_controller import LaminarController
from verl.experimental.trajectory_async.supply import (
    SupplyConfig,
    SupplyCoordinator,
    SupplyPolicy,
    queue_supply_snapshot,
)


def test_ready_watermarks_pause_new_work_without_starving_training():
    policy = SupplyPolicy(SupplyConfig(healthy_polls=2, loan_cooldown_s=10))
    assert policy.decide({"ready_batches": 1.6}, 0, 8, True, 0) == (True, 0, "hold")
    assert policy.decide({"ready_batches": 1.2}, 0, 8, True, 1) == (True, 1, "surplus")
    assert policy.decide({"ready_batches": 1.1}, 1, 8, True, 2) == (True, 1, "hold")
    assert policy.decide({"ready_batches": 0.4}, 1, 8, True, 3) == (False, 0, "refill")
    # Staleness never blocks an empty consumer, and no stale data is discarded.
    assert policy.decide({"ready_batches": 0, "window_staleness": 8}, 0, 8, True, 4)[0] is False


def test_staleness_and_live_sample_wait_control_supply():
    p = SupplyPolicy(SupplyConfig())
    assert p.decide({"ready_batches": 1, "window_staleness": 4}, 0, 8, False, 0)[0]
    assert p.decide({"ready_batches": 1, "sample_wait_s": 6}, 4, 8, True, 1) == (False, 0, "refill")
    assert p.decide({"ready_batches": 2}, 6, 8, True, 2)[1] == 6
    assert p.decide({"ready_batches": 2}, 3, 8, False, 3)[1] == 0
    assert p.decide({"ready_batches": 0, "training_finished": True}, 0, 8, True, 4) == (True, 8, "training_finished")


@pytest.mark.parametrize(
    "options",
    [
        {"low_ready_batches": 2},
        {"max_pending_batches": 1},
        {"healthy_polls": 0},
        {"feedback_timeout_s": 1},
        {"min_training_replicas": 0},
        {"target_staleness": -1},
    ],
)
def test_invalid_supply_config(options):
    with pytest.raises(ValueError):
        SupplyConfig(**options)


def test_only_whole_unsampled_successful_groups_count_as_supply():
    partition = {"p": {"is_prompt": True, "status": "finished"}, "q": {"is_prompt": True, "status": "running"}}
    for uid in ["p", "q", "already-sampled"]:
        for i in range(4):
            partition[f"{uid}_{i}_0"] = {"status": "success"}
    assert queue_supply_snapshot(partition, 4, 16) == {
        "ready_groups": 1,
        "pending_groups": 2,
        "ready_batches": 1 / 16,
        "unfinished_trajectories": 0,
    }
    partition["p_3_0"]["status"] = "failed"
    assert queue_supply_snapshot(partition, 4, 16)["ready_groups"] == 0


def test_release_fences_requests_pulls_reservations_and_recovery():
    async def scenario():
        relay = Handle(Relay())
        server = Handle(Server())
        c = LaminarController([{"server": server, "workers": [server]}], relay, enable_repack=False)
        r = c.replicas[0]
        for field, value in [("active", {"r": object()}), ("reserved", {"r"}), ("pulling", True), ("recovering", True)]:
            original = getattr(r, field)
            setattr(r, field, value)
            assert not await c.release_replica(0)
            assert r.owned and r.closed and r.releasing
            setattr(r, field, original)
        assert await c.release_replica(0)
        assert not r.owned
        await c.acquire_replica(0, admission_ceiling=1)
        assert r.owned and r.version == -1 and c._admission_limit(0) == 1
        with pytest.raises(RuntimeError):
            await c.acquire_replica(0)

    asyncio.run(scenario())


def test_borrowed_engine_serves_exact_eval_version_and_returns_latest_training_version():
    async def scenario():
        relay = Handle(Relay())
        servers = [Server() for _ in range(3)]
        specs = [{"server": Handle(s), "workers": [Handle(s)], "relay": relay} for s in servers]
        train = LaminarController(
            specs[:2], relay, batch_quota=1, max_running=1, check_interval_s=0.01, enable_repack=False
        )
        evaluation = EvaluationController(
            [specs[2]] + [{**s, "owned": False} for s in specs[:2]],
            relay,
            batch_quota=2,
            max_running=2,
            check_interval_s=0.01,
            enable_repack=False,
        )
        broker = SupplyCoordinator(
            Handle(train),
            Handle(evaluation),
            2,
            1,
            {"enabled": True, "min_training_replicas": 1, "healthy_polls": 1, "loan_cooldown_s": 0},
        )
        tasks = []
        try:
            await train.start(20)
            await evaluation.register_version(10)
            tasks = [
                asyncio.create_task(
                    evaluation.generate(f"eval-{i}", [1], {"max_tokens": 2, "_laminar_policy_version": 10})
                )
                for i in range(3)
            ]
            await until(lambda: bool(evaluation.request_versions))
            broker.update({"ready_batches": 1, "window_staleness": 4})
            await broker.tick()
            assert broker.borrowed == {0}
            assert not train.replicas[0].owned and evaluation.replicas[1].owned
            assert evaluation._admission_limit(1) == 1  # Do not exceed donor safety envelope.
            await until(lambda: bool(servers[0].requests))
            assert servers[0].version == 10
            await train.notify_weights(30)
            await asyncio.sleep(0.03)
            assert servers[0].version == 10  # Training cannot pull on a loaned engine.
            broker.update({"ready_batches": 0})
            await broker.tick()
            assert broker.transitions == {0: "training"}
            assert evaluation.replicas[1].releasing and not train.replicas[0].owned
            for s in servers:
                s.finish()
            await asyncio.gather(*tasks)
            await evaluation.unregister_version(10)
            await broker.tick()
            assert not broker.borrowed
            assert train.replicas[0].owned and not evaluation.replicas[1].owned
            await until(lambda: train.replicas[0].version == 30)
            assert servers[0].version == 30
            assert all(not (train.replicas[i].owned and evaluation.replicas[i + 1].owned) for i in range(2))
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await train.stop()
            await evaluation.stop()

    asyncio.run(scenario())


def test_lost_feedback_reclaims_loans_and_resumes_admission():
    async def scenario():
        relay = Handle(Relay())
        servers = [Server(), Server()]
        specs = [{"server": Handle(s), "workers": [Handle(s)]} for s in servers]
        train = LaminarController(specs, relay, enable_repack=False)
        evaluation = EvaluationController([{**s, "owned": False} for s in specs], relay, enable_repack=False)
        evaluation.request_versions["queued"] = 5
        broker = SupplyCoordinator(
            Handle(train),
            Handle(evaluation),
            2,
            0,
            {"enabled": True, "healthy_polls": 1, "min_training_replicas": 1, "loan_cooldown_s": 0},
        )
        broker.update({"ready_batches": 2})
        await broker.tick()
        assert broker.borrowed == {0}
        await broker.tick(now=broker.feedback_at + 20)
        assert not broker.borrowed and not train.supply_paused
        assert broker.status()["reason"] == "feedback_timeout"

    asyncio.run(scenario())


def test_supply_recovery_cancels_an_unfinished_loan():
    async def scenario():
        relay = Handle(Relay())
        specs = [{"server": Handle(Server()), "workers": []} for _ in range(2)]
        train = LaminarController(specs, relay, enable_repack=False)
        evaluation = EvaluationController([{**s, "owned": False} for s in specs], relay, enable_repack=False)
        evaluation.request_versions["queued"] = 5
        # Both engines are busy; the loan must drain before transferring.
        train.replicas[0].reserved.add("one")
        train.replicas[1].reserved.add("two")
        broker = SupplyCoordinator(
            Handle(train),
            Handle(evaluation),
            2,
            0,
            {"enabled": True, "healthy_polls": 1, "min_training_replicas": 1, "loan_cooldown_s": 0},
        )
        broker.update({"ready_batches": 2})
        await broker.tick()
        assert broker.transitions == {0: "evaluation"}
        broker.update({"ready_batches": 0})
        await broker.tick()
        assert not broker.transitions and train.replicas[0].owned and not train.replicas[0].releasing

    asyncio.run(scenario())


def test_repacking_does_not_poll_or_migrate_into_unowned_engines():
    async def scenario():
        class ForbiddenServer:
            def trajectory_status(self):
                raise AssertionError("The other pool owns this engine")

        c = LaminarController(
            [
                {"server": Handle(Server()), "workers": []},
                {"server": Handle(ForbiddenServer()), "workers": [], "owned": False},
            ],
            Handle(Relay()),
            enable_repack=True,
        )
        await c.repack_once()
        with pytest.raises(ValueError):
            await c.migrate(0, 1)

    asyncio.run(scenario())


def test_scheduled_weight_loads_block_ownership_transfer():
    async def scenario():
        spec = {"server": Handle(Server()), "workers": []}
        train = LaminarController([spec], Handle(Relay()))
        train.scheduled_pulls.add(0)
        assert not await train.release_replica(0)
        train.scheduled_pulls.clear()
        assert await train.release_replica(0)
        evaluation = EvaluationController([spec], Handle(Relay()))
        evaluation.loading_versions[0] = 5
        assert not await evaluation.release_replica(0)
        evaluation.loading_versions.clear()
        assert await evaluation.release_replica(0)

    asyncio.run(scenario())


def test_just_in_time_warmup_respects_prompt_capacity(monkeypatch):
    from threading import RLock
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1 import trainer_laminar as module

    trainer = object.__new__(module.PPOTrainerLaminar)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"v1": {"laminar": {"initial_experience_dir": None, "prompt_buffer_groups": 64}}},
            "data": {"train_batch_size": 16},
        }
    )
    trainer.supply_config = SupplyConfig(enabled=True)
    trainer.supply_coordinator = SimpleNamespace(start=SimpleNamespace(remote=lambda: None))
    trainer._prompt_lock = RLock()
    trainer._add_batch_to_generate = lambda: None
    captured = {}

    class Producer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    class Observer:
        def __init__(self, *args):
            pass

        def start(self):
            pass

    def unbounded_warmup(self):
        raise AssertionError("The bounded feeder must own warmup as well as steady state")

    monkeypatch.setattr(module.PPOTrainerSeparateAsync, "on_train_begin", unbounded_warmup)
    monkeypatch.setattr(module, "PromptFeeder", Producer)
    monkeypatch.setattr(module, "SupplyMonitor", Observer)
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    trainer.on_train_begin()
    assert captured["capacity"] == 32 and captured["started"]


def test_surplus_never_lends_the_last_healthy_training_replica():
    async def scenario():
        relay = Handle(Relay())
        specs = [{"server": Handle(Server()), "workers": []} for _ in range(3)]
        train = LaminarController(specs, relay)
        train.replicas[1].healthy = train.replicas[2].healthy = False
        evaluation = EvaluationController([{**s, "owned": False} for s in specs], relay)
        evaluation.request_versions["queued"] = 5
        broker = SupplyCoordinator(
            Handle(train), Handle(evaluation), 3, 0, {"enabled": True, "healthy_polls": 1, "min_training_replicas": 1}
        )
        broker.update({"ready_batches": 2})
        await broker.tick()
        assert not broker.borrowed and train.replicas[0].owned

    asyncio.run(scenario())


def test_pausing_supply_preserves_inflight_work_and_blocks_only_new_admission():
    async def scenario():
        relay = Handle(Relay())
        server = Server()
        controller = LaminarController(
            [{"server": Handle(server), "workers": [Handle(server)], "relay": relay}],
            relay,
            batch_quota=2,
            max_running=2,
            check_interval_s=0.01,
            enable_repack=False,
        )
        tasks = []
        try:
            await controller.start(10)
            tasks.append(asyncio.create_task(controller.generate("running", [1], {"max_tokens": 2})))
            await until(lambda: bool(server.requests))
            await controller.set_supply_paused(True)
            tasks.append(asyncio.create_task(controller.generate("waiting", [1], {"max_tokens": 2})))
            await until(lambda: len(controller.pending) == 2)
            server.finish()
            result = await tasks[0]
            assert result.extra_fields["global_steps"] == 10
            await asyncio.sleep(0.03)
            assert not tasks[1].done() and not server.requests
            await controller.set_supply_paused(False)
            await until(lambda: bool(server.requests))
            server.finish()
            assert (await tasks[1]).extra_fields["global_steps"] == 10
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await controller.stop()

    asyncio.run(scenario())


def test_pending_loan_survives_cooldown_while_supply_remains_healthy():
    async def scenario():
        relay = Handle(Relay())
        specs = [{"server": Handle(Server()), "workers": []} for _ in range(2)]
        train = LaminarController(specs, relay)
        for r in train.replicas:
            r.reserved.add("in-flight")
        evaluation = EvaluationController([{**s, "owned": False} for s in specs], relay)
        evaluation.request_versions["queued"] = 5
        broker = SupplyCoordinator(
            Handle(train),
            Handle(evaluation),
            2,
            0,
            {"enabled": True, "healthy_polls": 1, "min_training_replicas": 1, "loan_cooldown_s": 60},
        )
        broker.update({"ready_batches": 2})
        await broker.tick()
        assert broker.transitions == {0: "evaluation"}
        await broker.tick()
        assert broker.transitions == {0: "evaluation"} and train.replicas[0].releasing
        train.replicas[0].reserved.clear()
        await broker.tick()
        assert broker.borrowed == {0} and not broker.transitions

    asyncio.run(scenario())


def test_training_serving_lag_excludes_loaned_instances():
    from verl.experimental.trajectory_async.staleness_metrics import compute_replica_policy_metrics

    replica = {"healthy": True, "pulling": False, "closed": False, "active": 0, "reserved": 0}
    metrics = compute_replica_policy_metrics(
        {
            "latest": 30,
            "replicas": [
                {**replica, "version": 30, "owned": True},
                {**replica, "version": 10, "owned": False},
            ],
        }
    )
    assert metrics["laminar/replicas/count"] == 1
    assert metrics["laminar/serving_policy/lag_mean"] == 0


@pytest.mark.parametrize("split", [False, True])
def test_active_handoff_splits_requests_across_same_version_targets_without_ramp_down(split):
    async def scenario():
        relay = Handle(Relay())

        class CachedServer(Server):
            def trajectory_status(self):
                return {**super().trajectory_status(), "kv_usage": 0.4 if len(self.requests) == 2 else 0.2}

        servers = [CachedServer() for _ in range(3 if split else 2)]
        specs = [{"server": Handle(s), "workers": [Handle(s)], "relay": relay} for s in servers]
        controller = LaminarController(
            specs, relay, batch_quota=2, max_running=2, repack_max_running=1 if split else 2, enable_repack=False
        )
        tasks = []
        try:
            await controller.notify_weights(10)
            await asyncio.gather(*(controller._pull(i) for i in range(len(servers))))
            for r in controller.replicas[1:]:
                r.closed = True
            tasks = [asyncio.create_task(controller.generate(f"drain-{i}", [1], {"max_tokens": 3})) for i in range(2)]
            await until(lambda: len(servers[0].requests) == 2)
            assert not any(r.ramp_down for r in controller.replicas)
            for r in controller.replicas[1:]:
                r.closed = False
            await controller.release_replica(0)
            await until(lambda: not controller.replicas[0].load and all(s.requests for s in servers[1:]))
            assert await controller.release_replica(0)
            assert not controller.replicas[0].owned
            assert all(r.version == 10 and r.load == (1 if split else 2) for r in controller.replicas[1:])
            for server in servers[1:]:
                server.finish()
            outputs = await asyncio.wait_for(asyncio.gather(*tasks), 5)
            assert all(o.token_ids == [2, 3] and o.log_probs == [-0.2, -0.2] for o in outputs)
            assert all(o.extra_fields["global_steps"] == 10 and o.extra_fields["migrations"] == 1 for o in outputs)
            assert controller.metrics["drain_moves"] == 2
            assert all(not r.reserved for r in controller.replicas)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await controller.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "blocked",
    [
        "different_version",
        "full",
        "releasing",
        "unowned",
        "recovering",
        "pulling",
        "stale",
        "kv_capacity",
        "waiting",
        "backend_not_ready",
        "loading",
        "unobserved_arrival",
    ],
)
def test_active_drain_refuses_unsafe_destinations(blocked):
    from verl.experimental.trajectory_async.laminar_controller import Request

    async def scenario():
        aborts = []

        class Telemetry:
            def __init__(self, source):
                self.source = source

            def trajectory_status(self):
                return {
                    "request_ids": ["backend"] if self.source and blocked != "backend_not_ready" else [],
                    "kv_usage": 0.89 if not self.source and blocked == "kv_capacity" else 0.1,
                    "waiting": int(not self.source and blocked == "waiting"),
                    "observed_at": time.time() - (100 if blocked == "stale" else 0),
                }

            def abort_trajectory(self, request_id):
                aborts.append(request_id)
                return {"aborted": True}

        specs = [{"server": Handle(Telemetry(source)), "workers": [], "version": 10} for source in [True, False]]
        controller = EvaluationController(
            specs, Handle(Relay()), batch_quota=2, max_running=2 if blocked == "unobserved_arrival" else 1
        )
        controller.replicas[0].active["request"] = Request("request", 0, 10, backend_id="backend")
        target = controller.replicas[1]
        if blocked == "different_version":
            target.version = 11
        elif blocked == "full":
            target.reserved.add("other-arrival")
        elif blocked == "releasing":
            target.releasing = True
        elif blocked == "unowned":
            target.owned = False
        elif blocked == "recovering":
            target.recovering = True
        elif blocked == "pulling":
            target.pulling = True
        elif blocked == "loading":
            controller.loading_versions[1] = 10
        elif blocked == "unobserved_arrival":
            target.active["arriving"] = Request("arriving", 1, 10, backend_id="not-yet-in-telemetry")
        assert not await controller.release_replica(0)
        assert not aborts and controller.replicas[0].owned
        assert controller.replicas[0].active["request"].destination is None

    asyncio.run(scenario())


def test_active_drain_preserves_durable_prefix_and_behavior_logprobs(tmp_path):
    from tests.trainer.ppo.v1.test_partial_store_on_cpu import durable_controller

    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("durable-drain", [1], {"max_tokens": 3, "logprobs": True}))
        try:
            await until(lambda: bool(servers[0].requests))
            await controller.release_replica(0)
            await until(lambda: bool(servers[1].requests) and not controller.replicas[0].load)
            assert await controller.release_replica(0)
            assert servers[1].calls[-1][1:] == ([1, 2], 2)
            servers[1].finish()
            output = await asyncio.wait_for(task, 5)
            assert output.token_ids == [2, 3, 4]
            assert output.log_probs == [-0.25, -0.25, -0.5]
            assert output.extra_fields["migrations"] == 1
            assert output.extra_fields["global_steps"] == 0
            assert not any(r.reserved for r in controller.replicas)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await controller.stop()
            store.close()

    asyncio.run(scenario())


def test_refill_counts_missing_trajectories_instead_of_blocking_on_tail_groups():
    from verl.experimental.trajectory_async.supply import should_refill

    config = SupplyConfig(low_ready_batches=1, target_ready_batches=1.5, high_ready_batches=2.5, max_pending_batches=4)
    runtime = {"observed_at": 100, "admission_total": 192}
    # The old two-batch bound would stop: 32 prompts, but only 60 trajectories
    # still need compute and 14 full groups cannot supply the next 16-group batch.
    queue = {"pending_groups": 32, "ready_groups": 14, "unfinished_trajectories": 60}
    assert should_refill(config, queue, runtime, 16, 101)
    assert not should_refill(config, {**queue, "unfinished_trajectories": 192}, runtime, 16, 101)
    assert not should_refill(config, {**queue, "ready_groups": 24}, runtime, 16, 101)
    assert should_refill(config, queue, {}, 16, 101)


def test_small_prompt_refills_use_idle_capacity_and_stop_at_ready_watermark():
    import threading

    from tests.trainer.ppo.v1.test_prompt_feeder_on_cpu import wait_for
    from verl.experimental.trajectory_async.prompt_feeder import PromptFeeder
    from verl.experimental.trajectory_async.supply import should_refill

    config = SupplyConfig(low_ready_batches=1, target_ready_batches=1.5, high_ready_batches=2.5, max_pending_batches=4)
    state = {"pending_groups": 32, "ready_groups": 14, "unfinished_trajectories": 60}
    runtime = {"observed_at": time.monotonic(), "admission_total": 100}
    lock = threading.RLock()
    submitted = []

    def submit():
        state["pending_groups"] += 4
        state["unfinished_trajectories"] += 16
        submitted.append(4)

    feeder = PromptFeeder(
        lambda: state["pending_groups"],
        submit,
        batch_size=4,
        capacity=64,
        lock=lock,
        poll_interval=0.001,
        can_submit=lambda: should_refill(config, state, runtime, 16, time.monotonic()),
    )
    feeder.start()
    try:
        wait_for(lambda: state["unfinished_trajectories"] >= 100)
        assert submitted == [4, 4, 4] and state["pending_groups"] == 44
        with lock:
            state.update(ready_groups=24, unfinished_trajectories=20)
            before = len(submitted)
        time.sleep(0.02)
        assert len(submitted) == before
        feeder.check()
    finally:
        feeder.stop()
