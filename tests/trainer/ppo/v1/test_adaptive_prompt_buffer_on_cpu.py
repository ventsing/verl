# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded prompt horizons respond to actual complete-group supply."""

import threading
import time

import pytest

from verl.experimental.trajectory_async.prompt_feeder import AdaptivePromptBuffer, PromptFeeder


def feedback(ready=0, unfinished=200, pending=256, active=200, admission=200, now=0):
    return (
        {"ready_groups": ready, "unfinished_trajectories": unfinished, "pending_groups": pending},
        {"observed_at": now, "active_total": active, "admission_total": admission},
    )


def decide(policy, now, **kwargs):
    lag = kwargs.pop("lag", 0)
    wait = kwargs.pop("wait", 60)
    queue, runtime = feedback(now=now, **kwargs)
    return policy.observe(queue, runtime, lag, wait, now)


def test_saturated_engines_do_not_expand_for_training_wait_alone():
    p = AdaptivePromptBuffer(64, 256, {"initial_batches": 2})
    for now in range(10):
        assert decide(p, now, pending=128, active=200, unfinished=900) == 128
    assert p.adjustments == 0


def test_growth_requires_sustained_starvation_engine_room_and_full_window():
    p = AdaptivePromptBuffer(64, 256, {"initial_batches": 2})
    args = dict(pending=128, active=50, unfinished=100)
    assert decide(p, 0, **args) == 128
    assert decide(p, 5, **args) == 128
    assert decide(p, 10, **args) == 144
    for now in [15, 20, 25, 30]:
        assert decide(p, now, **args) == 144  # Cooldown and room for a refill.
    assert p.adjustments == 1


def test_completed_surplus_reduces_window_with_bounds_and_no_cancellation():
    p = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    for now in range(0, 1200, 60):
        decide(p, now, ready=96, unfinished=1000, pending=256)
    assert p.current == 128
    assert p.adjustments == 8


def test_partial_groups_and_pending_requests_do_not_count_as_ready_surplus():
    p = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    assert decide(p, 0, ready=0, unfinished=1800) == 256
    assert decide(p, 60, ready=40, unfinished=1800) == 240
    assert p.reason == "excess_queued_work"


def test_staleness_reduces_prefetch_only_when_ready_groups_exist():
    p = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    assert decide(p, 0, ready=0, lag=5) == 256
    assert decide(p, 60, ready=32, lag=5) == 240


def test_stale_or_future_feedback_holds_window_and_resets_hysteresis():
    p = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    queue, runtime = feedback(ready=128, now=0)
    assert p.observe(queue, runtime, 0, 60, 20) == 256
    runtime["observed_at"] = 30
    assert p.observe(queue, runtime, 0, 60, 20) == 256
    assert p.streak == 0


@pytest.mark.parametrize(
    "options",
    [
        {"min_batches": 0.5},
        {"max_batches": 1},
        {"initial_batches": 1},
        {"step_groups": 0},
        {"refill_groups": 129},
        {"healthy_polls": 0},
        {"adjust_interval_s": 0},
        {"target_staleness": -1},
    ],
)
def test_invalid_window_configuration(options):
    with pytest.raises(ValueError):
        AdaptivePromptBuffer(64, 256, options)


def test_feeder_respects_reduced_capacity_without_cancelling_existing_work():
    lock = threading.RLock()
    backlog = list(range(8))
    submitted = threading.Event()

    def submit():
        backlog.extend(["new"] * 2)
        submitted.set()

    feeder = PromptFeeder(lambda: len(backlog), submit, batch_size=2, capacity=4, lock=lock, poll_interval=0.001)
    feeder.start()
    try:
        assert not submitted.wait(0.03)
        assert backlog == list(range(8))
        with lock:
            del backlog[:6]  # Consumer removes completed groups.
        assert submitted.wait(1)
        assert len(backlog) == 4
    finally:
        feeder.stop()


def test_trainer_poll_updates_feeder_capacity_without_replacing_existing_queue(monkeypatch):
    from types import SimpleNamespace

    import verl.trainer.ppo.v1.trainer_laminar as module

    trainer = module.PPOTrainerLaminar.__new__(module.PPOTrainerLaminar)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=8)), data=SimpleNamespace(train_batch_size=64)
    )
    trainer._adaptive_buffer = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    trainer._prompt_lock = threading.RLock()
    trainer._prompt_feeder = SimpleNamespace(capacity=256)
    trainer._sample_started_at = time.monotonic() - 60
    trainer._buffer_staleness = 0.0
    trainer.laminar_controller = SimpleNamespace(status=SimpleNamespace(remote=lambda: object()))
    queue, _ = feedback(ready=96, unfinished=800)
    monkeypatch.setattr(module.tq, "kv_list", lambda: {"train": {}})
    monkeypatch.setattr(module, "queue_supply_snapshot", lambda *args: queue)
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kwargs: (refs, []))
    monkeypatch.setattr(
        module.ray,
        "get",
        lambda *args, **kwargs: {"replicas": [{"healthy": True, "active": 200, "admission_limit": 200}]},
    )
    trainer._poll_adaptive_buffer()
    assert trainer._prompt_feeder.capacity == 240
    assert trainer._adaptive_buffer.current == 240
    assert queue["pending_groups"] == 256


def status_poll_fixture(monkeypatch):
    from types import SimpleNamespace

    import verl.trainer.ppo.v1.trainer_laminar as module

    trainer = module.PPOTrainerLaminar.__new__(module.PPOTrainerLaminar)
    trainer._adaptive_buffer = AdaptivePromptBuffer(64, 256, {"healthy_polls": 1})
    trainer._prompt_feeder = SimpleNamespace(capacity=256)
    requests = []

    def status():
        ref = object()
        requests.append(ref)
        return ref

    trainer.laminar_controller = SimpleNamespace(status=SimpleNamespace(remote=status))
    return module, trainer, requests


def test_busy_controller_keeps_one_query_and_never_fails_or_changes_window(monkeypatch):
    module, trainer, requests = status_poll_fixture(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kwargs: ([], refs))
    monkeypatch.setattr(module.ray, "get", lambda *args, **kwargs: pytest.fail("Unready result must not be fetched"))
    for now in [0, 5, 15, 30, 120]:
        clock[0] = now
        trainer._poll_adaptive_buffer()
    assert len(requests) == 1
    assert trainer._adaptive_buffer.current == trainer._prompt_feeder.capacity == 256
    assert trainer._adaptive_buffer_status_warned


def test_late_controller_reply_is_ignored_then_a_fresh_query_can_be_submitted(monkeypatch):
    module, trainer, requests = status_poll_fixture(monkeypatch)
    clock = [0.0]
    ready = [False]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kwargs: (refs, []) if ready[0] else ([], refs))
    monkeypatch.setattr(module.ray, "get", lambda ref: {})
    trainer._poll_adaptive_buffer()
    clock[0] = 30
    ready[0] = True
    trainer._poll_adaptive_buffer()
    assert trainer._adaptive_buffer_status_ref is None
    assert trainer._adaptive_buffer.current == 256
    ready[0] = False
    trainer._poll_adaptive_buffer()
    assert len(requests) == 2


def test_controller_failure_is_still_fatal_even_when_reply_is_ready(monkeypatch):
    module, trainer, _ = status_poll_fixture(monkeypatch)
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kwargs: (refs, []))

    def failed(ref):
        raise RuntimeError("controller failed")

    monkeypatch.setattr(module.ray, "get", failed)
    with pytest.raises(RuntimeError, match="controller failed"):
        trainer._poll_adaptive_buffer()
