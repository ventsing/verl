# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""The synchronous control must neither prefetch nor consume stale experience."""

from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1 import trainer_laminar
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_laminar import PPOTrainerLaminar


def make_trainer(monkeypatch):
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.synchronous = True
    trainer.global_steps = 3
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            v1=SimpleNamespace(laminar=SimpleNamespace(weight_budget=SimpleNamespace(capacity_timeout_s=0.01)))
        )
    )
    replica = {"version": 2, "healthy": True, "active": 0, "reserved": 0, "pulling": False}
    trainer.status = {"pending": 0, "replicas": [replica.copy() for _ in range(4)]}
    trainer.laminar_controller = SimpleNamespace(status=SimpleNamespace(remote=lambda: trainer.status))
    trainer._pending_prompt_count = lambda: 0
    monkeypatch.setattr(trainer_laminar.ray, "get", lambda x: x)
    return trainer


def test_submits_one_batch_without_background_warmup(monkeypatch):
    trainer = make_trainer(monkeypatch)
    submitted = []
    monkeypatch.setattr(PPOTrainer, "_add_batch_to_generate", lambda self: submitted.append(self.global_steps))
    trainer.on_train_begin()
    assert submitted == []
    assert trainer.prepare_step() == {"laminar/synchronous": 1.0}
    assert submitted == [3]
    with pytest.raises(RuntimeError, match="cannot refill"):
        trainer._add_prompts_to_generate(1)


@pytest.mark.parametrize(
    "field,value", [("version", 1), ("active", 1), ("reserved", 1), ("pulling", True), ("healthy", False)]
)
def test_rejects_non_idle_or_old_replica(monkeypatch, field, value):
    trainer = make_trainer(monkeypatch)
    trainer.status["replicas"][3][field] = value
    with pytest.raises(RuntimeError, match="not idle"):
        trainer.prepare_step()


def test_weight_barrier_waits_for_all_replicas(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.status["replicas"][3]["version"] = 1
    sleeps = []

    def complete_pull(seconds):
        sleeps.append(seconds)
        trainer.status["replicas"][3]["version"] = 2

    monkeypatch.setattr(trainer_laminar.time, "sleep", complete_pull)
    trainer._wait_synchronous_version(2)
    assert sleeps == [0.1]


def test_weight_barrier_times_out_instead_of_admitting_stale_batch(monkeypatch):
    trainer = make_trainer(monkeypatch)
    trainer.status["replicas"][3]["version"] = 1
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(trainer_laminar.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="timed out"):
        trainer._wait_synchronous_version(2)


def test_behavior_logprobs_require_current_rollout_version(monkeypatch):
    trainer = make_trainer(monkeypatch)
    monkeypatch.setattr(PPOTrainer, "_compute_old_log_prob", lambda self, batch, metrics: batch)
    batch = SimpleNamespace(tags=[{"min_global_steps": 2, "max_global_steps": 2}])
    assert trainer._compute_old_log_prob(batch, {}) is batch
    batch.tags[0].update(min_global_steps=1, max_global_steps=1)
    with pytest.raises(ValueError, match="requires behavior version 2"):
        trainer._compute_old_log_prob(batch, {})
