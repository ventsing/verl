# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Wall-time stopping must preserve the completed-checkpoint boundary."""

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.trajectory_async import checkpoint_wall_time
from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback


@pytest.fixture
def context(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(checkpoint_wall_time, "time", SimpleNamespace(monotonic=lambda: now[0]))
    config = OmegaConf.create({
        "trainer": {"checkpoint_wall_time_hours": 1, "save_freq": 5,
                    "checkpoint_callback_class": "verl.experimental.trajectory_async.checkpoint_wall_time."
                    "WallTimeCheckpointCallback"},
        "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
    })
    directory = tmp_path / "global_step_55"
    directory.mkdir()
    (directory / "data.pt").write_bytes(b"dataloader fixture")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("55\n")
    trainer = SimpleNamespace(global_steps=55, total_training_steps=1000000)
    return now, config, directory, trainer


def test_native_factory_stops_only_after_elapsed_budget_and_completed_save(context):
    now, config, directory, trainer = context
    callback = build_checkpoint_callback(config)
    callback.on_save(trainer, 55, str(directory))
    assert trainer.total_training_steps == 1000000
    assert not (directory.parent / "wall_time_stop.json").exists()
    now[0] += 3600
    callback.on_save(trainer, 55, str(directory))
    assert trainer.total_training_steps == trainer.global_steps == 55
    receipt = json.loads((directory.parent / "wall_time_stop.json").read_text())
    assert receipt["elapsed_seconds"] == receipt["budget_seconds"] == 3600
    assert receipt["stop_after_step"] == 55
    callback.on_save(trainer, 55, str(directory))
    assert trainer.total_training_steps == 55


@pytest.mark.parametrize("hours", [0, -1, float("nan"), float("inf"), 1e308, True, "36", None])
def test_invalid_duration_rejected(context, hours):
    _, config, _, _ = context
    config.trainer.checkpoint_wall_time_hours = hours
    with pytest.raises(ValueError):
        build_checkpoint_callback(config)


@pytest.mark.parametrize("problem", ["missing_data", "wrong_tracker", "wrong_step", "async_save"])
def test_invalid_save_never_shortens_training(context, problem):
    now, config, directory, trainer = context
    callback = build_checkpoint_callback(config)
    now[0] += 3601
    if problem == "missing_data":
        (directory / "data.pt").unlink()
    elif problem == "wrong_tracker":
        (directory.parent / "latest_checkpointed_iteration.txt").write_text("50\n")
    elif problem == "wrong_step":
        trainer.global_steps = 56
    with pytest.raises((ValueError, RuntimeError)):
        callback.on_save(trainer, 55, str(directory), async_save=problem == "async_save")
    assert trainer.total_training_steps == 1000000
    assert not (directory.parent / "wall_time_stop.json").exists()


@pytest.mark.parametrize("setting", ["save_disabled", "async_checkpoint_config"])
def test_unsupported_checkpoint_config_rejected(context, setting):
    _, config, _, _ = context
    if setting == "save_disabled":
        config.trainer.save_freq = 0
    else:
        config.actor_rollout_ref.actor.checkpoint.async_save = True
    with pytest.raises(ValueError):
        build_checkpoint_callback(config)
