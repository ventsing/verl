# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in wall-time stopping at a completed synchronous v1 checkpoint.

Set trainer.checkpoint_callback_class to this module's WallTimeCheckpointCallback
and add trainer.checkpoint_wall_time_hours. Elapsed time starts when the trainer
constructs the callback, including initialization and validation. Stopping occurs
at the next completed checkpoint after that duration; an update is never killed
mid-flight. A resumed driver starts a new uninterrupted observation interval.
"""

import json
import math
import time
from pathlib import Path

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback


class WallTimeCheckpointCallback(CheckpointCallback):
    """End the v1 training loop after the wall-time budget and a durable save."""

    def __init__(self, config=None):
        super().__init__(config=config)
        hours = config.trainer.get("checkpoint_wall_time_hours")
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
            raise ValueError("trainer.checkpoint_wall_time_hours must be a finite positive number")
        seconds = hours * 3600
        if not math.isfinite(seconds):
            raise ValueError("Checkpoint wall-time budget in seconds must remain finite")
        if config.trainer.save_freq <= 0:
            raise ValueError("Wall-time stopping requires a positive trainer.save_freq")
        if config.actor_rollout_ref.actor.get("checkpoint", {}).get("async_save", False):
            raise ValueError("Wall-time stopping requires synchronous checkpoint saves")
        self.budget_seconds = seconds
        self.started_at = time.monotonic()
        self.stop_requested = False

    def on_save(self, trainer, global_step: int, checkpoint_dir: str, async_save: bool = False, **kwargs):
        if self.stop_requested:
            return
        if async_save:
            raise ValueError("An asynchronous checkpoint cannot establish a durable stopping boundary")
        elapsed = time.monotonic() - self.started_at
        if elapsed < self.budget_seconds:
            return
        directory = Path(checkpoint_dir)
        if directory.name != f"global_step_{global_step}" or global_step != trainer.global_steps:
            raise ValueError("Checkpoint and current trainer step must agree")
        if not (directory / "data.pt").is_file():
            raise RuntimeError("Completed checkpoint must include its dataloader state")
        tracker = directory.parent / "latest_checkpointed_iteration.txt"
        if int(tracker.read_text().strip()) != global_step:
            raise RuntimeError("Latest-checkpoint tracker must identify the completed stopping checkpoint")
        previous_limit = trainer.total_training_steps
        if previous_limit < global_step:
            raise ValueError("The checkpoint step exceeds the current training limit")
        receipt = {
            "reason": "wall_time_budget_reached_at_completed_checkpoint",
            "elapsed_seconds": elapsed,
            "budget_seconds": self.budget_seconds,
            "global_step": global_step,
            "checkpoint_dir": str(directory),
            "previous_total_training_steps": previous_limit,
            "stop_after_step": global_step,
            "includes_initialization_and_validation": True,
            "asynchronous_checkpoint": False,
        }
        with (directory.parent / "wall_time_stop.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        # TrainerBase.fit checks this limit before its next iteration. Laminar
        # on_step_end also sees the new limit and stops its prompt feeder.
        trainer.total_training_steps = global_step
        self.stop_requested = True
