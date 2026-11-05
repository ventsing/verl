# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Exercise a fixed startup cohort across real sampler metadata transitions."""

import json
from copy import deepcopy

import pytest

from verl.experimental.trajectory_async import startup_sampler, tq_sampler
from verl.trainer.ppo.v1 import replay_buffer


@pytest.fixture
def store(monkeypatch):
    data = {"train": {}, "val": {}}
    monkeypatch.setattr(replay_buffer.tq, "kv_list", lambda: deepcopy(data))

    def clear(partition_id, keys):
        for key in keys:
            data[partition_id].pop(key, None)

    monkeypatch.setattr(tq_sampler.tq, "kv_clear", clear)

    def no_reward_access(*args, **kwargs):
        raise AssertionError("Startup reservation must not fetch rewards or tensor data")

    monkeypatch.setattr(replay_buffer.tq, "kv_batch_get", no_reward_access)
    return data


def make_sampler(size=2, **kwargs):
    options = {"rollout_n": 4, "val_rollout_n": 1, "startup_batch_groups": size,
               "selection_strategy": "mixed_ready", "selection_seed": 42}
    options.update(kwargs.pop("sampler_kwargs", {}))
    return startup_sampler.StartupCohortReplayBuffer(
        trainer_mode="laminar", trainer_config={}, sampler_kwargs=options,
        max_off_policy_threshold=1, max_off_policy_strategy="none", **kwargs,
    )


def prompt(store, uid, status="running", partition="train", step=1):
    store[partition][uid] = {"is_prompt": True, "status": status, "global_steps": step}


def response(store, uid, session, partition="train", behavior=0):
    store[partition][f"{uid}_{session}_0"] = {
        "status": "success", "global_steps": behavior, "min_global_steps": behavior,
        "max_global_steps": behavior, "response_len": 8,
    }


def finish(store, uid, partition="train", behavior=0):
    store[partition][uid]["status"] = "finished"
    for session in range(1 if partition == "val" else 4):
        response(store, uid, session, partition, behavior)


def sampleable(sampler, partition="train"):
    sampler._sync_metadata_from_transfer_queue()
    reasons = sampler._terminal_eviction_reasons(1, partition)
    return sampler._sampleable_terminal_keys(partition, reasons)


def test_waits_for_reserved_siblings_and_terminal_markers_then_keeps_fast_groups(store):
    for uid in ["first", "second", "fast_a", "fast_b"]:
        prompt(store, uid)
    finish(store, "fast_a")
    finish(store, "fast_b")
    sampler = make_sampler()
    assert sampleable(sampler) == set()
    assert sampler._startup_cohort == ("first", "second")
    for session in range(3):
        response(store, "first", session)
    assert sampleable(sampler) == set()
    response(store, "first", 3)
    assert sampleable(sampler) == set()
    store["train"]["first"]["status"] = "finished"
    assert sampleable(sampler) == {"first"}
    finish(store, "second")
    batch, metrics = sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert batch.keys == [f"{uid}_{session}_0" for uid in ["first", "second"] for session in range(4)]
    assert sampler._startup_complete
    assert metrics["trajectory/train/startup/cohort_batch"] == 1
    assert "fast_a" in store["train"] and "fast_b" in store["train"]
    later, metrics = sampler.sample(global_steps=100, partition_id="train", batch_size=2)
    assert {key.rsplit("_", 2)[0] for key in later.keys} == {"fast_a", "fast_b"}
    assert metrics["trajectory/train/startup/cohort_batch"] == 0
    assert metrics["trajectory/train/behavior_policy/lag_max"] == 99


def test_metadata_reordering_and_new_ready_groups_do_not_replace_reserved_prompts(store):
    for uid in ["slow_a", "slow_b", "fast"]:
        prompt(store, uid)
    finish(store, "fast")
    sampler = make_sampler()
    assert sampleable(sampler) == set()
    store["train"] = dict(reversed(list(store["train"].items())))
    prompt(store, "new_fast", step=-100)
    finish(store, "new_fast")
    assert sampleable(sampler) == set()
    assert sampler._startup_cohort == ("slow_a", "slow_b")
    finish(store, "slow_b")
    finish(store, "slow_a")
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert {key.rsplit("_", 2)[0] for key in batch.keys} == {"slow_a", "slow_b"}


def test_failure_eviction_refills_and_replaces_only_failed_reserved_member(store):
    for uid in ["failed", "slow", "replacement", "other"]:
        prompt(store, uid)
    finish(store, "replacement")
    finish(store, "other")
    refills = []
    sampler = make_sampler(refill_fn=refills.append)
    assert sampleable(sampler) == set()
    store["train"]["failed"]["status"] = "failure"
    finish(store, "slow")
    batch, metrics = sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert {key.rsplit("_", 2)[0] for key in batch.keys} == {"slow", "replacement"}
    assert refills == [1]
    assert metrics["trajectory/train/startup/evicted_groups"] == 1
    assert "other" in store["train"] and "failed" not in store["train"]


def test_validation_does_not_consume_startup_reservation(store):
    for uid in ["first", "second"]:
        prompt(store, uid)
    prompt(store, "validation", partition="val")
    finish(store, "validation", partition="val")
    sampler = make_sampler()
    batch, metrics = sampler.sample(global_steps=0, partition_id="val", batch_size=1)
    assert batch.keys == ["validation_0_0"]
    assert "trajectory/train/startup/cohort_batch" not in metrics
    assert not sampler._startup_complete and sampler._startup_cohort == ()
    finish(store, "second")
    assert sampleable(sampler) == {"second"}
    assert sampler._startup_cohort == ("first", "second")


def test_missing_reserved_metadata_fails_instead_of_waiting_forever(store):
    for uid in ["first", "second"]:
        prompt(store, uid)
    sampler = make_sampler()
    sampleable(sampler)
    store["train"].pop("first")
    with pytest.raises(RuntimeError, match="disappeared without terminal eviction"):
        sampler._sync_metadata_from_transfer_queue()
    assert not sampler._startup_complete


def test_wait_rechecks_producer_health(store, monkeypatch):
    for uid in ["first", "second", "fast_a", "fast_b"]:
        prompt(store, uid)
    finish(store, "fast_a")
    finish(store, "fast_b")
    sampler = make_sampler()

    def fail_next_poll(partition_id, last_debug_time):
        sampler.health_check = lambda: (_ for _ in ()).throw(RuntimeError("producer failed"))
        return last_debug_time

    monkeypatch.setattr(sampler, "_wait_for_next_poll", fail_next_poll)
    with pytest.raises(RuntimeError, match="producer failed"):
        sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert not sampler._startup_complete


def test_steady_selection_delegates_to_existing_mixed_ready_policy(store):
    sampler = make_sampler()
    sampler._startup_complete = True
    reference = tq_sampler.TrajectoryReplayBuffer(
        trainer_mode="laminar", trainer_config={},
        sampler_kwargs={"rollout_n": 4, "selection_strategy": "mixed_ready", "selection_seed": 42},
        max_off_policy_threshold=1, max_off_policy_strategy="none",
    )
    for i in range(8):
        prompt(store, f"group{i}")
        finish(store, f"group{i}")
    sampler._sync_metadata_from_transfer_queue()
    reference._sync_metadata_from_transfer_queue()
    for i in range(8):
        sampler.ready_groups["train"][f"group{i}"].complete_time = i
        reference.ready_groups["train"][f"group{i}"].complete_time = i
    keys = {f"group{i}" for i in range(8)}
    assert sampler._select_prompt_uids("train", keys, 2)[0] == reference._select_prompt_uids("train", keys, 2)[0]


@pytest.mark.parametrize("size", [None, True, False, 0, -1, 1.5, "2"])
def test_invalid_startup_group_count_rejected(store, size):
    with pytest.raises(ValueError, match="positive integer"):
        make_sampler(size=size)


def test_training_batch_mismatch_rejected_before_polling(store):
    sampler = make_sampler(size=2)
    with pytest.raises(ValueError, match="training prompt batch size"):
        sampler.sample(global_steps=1, partition_id="train", batch_size=1)


def test_remembers_admission_observed_before_a_full_cohort_is_available(store):
    prompt(store, "first")
    sampler = make_sampler()
    assert sampleable(sampler) == set()
    assert sampler._startup_cohort == ()
    prompt(store, "second")
    prompt(store, "fast")
    finish(store, "fast")
    store["train"] = dict(reversed(list(store["train"].items())))
    eligible = sampleable(sampler)
    assert eligible == {"fast"}
    assert not sampler._has_enough_samples(1, "train", 2, eligible)
    assert sampler._startup_cohort == ("first", "fast")
    # The first observation of "first" is retained. Metadata order for newly
    # observed prompts is explicitly not a certified dataloader-order signal.


def test_materialization_failure_does_not_mark_startup_complete(store, monkeypatch):
    for uid in ["first", "second"]:
        prompt(store, uid)
        finish(store, uid)
    sampler = make_sampler()

    def fail_store(*args, **kwargs):
        raise RuntimeError("store failed")

    monkeypatch.setattr(tq_sampler.TrajectoryReplayBuffer, "_materialize_batch", fail_store)
    with pytest.raises(RuntimeError, match="store failed"):
        sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert not sampler._startup_complete
    assert not sampler._startup_materializing


def test_logs_reserved_cohort_before_it_completes_without_reward_fields(store, tmp_path):
    for uid in ["first", "second"]:
        prompt(store, uid)
    sampler = make_sampler(sampler_kwargs={"trajectory_log_dir": str(tmp_path)})
    assert sampleable(sampler) == set()
    events = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "startup_cohort_reserved"
    assert events[-1]["uids"] == ["first", "second"]
    for uid in ["second", "first"]:
        finish(store, uid)
    sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    events = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    cohort_events = [row for row in events if row["event"].startswith("startup_cohort_")]
    assert [row["event"] for row in cohort_events] == ["startup_cohort_reserved", "startup_cohort_consumed"]
    assert all(row["uids"] == ["first", "second"] for row in cohort_events)
    assert all(not any("reward" in key or "length" in key for key in row) for row in cohort_events)
