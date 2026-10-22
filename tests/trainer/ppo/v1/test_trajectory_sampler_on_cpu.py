# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise trainer-side trajectory aggregation over successive TQ snapshots."""

import json
from copy import deepcopy

import pytest

from verl.experimental.trajectory_async import tq_sampler
from verl.trainer.ppo.v1 import replay_buffer


@pytest.fixture
def store(monkeypatch):
    data = {"train": {}, "val": {}}
    monkeypatch.setattr(replay_buffer.tq, "kv_list", lambda: deepcopy(data))

    def clear(partition_id, keys):
        for key in keys:
            data[partition_id].pop(key, None)

    monkeypatch.setattr(tq_sampler.tq, "kv_clear", clear)
    return data


def make_sampler(**kwargs):
    sampler_kwargs = kwargs.pop("sampler_kwargs", {"rollout_n": 2, "val_rollout_n": 1})
    return tq_sampler.TrajectoryReplayBuffer(
        trainer_mode="separate_async",
        trainer_config={},
        sampler_kwargs=sampler_kwargs,
        max_off_policy_threshold=8,
        max_off_policy_strategy=kwargs.pop("max_off_policy_strategy", "drop"),
        **kwargs,
    )


def test_laminar_has_no_configured_staleness_barrier(store):
    sampler = make_sampler(max_off_policy_strategy="none")
    prompt(store, "long", status="running", step=1)
    prompt(store, "ready", status="finished", step=1)
    response(store, "ready", 0, step=1)
    response(store, "ready", 1, step=1)
    batch, _ = sampler.sample(global_steps=50, partition_id="train", batch_size=1)
    assert set(batch.keys) == {"ready_0_0", "ready_1_0"}
    assert "long" in store["train"]


def test_producer_failure_interrupts_sampler_wait(store):
    sampler = make_sampler()

    def fail():
        raise RuntimeError("producer failed")

    sampler.health_check = fail
    with pytest.raises(RuntimeError, match="producer failed"):
        sampler.sample(global_steps=1, partition_id="train", batch_size=1)


def prompt(store, uid, status="running", step=1, partition="train"):
    store[partition][uid] = {"is_prompt": True, "status": status, "global_steps": step}


def response(store, uid, session, step=1, partition="train", output_index=0):
    store[partition][f"{uid}_{session}_{output_index}"] = {
        "status": "success",
        "global_steps": step,
        "min_global_steps": step,
        "response_len": 8,
    }


def test_ingests_before_siblings_finish_and_never_replays_sampled_rows(store):
    sampler = make_sampler()
    prompt(store, "a")
    response(store, "a", 1)
    sampler._sync_metadata_from_transfer_queue()
    assert sampler.aggregators["train"].num_buffered_trajectories == 1
    assert sampler.aggregators["train"].num_partial_groups == 1
    assert not sampler.ready_groups["train"]

    # A second poll sees the same stored row, not a second trajectory.
    sampler._sync_metadata_from_transfer_queue()
    assert sampler.aggregators["train"].total_added == 1
    response(store, "a", 0)
    sampler._sync_metadata_from_transfer_queue()
    assert "a" in sampler.ready_groups["train"]
    assert not sampler._sampleable_terminal_keys("train", (set(), set(), set(), {}))

    # The terminal marker prevents a late producer status write after retirement.
    prompt(store, "a", status="finished")
    batch, metrics = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
    assert batch.keys == ["a_0_0", "a_1_0"]
    assert metrics["trajectory/train/aggregator/groups_completed"] == 1
    assert "a" not in store["train"]
    assert "a_0_0" in store["train"]  # tensor rows are retained for the actor
    sampler._sync_metadata_from_transfer_queue()
    assert not sampler.ready_groups["train"]
    assert sampler.aggregators["train"].total_added == 2


def test_completed_groups_are_selected_in_completion_order(store):
    sampler = make_sampler()
    prompt(store, "slow")
    response(store, "slow", 0)
    prompt(store, "fast", status="finished")
    response(store, "fast", 1)
    response(store, "fast", 0)
    sampler._sync_metadata_from_transfer_queue()
    response(store, "slow", 1)
    prompt(store, "slow", status="finished")
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
    assert batch.keys == ["fast_0_0", "fast_1_0"]
    assert "slow" in sampler.ready_groups["train"]


def test_ready_mixing_keeps_oldest_and_never_waits_for_incomplete_groups(store):
    sampler = make_sampler(sampler_kwargs={"rollout_n": 2, "selection_strategy": "mixed_ready", "selection_seed": 7})
    for uid in ["a", "b", "c", "d", "e", "f"]:
        prompt(store, uid, status="finished")
        response(store, uid, 0)
        response(store, uid, 1)
    prompt(store, "unfinished")
    response(store, "unfinished", 0)
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=2)
    assert batch.keys == ["a_0_0", "a_1_0", "d_0_0", "d_1_0"]
    assert "unfinished" in store["train"]
    assert len(sampler.ready_groups["train"]) == 4


def test_ready_mixing_does_not_delay_an_exactly_ready_batch(store):
    sampler = make_sampler(sampler_kwargs={"rollout_n": 2, "selection_strategy": "mixed_ready"})
    prompt(store, "ready", status="finished")
    response(store, "ready", 0)
    response(store, "ready", 1)
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
    assert batch.keys == ["ready_0_0", "ready_1_0"]


@pytest.mark.parametrize("failure", [True, False])
def test_failed_or_stale_groups_are_cleared_and_refilled(store, failure):
    prompt(store, "old", status="failure" if failure else "finished", step=0)
    response(store, "old", 0, step=0)
    if not failure:
        response(store, "old", 1, step=0)
    calls = []

    def refill(count):
        calls.append(count)
        prompt(store, "fresh", status="finished", step=10)
        response(store, "fresh", 0, step=10)
        response(store, "fresh", 1, step=10)

    sampler = make_sampler(refill_fn=refill)
    batch, _ = sampler.sample(global_steps=10, partition_id="train", batch_size=1)
    assert calls == [1]
    assert batch.keys == ["fresh_0_0", "fresh_1_0"]
    assert all(not key.startswith("old") for key in store["train"])
    assert sampler.aggregators["train"].num_partial_groups == 0


def test_validation_uses_its_own_group_size(store):
    prompt(store, "val", status="finished", partition="val")
    response(store, "val", 0, partition="val")
    batch, _ = make_sampler().sample(global_steps=1, partition_id="val", batch_size=1)
    assert batch.keys == ["val_0_0"]


def test_validation_failure_is_reported_instead_of_waiting_forever(store):
    prompt(store, "bad", status="failure", partition="val")
    with pytest.raises(RuntimeError, match="Validation trajectories failed"):
        make_sampler().sample(global_steps=1, partition_id="val", batch_size=1)


def test_wrong_group_size_is_reported_instead_of_waiting_forever(store):
    prompt(store, "bad", status="finished")
    response(store, "bad", 0)
    with pytest.raises(RuntimeError, match="Finished prompts lack 2 trajectories"):
        make_sampler().sample(global_steps=1, partition_id="train", batch_size=1)


def test_unrelated_partition_does_not_apply_training_group_size(store):
    store["other-app"] = {}
    prompt(store, "foreign", status="finished", partition="other-app")
    response(store, "foreign", 0, partition="other-app")
    prompt(store, "ready", status="finished")
    response(store, "ready", 0)
    response(store, "ready", 1)
    sampler = make_sampler()
    batch, _ = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
    assert batch.keys == ["ready_0_0", "ready_1_0"]
    assert not sampler.aggregators["other-app"].total_added
    assert "foreign" in store["other-app"]


def test_multiple_outputs_per_session_are_rejected(store):
    prompt(store, "bad")
    response(store, "bad", 0, output_index=1)
    with pytest.raises(ValueError, match="single-output"):
        make_sampler()._sync_metadata_from_transfer_queue()


def test_length_log_includes_partial_arrivals_once_and_marks_selection(store, tmp_path):
    sampler = make_sampler(
        sampler_kwargs={
            "rollout_n": 2,
            "val_rollout_n": 1,
            "trajectory_log_dir": str(tmp_path),
            "response_limit": 24576,
        }
    )
    prompt(store, "long")
    response(store, "long", 0)
    store["train"]["long_0_0"]["response_len"] = 1024
    sampler._sync_metadata_from_transfer_queue()
    sampler._sync_metadata_from_transfer_queue()
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [(r["key"], r["response_tokens"]) for r in rows] == [("long_0_0", 1024)]
    response(store, "long", 1)
    store["train"]["long_1_0"]["response_len"] = 24576
    prompt(store, "long", status="finished")
    _, metrics = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
    assert metrics["trajectory/train/length/p50"] == 12800
    assert metrics["trajectory/train/length/ge_16384_ratio"] == 0.5
    assert metrics["trajectory/train/length/budget_reached_ratio"] == 0.5
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "arrived"]) == 2
    assert {r["key"] for r in rows if r["event"] == "selected"} == {"long_0_0", "long_1_0"}
