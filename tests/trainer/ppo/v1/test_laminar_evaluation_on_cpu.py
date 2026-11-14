# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Dedicated evaluation keeps one policy version while training advances."""

from threading import Event
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import laminar_evaluation as evaluation
from verl.trainer.ppo.v1 import trainer_laminar as laminar
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_laminar import PPOTrainerLaminar


def config():
    return OmegaConf.create(
        {
            "trainer": {
                "v1": {
                    "laminar": {
                        "evaluation": {
                            "enabled": True,
                            "nnodes": 1,
                            "n_gpus_per_node": 2,
                            "tensor_model_parallel_size": 2,
                            "num_workers": 1,
                            "timeout_s": 10,
                        }
                    }
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "nnodes": 2,
                    "n_gpus_per_node": 8,
                    "tensor_model_parallel_size": 4,
                    "max_num_seqs": 8,
                    "enforce_eager": False,
                    "cudagraph_capture_sizes": [1, 2, 4, 8],
                    "checkpoint_engine": {"backend": "mooncake_relay"},
                    "agent": {"num_workers": 8},
                }
            },
        }
    )


def remote(fn):
    return SimpleNamespace(remote=fn)


def runner(monkeypatch, concurrency=1):
    cfg = evaluation.evaluation_config(config())
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 2 * concurrency
    cfg.trainer.v1.laminar.evaluation.max_concurrent_versions = concurrency
    versions, pins, released = [], {}, []
    controller = SimpleNamespace(
        register_version=remote(lambda v: versions.append(v)),
        unregister_version=remote(lambda v: None),
        set_minimum_version=remote(lambda v: None),
        status=remote(lambda: {}),
    )
    relay = SimpleNamespace(
        retain=remote(lambda v, owner: pins.update({owner: v})),
        release_owner=remote(lambda owner: released.append(pins.pop(owner))),
    )
    # Ray handles are hashable; use a normal class for the fake relay as well.
    relay = type("Relay", (), vars(relay))()
    manager = SimpleNamespace(shutdown=lambda: None)
    trainer = SimpleNamespace(
        config=config(),
        global_steps=3,
        replay_buffer=object(),
        _init_dump_executor=lambda: None,
        _shutdown_dump_executor=lambda: None,
    )
    monkeypatch.setattr(evaluation.ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(PPOTrainer, "_maybe_log_val_generations", lambda *args, **kwargs: None)
    instance = evaluation.LaminarEvaluation(trainer, cfg, controller, manager, [relay])
    return instance, trainer, versions, pins, released


def test_evaluation_resource_config_is_independent():
    original = config()
    result = evaluation.evaluation_config(original)
    assert result.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert result.actor_rollout_ref.rollout.n_gpus_per_node == 2
    assert original.actor_rollout_ref.rollout.n_gpus_per_node == 8
    assert original.actor_rollout_ref.rollout.agent.num_workers == 8


def test_evaluation_admission_extends_graph_gears_without_changing_training():
    original = config()
    options = original.trainer.v1.laminar.evaluation
    options.max_num_seqs = 32
    options.max_inflight_per_replica = 24
    result = evaluation.evaluation_config(original)
    assert result.actor_rollout_ref.rollout.max_num_seqs == 32
    assert list(result.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8, 16, 24]
    assert original.actor_rollout_ref.rollout.max_num_seqs == 8
    assert list(original.actor_rollout_ref.rollout.cudagraph_capture_sizes) == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_inflight_per_replica": 9},
        {"max_num_seqs": 0},
        {"max_inflight_per_replica": -1},
        {"cudagraph_capture_sizes": [1, 4]},
        {"overflow_policy": "unbounded"},
        {"allocation_policy": "invalid"},
    ],
)
def test_invalid_evaluation_capacity_and_policy(overrides):
    cfg = config()
    cfg.trainer.v1.laminar.evaluation.update(overrides)
    with pytest.raises(ValueError):
        evaluation.evaluation_config(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nnodes", 0),
        ("n_gpus_per_node", -1),
        ("num_workers", 0),
        ("tensor_model_parallel_size", 3),
        ("timeout_s", 0),
    ],
)
def test_rejects_invalid_placement_and_timeout(field, value):
    cfg = config()
    cfg.trainer.v1.laminar.evaluation[field] = value
    with pytest.raises(ValueError):
        evaluation.evaluation_config(cfg)


def test_background_evaluation_keeps_version_and_releases_pins(monkeypatch):
    instance, trainer, versions, pins, released = runner(monkeypatch)
    entered, resume = Event(), Event()
    snapshots = []

    def validate(view, **kwargs):
        snapshots.append((view.global_steps, view.config, view.replay_buffer))
        entered.set()
        assert resume.wait(5)
        assert view.global_steps == snapshots[-1][0]
        return {"val-core/score": 0.75}

    monkeypatch.setattr(PPOTrainer, "_validate", validate)
    try:
        instance.submit(3)
        assert entered.wait(5)
        trainer.global_steps = 20
        assert instance.busy
        assert instance.collect() == {}
        assert list(pins.values()) == [3]
        with pytest.raises(RuntimeError, match="collected"):
            instance.submit(20)
        resume.set()
        metrics = instance.collect(wait=True)
        assert metrics["evaluation/policy_version"] == 3
        assert metrics["val-core/score"] == 0.75
        assert snapshots[0][1] is instance.config
        assert snapshots[0][2] is not trainer.replay_buffer
        assert not pins and released == [3]
        instance.submit(20)
        # A second validation must announce a new exact version, not restart
        # the controller or silently use the relay's latest publication.
        assert instance.collect(wait=True)["evaluation/policy_version"] == 20
        assert versions == [3, 20]
        assert not pins and released == [3, 20]
    finally:
        resume.set()
        instance.close()


def test_evaluation_error_is_propagated_and_unpins_weights(monkeypatch):
    instance, _, _, pins, released = runner(monkeypatch)

    def fail(view, **kwargs):
        raise ValueError("scorer failed")

    monkeypatch.setattr(PPOTrainer, "_validate", fail)
    try:
        instance.submit(7)
        with pytest.raises(ValueError, match="scorer failed"):
            instance.collect(wait=True)
        assert not pins and released == [7]
    finally:
        instance.close()


def test_validation_schedule_defers_busy_jobs_but_waits_at_end():
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.config = OmegaConf.create({"trainer": {"val_only": False}})
    trainer.evaluation_config = config()
    trainer._laminar_training_started = True
    trainer.global_steps, trainer.total_training_steps = 4, 10
    submitted, waits, deferred, discarded = [], [], [], []
    state = {"busy": True}

    def collect(wait=False):
        waits.append(wait)
        if wait:
            state["busy"] = False
        return {}

    class Evaluator:
        skipped = 0
        busy = property(lambda self: state["busy"])

    trainer.evaluation = Evaluator()
    trainer.evaluation.collect = collect
    trainer.evaluation.submit = submitted.append
    trainer.evaluation.defer = deferred.append
    trainer.evaluation.discard_pending = lambda replaced=False: discarded.append(replaced)
    trainer.evaluation.queue_metrics = lambda: {"evaluation/deferred": len(deferred)}
    assert trainer._validate()["evaluation/deferred"] == 1
    assert deferred == [4]
    assert not submitted
    trainer.global_steps = 10
    trainer._validate()
    assert submitted == [10] and waits[-2:] == [True, True]
    # Initial validation after checkpoint resume must still block even though
    # its policy version is greater than zero (the base trainer requires metrics).
    trainer._laminar_training_started = False
    trainer.global_steps = 6
    trainer._validate()
    assert submitted[-1] == 6 and waits[-2:] == [True, True]


@pytest.mark.parametrize("collection_point", ["step_end", "final_barrier"])
def test_final_validation_preserves_previous_version_score(monkeypatch, collection_point):
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.config = config()
    trainer.config.trainer.val_only = False
    trainer.config.trainer.v1.laminar.prompt_buffer_groups = 64
    trainer.evaluation_config = config()
    trainer._laminar_training_started = True
    trainer.global_steps = trainer.total_training_steps = 100
    logged, submitted = [], []
    trainer.logger = SimpleNamespace(log=lambda data, step: logged.append((step, dict(data))))
    older = {"evaluation/policy_version": 95, "evaluation/samples": 512, "accuracy": 0.6875}
    final = {"evaluation/policy_version": 100, "evaluation/samples": 512, "accuracy": 0.671875}
    results = iter([older, {}, final] if collection_point == "step_end" else [older, final])
    trainer.evaluation = SimpleNamespace(
        collect=lambda wait=False: next(results),
        busy=False,
        discard_pending=lambda replaced=False: None,
        submit=submitted.append,
        queue_metrics=lambda: {"evaluation/skipped": 0},
    )
    if collection_point == "step_end":
        status = dict.fromkeys(
            [
                "completed",
                "migrations",
                "pulls",
                "repack_checks",
                "pending",
                "replayed_experiences",
                "recoveries",
                "quarantined_replicas",
            ],
            0,
        )
        trainer._check_prompt_feeder = trainer._stop_prompt_feeder = lambda: None
        trainer._publish = lambda: {}
        trainer._pending_prompt_count = lambda: 0
        trainer._completion_rate_metrics = SimpleNamespace(update=lambda *args: {})
        trainer.timing_raw = {}
        trainer.laminar_controller = SimpleNamespace(
            notify_weights=remote(lambda version: None), status=remote(lambda: status)
        )
        trainer.evaluation_controller = SimpleNamespace(
            status=remote(lambda: {"evaluation_requests": {}, "replicas": [], "pulls": 0, "migrations": 0})
        )
        monkeypatch.setattr(laminar.ray, "get", lambda value: value)
        monkeypatch.setattr(laminar, "compute_replica_policy_metrics", lambda value: {})
        trainer.on_step_end()
        metrics = trainer._consume_sync_metrics()
    else:
        metrics = {}
    metrics.update(trainer._validate())
    trainer.logger.log(data=metrics, step=100)
    assert submitted == [100]
    assert [row["evaluation/policy_version"] for _, row in logged] == [95, 100]
    assert [row["accuracy"] for _, row in logged] == [0.6875, 0.671875]
    assert all(step == 100 and row["evaluation/samples"] == 512 for step, row in logged)


def test_pending_latest_is_pinned_replaced_and_evaluated_exactly(monkeypatch):
    instance, trainer, versions, pins, released = runner(monkeypatch)
    seen = []
    monkeypatch.setattr(PPOTrainer, "_validate", lambda view, **kwargs: seen.append(view.global_steps) or {})
    try:
        instance.defer(4)
        assert list(pins.values()) == [4]
        instance.defer(5)
        assert list(pins.values()) == [5] and released == [4]
        assert instance.coalesced == instance.skipped == 1
        assert instance.deferred == 2
        trainer.global_steps = 99
        assert instance.pump_pending()
        assert instance.collect(wait=True)["evaluation/policy_version"] == 5
        assert versions == seen == [5]
        assert not pins and released == [4, 5]
        assert not instance.pump_pending()
    finally:
        instance.close()


def test_pending_pin_failure_keeps_old_snapshot(monkeypatch):
    instance, _, _, pins, released = runner(monkeypatch)
    try:
        instance.defer(4)
        relay = instance.relays[0]
        retain = relay.retain.remote

        def failing_retain(version, owner):
            retain(version, owner)
            raise RuntimeError("relay unavailable")

        relay.retain = remote(failing_retain)
        with pytest.raises(RuntimeError, match="relay unavailable"):
            instance.defer(5)
        assert instance.pending[0] == 4 and list(pins.values()) == [4]
        assert instance.coalesced == 0
    finally:
        instance.close()
    assert not pins and released == [5, 4]


def test_skip_overflow_does_not_pin_pending_weights(monkeypatch):
    instance, _, _, pins, _ = runner(monkeypatch)
    try:
        instance.overflow_policy = "skip"
        instance.defer(4)
        assert instance.skipped == 1 and instance.pending is None and not pins
    finally:
        instance.close()


def test_cancelled_evaluation_releases_pins_without_scoring(monkeypatch):
    instance, _, _, pins, released = runner(monkeypatch)
    monkeypatch.setattr(
        PPOTrainer, "_validate", lambda view, **kwargs: pytest.fail("Cancelled evaluation must not score")
    )
    instance.cancelled.set()
    try:
        instance.submit(4)
        with pytest.raises(RuntimeError, match="cancelled"):
            instance.collect(wait=True)
        assert not pins and released == [4]
    finally:
        instance.close()


def test_dump_initialization_failure_still_unpins(monkeypatch):
    instance, trainer, _, pins, released = runner(monkeypatch)

    def fail():
        raise OSError("cannot create dump executor")

    trainer._init_dump_executor = fail
    try:
        instance.submit(9)
        with pytest.raises(OSError, match="dump executor"):
            instance.collect(wait=True)
        assert not pins and released == [9]
    finally:
        instance.close()


def test_completed_result_occupies_slot_until_collected(monkeypatch):
    instance, _, _, _, _ = runner(monkeypatch)
    monkeypatch.setattr(PPOTrainer, "_validate", lambda view, **kwargs: {"score": 1.0})
    try:
        instance.submit(3)
        instance.future.result(timeout=5)
        # Completion between a nonblocking poll and the admission check must
        # not admit a second job over an uncollected result.
        assert instance.busy
        assert instance.collect()["evaluation/policy_version"] == 3
        assert not instance.busy
    finally:
        instance.close()


def test_three_versions_use_distinct_partitions_and_complete_full_sets(monkeypatch):
    instance, trainer, versions, pins, released = runner(monkeypatch, concurrency=3)
    resume = Event()
    entered = {version: Event() for version in [1, 2, 3]}
    partitions = {}
    logged = []
    trainer.logger = SimpleNamespace(log=lambda data, step: logged.append(data))

    def validate(view, partition_id):
        partitions[view.global_steps] = partition_id
        entered[view.global_steps].set()
        assert resume.wait(5)
        view.replay_buffer.sampled_rows = 512
        return {"score": float(view.global_steps)}

    monkeypatch.setattr(PPOTrainer, "_validate", validate)
    try:
        for version in [1, 2, 3]:
            instance.submit(version)
        assert all(event.wait(5) for event in entered.values())
        assert instance.busy and len(pins) == 3
        assert len(set(partitions.values())) == 3
        trainer.global_steps = 80
        resume.set()
        last = instance.collect(wait=True)
        assert [row["evaluation/policy_version"] for row in logged] == [1, 2]
        assert last["evaluation/policy_version"] == 3
        assert all(row["evaluation/samples"] == 512 for row in logged + [last])
        assert not pins and sorted(released) == [1, 2, 3]
    finally:
        resume.set()
        instance.close()


def test_sampler_rejects_a_response_from_another_version(monkeypatch):
    batch = SimpleNamespace(keys=["uid_0_0"], tags=[{"min_global_steps": 1, "max_global_steps": 2}])
    monkeypatch.setattr(evaluation.ReplayBufferAsync, "sample", lambda *args: (batch, {}))
    sampler = evaluation.EvaluationReplayBuffer(config(), lambda: None, "val-evaluation-1")
    with pytest.raises(ValueError, match="changed policy version"):
        sampler.sample(1, "val-evaluation-1", 1)


@pytest.mark.parametrize("version,expected_rows", [(190, 500), (200, 5000)])
def test_final_validation_files_are_private_to_terminal_version(monkeypatch, version, expected_rows):
    import copy

    cfg = config()
    cfg.data = {"val_files": ["math500.parquet"], "val_max_samples": -1}
    cfg.trainer.v1.laminar.evaluation.final_val_files = ["math_lighteval.parquet"]
    trainer = SimpleNamespace(
        config=cfg,
        val_dataset=list(range(500)),
        total_training_steps=200,
        global_steps=version,
        tokenizer=object(),
        processor=None,
    )
    view = copy.copy(trainer)
    calls = []

    def dataset(files, data, tokenizer, processor, **kwargs):
        calls.append(list(files))
        assert list(data.val_files) == ["math_lighteval.parquet"]
        assert tokenizer is trainer.tokenizer and processor is None
        assert kwargs == {"is_train": False, "max_samples": -1}
        return list(range(5000))

    monkeypatch.setattr("verl.trainer.ppo.utils.create_rl_dataset", dataset)
    evaluation.prepare_validation_dataset(view)
    assert len(view.val_dataset) == expected_rows
    assert len(trainer.val_dataset) == 500
    assert list(trainer.config.data.val_files) == ["math500.parquet"]
    assert calls == ([["math_lighteval.parquet"]] if version == 200 else [])
    assert (view.config is trainer.config) == (version < 200)


def test_background_final_evaluation_scores_only_final_files(monkeypatch):
    instance, trainer, versions, pins, released = runner(monkeypatch)
    instance.config.data = {"val_files": ["math500.parquet"], "val_batch_size": 64, "val_max_samples": -1}
    instance.config.trainer.v1.laminar.evaluation.final_val_files = ["math_lighteval.parquet"]
    trainer.val_dataset = list(range(500))
    trainer.tokenizer, trainer.processor = object(), None
    trainer.total_training_steps = 200
    monkeypatch.setattr("verl.trainer.ppo.utils.create_rl_dataset", lambda *args, **kwargs: list(range(5000)))
    seen = []

    def validate(view, **kwargs):
        seen.append((view.global_steps, list(view.config.data.val_files), len(view.val_dataset)))
        view.replay_buffer.sampled_rows = len(view.val_dataset)
        return {"score": 1.0}

    monkeypatch.setattr(PPOTrainer, "_validate", validate)
    try:
        instance.submit(10)
        assert instance.collect(wait=True)["evaluation/samples"] == 500
        instance.submit(200)
        result = instance.collect(wait=True)
        assert result["evaluation/samples"] == 5000
        assert result["evaluation/policy_version"] == 200
        assert seen == [(10, ["math500.parquet"], 500), (200, ["math_lighteval.parquet"], 5000)]
        assert len(trainer.val_dataset) == 500
        assert list(instance.config.data.val_files) == ["math500.parquet"]
        assert not pins and released == [10, 200]
    finally:
        instance.close()


def test_blocking_policy_waits_for_one_slot_without_skipping_versions(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    instance, trainer, versions, pins, released = runner(monkeypatch, concurrency=3)
    instance.overflow_policy = "block"
    entered = {v: Event() for v in [10, 20, 30, 40]}
    resume = {v: Event() for v in entered}
    logged = []
    trainer.logger = SimpleNamespace(log=lambda data, step: logged.append(dict(data)))

    def validate(view, **kwargs):
        entered[view.global_steps].set()
        assert resume[view.global_steps].wait(5)
        return {"score": float(view.global_steps)}

    monkeypatch.setattr(PPOTrainer, "_validate", validate)
    try:
        for version in [10, 20, 30]:
            instance.submit(version)
        assert all(entered[v].wait(2) for v in [10, 20, 30])
        with ThreadPoolExecutor(max_workers=1) as caller:
            waiting = caller.submit(instance.wait_for_slot)
            assert not waiting.done()
            resume[20].set()
            metrics = waiting.result(timeout=2)
        assert metrics["evaluation/policy_version"] == 20
        assert set(instance.futures) == {10, 30} and not instance.busy
        instance.submit(40)
        assert entered[40].wait(2)
        for event in resume.values():
            event.set()
        final = instance.collect(wait=True)
        scored = [metrics, *logged, final]
        assert sorted(row["evaluation/policy_version"] for row in scored) == [10, 20, 30, 40]
        assert versions == [10, 20, 30, 40]
        assert not pins and sorted(released) == [10, 20, 30, 40]
        assert instance.skipped == instance.coalesced == 0 and instance.pending is None
    finally:
        for event in resume.values():
            event.set()
        instance.close()


def test_validation_backpressures_full_pool_instead_of_deferring():
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.config = OmegaConf.create({"trainer": {"val_only": False}})
    trainer.evaluation_config = config()
    trainer.evaluation_config.trainer.v1.laminar.evaluation.overflow_policy = "block"
    resolved = evaluation.evaluation_config(trainer.evaluation_config)
    assert resolved.trainer.v1.laminar.evaluation.overflow_policy == "block"
    trainer._laminar_training_started = True
    trainer.global_steps, trainer.total_training_steps = 40, 200
    state, submitted = {"busy": True}, []

    def slot():
        state["busy"] = False
        return {"evaluation/policy_version": 20}

    class Evaluator:
        busy = property(lambda self: state["busy"])

    trainer.evaluation = Evaluator()
    trainer.evaluation.collect = lambda wait=False: {}
    trainer.evaluation.wait_for_slot = slot
    trainer.evaluation.defer = lambda version: pytest.fail("A required version must never enter the latest queue")
    trainer.evaluation.discard_pending = lambda replaced=False: None
    trainer.evaluation.submit = submitted.append
    trainer.evaluation.queue_metrics = lambda: {"evaluation/skipped": 0}
    metrics = trainer._validate()
    assert submitted == [40]
    assert metrics == {"evaluation/policy_version": 20, "evaluation/skipped": 0}


@pytest.mark.parametrize("continuous", [False, True])
def test_evaluation_submission_does_not_wait_for_batch_tails(monkeypatch, continuous):
    import numpy as np

    from verl.trainer.ppo.v1 import trainer_base

    submitted = []
    registered = []
    trainer = SimpleNamespace(
        global_steps=10,
        _continuous_evaluation=continuous,
        val_dataloader=[{"raw_prompt": np.array([f"question-{i}", f"question-{i + 1}"])} for i in range(0, 6, 2)],
        agent_loop_manager=SimpleNamespace(generate_sequences=lambda batch: submitted.append(batch)),
    )
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: registered.append(kwargs))
    sizes = PPOTrainer._validation_batch_sizes(trainer, "val-evaluation-10")
    # Even with every response unfinished, later questions must be queued so
    # they can fill slots while the first batch has a long-running tail.
    assert next(sizes) == 2
    assert len(submitted) == (3 if continuous else 1)
    assert list(sizes) == [2, 2]
    assert len(submitted) == len(registered) == 3
    ids = [uid for entry in registered for uid in entry["keys"]]
    assert len(ids) == len(set(ids)) == 6
    assert all(entry["partition_id"] == "val-evaluation-10" for entry in registered)
    assert all(tag["global_steps"] == 10 for entry in registered for tag in entry["tags"])
    assert all(len(batch) == 2 for batch in submitted)
