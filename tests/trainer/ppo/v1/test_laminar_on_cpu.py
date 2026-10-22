# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Exercise independent pulls and running-request migration under races."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import BatchEncoding

from verl.experimental.trajectory_async.cpu_relay import CPUWeightRelay
from verl.experimental.trajectory_async.laminar_controller import LaminarController, consolidation_plan
from verl.trainer.ppo.v1 import trainer_base, trainer_laminar
from verl.trainer.ppo.v1.trainer_laminar import PPOTrainerLaminar
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode
from verl.workers.rollout.replica import TokenOutput


class RemoteMethod:
    def __init__(self, method):
        self.method = method

    async def remote(self, *args, **kwargs):
        result = self.method(*args, **kwargs)
        return await result if asyncio.iscoroutine(result) else result


@pytest.mark.parametrize("original_failure", [True, False])
def test_fit_preserves_original_error_and_still_reports_cleanup_failure(monkeypatch, original_failure):
    def fit(self, manager):
        if original_failure:
            raise ValueError("original actor failure")
        return "complete"

    def bad_shutdown():
        raise RuntimeError("cleanup failure")

    stopped = []
    trainer = object.__new__(PPOTrainerLaminar)
    trainer._prompt_feeder = None
    trainer.laminar_controller = SimpleNamespace(stop=SimpleNamespace(remote=lambda: stopped.append(True)))
    monkeypatch.setattr(trainer_base.PPOTrainer, "fit", fit)
    monkeypatch.setattr(trainer_laminar.ray, "get", lambda value: value)
    expected = ValueError if original_failure else RuntimeError
    with pytest.raises(expected, match="original actor failure" if original_failure else "cleanup failure"):
        trainer.fit(SimpleNamespace(shutdown=bad_shutdown))
    assert stopped == [True]


class Handle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.target, name))


class Relay:
    def __init__(self):
        self.latest = 0
        self.leases = set()

    def acquire(self):
        lease = str(time.monotonic_ns())
        self.leases.add(lease)
        return {"version": self.latest, "lease": lease}

    def release(self, lease):
        self.leases.remove(lease)


class Server:
    def __init__(self):
        self.version = -1
        self.calls = []
        self.requests = {}
        self.usage = 0.1

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        assert sampling_params["max_tokens"] > 0
        assert "_laminar_validate" not in sampling_params
        future = asyncio.get_running_loop().create_future()
        self.requests[request_id] = future
        self.calls.append((request_id, prompt_ids, sampling_params["max_tokens"]))
        version = self.version
        try:
            aborted = await future
            return TokenOutput(
                token_ids=[prompt_ids[-1] + 1],
                log_probs=[-0.25],
                stop_reason="aborted" if aborted else "completed",
                extra_fields={"global_steps": version},
            )
        finally:
            del self.requests[request_id]

    async def abort_trajectory(self, request_id):
        if request_id not in self.requests or self.requests[request_id].done():
            return {"aborted": False}
        self.requests[request_id].set_result(True)
        return {"aborted": True}

    def finish(self, request_id=None):
        if request_id is None:
            request_id = next(iter(self.requests))
        self.requests[request_id].set_result(False)

    def trajectory_status(self):
        return {"kv_usage": self.usage, "waiting": 0, "request_ids": list(self.requests), "observed_at": time.time()}

    async def wait_for_requests_to_drain(self):
        assert not self.requests, "Weights may never change while requests remain active"


class Worker:
    def __init__(self, server):
        self.server = server
        self.manifest = None

    def execute_checkpoint_engine(self, method, manifest=None):
        if method == "set_manifest":
            self.manifest = manifest
        elif method == "finalize":
            self.manifest = None
        else:
            raise AssertionError(method)

    def update_weights(self, global_steps):
        assert not self.server.requests
        assert self.manifest["version"] == global_steps
        self.server.version = global_steps


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def setup_controller(tmp_path):
    servers = [Server(), Server()]
    relay = Relay()
    controller = LaminarController(
        [{"server": Handle(s), "workers": [Handle(Worker(s))]} for s in servers],
        Handle(relay),
        batch_quota=1,
        max_running=4,
        check_interval_s=0.1,
        enable_repack=False,
        event_path=str(tmp_path / "scheduler.jsonl"),
    )
    await controller.start(0)
    return controller, relay, servers


def test_pull_does_not_wait_for_other_replica(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        first = asyncio.create_task(controller.generate("a", [1], {"max_tokens": 2}))
        second = asyncio.create_task(controller.generate("b", [9], {"max_tokens": 2}))
        await until(lambda: all(s.requests for s in servers))
        relay.latest = 1
        await controller.notify_weights(1)
        servers[0].finish()
        assert (await first).extra_fields["global_steps"] == 0
        await until(lambda: controller.replicas[0].version == 1)
        assert controller.replicas[1].version == 0
        assert not second.done()
        servers[1].finish()
        assert (await second).extra_fields["global_steps"] == 0
        await controller.stop()
        assert not relay.leases

    asyncio.run(scenario())
    events = [json.loads(line) for line in (tmp_path / "scheduler.jsonl").read_text().splitlines()]
    assert any(e["event"] == "pull_begin" and e["version"] == 1 and e["other_active"] == 1 for e in events)


def test_tokenizer_batch_encoding_is_normalized_before_backend_request(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        output = asyncio.create_task(
            controller.generate("encoded", BatchEncoding({"input_ids": [1], "attention_mask": [1]}), {"max_tokens": 2})
        )
        await until(lambda: bool(servers[0].requests))
        assert servers[0].calls[0][1] == [1]
        servers[0].finish()
        assert (await output).token_ids == [2]
        await controller.stop()

    asyncio.run(scenario())


def test_validation_uses_new_version_ahead_of_queued_training(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        old = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(2)]
        await until(lambda: all(server.requests for server in servers))
        relay.latest = 1
        await controller.notify_weights(1)
        await controller.set_minimum_version(1)
        queued = asyncio.create_task(controller.generate("queued-train", [10], {"max_tokens": 2}))
        validate = asyncio.create_task(controller.generate("val", [20], {"max_tokens": 2, "_laminar_validate": True}))
        servers[0].finish()
        await old[0]
        await until(lambda: len(servers[0].calls) == 2)
        assert servers[0].calls[-1][1] == [20]
        assert not queued.done() and not old[1].done()
        servers[0].finish()
        assert (await validate).extra_fields["global_steps"] == 1
        await controller.set_minimum_version(-1)
        await until(lambda: len(servers[0].calls) == 3)
        servers[0].finish()
        servers[1].finish()
        await asyncio.gather(queued, old[1])
        await controller.stop()

    asyncio.run(scenario())


def test_shutdown_cancels_active_and_waiting_requests_without_fake_outputs(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        tasks = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(3)]
        await until(lambda: len(controller.pending) == 3 and all(server.requests for server in servers))
        await controller.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(task.cancelled() for task in tasks)
        assert not controller.pending and not controller.request_tasks
        assert controller.metrics["completed"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("trigger", ["automatic", "manual"])
def test_repack_preserves_tokens_logprobs_budget_and_policy(tmp_path, trigger):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        moving = asyncio.create_task(controller.generate("a", [1], {"max_tokens": 2}))
        staying = asyncio.create_task(controller.generate("b", [9], {"max_tokens": 1}))
        await until(lambda: all(s.requests for s in servers))
        relay.latest = 1
        await controller.notify_weights(1)
        controller.replicas[0].previous_usage = 0.3
        controller.replicas[1].previous_usage = 0.5
        servers[1].usage = 0.2
        moved = await controller.repack_once() if trigger == "automatic" else await controller.migrate(0, 1)
        assert moved == 1
        await until(lambda: len(servers[1].calls) == 2)
        assert servers[1].calls[-1][1:] == ([1, 2], 1)
        await until(lambda: controller.replicas[0].version == 1)
        assert controller.replicas[1].version == 0
        for future in list(servers[1].requests.values()):
            future.set_result(False)
        output, other = await asyncio.gather(moving, staying)
        assert output.token_ids == [2, 3]
        assert output.log_probs == [-0.25, -0.25]
        assert output.extra_fields["min_global_steps"] == output.extra_fields["max_global_steps"] == 0
        assert output.extra_fields["migrations"] == 1
        assert other.token_ids == [10]
        assert not any(r.reserved for r in controller.replicas)
        assert controller.metrics["completed"] == 2
        await controller.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("violation", ["version", "capacity"])
def test_manual_migration_checks_version_and_capacity_before_reserving(tmp_path, violation):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        tasks = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(2)]
        await until(lambda: all(s.requests for s in servers))
        if violation == "version":
            controller.replicas[1].version = 1
        else:
            controller.max_running = 1
        with pytest.raises(ValueError, match="same-version|capacity"):
            await controller.migrate(0, 1)
        assert not any(replica.reserved for replica in controller.replicas)
        controller.replicas[1].version = 0
        for server in servers:
            server.finish()
        await asyncio.gather(*tasks)
        await controller.stop()

    asyncio.run(scenario())


def test_natural_completion_during_abort_releases_reservation(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        tasks = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(2)]
        await until(lambda: all(s.requests for s in servers))
        controller.replicas[0].previous_usage = controller.replicas[1].previous_usage = 0.5
        servers[1].usage = 0.2

        async def finish_instead(request_id):
            servers[0].finish(request_id)
            await asyncio.sleep(0)
            return {"aborted": False}

        servers[0].abort_trajectory = finish_instead
        await controller.repack_once()
        servers[1].finish()
        await asyncio.gather(*tasks)
        assert controller.metrics["completed"] == 2
        assert controller.metrics["migrations"] == 0
        assert not any(r.reserved for r in controller.replicas)
        await controller.stop()

    asyncio.run(scenario())


def test_repack_never_crosses_policy_versions(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        controller.replicas[1].version = servers[1].version = 1
        tasks = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(2)]
        await until(lambda: all(s.requests for s in servers))
        for replica in controller.replicas:
            replica.previous_usage = 0.5
        assert await controller.repack_once() == 0
        for server in servers:
            server.finish()
        await asyncio.gather(*tasks)
        await controller.stop()

    asyncio.run(scenario())


def test_repack_remembers_rampdown_across_staggered_completions(tmp_path):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)
        tasks = [asyncio.create_task(controller.generate(str(i), [i], {"max_tokens": 2})) for i in range(2)]
        await until(lambda: all(server.requests for server in servers))
        controller.replicas[0].previous_usage = 0.3
        controller.replicas[1].previous_usage = 0.1
        servers[0].usage, servers[1].usage = 0.1, 0.2
        assert await controller.repack_once() == 0
        servers[0].usage, servers[1].usage = 0.15, 0.15
        assert await controller.repack_once() == 1
        await until(lambda: len(servers[1].requests) == 2)
        for future in list(servers[1].requests.values()):
            future.set_result(False)
        await asyncio.gather(*tasks)
        await controller.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize("error", ["version", "logprob"])
def test_bad_behavior_metadata_fails_closed(tmp_path, error):
    async def scenario():
        controller, relay, servers = await setup_controller(tmp_path)

        async def bad_generate(**kwargs):
            return TokenOutput(
                token_ids=[2],
                log_probs=None if error == "logprob" else [-0.1],
                stop_reason="completed",
                extra_fields={"global_steps": 3 if error == "version" else 0},
            )

        servers[0].generate = bad_generate
        with pytest.raises(RuntimeError, match="Behavior policy|behavior log"):
            await controller.generate("bad", [1], {"max_tokens": 2})
        assert not controller.pending and not controller.replicas[0].active
        await controller.stop()

    asyncio.run(scenario())


def test_packing_respects_capacity_and_has_no_transfer_chains():
    snapshots = [{"id": i, "usage": load, "count": 1} for i, load in enumerate([0.1, 0.2, 0.3, 0.4])]
    plan = consolidation_plan(snapshots, max_requests=3, capacity=0.7)
    sources, targets = {s for s, _ in plan}, {d for _, d in plan}
    assert not sources & targets
    for target in targets:
        members = [target] + [s for s, d in plan if d == target]
        assert sum(snapshots[i]["usage"] for i in members) <= 0.7 + 1e-12
        assert len(members) <= 3


def test_relay_lease_retains_inflight_version_and_publication_is_atomic():
    relay = CPUWeightRelay(keep_last=1)
    relay.commit({"version": 0, "nbytes": 4, "tensors": ["v0"]})
    manifest = relay.acquire()
    relay.commit({"version": 1, "nbytes": 4, "tensors": ["v1"]})
    assert relay.status()["versions"] == [0, 1]
    with pytest.raises(ValueError):
        relay.commit({"version": 2, "nbytes": 0, "tensors": []})
    assert relay.status()["latest"] == 1
    relay.release(manifest["lease"])
    assert relay.status()["versions"] == [1]
    with pytest.raises(KeyError):
        relay.acquire(0)


def test_old_log_probs_are_exact_behavior_values_without_actor_forward(monkeypatch):
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.config = SimpleNamespace(algorithm={"rollout_correction": {"bypass_mode": True}})
    # Deliberately provide no actor worker group: calling a new-policy forward
    # here must fail the test rather than silently alter the PPO denominator.
    behavior = torch.tensor([[-1.2, -0.3]])
    captured = {}
    monkeypatch.setattr(
        trainer_base.tq,
        "kv_batch_get",
        lambda **kwargs: TensorDict({"rollout_log_probs": behavior.clone()}, batch_size=1),
    )
    monkeypatch.setattr(trainer_base.tq, "kv_batch_put", lambda **kwargs: captured.update(kwargs))
    batch = SimpleNamespace(keys=["a"], partition_id="train", tags=[{"min_global_steps": 3, "max_global_steps": 3}])
    metrics = {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert torch.equal(captured["fields"]["old_log_probs"], behavior)
    assert metrics["laminar/old_log_probs_from_rollout"] == 1.0
    batch.tags = [{"min_global_steps": 3, "max_global_steps": 4}]
    with pytest.raises(ValueError, match="one behavior-policy version"):
        trainer._compute_old_log_prob(batch, {})


def test_step_and_sample_hooks_keep_fixed_placement_and_initialize_timers(monkeypatch):
    trainer = object.__new__(PPOTrainerLaminar)
    trainer.laminar_controller = SimpleNamespace(status=SimpleNamespace(remote=lambda: {}))
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.current_mode = HybridEngineMode.TRAINER
    monkeypatch.setattr(trainer_base.ray, "get", lambda value: value)
    # No checkpoint manager is provided: any global pause/switch would fail.
    trainer.on_step_begin()
    trainer.on_sample_begin()
    trainer.on_sample_end()
    assert trainer._step_sample_wait_seconds >= 0
    assert trainer._step_wait_samples == 0
    assert trainer.current_mode == HybridEngineMode.TRAINER
    trainer.on_step_begin()
    assert trainer._step_sample_wait_seconds == 0
