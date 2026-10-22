# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fault injection against the actual journal and controller continuation path."""

import asyncio
import subprocess
import sys

import pytest
from test_laminar_on_cpu import Handle, Server, Worker, until

from verl.experimental.trajectory_async.cpu_relay import CPUWeightRelay
from verl.experimental.trajectory_async.laminar_controller import LaminarController
from verl.experimental.trajectory_async.partial_store import PartialTrajectoryStore, StreamJournal
from verl.workers.rollout.replica import TokenOutput


def make_store(tmp_path):
    return PartialTrajectoryStore(str(tmp_path / "partial.sqlite"))


def create(store, request="r", budget=5, version=0):
    store.create(request, [1], {"logprobs": True}, budget, version)
    return store.begin_attempt(request, "attempt-1")


def test_durable_ack_survives_restart_and_incomplete_transaction_rolls_back(tmp_path):
    store = make_store(tmp_path)
    attempt = create(store)
    store.append("r", attempt["epoch"], 0, 0, [2, 3], [-0.2, -0.3])
    store.close()
    # Crash a real second process after an uncommitted metadata/chunk update.
    script = """
import os, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute('BEGIN IMMEDIATE')
conn.execute("UPDATE trajectories SET size=3 WHERE id='r'")
conn.execute("INSERT INTO chunks VALUES ('r', 2, '[4]', '[-0.4]')")
os._exit(17)
"""
    assert subprocess.run([sys.executable, "-c", script, str(tmp_path / "partial.sqlite")]).returncode == 17
    store = make_store(tmp_path)
    row = store.get("r")
    assert row["token_ids"] == [2, 3]
    assert row["log_probs"] == [-0.2, -0.3]
    assert row["budget"] - len(row["token_ids"]) == 3
    assert store.begin_attempt("r", "attempt-1")["epoch"] == attempt["epoch"]
    store.close()


@pytest.mark.parametrize("fault", ["stale", "version", "gap", "overlap", "conflict", "budget", "nan", "missing"])
def test_rejects_invalid_or_late_updates_without_mutating_prefix(tmp_path, fault):
    store = make_store(tmp_path)
    attempt = create(store, budget=3)
    store.append("r", attempt["epoch"], 0, 0, [2, 3], [-0.2, -0.3])
    epoch, version, offset, tokens, probs = attempt["epoch"], 0, 2, [4], [-0.4]
    if fault == "stale":
        store.fence("r")
    elif fault == "version":
        version = 1
    elif fault == "gap":
        offset = 3
    elif fault == "overlap":
        offset = 1
    elif fault == "conflict":
        offset = 0
    elif fault == "budget":
        tokens, probs = [4, 5], [-0.4, -0.5]
    elif fault == "nan":
        probs = [float("nan")]
    elif fault == "missing":
        probs = []
    with pytest.raises(ValueError):
        store.append("r", epoch, version, offset, tokens, probs)
    assert store.get("r")["token_ids"] == [2, 3]
    store.close()


def test_idempotent_chunk_and_terminal_ack_identity_and_capacity(tmp_path):
    store = PartialTrajectoryStore(tmp_path / "partial.sqlite", max_active=1)
    attempt = create(store)
    with pytest.raises(RuntimeError, match="capacity"):
        create(store, "other")
    with pytest.raises(ValueError, match="identity"):
        store.create("r", [99], {}, 5, 0)
    for _ in range(2):
        assert store.append("r", attempt["epoch"], 0, 0, [2], [-0.2], terminal=True)["status"] == "completed"
    assert store.get("r")["token_ids"] == [2]
    assert store.versions() == []
    create(store, "other")
    store.close()


def test_stream_submits_deltas_and_detects_rewritten_prefix(tmp_path):
    async def scenario():
        store = make_store(tmp_path)
        attempt = create(store)
        journal = StreamJournal({**attempt, "store": Handle(store), "request_id": "r", "chunk_tokens": 2})
        await journal.update([2], [-0.2], 0)
        assert store.get("r")["token_ids"] == []
        await journal.update([2, 3], [-0.2, -0.3], 0)
        assert store.get("r")["token_ids"] == [2, 3]
        with pytest.raises(ValueError, match="rewrote"):
            await journal.update([9, 3, 4], [-0.2, -0.3, -0.4], 0, finished=True)
        await journal.update([2, 3, 4], [-0.2, -0.3, -0.4], 0, finished=True)
        assert store.get("r")["status"] == "completed"
        store.close()

    asyncio.run(scenario())


class StreamingServer(Server):
    def trajectory_status(self):
        return {**super().trajectory_status(), "version": self.version}

    async def generate(self, request_id, prompt_ids, sampling_params, trajectory_progress, **kwargs):
        version = self.version
        journal = StreamJournal(trajectory_progress)
        tokens, probs = [prompt_ids[-1] + 1], [-0.25]
        await journal.update(tokens, probs, version, force=True)
        future = asyncio.get_running_loop().create_future()
        self.requests[request_id] = future
        self.calls.append((request_id, prompt_ids, sampling_params["max_tokens"]))
        try:
            result = await future
            if result == "crash":
                raise ConnectionError("injected rollout death")
            aborted = result is True
            if not aborted and sampling_params["max_tokens"] > 1:
                tokens.append(tokens[-1] + 1)
                probs.append(-0.5)
            await journal.update(tokens, probs, version, finished=not aborted, force=True)
            if result == "lost-final":
                raise ConnectionError("final result lost after durable EOS")
            return TokenOutput(
                token_ids=tokens,
                log_probs=probs,
                stop_reason="aborted" if aborted else "completed",
                extra_fields={"global_steps": version},
            )
        finally:
            self.requests.pop(request_id, None)


async def durable_controller(tmp_path, options=None):
    store = make_store(tmp_path)
    relay = CPUWeightRelay(keep_last=1)
    relay.commit({"version": 0, "tensors": ["v0"], "nbytes": 4})
    servers = [StreamingServer(), StreamingServer()]
    controller = LaminarController(
        [{"server": Handle(s), "workers": [Handle(Worker(s))]} for s in servers],
        Handle(relay),
        batch_quota=1,
        max_running=4,
        check_interval_s=0.01,
        enable_repack=False,
        partial_store=Handle(store),
        recovery_options=options,
    )
    await controller.start(0)
    return controller, relay, servers, store


@pytest.mark.parametrize("fault", ["crash", "heartbeat", "stall"])
def test_rollout_failure_resumes_acknowledged_prefix_under_original_weights(tmp_path, fault):
    async def scenario():
        controller, relay, servers, store = await durable_controller(
            tmp_path,
            {
                "heartbeat_interval_s": 0.01,
                "heartbeat_timeout_s": 0.02,
                "heartbeat_failures": 1,
                "progress_timeout_s": 0.05 if fault == "stall" else 10,
            },
        )
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        assert store.get("r")["token_ids"] == [2]
        # Both healthy replicas advance beyond version zero; root keep_last=1
        # would evict it without unfinished-trajectory retention.
        relay.commit({"version": 1, "tensors": ["v1"], "nbytes": 4})
        await controller.notify_weights(1)
        await until(lambda: controller.replicas[1].version == 1)
        assert 0 in relay.versions
        if fault == "crash":
            next(iter(servers[0].requests.values())).set_result("crash")
        elif fault == "heartbeat":

            async def dead():
                raise ConnectionError("heartbeat lost")

            servers[0].trajectory_status = dead
        await until(lambda: bool(servers[1].requests))
        assert servers[1].version == 0
        assert servers[1].calls[-1][1:] == ([1, 2], 2)
        servers[1].finish()
        output = await task
        assert output.token_ids == [2, 3, 4]
        assert output.log_probs == [-0.25, -0.25, -0.5]
        assert output.extra_fields["min_global_steps"] == output.extra_fields["max_global_steps"] == 0
        assert controller.metrics["recoveries"] == 1
        assert not controller.replicas[0].healthy
        assert store.get("r")["status"] == "completed"
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_lost_final_rpc_is_replayed_without_regeneration_or_duplicates(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        next(iter(servers[0].requests.values())).set_result("lost-final")
        output = await task
        replay = await controller.generate("r", [1], {"max_tokens": 3, "logprobs": True})
        assert output.token_ids == replay.token_ids == [2, 3]
        assert sum(len(s.calls) for s in servers) == 1
        assert controller.metrics["completed"] == 1
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_durable_repack_keeps_tokens_probs_and_reservations(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        await controller.migrate(0, 1)
        await until(lambda: bool(servers[1].requests))
        assert servers[1].calls[-1][1:] == ([1, 2], 2)
        servers[1].finish()
        output = await task
        assert output.token_ids == [2, 3, 4]
        assert output.log_probs == [-0.25, -0.25, -0.5]
        assert output.extra_fields["migrations"] == 1
        assert not any(r.reserved for r in controller.replicas)
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_reattach_reopens_partial_journal_and_fences_old_writer(tmp_path):
    async def scenario():
        store = make_store(tmp_path)
        attempt = create(store, budget=3)
        store.append("r", attempt["epoch"], 0, 0, [2], [-0.2])
        store.close()
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        assert servers[0].calls[-1][1:] == ([1, 2], 2)
        with pytest.raises(ValueError, match="Stale"):
            store.append("r", attempt["epoch"], 0, 1, [99], [-9.9])
        servers[0].finish()
        assert (await task).token_ids == [2, 3, 4]
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_no_healthy_target_fails_closed_and_keeps_recoverable_prefix(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        controller.replicas[1].healthy = False
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        next(iter(servers[0].requests.values())).set_result("crash")
        with pytest.raises(RuntimeError, match="No healthy"):
            await task
        assert store.get("r")["token_ids"] == [2]
        assert store.get("r")["status"] == "running"
        assert not controller.pending
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_database_rejects_another_models_run(tmp_path):
    path = tmp_path / "identity.sqlite"
    first = PartialTrajectoryStore(path, identity={"model": "qwen", "run": "a"})
    first.close()
    same = PartialTrajectoryStore(path, identity={"model": "qwen", "run": "a"})
    same.close()
    with pytest.raises(ValueError, match="another model/run"):
        PartialTrajectoryStore(path, identity={"model": "other", "run": "a"})


def test_second_target_failure_during_old_weight_load_uses_next_healthy_replica(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        third = StreamingServer()
        servers.append(third)
        controller.replicas.append(type(controller.replicas[0])(server=Handle(third), workers=[Handle(Worker(third))]))
        await controller._pull(2)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        relay.commit({"version": 1, "tensors": ["v1"], "nbytes": 4})
        await controller.notify_weights(1)
        await until(lambda: all(r.version == 1 for r in controller.replicas[1:]))
        original_update = controller.replicas[1].workers[0].target.update_weights

        def fail_old_load(global_steps):
            if global_steps == 0:
                raise ConnectionError("second target load failed")
            return original_update(global_steps)

        controller.replicas[1].workers[0].target.update_weights = fail_old_load
        next(iter(servers[0].requests.values())).set_result("crash")
        await until(lambda: bool(servers[2].requests))
        servers[2].finish()
        output = await task
        assert output.token_ids == [2, 3, 4]
        assert output.extra_fields["global_steps"] == 0
        assert controller.failure is None
        assert controller.metrics["quarantined_replicas"] == 2
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_cancel_fences_late_output_and_retains_prefix_without_fake_completion(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        epoch = store.get("r")["epoch"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.get("r")["token_ids"] == [2]
        assert store.get("r")["status"] == "running"
        with pytest.raises(ValueError, match="Stale"):
            store.append("r", epoch, 0, 1, [99], [-0.2])
        assert controller.metrics["completed"] == 0
        assert not controller.pending
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_cumulative_backend_list_growth_during_ack_does_not_skip_tokens(tmp_path):
    async def scenario():
        store = make_store(tmp_path)
        attempt = create(store)
        tokens, probs = [2, 3], [-0.2, -0.3]

        class GrowingStore:
            async def append(self, *args, **kwargs):
                result = store.append(*args, **kwargs)
                tokens.append(4)  # vLLM keeps decoding while the remote ACK is pending.
                await asyncio.sleep(0)
                return result

        journal = StreamJournal({**attempt, "store": Handle(GrowingStore()), "request_id": "r", "chunk_tokens": 1})
        await journal.update(tokens, probs, 0)
        assert journal.written == 2
        probs.append(-0.4)
        journal.store = Handle(store)
        await journal.update(tokens, probs, 0, finished=True)
        assert store.get("r")["token_ids"] == [2, 3, 4]
        assert store.get("r")["log_probs"] == [-0.2, -0.3, -0.4]
        store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", ["versions", "bytes"])
def test_publication_backpressure_waits_for_unused_old_version_without_evicting_active(tmp_path, limit):
    async def scenario():
        relay = CPUWeightRelay(
            keep_last=1,
            max_versions=2 if limit == "versions" else 4,
            max_bytes=8 if limit == "bytes" else 0,
            capacity_timeout_s=1,
        )
        relay.commit({"version": 0, "tensors": ["v0"], "nbytes": 4})
        relay.retain(0, "active-v0")
        relay.commit({"version": 1, "tensors": ["v1"], "nbytes": 4})
        waiting = asyncio.create_task(relay.wait_for_capacity(2, 4))
        await asyncio.sleep(0.02)
        assert not waiting.done()
        assert relay.status()["versions"] == [0, 1]
        assert relay.status()["capacity_waiters"] == 1
        relay.release_owner("active-v0")
        assert await waiting == 4
        relay.commit({"version": 2, "tensors": ["v2"], "nbytes": 4})
        assert relay.status()["versions"] == [2]
        assert relay.status()["accounted_bytes"] <= (relay.max_bytes or 100)

    asyncio.run(scenario())


def test_impossible_budget_and_stuck_retention_fail_explicitly():
    with pytest.raises(ValueError, match="keep_last"):
        CPUWeightRelay(keep_last=2, max_versions=2)

    async def scenario():
        relay = CPUWeightRelay(keep_last=1, max_versions=2, max_bytes=8, capacity_timeout_s=0.03)
        relay.commit({"version": 0, "tensors": ["v0"], "nbytes": 4})
        relay.retain(0, "unfinished")
        relay.commit({"version": 1, "tensors": ["v1"], "nbytes": 4})
        with pytest.raises(ValueError, match="byte budget"):
            await relay.wait_for_capacity(2, 9)
        with pytest.raises(TimeoutError):
            await relay.wait_for_capacity(2, 4)
        assert relay.status()["versions"] == [0, 1]
        assert relay.status()["capacity_waiters"] == 0

    asyncio.run(scenario())


def test_explicit_abandon_releases_version_and_fences_late_writer(tmp_path):
    store = make_store(tmp_path)
    attempt = create(store)
    store.append("r", attempt["epoch"], 0, 0, [2], [-0.2])
    assert store.versions() == [0]
    assert store.incomplete()[0]["tokens"] == 1
    store.abandon("r", "operator discarded irrecoverable prompt")
    assert store.versions() == []
    assert store.incomplete() == []
    with pytest.raises(ValueError, match="Stale"):
        store.append("r", attempt["epoch"], 0, 1, [3], [-0.3])
    store.close()


def test_second_store_writer_is_rejected_until_first_closes(tmp_path):
    first = make_store(tmp_path)
    with pytest.raises(RuntimeError, match="live writer"):
        make_store(tmp_path)
    first.close()
    replacement = make_store(tmp_path)
    replacement.close()


def test_failed_destination_after_abort_resumes_pinned_source_without_losing_prefix(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(tmp_path)
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))
        original_abort = servers[0].abort_trajectory

        async def abort_and_lose_destination(request_id):
            controller._quarantine(1, ConnectionError("destination died after reservation"))
            return await original_abort(request_id)

        servers[0].abort_trajectory = abort_and_lose_destination
        await controller.migrate(0, 1)
        await until(lambda: len(servers[0].calls) == 2)
        assert servers[0].calls[-1][1:] == ([1, 2], 2)
        servers[0].finish()
        assert (await task).token_ids == [2, 3, 4]
        assert not controller.replicas[1].reserved
        assert controller.failure is None
        await controller.stop()
        store.close()

    asyncio.run(scenario())


def test_source_death_during_abort_rpc_does_not_fail_scheduler(tmp_path):
    async def scenario():
        controller, relay, servers, store = await durable_controller(
            tmp_path,
            {
                "heartbeat_interval_s": 0.01,
                "heartbeat_timeout_s": 0.02,
                "heartbeat_failures": 1,
            },
        )
        task = asyncio.create_task(controller.generate("r", [1], {"max_tokens": 3, "logprobs": True}))
        await until(lambda: bool(servers[0].requests))

        async def dead_abort(request_id):
            raise ConnectionError("source died during abort")

        servers[0].abort_trajectory = dead_abort
        await controller.migrate(0, 1)
        await until(lambda: bool(servers[1].requests))
        servers[1].finish()
        assert (await task).token_ids == [2, 3, 4]
        assert controller.failure is None
        assert not controller.replicas[1].reserved
        await controller.stop()
        store.close()

    asyncio.run(scenario())
