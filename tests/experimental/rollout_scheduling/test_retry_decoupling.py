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
"""Unit tests for the retry-permission decoupling (design doc §4.4).

The abort-retry gate in ``FullyAsyncLLMServerClient.generate`` used to be
hard-wired to ``partial_rollout``. Decoupled semantics (Laminar-style
same-version migration):

- migration-caused abort (sticky server is draining/departed in the LB)
  → retry WITHOUT partial_rollout (keeper is same-weight-version);
- weight-update abort (server active, not draining) → partial_rollout gate
  unchanged (retry crosses versions);
- unknown (custom LB without the socket / no sticky entry) → conservative
  partial_rollout gate.

LB level covers the socket on the stock balancer (always False) and the
drain-aware override on ``TokenWeightedLoadBalancer``. Client level covers
the four retry-decision outcomes with fake LB/server handles.
"""

import asyncio
from types import SimpleNamespace

import pytest

from verl.experimental.rollout_scheduling.client import HintedFullyAsyncLLMServerClient
from verl.experimental.rollout_scheduling.load_balancer import TokenWeightedLoadBalancer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, GlobalRequestLoadBalancer
from verl.workers.rollout.replica import TokenOutput


# --------------------------------------------------------------------- fakes


class _FakeActorMethod:
    """Stand-in for a ray actor method (``.remote(**kw)``)."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, **kwargs):
        return self._fn(**kwargs)


class FakeServer:
    """Scripted rollout server: a list of (stop_reason, token_ids) per call.

    Honors ``max_tokens`` like a real engine (resume attempts carry the
    remaining budget, so the budget invariant lives server-side).
    """

    def __init__(self, script=None):
        self.script = script or [("completed", [9, 9])]
        self.calls = []
        self.generate = _FakeActorMethod(self._generate)

    async def _generate(self, **kw):
        # Snapshot: the client mutates the same sampling_params dict across
        # resume attempts (remaining-budget rewrite), so record a copy.
        self.calls.append({**kw, "sampling_params": dict(kw.get("sampling_params", {}))})
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        stop_reason, token_ids = self.script[idx]
        token_ids = list(token_ids)
        budget = kw.get("sampling_params", {}).get("max_tokens")
        if budget is not None and len(token_ids) > budget:
            token_ids = token_ids[:budget]
            stop_reason = "length"
        return TokenOutput(token_ids=token_ids, stop_reason=stop_reason, extra_fields={"global_steps": 7})


class MigrationFakeLB:
    """LB handle fake: one scripted server, configurable migration answer."""

    def __init__(self, server, migrating: bool = False):
        self._server = server
        self.migrating = migrating
        self.acquire_calls = []
        self.release_calls = []
        self.migrate_query_calls = []
        self.acquire_server = _FakeActorMethod(self._acquire)
        self.release_server = _FakeActorMethod(self._release)
        self.is_request_migrating = _FakeActorMethod(self._is_migrating)

    async def _acquire(self, **kw):
        self.acquire_calls.append(kw)
        return "s0", self._server

    def _release(self, **kw):
        self.release_calls.append(kw)

    async def _is_migrating(self, **kw):
        self.migrate_query_calls.append(kw)
        return self.migrating


def _config(partial_rollout=None):
    cfg = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(name="vllm", response_length=None, full_determinism=False)
        )
    )
    if partial_rollout is not None:
        cfg.async_training = SimpleNamespace(partial_rollout=partial_rollout)
    return cfg


# ------------------------------------------------------------------ LB level


def test_stock_lb_never_reports_migration():
    """Stock balancer has no drain concept: the socket always answers False."""
    lb = GlobalRequestLoadBalancer(servers={"s0": object()})
    lb.acquire_server(request_id="r1")  # sticky entry now points at s0
    assert lb.is_request_migrating("r1") is False
    assert lb.is_request_migrating("unknown") is False


def test_draining_sticky_is_migration():
    lb = TokenWeightedLoadBalancer(servers={"victim": object(), "keeper": object()})
    lb.acquire_server(request_id="r1")
    victim = lb._request_id_to_server["r1"]
    lb.begin_drain([victim])
    assert lb.is_request_migrating("r1") is True


def test_active_sticky_is_not_migration():
    lb = TokenWeightedLoadBalancer(servers={"s0": object(), "s1": object()})
    lb.acquire_server(request_id="r1")
    assert lb.is_request_migrating("r1") is False


def test_departed_sticky_is_migration():
    """Victim removed during the client's retry-sleep window (drain finished
    and remove_servers already ran): the stale sticky entry still identifies
    a migration abort."""
    lb = TokenWeightedLoadBalancer(servers={"victim": object(), "keeper": object()})
    lb.acquire_server(request_id="r1")
    victim = lb._request_id_to_server["r1"]
    lb.begin_drain([victim])
    lb.remove_servers([victim])  # also discards from _draining
    assert lb.is_request_migrating("r1") is True


def test_no_sticky_entry_is_not_migration():
    lb = TokenWeightedLoadBalancer(servers={"s0": object()})
    assert lb.is_request_migrating("never-seen") is False


def test_end_drain_rollback_restores_not_migrating():
    lb = TokenWeightedLoadBalancer(servers={"victim": object(), "keeper": object()})
    lb.acquire_server(request_id="r1")
    victim = lb._request_id_to_server["r1"]
    lb.begin_drain([victim])
    assert lb.is_request_migrating("r1") is True
    lb.end_drain([victim])
    assert lb.is_request_migrating("r1") is False


def test_full_lifecycle_query_then_reroute_lands_on_keeper():
    """End-to-end LB semantics: query during the drain window is True; the
    retry acquire lazily invalidates the sticky entry and reroutes to the
    keeper; the query after the retry is False again."""
    lb = TokenWeightedLoadBalancer(servers={"victim": object(), "keeper": object()})
    lb.acquire_server(request_id="r1", kv_hint=10)
    victim = lb._request_id_to_server["r1"]
    assert victim in ("victim", "keeper")
    other = "keeper" if victim == "victim" else "victim"
    lb.begin_drain([victim])

    assert lb.is_request_migrating("r1") is True  # abort was migration-caused

    # The retry: sticky invalidated (drain), rerouted to the non-drained server
    server_id, _ = lb.acquire_server(request_id="r1", kv_hint=10)
    assert server_id == other
    assert lb.is_request_migrating("r1") is False  # now stable on the keeper


# -------------------------------------------------------------- client level


@pytest.mark.asyncio
async def test_migration_abort_retries_without_partial_rollout():
    """The core decoupling: partial_rollout=False drops weight-update aborts,
    but a migration abort retries onto the same-version keeper."""
    server = FakeServer(script=[("aborted", [1, 2]), ("completed", [3])])
    lb = MigrationFakeLB(server, migrating=True)
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=lb
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.token_ids == [1, 2, 3]  # aborted attempt merged + retried continuation
    assert out.stop_reason == "completed"
    assert len(server.calls) == 2  # first attempt aborted, second completed
    assert len(lb.acquire_calls) == 2
    assert lb.migrate_query_calls  # the LB was consulted for the drain state


@pytest.mark.asyncio
async def test_completed_generation_never_queries_lb():
    """Hot-path guard: the migration query costs one LB RPC — it must only
    fire on the abort path, never for a normally completed generation."""
    server = FakeServer(script=[("completed", [1, 2])])
    lb = MigrationFakeLB(server, migrating=True)
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=lb
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.stop_reason == "completed"
    assert lb.migrate_query_calls == []  # no query on the happy path
    assert len(lb.acquire_calls) == 1


@pytest.mark.asyncio
async def test_weight_update_abort_drops_without_partial_rollout():
    """Server active and not draining → abort looks like a weight-update
    abort → sample dropped (existing semantics preserved)."""
    server = FakeServer(script=[("aborted", [1, 2])])
    lb = MigrationFakeLB(server, migrating=False)
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=lb
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.token_ids == [1, 2]
    assert out.stop_reason == "aborted"  # dropped, not retried
    assert len(server.calls) == 1
    assert len(lb.acquire_calls) == 1


@pytest.mark.asyncio
async def test_partial_rollout_true_retries_regardless_of_lb_answer():
    """partial_rollout=True keeps retrying even when the LB says the abort
    was not migration-caused (cross-version resume accepted by the trainer)."""
    server = FakeServer(script=[("aborted", [1]), ("completed", [2])])
    lb = MigrationFakeLB(server, migrating=False)
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=True), load_balancer_handle=lb
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.token_ids == [1, 2]
    assert len(server.calls) == 2


@pytest.mark.asyncio
async def test_v1_trainer_without_async_training_retries():
    """v1 trainers have no async_training config: retry is unconditional
    (pre-existing behaviour, unchanged)."""
    server = FakeServer(script=[("aborted", [1]), ("completed", [2])])
    lb = MigrationFakeLB(server, migrating=False)
    client = FullyAsyncLLMServerClient(config=_config(), load_balancer_handle=lb)

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.token_ids == [1, 2]
    assert len(server.calls) == 2


@pytest.mark.asyncio
async def test_custom_lb_without_socket_is_conservative():
    """A custom LB predating the is_request_migrating socket: AttributeError
    is swallowed and the abort is treated as weight-update-caused (drop)."""

    class LegacyLB:
        def __init__(self, server):
            self.acquire_server = _FakeActorMethod(self._acquire)
            self.release_server = _FakeActorMethod(lambda **kw: None)
            self._server = server

        async def _acquire(self, **kw):
            return "s0", self._server

    server = FakeServer(script=[("aborted", [1])])
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=LegacyLB(server)
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.stop_reason == "aborted"  # conservative: dropped
    assert len(server.calls) == 1


@pytest.mark.asyncio
async def test_hinted_client_inherits_decoupled_retry():
    """The P1 hinted FullyAsync client inherits the decoupled gate."""
    server = FakeServer(script=[("aborted", [1, 2]), ("completed", [3])])
    lb = MigrationFakeLB(server, migrating=True)
    client = HintedFullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=lb
    )

    out = await client.generate(request_id="r1", prompt_ids=[0], sampling_params={})

    assert out.token_ids == [1, 2, 3]
    assert out.stop_reason == "completed"
    assert len(server.calls) == 2


@pytest.mark.asyncio
async def test_migration_retry_respects_budget_invariant():
    """Same-version migration still obeys the token budget invariant: total
    tokens across attempts never exceed the original max_tokens."""
    server = FakeServer(script=[("aborted", [1] * 6), ("completed", [2] * 9)])
    lb = MigrationFakeLB(server, migrating=True)
    client = FullyAsyncLLMServerClient(
        config=_config(partial_rollout=False), load_balancer_handle=lb
    )

    out = await client.generate(
        request_id="r1", prompt_ids=[0], sampling_params={"max_tokens": 10}
    )

    assert len(out.token_ids) == 10  # first attempt 6 + retried continuation capped at 4
    assert out.stop_reason == "length"
    # the retry carried the remaining budget: max_tokens shrunk to 10 - 6
    assert server.calls[1]["sampling_params"]["max_tokens"] == 4
