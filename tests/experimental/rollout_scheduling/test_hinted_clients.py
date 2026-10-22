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
"""Unit tests for the hinted LLM server clients (design doc §3.3).

All Ray actor handles are fakes — the client only needs ``.remote()`` on the
methods it calls — so these run without a Ray cluster.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from verl.experimental.rollout_scheduling.client import (
    HintedFullyAsyncLLMServerClient,
    HintedLLMServerClient,
    upgrade_client_class,
)
from verl.experimental.rollout_scheduling.load_balancer import TokenWeightedLoadBalancer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput


# --------------------------------------------------------------------- fakes


class _FakeActorMethod:
    """Stand-in for a ray actor method (``.remote(**kw)``)."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, **kwargs):
        # An async fn returns a coroutine the caller awaits; a sync fn
        # returns a plain value (fire-and-forget semantics). Matches both
        # client call sites without Ray.
        return self._fn(**kwargs)


class FakeLBHandle:
    """Records acquire/release kwargs and delegates to a real balancer."""

    def __init__(self, lb):
        self._lb = lb
        self.acquire_calls = []
        self.release_calls = []
        self.acquire_server = _FakeActorMethod(self._acquire)
        self.release_server = _FakeActorMethod(self._release)

    async def _acquire(self, **kw):
        self.acquire_calls.append(kw)
        return self._lb.acquire_server(**kw)

    def _release(self, **kw):
        self.release_calls.append(kw)
        self._lb.release_server(**kw)


class FakeServer:
    """Scripted rollout server: a list of (stop_reason, token_ids) per call."""

    def __init__(self, script=None):
        self.script = script or [("completed", [9, 9])]
        self.calls = []
        self.generate = _FakeActorMethod(self._generate)

    async def _generate(self, **kw):
        self.calls.append(kw)
        stop_reason, token_ids = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return TokenOutput(token_ids=list(token_ids), stop_reason=stop_reason, extra_fields={"global_steps": 7})


def _config(response_length=None, partial_rollout=None):
    cfg = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                name="vllm", response_length=response_length, full_determinism=False
            )
        )
    )
    if partial_rollout is not None:
        cfg.async_training = SimpleNamespace(partial_rollout=partial_rollout)
    return cfg


def _client(client_cls, handle, **cfg_kw):
    return client_cls(config=_config(**cfg_kw), load_balancer_handle=handle)


# ---------------------------------------------------------------------- tests


class TestHintedSyncClient:
    def test_acquire_receives_hint_release_receives_request_id(self):
        async def run():
            server = FakeServer()
            lb = TokenWeightedLoadBalancer(servers={"s0": server, "s1": object()})
            handle = FakeLBHandle(lb)
            client = _client(HintedLLMServerClient, handle)
            output = await client.generate(
                "r1", prompt_ids=list(range(100)), sampling_params={"max_tokens": 50}
            )
            return lb, handle, output

        lb, handle, output = asyncio.run(run())
        assert output.token_ids == [9, 9]
        # acquire got the token hint: 100 prompt + 50 budget
        assert handle.acquire_calls == [{"request_id": "r1", "kv_hint": 150}]
        # release got the request id for exact per-request bookkeeping
        assert handle.release_calls == [{"server_id": "s0", "request_id": "r1"}]
        # and the real balancer's bookkeeping is exactly zeroed
        assert lb._inflight_requests == {"s0": 0, "s1": 0}
        assert lb._inflight_tokens == {"s0": 0, "s1": 0}

    def test_hint_estimation_fallbacks(self):
        handle = FakeLBHandle(TokenWeightedLoadBalancer(servers={"s0": object()}))
        client = _client(HintedLLMServerClient, handle)
        estimate = client._estimate_final_tokens
        # 1. explicit sampling-param budget wins
        assert estimate(list(range(10)), {"max_tokens": 5}) == 15
        assert estimate(list(range(10)), {"max_new_tokens": 5}) == 15
        # 2. rollout config response_length
        client2 = _client(HintedLLMServerClient, handle, response_length=32)
        assert client2._estimate_final_tokens(list(range(10)), {}) == 42
        # 3. prompt length alone (unknown budget), floor at 1
        assert estimate(list(range(10)), {}) == 10
        assert estimate([], {}) == 1

    def test_concurrent_generate_tasks_isolated(self):
        async def run():
            lb = TokenWeightedLoadBalancer(servers={"s0": FakeServer()})
            handle = FakeLBHandle(lb)
            client = _client(HintedLLMServerClient, handle)
            outputs = await asyncio.gather(
                client.generate("r1", prompt_ids=list(range(100)), sampling_params={"max_tokens": 50}),
                client.generate("r2", prompt_ids=list(range(7)), sampling_params={"max_tokens": 3}),
            )
            return handle, outputs

        handle, outputs = asyncio.run(run())
        assert all(o.token_ids == [9, 9] for o in outputs)
        hints = {c["request_id"]: c["kv_hint"] for c in handle.acquire_calls if "kv_hint" in c}
        assert hints == {"r1": 150, "r2": 10}  # no cross-task hint bleed


class TestHintedFullyAsyncClient:
    def test_retry_attempts_keep_hint_invariant(self):
        """P+B invariant: prompt grows / budget shrinks → hint constant per attempt."""

        async def run():
            server = FakeServer(script=[("aborted", [1, 2, 3]), ("completed", [4, 5])])
            lb = TokenWeightedLoadBalancer(servers={"s0": server})
            handle = FakeLBHandle(lb)
            client = _client(HintedFullyAsyncLLMServerClient, handle, partial_rollout=True)
            output = await client.generate(
                "r1", prompt_ids=list(range(100)), sampling_params={"max_tokens": 10}
            )
            return lb, handle, output

        with patch("asyncio.sleep", new=AsyncMock(return_value=None)):
            lb, handle, output = asyncio.run(run())
        # attempt 1: 100 prompt + 10 budget; attempt 2: 103 prompt + 7 budget
        assert [c["kv_hint"] for c in handle.acquire_calls] == [110, 110]
        assert output.token_ids == [1, 2, 3, 4, 5]
        assert output.extra_fields["global_steps"] == 7
        # bookkeeping zeroed after the successful resume
        assert lb._inflight_requests == {"s0": 0}
        assert lb._inflight_tokens == {"s0": 0}

    def test_mro_preserves_fully_async_semantics(self):
        """Retry loop and only_hybrid retry stay outside the hinted acquire."""
        assert issubclass(HintedFullyAsyncLLMServerClient, FullyAsyncLLMServerClient)
        client = _client(HintedFullyAsyncLLMServerClient, FakeLBHandle(object()), partial_rollout=True)
        assert client._only_hybrid is False


class TestClientClassUpgrader:
    def test_exact_match_mapping(self):
        assert upgrade_client_class(LLMServerClient) is HintedLLMServerClient
        assert upgrade_client_class(FullyAsyncLLMServerClient) is HintedFullyAsyncLLMServerClient

    def test_custom_subclass_passes_through(self):
        class CustomClient(LLMServerClient):
            pass

        assert upgrade_client_class(CustomClient) is None

    def test_plugin_package_registers_upgrader(self):
        # Importing the plugin package registers the upgrader with the
        # llm_server socket. Registration is global for the process and
        # behaviour-preserving (the stock balancer ignores hint kwargs), so
        # it is benign for other tests in the same session.
        import verl.experimental.rollout_scheduling  # noqa: F401
        from verl.workers.rollout.llm_server import _upgrade_client_class

        assert _upgrade_client_class(LLMServerClient) is HintedLLMServerClient
