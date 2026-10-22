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
"""Hinted LLM server clients — attach token-weight hints to acquire/release.

Implements design doc §3.3: the client knows the prompt length and the
response budget before acquiring a server, so it passes a ``kv_hint``
(estimated final token count) alongside the acquire, and the request id at
release so a token-weighted balancer can subtract exactly what it added.

Hint math is retry-invariant for :class:`FullyAsyncLLMServerClient`: on an
abort-resume retry the prompt grows by the tokens generated so far while the
budget shrinks by the same amount, so ``len(prompt) + budget`` stays constant
across attempts — a hint computed once per attempt is exact for every attempt.

Implementation note (MRO): the mixin subclasses :class:`LLMServerClient` so
that ``class HintedFullyAsyncLLMServerClient(FullyAsyncLLMServerClient,
_HintedAcquireMixin)`` linearizes to ``[HintedFully, FullyAsync, Mixin,
LLMServerClient]``. The FullyAsync retry wrapper therefore stays *outside*
the hinted acquire (its ``only_hybrid`` retry loop is preserved), and each
retry attempt's ``super().generate`` runs the mixin's per-attempt hint
bookkeeping with that attempt's current prompt and budget.
"""

from contextvars import ContextVar
from typing import Optional

from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient


class _HintRecord:
    """Per-generate-call hint context (one per asyncio task)."""

    __slots__ = ("request_id", "kv_hint")

    def __init__(self, request_id: str, kv_hint: int):
        self.request_id = request_id
        self.kv_hint = kv_hint


_client_hint_ctx: ContextVar[Optional[_HintRecord]] = ContextVar("verl_rollout_client_kv_hint", default=None)


class _HintedAcquireMixin(LLMServerClient):
    """Mixes token hints into acquire/release around the stock generate flow."""

    # Traced like the base/async generate: for the sync client this restores
    # the single span; for the async client each resume attempt gets its own
    # node showing that attempt's prompt/budget (the P+B invariant is then
    # visible in the trace). No-op unless tracing is enabled.
    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        **kwargs,
    ):
        record = _HintRecord(request_id=request_id, kv_hint=self._estimate_final_tokens(prompt_ids, sampling_params))
        token = _client_hint_ctx.set(record)
        try:
            return await super().generate(request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, **kwargs)
        finally:
            _client_hint_ctx.reset(token)

    async def _acquire_server(self, request_id: str) -> tuple:
        record = _client_hint_ctx.get()
        kv_hint = record.kv_hint if record is not None and record.request_id == request_id else None
        # Atomic acquire with hint: token-weighted balancers score on it; the
        # stock balancer accepts-and-ignores the kwarg (plugin socket).
        server_id, handle = await self._load_balancer.acquire_server.remote(request_id=request_id, kv_hint=kv_hint)
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget, same as base. Passing request_id lets a
        # token-weighted balancer pop the exact registered hint.
        record = _client_hint_ctx.get()
        if record is not None:
            self._load_balancer.release_server.remote(server_id=server_id, request_id=record.request_id)
        else:
            self._load_balancer.release_server.remote(server_id=server_id)

    def _estimate_final_tokens(self, prompt_ids, sampling_params: dict) -> int:
        """Estimate the request's final token footprint (prompt + budget).

        Order of preference: explicit ``max_tokens`` / ``max_new_tokens`` in
        the sampling params, then the rollout config's ``response_length``,
        then prompt length alone (response budget unknown). Always >= 1 so
        unhinted callers degrade to count-based routing.
        """
        try:
            prompt_len = len(prompt_ids)
        except TypeError:
            prompt_len = 0
        budget = None
        for key in ("max_tokens", "max_new_tokens"):
            value = sampling_params.get(key)
            if isinstance(value, int) and value > 0:
                budget = value
                break
        if budget is None:
            rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
            response_length = getattr(rollout_config, "response_length", None)
            if isinstance(response_length, int) and response_length > 0:
                budget = response_length
        if budget is None:
            return max(prompt_len, 1)
        return prompt_len + budget


class HintedLLMServerClient(_HintedAcquireMixin):
    """:class:`LLMServerClient` + token hints (sync / server-based rollout)."""


class HintedFullyAsyncLLMServerClient(FullyAsyncLLMServerClient, _HintedAcquireMixin):
    """:class:`FullyAsyncLLMServerClient` + token hints.

    The abort-resume retry loop is fully preserved (see module docstring for
    the MRO argument); per-attempt hints are exact by the P+B invariant.
    """


_CLIENT_UPGRADES = {
    LLMServerClient: HintedLLMServerClient,
    FullyAsyncLLMServerClient: HintedFullyAsyncLLMServerClient,
}


def upgrade_client_class(client_cls: type) -> type | None:
    """Client-class upgrader for :meth:`LLMServerManager.get_client`.

    Exact-match mapping only: custom client subclasses pass through unchanged
    (``None``), safely degrading to count-based routing for them.
    """
    return _CLIENT_UPGRADES.get(client_cls)
