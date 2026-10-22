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
"""Token-weighted rollout load balancer.

Implements the L1 routing upgrade from ``rollout_layered_scheduling_design.md`` §3:

- scores replicas by estimated in-flight *token* load instead of request count
  (RL rollout KV footprint is heavy-tailed; count is a one-dimensional
  projection of it);
- keeps the sticky-session and full-determinism hash paths semantically
  unchanged from :class:`GlobalRequestLoadBalancer`;
- adds the L2 orchestration sockets: drain set, per-replica scoring bias,
  and engine-truth rebase.

Drop-in compatible: callers that do not pass ``kv_hint`` get hint ``1`` per
request, which makes token scoring *exactly* equivalent to count-based
least-loaded selection.
"""

from verl.workers.rollout.llm_server import DEFAULT_ROUTING_CACHE_SIZE, GlobalRequestLoadBalancer


class TokenWeightedLoadBalancer(GlobalRequestLoadBalancer):
    """Least-loaded-by-estimated-tokens balancer with L2 orchestration sockets.

    Selected via the plugin socket in :class:`LLMServerManager`::

        export VERL_ROLLOUT_LOAD_BALANCER_CLS=\\
  verl.experimental.rollout_scheduling.load_balancer.TokenWeightedLoadBalancer

    State beyond the base class:

    - ``_inflight_tokens``: per-replica sum of registered hints (fast local
      approximation of KV footprint; corrected by :meth:`rebase_loads`);
    - ``_request_book``: ``request_id -> (server_id, hint)`` so release
      subtracts exactly what acquire added (no bookkeeping drift);
    - ``_draining``: replicas excluded from new acquires (L2 tail
      consolidation); in-flight requests on them still run to completion;
    - ``_bias``: multiplicative scoring penalty (L2 reacts to measured
      preemption pressure faster than a full consolidation cycle);
    - ``_hint_sum`` / ``_actual_sum``: calibration telemetry (Σ hints vs
      Σ actual final lengths per replica).
    """

    def __init__(
        self,
        servers: dict,
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
    ):
        super().__init__(servers=servers, max_cache_size=max_cache_size, full_determinism=full_determinism)
        self._inflight_tokens: dict[str, int] = {sid: 0 for sid in self._inflight_requests}
        self._request_book: dict[str, tuple[str, int]] = {}
        self._draining: set[str] = set()
        self._bias: dict[str, float] = {}
        self._hint_sum: dict[str, int] = {sid: 0 for sid in self._inflight_requests}
        self._actual_sum: dict[str, int] = {sid: 0 for sid in self._inflight_requests}

    # ------------------------------------------------------------------ acquire

    def acquire_server(self, request_id: str, kv_hint: int | None = None) -> tuple:
        """Acquire a server (sticky + token-weighted least-loaded).

        Args:
            request_id: request id for the sticky session.
            kv_hint: estimated final token count (prompt + response budget)
                for this request. ``None``/non-positive degrades to ``1``,
                i.e. count-based behaviour.

        Returns:
            ``(server_id, actor_handle)`` in a single atomic call.
        """
        hint = kv_hint if isinstance(kv_hint, int) and kv_hint > 0 else 1

        # Sticky session first — same semantics as the base class, plus the
        # drain check: a draining sticky target falls through to re-selection
        # (lazy invalidation, identical to the removed-server handling).
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            if server_id in self._inflight_requests and server_id not in self._draining:
                self._register(request_id, server_id, hint)
                return server_id, self._servers[server_id]
            del self._request_id_to_server[request_id]

        candidates = [sid for sid in self._inflight_requests if sid not in self._draining]
        if not candidates:
            # Message must stay identical: FullyAsyncLLMServerClient matches on it.
            raise RuntimeError("No available servers in load balancer")

        if self._full_determinism:
            # Unchanged from base: hash over the full pool. L2 drain/bias are
            # gated off under full_determinism by design (design doc §4.3 G1).
            server_id = list(self._servers)[hash(request_id) % len(self._servers)]
        else:
            # Token-weighted score, tie-break by request count then id order
            # (deterministic within a run, matching base-class tie behaviour).
            server_id = min(candidates, key=lambda sid: (self._score(sid), self._inflight_requests[sid], sid))
        self._request_id_to_server[request_id] = server_id
        self._register(request_id, server_id, hint)
        return server_id, self._servers[server_id]

    def _register(self, request_id: str, server_id: str, hint: int) -> None:
        self._request_book[request_id] = (server_id, hint)
        self._inflight_requests[server_id] += 1
        self._inflight_tokens[server_id] += hint

    def _score(self, server_id: str) -> float:
        return self._inflight_tokens.get(server_id, 0) * (1.0 + self._bias.get(server_id, 0.0))

    # ------------------------------------------------------------------ release

    def release_server(self, server_id: str, request_id: str | None = None, actual_kv: int | None = None) -> None:
        """Release a server after a request completes.

        Args:
            server_id: server to release (overridden by the request's booked
                server when ``request_id`` is provided).
            request_id: when provided, pops the exact registered hint so the
                bookkeeping subtracts exactly what acquire added.
            actual_kv: optional actual final token count; accumulated into the
                calibration telemetry only (never into the live bookkeeping).
        """
        hint = 1
        if request_id is not None:
            entry = self._request_book.pop(request_id, None)
            if entry is not None:
                server_id, hint = entry
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1
        self._inflight_tokens[server_id] = max(0, self._inflight_tokens[server_id] - hint)
        if actual_kv is not None:
            self._hint_sum[server_id] = self._hint_sum.get(server_id, 0) + hint
            self._actual_sum[server_id] = self._actual_sum.get(server_id, 0) + int(actual_kv)

    # ---------------------------------------------------------------- membership

    def add_servers(self, servers: dict) -> None:
        super().add_servers(servers)
        for sid in servers:
            self._inflight_tokens[sid] = 0
            self._hint_sum[sid] = 0
            self._actual_sum[sid] = 0

    def remove_servers(self, server_ids: list) -> None:
        super().remove_servers(server_ids)
        removed = set(server_ids)
        for sid in removed:
            self._inflight_tokens.pop(sid, None)
            self._hint_sum.pop(sid, None)
            self._actual_sum.pop(sid, None)
            self._draining.discard(sid)
        # Purge bookkeeping of in-flight requests that lived on removed
        # servers — they can never release onto a removed server.
        stale = [rid for rid, (sid, _) in self._request_book.items() if sid in removed]
        for rid in stale:
            self._request_book.pop(rid, None)

    # -------------------------------------------------------------- L2 sockets

    def begin_drain(self, server_ids: list) -> None:
        """Exclude servers from new acquires (in-flight requests keep running).

        Used by the L2 orchestration layer for tail consolidation: the drained
        replica finishes its long-tail requests while no new work arrives,
        after which it can be handed over (sleep / lease release / hot-switch).
        """
        for sid in server_ids:
            if sid in self._inflight_requests:
                self._draining.add(sid)

    def end_drain(self, server_ids: list) -> None:
        """Restore servers to the routing candidate set (rollback path)."""
        self._draining.difference_update(server_ids)

    def is_request_migrating(self, request_id: str) -> bool:
        """Whether this request's sticky server is draining or departed (migration abort).

        Override of the stock balancer's socket (design §4.4): the sticky
        entry is invalidated lazily at the *next* acquire, so at query time
        — right after an abort, before the retry — it still points at the
        victim. Two draining indicators:

        - sticky server in ``_draining``: L2 is actively consolidating it;
        - sticky server absent from the pool: it was removed mid-flight —
          only the drain path leaves a sticky entry pointing at a departed
          server (``rebalance_requests`` clears sticky before aborting, and
          elastic removal drains first).

        No sticky entry (evicted LRU entry) → conservative False: the caller
        falls back to the partial_rollout gate.
        """
        server_id = self._request_id_to_server.get(request_id)
        if server_id is None:
            return False
        return server_id in self._draining or server_id not in self._servers

    def set_bias(self, server_id: str, bias: float) -> None:
        """Set a multiplicative scoring bias (e.g. ``0.3`` = 30% penalty).

        Lets L2 steer routing away from a replica showing measured preemption
        pressure without waiting for a full consolidation cycle. Applied only
        to the live scoring; sticky sessions are unaffected.
        """
        self._bias[server_id] = float(bias)

    def rebase_loads(self, loads: dict) -> None:
        """Overwrite token bookkeeping with engine-measured ground truth.

        L2 polls engine stats (actual KV usage) and rebases the fast local
        approximation here. This also heals any hint drift from leaked
        requests (client crashed between acquire and release).
        """
        for sid, tokens in loads.items():
            if sid in self._inflight_tokens:
                self._inflight_tokens[sid] = max(0, int(tokens))

    # ------------------------------------------------------------------ status

    def get_status(self) -> dict:
        """Base status plus token loads / drain set / calibration telemetry."""
        status = super().get_status()
        status.update(
            token_loads=dict(self._inflight_tokens),
            total_inflight_tokens=sum(self._inflight_tokens.values()),
            draining=sorted(self._draining),
            calibration={
                sid: {"hint_sum": self._hint_sum.get(sid, 0), "actual_sum": self._actual_sum.get(sid, 0)}
                for sid in self._inflight_requests
            },
        )
        return status
