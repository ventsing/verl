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
"""Unit tests for the token-weighted rollout load balancer (design doc §3).

Pure-logic tests: the balancer is a plain Python class (wrapped with
``ray.remote`` only at instantiation by the server manager), so fake actor
handles suffice — no Ray cluster needed.
"""

import pytest

from verl.experimental.rollout_scheduling.load_balancer import TokenWeightedLoadBalancer


def _make_lb(server_ids, full_determinism=False):
    return TokenWeightedLoadBalancer(
        servers={sid: object() for sid in server_ids}, full_determinism=full_determinism
    )


class TestTokenScoring:
    def test_no_hint_degrades_to_count(self):
        """Without hints every request weighs 1 token → exact count behaviour."""
        lb = _make_lb(["s0", "s1"])
        sid1, _ = lb.acquire_server("r1")
        sid2, _ = lb.acquire_server("r2")
        assert sid1 == "s0"  # tie → id order, same as base class
        assert sid2 == "s1"
        assert lb._inflight_requests == {"s0": 1, "s1": 1}
        assert lb._inflight_tokens == {"s0": 1, "s1": 1}

    def test_token_scoring_routes_around_heavy_request(self):
        """A long request must shift the next one even with equal counts."""
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1", kv_hint=1000)  # lands on s0 (tie → id order)
        sid, _ = lb.acquire_server("r2", kv_hint=10)  # s0 has 1000 tokens → s1
        assert sid == "s1"
        assert lb._inflight_tokens == {"s0": 1000, "s1": 10}

    def test_tie_break_by_count_then_id(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1", kv_hint=10)  # s0
        lb.acquire_server("r2", kv_hint=10)  # s1 (s0 count 1)
        # both have 10 tokens, s1 has 1 request vs s0... equal counts 1/1, equal
        # tokens → id order → s0
        sid, _ = lb.acquire_server("r3", kv_hint=10)
        assert sid == "s0"

    def test_bias_penalizes_loaded_replica(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("a", kv_hint=4)  # s0
        lb.acquire_server("b", kv_hint=5)  # s1
        sid, _ = lb.acquire_server("c")  # s0 lighter (4 < 5)
        assert sid == "s0"
        lb.release_server("s0", request_id="c")
        lb.set_bias("s0", 0.5)  # s0 effective score 4 * 1.5 = 6 > 5
        sid, _ = lb.acquire_server("d")
        assert sid == "s1"


class TestStickySession:
    def test_sticky_returns_to_same_server(self):
        lb = _make_lb(["s0", "s1"])
        sid1, _ = lb.acquire_server("r1", kv_hint=100)  # s0 (tie → id order)
        lb.release_server("s0", request_id="r1")
        lb.acquire_server("r2", kv_hint=5)  # both 0 again → s0
        # r1 re-acquired: sticky hit on s0 even though a lighter-looking r2
        # request also lives there — session wins over scoring.
        sid_again, _ = lb.acquire_server("r1", kv_hint=100)
        assert sid_again == sid1

    def test_sticky_skips_draining_server(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1")  # s0
        lb.begin_drain(["s0"])
        sid, _ = lb.acquire_server("r1")  # sticky target draining → re-select
        assert sid == "s1"


class TestDrain:
    def test_drain_excludes_from_new_acquires(self):
        lb = _make_lb(["s0", "s1"])
        lb.begin_drain(["s0"])
        for i in range(5):
            sid, _ = lb.acquire_server(f"r{i}")
            assert sid == "s1"

    def test_drain_all_raises_base_message(self):
        """The error message is load-bearing: FullyAsync clients match on it."""
        lb = _make_lb(["s0"])
        lb.begin_drain(["s0"])
        with pytest.raises(RuntimeError, match="No available servers in load balancer"):
            lb.acquire_server("r1")

    def test_end_drain_restores_routing(self):
        lb = _make_lb(["s0", "s1"])
        lb.begin_drain(["s0", "s1"])
        lb.end_drain(["s1"])
        sid, _ = lb.acquire_server("r1")
        assert sid == "s1"


class TestRelease:
    def test_release_with_request_id_exact_bookkeeping(self):
        lb = _make_lb(["s0"])
        lb.acquire_server("r1", kv_hint=500)
        assert lb._inflight_tokens["s0"] == 500
        lb.release_server("s0", request_id="r1")
        assert lb._inflight_requests["s0"] == 0
        assert lb._inflight_tokens["s0"] == 0
        assert "r1" not in lb._request_book

    def test_release_overridden_by_book_entry(self):
        """release_server's sid arg is advisory; the book entry is authoritative."""
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1", kv_hint=7)  # s0
        lb.release_server("s1", request_id="r1")  # wrong sid on purpose
        assert lb._inflight_tokens["s0"] == 0  # subtracted from the booked server
        assert lb._inflight_requests["s1"] == 0  # untouched

    def test_legacy_release_degrades_to_one_token(self):
        """Legacy callers (no request_id) subtract 1 — documented degradation."""
        lb = _make_lb(["s0"])
        lb.acquire_server("r1", kv_hint=500)
        lb.release_server("s0")
        assert lb._inflight_requests["s0"] == 0
        assert lb._inflight_tokens["s0"] == 499  # drift, healed by rebase

    def test_actual_kv_accumulates_calibration(self):
        lb = _make_lb(["s0"])
        lb.acquire_server("r1", kv_hint=500)
        lb.release_server("s0", request_id="r1", actual_kv=420)
        assert lb._hint_sum["s0"] == 500
        assert lb._actual_sum["s0"] == 420


class TestRebaseAndMembership:
    def test_rebase_overwrites_token_loads(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1", kv_hint=100)
        lb.rebase_loads({"s0": 3, "s1": 7, "gone": 99})  # unknown ids ignored
        assert lb._inflight_tokens == {"s0": 3, "s1": 7}

    def test_add_servers_resets_token_state(self):
        lb = _make_lb(["s0"])
        lb.add_servers({"s1": object()})
        assert lb._inflight_tokens["s1"] == 0

    def test_remove_servers_purges_book_entries(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1")  # s0
        lb.remove_servers(["s0"])
        assert "s0" not in lb._inflight_tokens
        assert "r1" not in lb._request_book  # cannot release onto a removed server
        sid, _ = lb.acquire_server("r2")
        assert sid == "s1"

    def test_remove_clears_drain_flag(self):
        lb = _make_lb(["s0", "s1"])
        lb.begin_drain(["s0"])
        lb.remove_servers(["s0"])
        lb.add_servers({"s0": object()})
        assert "s0" not in lb._draining


class TestDeterminism:
    def test_full_determinism_hash_routing(self):
        lb = _make_lb(["s0", "s1"], full_determinism=True)
        sid1, _ = lb.acquire_server("det-1", kv_hint=1000)
        lb.release_server(sid1, request_id="det-1")
        sid2, _ = lb.acquire_server("det-1", kv_hint=1000)
        assert sid1 == sid2  # sticky+hash: same request → same replica
        # bookkeeping still exact under determinism
        assert lb._inflight_requests[sid1] == 1
        assert lb._inflight_tokens[sid1] == 1000


class TestStatus:
    def test_get_status_compat_and_extensions(self):
        lb = _make_lb(["s0", "s1"])
        lb.acquire_server("r1", kv_hint=42)
        lb.begin_drain(["s1"])
        status = lb.get_status()
        # base (compat): flat sid -> request-count map
        assert status["servers"] == {"s0": 1, "s1": 0}
        # extensions
        assert status["token_loads"] == {"s0": 42, "s1": 0}
        assert status["total_inflight_tokens"] == 42
        assert status["draining"] == ["s1"]
        assert set(status["calibration"].keys()) == {"s0", "s1"}
