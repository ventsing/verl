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
"""Mock async inference engine for CPU-only end-to-end runs.

Simulates the properties of a real async rollout server that matter for
trajectory-level scheduling:

* **continuous batching**: ``max_concurrency`` slots; a request occupies one
  slot for its full prefill+decode duration, so short requests free capacity
  early — this is what creates the long-tail dynamics trajectory-level
  delivery exploits;
* **long-tail response lengths**: lognormal length distribution, seeded
  per-request, so the *same* workload (identical lengths, identical failures)
  can be replayed across modes for honest A/B comparison;
* **deterministic failures**: the per-request RNG also decides failure, so
  retries behave identically across modes;
* **weight versioning**: :meth:`set_version` simulates a parameter sync from
  the trainer; each request records the version it *started* under (the
  policy that generated its first token — the convention used for staleness
  accounting).
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass


class MockRolloutError(Exception):
    """A single generation attempt failed (transient; retryable)."""


@dataclass
class MockEngineConfig:
    # throughput of one concurrency slot
    tokens_per_s: float = 2500.0
    # fixed per-request overhead (queueing + prefill setup)
    base_latency_s: float = 0.02
    # continuous-batching capacity (number of concurrent requests)
    max_concurrency: int = 64
    # per-attempt failure probability (drawn from the request's own RNG)
    failure_rate: float = 0.0
    # lognormal response length distribution
    length_mean_tokens: float = 1500.0
    length_sigma: float = 1.0
    min_tokens: int = 16
    # response-length cap, the analogue of rollout.max_response_length —
    # without it a single lognormal draw can dominate the whole makespan
    max_tokens: int | None = 8192
    # global seed (only used to derive per-request seeds; per-request draws
    # are fully determined by the request seed)
    seed: int = 7


@dataclass
class MockGenResult:
    num_tokens: int
    latency_s: float
    model_version: int


@dataclass
class MockEngineStats:
    requests_started: int = 0
    requests_finished: int = 0
    requests_failed: int = 0
    busy_slots_peak: int = 0
    busy_time_s: float = 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "engine/requests_started": self.requests_started,
            "engine/requests_finished": self.requests_finished,
            "engine/requests_failed": self.requests_failed,
            "engine/busy_slots_peak": self.busy_slots_peak,
            "engine/busy_time_s": round(self.busy_time_s, 4),
            "engine/utilization": round(self.busy_time_s, 4),
        }


def request_seed(uid: str, traj_index: int, attempt: int, base_seed: int = 0) -> int:
    """Deterministic per-attempt seed.

    The same ``(uid, traj_index, attempt)`` always yields the same response
    length and the same failure outcome, regardless of mode or interleaving —
    the property that makes the A/B comparison data-equivalent.
    """
    return random.Random(f"{base_seed}:{uid}:{traj_index}:{attempt}").getrandbits(63)


class MockRolloutEngine:
    """A slot-bounded fake inference server."""

    def __init__(self, config: MockEngineConfig | None = None) -> None:
        self.config = config or MockEngineConfig()
        self._slots = asyncio.Semaphore(self.config.max_concurrency)
        self._version = 0
        self.stats = MockEngineStats()
        self._version_history: list[tuple[float, int]] = []

    # ---------------------------------------------------------------- version

    def set_version(self, version: int) -> None:
        """Simulate a parameter sync (weights broadcast) from the trainer."""
        self._version = version
        self._version_history.append((asyncio.get_running_loop().time(), version))

    @property
    def version(self) -> int:
        return self._version

    # ------------------------------------------------------------------ gen

    async def generate(self, seed: int, prompt_tokens: int = 256) -> MockGenResult:
        """Generate one trajectory. Raises :class:`MockRolloutError` on a
        (deterministic, retryable) failure."""
        rng = random.Random(seed)
        num_tokens = max(
            self.config.min_tokens,
            int(round(math.exp(rng.gauss(math.log(self.config.length_mean_tokens), self.config.length_sigma)))),
        )
        if self.config.max_tokens is not None:
            num_tokens = min(num_tokens, self.config.max_tokens)
        if rng.random() < self.config.failure_rate:
            # consume a slot briefly even when failing, then raise
            async with self._slots:
                self.stats.requests_started += 1
                self.stats.requests_failed += 1
                await asyncio.sleep(self.config.base_latency_s)
            raise MockRolloutError(f"request seed={seed} failed")

        latency = self.config.base_latency_s + num_tokens / self.config.tokens_per_s + prompt_tokens / (
            self.config.tokens_per_s * 8
        )

        loop = asyncio.get_running_loop()
        start_version = self._version
        async with self._slots:
            self.stats.requests_started += 1
            self.stats.busy_slots_peak = max(self.stats.busy_slots_peak, self._count_slots())
            t0 = loop.time()
            await asyncio.sleep(latency)
            self.stats.busy_time_s += loop.time() - t0
            self.stats.requests_finished += 1
        return MockGenResult(
            num_tokens=num_tokens,
            latency_s=latency,
            model_version=start_version,
        )

    # ---------------------------------------------------------------- private

    def _count_slots(self) -> int:
        # asyncio.Semaphore internal: value = free slots
        return self.config.max_concurrency - self._slots._value
