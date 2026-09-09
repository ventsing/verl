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
"""Hierarchical weight relay service — the mock of Laminar §4.

Laminar replaces global weight synchronization (trainer broadcasts to all
rollouts, everyone stalls) with a tier of relay workers:

1. after each update the trainer pushes weights to ONE master relay and
   immediately resumes training — the only actor stall is this one hop
   (measured 0.64s@32B / 1.40s@72B in the paper);
2. the master relay broadcasts to every rollout machine's colocated relay
   via chain-based pipelined RDMA — background, near-constant time in the
   number of relays (< 1.6s for 72B to 127 relays);
3. each rollout pulls the latest weights from its colocated relay through
   PCIe whenever it decides to (typically at batch completion or when
   released by a repack) — different rollouts may then run different
   versions concurrently, which is the point: no lockstep.

This module models that timing with asyncio sleeps:

* :meth:`WeightRelayService.publish` blocks for ``actor_to_master_s``
  (the actor stall) and schedules the chain broadcast in the background;
* relay ``i`` receives a version at ``publish_time + actor_to_master_s +
  hop_latency_s * (i + 1)`` (chain order);
* :meth:`WeightRelayService.pull` waits until the replica's relay has the
  newest version that was published when the pull started, then sleeps
  ``pcie_pull_s`` — so a pull right after a publish waits for chain
  propagation, while a pull after the chain passed is PCIe-only (the
  best case in the paper's Figure 14).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class RelayConfig:
    num_relays: int = 8
    # actor → master relay transfer: the ONLY synchronous cost the trainer pays
    actor_to_master_s: float = 0.5
    # per-hop chain propagation latency (master → relay 0 → relay 1 → ...)
    hop_latency_s: float = 0.1
    # colocated relay → rollout GPUs (PCIe) load time
    pcie_pull_s: float = 0.2


@dataclass
class _Publication:
    version: int
    publish_start: float  # loop-time when publish() was called
    master_ready: float  # actor stall done, master has it
    chain_ready: list[float]  # per-relay arrival times


@dataclass
class RelayStats:
    publishes: int = 0
    actor_stall_total_s: float = 0.0
    pulls: int = 0
    pull_wait_total_s: float = 0.0  # chain propagation wait inside pulls
    pcie_total_s: float = 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            "relay/publishes": self.publishes,
            "relay/actor_stall_total_s": round(self.actor_stall_total_s, 4),
            "relay/pulls": self.pulls,
            "relay/pull_wait_total_s": round(self.pull_wait_total_s, 4),
            "relay/pcie_total_s": round(self.pcie_total_s, 4),
        }


class WeightRelayService:
    """Mock hierarchical parameter service (Laminar §4)."""

    def __init__(self, config: RelayConfig | None = None) -> None:
        self.config = config or RelayConfig()
        self._publications: list[_Publication] = []
        self.stats = RelayStats()

    # ---------------------------------------------------------------- clock

    @staticmethod
    def _now() -> float:
        return asyncio.get_running_loop().time()

    # -------------------------------------------------------------- publish

    async def publish(self, version: int) -> float:
        """Trainer-side weight publication.

        Blocks only for the actor→master transfer, then returns: the chain
        broadcast to the other relays happens in the background. Returns
        the loop-time the master relay became ready.
        """
        start = self._now()
        await asyncio.sleep(self.config.actor_to_master_s)
        self.stats.publishes += 1
        self.stats.actor_stall_total_s += self.config.actor_to_master_s
        master_ready = self._now()
        publication = _Publication(
            version=version,
            publish_start=start,
            master_ready=master_ready,
            chain_ready=[
                master_ready + self.config.hop_latency_s * (i + 1)
                for i in range(self.config.num_relays)
            ],
        )
        self._publications.append(publication)
        return master_ready

    # ------------------------------------------------------------- versions

    def latest_published_version(self) -> int:
        """Newest version published so far (may not have reached all relays)."""
        return self._publications[-1].version if self._publications else 0

    def _latest_publication_for_relay(self, replica_id: int) -> _Publication | None:
        """Newest publication that has already ARRIVED at this replica's relay."""
        for pub in reversed(self._publications):
            if self._now() >= pub.chain_ready[replica_id]:
                return pub
        return None

    def relay_version(self, replica_id: int) -> int:
        """Version currently available at this replica's colocated relay."""
        pub = self._latest_publication_for_relay(replica_id)
        return pub.version if pub else 0

    # ----------------------------------------------------------------- pull

    async def pull(self, replica_id: int) -> int:
        """Fetch the latest weights into a rollout replica.

        Targets the newest version published when the pull started (matching
        "rollouts fetch the latest weights ... without waiting for the
        resharding and broadcast to complete [globally]"): if the chain has
        not reached this relay yet, wait for it; then pay the PCIe load.
        """
        target = next((p for p in reversed(self._publications)), None)
        if target is None:
            return 0
        self.stats.pulls += 1

        arrival = target.chain_ready[replica_id]
        now = self._now()
        if now < arrival:
            self.stats.pull_wait_total_s += arrival - now
            await asyncio.sleep(arrival - now)
        self.stats.pcie_total_s += self.config.pcie_pull_s
        await asyncio.sleep(self.config.pcie_pull_s)
        return target.version
