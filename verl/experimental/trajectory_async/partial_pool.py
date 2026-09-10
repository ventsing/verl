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
"""Partial Response Pool (paper §3.1): the fault-tolerance substrate.

The paper streams every in-progress trajectory to a central pool so that
when a replica fails, generation resumes from the saved partial on a
SAME-WEIGHT-VERSION replica instead of restarting from the prompt.

On this stack the pool is the trajectory-keyed, version-gated store
below; its honest wiring map (see the package README, TODO-16):

* **Store (real, this module)**: put/get/discard/complete with
  ``model_version`` gating — a partial generated under version v is only
  reusable while the generating fleet still runs v (the paper's
  same-version redirect rule; a cross-version resume would mix weight
  versions within one trajectory, the exact property trajectory-level
  delivery exists to protect). TTL + LRU + byte-quota eviction bound the
  memory: partials of retired versions age out even if nobody discards
  them.
* **Consumer (real)**: the producer's row-retry loop consults the pool
  on retry attempts — a same-version partial turns the retry into a
  resume (``resume_tokens`` hint attached to the row; see
  ``row_retry.py``), a miss restarts clean.
* **Writer (seam, deliberately external)**: token-level checkpointing
  must sit where partial tokens are observable — the LLM client's
  generation loop or a resume-aware agent loop. Both are shared rollout
  infrastructure outside this package's boundary today; the pool actor
  exposes ``put`` for whoever lands there (the L2 scheduling stack's
  abort-resume client is the natural home).

The stdlib core is CPU-testable without ray; :func:`make_partial_pool_actor`
wraps it as a Ray actor.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PartialProgress:
    """One trajectory's saved partial progress.

    Attributes:
        uid: group (prompt) id.
        traj_index: trajectory index within the group.
        model_version: the weight version the partial was generated
            under — the reuse gate.
        tokens: the response tokens generated so far.
        updated_at: monotonic timestamp of the last update.
        complete: True once the trajectory finished (terminal marker —
            retrievable for diagnostics, never resumed from).
    """

    uid: str
    traj_index: int
    model_version: int
    tokens: list = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)
    complete: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.uid, self.traj_index)


class PartialResponsePool:
    """Central, version-gated store of in-progress trajectories.

    Bounded three ways: entry count (LRU), total bytes (oldest-first
    eviction — never the entry just put), and TTL (partials of versions
    the fleet has moved past expire even without explicit discards).
    All bounds are soft-best-effort: the just-put entry always survives
    its own put.
    """

    def __init__(
        self,
        max_entries: int = 4096,
        max_bytes: int | None = None,
        ttl_s: float | None = None,
        clock: Any = time.monotonic,
    ):
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = int(max_bytes) if max_bytes is not None else None
        self.ttl_s = float(ttl_s) if ttl_s is not None else None
        self._clock = clock
        # (uid, traj_index) -> PartialProgress, LRU order (oldest first)
        self._entries: OrderedDict[tuple[str, int], PartialProgress] = OrderedDict()

        # counters
        self.puts = 0
        self.hits = 0
        self.misses = 0
        self.version_mismatches = 0
        self.expired = 0
        self.evicted_lru = 0
        self.evicted_bytes = 0
        self.discarded = 0

    # ------------------------------------------------------------- size

    @staticmethod
    def _entry_bytes(progress: PartialProgress) -> int:
        # token ids are ints; a Python-level estimate (the real payload
        # may be tensors — whatever the writer puts, we count len())
        try:
            return 8 * len(progress.tokens)
        except TypeError:
            return 8

    def _now(self) -> float:
        return self._clock()

    # ------------------------------------------------------------ writes

    def put(
        self,
        uid: str,
        traj_index: int,
        model_version: int,
        tokens,
        complete: bool = False,
    ) -> None:
        """Save/overwrite one trajectory's partial (same key, same-or-newer
        version). A stale-version overwrite is refused (the newer partial
        is what a resume wants)."""
        key = (uid, traj_index)
        existing = self._entries.get(key)
        if existing is not None and existing.model_version > model_version:
            return  # never regress to an older version's partial
        self._entries[key] = PartialProgress(
            uid=uid,
            traj_index=traj_index,
            model_version=model_version,
            tokens=list(tokens) if tokens is not None else [],
            updated_at=self._now(),
            complete=complete,
        )
        self._entries.move_to_end(key)
        self.puts += 1
        self._evict_unless(key)

    def discard(self, uid: str, traj_index: int) -> None:
        """Drop one entry (trajectory delivered / group resolved)."""
        if self._entries.pop((uid, traj_index), None) is not None:
            self.discarded += 1

    def purge_expired(self) -> int:
        """Drop entries older than the TTL. Returns the count purged."""
        if self.ttl_s is None:
            return 0
        now = self._now()
        dead = [k for k, e in self._entries.items() if now - e.updated_at > self.ttl_s]
        for k in dead:
            self._entries.pop(k, None)
        self.expired += len(dead)
        return len(dead)

    def _evict_unless(self, keep: tuple[str, int]) -> None:
        """Enforce entry and byte bounds, oldest-first, never `keep`."""
        if len(self._entries) > self.max_entries:
            overflow = len(self._entries) - self.max_entries
            for key in list(self._entries.keys()):
                if overflow <= 0:
                    break
                if key == keep:
                    continue
                self._entries.pop(key, None)
                self.evicted_lru += 1
                overflow -= 1
        if self.max_bytes is not None:
            total = sum(self._entry_bytes(e) for e in self._entries.values())
            for key in list(self._entries.keys()):
                if total <= self.max_bytes or len(self._entries) <= 1:
                    break
                if key == keep:
                    continue
                entry = self._entries.pop(key, None)
                if entry is not None:
                    freed = self._entry_bytes(entry)
                    total -= freed
                    self.evicted_bytes += freed

    # ------------------------------------------------------------- reads

    def get(self, uid: str, traj_index: int, current_version: int) -> PartialProgress | None:
        """Fetch a REUSABLE partial: same version, not complete, not
        expired. Anything else is a miss (with the mismatch counted)."""
        self.purge_expired()
        entry = self._entries.get((uid, traj_index))
        if entry is None:
            self.misses += 1
            return None
        if entry.complete:
            self.misses += 1  # terminal — nothing to resume
            return None
        if entry.model_version != current_version:
            # same-version redirect rule: the partial's tokens were
            # sampled under other weights; resuming would mix versions
            # within one trajectory — refuse and drop the stale entry
            self._entries.pop((uid, traj_index), None)
            self.version_mismatches += 1
            return None
        self._entries.move_to_end((uid, traj_index))
        self.hits += 1
        return entry

    # ---------------------------------------------------------- metrics

    def snapshot(self) -> dict[str, Any]:
        return {
            "partial_pool/entries": len(self._entries),
            "partial_pool/bytes": sum(self._entry_bytes(e) for e in self._entries.values()),
            "partial_pool/puts": self.puts,
            "partial_pool/hits": self.hits,
            "partial_pool/misses": self.misses,
            "partial_pool/version_mismatches": self.version_mismatches,
            "partial_pool/expired": self.expired,
            "partial_pool/evicted_lru": self.evicted_lru,
            "partial_pool/evicted_bytes": self.evicted_bytes,
            "partial_pool/discarded": self.discarded,
        }


def make_partial_pool_actor():
    """Wrap the pool as a Ray actor (named for discovery by writers)."""
    import ray

    @ray.remote(num_cpus=0, max_concurrency=10)
    class PartialPoolActor:
        """Central partial-response store. Writers (the LLM client seam)
        put; the producer's row-retry loop gets; the trainer snapshots."""

        def __init__(self, max_entries: int = 4096, max_bytes=None, ttl_s=None):
            self._pool = PartialResponsePool(
                max_entries=max_entries, max_bytes=max_bytes, ttl_s=ttl_s
            )

        def put(self, uid, traj_index, model_version, tokens, complete=False):
            self._pool.put(uid, traj_index, model_version, tokens, complete=complete)

        def get(self, uid, traj_index, current_version):
            progress = self._pool.get(uid, traj_index, current_version)
            if progress is None:
                return None
            return {
                "uid": progress.uid,
                "traj_index": progress.traj_index,
                "model_version": progress.model_version,
                "tokens": list(progress.tokens),
            }

        def discard(self, uid, traj_index):
            self._pool.discard(uid, traj_index)

        def snapshot(self):
            return self._pool.snapshot()

    return PartialPoolActor
