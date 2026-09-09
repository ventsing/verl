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
# WITHOUT WARRANTIES OR CONDITIONS of ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-process trajectory queue.

The trajectory-level analogue of ``fully_async_policy``'s ``MessageQueue``
Ray actor, shrunk to a stdlib asyncio queue so the whole data plane runs —
and is tested — without a Ray cluster. For a real multi-process deployment,
reuse ``MessageQueue`` verbatim: its ``put_sample``/``get_sample`` are
payload-agnostic, so a serialized :class:`TrajectorySample` flows through
unchanged (see README wiring guide).
"""

from __future__ import annotations

import asyncio
from typing import Any

from verl.experimental.trajectory_async.types import TrajectorySample


class InProcessTrajectoryQueue:
    """FIFO queue of trajectory units with close semantics and stats.

    ``None`` is the end-of-stream sentinel: the producer puts one sentinel
    per consumer after all real trajectories; ``get`` returns ``None`` once
    the queue is drained, unblocking consumers.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self.total_put = 0
        self.total_get = 0
        self.total_sentinels_put = 0
        self._peak_size = 0
        self._put_wait_total_s = 0.0
        self._get_wait_total_s = 0.0

    # ------------------------------------------------------------------ api

    async def put(self, item: TrajectorySample | None) -> None:
        """Put one trajectory (or ``None`` sentinel). Blocks on backpressure."""
        if item is None:
            self.total_sentinels_put += 1
        loop = asyncio.get_running_loop()
        start = loop.time()
        await self._queue.put(item)
        self._put_wait_total_s += loop.time() - start
        if item is not None:
            self.total_put += 1
        self._peak_size = max(self._peak_size, self._queue.qsize())

    def put_nowait(self, item: TrajectorySample | None) -> None:
        if item is None:
            self.total_sentinels_put += 1
        else:
            self.total_put += 1
        self._queue.put_nowait(item)
        self._peak_size = max(self._peak_size, self._queue.qsize())

    async def get(self) -> TrajectorySample | None:
        """Get the next trajectory; blocks while the queue is open and empty.

        Returns ``None`` only when a sentinel was consumed — after seeing a
        sentinel the caller should stop (with one sentinel per consumer,
        racing consumers each get exactly one).
        """
        loop = asyncio.get_running_loop()
        start = loop.time()
        item = await self._queue.get()
        self._get_wait_total_s += loop.time() - start
        if item is not None:
            self.total_get += 1
        return item

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    def task_done(self) -> None:
        self._queue.task_done()

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue/size": self._queue.qsize(),
            "queue/peak_size": self._peak_size,
            "queue/total_put": self.total_put,
            "queue/total_get": self.total_get,
            "queue/sentinels_put": self.total_sentinels_put,
            "queue/put_wait_total_s": round(self._put_wait_total_s, 6),
            "queue/get_wait_total_s": round(self._get_wait_total_s, 6),
        }
