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
"""Trajectory-level producer: one response per queue message.

The stock ``FullyAsyncRollouter`` generates a whole prompt group (``rollout.n``
responses) as ONE blocking call and delivers ONE group-level message; the
slowest response of a group delays delivery of all its siblings. This subclass
delivers at single-response (trajectory) granularity:

* the group's ``n`` rows are submitted as ``n`` independent 1-row generations;
* each row delivers its own 1-row message the moment it completes, stamped
  with the group's ``uid``, its ``traj_index``, the ``group_size`` and the
  ``model_version`` it generated under;
* a row whose generation raises delivers a FAILED row (``rollout_failed``)
  so the trainer-side aggregator evicts just that group (failure isolation
  at trajectory granularity — no batch is held hostage);
* the group is re-assembled TRAINER-side by ``TrajectoryBatchCollector``
  (uid + traj_index), which also enforces per-group staleness.

Weight pulls: when a relay controller is attached
(``set_relay_controller``), each group submission is preceded by a
batch-boundary pull — if a newer version was published, the fleet pulls it
here, between groups, and the group is stamped with the version it will
generate under. This is the rollout-driven half of the pull-based weight
path (``relay_controller.py``).

Deployment-path note (README "Relation to the v1 separate-async stack"):
the v1 TransferQueue stack already produces at trajectory granularity
(``agent_loop_tq.py``); this producer is the MessageQueue-path equivalent,
chosen for P0 because ``TrajectoryAsyncTrainer`` extends the
``fully_async_policy`` trainer. On the v1 path this module is unnecessary.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import ray

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10, max_concurrency=100)
class TrajectoryLevelRollouter(FullyAsyncRollouter):
    """Fully-async rollouter delivering one message per response.

    Configured by presence: use this class instead of
    ``FullyAsyncRollouter`` in the launcher (``trajectory_async_main.py``)
    to switch the producer to trajectory granularity. The trainer side
    accepts either granularity interchangeably.
    """

    async def set_relay_controller(self, relay_controller):
        """Attach the versioned-weight relay controller (optional; enables
        rollout-driven batch-boundary pulls)."""
        self.relay_controller = relay_controller
        self._pull_version = 0  # version this fleet generates under
        # group submissions run concurrently (streaming processor); the pull
        # DECISION must be serialized so concurrent groups don't both see
        # "behind" and trigger back-to-back fleet pulls (the second would
        # abort the first's freshly resumed in-flight requests for nothing)
        self._pull_lock = asyncio.Lock()

    # ------------------------------------------------------------ delivery

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Split the group into per-row generations and deliver each row
        independently (trajectory-level producer)."""
        batch = rollout_sample.full_batch
        uid = f"uid_{rollout_sample.sample_id}"
        group_size = len(batch)

        # stock parity: embed uid BEFORE generation (agent-loop skip management)
        batch.non_tensor_batch["uid"] = np.array([uid] * group_size, dtype=object)

        # batch boundary: pull newer weights BEFORE submitting this group,
        # then stamp every row with the version it generates under
        version = await self._maybe_pull_weights()

        rows = batch.chunk(group_size)
        tasks = [
            asyncio.ensure_future(self._generate_and_deliver_row(row, uid, traj_index, group_size, version))
            for traj_index, row in enumerate(rows)
        ]
        # a row failing must not cancel its siblings — deliver FAILED rows
        # and let the trainer-side aggregator evict the group
        await asyncio.gather(*tasks, return_exceptions=False)
        self.processed_sample_count += 1

    async def _generate_and_deliver_row(
        self,
        row,
        uid: str,
        traj_index: int,
        group_size: int,
        version: int,
    ):
        try:
            ret = await self.async_rollout_manager.generate_sequences_single(row)
        except Exception:
            logger.exception("trajectory %s[%d] generation failed", uid, traj_index)
            await self._deliver_row(
                self._failed_row(uid, traj_index, group_size, version), uid, traj_index
            )
            return

        await self._deliver_row(ret, uid, traj_index, group_size, version)

    async def _deliver_row(self, row_batch, uid: str, traj_index: int, group_size: int, version: int):
        """Put one 1-row message into the queue, stamped for group re-assembly."""
        from verl.protocol import DataProto

        if not isinstance(row_batch, DataProto):
            row_batch = self._failed_row(uid, traj_index, group_size, version)

        ntb = row_batch.non_tensor_batch
        ntb["uid"] = np.array([uid], dtype=object)
        ntb["traj_index"] = np.array([traj_index], dtype=np.int64)
        ntb["group_size"] = np.array([group_size], dtype=np.int64)
        ntb["model_version"] = np.array([version], dtype=np.int64)
        ntb.setdefault("rollout_failed", np.array([False], dtype=bool))

        sample = RolloutSample(
            full_batch=row_batch,
            sample_id=f"{uid}:{traj_index}",
            epoch=0,
            rollout_status={},
        )
        success = await self.message_queue_client.put_sample(sample=ray.cloudpickle.dumps(sample))
        if success:
            self.total_generated_samples += 1
            self._step_generated_samples += 1
        else:
            self.dropped_stale_samples += 1

    def _failed_row(self, uid: str, traj_index: int, group_size: int, version: int):
        """Minimal 1-row message for a failed trajectory: no tensors, just
        the re-assembly keys + ``rollout_failed`` — the trainer-side
        aggregator's FAILED-sentinel protocol evicts the group."""
        from verl.protocol import DataProto

        row = DataProto(
            non_tensor_batch={
                "uid": np.array([uid], dtype=object),
                "traj_index": np.array([traj_index], dtype=np.int64),
                "group_size": np.array([group_size], dtype=np.int64),
                "model_version": np.array([version], dtype=np.int64),
                "rollout_failed": np.array([True], dtype=bool),
            }
        )
        return row

    # -------------------------------------------------------------- pulls

    async def _maybe_pull_weights(self) -> int:
        """Batch-boundary weight pull, decision serialized across concurrent
        group submissions: if the trainer published a newer version, pull it
        here (fleet-synchronized on the P0 topology) and return the version
        the next group generates under. In-flight requests on the replicas
        are aborted and RESUMED around the load (stock partial-rollout
        semantics — the interruption is invisible to the agent loop)."""
        controller = getattr(self, "relay_controller", None)
        if controller is None:
            return getattr(self, "_pull_version", 0)

        lock = getattr(self, "_pull_lock", None)
        if lock is None:  # set_relay_controller raced ahead of attribute init
            return getattr(self, "_pull_version", 0)

        async with lock:
            try:
                if not await controller.behind_latest.remote(self._pull_version):
                    return self._pull_version
                pulled = await controller.pull.remote()
                if pulled is not None:
                    self._pull_version = pulled
            except Exception:
                logger.exception("relay pull failed; generating on v%d", self._pull_version)
            return self._pull_version
