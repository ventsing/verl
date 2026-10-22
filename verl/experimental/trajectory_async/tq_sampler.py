# Copyright 2026 verl contributors
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
"""Real-engine trajectory ingestion through V1's TransferQueue data plane.

AgentLoopWorkerTQ already writes each scored response independently. This sampler
feeds those responses to GroupAggregator on the trainer side, including responses
whose siblings are still running. Tensor payloads remain in TransferQueue.

GRPO/GSPO still update on complete groups. The prompt's terminal marker is also
required before retirement: it guarantees no producer can recreate a cleared key.
The existing async sampler retains responsibility for staleness eviction/refill.
Only single-output agent sessions are supported (the single-turn Qwen3 launcher).
Weight synchronization uses the trainer's checkpoint engine, not the CPU demo's
simulated relay or its experimental multi-version P2P store.
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.types import TrajectorySample
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

logger = logging.getLogger(__name__)


class TrajectoryReplayBuffer(ReplayBufferAsync):
    """Observe one response at a time and assemble prompt groups on the trainer."""

    def __init__(self, **kwargs):
        kwargs.setdefault("poll_interval", 0.2)
        super().__init__(**kwargs)
        if self.trainer_mode == "sync":
            raise ValueError("TrajectoryReplayBuffer requires an asynchronous trainer mode")
        self.rollout_n = int(self.sampler_kwargs.get("rollout_n", 0))
        self.val_rollout_n = int(self.sampler_kwargs.get("val_rollout_n", 1))
        if self.rollout_n < 2 or self.val_rollout_n < 1:
            raise ValueError("Set sampler_kwargs.rollout_n >= 2 and val_rollout_n >= 1")
        self.aggregators = defaultdict(GroupAggregator)
        self.ready_groups = defaultdict(dict)
        self.seen_keys = defaultdict(set)
        self.last_selected_tags = {}
        self.selection_strategy = self.sampler_kwargs.get("selection_strategy", "completion")
        if self.selection_strategy not in {"completion", "mixed_ready"}:
            raise ValueError("Unknown trajectory selection_strategy")
        self.selection_rng = random.Random(self.sampler_kwargs.get("selection_seed", 42))
        log_dir = self.sampler_kwargs.get("trajectory_log_dir")
        self.trajectory_log_dir = Path(log_dir) if log_dir else None
        if self.trajectory_log_dir is not None:
            self.trajectory_log_dir.mkdir(parents=True, exist_ok=True)

    def _record_trajectories(self, partition_id, records):
        if self.trajectory_log_dir is not None and records:
            with (self.trajectory_log_dir / f"{partition_id}.jsonl").open("a") as destination:
                for record in records:
                    destination.write(json.dumps(record) + "\n")

    def _sync_metadata_from_transfer_queue(self):
        if health_check := getattr(self, "health_check", None):
            health_check()
        super()._sync_metadata_from_transfer_queue()
        for partition_id, partition in self.partitions.items():
            # V1 owns these two rollout partitions. TQ can also contain
            # unrelated application/test partitions with different group sizes.
            if partition_id not in ("train", "val"):
                continue
            arrivals = []
            aggregator = self.aggregators[partition_id]
            group_size = self.val_rollout_n if partition_id == "val" else self.rollout_n
            for key, tag in partition.items():
                if key in self.seen_keys[partition_id] or tag.get("status") != "success":
                    continue
                uid, session_id, output_index = key.rsplit("_", 2)
                if uid not in self.prompt_global_steps[partition_id]:
                    # Sampled tensor rows remain in TQ until the training step clears them.
                    continue
                if output_index != "0":
                    raise ValueError("TrajectoryReplayBuffer only supports single-output agent sessions")
                version = tag.get("min_global_steps")
                if version is None:
                    version = tag["global_steps"]
                record = aggregator.add_trajectory(
                    TrajectorySample(
                        uid=uid,
                        traj_index=int(session_id),
                        group_size=group_size,
                        payload=key,
                        model_version=int(version),
                        num_tokens=int(tag.get("response_len", 0)),
                    )
                )
                self.seen_keys[partition_id].add(key)
                arrivals.append(
                    {
                        "event": "arrived",
                        "key": key,
                        "uid": uid,
                        "response_tokens": int(tag.get("response_len", 0)),
                        "model_version": int(version),
                        "observed_at": time.time(),
                        "scheduler_request_id": tag.get("trajectory_request_id"),
                        "migrations": int(tag.get("migrations", 0)),
                    }
                )
                if record is not None:
                    self.ready_groups[partition_id][uid] = record

            self._record_trajectories(partition_id, arrivals)

            if partition_id == "val" and self.failure_keys[partition_id]:
                raise RuntimeError(f"Validation trajectories failed: {sorted(self.failure_keys[partition_id])}")
            incomplete = self.finished_keys[partition_id] - self.ready_groups[partition_id].keys()
            if incomplete:
                raise RuntimeError(
                    f"Finished prompts lack {group_size} trajectories: {sorted(incomplete)}. "
                    "Check sampler_kwargs.rollout_n against rollout.n."
                )

    def _sampleable_terminal_keys(self, partition_id, eviction_reasons):
        keys = super()._sampleable_terminal_keys(partition_id, eviction_reasons)
        return keys & self.ready_groups[partition_id].keys() & self.finished_keys[partition_id]

    def _select_prompt_uids(self, partition_id, sampleable_keys, batch_size):
        ordered = sorted(sampleable_keys, key=lambda uid: self.ready_groups[partition_id][uid].complete_time)
        # Mix only already-complete groups: never wait for a slow group, inspect
        # its reward, or cross a group boundary. Retire the oldest every update
        # to bound starvation while mixing the other ready groups reproducibly.
        if partition_id == "train" and self.selection_strategy == "mixed_ready" and len(ordered) > batch_size:
            ordered = ordered[:1] + self.selection_rng.sample(ordered[1:], batch_size - 1)
        return (
            ordered[:batch_size],
            dict(self.partitions[partition_id]),
            dict(self.prompt_global_steps[partition_id]),
        )

    def _forget_groups(self, partition_id, uids):
        for uid in uids:
            self.aggregators[partition_id].evict_group(uid, reason="terminal eviction")
            self.ready_groups[partition_id].pop(uid, None)
        self.seen_keys[partition_id].difference_update(
            key for key in list(self.seen_keys[partition_id]) if key.rsplit("_", 2)[0] in uids
        )

    def _clear_groups(self, partition_id, uids):
        super()._clear_groups(partition_id, uids)
        self._forget_groups(partition_id, uids)

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        keys = [
            trajectory.payload
            for uid in selected_prompt_uids
            for trajectory in self.ready_groups[partition_id][uid].trajectories
        ]
        tags = [partition_snapshot[key] for key in keys]
        self.last_selected_tags[partition_id] = tags
        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)
        self._forget_groups(partition_id, set(selected_prompt_uids))
        return KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)

    def sample(self, global_steps, partition_id, batch_size):
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        tags = self.last_selected_tags[partition_id]
        lengths = np.array([tag.get("response_len", 0) for tag in tags], dtype=np.int64)
        if lengths.size:
            prefix = f"trajectory/{partition_id}/length"
            for percentile in (50, 90, 95, 99):
                metrics[f"{prefix}/p{percentile}"] = float(np.percentile(lengths, percentile))
            for threshold in (4096, 8192, 16384, 24576, 32768, 49152):
                metrics[f"{prefix}/ge_{threshold}_ratio"] = float(np.mean(lengths >= threshold))
            limit = self.sampler_kwargs.get("response_limit")
            if limit:
                metrics[f"{prefix}/budget_reached_ratio"] = float(np.mean(lengths >= int(limit)))
        self._record_trajectories(
            partition_id, [{"event": "selected", "key": key, "step": global_steps} for key in batch.keys]
        )
        metrics.update(
            {
                f"trajectory/{partition_id}/{name}": value
                for name, value in self.aggregators[partition_id].snapshot().items()
            }
        )
        logger.info(
            "Trajectory-level sample: partition=%s, groups=%d, responses=%d, partial_groups=%d",
            partition_id,
            batch_size,
            len(batch.keys),
            self.aggregators[partition_id].num_partial_groups,
        )
        return batch, metrics
