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
"""Trajectory-level asynchronous RL on verl (experimental).

Real components only — what is not implemented is a tracked TODO, never
a simulation. Two planes:

* **data plane** — trainer-side trajectory-level consumption:
  :class:`GroupAggregator` reassembles GRPO groups from per-trajectory
  rows, :class:`TrajectoryBatchCollector` forms exact mini-batches of
  complete, fresh groups under staleness control;
  :class:`TrajectoryAsyncTrainer` (async_trainer.py) wires this into the
  fully-async separate-deployment trainer.
* **systems plane** — Laminar-style weight and replica management:
  :class:`RelayService` (hierarchical relay tier over the P2P weight
  backends), :class:`VersionedWeightStore` + kimi/mooncake adapters
  (multi-version pull-based weights), :class:`RepackManager` +
  :func:`best_fit_consolidation` + :class:`RolloutRepackExecutor`
  (active scheduling over real rollout replicas).

See ``README.md`` for the design, the real-engine wiring guide, and the
cluster TODO list.
"""

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.group_collector import (
    TrajectoryBatchCollector,
    row_from_sample_batch,
)
from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.repack import (
    MigrationResult,
    RepackConfig,
    RepackExecutor,
    RepackManager,
    RepackStats,
    ReplicaState,
    best_fit_consolidation,
)
from verl.experimental.trajectory_async.relay_controller import (
    RelayController,
    build_relay_controller,
    make_relay_controller_actor,
)
from verl.experimental.trajectory_async.relay_controller import derive_replica_partition
from verl.experimental.trajectory_async.repack_bridge import (
    FleetRepackExecutor,
    RolloutReplicaView,
    build_repack_controller,
)
from verl.experimental.trajectory_async.row_retry import generate_row_with_retry
from verl.experimental.trajectory_async.staleness_correction import (
    StalenessCorrectionConfig,
    adaptive_clip_scale,
    apply_staleness_correction,
    cohort_advantages,
    cohort_stats,
    staleness_weight,
    staleness_weights,
)
from verl.experimental.trajectory_async.relay_tier import (
    RelayNode,
    RelayService,
    RelayTierAdapter,
    RelayTierConfig,
    RelayTierStats,
    RolloutRepackExecutor,
    RolloutReplicaHandle,
    RunningRequest,
)
from verl.experimental.trajectory_async.types import (
    GroupRecord,
    TrajectorySample,
    TrajectoryStatus,
)
from verl.experimental.trajectory_async.versioned_weight_store import (
    FakeP2PBackend,
    KimiP2PBackend,
    MooncakeP2PBackend,
    P2P_BACKENDS,
    P2PWeightBackend,
    ReadStats,
    VersionedWeightStore,
    WeightManifest,
    make_p2p_backend,
)

__all__ = [
    # data plane
    "GroupAggregator",
    "TrajectoryBatchCollector",
    "row_from_sample_batch",
    "MiniBatcher",
    "GroupRecord",
    "TrajectorySample",
    "TrajectoryStatus",
    # repack (algorithm + executor seam)
    "MigrationResult",
    "ReplicaState",
    "RepackConfig",
    "RepackExecutor",
    "RepackManager",
    "RepackStats",
    "best_fit_consolidation",
    "RolloutRepackExecutor",
    "RolloutReplicaHandle",
    "RunningRequest",
    # relay tier (weights)
    "RelayNode",
    "RelayService",
    "RelayTierAdapter",
    "RelayTierConfig",
    "RelayTierStats",
    # relay controller (Ray-native control plane of the versioned pull path)
    "RelayController",
    "build_relay_controller",
    "make_relay_controller_actor",
    "derive_replica_partition",
    # repack closed loop over the real rollout fleet
    "FleetRepackExecutor",
    "RolloutReplicaView",
    "build_repack_controller",
    # bounded row-generation retry (long-tail mitigation)
    "generate_row_with_retry",
    # loss-side staleness correction (version-aware off-policy layer)
    "StalenessCorrectionConfig",
    "adaptive_clip_scale",
    "apply_staleness_correction",
    "cohort_advantages",
    "cohort_stats",
    "staleness_weight",
    "staleness_weights",
    # weight store + P2P backends
    "FakeP2PBackend",
    "KimiP2PBackend",
    "MooncakeP2PBackend",
    "P2P_BACKENDS",
    "P2PWeightBackend",
    "ReadStats",
    "VersionedWeightStore",
    "WeightManifest",
    "make_p2p_backend",
]
