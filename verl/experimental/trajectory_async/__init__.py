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

Streams *single responses* (trajectories) from the rollout side as soon as
each one finishes, and reassembles prompt groups on the trainer side for
group-based advantage estimation (GRPO/DAPO). See ``README.md`` for the
design, the honest benefit model, and the real-engine wiring guide.
"""

from verl.experimental.trajectory_async.group_aggregator import GroupAggregator
from verl.experimental.trajectory_async.group_collector import (
    TrajectoryBatchCollector,
    row_from_sample_batch,
)
from verl.experimental.trajectory_async.mini_batcher import MiniBatcher
from verl.experimental.trajectory_async.mock_rollout import (
    MockEngineConfig,
    MockRolloutEngine,
    MockRolloutError,
)
from verl.experimental.trajectory_async.multi_replica_engine import (
    MultiReplicaEngine,
    MultiReplicaEngineConfig,
    ReplicaState,
)
from verl.experimental.trajectory_async.repack import (
    RepackConfig,
    RepackManager,
    RepackStats,
    best_fit_consolidation,
)
from verl.experimental.trajectory_async.rollouter import (
    PromptRecord,
    RollouterConfig,
    RollouterStats,
    TrajectoryRollouter,
)
from verl.experimental.trajectory_async.trainer import (
    TrainerBatch,
    TrainerConfig,
    TrainerStats,
    TrajectoryTrainer,
    grpo_group_advantages,
)
from verl.experimental.trajectory_async.trajectory_queue import InProcessTrajectoryQueue
from verl.experimental.trajectory_async.types import (
    GroupRecord,
    TrajectorySample,
    TrajectoryStatus,
)
from verl.experimental.trajectory_async.relay_tier import (
    RelayNode,
    RelayService,
    RelayTierAdapter,
    RelayTierConfig,
    RelayTierStats,
    RepackExecutor,
    RolloutRepackExecutor,
    RolloutReplicaHandle,
    RunningRequest,
)
from verl.experimental.trajectory_async.versioned_weight_store import (
    FakeP2PBackend,
    KimiP2PBackend,
    MooncakeP2PBackend,
    P2P_BACKENDS,
    P2PWeightBackend,
    ReadStats,
    VersionedStoreRelayAdapter,
    VersionedWeightStore,
    WeightManifest,
    make_p2p_backend,
)
from verl.experimental.trajectory_async.weight_relay import (
    RelayConfig,
    RelayStats,
    WeightRelayService,
)

__all__ = [
    "GroupAggregator",
    "MiniBatcher",
    "MockEngineConfig",
    "MockRolloutEngine",
    "MockRolloutError",
    "MultiReplicaEngine",
    "MultiReplicaEngineConfig",
    "ReplicaState",
    "RepackConfig",
    "RepackManager",
    "RepackStats",
    "best_fit_consolidation",
    "PromptRecord",
    "RollouterConfig",
    "RollouterStats",
    "TrajectoryRollouter",
    "TrainerBatch",
    "TrainerConfig",
    "TrainerStats",
    "TrajectoryTrainer",
    "grpo_group_advantages",
    "InProcessTrajectoryQueue",
    "GroupRecord",
    "TrajectorySample",
    "TrajectoryStatus",
    "FakeP2PBackend",
    "KimiP2PBackend",
    "MooncakeP2PBackend",
    "P2P_BACKENDS",
    "P2PWeightBackend",
    "ReadStats",
    "RelayNode",
    "RelayService",
    "RelayTierAdapter",
    "RelayTierConfig",
    "RelayTierStats",
    "RepackExecutor",
    "RolloutRepackExecutor",
    "RolloutReplicaHandle",
    "RunningRequest",
    "VersionedStoreRelayAdapter",
    "VersionedWeightStore",
    "WeightManifest",
    "make_p2p_backend",
    "RelayConfig",
    "RelayStats",
    "WeightRelayService",
]
