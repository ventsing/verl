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
"""Cluster-gated wiring smoke: the P0 launch path imports cleanly and the
component relationships hold. Skipped without ray/numpy (a full install);
run on the cluster or CI:

    python -m unittest tests.experimental.trajectory_async.test_wiring_smoke -v
"""

import inspect
import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

try:
    import ray  # noqa: F401

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@unittest.skipUnless(HAS_RAY, "ray required for wiring smoke (full install)")
class TestP0Wiring(unittest.TestCase):
    def test_producer_and_trainer_import(self):
        from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
        from verl.experimental.trajectory_async.async_trainer import TrajectoryAsyncTrainer
        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter

        # producer is a drop-in rollouter replacement
        self.assertTrue(issubclass(TrajectoryLevelRollouter, FullyAsyncRollouter))
        # trainer overrides the trajectory-async extension points
        for method in (
            "_get_samples_from_queue",
            "_fit_update_weights",
            "_setup_checkpoint_manager",
            "_publish_versioned_weights",
        ):
            self.assertTrue(
                hasattr(TrajectoryAsyncTrainer, method),
                f"TrajectoryAsyncTrainer missing {method}",
            )

    def test_launcher_import(self):
        from verl.experimental.trajectory_async import trajectory_async_main

        self.assertTrue(hasattr(trajectory_async_main, "TrajectoryAsyncTaskRunner"))
        self.assertTrue(hasattr(trajectory_async_main, "main"))

    def test_engine_worker_methods_exist(self):
        """The kimi engine + worker classes expose the versioned path."""
        from verl.checkpoint_engine.kimi_checkpoint_engine import KIMICheckpointEngine

        for method in (
            "stage_version",
            "gather_version_metas",
            "receive_weights_version",
            "unstage_version",
            "drop_version",
        ):
            self.assertTrue(hasattr(KIMICheckpointEngine, method), f"kimi engine missing {method}")

        from verl.workers.engine_workers import TrainingWorker

        self.assertTrue(hasattr(TrainingWorker, "stage_weights_version"))

        from verl.checkpoint_engine.base import CheckpointEngineWorker

        for method in ("gather_version_metas", "pull_weights_version"):
            self.assertTrue(hasattr(CheckpointEngineWorker, method), f"engine worker missing {method}")

    def test_per_replica_pull_signatures(self):
        """The per-replica seam: engine partition + replica-scoped pulls."""
        import inspect

        from verl.checkpoint_engine.kimi_checkpoint_engine import KIMICheckpointEngine

        engine_params = inspect.signature(KIMICheckpointEngine.receive_weights_version).parameters
        self.assertIn("replica_id", engine_params)
        init_params = inspect.signature(KIMICheckpointEngine.init_process_group).parameters
        self.assertIn("replica_partition", init_params)
        topology_params = inspect.signature(KIMICheckpointEngine.build_topology).parameters
        self.assertIn("replica_partition", topology_params)
        # stage_version reports staged sizes (host-memory quota accounting)
        from verl.workers.engine_workers import TrainingWorker

        self.assertTrue(hasattr(TrainingWorker, "stage_weights_version"))

    def test_controller_and_bridge_wiring(self):
        """The per-replica/quota controller + the repack closed loop."""
        from verl.experimental.trajectory_async.relay_controller import (
            RelayController,
            build_relay_controller,
            derive_replica_partition,
            make_relay_controller_actor,
        )

        for method in ("pull_replica", "replica_version"):
            self.assertTrue(hasattr(RelayController, method), f"controller missing {method}")
        controller_params = inspect.signature(build_relay_controller).parameters
        self.assertIn("max_staged_bytes", controller_params)

        from verl.experimental.trajectory_async.repack_bridge import (
            FleetRepackExecutor,
            RolloutReplicaView,
            build_repack_controller,
            make_repack_controller_actor,
        )

        for method in ("refresh_idle", "migrate", "snapshot"):
            self.assertTrue(hasattr(FleetRepackExecutor, method), f"bridge executor missing {method}")
        for method in ("pull_weights", "running_requests", "remove_request", "admit_request"):
            self.assertTrue(hasattr(RolloutReplicaView, method), f"replica view missing {method}")

    def test_row_retry_wiring(self):
        """The producer's retry seam: policy function importable, deliver
        stamps attempts, the trainer passes the budget."""
        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter
        from verl.experimental.trajectory_async.row_retry import generate_row_with_retry

        import inspect

        deliver_params = inspect.signature(TrajectoryLevelRollouter._deliver_row).parameters
        self.assertIn("attempts", deliver_params)
        set_rc_params = inspect.signature(TrajectoryLevelRollouter.set_relay_controller).parameters
        self.assertIn("row_max_attempts", set_rc_params)
        # attempts default 1 keeps non-retry deliveries single-shot
        self.assertEqual(deliver_params["attempts"].default, 1)
        self.assertTrue(callable(generate_row_with_retry))

    def test_producer_lb_probes_exist(self):
        """The producer exposes the LB probes the repack bridge consumes."""
        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter

        for method in ("replica_server_ids", "replica_inflight"):
            self.assertTrue(hasattr(TrajectoryLevelRollouter, method), f"producer missing {method}")

    def test_rollouter_row_message_fields(self):
        """A delivered row message carries the re-assembly keys; a failed row
        carries rollout_failed=True (the FAILED-sentinel protocol)."""
        import numpy as np  # noqa: F401

        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter
        from verl.protocol import DataProto

        class _Bare:
            """Drive the unbound methods without instantiating the Ray actor."""

        bare = _Bare()
        failed = TrajectoryLevelRollouter._failed_row(bare, "uid_x", 2, 4, 9)
        self.assertIsInstance(failed, DataProto)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed.non_tensor_batch["uid"][0], "uid_x")
        self.assertEqual(failed.non_tensor_batch["traj_index"][0], 2)
        self.assertEqual(failed.non_tensor_batch["group_size"][0], 4)
        self.assertEqual(failed.non_tensor_batch["model_version"][0], 9)
        self.assertEqual(failed.non_tensor_batch["rollout_failed"][0], True)

        # the trainer-side adapter accepts exactly this shape
        from verl.experimental.trajectory_async.group_collector import row_from_sample_batch

        row = row_from_sample_batch("uid_x", failed, 0)
        self.assertEqual(row["uid"], "uid_x")
        self.assertEqual(row["traj_index"], 2)
        self.assertEqual(row["group_size"], 4)
        self.assertEqual(row["model_version"], 9)
        self.assertTrue(row["failed"])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
