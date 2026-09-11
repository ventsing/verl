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
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    # the mooncake ENGINE module needs torch + ray + mooncake + vllm/sglang;
    # gate the engine protocol smoke on the pieces it actually executes
    from mooncake.engine import TransferEngine  # noqa: F401

    HAS_MOONCAKE = True
except ImportError:
    HAS_MOONCAKE = False

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

    @unittest.skipUnless(
        HAS_MOONCAKE and HAS_TORCH, "mooncake engine + torch required (full install)"
    )
    def test_mooncake_versioned_engine_protocol(self):
        """The mooncake engine's versioned path (TODO-9) against stub
        transport/store: stage -> descriptor rendezvous -> per-bucket direct
        reads -> unstage/drop, with REAL tensor bytes flowing (the stub
        reads actually copy)."""
        import torch

        import verl.checkpoint_engine.mooncake_checkpoint_engine as mce
        from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine

        class _CpuDeviceShim:  # stage/receive call get_torch_device().synchronize()
            def synchronize(self):
                pass

            def empty_cache(self):
                pass

            def is_available(self):
                return False  # -> pageable staging buffers (CPU-only host)

        mce.get_torch_device = lambda: _CpuDeviceShim()
        from verl.utils.device import get_torch_device as _real_get_torch_device

        self.addCleanup(lambda: setattr(mce, "get_torch_device", _real_get_torch_device))

        class StubTransferEngine:
            def __init__(self):
                self.buffers = {}  # data_ptr -> tensor
                self.unregistered = []

            def _find(self, ptr):
                for base, buf in self.buffers.items():
                    if base <= ptr < base + buf.numel():
                        return base, buf
                raise KeyError(f"ptr {ptr:#x} not registered")

            def batch_register_memory(self, ptrs, sizes):
                return 0

            def register_buffer(self, buf):
                self.buffers[buf.data_ptr()] = buf

            def unregister_memory(self, ptr):
                self.unregistered.append(ptr)
                self.buffers.pop(ptr, None)
                return 0

            def transfer_sync_read(self, src_session, dst_ptr, src_ptr, length):
                src_base, src_buf = self._find(src_ptr)
                dst_base, dst_buf = self._find(dst_ptr)
                src_rel, dst_rel = src_ptr - src_base, dst_ptr - dst_base
                dst_buf.view(-1)[dst_rel : dst_rel + length].copy_(
                    src_buf.view(-1)[src_rel : src_rel + length]
                )
                return 0

        class StubStore:
            """Sequential all_gather_obj: rank 0 contributes the object,
            everyone else None (the engine only consumes info_list[0])."""

            def __init__(self):
                self.descriptor = None

            def all_gather_obj(self, obj):
                if obj is not None:
                    self.descriptor = obj
                return [self.descriptor, None]

        BUCKET = 4096

        def make_engine(te, store, rank):
            e = MooncakeCheckpointEngine.__new__(MooncakeCheckpointEngine)
            e.rank = rank
            e.bucket_size = BUCKET
            e.rollout_dtype = torch.float32
            e.device = "cpu"
            e.engine = te
            e.store = store
            e.session_id = "actor-session" if rank <= 0 else f"rollout-{rank}"
            e.buf = torch.zeros(2 * BUCKET, dtype=torch.uint8)
            te.register_buffer(e.buf)
            return e

        async def scenario():
            te = StubTransferEngine()
            store = StubStore()
            actor = make_engine(te, store, 0)  # the single sender (store rank 0)
            drain_actor = make_engine(te, store, -1)  # a non-sender actor rank
            rollout = make_engine(te, store, 1)  # a rollout consumer rank

            w1 = torch.arange(1024, dtype=torch.float32)  # 4096B = exactly one bucket
            w2 = torch.full((512,), 7, dtype=torch.float32)  # 2048B

            def weights_gen():
                yield "w.a", w1
                yield "w.b", w2

            # non-sender actor ranks drain (stock single-sender convention)
            drained = await drain_actor.stage_version(5, weights_gen())
            self.assertEqual(drained, {"staged_bytes": 0, "staged_params": 0})

            # stage + gather run concurrently in the real driver; sequentially
            # here: stage publishes the descriptor via the store rendezvous,
            # gather snapshots it
            metrics = await actor.stage_version(5, weights_gen())
            self.assertEqual(metrics["staged_params"], 2)
            self.assertEqual(metrics["staged_bytes"], 4096 + 2048)
            te.register_buffer(actor._staged_versions[5]["buf"])
            rollout.gather_version_metas(5)

            # whole-tensor buckets: w.a fills a bucket exactly, w.b the next
            desc = rollout._versioned_descriptors[5]
            self.assertEqual(len(desc["buckets"]), 2)
            self.assertIn("w.a", desc["buckets"][0]["tensors"])
            self.assertIn("w.b", desc["buckets"][1]["tensors"])

            # pull: per-bucket direct reads yield the true tensor values
            got = {}
            async for name, tensor in rollout.receive_weights_version(5, replica_id=0):
                got[name] = tensor.clone()
            self.assertTrue(torch.equal(got["w.a"], w1))
            self.assertTrue(torch.equal(got["w.b"], w2))

            # anytime + repeatable: a second pull of the same version works
            got2 = {}
            async for name, tensor in rollout.receive_weights_version(5):
                got2[name] = tensor.clone()
            self.assertTrue(torch.equal(got2["w.b"], w2))

            # unknown version -> the kimi-contract LookupError
            with self.assertRaises(LookupError):
                async for _ in rollout.receive_weights_version(99):
                    pass

            # retire: actor unregisters the staging buffer, rollout drops
            actor.unstage_version(5)
            self.assertEqual(te.unregistered, [desc["ptr"]])
            rollout.drop_version(5)
            self.assertNotIn(5, rollout._versioned_descriptors)

        import asyncio

        asyncio.run(scenario())

    @unittest.skipUnless(
        HAS_MOONCAKE and HAS_TORCH, "mooncake engine + torch required (full install)"
    )
    def test_mooncake_build_topology_accepts_replica_partition(self):
        """mooncake ignores the partition (direct P2P reads need no
        subgroups) but must ACCEPT it — the versioned driver passes it
        uniformly across backends."""
        from verl.checkpoint_engine.mooncake_checkpoint_engine import MooncakeCheckpointEngine

        actor_kw, rollout_kw = MooncakeCheckpointEngine.build_topology(
            2, 4, [{"addr": "h", "port": 1}], replica_partition=[[2, 3], [4, 5], [6, 7]]
        )
        self.assertNotIn("replica_partition", actor_kw)
        self.assertNotIn("replica_partition", rollout_kw)

    def test_kimi_topology_change_fails_loud(self):
        """Elastic-safety guard: a re-init with a CHANGED replica
        partition must raise (silently keeping stale subgroups would
        route per-replica pulls over the wrong topology); the SAME
        partition is an idempotent no-op (the failover factory rebuilds
        the controller without changing the fleet)."""
        from verl.checkpoint_engine.kimi_checkpoint_engine import KIMICheckpointEngine

        engine = KIMICheckpointEngine.__new__(KIMICheckpointEngine)  # no model paths
        engine.initialized = True
        engine._installed_partition = [[8, 9, 10, 11], [12, 13, 14, 15]]

        # same topology: silent no-op (failover re-init)
        engine.init_process_group(
            rank=9, actor_wg_world_size=8, rollout_world_size=8,
            master_metadata=None, replica_partition=[[8, 9, 10, 11], [12, 13, 14, 15]],
        )

        # elastic change: fail loud, never a silent stale topology
        with self.assertRaises(RuntimeError):
            engine.init_process_group(
                rank=9, actor_wg_world_size=8, rollout_world_size=8,
                master_metadata=None, replica_partition=[[8, 9, 10, 11]],
            )

        # flat -> partitioned is also a topology change
        flat = KIMICheckpointEngine.__new__(KIMICheckpointEngine)
        flat.initialized = True
        flat._installed_partition = None
        with self.assertRaises(RuntimeError):
            flat.init_process_group(
                rank=9, actor_wg_world_size=8, rollout_world_size=8,
                master_metadata=None, replica_partition=[[8, 9, 10, 11]],
            )
        # flat -> flat stays a no-op (the stock path)
        flat.init_process_group(
            rank=9, actor_wg_world_size=8, rollout_world_size=8,
            master_metadata=None, replica_partition=None,
        )

    def test_fault_tolerance_wiring(self):
        """§3.3/§4.3 wiring: producer lifecycle RPCs, controller
        ping/recover, pool actor factory, supervisor + pool exports."""
        from verl.experimental.trajectory_async import (
            PartialResponsePool,
            RelaySupervisor,
            ReplicaHealthMonitor,
            generate_row_with_retry,
        )
        from verl.experimental.trajectory_async.partial_pool import make_partial_pool_actor
        from verl.experimental.trajectory_async.relay_controller import make_relay_controller_actor
        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter

        import inspect

        for rpc in ("replica_probe_all", "replica_retire", "replica_revive", "set_partial_pool"):
            self.assertTrue(hasattr(TrajectoryLevelRollouter, rpc), rpc)

        # the retry policy consults the pool on retries (§3.1 consumer)
        src = inspect.getsource(generate_row_with_retry)
        self.assertIn("pool_consult_fn", src)

        # actor factories exist
        self.assertTrue(callable(make_partial_pool_actor))
        self.assertTrue(callable(make_relay_controller_actor))

    def test_drain_lifecycle_wiring(self):
        """Migration execution path: the producer exposes the drain
        lifecycle RPCs the repack bridge drives."""
        from verl.experimental.trajectory_async.rollout_producer import TrajectoryLevelRollouter

        import inspect

        for rpc in ("replica_drain", "replica_abort_all", "replica_resume"):
            self.assertTrue(hasattr(TrajectoryLevelRollouter, rpc), rpc)
            params = inspect.signature(getattr(TrajectoryLevelRollouter, rpc)).parameters
            self.assertTrue(all(p.kind is not p.VAR_POSITIONAL for p in params.values()), rpc)

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
