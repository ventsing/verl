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
"""The REAL trainer for trajectory-level async RL — separate deployment.

This is the deployable trainer, structured after
``verl/experimental/fully_async_policy/fully_async_trainer.py``:

* **separate deployment** — inherited from :class:`FullyAsyncTrainer`
  (itself a :class:`SeparateRayPPOTrainer`): training workers are
  ``Role.Actor`` on the trainer GPUs, rollout replicas live on the rollout
  side managed by the rollout stack, connected through the message queue; weight
  sync goes through the ``CheckpointEngineManager`` over the replicas.
* **trajectory-level consumption** — the stock trainer collects samples
  by COUNT (``required_samples``), where one queue message is a whole
  prompt group (``rollout.n`` rows). This subclass routes every consumed
  message through :class:`TrajectoryBatchCollector`, which re-assembles
  GRPO groups from per-trajectory rows and only emits mini-batches of
  *complete, fresh* groups:
  ``ppo_mini_batch_size`` groups per update, exact, under staleness
  control — so a trajectory-level producer (one response per message)
  becomes a drop-in, while a group-level producer still works (rows are
  split and re-aggregated, gaining per-group staleness accounting).
* **group-granular staleness/refusal accounting** — per-update metrics
  ``trajectory_async/*``: groups trained / evicted / dropped-stale /
  leftover / incomplete, staleness + version-span summaries.

NOTE — v1 stack: verl's newer separate-async path
(`verl/trainer/ppo/v1/`) stores experiences in a TransferQueue-backed
`ReplayBuffer` (per-trajectory keys, group re-assembly, staleness
eviction, refill) fed by `agent_loop_tq.py` (one kv_put per agent-loop
output). On that path this trainer's collector is redundant; what ports
is the relay tier (`relay_tier.py`) and the repack executor. See the
README section "Relation to the v1 separate-async stack".

Weight versioning: two paths. Default = the stock push-based
``CheckpointEngineManager.update_weights`` (versioned by
``global_steps``). With ``async_training.weight_store`` set, the
Laminar-style multi-version pull path takes over: the trainer only
PUBLISHES (``stage_weights_version`` on the actor ranks — version-scoped
registration in the kimi P2P store, no tensor push; the metadata gather
is the whole trainer-side stall) through the relay controller, and the
rollout side pulls at its batch boundaries
(``rollout_producer.py``). See ``relay_controller.py`` and the package
README's real-engine wiring guide.
"""

from __future__ import annotations

import logging
import time

import ray

from verl.experimental.fully_async_policy.detach_utils import assemble_batch_from_rollout_samples
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.trajectory_async.group_collector import (
    TrajectoryBatchCollector,
    row_from_sample_batch,
)

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=10)
class TrajectoryAsyncTrainer(FullyAsyncTrainer):
    """Fully-async PPO trainer with trajectory-level, group-aware consumption.

    Extends :class:`FullyAsyncTrainer` (separate deployment:
    ``SeparateRayPPOTrainer``) with:

    1. group re-assembly on the trainer side
       (:class:`TrajectoryBatchCollector`) — accepts both group-level and
       trajectory-level producers;
    2. freshness control (``async_training.staleness_drop``) with exact
       ``ppo_mini_batch_size`` batches;
    3. the ``trajectory_async/*`` metric family;
    4. an optional multi-version pull-based weight path
       (``async_training.weight_store.*``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        async_cfg = self.config.async_training
        self.trajectory_staleness_drop = async_cfg.get("staleness_drop", None)
        self.trajectory_group_assembly = async_cfg.get("trajectory_group_assembly", True)
        rollout_n = self.config.actor_rollout_ref.rollout.n

        self.trajectory_collector = TrajectoryBatchCollector(
            mini_batch_groups=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            max_staleness_drop=self.trajectory_staleness_drop,
            group_size=rollout_n,
            on_group_complete=self._on_group_complete,
        )
        self._terminated = False

    # ------------------------------------------------------------ callbacks

    def _on_group_complete(self, group) -> None:
        logger.info(
            "group %s complete (%d trajectories, version span %d, train-ready %.3fs)",
            group.uid,
            group.group_size,
            group.version_span,
            group.train_ready_latency_s,
        )

    # ------------------------------------------------------------- intake

    def _feed_collector(self, rollout_sample) -> None:
        """Route one queue message (either granularity) into the collector.

        Group-level producer: the sample's ``n`` rows share one uid;
        ``row_from_sample_batch`` reads per-row ``traj_index`` /
        ``model_version`` / ``rollout_failed`` when the producer set them
        (trajectory-mode producers do), falling back to positional split.
        Row payloads ride inside the GroupRecord (``trajectory.payload``)
        so a completed group re-assembles directly into a DataProto.
        """
        batch = rollout_sample.full_batch
        uid = str(rollout_sample.sample_id)
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        if "uid" in ntb and len(ntb["uid"]):
            uid = str(ntb["uid"][0])

        rows = [row_from_sample_batch(uid, batch, position) for position in range(len(batch))]
        if len(rows) == 1:
            row = rows[0]
            self.trajectory_collector.add_trajectory(
                uid=uid,
                traj_index=row["traj_index"],
                group_size=row["group_size"],
                model_version=row["model_version"],
                reward=row["reward"],
                num_tokens=row["num_tokens"],
                failed=row["failed"],
                payload=row["payload"],
            )
        else:
            self.trajectory_collector.add_sample(uid, rows)

    def _assemble_group_batch(self, groups: list):
        """Concat per-group row payloads (traj_index order) into one
        gen_batch_output, reusing the stock assembly utilities."""
        from verl.experimental.fully_async_policy.detach_utils import RolloutSample

        group_samples = []
        for group in groups:
            payloads = [t.payload for t in group.trajectories]  # traj_index order
            payloads = [p for p in payloads if p is not None]
            if not payloads:
                continue
            concat = payloads[0]
            if len(payloads) > 1:
                concat = concat.concat(payloads[1:])
            group_samples.append(
                RolloutSample(
                    full_batch=concat,
                    sample_id=group.uid,
                    epoch=0,
                    rollout_status={},
                )
            )
        if not group_samples:
            return None
        if self.config.trainer.balance_batch:
            return assemble_batch_from_rollout_samples(
                group_samples, self.tokenizer, self.config, self._balance_batch
            )
        return assemble_batch_from_rollout_samples(group_samples, self.tokenizer, self.config, None)

    # ------------------------------------------------------------- override

    async def _get_samples_from_queue(self):
        """Trajectory-level consumption: collect messages until
        ``ppo_mini_batch_size`` complete, fresh groups are pending, then
        assemble their rows into one gen_batch_output."""
        if not self.trajectory_group_assembly:
            return await super()._get_samples_from_queue()

        consumer_start = time.time()
        while True:
            batch_groups = self.trajectory_collector.take_mini_batch(self.current_param_version)
            if batch_groups is not None:
                break
            if self._terminated:
                # stream ended without enough fresh groups: final accounting
                self.trajectory_collector.finalize()
                return None, None
            sample, queue_len = await self.message_queue_client.get_sample()
            if sample is None:
                self._terminated = True
                continue
            self._feed_collector(sample)

        batch = self._assemble_group_batch(batch_groups)
        if batch is None:
            return None, None
        total_wait_time = time.time() - consumer_start
        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        self._step_wait_times.append(total_wait_time)
        self._step_wait_samples.append(len(batch_groups))
        self._log_trajectory_metrics(batch_groups)
        return 0, batch

    def _log_trajectory_metrics(self, batch_groups: list) -> None:
        """Emit the trajectory_async/* metric family for this update."""
        stats = self.trajectory_collector.stats
        self.metrics.update(stats.snapshot())
        # per-batch (not cumulative) summaries
        spans = [g.version_span for g in batch_groups]
        if spans:
            self.metrics["trajectory_async/batch_version_span_mean"] = sum(spans) / len(spans)
            self.metrics["trajectory_async/batch_version_span_max"] = max(spans)
        self.metrics["trajectory_async/current_param_version"] = self.current_param_version

    async def _setup_checkpoint_manager(self):
        """After the stock manager setup: when the multi-version pull path is
        selected, also build the relay controller (a Ray actor over the actor
        worker group + rollout replicas) and hand it to the rollouter so its
        group submissions pull weights at their batch boundaries."""
        await super()._setup_checkpoint_manager()

        weight_store_cfg = self.config.async_training.get("weight_store", None)
        if weight_store_cfg is None:
            return

        from verl.experimental.trajectory_async.relay_controller import make_relay_controller_actor

        backend = weight_store_cfg.get("backend", "kimi")
        if backend == "kimi":
            config_backend = self.config.actor_rollout_ref.rollout.checkpoint_engine.get("backend", None)
            if config_backend not in ("kimi_ckpt_engine", None):
                logger.warning(
                    "weight_store.backend=kimi but rollout.checkpoint_engine.backend=%s; "
                    "the versioned stage path requires the kimi engine",
                    config_backend,
                )
        else:
            raise NotImplementedError(
                f"async_training.weight_store.backend={backend!r}: only 'kimi' is "
                "implemented for the multi-version pull path (mooncake needs "
                "per-version RDMA staging buffers — see the package README TODO)"
            )

        keep_last = int(weight_store_cfg.get("keep_last", 2))
        max_staged_bytes = weight_store_cfg.get("max_staged_bytes", None)
        controller_cls = make_relay_controller_actor()
        self.relay_controller = controller_cls.remote(
            self.checkpoint_manager, keep_last=keep_last, max_staged_bytes=max_staged_bytes
        )

        # the rollout side drives pulls at ITS batch boundaries
        if not hasattr(self.rollouter, "set_relay_controller"):
            raise RuntimeError(
                "async_training.weight_store requires a rollouter with "
                "batch-boundary pull support (TrajectoryLevelRollouter); the "
                "stock FullyAsyncRollouter cannot drive versioned pulls — "
                "launch via trajectory_async_main.py or set weight_store=null"
            )
        ray.get(self.rollouter.set_relay_controller.remote(self.relay_controller))
        logger.info(
            "relay controller attached (backend=%s, keep_last=%d, max_staged_bytes=%s): "
            "publish is a stage-only metadata phase; pulls are rollout-driven",
            backend,
            keep_last,
            max_staged_bytes,
        )

        # repack closed loop (paper §5): a manager actor that refreshes IDLE
        # replicas lagging the fresh version right after each publish
        # (per-replica pulls) and periodically; migration-dependent
        # consolidation waits on rollout-server request-control RPCs
        repack_cfg = self.config.async_training.get("repack", None)
        if repack_cfg is not None and repack_cfg.get("enabled", True):
            from verl.experimental.trajectory_async.repack import RepackConfig
            from verl.experimental.trajectory_async.repack_bridge import make_repack_controller_actor

            field_names = RepackConfig.__dataclass_fields__
            manager_cfg = RepackConfig(
                **{k: v for k, v in dict(repack_cfg).items() if k in field_names}
            )
            server_ids = repack_cfg.get("server_ids", None)
            if server_ids is None and hasattr(self.rollouter, "replica_server_ids"):
                # convention: replica index i (engine partition order) <->
                # i-th sorted LB server id; override via repack.server_ids
                server_ids = ray.get(self.rollouter.replica_server_ids.remote())
            repack_cls = make_repack_controller_actor()
            self.repack_controller = repack_cls.remote(
                self.relay_controller,
                rollouter=self.rollouter,
                server_ids=list(server_ids) if server_ids else None,
                config=manager_cfg,
            )
            logger.info(
                "repack controller attached (check_interval_s=%.1f, replicas=%d): "
                "idle replicas refresh to fresh versions per-replica after each publish",
                manager_cfg.check_interval_s,
                len(server_ids) if server_ids else 0,
            )

    def _fit_compute_advantage(self, batch):
        """Standard advantages + the version-staleness correction layer.

        The correction applies ONLY to version-stamped batches (rows carry
        ``model_version`` from the trajectory-level producer) on the
        versioned weight path (relay controller attached): it reweights
        each trajectory by its version age, optionally widens/tightens its
        clip bounds and (experimentally) renormalizes mixed-version groups
        by version cohort — see ``staleness_correction.py`` for the
        positioning vs the PPO ratio's built-in per-token cross-version IS.
        """
        batch = super()._fit_compute_advantage(batch)

        from verl.experimental.trajectory_async.staleness_correction import (
            StalenessCorrectionConfig,
            apply_staleness_correction,
        )

        cfg = StalenessCorrectionConfig.from_config(self.config.async_training.get("staleness_correction", None))
        if cfg is None:
            return batch
        if getattr(self, "relay_controller", None) is None:
            logger.debug(
                "staleness_correction set but no relay controller attached "
                "(stock push path or unstamped producer); skipping"
            )
            return batch
        if "model_version" not in (getattr(batch, "non_tensor_batch", None) or {}):
            return batch  # group-level producer without version stamps

        # during this update the actor embodies the version that will be
        # published at the end of the step (the weight_store path publishes
        # current_param_version + 1 in _fit_update_weights)
        apply_staleness_correction(
            batch,
            current_version=self.current_param_version + 1,
            config=cfg,
            metrics_out=self.metrics,
        )
        return batch

    async def _fit_update_weights(self):
        """Weight sync with a multi-version pull-based extension point.

        Default: the stock push through ``CheckpointEngineManager``
        (inherited — actor workers ``send_weights`` their shards, replicas
        receive; versioned by ``global_steps``). Setting
        ``async_training.weight_store.backend`` selects the Laminar-style
        pull path (stage-only publish, per-replica pulls at batch
        boundaries) — see :meth:`_publish_versioned_weights`.
        """
        if self.config.async_training.get("weight_store", None) is None:
            return await super()._fit_update_weights()

        if self.local_trigger_step != 1:
            return None

        with self._marked_param_sync():
            await self._publish_versioned_weights(self.current_param_version + 1)
        return None

    def _marked_param_sync(self):
        """Timer context matching the stock ``timing_s/param_sync``."""
        from verl.utils.debug import marked_timer

        return marked_timer("timing_s/param_sync", self.timing_raw)

    async def _publish_versioned_weights(self, version: int) -> None:
        """Stage the current actor weights as pullable version ``version``.

        The real path (P0 topology): the relay controller fires
        ``stage_weights_version`` on every actor rank (CPU offload +
        version-scoped registration in the kimi P2P store, metas gather — no
        tensor push, no unregister) concurrently with ``gather_version_metas``
        on the rollout ranks (gather_metas is collective over the whole
        engine group). That metadata phase is the WHOLE trainer-side stall;
        the version stays pullable until retention retires it, and the
        rollout side pulls at its batch boundaries
        (``TrajectoryLevelRollouter._maybe_pull_weights`` →
        ``RelayControllerActor.pull``). Never blocks training compute.
        """
        controller = getattr(self, "relay_controller", None)
        if controller is None:
            raise RuntimeError(
                "weight_store configured but no relay controller was built "
                "(expected _setup_checkpoint_manager to attach one)"
            )
        snapshot = await controller.publish.remote(version)
        self.metrics.update(snapshot)
        # repack post-update trigger (paper §5.1): the best moment to move
        # idle replicas onto the fresh version — fire-and-forget; the repack
        # actor's own loop does the work
        repack = getattr(self, "repack_controller", None)
        if repack is not None:
            repack.notify_update.remote()

    async def fit(self):
        """Training loop; finalizes collector accounting on the way out."""
        try:
            return await super().fit()
        finally:
            repack = getattr(self, "repack_controller", None)
            if repack is not None:
                try:
                    ray.get(repack.stop.remote())
                except Exception:  # noqa: BLE001 — teardown must not mask results
                    logger.warning("repack controller stop failed", exc_info=True)
            leftover = self.trajectory_collector.finalize()
            if any(leftover.values()):
                logger.warning(
                    "trajectory collector at end of stream: %d leftover complete "
                    "group(s), %d incomplete group(s)",
                    leftover["leftover"],
                    leftover["incomplete"],
                )
