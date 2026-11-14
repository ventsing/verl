# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Laminar execution with independently loaded node-local CPU relays."""

import logging
import math
import sys
import time
from collections import deque
from pathlib import Path
from threading import RLock

import ray
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.experimental.trajectory_async.cpu_relay import CPUWeightRelay
from verl.experimental.trajectory_async.experience_cache import load_initial_groups
from verl.experimental.trajectory_async.laminar_controller import LaminarController, LaminarServerClient
from verl.experimental.trajectory_async.prompt_feeder import AdaptivePromptBuffer, PromptFeeder
from verl.experimental.trajectory_async.reward_diagnostics import RewardDiagnostics
from verl.experimental.trajectory_async.staleness_metrics import CompletionRateTracker, compute_replica_policy_metrics
from verl.experimental.trajectory_async.supply import (
    SupplyConfig,
    SupplyCoordinator,
    SupplyMonitor,
    queue_supply_snapshot,
    should_refill,
)
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import LLMServerManager

logger = logging.getLogger(__name__)


@register_trainer("laminar")
class PPOTrainerLaminar(PPOTrainerSeparateAsync):
    """Fixed trainer/rollout placement; independent pulls by default.

    The opt-in synchronous control submits one batch at a time and waits for
    every rollout replica to load the new version before the next batch.

    The behavior policy is the old policy: old_log_probs are copied from
    rollout_log_probs, never recomputed with a newer actor snapshot.
    """

    def __init__(self, config):
        self.synchronous = bool(config.trainer.v1.laminar.get("synchronous", False))
        if self.synchronous and config.trainer.v1.laminar.initial_experience_dir:
            raise ValueError("Synchronous comparison requires fresh, on-policy experiences")
        rollout = config.actor_rollout_ref.rollout
        if rollout.name != "vllm" or rollout.checkpoint_engine.backend not in {"cpu_relay", "mooncake_relay"}:
            raise ValueError("Laminar requires vllm with cpu_relay or mooncake_relay")
        if rollout.checkpoint_engine.backend == "cpu_relay" and (rollout.nnodes != 1 or config.trainer.nnodes != 1):
            raise ValueError("The shared-memory CPU relay currently supports a single node")
        if rollout.disaggregation.enabled or rollout.data_parallel_size != 1:
            raise ValueError("Laminar currently requires engine DP=1 and no PD disaggregation")
        if rollout.pipeline_model_parallel_size != 1:
            raise ValueError("Laminar currently requires rollout PP=1")
        if rollout.enable_rollout_routing_replay:
            raise ValueError("Laminar recompute migration currently does not support MoE routing replay")
        if config.trainer.v1.separate_async.hybrid_rollout.enable_switch:
            raise ValueError("Laminar uses fixed trainer/rollout placement; disable hybrid switching")
        if not rollout.calculate_log_probs or not config.algorithm.rollout_correction.bypass_mode:
            raise ValueError("Laminar requires vLLM behavior logprobs and bypass_mode=True")
        if rollout.top_p != 1.0 or rollout.top_k != -1 or rollout.get("repetition_penalty", 1.0) != 1.0:
            raise ValueError("Behavior/actor probability alignment requires top_p=1, top_k=-1, repetition_penalty=1")
        if rollout.logprobs_mode != "processed_logprobs":
            raise ValueError("Laminar requires processed_logprobs to include the sampling temperature")
        if config.trainer.v1.laminar.parameter_sync_step != 1:
            raise ValueError("Laminar currently publishes after each single training update")
        if config.data.dataloader_num_workers != 0:
            raise ValueError("Laminar prompt production requires data.dataloader_num_workers=0")
        if config.skip.rollout_tq.enable:
            raise ValueError("Laminar requires live rollout; rollout_tq skip mode is unsupported")
        self._length_calibration_enabled = config.trainer.v1.laminar.online_capacity.get(
            "response_step_calibration", True
        )
        self.supply_config = SupplyConfig(**dict(config.trainer.v1.laminar.get("supply", {})))
        if self.supply_config.enabled and self.synchronous:
            raise ValueError("Just-in-time supply requires asynchronous training")
        self.adaptive_buffer_options = dict(config.trainer.v1.laminar.get("adaptive_buffer", {}))
        if self.adaptive_buffer_options.get("enabled", False) and (self.synchronous or self.supply_config.enabled):
            raise ValueError("Adaptive buffer requires asynchronous training without the separate supply controller")
        self._adaptive_buffer = None
        self._adaptive_buffer_monitor = None
        self._buffer_staleness = 0.0
        self._supply_monitor = None
        self._supply_feedback = {}
        self._supply_runtime = {}
        self._supply_staleness = deque(maxlen=self.supply_config.staleness_window)
        self._sample_started_at = None
        self._supply_training_finished = False
        self.evaluation = None
        self.evaluation_config = None
        if config.trainer.v1.laminar.get("evaluation", {}).get("enabled", False):
            from verl.trainer.ppo.v1.laminar_evaluation import evaluation_config

            self.evaluation_config = evaluation_config(config)
            if (
                self.supply_config.enabled
                and self.supply_config.elastic_evaluation
                and rollout.tensor_model_parallel_size
                != self.evaluation_config.actor_rollout_ref.rollout.tensor_model_parallel_size
            ):
                raise ValueError("Elastic evaluation requires matching training/evaluation TP sizes")
        super().__init__(config)
        self._prompt_lock = RLock()
        self._prompt_feeder = None
        self.reward_diagnostics = RewardDiagnostics()
        self._completion_rate_metrics = CompletionRateTracker()
        self.replay_buffer.health_check = self._check_prompt_feeder

    def add_replicas_to_balancer(self):
        # Fixed trainer roles are never added to the inference pool.
        pass

    def _create_hybrid_server_manager(self, actor_rollout_resource_pool):
        return LLMServerManager.empty(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

    def _setup(self):
        super()._setup()
        self.current_mode = HybridEngineMode.TRAINER
        options = self.config.trainer.v1.laminar
        budget = OmegaConf.to_container(options.weight_budget, resolve=True)
        world_size = self.actor_rollout_wg.world_size
        replicas = [
            {"server": replica._server_handle, "workers": replica.workers}
            for replica in self.standalone_server_manager.get_replicas()
        ]
        training_replica_count = len(replicas)
        evaluation_replicas = []
        if self.evaluation_config is not None:
            self.evaluation_server_manager = LLMServerManager.create(
                config=self.evaluation_config, start_rank=training_replica_count
            )
            evaluation_replicas = [
                {"server": replica._server_handle, "workers": replica.workers}
                for replica in self.evaluation_server_manager.get_replicas()
            ]
            # Evaluation has its own devices and controller, but participates in
            # the same Mooncake publication tree (including node-local relays).
            replicas.extend(evaluation_replicas)
        self.relay_nodes = []
        backend = self.config.actor_rollout_ref.rollout.checkpoint_engine.backend
        if backend == "mooncake_relay":
            from verl.experimental.trajectory_async.mooncake_relay import create_relay_hierarchy

            actor_nodes = ray.get(self.actor_rollout_wg.execute_checkpoint_engine(method=["node_info"] * world_size))
            replica_nodes = []
            for replica in replicas:
                locations = ray.get(
                    [worker.execute_checkpoint_engine.remote("node_info") for worker in replica["workers"]]
                )
                if not locations or len({node["node_id"] for node in locations}) != 1:
                    raise ValueError("Mooncake Laminar currently requires each TP replica to fit within one node")
                replica_nodes.append(locations[0])
            relay_options = OmegaConf.to_container(options.mooncake, resolve=True)
            relay_options["keep_last"] = options.keep_last_versions
            relay_options.update(budget)
            relay_recovery = relay_options.pop("recovery")
            if relay_recovery.pop("enabled"):
                from verl.experimental.trajectory_async.relay_recovery import create_resilient_relay_hierarchy

                recovery_dir = relay_recovery.pop("directory") or str(
                    Path(options.event_dir or self.config.trainer.default_local_dir) / "relay_recovery"
                )
                self.relay_nodes, local_relays = create_resilient_relay_hierarchy(
                    actor_nodes[0],
                    replica_nodes,
                    relay_options,
                    relay_recovery,
                    recovery_dir,
                    identity={
                        "model": self.config.actor_rollout_ref.model.path,
                        "run_dir": self.config.trainer.default_local_dir,
                    },
                )
            else:
                self.relay_nodes, local_relays = create_relay_hierarchy(actor_nodes[0], replica_nodes, relay_options)
            self.weight_relay = self.relay_nodes[0]
            for replica, local_relay in zip(replicas, local_relays, strict=True):
                replica["relay"] = local_relay
            ray.get(
                self.actor_rollout_wg.execute_checkpoint_engine(
                    method=["configure"] * world_size,
                    relay=[self.weight_relay] * world_size,
                    root_node_id=[actor_nodes[0]["node_id"]] * world_size,
                )
            )
        else:
            self.weight_relay = (
                ray.remote(CPUWeightRelay).options(num_cpus=1).remote(keep_last=options.keep_last_versions, **budget)
            )
            ray.get(
                self.actor_rollout_wg.execute_checkpoint_engine(
                    method=["configure"] * world_size,
                    relay=[self.weight_relay] * world_size,
                )
            )
        replicas = replicas[:training_replica_count]
        elastic = self.supply_config.enabled and self.supply_config.elastic_evaluation and bool(evaluation_replicas)
        if elastic:
            evaluation_replicas = evaluation_replicas + [{**replica, "owned": False} for replica in replicas]
        if evaluation_replicas:
            from verl.experimental.trajectory_async.evaluation_controller import EvaluationController

            self.evaluation_controller = (
                ray.remote(EvaluationController)
                .options(num_cpus=1, max_concurrency=4096)
                .remote(
                    replicas=evaluation_replicas,
                    relay=self.weight_relay,
                    batch_quota=(
                        options.evaluation.get("max_inflight_per_replica")
                        or self.evaluation_config.actor_rollout_ref.rollout.max_num_seqs
                    ),
                    max_running=(
                        options.evaluation.get("max_inflight_per_replica")
                        or self.evaluation_config.actor_rollout_ref.rollout.max_num_seqs
                    ),
                    allocation_policy=options.evaluation.get("allocation_policy", "finish_first"),
                    enable_repack=options.evaluation.enable_repack,
                    check_interval_s=options.evaluation.repack_interval_s,
                    event_path=str(
                        Path(options.event_dir or self.config.trainer.default_local_dir) / "evaluation_scheduler.jsonl"
                    ),
                )
            )
        recovery = options.partial_store
        self.partial_store = None
        if recovery.enabled:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            from verl.experimental.trajectory_async.partial_store import PartialTrajectoryStore

            persistent = recovery.get("persist_to_disk", False)
            database = (
                (
                    recovery.path
                    or str(
                        Path(options.event_dir or self.config.trainer.default_local_dir) / "partial_trajectories.sqlite"
                    )
                )
                if persistent
                else ":memory:"
            )
            self.partial_store = (
                ray.remote(PartialTrajectoryStore)
                .options(
                    num_cpus=1,
                    max_restarts=3 if persistent else 0,
                    max_task_retries=3 if persistent else 0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        ray.get_runtime_context().get_node_id(), soft=False
                    ),
                )
                .remote(
                    database,
                    recovery.max_active,
                    identity={
                        "model": self.config.actor_rollout_ref.model.path,
                        "run_dir": self.config.trainer.default_local_dir,
                    },
                )
            )
            # A volatile store must fail-stop on actor loss, never restart empty.
            ray.get(self.partial_store.status.remote())
        roofline = OmegaConf.to_container(options.roofline, resolve=True)
        configurations = []
        if roofline["enabled"]:
            configurations = ray.get([r["server"].configure_trajectory_roofline.remote(roofline) for r in replicas])
            logger.debug("Laminar online roofline probes: %s", configurations)
        event_path = str(Path(options.event_dir) / "scheduler.jsonl") if options.event_dir else None
        self.laminar_controller = (
            ray.remote(LaminarController)
            .options(num_cpus=1, max_concurrency=4096)
            .remote(
                replicas=replicas,
                relay=self.weight_relay,
                batch_quota=options.batch_per_replica,
                max_running=self.config.actor_rollout_ref.rollout.max_num_seqs,
                repack_max_running=options.repack_max_running,
                roofline_options=roofline,
                capacity_options=OmegaConf.to_container(options.online_capacity, resolve=True),
                capacity_profiles=[c.get("capacity") for c in configurations],
                check_interval_s=options.repack_interval_s,
                enable_repack=options.enable_repack,
                event_path=event_path,
                partial_store=self.partial_store,
                recovery_options=OmegaConf.to_container(recovery, resolve=True),
            )
        )

        if self.supply_config.enabled:
            self.supply_coordinator = (
                ray.remote(SupplyCoordinator)
                .options(num_cpus=1, max_concurrency=16)
                .remote(
                    self.laminar_controller,
                    getattr(self, "evaluation_controller", None) if elastic else None,
                    training_replica_count,
                    len(evaluation_replicas) - training_replica_count if elastic else 0,
                    OmegaConf.to_container(options.supply, resolve=True),
                    event_path=str(
                        Path(options.event_dir or self.config.trainer.default_local_dir) / "supply_scheduler.jsonl"
                    ),
                )
            )

    def get_llm_client(self):
        return LaminarServerClient(self.config, controller=self.laminar_controller)

    def _publish(self):
        results = ray.get(
            self.actor_rollout_wg.update_weights(
                global_steps=self.global_steps, mode=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend
            )
        )
        metrics = {}
        for result in results:
            if isinstance(result, dict):
                metrics.update(result)
        return metrics

    def on_init_end(self):
        self._publish()
        ray.get(self.laminar_controller.start.remote(self.global_steps))
        if self.config.trainer.v1.laminar.run_migration_probe:
            tokens = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Explain step by step why there are infinitely many prime numbers."}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            report = ray.get(self.laminar_controller.migration_probe.remote(tokens))
            logger.debug("Laminar real-backend migration preflight passed: %s", report)

    def on_step_begin(self):
        self._check_prompt_feeder()
        ray.get(self.laminar_controller.status.remote())
        if getattr(self, "evaluation", None) is not None:
            metrics = self.evaluation.collect()
            if metrics:
                self.logger.log(data=metrics, step=self.global_steps)
            self.evaluation.pump_pending()
        # Preserve the inherited sampling timers. In fixed TRAINER mode this
        # resets accounting and returns without touching rollout replicas.
        super().on_step_begin()

    def on_step_end(self):
        self._check_prompt_feeder()
        if getattr(self, "synchronous", False):
            self._assert_synchronous_idle(self.global_steps - 1)
        if self.global_steps >= self.total_training_steps:
            self._supply_training_finished = True
            self._stop_prompt_feeder()
        with marked_timer("publish_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = self._publish()
        ray.get(self.laminar_controller.notify_weights.remote(self.global_steps))
        if getattr(self, "synchronous", False):
            with marked_timer("sync_rollout_barrier", self.timing_raw, color="red"):
                self._wait_synchronous_version(self.global_steps)
        status = ray.get(self.laminar_controller.status.remote())
        for name in (
            "completed",
            "migrations",
            "pulls",
            "repack_checks",
            "pending",
            "replayed_experiences",
            "recoveries",
            "quarantined_replicas",
        ):
            self._pending_sync_metrics[f"laminar/{name}"] = status[name]
        self._pending_sync_metrics["laminar/drain_moves"] = status.get("drain_moves", 0)
        self._pending_sync_metrics.update(compute_replica_policy_metrics(status))
        self._pending_sync_metrics.update(self._completion_rate_metrics.update(status["completed"], time.monotonic()))
        self._pending_sync_metrics["laminar/queue/prompt_groups"] = self._pending_prompt_count()
        self._pending_sync_metrics["laminar/queue/prompt_capacity"] = (
            self.config.trainer.v1.laminar.prompt_buffer_groups
        )
        if getattr(self, "evaluation", None) is not None:
            evaluation_metrics = self.evaluation.collect()
            if evaluation_metrics:
                # Validation below may return a newer version in the same step.
                self.logger.log(data=evaluation_metrics, step=self.global_steps)
            self._pending_sync_metrics.update(self.evaluation.queue_metrics())
            evaluation_status = ray.get(self.evaluation_controller.status.remote())
            self._pending_sync_metrics.update(
                {
                    "evaluation/pool/active_versions": len(evaluation_status["evaluation_requests"]),
                    "evaluation/pool/active_requests": sum(r["active"] for r in evaluation_status["replicas"]),
                    "evaluation/pool/busy_replicas": sum(
                        bool(r["active"] or r["reserved"]) for r in evaluation_status["replicas"]
                    ),
                    "evaluation/pool/pulls": evaluation_status["pulls"],
                    "evaluation/pool/migrations": evaluation_status["migrations"],
                    "evaluation/pool/drain_moves": evaluation_status.get("drain_moves", 0),
                }
            )

    def on_sample_begin(self):
        self._sample_started_at = time.monotonic()
        super().on_sample_begin()

    def on_sample_end(self):
        super().on_sample_end()
        if getattr(self, "_length_calibration_enabled", False):
            tags = self.replay_buffer.last_selected_tags.get("train", [])
            records = []
            for tag in tags:
                oldest = tag.get("min_global_steps")
                oldest = tag["global_steps"] if oldest is None else oldest
                newest = tag.get("max_global_steps")
                newest = oldest if newest is None else newest
                length = int(tag.get("response_len", 0))
                if not tag.get("is_padding", False) and oldest == newest and length > 0:
                    records.append((int(oldest), length))
            if records:
                # This ordered step RPC follows the controller's normal lifecycle;
                # a busy expanded rollout pool must not turn a short deadline into a fatal error.
                ray.get(self.laminar_controller.calibrate_response_lengths.remote(self.global_steps, records))
        self._sample_started_at = None

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        buffer = getattr(self, "_adaptive_buffer", None)
        if buffer is not None:
            self._buffer_staleness = float(metrics["training/off_policy/trajectory_staleness/mean"])
            metrics.update(
                {"laminar/buffer/capacity_groups": buffer.current, "laminar/buffer/adjustments": buffer.adjustments}
            )
        if getattr(self, "supply_config", None) is not None and self.supply_config.enabled:
            self._supply_staleness.append(float(metrics["training/off_policy/trajectory_staleness/mean"]))
            self._supply_feedback = {"window_staleness": sum(self._supply_staleness) / len(self._supply_staleness)}
            status = ray.get(self.supply_coordinator.status.remote(), timeout=10)
            metrics.update(
                {
                    "laminar/supply/paused": int(status.get("paused", False)),
                    "laminar/supply/borrowed_replicas": len(status["borrowed"]),
                    "laminar/supply/transitions": len(status["transitions"]),
                    "laminar/supply/transfers": status["transfers"],
                    "laminar/supply/window_staleness": self._supply_feedback["window_staleness"],
                    "laminar/supply/ready_groups": status["feedback"].get("ready_groups", 0),
                    "laminar/supply/ready_batches": status["feedback"].get("ready_batches", 0),
                    "laminar/supply/pending_groups": status["feedback"].get("pending_groups", 0),
                    "laminar/supply/unfinished_trajectories": status["feedback"].get("unfinished_trajectories", 0),
                    "laminar/supply/admission_total": status["feedback"].get("admission_total", 0),
                    "laminar/supply/active_total": status["feedback"].get("active_total", 0),
                    "laminar/supply/idle_replicas": status["feedback"].get("idle_replicas", 0),
                    "laminar/supply/feedback_fresh": int(status.get("feedback_fresh", False)),
                }
            )

    def _poll_supply(self):
        partitions = tq.kv_list() or {}
        feedback = queue_supply_snapshot(
            partitions.get("train", {}), self.config.actor_rollout_ref.rollout.n, self.config.data.train_batch_size
        )
        feedback.update(self._supply_feedback)
        started = self._sample_started_at
        feedback["sample_wait_s"] = max(0.0, time.monotonic() - started) if started is not None else 0.0
        feedback["training_finished"] = self._supply_training_finished
        training_status = ray.get(self.laminar_controller.status.remote(), timeout=10)
        owned = [
            r
            for r in training_status["replicas"]
            if r.get("owned", True) and r["healthy"] and not r.get("releasing", False)
        ]
        self._supply_runtime = {
            "observed_at": time.monotonic(),
            "admission_total": sum(r["admission_limit"] for r in owned),
            "active_total": sum(r["active"] for r in owned),
            "idle_replicas": sum(not r["active"] and not r["pulling"] for r in owned),
        }
        feedback.update(self._supply_runtime)
        ray.get(self.supply_coordinator.update.remote(feedback), timeout=10)
        ray.get(self.supply_coordinator.status.remote(), timeout=10)

    def _poll_adaptive_buffer(self):
        # This optional tuning loop must not fail training because a busy
        # controller takes longer than one poll to answer. Keep only one RPC
        # outstanding, and never make adjustments from a late response.
        now = time.monotonic()
        ref = getattr(self, "_adaptive_buffer_status_ref", None)
        if ref is None:
            ref = self.laminar_controller.status.remote()
            self._adaptive_buffer_status_ref = ref
            self._adaptive_buffer_status_requested_at = now
            self._adaptive_buffer_status_warned = False
        ready, _ = ray.wait([ref], timeout=0)
        if not ready:
            if (
                now - self._adaptive_buffer_status_requested_at > self._adaptive_buffer.feedback_timeout
                and not self._adaptive_buffer_status_warned
            ):
                logger.warning(
                    "Adaptive buffer status delayed; keeping capacity=%d until fresh feedback arrives",
                    self._adaptive_buffer.current,
                )
                self._adaptive_buffer_status_warned = True
            return
        self._adaptive_buffer_status_ref = None
        status = ray.get(ref)  # Surface real controller errors; the result is ready.
        observed_at = self._adaptive_buffer_status_requested_at
        if now - observed_at > self._adaptive_buffer.feedback_timeout:
            return
        queue = queue_supply_snapshot(
            (tq.kv_list() or {}).get("train", {}),
            self.config.actor_rollout_ref.rollout.n,
            self.config.data.train_batch_size,
        )
        owned = [
            r for r in status["replicas"] if r.get("owned", True) and r["healthy"] and not r.get("releasing", False)
        ]
        now = time.monotonic()
        runtime = {
            "observed_at": observed_at,
            "admission_total": sum(r["admission_limit"] for r in owned),
            "active_total": sum(r["active"] for r in owned),
        }
        wait = max(0.0, now - self._sample_started_at) if self._sample_started_at is not None else 0.0
        with self._prompt_lock:
            previous = self._adaptive_buffer.current
            capacity = self._adaptive_buffer.observe(queue, runtime, self._buffer_staleness, wait, now)
            self._prompt_feeder.capacity = capacity
            if capacity != previous:
                logger.info(
                    "Adaptive prompt buffer: %d -> %d groups; reason=%s ready=%d unfinished=%d active=%d",
                    previous,
                    capacity,
                    self._adaptive_buffer.reason,
                    queue["ready_groups"],
                    queue["unfinished_trajectories"],
                    runtime["active_total"],
                )

    def _can_refill_supply(self):
        queue = queue_supply_snapshot(
            (tq.kv_list() or {}).get("train", {}),
            self.config.actor_rollout_ref.rollout.n,
            self.config.data.train_batch_size,
        )
        return should_refill(
            self.supply_config, queue, self._supply_runtime, self.config.data.train_batch_size, time.monotonic()
        )

    def _stop_supply(self):
        try:
            if getattr(self, "_supply_monitor", None) is not None:
                self._supply_monitor.stop()
        finally:
            if hasattr(self, "supply_coordinator"):
                try:
                    ray.get(self.supply_coordinator.stop.remote(), timeout=30)
                finally:
                    ray.kill(self.supply_coordinator, no_restart=True)

    def _pending_prompt_count(self):
        partitions = tq.kv_list() or {}
        return sum(bool(tag.get("is_prompt")) for tag in partitions.get("train", {}).values())

    def _check_prompt_feeder(self):
        monitor = getattr(self, "_supply_monitor", None)
        if monitor is not None:
            monitor.check()
        adaptive = getattr(self, "_adaptive_buffer_monitor", None)
        if adaptive is not None:
            adaptive.check()
        feeder = getattr(self, "_prompt_feeder", None)
        if feeder is not None:
            feeder.check()

    def _stop_prompt_feeder(self):
        adaptive = getattr(self, "_adaptive_buffer_monitor", None)
        if adaptive is not None:
            adaptive.stop()
        if self._prompt_feeder is not None:
            self._prompt_feeder.stop()

    def on_train_begin(self):
        self._laminar_training_started = True
        if getattr(self, "synchronous", False):
            # No warmup or producer thread: prepare_step submits exactly one batch.
            self._wait_synchronous_version(self.global_steps - 1)
            return
        cache = self.config.trainer.v1.laminar.initial_experience_dir
        if cache:
            if self.global_steps != 1 or self.config.trainer.resume_mode != "disable":
                raise ValueError("Initial experience replay requires a fresh, non-resumed actor")
            groups = load_initial_groups(
                cache,
                self.config.actor_rollout_ref.model.path,
                self.config.actor_rollout_ref.rollout.n,
                self.config.data.train_batch_size,
            )
            records = [
                {
                    "request": row["tag"]["trajectory_request_id"],
                    "version": 0,
                    "tokens": row["tag"]["response_len"],
                    "migrations": row["tag"]["migrations"],
                    "source_output": row["source_output"],
                    "source_job": row["source_job"],
                }
                for group in groups
                for row in group
            ]
            ray.get(self.laminar_controller.record_initial_experiences.remote(records))
            for group in groups:
                for row in group:
                    tq.kv_batch_put(
                        keys=[row["key"]],
                        partition_id="train",
                        fields=row["data"],
                        tags=[{**row["tag"], "global_steps": 0}],
                    )
                uid = group[0]["key"].rsplit("_", 2)[0]
                tq.kv_put(
                    key=uid, partition_id="train", tag={"is_prompt": True, "status": "finished", "global_steps": 0}
                )
            print(
                f"Replayed {len(records)} initial-policy experiences; longest groups selected for memory validation",
                flush=True,
            )
        if not self.supply_config.enabled:
            super().on_train_begin()
        # With demand-based supply the feeder also owns bounded warmup.
        capacity = self.config.trainer.v1.laminar.prompt_buffer_groups
        batch_size = self.config.data.train_batch_size
        if self.supply_config.enabled:
            capacity = min(capacity, math.floor(self.supply_config.max_pending_batches * batch_size))
            ray.get(self.supply_coordinator.start.remote())
            self._supply_monitor = SupplyMonitor(self._poll_supply, self.supply_config.poll_interval_s)
            self._supply_monitor.start()
        if capacity < batch_size:
            raise ValueError("Laminar prompt_buffer_groups must hold at least one training batch")
        refill = (self.supply_config.refill_groups or batch_size) if self.supply_config.enabled else batch_size
        adaptive_options = getattr(self, "adaptive_buffer_options", {})
        if adaptive_options.get("enabled", False):
            self._adaptive_buffer = AdaptivePromptBuffer(batch_size, capacity, adaptive_options)
            capacity = self._adaptive_buffer.current
            refill = self._adaptive_buffer.refill_groups
        granularity = self.config.data.get("gen_batch_size", None) or batch_size
        if refill % granularity:
            raise ValueError("Supply refill_groups must be a multiple of data.gen_batch_size")
        self._prompt_feeder = PromptFeeder(
            count_pending=self._pending_prompt_count,
            submit_batch=(lambda: self._add_prompts_to_generate(refill))
            if refill != batch_size
            else self._add_batch_to_generate,
            batch_size=refill,
            capacity=capacity,
            lock=self._prompt_lock,
            can_submit=self._can_refill_supply if self.supply_config.enabled else None,
        )
        self._prompt_feeder.start()
        if adaptive_options.get("enabled", False):
            interval = float(adaptive_options.get("poll_interval_s", 5.0))
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError("Adaptive buffer poll interval must be finite and positive")
            self._adaptive_buffer_monitor = SupplyMonitor(self._poll_adaptive_buffer, interval)
            self._adaptive_buffer_monitor.start()

    def prepare_step(self):
        if getattr(self, "synchronous", False):
            self._assert_synchronous_idle(self.global_steps - 1)
            PPOTrainer._add_batch_to_generate(self)
            return {"laminar/synchronous": 1.0}
        # The producer refills independently while the actor is busy. It also
        # counts completed, unsampled groups, providing experience backpressure.
        self._check_prompt_feeder()
        return {}

    def _assert_synchronous_idle(self, version):
        status = ray.get(self.laminar_controller.status.remote())
        replicas = status["replicas"]
        if status["pending"] or self._pending_prompt_count() or not replicas:
            raise RuntimeError(f"Synchronous batch boundary has outstanding prompts: {status}")
        if any(
            not r["healthy"] or r["active"] or r["reserved"] or r["pulling"] or r["version"] != version
            for r in replicas
        ):
            raise RuntimeError(f"Synchronous batch boundary is not idle at version {version}: {status}")

    def _wait_synchronous_version(self, version):
        timeout = float(self.config.trainer.v1.laminar.weight_budget.capacity_timeout_s)
        deadline = time.monotonic() + timeout
        while True:
            status = ray.get(self.laminar_controller.status.remote())
            replicas = status["replicas"]
            if any(not r["healthy"] for r in replicas):
                raise RuntimeError(f"Synchronous weight barrier has an unhealthy replica: {status}")
            if replicas and all(r["version"] == version and not r["pulling"] for r in replicas):
                self._assert_synchronous_idle(version)
                logger.debug("Synchronous rollout barrier passed: version=%d replicas=%d", version, len(replicas))
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Synchronous weight barrier timed out for version {version}: {status}")
            time.sleep(0.1)

    def _add_batch_to_generate(self):
        with self._prompt_lock:
            return super()._add_batch_to_generate()

    def _add_prompts_to_generate(self, num_prompts):
        if getattr(self, "synchronous", False):
            raise RuntimeError("Synchronous comparison cannot refill or prefetch extra prompt groups")
        with self._prompt_lock:
            room = self.config.trainer.v1.laminar.prompt_buffer_groups - self._pending_prompt_count()
            # The producer may already have replaced a just-evicted group.
            amount = min(num_prompts, max(0, room))
            granularity = self.config.data.get("gen_batch_size", None) or self.config.data.train_batch_size
            amount = amount // granularity * granularity
            return super()._add_prompts_to_generate(amount) if amount else 0

    def _save_checkpoint(self):
        with self._prompt_lock:
            return super()._save_checkpoint()

    def _compute_old_log_prob(self, batch, metrics):
        for tag in batch.tags:
            if tag.get("is_padding", False):
                continue
            minimum, maximum = tag.get("min_global_steps"), tag.get("max_global_steps")
            if minimum is None or minimum != maximum:
                raise ValueError(f"Laminar requires one behavior-policy version per trajectory: {tag}")
            if getattr(self, "synchronous", False) and minimum != self.global_steps - 1:
                raise ValueError(f"Synchronous batch requires behavior version {self.global_steps - 1}: {tag}")
        metrics["laminar/old_log_probs_from_rollout"] = 1.0
        return PPOTrainer._compute_old_log_prob(self, batch, metrics)

    def _update_reward_diagnostics(self, keys, data, non_padding_mask):
        scores = data["rm_scores"].sum(dim=-1).tolist()
        lengths = data["response_length"].tolist()
        records = [
            (key, score, length)
            for key, score, length, keep in zip(keys, scores, lengths, non_padding_mask, strict=True)
            if keep
        ]
        return self.reward_diagnostics.update(records, self.config.actor_rollout_ref.rollout.response_length)

    def on_validate_begin(self):
        if getattr(self, "evaluation_config", None) is not None:
            return
        # New validation requests wait for a freshly updated replica; old
        # training trajectories continue on the remaining replicas.
        ray.get(self.laminar_controller.set_minimum_version.remote(self.global_steps))

    def on_validate_end(self):
        if getattr(self, "evaluation_config", None) is not None:
            return
        ray.get(self.laminar_controller.set_minimum_version.remote(-1))

    def _validate(self):
        if self.evaluation_config is None:
            return super()._validate()
        if self.evaluation is None:
            from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
            from verl.trainer.ppo.v1.laminar_evaluation import LaminarEvaluation
            from verl.utils.import_utils import load_class_from_fqn

            manager_fqn = self.evaluation_config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
            manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ

            def manager(version):
                return manager_cls.create(
                    config=self.evaluation_config,
                    llm_client=LaminarServerClient(
                        self.evaluation_config, controller=self.evaluation_controller, policy_version=version
                    ),
                    reward_loop_worker_handles=self.get_reward_handles(),
                )

            self.evaluation = LaminarEvaluation(
                self,
                self.evaluation_config,
                self.evaluation_controller,
                manager,
                # Pin every relay: hierarchy recovery/GC can run independently
                # on each node while training publishes subsequent versions.
                self.relay_nodes,
            )
        # Initial/final validation are barriers; intermediate versions share
        # the evaluation pool without accumulating an unbounded backlog.
        wait = not getattr(self, "_laminar_training_started", False)
        wait = wait or self.global_steps >= self.total_training_steps
        wait = wait or self.config.trainer.get("val_only", False)
        if wait:
            self.evaluation.discard_pending(replaced=True)
        metrics = self.evaluation.collect(wait=wait)
        if wait and metrics:
            # The final barrier drains older versions before submitting the
            # final version; keep their scores before collecting that result.
            self.logger.log(data=metrics, step=self.global_steps)
            metrics = {}
        if self.evaluation.busy:
            if self.evaluation_config.trainer.v1.laminar.evaluation.get("overflow_policy", "latest") == "block":
                metrics.update(self.evaluation.wait_for_slot())
            else:
                self.evaluation.defer(self.global_steps)
                return {**metrics, **self.evaluation.queue_metrics()}
        self.evaluation.discard_pending(replaced=True)
        self.evaluation.submit(self.global_steps)
        if wait:
            metrics.update(self.evaluation.collect(wait=True))
        return {**metrics, **self.evaluation.queue_metrics()}

    def _close_relay_hierarchy(self):
        # Upstream first: wait native writes before releasing downstream memory.
        relays = getattr(self, "relay_nodes", [])
        resources = []
        for relay in relays:
            resources.extend(getattr(relay, "resources", [relay]))
        try:
            for relay in relays:
                ray.get(relay.close.remote())
        finally:
            # These handles come only from this trainer's relay factory. Stop
            # automatic restarts and break coordinator/endpoint reference cycles.
            for actor in dict.fromkeys(resources):
                ray.kill(actor, no_restart=True)

    def _close_evaluation_resources(self):
        """Release only the actors and placement groups owned by evaluation."""
        actors = [self.evaluation_controller]
        manager = self.evaluation_server_manager
        actors.append(manager.global_load_balancer)
        pools = []
        for replica in manager.get_replicas():
            actors.extend(replica.workers)
            actors.extend(replica.servers)
            pools.extend(replica.resource_pool.pgs or [])
        for actor in dict.fromkeys(actors):
            ray.kill(actor, no_restart=True)
        for pool in pools:
            ray.util.remove_placement_group(pool)

    def fit(self, agent_loop_manager):
        try:
            return super().fit(agent_loop_manager)
        finally:
            original_error = sys.exc_info()[0] is not None
            errors = []
            cleanups = [self._stop_prompt_feeder, self._stop_supply]
            if getattr(self, "evaluation", None) is not None:
                cleanups.append(lambda: self.evaluation.close(cancel=original_error))
            if hasattr(self, "evaluation_controller"):
                cleanups.append(lambda: ray.get(self.evaluation_controller.stop.remote()))
                cleanups.append(self._close_evaluation_resources)
            if hasattr(agent_loop_manager, "shutdown"):
                cleanups.append(agent_loop_manager.shutdown)
            cleanups.append(lambda: ray.get(self.laminar_controller.stop.remote()))
            cleanups.append(self._close_relay_hierarchy)
            for cleanup in cleanups:
                try:
                    cleanup()
                except Exception as error:
                    errors.append(error)
                    logger.exception("Laminar cleanup failed")
            if errors and not original_error:
                raise errors[0]
