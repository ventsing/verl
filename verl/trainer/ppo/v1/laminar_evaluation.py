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
"""Dedicated, concurrent evaluation of immutable Laminar weight versions."""

import copy
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from threading import Event
from uuid import uuid4

import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def prepare_validation_dataset(view):
    """Select final-only files on the private version view, never the trainer."""
    files = view.config.trainer.v1.laminar.evaluation.get("final_val_files")
    if not files or view.global_steps < getattr(view, "total_training_steps", float("inf")):
        return
    from verl.trainer.ppo.utils import create_rl_dataset

    view.config = copy.deepcopy(view.config)
    view.config.data.val_files = files
    view.val_dataset = create_rl_dataset(
        view.config.data.val_files,
        view.config.data,
        view.tokenizer,
        view.processor,
        is_train=False,
        max_samples=view.config.data.get("val_max_samples", -1),
    )


def evaluation_config(config):
    """Copy rollout configuration and validate dedicated evaluation placement."""
    options = config.trainer.v1.laminar.evaluation
    result = copy.deepcopy(config)
    rollout = result.actor_rollout_ref.rollout
    rollout.nnodes = options.nnodes
    rollout.n_gpus_per_node = options.n_gpus_per_node
    if options.tensor_model_parallel_size is not None:
        rollout.tensor_model_parallel_size = options.tensor_model_parallel_size
    rollout.agent.num_workers = options.num_workers
    engine_limit = options.get("max_num_seqs")
    if engine_limit is not None:
        rollout.max_num_seqs = engine_limit
    engine_limit = rollout.get("max_num_seqs", 1024)
    inflight_limit = options.get("max_inflight_per_replica")
    inflight_limit = engine_limit if inflight_limit is None else inflight_limit
    for name, value in (("max_num_seqs", engine_limit), ("max_inflight_per_replica", inflight_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Dedicated evaluation {name} must be a positive integer")
    if inflight_limit > engine_limit:
        raise ValueError("Dedicated evaluation max_inflight_per_replica exceeds max_num_seqs")
    captures = options.get("cudagraph_capture_sizes")
    override_capacity = options.get("max_num_seqs") is not None or options.get("max_inflight_per_replica") is not None
    if captures is None and override_capacity and not rollout.get("enforce_eager", False):
        captures = list(rollout.get("cudagraph_capture_sizes") or [])
        size = 1
        while size < inflight_limit:
            captures.append(size)
            size *= 2
        captures.append(inflight_limit)
        captures = sorted(set(size for size in captures if size <= engine_limit))
    if captures is not None:
        if not captures or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in captures):
            raise ValueError("Dedicated evaluation graph capture sizes must be positive integers")
        if max(captures) > engine_limit:
            raise ValueError("Dedicated evaluation graph capture sizes exceed max_num_seqs")
        if not rollout.get("enforce_eager", False) and max(captures) < inflight_limit:
            raise ValueError("Dedicated evaluation graph gears do not cover its admission ceiling")
        rollout.cudagraph_capture_sizes = sorted(set(captures))
        compilation = OmegaConf.select(rollout, "engine_kwargs.vllm.compilation_config")
        if isinstance(compilation, dict | DictConfig) and "max_cudagraph_capture_size" in compilation:
            compilation["max_cudagraph_capture_size"] = max(captures)
    tp = rollout.tensor_model_parallel_size
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (rollout.nnodes, rollout.n_gpus_per_node, tp, options.num_workers)
    ):
        raise ValueError("Dedicated evaluation requires positive integer node, device, TP and worker counts")
    if tp > rollout.n_gpus_per_node or rollout.n_gpus_per_node % tp:
        raise ValueError("Dedicated evaluation TP must divide n_gpus_per_node and fit within one node")
    concurrency = options.get("max_concurrent_versions", 1)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("Dedicated evaluation max_concurrent_versions must be a positive integer")
    if options.get("repack_interval_s", 5.0) <= 0:
        raise ValueError("Dedicated evaluation repack_interval_s must be positive")
    if options.get("overflow_policy", "latest") not in {"latest", "skip", "block"}:
        raise ValueError("Dedicated evaluation overflow_policy must be latest, skip or block")
    if options.get("allocation_policy", "finish_first") not in {"finish_first", "fair"}:
        raise ValueError("Dedicated evaluation allocation_policy must be finish_first or fair")
    if options.timeout_s <= 0:
        raise ValueError("Dedicated evaluation timeout_s must be positive")
    if rollout.checkpoint_engine.backend != "mooncake_relay":
        raise ValueError("Dedicated evaluation requires the mooncake_relay checkpoint backend")
    # Evaluation must use the same model representation and checkpoint engine;
    # sampling still comes from val_kwargs, independently of training sampling.
    return result


class EvaluationReplayBuffer(ReplayBufferAsync):
    """Private validation metadata with bounded waits and explicit cancellation."""

    def __init__(self, config, check, partition_id="val"):
        super().__init__(
            trainer_mode="laminar",
            trainer_config=config.trainer.v1.laminar,
            max_off_policy_threshold=1,
            max_off_policy_strategy="none",
            sampler_kwargs=OmegaConf.create({}),
            poll_interval=0.1,
        )
        self.check = check
        self.partition_id = partition_id
        self.sampled_rows = 0

    def _sync_metadata_from_transfer_queue(self):
        self.check()
        super()._sync_metadata_from_transfer_queue()
        if self.failure_keys[self.partition_id]:
            raise RuntimeError(f"Dedicated evaluation rollout failed: {sorted(self.failure_keys[self.partition_id])}")

    def sample(self, global_steps, partition_id, batch_size):
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        for tag in batch.tags:
            if tag.get("min_global_steps") != global_steps or tag.get("max_global_steps") != global_steps:
                raise ValueError(f"Evaluation trajectory changed policy version; expected {global_steps}: {tag}")
        self.sampled_rows += len(batch.keys)
        return batch, metrics


class LaminarEvaluation:
    """Bounded concurrent evaluation jobs sharing an exact-version replica pool.

    Each version owns its manager, validation iterator, TQ partition, sampler
    and dump executor. Only the training thread submits jobs and logs results.
    Pins protect all scheduled versions until scoring and dumping finish.
    """

    def __init__(self, trainer, config, controller, manager, relays):
        self.trainer = trainer
        self.config = config
        self.controller = controller
        self.manager_factory = manager if callable(manager) else lambda version: manager
        self.relays = list(dict.fromkeys(relays))
        rollout = config.actor_rollout_ref.rollout
        replicas = rollout.nnodes * rollout.n_gpus_per_node // rollout.tensor_model_parallel_size
        self.concurrency = min(config.trainer.v1.laminar.evaluation.get("max_concurrent_versions", 1), replicas)
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="laminar-evaluation")
        self.futures = {}
        self.managers = {}
        self.cancelled = Event()
        self.skipped = 0
        self.deferred = 0
        self.coalesced = 0
        self.pending = None
        self.overflow_policy = config.trainer.v1.laminar.evaluation.get("overflow_policy", "latest")

    @property
    def future(self):
        """The sole outstanding result when operating with one version slot."""
        return next(iter(self.futures.values()), None)

    @property
    def busy(self):
        # Completed but uncollected results still occupy their slot.
        return len(self.futures) >= self.concurrency

    def submit(self, version):
        """Pin and schedule a version without accumulating an unbounded backlog."""
        if self.busy or version in self.futures:
            raise RuntimeError("Previous evaluation must be collected before submitting another")
        owner, pinned = self._pin(version)
        self._start(version, owner, pinned)

    def _pin(self, version):
        owner = f"evaluation:{uuid4().hex}:{version}"
        pinned = []
        try:
            for relay in self.relays:
                pinned.append(relay)
                ray.get(relay.retain.remote(version, owner))
        except BaseException:
            for relay in pinned:
                ray.get(relay.release_owner.remote(owner))
            raise
        return owner, pinned

    def defer(self, version):
        """Keep at most one pending exact snapshot, replacing older pending work."""
        if version in self.futures or (self.pending is not None and self.pending[0] == version):
            return
        if self.overflow_policy == "block":
            raise RuntimeError("Blocking evaluation must wait for a slot instead of deferring a version")
        if self.overflow_policy == "skip":
            self.skipped += 1
            return
        # Pin first: failure to retain the new snapshot must preserve the old one.
        owner, pinned = self._pin(version)
        self.discard_pending(replaced=True)
        self.pending = (version, owner, pinned)
        self.deferred += 1

    def discard_pending(self, replaced=False):
        if self.pending is None:
            return
        _, owner, pinned = self.pending
        self.pending = None
        for relay in pinned:
            ray.get(relay.release_owner.remote(owner))
        if replaced:
            self.coalesced += 1
            self.skipped += 1

    def pump_pending(self):
        if self.pending is None or self.busy:
            return False
        pending, self.pending = self.pending, None
        self._start(*pending)
        return True

    def queue_metrics(self):
        return {
            "evaluation/skipped": self.skipped,
            "evaluation/deferred": self.deferred,
            "evaluation/coalesced": self.coalesced,
            "evaluation/pending_version": self.pending[0] if self.pending else -1,
        }

    def _start(self, version, owner, pinned):
        registered = False
        try:
            ray.get(self.controller.register_version.remote(version))
            registered = True
            manager = self.manager_factory(version)
            self.managers[version] = manager
            view = copy.copy(self.trainer)
            view.config = self.config
            view.global_steps = version
            view.agent_loop_manager = manager
            self.futures[version] = self.executor.submit(self._evaluate, view, owner, pinned)
        except BaseException:
            if registered:
                ray.get(self.controller.unregister_version.remote(version))
            for relay in pinned:
                ray.get(relay.release_owner.remote(owner))
            self._close_manager(version)
            raise

    def _evaluate(self, view, owner, pinned):
        started = time.monotonic()
        deadline = started + self.config.trainer.v1.laminar.evaluation.timeout_s
        generations = {}
        partition_id = f"val-evaluation-{view.global_steps}"

        def get(ref):
            while True:
                if self.cancelled.is_set():
                    raise RuntimeError("Dedicated evaluation cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Dedicated evaluation timed out at policy version {view.global_steps}")
                try:
                    return ray.get(ref, timeout=min(1.0, remaining))
                except ray.exceptions.GetTimeoutError:
                    continue

        def check():
            monitor = getattr(self.trainer, "_supply_monitor", None)
            if monitor is not None:
                monitor.check()
            if self.cancelled.is_set():
                raise RuntimeError("Dedicated evaluation cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Dedicated evaluation timed out at policy version {view.global_steps}")
            get(self.controller.status.remote())

        dump_initialized = False
        try:
            prepare_validation_dataset(view)
            view._continuous_evaluation = True
            view.replay_buffer = EvaluationReplayBuffer(self.config, check, partition_id)
            if hasattr(view, "val_dataset"):
                from torchdata.stateful_dataloader import StatefulDataLoader

                from verl.utils.dataset.rl_dataset import collate_fn

                view.val_dataloader = StatefulDataLoader(
                    dataset=view.val_dataset,
                    batch_size=view.config.data.val_batch_size or len(view.val_dataset),
                    num_workers=0,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            view._maybe_log_val_generations = lambda **kwargs: generations.update(kwargs)
            view._init_dump_executor()
            dump_initialized = True
            check()
            metrics = PPOTrainer._validate(view, partition_id=partition_id)
            get(self.controller.unregister_version.remote(view.global_steps))
            metrics.update(
                {
                    "evaluation/policy_version": view.global_steps,
                    "evaluation/seconds": time.monotonic() - started,
                    "evaluation/samples": view.replay_buffer.sampled_rows,
                }
            )
            return view, metrics, generations
        finally:
            try:
                if dump_initialized:
                    view._shutdown_dump_executor()
            finally:
                for relay in pinned:
                    ray.get(relay.release_owner.remote(owner))

    def _close_manager(self, version):
        manager = self.managers.pop(version, None)
        if manager is not None:
            try:
                if hasattr(manager, "shutdown"):
                    manager.shutdown()
            finally:
                for worker in getattr(manager, "agent_loop_workers", []):
                    ray.kill(worker, no_restart=True)

    def collect(self, wait=False):
        """Collect independent results and log on the training thread."""
        results = []
        for version, future in list(self.futures.items()):
            if not wait and not future.done():
                continue
            self.futures.pop(version)
            try:
                view, metrics, generations = future.result()
                if generations:
                    PPOTrainer._maybe_log_val_generations(view, **generations)
                metrics.update(self.queue_metrics())
                metrics["evaluation/concurrent_versions"] = len(self.futures)
                results.append(metrics)
            finally:
                self._close_manager(version)
        # Preserve every version if several completed between two training steps.
        for metrics in results[:-1]:
            self.trainer.logger.log(data=metrics, step=self.trainer.global_steps)
        return results[-1] if results else {}

    def wait_for_slot(self):
        """Apply bounded backpressure; keep all other version jobs running."""
        if self.busy:
            wait_futures(list(self.futures.values()), return_when=FIRST_COMPLETED)
        return self.collect()

    def close(self, cancel=True):
        """Cancel or drain jobs before releasing their managers and weight pins."""
        try:
            if not cancel:
                while True:
                    metrics = self.collect(wait=True)
                    if metrics:
                        self.trainer.logger.log(data=metrics, step=self.trainer.global_steps)
                    if not self.pump_pending():
                        break
        finally:
            self.cancelled.set()
            try:
                self.discard_pending()
                for manager in self.managers.values():
                    if hasattr(manager, "shutdown"):
                        manager.shutdown()
            finally:
                self.executor.shutdown(wait=True)
                for version in list(self.managers):
                    self._close_manager(version)
