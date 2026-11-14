# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in vLLM worker hooks for sampled, nonblocking decode timing.

All hooks live in verl; installed vLLM/Ascend sources are left untouched. Timing
uses device events around worker execute + sampling, polled later with query().
It never inserts synchronize(), CPU copies of device tensors, or TP collectives.
"""

import time
from collections import Counter, deque

from verl.experimental.trajectory_async.capacity import engine_capacity
from verl.experimental.trajectory_async.roofline import RooflineConfig


class DecodeStepProbe:
    def __init__(self, worker, options, device_api=None, context_reader=None, counters=None):
        self.worker = worker
        self.config = RooflineConfig.from_dict(options)
        self.enabled = self.config.enabled
        self.suspended = False
        self.version = -1
        self.versions = set()
        self.step = 0
        self.samples = deque()
        self.pending = deque()
        self.pool = []
        self.current = None
        self.shapes = Counter()
        self.bins = Counter()
        self.coverage_attempts = Counter()
        self.counts = Counter()
        self.error = None
        self.last_context = 0
        self.last_mode = "unknown"
        self.capacity = engine_capacity(worker)
        if device_api is None:
            import torch

            device_type = getattr(worker.device, "type", str(worker.device).split(":")[0])
            if device_type not in {"cuda", "npu"}:
                raise ValueError("Online roofline requires a CUDA/NPU device-event clock")
            device_api = getattr(torch, device_type)
        self.device = device_api
        if context_reader is None:
            from vllm.forward_context import get_forward_context

            context_reader = get_forward_context
        self.context_reader = context_reader
        if counters is None:
            from vllm.compilation.counter import compilation_counter

            counters = compilation_counter
        self.counters = counters

    def _compilations(self):
        return tuple(getattr(self.counters, key, 0) for key in ("num_cudagraph_captured", "num_backend_compilations"))

    def set_version(self, version):
        if type(version) is not int or version < 0:
            raise ValueError("Roofline requires a nonnegative policy version")
        self.version = version
        self.versions.add(version)
        if not self.config.continuous_sampling and len(self.versions) > self.config.calibration_rounds:
            self.enabled = False

    def fail(self, error):
        self.error = repr(error)
        self.enabled = False
        self.current = None

    def poll(self):
        while self.pending and self.pending[0][1].query():
            start, end, sample = self.pending.popleft()
            elapsed = float(start.elapsed_time(end))
            self.pool.append((start, end))
            if elapsed > 0:
                sample["elapsed_ms"] = elapsed
                self.samples.append(sample)
                self.counts["sampled"] += 1

    def begin(self, scheduled):
        self.step += 1
        self.current = None
        if self.step % self.config.sample_every == 0:
            self.poll()
        if not self.enabled or self.version < 0:
            return
        if self.counts["queued"] >= self.config.max_samples:
            self.enabled = False
            return
        counts = scheduled.num_scheduled_tokens
        if (
            not counts
            or scheduled.scheduled_new_reqs
            or scheduled.scheduled_encoder_inputs
            or scheduled.scheduled_spec_decode_tokens
            or any(n != 1 for n in counts.values())
        ):
            self.counts["mixed_or_non_decode"] += 1
            return
        cached = scheduled.scheduled_cached_reqs
        computed = dict(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        # The dense phase is driven exclusively by scheduled CPU metadata and
        # attempted steps, not rank-local event completion or timing results.
        # All TP ranks therefore select the same steps even if one device lags.
        maximum = max((computed.get(request_id, -1) + 1 for request_id in counts), default=0)
        coverage_key = ((maximum - 1) // self.config.context_bucket_tokens, len(counts))
        target = 2 * self.config.min_samples_per_bin + self.config.warmup_per_shape
        known = coverage_key in self.coverage_attempts
        dense = (known or len(self.coverage_attempts) < self.config.max_samples) and (
            self.coverage_attempts[coverage_key] < target
        )
        interval = self.config.coverage_sample_every if dense else self.config.sample_every
        if self.step % interval:
            return
        if dense:
            self.coverage_attempts[coverage_key] += 1
        self.counts["coverage_attempts" if dense else "periodic_attempts"] += 1
        self.poll()
        contexts = []
        for request_id in counts:
            state = self.worker.model_runner.requests.get(request_id)
            done = computed.get(request_id)
            prompt = getattr(state, "num_prompt_tokens", None)
            if prompt is None:
                prompt_ids = getattr(state, "prompt_token_ids", None)
                prompt = len(prompt_ids) if prompt_ids is not None else None
            if state is None or done is None or prompt is None or done < prompt:
                self.counts["prefill_or_unknown_context"] += 1
                return
            contexts.append(done + 1)
        self.last_context = max(contexts)
        if len(self.pending) >= self.config.max_pending_events:
            self.counts["pending_limit"] += 1
            return
        rough_key = (self.last_mode, (max(contexts) - 1) // self.config.context_bucket_tokens, len(counts))
        if self.bins[rough_key] >= self.config.max_samples_per_bin:
            return
        start, end = (
            self.pool.pop()
            if self.pool
            else (self.device.Event(enable_timing=True), self.device.Event(enable_timing=True))
        )
        self.current = {
            "start": start,
            "end": end,
            "host_start": time.perf_counter(),
            "compilations": self._compilations(),
            "sample": {
                "step": self.step,
                "version": self.version,
                "batch": len(counts),
                "context_tokens": sum(contexts),
                "max_context_tokens": max(contexts),
                "pure_decode": True,
                "mode": "unknown",
            },
        }
        start.record()

    def annotate(self):
        if self.current is None:
            return
        context = self.context_reader()
        mode = getattr(context, "cudagraph_runtime_mode", None)
        if mode is None:
            return
        name = getattr(mode, "name", str(mode))
        name = "eager" if name in {"NONE", "0"} else str(name).lower()
        descriptor = getattr(context, "batch_descriptor", None)
        padded = getattr(descriptor, "num_tokens", self.current["sample"]["batch"])
        self.current["sample"].update(mode=name, padded_batch=int(padded))
        capturing = getattr(self.device, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            self.current["capturing"] = True

    def finish(self):
        current, self.current = self.current, None
        if current is None:
            return
        sample = current["sample"]
        # Eager runners without a forward adapter are supported only when the
        # engine explicitly disables graphs. Unknown graph modes are rejected.
        if sample["mode"] == "unknown" and getattr(self.worker.model_runner.model_config, "enforce_eager", False):
            sample.update(mode="eager", padded_batch=sample["batch"])
        self.last_mode = sample["mode"]
        shape = (sample["mode"], sample.get("padded_batch", sample["batch"]))
        self.shapes[shape] += 1
        key = (sample["mode"], (sample["max_context_tokens"] - 1) // self.config.context_bucket_tokens, sample["batch"])
        if (
            sample["mode"] == "unknown"
            or current.get("capturing")
            or current["compilations"] != self._compilations()
            or self.shapes[shape] <= self.config.warmup_per_shape
            or self.bins[key] >= self.config.max_samples_per_bin
        ):
            # These events have no reader yet. Recording end also makes reuse
            # safe on the same ordered stream without synchronizing the device.
            current["end"].record()
            self.pool.append((current["start"], current["end"]))
            self.counts["warmup_capture_or_unknown"] += 1
            return
        current["end"].record()
        sample["host_enqueue_ms"] = 1000 * (time.perf_counter() - current["host_start"])
        self.pending.append((current["start"], current["end"], sample))
        self.bins[key] += 1
        self.counts["queued"] += 1

    def drain(self):
        self.poll()
        samples = [self.samples.popleft() for _ in range(min(len(self.samples), 512))]
        return {
            "rank": int(getattr(self.worker, "rank", getattr(self.worker, "local_rank", 0))),
            "samples": samples,
            "counts": dict(self.counts),
            "enabled": self.enabled and not self.suspended,
            "suspended": self.suspended,
            "error": self.error,
            "mode": self.last_mode,
            "max_context_tokens": self.last_context,
            "versions": sorted(self.versions),
            "pending_events": len(self.pending),
            "capacity": self.capacity,
        }


def install_decode_probe(worker, options):
    existing = getattr(worker, "_trajectory_decode_probe", None)
    if existing is not None:
        return {
            "available": existing.error is None,
            "already_installed": True,
            "error": existing.error,
            "capacity": existing.capacity,
        }
    try:
        probe = DecodeStepProbe(worker, options)
        execute = worker.execute_model
        sample = worker.sample_tokens
    except (AttributeError, ImportError, ValueError) as error:
        return {"available": False, "error": repr(error)}
    worker._trajectory_decode_probe = probe

    def execute_timed(scheduler_output, *args, **kwargs):
        if probe.suspended:
            return execute(scheduler_output, *args, **kwargs)
        try:
            probe.begin(scheduler_output)
        except Exception as error:
            probe.fail(error)
        try:
            result = execute(scheduler_output, *args, **kwargs)
        except BaseException:
            probe.current = None
            raise
        if result is not None:
            try:
                probe.finish()
            except Exception as error:
                probe.fail(error)
        return result

    def sample_timed(*args, **kwargs):
        if probe.suspended:
            return sample(*args, **kwargs)
        try:
            result = sample(*args, **kwargs)
        except BaseException:
            probe.current = None
            raise
        try:
            probe.finish()
        except Exception as error:
            probe.fail(error)
        return result

    worker.execute_model = execute_timed
    worker.sample_tokens = sample_timed
    runner = worker.model_runner
    forward = getattr(runner, "_model_forward", None)
    if forward is not None:

        def forward_timed(*args, **kwargs):
            if probe.suspended:
                return forward(*args, **kwargs)
            try:
                probe.annotate()
            except Exception as error:
                probe.fail(error)
            return forward(*args, **kwargs)

        runner._model_forward = forward_timed
    return {
        "available": True,
        "clock": "device_events",
        "worker": type(worker).__name__,
        "runner": type(runner).__name__,
        "capacity": probe.capacity,
    }


class TPRooflineJoin:
    """Join one sampled step across all TP ranks; never fit rank-zero timings."""

    def __init__(self, tp_size, max_pending=1024):
        self.tp_size = tp_size
        self.max_pending = max_pending
        self.pending = {}
        self.rejected = 0

    def add(self, packets):
        joined = []
        for packet in packets:
            rank = packet["rank"]
            for sample in packet["samples"]:
                key = (sample["version"], sample["step"])
                self.pending.setdefault(key, {})[rank] = sample
        for key in list(self.pending):
            ranks = self.pending[key]
            if len(ranks) != self.tp_size:
                continue
            values = list(ranks.values())
            signatures = {
                (s["batch"], s["context_tokens"], s["max_context_tokens"], s["mode"], s.get("padded_batch"))
                for s in values
            }
            if len(signatures) == 1:
                joined.append(
                    {
                        **values[0],
                        "elapsed_ms": max(s["elapsed_ms"] for s in values),
                        "host_enqueue_ms": max(s["host_enqueue_ms"] for s in values),
                    }
                )
            else:
                self.rejected += 1
            del self.pending[key]
        while len(self.pending) > self.max_pending:
            del self.pending[next(iter(self.pending))]
            self.rejected += 1
        return joined


def set_decode_probe_sampling(worker, enabled, fresh_measurement=False):
    """Suspend between drained workloads for paired overhead measurements.

    The normal switch preserves calibration state and the version/sample window.
    Explicit fresh_measurement starts a drained, independent diagnostic trial;
    it preserves version identity and cannot revive an expired/failed probe.
    Existing wrappers remain installed; suspension skips their timing work.
    """
    if type(enabled) is not bool or type(fresh_measurement) is not bool:
        raise ValueError("Roofline sampling switch must be boolean")
    probe = getattr(worker, "_trajectory_decode_probe", None)
    if probe is None:
        raise RuntimeError("Roofline probe is not installed")
    if probe.current is not None:
        raise RuntimeError("Cannot change sampling inside a model step")
    if fresh_measurement:
        if probe.pending or probe.samples:
            raise RuntimeError("Drain the previous measurement before starting another")
        if probe.error or (
            not probe.config.continuous_sampling and len(probe.versions) > probe.config.calibration_rounds
        ):
            raise RuntimeError("Cannot restart a failed or version-expired probe")
        probe.bins.clear()
        probe.coverage_attempts.clear()
        probe.counts.clear()
        probe.enabled = probe.config.enabled
    probe.suspended = not enabled
    return {"enabled": probe.enabled and enabled, "suspended": probe.suspended, "error": probe.error}
