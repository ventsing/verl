# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Real-server control plane: independent pulls and same-version repacking.

The portable migration transport aborts a request and re-prefills its prefix on
the destination. It preserves policy identity, not the source KV/SSM storage.
This is deliberately reported as ``recompute`` in event traces.
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
import ray

from verl.utils import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


@dataclass
class Replica:
    server: object
    workers: list
    version: int = -1
    admitted: int = 0
    closed: bool = False
    pulling: bool = False
    active: dict = field(default_factory=dict)
    reserved: set = field(default_factory=set)
    previous_usage: float = 0.0
    ramp_down: bool = False
    relay: object = None
    healthy: bool = True
    recovering: bool = False
    owned: bool = True
    releasing: bool = False
    admission_ceiling: int | None = None

    @property
    def load(self):
        return len(self.active) + len(self.reserved)


@dataclass
class Request:
    request_id: str
    slot: int
    version: int
    backend_id: str = ""
    destination: int | None = None
    migrations: int = 0
    recoveries: int = 0
    prompt_tokens: int = 0
    response_budget: int = 0
    generated_tokens: int = 0
    generated_base: int = 0
    last_pressure_migration: float = -math.inf


def consolidation_plan(snapshots, max_requests, capacity=0.9, request_limit=None, usage_projection=None):
    """Best-fit consolidation with projected source AND destination loads.

    Destinations are never subsequently used as sources in this round. This
    avoids chain plans whose source load would omit already reserved arrivals.
    Caller supplies one version group, with nonempty ramp-down candidates.
    """
    ordered = sorted(snapshots, key=lambda s: s["usage"])
    loads = {s["id"]: [s["usage"], s["count"], s.get("context_tokens", 0)] for s in ordered}
    emptied, destinations, plan = set(), set(), []
    for source in ordered:
        sid = source["id"]
        if sid in emptied or sid in destinations:
            continue
        fits = [
            d
            for d in ordered
            if d["id"] != sid
            and d["id"] not in emptied
            and (
                usage_projection(d["id"], loads[d["id"]][0], loads[sid][1])
                if usage_projection
                else loads[d["id"]][0] + loads[sid][0]
            )
            <= capacity
            and loads[d["id"]][1] + loads[sid][1]
            <= (request_limit(d["id"], max(loads[d["id"]][2], loads[sid][2])) if request_limit else max_requests)
        ]
        if not fits:
            continue
        target = max(fits, key=lambda d: loads[d["id"]][0])["id"]
        loads[target][0] = (
            usage_projection(target, loads[target][0], loads[sid][1])
            if usage_projection
            else loads[target][0] + loads[sid][0]
        )
        loads[target][1] += loads[sid][1]
        loads[target][2] = max(loads[target][2], loads[sid][2])
        emptied.add(sid)
        destinations.add(target)
        plan.append((sid, target))
    return plan


class LaminarController:
    """Async Ray actor owning admission, drain boundaries, pulls and migration.

    No trainer collective or rollout-wide pause is used by this controller.
    In-flight requests reserve their version on admission; replicas can update
    only after both active requests and migration reservations reach zero.
    """

    def __init__(
        self,
        replicas,
        relay,
        batch_quota=8,
        max_running=8,
        check_interval_s=5.0,
        enable_repack=True,
        event_path=None,
        partial_store=None,
        recovery_options=None,
        repack_max_running=0,
        roofline_options=None,
        capacity_options=None,
        capacity_profiles=None,
    ):
        if batch_quota < 1 or max_running < 1 or check_interval_s <= 0:
            raise ValueError("Quotas, max_running and check_interval_s must be positive")
        self.replicas = [Replica(**r) for r in replicas]
        self.relay = relay
        self.partial_store = partial_store
        self.recovery_options = dict(recovery_options or {})
        self.pinned_versions = set()
        self.pin_owner = "laminar-partial-pool"
        self.batch_quota = batch_quota
        self.max_running = max_running
        if repack_max_running < 0:
            raise ValueError("Repack capacity cannot be negative")
        self.repack_max_running = min(max_running, repack_max_running or max_running)
        self.roofline = {}
        self.roofline_last_fit = {}
        self.roofline_dirty = set()
        self.roofline_lock = asyncio.Lock()
        if roofline_options and roofline_options.get("enabled", False):
            from verl.experimental.trajectory_async.roofline import OnlineRoofline

            self.roofline = {i: OnlineRoofline(roofline_options) for i in range(len(replicas))}
        self.capacity = {}
        self._budget_cache = {}
        if (
            capacity_options
            and capacity_options.get("enabled", False)
            and (self.roofline or capacity_options.get("token_budget_admission", False))
        ):
            from verl.experimental.trajectory_async.capacity import OnlineCapacity

            self.capacity = {
                i: OnlineCapacity(batch_quota, self.max_running, capacity_options) for i in range(len(replicas))
            }
        if self.capacity:
            shared = next(iter(self.capacity.values())).responses
            for governor in self.capacity.values():
                governor.responses = shared
            history_path = next(iter(self.capacity.values())).config.response_history_path
            if history_path and next(iter(self.capacity.values())).config.token_budget_admission:
                shared.seed(history_path)
        for index, profile in enumerate(capacity_profiles or []):
            if index in self.capacity and profile and profile.get("available"):
                governor = self.capacity[index]
                governor.pool = profile
                if governor.memory_limit() < 1:
                    raise ValueError("Initial KV/SSM pool cannot fit one request within its safety margin")
                governor.current = min(governor.current, governor.memory_limit())
                governor.reason = "profiled_initial_pool"
        if any(g.config.token_budget_admission and not g.pool.get("available") for g in self.capacity.values()):
            raise ValueError("Token-budget admission requires an actual profiled KV pool before startup")
        self.check_interval = check_interval_s
        if any(
            g.config.rolling_kv_enabled and g.config.rolling_kv_horizon_s < 2 * check_interval_s
            for g in self.capacity.values()
        ):
            raise ValueError("Rolling KV horizon must cover at least two telemetry polling intervals")
        self.enable_repack = enable_repack
        self.latest = -1
        self.minimum_version = -1
        self.supply_paused = False
        self.condition = asyncio.Condition()
        self.wakeup = asyncio.Event()
        self.repack_requested = False
        self.task = None
        self.pulls = set()
        self.scheduled_pulls = set()
        self.failure = None
        self.stopped = False
        self.pending = set()
        self.request_tasks = {}
        self.imported_requests = set()
        self.event_path = Path(event_path) if event_path else None
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "completed": 0,
            "migrations": 0,
            "drain_moves": 0,
            "pulls": 0,
            "repack_checks": 0,
            "roofline_polls": 0,
            "preflight_completed": 0,
            "preflight_migrations": 0,
            "replayed_experiences": 0,
            "recoveries": 0,
            "quarantined_replicas": 0,
        }
        self.last_telemetry = []

    def _event(self, event, **fields):
        record = {"event": event, "time": time.time(), **fields}
        if self.event_path:
            with self.event_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def _check(self):
        if self.failure:
            raise RuntimeError(f"Laminar controller failed: {self.failure}")
        if self.stopped:
            raise RuntimeError("Laminar controller stopped")

    async def start(self, version):
        await self.notify_weights(version)
        if self.partial_store is not None:
            for retained in await self.partial_store.versions.remote():
                await self._retain_version(retained)
        await asyncio.gather(*(self._pull(i) for i in range(len(self.replicas))))
        self.task = asyncio.create_task(self._run())
        return self.status()

    def _relays(self):
        # Ray handle equality is stable; plain test handles are also hashable.
        return list(dict.fromkeys([self.relay] + [r.relay for r in self.replicas if r.relay is not None]))

    async def _retain_version(self, version):
        if self.partial_store is not None and version not in self.pinned_versions:
            for relay in self._relays():
                await relay.retain.remote(version, f"{self.pin_owner}:{version}")
            self.pinned_versions.add(version)

    async def _collect_versions(self):
        if self.partial_store is None:
            return
        needed = set(await self.partial_store.versions.remote()) | {self.latest}
        needed.update(r.version for r in self.replicas if r.owned and r.healthy and r.version >= 0)
        for version in list(self.pinned_versions - needed):
            for relay in self._relays():
                await relay.release_owner.remote(f"{self.pin_owner}:{version}")
            self.pinned_versions.remove(version)

    async def notify_weights(self, version):
        await self._retain_version(version)
        async with self.condition:
            if version < self.latest:
                raise ValueError("Published versions must not decrease")
            self.latest = version
            self.repack_requested = True
            for replica in self.replicas:
                if replica.version < version:
                    replica.closed = True
            self._event("published", version=version, active=sum(r.load for r in self.replicas))
            self.wakeup.set()
            self.condition.notify_all()

    async def set_supply_paused(self, paused):
        async with self.condition:
            self.supply_paused = bool(paused)
            self.condition.notify_all()

    async def release_replica(self, index):
        """Fence admission, then relinquish an idle instance without stopping it."""
        replica = self.replicas[index]
        replica.releasing = True
        replica.closed = True
        self.wakeup.set()
        if replica.load:
            await self._drain_replica(index)
        if replica.load or replica.pulling or replica.recovering or index in self.scheduled_pulls:
            return False
        replica.owned = False
        return True

    def _migration_target_available(self, index, version):
        replica = self.replicas[index]
        return (
            replica.owned
            and replica.healthy
            and not replica.releasing
            and not replica.recovering
            and not replica.pulling
            and index not in self.scheduled_pulls
            and not any(request.destination is not None for request in replica.active.values())
            and replica.version == version
        )

    async def _drain_replica(self, index):
        """Evacuate to same-version capacity without waiting for a ramp-down phase.

        Plan partial moves across multiple targets if one cannot hold the whole
        source. Reserve every arrival before aborting any source request. The
        normal generation path preserves tokens/logprobs and confirms completion
        of the move before release_replica can acknowledge the handoff.
        """
        source = self.replicas[index]
        if (
            not source.owned
            or not source.healthy
            or not source.releasing
            or source.pulling
            or source.recovering
            or source.reserved
            or index in self.scheduled_pulls
            or not source.active
        ):
            return 0
        candidates = [
            i for i in range(len(self.replicas)) if i != index and self._migration_target_available(i, source.version)
        ]
        if not candidates:
            return 0
        indices = [index] + candidates
        packets = await asyncio.gather(
            *[
                asyncio.wait_for(
                    self.replicas[i].server.trajectory_status.remote(),
                    self.recovery_options.get("heartbeat_timeout_s", 10.0),
                )
                for i in indices
            ],
            return_exceptions=True,
        )
        telemetry = {
            i: packet for i, packet in zip(indices, packets, strict=True) if not isinstance(packet, BaseException)
        }
        now = time.time()

        def fresh(i):
            data = telemetry.get(i, {})
            usage = data.get("kv_usage", math.inf)
            return (
                math.isfinite(usage)
                and usage >= 0
                and data.get("version", self.replicas[i].version) == self.replicas[i].version
                and 0 <= now - data.get("observed_at", -math.inf) < 2 * self.check_interval
            )

        # Ownership and weight loads may change during the telemetry RPCs.
        if (
            not fresh(index)
            or not source.owned
            or not source.releasing
            or not source.healthy
            or source.pulling
            or source.recovering
            or source.reserved
        ):
            return 0
        counts = {i: self.replicas[i].load for i in candidates}
        usages = {}
        for target in candidates:
            if not fresh(target):
                continue
            replica = self.replicas[target]
            backend_ids = set(telemetry[target].get("request_ids", []))
            unobserved = len(replica.reserved) + sum(
                request.backend_id not in backend_ids for request in replica.active.values()
            )
            governor = self.capacity.get(target)
            # Metadata may include arrivals not yet visible in backend KV usage.
            usages[target] = (
                governor.projected_usage(telemetry[target]["kv_usage"], unobserved)
                if governor is not None
                else math.inf
                if unobserved
                else telemetry[target]["kv_usage"]
            )
        moves = []
        charged_targets = set()
        for request in list(source.active.values()):
            if request.destination is not None or request.backend_id not in telemetry[index].get("request_ids", []):
                continue
            fits = []
            for target in usages:
                if not self._migration_target_available(target, request.version) or telemetry[target].get("waiting", 0):
                    continue
                context = max(telemetry[i].get("roofline", {}).get("max_context_tokens", 0) for i in (index, target))
                limit = self._repack_limit(target, telemetry, context)
                governor = self.capacity.get(target)
                # Without a calibrated pool, charge the full source once per
                # destination: this bounds any subset of the source's prefixes.
                projected = (
                    governor.projected_usage(usages[target], 1)
                    if governor is not None
                    else usages[target] + (0 if target in charged_targets else telemetry[index]["kv_usage"])
                )
                margin = governor.config.kv_usage_limit if governor is not None else 0.9
                if counts[target] < limit and projected <= margin:
                    fits.append((counts[target], target, projected))
            if not fits:
                continue
            _, target, projected = min(fits)
            destination = self.replicas[target]
            destination.closed = True
            destination.reserved.add(request.request_id)
            request.destination = target
            counts[target] += 1
            usages[target] = projected
            charged_targets.add(target)
            moves.append((index, request, request.backend_id))
        if moves:
            self.metrics["drain_moves"] += len(moves)
            self._event(
                "drain_repack",
                source=index,
                version=source.version,
                targets=[request.destination for _, request, _ in moves],
                requests=len(moves),
            )
        return await self._execute_moves(moves)

    async def acquire_replica(self, index, admission_ceiling=None):
        """Called only after the other controller acknowledged release."""
        replica = self.replicas[index]
        if replica.owned or replica.load or replica.pulling or replica.recovering:
            raise RuntimeError("Replica acquisition requires an unowned idle slot")
        replica.admission_ceiling = admission_ceiling
        replica.version = -1  # The other pool may have loaded a different version.
        replica.admitted = 0
        replica.previous_usage = 0.0
        replica.ramp_down = False
        replica.releasing = False
        replica.closed = True
        replica.owned = True
        self._event("replica_acquired", replica=index)
        self.wakeup.set()

    async def cancel_release(self, index):
        replica = self.replicas[index]
        if not replica.owned:
            raise RuntimeError("Cannot cancel release after ownership was transferred")
        replica.releasing = False
        self.wakeup.set()

    async def set_minimum_version(self, version):
        async with self.condition:
            self.minimum_version = version
            self.condition.notify_all()

    async def _pull(self, index, exact_version=None):
        replica = self.replicas[index]
        # Reserve before any await, so admission cannot race a load.
        if not replica.owned or replica.releasing or replica.load or replica.pulling:
            return
        replica.pulling = True
        manifest = None
        relay = replica.relay if replica.relay is not None else self.relay
        started = time.monotonic()
        try:
            # Hierarchical relays may still be receiving this committed version.
            manifest = (
                await relay.acquire.remote(exact_version if exact_version is not None else self.latest)
                if replica.relay is not None or exact_version is not None
                else await relay.acquire.remote()
            )
            version = manifest["version"]
            self._event(
                "pull_begin",
                replica=index,
                version=version,
                other_active=sum(r.load for i, r in enumerate(self.replicas) if i != index),
            )
            await replica.server.wait_for_requests_to_drain.remote()
            await asyncio.gather(
                *[worker.execute_checkpoint_engine.remote("set_manifest", manifest) for worker in replica.workers]
            )
            await asyncio.gather(*[worker.update_weights.remote(global_steps=version) for worker in replica.workers])
            replica.version = version
            if self._continuous_admission(index):
                self.capacity[index].reset_observation()
            replica.admitted = 0
            replica.previous_usage = 0.0
            replica.ramp_down = False
            replica.closed = version < self.latest
            self.metrics["pulls"] += 1
            self._event("pull_end", replica=index, version=version, seconds=time.monotonic() - started)
        except Exception as error:
            if self.partial_store is not None and self._recoverable(error):
                self._quarantine(index, error)
            else:
                self.failure = repr(error)
                self._event("failure", replica=index, error=self.failure)
            raise
        finally:
            # Hold the relay lease until every TP worker has relinquished its
            # manifest. Failed loads remain unroutable, never partially fresh.
            try:
                results = await asyncio.gather(
                    *[
                        asyncio.wait_for(
                            worker.execute_checkpoint_engine.remote("finalize"),
                            self.recovery_options.get("heartbeat_timeout_s", 10.0),
                        )
                        for worker in replica.workers
                    ],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        if self.partial_store is not None and self._recoverable(result):
                            self._quarantine(index, result)
                        else:
                            raise result
            finally:
                if manifest is not None:
                    await relay.release.remote(manifest["lease"])
                async with self.condition:
                    replica.pulling = False
                    self.condition.notify_all()

    def _continuous_admission(self, index):
        return index in self.capacity and self.capacity[index].token_budget_enabled()

    def _admission_projection(self, index, request=None):
        governor = self.capacity.get(index)
        if governor is None or not governor.token_budget_enabled():
            return None
        replica = self.replicas[index]
        reserved = {
            q.request_id: q for r in self.replicas for q in r.active.values() if q.request_id in replica.reserved
        }
        if len(reserved) != len(replica.reserved):
            return math.inf  # Unknown migration arrivals cannot spend free KV.
        now = time.time()

        def signature(q):
            return (
                q.request_id,
                q.backend_id,
                q.prompt_tokens,
                q.response_budget,
                q.generated_tokens,
                q.generated_base,
            )

        # Thousands of condition waiters can retry against exactly the same
        # ledger. Cache only its existing-reservation sum, never an incoming
        # request's cost. Identity, progress, arrivals, history and freshness
        # all invalidate it, including expiry without a new telemetry tick.
        key = (
            id(governor.pool),
            governor.profile_available,
            governor.last_observed,
            governor.observed_usage,
            governor.fresh(now),
            governor.responses.revision,
            governor.rate_revision,
            governor.config.rolling_kv_enabled,
            int(max(0, now - governor.last_observed)) if math.isfinite(governor.last_observed) else -1,
            tuple(signature(q) for q in replica.active.values()),
            tuple(signature(q) for q in reserved.values()),
        )
        cached = self._budget_cache.get(index)
        if cached is None or cached[0] != key:
            base = governor.reserved_usage(list(replica.active.values()), list(reserved.values()), now)
            self._budget_cache[index] = (key, base)
        else:
            base = cached[1]
        if request is None or base is None:
            return base
        return base + governor.request_reserve(request, now, incoming=True, version=replica.version)

    def _select_admission(self, available, request):
        scores = {}
        for index in available:
            if index in self.capacity and self.capacity[index].pressure_paused:
                continue
            projected = self._admission_projection(index, request)
            if projected is None:
                scores[index] = (self.replicas[index].load, self.replicas[index].load)
            elif projected <= self.capacity[index].config.kv_usage_limit:
                scores[index] = (projected, self.replicas[index].load)
        return min(scores, key=scores.get) if scores else None

    def _reserved_usage_status(self, index):
        projection = self._admission_projection(index)
        return projection if projection is not None and math.isfinite(projection) else None

    def calibrate_response_lengths(self, step, records):
        """Calibrate once per consumed batch, preserving completion-history counts."""
        if not self.capacity:
            return {"updated": False}
        estimator = next(iter(self.capacity.values())).responses
        if not getattr(estimator.config, "response_step_calibration", False):
            return {"updated": False}
        updated = estimator.calibrate(step, records)
        if updated:
            self._budget_cache.clear()
            self._event("response_length_calibration", step=step, records=len(records), **estimator.status())
        return {"updated": updated, **estimator.status()}

    def _observe_response(self, tokens, is_probe=False):
        if self.capacity and not is_probe:
            next(iter(self.capacity.values())).responses.observe(tokens)

    async def _assign(self, request_id, is_validation=False, prompt_tokens=0, response_budget=0):
        async with self.condition:
            while True:
                self._check()
                if self.partial_store is not None and not any(r.healthy for r in self.replicas):
                    raise RuntimeError("No healthy rollout replica remains; durable prefixes are retained")
                if (self.minimum_version >= 0 or self.supply_paused) and not is_validation:
                    await self.condition.wait()
                    continue
                required_version = max(self.minimum_version, self.latest) if is_validation else -1
                available = [
                    i
                    for i, r in enumerate(self.replicas)
                    if r.owned
                    and not r.releasing
                    and r.healthy
                    and not r.recovering
                    and not r.closed
                    and not r.pulling
                    and r.load < self._admission_limit(i)
                    and r.version >= required_version
                    and (self._continuous_admission(i) or r.admitted < self._admission_limit(i))
                ]
                request = Request(request_id, -1, -1, prompt_tokens=prompt_tokens, response_budget=response_budget)
                index = self._select_admission(available, request)
                if index is not None:
                    replica = self.replicas[index]
                    request.slot, request.version = index, replica.version
                    replica.active[request_id] = request
                    replica.admitted += 1
                    replica.closed = not self._continuous_admission(
                        index
                    ) and replica.admitted >= self._admission_limit(index)
                    return request
                await self.condition.wait()

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        if self.partial_store is not None:
            return await self._generate_durable(request_id, prompt_ids, sampling_params, **kwargs)
        prompt_ids = normalize_token_ids(prompt_ids)
        if request_id in self.pending or request_id in self.imported_requests:
            raise ValueError(f"Duplicate in-flight request ID {request_id}")
        self.pending.add(request_id)
        request = None
        tokens, logprobs, experts = [], [], []
        sampling_params = dict(sampling_params)
        is_validation = bool(sampling_params.pop("_laminar_validate", False))
        budget = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", None))
        if not isinstance(budget, int) or budget <= 0:
            self.pending.remove(request_id)
            raise ValueError("Laminar requests require an explicit positive response budget")
        if kwargs.get("image_data") or kwargs.get("video_data") or kwargs.get("audio_data"):
            self.pending.remove(request_id)
            raise ValueError("Laminar recompute migration currently supports text-only trajectories")
        started = time.monotonic()
        self.request_tasks[request_id] = asyncio.current_task()
        try:
            request = await self._assign(
                request_id, is_validation=is_validation, prompt_tokens=len(prompt_ids), response_budget=budget
            )
            is_probe = request_id.startswith("preflight-")
            self._event(
                "request_begin", request=request_id, replica=request.slot, version=request.version, is_probe=is_probe
            )
            while len(tokens) < budget:
                replica = self.replicas[request.slot]
                if replica.version != request.version or replica.pulling:
                    raise RuntimeError("In-flight trajectory policy version changed")
                request.backend_id = f"laminar-{request_id}-{uuid4().hex}"
                request.generated_base = len(tokens)
                request.generated_tokens = len(tokens)
                output = await replica.server.generate.remote(
                    request_id=request.backend_id,
                    prompt_ids=list(prompt_ids) + tokens,
                    sampling_params={**sampling_params, "max_tokens": budget - len(tokens)},
                    **kwargs,
                )
                version = output.extra_fields.get("global_steps")
                if version != request.version:
                    raise RuntimeError(f"Behavior policy mismatch: expected {request.version}, got {version}")
                if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                    raise RuntimeError("Each generated token must retain its behavior log probability")
                if not all(math.isfinite(value) for value in output.log_probs):
                    raise RuntimeError("Non-finite behavior log probability")
                tokens.extend(output.token_ids)
                request.generated_tokens = len(tokens)
                logprobs.extend(output.log_probs)
                if len(tokens) > budget:
                    raise RuntimeError("Backend exceeded the remaining trajectory budget")
                if request.migrations:
                    self._event(
                        "continuation_segment",
                        request=request_id,
                        version=request.version,
                        tokens=len(output.token_ids),
                        prefill_seconds=output.extra_fields.get("prefill_seconds"),
                    )
                if output.routed_experts is not None:
                    experts.append(output.routed_experts)
                if output.stop_reason not in ("aborted", "abort") or len(tokens) == budget:
                    self.metrics["preflight_completed" if is_probe else "completed"] += 1
                    self._observe_response(len(tokens), is_probe)
                    self._event(
                        "request_end",
                        request=request_id,
                        version=request.version,
                        tokens=len(tokens),
                        migrations=request.migrations,
                        seconds=time.monotonic() - started,
                        is_probe=is_probe,
                    )
                    return TokenOutput(
                        token_ids=tokens,
                        log_probs=logprobs,
                        routed_experts=np.concatenate(experts, axis=0) if experts else None,
                        stop_reason="completed",
                        extra_fields={
                            **output.extra_fields,
                            "global_steps": request.version,
                            "min_global_steps": request.version,
                            "max_global_steps": request.version,
                            "migrations": request.migrations,
                            "migration_transport": "recompute",
                            "trajectory_request_id": request_id,
                        },
                    )
                async with self.condition:
                    if request.destination is None:
                        raise RuntimeError("Unexpected abort: no same-version migration was reserved")
                    source = request.slot
                    target = request.destination
                    destination = self.replicas[target]
                    if destination.version != request.version or destination.pulling:
                        raise RuntimeError("Migration destination policy version changed")
                    self.replicas[source].active.pop(request_id)
                    destination.reserved.remove(request_id)
                    destination.active[request_id] = request
                    request.slot, request.destination = target, None
                    request.migrations += 1
                    self.metrics["preflight_migrations" if is_probe else "migrations"] += 1
                    self._event(
                        "migration",
                        request=request_id,
                        source=source,
                        destination=target,
                        version=request.version,
                        prefix_tokens=len(prompt_ids) + len(tokens),
                        transport="recompute",
                        is_probe=is_probe,
                    )
                    self.wakeup.set()
                    self.condition.notify_all()
        finally:
            async with self.condition:
                self.pending.discard(request_id)
                self.request_tasks.pop(request_id, None)
                if request is not None:
                    self.replicas[request.slot].active.pop(request_id, None)
                    if request.destination is not None:
                        self.replicas[request.destination].reserved.discard(request_id)
                self.wakeup.set()
                self.condition.notify_all()

    def _quarantine(self, index, error):
        replica = self.replicas[index]
        if replica.healthy:
            replica.healthy = False
            replica.closed = True
            self.metrics["quarantined_replicas"] += 1
            self._event("replica_quarantined", replica=index, error=repr(error))
            if not any(r.healthy for r in self.replicas):
                self.failure = "No healthy rollout replica remains; durable prefixes are retained"
                self._event("failure", error=self.failure)
        self.wakeup.set()

    async def _wait_backend(self, request, reference):
        task = asyncio.ensure_future(reference)
        heartbeat = self.recovery_options.get("heartbeat_interval_s", 5.0)
        timeout = self.recovery_options.get("heartbeat_timeout_s", 10.0)
        progress_timeout = self.recovery_options.get("progress_timeout_s", 600.0)
        failures = 0
        try:
            while True:
                done, _ = await asyncio.wait([task], timeout=heartbeat)
                if done:
                    return await task
                if not self.replicas[request.slot].healthy:
                    raise ConnectionError("Rollout replica was quarantined")
                try:
                    status = await asyncio.wait_for(
                        self.replicas[request.slot].server.trajectory_status.remote(), timeout
                    )
                except Exception as error:
                    failures += 1
                    if failures >= self.recovery_options.get("heartbeat_failures", 2):
                        raise ConnectionError("Rollout heartbeat failed") from error
                    continue
                failures = 0
                if status["version"] != request.version:
                    raise ValueError("Heartbeat reports a different behavior policy")
                progress = await self.partial_store.progress.remote(request.request_id)
                if time.time() - progress["updated"] > progress_timeout:
                    raise TimeoutError("Rollout made no durable token progress")
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _recoverable(error):
        cause = getattr(error, "cause", error)
        return isinstance(
            cause,
            ConnectionError
            | TimeoutError
            | ray.exceptions.RayActorError
            | ray.exceptions.RaySystemError
            | ray.exceptions.ObjectLostError,
        ) or (type(cause).__name__ == "EngineDeadError")

    async def _recover_slot(self, request):
        """Reserve an exact-version target, reloading retained weights if needed."""
        source = request.slot
        if request.destination is not None:
            self.replicas[request.destination].reserved.discard(request.request_id)
            request.destination = None
        self.replicas[source].active.pop(request.request_id, None)
        self.wakeup.set()
        async with asyncio.timeout(self.recovery_options.get("recovery_timeout_s", 300.0)):
            while True:
                async with self.condition:
                    self._check()
                    healthy = [i for i, r in enumerate(self.replicas) if r.owned and not r.releasing and r.healthy]
                    if not healthy:
                        raise RuntimeError("No healthy rollout replica remains; durable prefix is retained")
                    same = [
                        i
                        for i in healthy
                        if not self.replicas[i].pulling
                        and not self.replicas[i].recovering
                        and self.replicas[i].version == request.version
                        and self.replicas[i].load < self._admission_limit(i)
                    ]
                    if same:
                        target = min(same, key=lambda i: self.replicas[i].load)
                        self.replicas[target].active[request.request_id] = request
                        self.replicas[target].closed = True
                        request.slot = target
                        break
                    idle = [
                        i
                        for i in healthy
                        if not self.replicas[i].pulling
                        and not self.replicas[i].recovering
                        and not self.replicas[i].load
                    ]
                    if idle:
                        target = idle[0]
                        self.replicas[target].recovering = True
                    else:
                        await self.condition.wait()
                        continue
                try:
                    try:
                        await self._pull(target, exact_version=request.version)
                    except Exception as error:
                        if self._recoverable(error):
                            continue  # A second failed target does not stop other healthy replicas.
                        raise
                    async with self.condition:
                        self.replicas[target].active[request.request_id] = request
                        self.replicas[target].closed = True
                        request.slot = target
                    break
                finally:
                    self.replicas[target].recovering = False
        self.metrics["recoveries"] += 1
        request.recoveries += 1
        self._event(
            "trajectory_recovered",
            request=request.request_id,
            source=source,
            destination=request.slot,
            version=request.version,
        )

    def _stored_output(self, state, migrations=None):
        version = state["version"]
        return TokenOutput(
            token_ids=state["token_ids"],
            log_probs=state["log_probs"],
            stop_reason="completed",
            extra_fields={
                **state["extra_fields"],
                "global_steps": version,
                "min_global_steps": version,
                "max_global_steps": version,
                "migrations": state["extra_fields"].get("migrations", 0) if migrations is None else migrations,
                "migration_transport": "recompute",
                "trajectory_request_id": state["request_id"],
                "durable_partial": True,
            },
        )

    async def _generate_durable(self, request_id, prompt_ids, sampling_params, **kwargs):
        prompt_ids = normalize_token_ids(prompt_ids)
        params = dict(sampling_params)
        is_validation = bool(params.pop("_laminar_validate", False))
        budget = params.pop("max_tokens", params.pop("max_new_tokens", None))
        if type(budget) is not int or budget <= 0:
            raise ValueError("Laminar requests require an explicit positive response budget")
        if any(kwargs.get(k) for k in ("image_data", "video_data", "audio_data")):
            raise ValueError("Durable trajectories currently support text-only generation")
        if request_id in self.pending or request_id in self.imported_requests:
            raise ValueError(f"Duplicate in-flight request ID {request_id}")
        self.pending.add(request_id)
        self.request_tasks[request_id] = asyncio.current_task()
        request = None
        started = time.monotonic()
        try:
            state = await self.partial_store.lookup.remote(request_id)
            if state is not None:
                # create() verifies immutable input identity before replay/resume.
                state = await self.partial_store.create.remote(request_id, prompt_ids, params, budget, state["version"])
                if state["status"] == "completed":
                    return self._stored_output(state)
                if state["status"] == "abandoned":
                    raise ValueError("Trajectory was explicitly abandoned")
                request = Request(
                    request_id,
                    -1,
                    state["version"],
                    migrations=state["extra_fields"].get("migrations", 0),
                    recoveries=state["extra_fields"].get("recoveries", 0),
                    prompt_tokens=len(prompt_ids),
                    response_budget=budget,
                    generated_tokens=len(state["token_ids"]),
                )
                # There is no live source on a reattach; fence any previous writer.
                await self.partial_store.fence.remote(request_id)
                request.slot = 0
                await self._recover_slot(request)
            else:
                request = await self._assign(
                    request_id, is_validation=is_validation, prompt_tokens=len(prompt_ids), response_budget=budget
                )
                await self._retain_version(request.version)
                state = await self.partial_store.create.remote(request_id, prompt_ids, params, budget, request.version)
            is_probe = request_id.startswith("preflight-")
            self._event(
                "request_begin", request=request_id, replica=request.slot, version=request.version, is_probe=is_probe
            )
            failures = 0
            while state["status"] != "completed":
                if len(state["token_ids"]) == budget:
                    attempt = await self.partial_store.begin_attempt.remote(request_id, uuid4().hex)
                    await self.partial_store.append.remote(
                        request_id, attempt["epoch"], request.version, budget, [], [], terminal=True
                    )
                    state = await self.partial_store.get.remote(request_id)
                    break
                replica = self.replicas[request.slot]
                if replica.version != request.version or replica.pulling:
                    raise RuntimeError("In-flight trajectory policy version changed")
                attempt = await self.partial_store.begin_attempt.remote(request_id, uuid4().hex)
                request.backend_id = f"laminar-{request_id}-{uuid4().hex}"
                base = len(state["token_ids"])
                request.generated_base = request.generated_tokens = base
                context = {
                    **attempt,
                    "store": self.partial_store,
                    "request_id": request_id,
                    "chunk_tokens": self.recovery_options.get("chunk_tokens", 128),
                    "interval_s": self.recovery_options.get("flush_interval_s", 1.0),
                    "extra_fields": {"migrations": request.migrations, "recoveries": request.recoveries},
                }
                try:
                    output = await self._wait_backend(
                        request,
                        replica.server.generate.remote(
                            request_id=request.backend_id,
                            prompt_ids=prompt_ids + state["token_ids"],
                            sampling_params={**params, "max_tokens": budget - base},
                            trajectory_progress=context,
                            **kwargs,
                        ),
                    )
                except Exception as error:
                    if not self._recoverable(error):
                        raise
                    # Fencing is committed before any new generation is admitted.
                    state = await self.partial_store.fence.remote(request_id)
                    self._quarantine(request.slot, error)
                    failures += 1
                    if state["status"] == "completed":
                        break  # Durable EOS/length survived a lost final RPC response.
                    if failures > self.recovery_options.get("max_retries", 3):
                        raise RuntimeError("Trajectory recovery retry limit exceeded; prefix retained") from error
                    await self._recover_slot(request)
                    continue
                state = await self.partial_store.get.remote(request_id)
                if output.extra_fields.get("global_steps") != request.version:
                    raise RuntimeError("Behavior policy mismatch after durable generation")
                if state["token_ids"][base:] != output.token_ids or state["log_probs"][base:] != output.log_probs:
                    raise RuntimeError("Backend result differs from acknowledged durable prefix")
                if state["status"] == "completed":
                    break
                async with self.condition:
                    if output.stop_reason not in ("abort", "aborted") or request.destination is None:
                        raise RuntimeError("Unexpected partial segment without a migration reservation")
                    target = request.destination
                    destination = self.replicas[target]
                    if not destination.healthy:
                        # Destination died after reservation/abort. Keep the
                        # durable prefix on the still-pinned source and resume.
                        destination.reserved.discard(request_id)
                        request.destination = None
                        self._event(
                            "migration_cancelled",
                            request=request_id,
                            source=request.slot,
                            destination=target,
                            version=request.version,
                        )
                        self.condition.notify_all()
                        continue
                    if destination.version != request.version or destination.pulling:
                        raise RuntimeError("Migration destination changed version")
                    source = request.slot
                    replica.active.pop(request_id, None)
                    destination.reserved.discard(request_id)
                    destination.active[request_id] = request
                    request.slot, request.destination = target, None
                    request.migrations += 1
                    self.metrics["preflight_migrations" if is_probe else "migrations"] += 1
                    self._event(
                        "migration",
                        request=request_id,
                        source=source,
                        destination=target,
                        version=request.version,
                        prefix_tokens=len(prompt_ids) + len(state["token_ids"]),
                        transport="recompute",
                        is_probe=is_probe,
                    )
                    self.wakeup.set()
                    self.condition.notify_all()
            self.metrics["preflight_completed" if is_probe else "completed"] += 1
            self._observe_response(len(state["token_ids"]), is_probe)
            self._event(
                "request_end",
                request=request_id,
                version=request.version,
                tokens=len(state["token_ids"]),
                migrations=request.migrations,
                seconds=time.monotonic() - started,
                is_probe=is_probe,
            )
            return self._stored_output(state, request.migrations)
        except Exception as error:
            # Probability/version/prefix invariants must stop training, rather
            # than be swallowed as a bad GRPO group and silently keep going.
            if request is not None:
                self.failure = repr(error)
                self._event("failure", request=request_id, error=self.failure)
                self.wakeup.set()
            raise
        finally:
            # Stop/caller cancellation also fences a potentially still-live worker.
            if request is not None:
                saved = await self.partial_store.fence.remote(request_id)
                if saved["status"] != "completed" and request.backend_id:
                    try:
                        await asyncio.wait_for(
                            self.replicas[request.slot].server.abort_trajectory.remote(request.backend_id),
                            self.recovery_options.get("heartbeat_timeout_s", 10.0),
                        )
                    except Exception:
                        pass  # The committed fence already makes late output harmless.
            async with self.condition:
                self.pending.discard(request_id)
                self.request_tasks.pop(request_id, None)
                if request is not None:
                    self.replicas[request.slot].active.pop(request_id, None)
                    if request.destination is not None:
                        self.replicas[request.destination].reserved.discard(request_id)
                self.wakeup.set()
                self.condition.notify_all()

    async def durability_status(self):
        if self.partial_store is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "pool": await self.partial_store.status.remote(),
            "incomplete": await self.partial_store.incomplete.remote(),
            "relays": [await relay.status.remote() for relay in self._relays()],
        }

    async def abandon(self, request_id, reason):
        if self.partial_store is None:
            raise RuntimeError("Partial response storage is disabled")
        if request_id in self.pending:
            raise ValueError("Cancel and settle the active request before abandoning its durable prefix")
        state = await self.partial_store.abandon.remote(request_id, reason)
        self._event("trajectory_abandoned", request=request_id, version=state["version"], reason=reason)
        await self._collect_versions()
        return {"request_id": request_id, "status": state["status"]}

    def record_initial_experiences(self, records):
        self._check()
        if self.latest != 0 or self.pending or self.imported_requests:
            raise ValueError("Initial experience replay must precede live generation on fresh policy version 0")
        for record in records:
            if record["version"] != 0 or record["request"] in self.imported_requests:
                raise ValueError("Initial replay requires unique version-zero experiences")
            self._event("request_import", **record)
            self.imported_requests.add(record["request"])
        self.metrics["replayed_experiences"] = len(records)

    async def _ingest_roofline(self, telemetry):
        if not self.roofline:
            return
        # Only this polling path drains samples. Heartbeat/manual status calls
        # read cached metadata and cannot steal points before manager fitting.
        indices = [i for i in self.roofline if self.replicas[i].owned]
        packets = await asyncio.gather(
            *[
                asyncio.wait_for(
                    self.replicas[index].server.trajectory_roofline.remote(),
                    self.recovery_options.get("heartbeat_timeout_s", 10.0),
                )
                for index in indices
            ],
            return_exceptions=True,
        )
        for index, packet in zip(indices, packets, strict=True):
            telemetry[index]["roofline"] = (
                packet
                if not isinstance(packet, BaseException)
                else {
                    "available": False,
                    "samples": [],
                    "errors": [repr(packet)],
                }
            )
        async with self.roofline_lock:
            for index in indices:
                estimator = self.roofline[index]
                observations = telemetry[index].get("roofline", {})
                samples = observations.get("samples", []) if observations.get("available", False) else []
                before = estimator.accepted
                estimator.add(samples)
                if estimator.accepted > before:
                    self.roofline_dirty.add(index)
                now = time.monotonic()
                if (
                    index in self.roofline_dirty
                    and now - self.roofline_last_fit.get(index, -math.inf) >= estimator.config.fit_interval_s
                ):
                    status = await asyncio.to_thread(estimator.fit)
                    self.roofline_last_fit[index] = time.monotonic()
                    self.roofline_dirty.discard(index)
                    self._event("roofline_update", replica=index, **status)

    def _admission_limit(self, index):
        limit = self.capacity[index].limit() if index in self.capacity else min(self.batch_quota, self.max_running)
        ceiling = self.replicas[index].admission_ceiling if index < len(self.replicas) else None
        return min(limit, ceiling) if ceiling is not None else limit

    def _repack_limit(self, index, telemetry, context_tokens=None):
        fallback = min(self.max_running, self.repack_max_running)
        ceiling = self.replicas[index].admission_ceiling if index < len(self.replicas) else None
        if ceiling is not None:
            fallback = min(fallback, ceiling)
        if index in self.capacity:
            fallback = min(fallback, self.capacity[index].limit())
        estimator = self.roofline.get(index)
        if estimator is None:
            return fallback
        observations = telemetry[index].get("roofline", {})
        if not observations.get("available", False):
            return fallback
        context = context_tokens if context_tokens is not None else observations.get("max_context_tokens", 0)
        return estimator.limit(observations.get("mode", "unknown"), context, fallback)

    async def repack_once(self, perform_moves=True):
        """Read backend KV telemetry, reserve destinations, then cancel sources."""
        self.metrics["repack_checks"] += int(perform_moves)
        self.metrics["roofline_polls"] += int(bool(self.roofline))
        telemetry = await asyncio.gather(
            *[
                asyncio.wait_for(
                    r.server.trajectory_status.remote(), self.recovery_options.get("heartbeat_timeout_s", 10.0)
                )
                if r.owned
                else asyncio.sleep(
                    0, result={"kv_usage": 0.0, "waiting": 0, "request_ids": [], "observed_at": time.time()}
                )
                for r in self.replicas
            ],
            return_exceptions=self.partial_store is not None,
        )
        if self.partial_store is not None:
            for index, data in enumerate(telemetry):
                if isinstance(data, BaseException):
                    self._quarantine(index, data)
                    telemetry[index] = {"kv_usage": 0.0, "waiting": 0, "request_ids": [], "observed_at": time.time()}

        await self._ingest_roofline(telemetry)
        for index, governor in self.capacity.items():
            if not self.replicas[index].owned:
                continue
            if governor.update(telemetry[index], self.roofline.get(index), time.time()):
                self._event("capacity_update", replica=index, **governor.status())
            replica = self.replicas[index]
            if (
                governor.token_budget_enabled()
                and not replica.load
                and not replica.pulling
                and not telemetry[index].get("request_ids")
                and telemetry[index].get("version", replica.version) == replica.version
                and not governor.fresh(time.time())
            ):
                governor.reset_observation(telemetry[index].get("roofline", {}).get("capacity"))
            for request in replica.active.values():
                generated = governor.generated_tokens.get(request.backend_id, 0)
                request.generated_tokens = max(request.generated_tokens, request.generated_base + generated)
            was_paused = governor.pressure_paused
            governor.update_pressure(self._admission_projection(index), time.time())
            if governor.pressure_paused != was_paused:
                self._event(
                    "kv_pressure_admission",
                    replica=index,
                    paused=governor.pressure_paused,
                    kv_usage=governor.observed_usage,
                    predicted_kv_usage=self._reserved_usage_status(index),
                )
            continuous = self._continuous_admission(index)
            if continuous and replica.version == self.latest:
                replica.ramp_down = False
            # Exploration only tops up the current version's batch. A newer
            # actor version, migration reservations or drain operations win.
            if (
                replica.owned
                and not replica.releasing
                and replica.healthy
                and not replica.recovering
                and not replica.pulling
                and not replica.reserved
                and not any(q.destination is not None for q in replica.active.values())
                and replica.version == self.latest
                and (continuous or replica.admitted < governor.limit())
                and not replica.ramp_down
                and not governor.pressure_paused
                and (continuous or telemetry[index].get("kv_usage", 0) >= replica.previous_usage)
            ):
                replica.closed = False
            if governor.pressure_paused:
                replica.closed = True
        if self.capacity:
            async with self.condition:
                self.condition.notify_all()
        self.last_telemetry = [
            {
                "replica": index,
                "owned": replica.owned,
                "releasing": replica.releasing,
                "version": replica.version,
                "active": len(replica.active),
                "reserved": len(replica.reserved),
                "closed": replica.closed,
                "pulling": replica.pulling,
                "kv_usage": data["kv_usage"],
                "previous_kv_usage": replica.previous_usage,
                "waiting": data["waiting"],
                "observed_at": data["observed_at"],
                "repack_limit": self._repack_limit(index, telemetry),
                "admission_limit": self._admission_limit(index),
                "capacity": self.capacity[index].status() if index in self.capacity else {"enabled": False},
                "predicted_kv_usage": self._reserved_usage_status(index),
                "roofline": {k: v for k, v in data.get("roofline", {}).items() if k != "samples"},
            }
            for index, (replica, data) in enumerate(zip(self.replicas, telemetry, strict=True))
        ]
        moves = self._reserve_pressure_moves(telemetry) if perform_moves else []
        groups = {}
        for index, (replica, data) in enumerate(zip(self.replicas, telemetry, strict=True)):
            usage = data["kv_usage"]
            previous, replica.previous_usage = replica.previous_usage, usage
            capacity_limit = self.capacity[index].config.kv_usage_limit if index in self.capacity else 0.9
            if data["waiting"] or usage >= capacity_limit:
                replica.ramp_down = False
            elif 0 < usage < previous:
                # Remember the observed drain phase until the batch boundary.
                # Completions on different replicas need not coincide in the
                # same five-second polling window to permit consolidation.
                replica.ramp_down = True
            self.last_telemetry[index]["ramp_down"] = replica.ramp_down
            # Stale metrics and pre-engine requests cannot authorize migration.
            ids = set(data["request_ids"])
            ready = all(q.backend_id in ids and q.destination is None for q in replica.active.values())
            if (
                replica.owned
                and replica.healthy
                and not replica.recovering
                and replica.closed
                and not (self._continuous_admission(index) and replica.version == self.latest)
                and not replica.pulling
                and not replica.reserved
                and ready
                and 0 < replica.load < self.max_running
                and data["waiting"] == 0
                and replica.ramp_down
                and not (index in self.capacity and self.capacity[index].pressure_paused)
                and 0 < usage < capacity_limit
                and time.time() - data["observed_at"] < 2 * self.check_interval
            ):
                groups.setdefault(replica.version, []).append(
                    {
                        "id": index,
                        "usage": usage,
                        "count": replica.load,
                        "context_tokens": data.get("roofline", {}).get("max_context_tokens", 0),
                    }
                )
        self._event("replica_snapshot", latest=self.latest, replicas=self.last_telemetry)
        if not perform_moves:
            return 0  # Keep identical calibration telemetry in repack-off A/B runs.
        # No await during reservations: admission/completion cannot interleave.
        for group in groups.values():
            if all(self._continuous_admission(row["id"]) for row in group):
                plan = self._plan_request_consolidation(group, telemetry)
                for source, target, request in plan:
                    self.replicas[target].closed = True
                    request.destination = target
                    self.replicas[target].reserved.add(request.request_id)
                    moves.append((source, request, request.backend_id))
                continue
            for source, target in consolidation_plan(
                group,
                self.repack_max_running,
                request_limit=lambda target, context: 0
                if self.replicas[target].releasing
                else self._repack_limit(target, telemetry, context),
                capacity=next(iter(self.capacity.values())).config.kv_usage_limit if self.capacity else 0.9,
                usage_projection=(lambda target, usage, count: self.capacity[target].projected_usage(usage, count))
                if self.capacity
                else None,
            ):
                destination = self.replicas[target]
                destination.closed = True
                for request in list(self.replicas[source].active.values()):
                    request.destination = target
                    destination.reserved.add(request.request_id)
                    moves.append((source, request, request.backend_id))
        return await self._execute_moves(moves)

    def _reserve_pressure_moves(self, telemetry):
        """Reserve bounded partial evacuations before any abort RPC is issued.

        Unlike consolidation this does not require a draining/closed batch.
        A pressured instance may have no safe peer: it stays admission-paused.
        """
        now = time.time()
        moves = []
        for source, governor in self.capacity.items():
            src = self.replicas[source]
            if not (
                governor.config.rolling_kv_enabled
                and governor.pressure_paused
                and governor.fresh(now)
                and src.owned
                and src.healthy
                and not src.pulling
                and not src.recovering
                and not src.reserved
            ):
                continue
            projected = self._admission_projection(source)
            if projected is None or not math.isfinite(projected):
                continue
            actual = governor.observed_usage
            moved = 0
            for request in sorted(
                src.active.values(), key=lambda q: q.prompt_tokens + q.generated_tokens, reverse=True
            ):
                if (
                    request.destination is not None
                    or request.backend_id not in governor.backend_ids
                    or now - getattr(request, "last_pressure_migration", -math.inf)
                    < governor.config.rolling_kv_migration_cooldown_s
                ):
                    continue
                fits = []
                for target, peer in self.capacity.items():
                    if (
                        target == source
                        or not peer.config.rolling_kv_enabled
                        or peer.pressure_paused
                        or not peer.fresh(now)
                        or not self._migration_target_available(target, src.version)
                    ):
                        continue
                    dst = self.replicas[target]
                    context = max(
                        telemetry[source].get("roofline", {}).get("max_context_tokens", 0),
                        telemetry[target].get("roofline", {}).get("max_context_tokens", 0),
                    )
                    if dst.load + 1 > min(
                        self._admission_limit(target), self._repack_limit(target, telemetry, context)
                    ):
                        continue
                    base = self._admission_projection(target)
                    cost = peer.request_reserve(request, now, incoming=True)
                    if (
                        base is not None
                        and math.isfinite(base)
                        and base + cost < peer.config.rolling_kv_pressure_threshold
                    ):
                        fits.append((base + cost, target))
                if not fits:
                    continue
                target_usage, target = min(fits)
                self.replicas[target].closed = True
                self.replicas[target].reserved.add(request.request_id)
                request.destination = target
                request.last_pressure_migration = now
                moves.append((source, request, request.backend_id))
                # Only credit prefix pages already visible on the source, never
                # forecasted completions. Source remains paused until telemetry
                # confirms the actual reclamation on a subsequent poll.
                visible = governor.generated_tokens.get(request.backend_id, 0)
                released = governor.fraction_for_tokens(request.prompt_tokens + request.generated_base + visible)
                projected -= released + governor.request_reserve(request, now)
                actual -= released
                moved += 1
                self._event(
                    "pressure_migration_reserved",
                    source=source,
                    destination=target,
                    request=request.request_id,
                    version=request.version,
                    source_kv=governor.observed_usage,
                    target_projected_kv=target_usage,
                )
                if moved >= governor.config.rolling_kv_max_moves or (
                    actual <= governor.config.rolling_kv_resume_threshold
                    and projected < governor.config.rolling_kv_pressure_threshold
                ):
                    break
        return moves

    def _plan_request_consolidation(self, group, telemetry):
        """Drain complete sources across same-version targets using request KV budgets.

        Plan against a private ledger first: a source that cannot be fully drained
        contributes no migrations or reservations. Recipients never become sources
        in the same round, and every target includes all earlier planned arrivals.
        """
        now = time.time()
        rows = {row["id"]: row for row in group}
        arrivals = {index: [] for index in rows}
        contexts = {index: row["context_tokens"] for index, row in rows.items()}
        emptied, recipients, plan = set(), set(), []
        for source in sorted(rows, key=lambda index: (rows[index]["count"], rows[index]["usage"])):
            if source in emptied or source in recipients:
                continue
            requests = list(self.replicas[source].active.values())
            # Place expensive prefixes first to reduce fragmentation.
            requests.sort(key=lambda q: q.prompt_tokens + q.generated_tokens, reverse=True)
            trial = {index: list(pending) for index, pending in arrivals.items()}
            trial_contexts = dict(contexts)
            assignments = []
            for request in requests:
                fits = []
                for target in rows:
                    if target == source or target in emptied or self.replicas[target].releasing:
                        continue
                    if self.replicas[target].version != self.replicas[source].version:
                        continue
                    governor = self.capacity[target]
                    context = max(trial_contexts[target], contexts[source])
                    count = self.replicas[target].load + len(trial[target]) + 1
                    if count > self._repack_limit(target, telemetry, context):
                        continue
                    base = self._admission_projection(target)
                    if base is None or not math.isfinite(base):
                        continue
                    projected = base + sum(
                        governor.request_reserve(q, now, incoming=True) for q in trial[target] + [request]
                    )
                    if projected <= governor.config.kv_usage_limit:
                        fits.append((projected, target, context))
                if not fits:
                    break
                _, target, context = max(fits)
                trial[target].append(request)
                trial_contexts[target] = context
                assignments.append((source, target, request))
            if len(assignments) != len(requests) or not assignments:
                continue
            arrivals, contexts = trial, trial_contexts
            emptied.add(source)
            recipients.update(target for _, target, _ in assignments)
            plan.extend(assignments)
        return plan

    async def _execute_moves(self, moves):
        for source, request, backend_id in moves:
            try:
                result = await self.replicas[source].server.abort_trajectory.remote(backend_id)
            except Exception as error:
                if self.partial_store is not None and self._recoverable(error):
                    self._quarantine(source, error)
                    continue  # generate() owns fencing and target reservation cleanup.
                raise
            if not result["aborted"] and request.slot == source and request.destination is not None:
                self.replicas[request.destination].reserved.discard(request.request_id)
                request.destination = None
        return len(moves)

    async def migrate(self, source, target):
        """Capacity-checked migration, also used by the real-backend preflight.

        Manual invocation bypasses only ramp-down detection, never version,
        request-readiness, capacity or reservation checks.
        """
        self._check()
        if source == target or not (0 <= source < len(self.replicas) and 0 <= target < len(self.replicas)):
            raise ValueError("Migration requires distinct valid source/destination replicas")
        src, dst = self.replicas[source], self.replicas[target]
        if not src.owned or not dst.owned or dst.releasing:
            raise ValueError("Migration requires owned source and destination replicas")
        before = await asyncio.gather(src.server.trajectory_status.remote(), dst.server.trajectory_status.remote())
        if (
            not src.owned
            or not dst.owned
            or dst.releasing
            or not src.healthy
            or not dst.healthy
            or src.version != dst.version
            or src.version < 0
            or src.pulling
            or dst.pulling
            or src.reserved
            or dst.reserved
            or not src.active
        ):
            raise ValueError("Migration requires ready same-version replicas with no pending reservations")
        context = max(s.get("roofline", {}).get("max_context_tokens", 0) for s in before)
        data = {target: before[1]}
        projected = sum(s["kv_usage"] for s in before)
        margin = 0.9
        if target in self.capacity:
            governor = self.capacity[target]
            projected = governor.projected_usage(before[1]["kv_usage"], src.load)
            margin = governor.config.kv_usage_limit
        if src.load + dst.load > self._repack_limit(target, data, context) or projected > margin:
            raise ValueError("Migration exceeds destination request/KV capacity")
        if any(q.backend_id not in before[0]["request_ids"] or q.destination is not None for q in src.active.values()):
            raise ValueError("Source requests are not ready for migration")
        moves = []
        src.closed = dst.closed = True
        for request in src.active.values():
            request.destination = target
            dst.reserved.add(request.request_id)
            moves.append((source, request, request.backend_id))
        self._event("manual_repack", source=source, destination=target, version=src.version, requests=len(moves))
        return await self._execute_moves(moves)

    async def migration_probe(self, prompt_ids):
        """Check real token/logprob continuation before admitting training data."""
        if any(r.load for r in self.replicas) or len(self.replicas) < 2:
            raise RuntimeError("Migration preflight requires at least two idle replicas")
        sampling = {
            "max_tokens": 256,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": True,
            "ignore_eos": True,
        }

        # Compile/warm both independent engines before timing their overlap.
        # A cold engine can still be in prefill when the other has generated
        # all 256 tokens; that says nothing about migration correctness.
        async def warm(replica):
            output = await replica.server.generate.remote(
                request_id=f"preflight-warmup-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params={**sampling, "max_tokens": 32},
            )
            if (
                output.extra_fields.get("global_steps") != replica.version
                or len(output.token_ids) != 32
                or output.log_probs is None
                or len(output.log_probs) != 32
                or not all(math.isfinite(value) for value in output.log_probs)
            ):
                raise RuntimeError("Migration preflight warmup returned invalid tokens, logprobs or policy version")

        warmups = [asyncio.create_task(warm(replica)) for replica in self.replicas[:2]]
        try:
            async with asyncio.timeout(120):
                await asyncio.gather(*warmups)
        finally:
            for task in warmups:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*warmups, return_exceptions=True)
        self._event("migration_probe_warmup_passed", responses=2, tokens=64)
        requests = [
            asyncio.create_task(self.generate(f"preflight-{uuid4().hex}", prompt_ids, sampling)) for _ in range(2)
        ]
        started = time.monotonic()
        try:
            async with asyncio.timeout(120):
                while True:
                    progress = await asyncio.gather(*[r.server.trajectory_status.remote() for r in self.replicas[:2]])
                    generated = [list(s.get("generated_tokens", {}).values()) for s in progress]
                    if all(values and max(values) >= 16 for values in generated):
                        break
                    for task in requests:
                        if task.done():
                            output = await task  # Preserve the backend's original exception, if any.
                            self._event(
                                "migration_probe_no_overlap",
                                progress=progress,
                                completed_tokens=len(output.token_ids),
                            )
                            raise RuntimeError("Preflight request ended before migration could be tested")
                    await asyncio.sleep(0.1)
                if await self.migrate(0, 1) != 1:
                    raise RuntimeError("Preflight did not reserve exactly one running trajectory")
                outputs = await asyncio.gather(*requests)
                if sorted(output.extra_fields["migrations"] for output in outputs) != [0, 1]:
                    raise RuntimeError("Preflight did not complete a migrated response")
                if any(len(output.token_ids) != 256 or len(output.log_probs) != 256 for output in outputs):
                    raise RuntimeError("Preflight lost tokens or behavior logprobs")
                report = {
                    "seconds": time.monotonic() - started,
                    "responses": 2,
                    "tokens": 512,
                    "version": outputs[0].extra_fields["global_steps"],
                }
                self._event("migration_probe_passed", **report)
                return report
        finally:
            for request in requests:
                if not request.done():
                    request.cancel()
            await asyncio.gather(*requests, return_exceptions=True)

    async def _load_idle_replica(self, index):
        try:
            await self._pull(index)
        finally:
            self.scheduled_pulls.discard(index)
            self.wakeup.set()

    async def _run(self):
        last_repack = time.monotonic()
        try:
            while not self.stopped:
                try:
                    timeout = max(0.001, self.check_interval - (time.monotonic() - last_repack))
                    await asyncio.wait_for(self.wakeup.wait(), timeout=timeout)
                except TimeoutError:
                    pass
                self.wakeup.clear()
                if self.stopped:
                    break
                self._check()
                if any(replica.relay is not None for replica in self.replicas):
                    await self.relay.health.remote()
                for index, replica in enumerate(self.replicas):
                    if (
                        not replica.owned
                        or replica.releasing
                        or not replica.healthy
                        or replica.recovering
                        or replica.load
                        or replica.pulling
                    ):
                        continue
                    if replica.version < self.latest:
                        self.scheduled_pulls.add(index)
                        task = asyncio.create_task(self._load_idle_replica(index))
                        self.pulls.add(task)
                        task.add_done_callback(self.pulls.discard)
                        task.add_done_callback(self._pull_done)
                    else:
                        replica.admitted = 0
                        replica.closed = index in self.capacity and self.capacity[index].pressure_paused
                        replica.ramp_down = False
                await self._collect_versions()
                async with self.condition:
                    self.condition.notify_all()
                if (self.enable_repack or self.roofline) and (
                    self.repack_requested or time.monotonic() - last_repack >= self.check_interval
                ):
                    self.repack_requested = False
                    await self.repack_once(perform_moves=self.enable_repack)
                    last_repack = time.monotonic()
                elif not self.enable_repack and not self.roofline:
                    last_repack = time.monotonic()
        except Exception as error:
            self.failure = repr(error)
            self._event("failure", error=self.failure)
            async with self.condition:
                self.condition.notify_all()

    def _pull_done(self, task):
        if not task.cancelled() and task.exception() is not None:
            if self.partial_store is None or not self._recoverable(task.exception()):
                self.failure = repr(task.exception())
            self.wakeup.set()

    def status(self):
        self._check()
        return {
            "latest": self.latest,
            **self.metrics,
            "pending": len(self.pending),
            "supply_paused": self.supply_paused,
            "repack_max_running": self.repack_max_running,
            "capacity": {i: governor.status() for i, governor in self.capacity.items()},
            "roofline": {i: estimator.status() for i, estimator in self.roofline.items()},
            "replicas": [
                {
                    "version": r.version,
                    "owned": r.owned,
                    "admission_limit": self._admission_limit(i),
                    "predicted_kv_usage": self._reserved_usage_status(i),
                    "releasing": r.releasing,
                    "healthy": r.healthy,
                    "active": len(r.active),
                    "reserved": len(r.reserved),
                    "pulling": r.pulling,
                    "closed": r.closed,
                }
                for i, r in enumerate(self.replicas)
            ],
        }

    async def stop(self):
        self.stopped = True
        self.wakeup.set()
        current = asyncio.current_task()
        requests = [task for task in self.request_tasks.values() if task is not current]
        for request in requests:
            request.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        if self.task:
            await self.task
        await asyncio.gather(*self.pulls, return_exceptions=True)
        async with self.condition:
            self.condition.notify_all()


class LaminarServerClient(LLMServerClient):
    def __init__(self, config, controller, policy_version=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.controller = controller
        self.policy_version = policy_version

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        if self.policy_version is not None:
            params["_laminar_policy_version"] = self.policy_version
        if "max_tokens" not in params and "max_new_tokens" not in params:
            params["max_tokens"] = self.config.actor_rollout_ref.rollout.response_length
        logical_id = params.pop("_laminar_trajectory_id", None) or f"{request_id}-{uuid4().hex}"
        return await self.controller.generate.remote(
            request_id=logical_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
            **kwargs,
        )
