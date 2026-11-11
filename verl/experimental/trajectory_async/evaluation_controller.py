# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Share evaluation replicas across immutable versions, with same-version repacking."""

import asyncio
import time
from collections import Counter

from verl.experimental.trajectory_async.laminar_controller import LaminarController, Request


def evaluation_allocation(demands, slots, policy="fair"):
    """Allocate work without idling replicas or starving a resident version."""
    if policy not in {"fair", "finish_first"}:
        raise ValueError(f"Unknown evaluation allocation policy: {policy}")
    allocation = {version: 0 for version, count in sorted(demands.items()) if count > 0}
    if policy == "finish_first":
        ordered = sorted(allocation, key=lambda version: (demands[version], version))
        for version in ordered[:slots]:
            allocation[version] = 1
            slots -= 1
        for version in ordered:
            extra = min(slots, demands[version] - allocation[version])
            allocation[version] += extra
            slots -= extra
        return allocation
    while slots > 0:
        eligible = [v for v in allocation if allocation[v] < demands[v]]
        if not eligible:
            break
        for version in eligible:
            if slots == 0:
                break
            allocation[version] += 1
            slots -= 1
    return allocation


class EvaluationController(LaminarController):
    """Exact-version admission and elastic version cohorts on one evaluation pool.

    Idle replicas change versions independently. Surplus replicas drain while
    busy cohorts continue serving; the inherited telemetry/capacity checked
    repacker can move tails only within the same version. No request is routed
    to a newer policy merely because training published another snapshot.
    """

    def __init__(self, *args, allocation_policy="finish_first", **kwargs):
        super().__init__(*args, **kwargs)
        if allocation_policy not in {"finish_first", "fair"}:
            raise ValueError(f"Unknown evaluation allocation policy: {allocation_policy}")
        self.allocation_policy = allocation_policy
        self.versions = set()
        self.request_versions = {}
        self.loading_versions = {}
        self.draining = set()
        self._last_targets = None

    async def register_version(self, version):
        self._check()
        if version in self.versions:
            raise ValueError(f"Evaluation version already registered: {version}")
        self.versions.add(version)
        self.latest = max(self.versions)
        self._event("evaluation_registered", version=version)
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        self.wakeup.set()

    async def unregister_version(self, version):
        if version in self.request_versions.values():
            raise RuntimeError(f"Evaluation version {version} still has requests")
        self.versions.discard(version)
        self._event("evaluation_finished", version=version)
        self.wakeup.set()

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        params = dict(sampling_params)
        version = params.pop("_laminar_policy_version", None)
        if version not in self.versions:
            raise ValueError(f"Unregistered evaluation policy version: {version}")
        if request_id in self.request_versions:
            raise ValueError(f"Duplicate evaluation request: {request_id}")
        self.request_versions[request_id] = version
        params["_laminar_validate"] = True
        self.wakeup.set()
        try:
            return await super().generate(request_id, prompt_ids, params, **kwargs)
        finally:
            self.request_versions.pop(request_id, None)
            self.wakeup.set()

    async def _assign(self, request_id, is_validation=False):
        version = self.request_versions[request_id]
        async with self.condition:
            while True:
                self._check()
                available = [
                    i
                    for i, r in enumerate(self.replicas)
                    if r.owned
                    and not r.releasing
                    and r.healthy
                    and not r.recovering
                    and not r.pulling
                    and i not in self.loading_versions
                    and i not in self.draining
                    and not r.closed
                    and r.version == version
                    and r.load < self._admission_limit(i)
                ]
                if available:
                    index = min(available, key=lambda i: self.replicas[i].load)
                    request = Request(request_id, index, version)
                    self.replicas[index].active[request_id] = request
                    return request
                await self.condition.wait()

    def _migration_target_available(self, index, version):
        return (
            super()._migration_target_available(index, version)
            and index not in self.loading_versions
            and index not in self.draining
        )

    async def release_replica(self, index):
        # A version-load task is reserved before its coroutine begins running.
        # Do not acknowledge a handoff while that task can still touch the engine.
        if index in self.loading_versions:
            self.replicas[index].releasing = True
            self.replicas[index].closed = True
            self.wakeup.set()
            return False
        return await super().release_replica(index)

    async def _load_version(self, index, version):
        try:
            await super()._pull(index, exact_version=version)
        finally:
            self.loading_versions.pop(index, None)
            self.wakeup.set()

    def _rebalance(self):
        demands = Counter(self.request_versions.values())
        targets = evaluation_allocation(
            demands, sum(r.owned and not r.releasing and r.healthy for r in self.replicas), self.allocation_policy
        )
        if targets != self._last_targets:
            self._event("evaluation_allocation", targets=targets, demands=dict(demands))
            self._last_targets = targets.copy()
        groups = {}
        for index, replica in enumerate(self.replicas):
            if replica.owned and not replica.releasing and replica.healthy:
                version = self.loading_versions.get(index, replica.version)
                groups.setdefault(version, []).append(index)
        self.draining = set()
        for version, indexes in groups.items():
            # Keep the busiest replicas; drain low-load tails first.
            indexes.sort(key=lambda i: (self.replicas[i].load, i in self.loading_versions), reverse=True)
            self.draining.update(indexes[targets.get(version, 0) :])
        idle = [i for i in sorted(self.draining) if not self.replicas[i].load and i not in self.loading_versions]
        for version, target in targets.items():
            assigned = sum(i not in self.draining for i in groups.get(version, []))
            for _ in range(target - assigned):
                if not idle:
                    break
                index = idle.pop(0)
                self.loading_versions[index] = version
                self.replicas[index].closed = True
                task = asyncio.create_task(self._load_version(index, version))
                self.pulls.add(task)
                task.add_done_callback(self.pulls.discard)
                task.add_done_callback(self._pull_done)
        for index, replica in enumerate(self.replicas):
            replica.closed = (
                not replica.owned
                or replica.releasing
                or index in self.draining
                or index in self.loading_versions
                or any(request.destination is not None for request in replica.active.values())
            )

    async def _run(self):
        last_repack = time.monotonic()
        try:
            while not self.stopped:
                try:
                    await asyncio.wait_for(self.wakeup.wait(), timeout=self.check_interval)
                except TimeoutError:
                    pass
                self.wakeup.clear()
                if self.stopped:
                    break
                self._check()
                self._rebalance()
                if self.enable_repack and time.monotonic() - last_repack >= self.check_interval:
                    # The base repacker needs a closed batch boundary and fresh
                    # ramp-down telemetry on both source and destination. Briefly
                    # fence admission while checking; its version/KV limits and
                    # atomic destination reservations remain unchanged.
                    for replica in self.replicas:
                        replica.closed = True
                    await self.repack_once()
                    self._rebalance()
                    last_repack = time.monotonic()
                async with self.condition:
                    self.condition.notify_all()
        except Exception as error:
            self.failure = repr(error)
            self._event("failure", error=self.failure)
            async with self.condition:
                self.condition.notify_all()

    def status(self):
        return {
            **super().status(),
            "evaluation_versions": sorted(self.versions),
            "evaluation_requests": dict(Counter(self.request_versions.values())),
            "loading_versions": dict(self.loading_versions),
            "allocation_policy": self.allocation_policy,
        }
