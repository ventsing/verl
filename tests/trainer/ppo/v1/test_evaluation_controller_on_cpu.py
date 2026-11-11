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

"""Exact-version admission and migration through the real evaluation controller."""

import asyncio
import time

import pytest

from verl.experimental.trajectory_async.evaluation_controller import EvaluationController, evaluation_allocation
from verl.workers.rollout.replica import TokenOutput


class Handle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            value = getattr(self.target, name)(*args, **kwargs)
            return await value if asyncio.iscoroutine(value) else value

        return type("Method", (), {"remote": staticmethod(call)})()


class Relay:
    def __init__(self):
        self.leases = set()

    def acquire(self, version):
        lease = str(time.monotonic_ns())
        self.leases.add(lease)
        return {"version": version, "lease": lease}

    def release(self, lease):
        self.leases.remove(lease)

    def health(self):
        pass


class Server:
    def __init__(self):
        self.version = -1
        self.requests = {}

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        version = self.version
        future = asyncio.get_running_loop().create_future()
        self.requests[request_id] = future
        try:
            aborted = await future
            return TokenOutput(
                token_ids=[prompt_ids[-1] + 1],
                log_probs=[-0.2],
                stop_reason="aborted" if aborted else "completed",
                extra_fields={"global_steps": version},
            )
        finally:
            self.requests.pop(request_id, None)

    def trajectory_status(self):
        return {
            "request_ids": list(self.requests),
            "kv_usage": 0.02 * len(self.requests),
            "waiting": 0,
            "observed_at": time.time(),
        }

    def abort_trajectory(self, request_id):
        future = self.requests.get(request_id)
        if future is None or future.done():
            return {"aborted": False}
        future.set_result(True)
        return {"aborted": True}

    def finish(self):
        for future in list(self.requests.values()):
            if not future.done():
                future.set_result(False)

    def wait_for_requests_to_drain(self):
        assert not self.requests

    def execute_checkpoint_engine(self, method, manifest=None):
        if method == "set_manifest":
            self.manifest = manifest
        elif method == "finalize":
            self.manifest = None
        else:
            raise AssertionError(method)

    def update_weights(self, global_steps):
        assert not self.requests and self.manifest["version"] == global_steps
        self.version = global_steps


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def test_replica_allocation_is_fair_and_does_not_reserve_idle_work():
    assert evaluation_allocation({1: 512, 2: 512, 3: 512}, 8) == {1: 3, 2: 3, 3: 2}
    assert evaluation_allocation({1: 0, 2: 1}, 8) == {2: 1}


def test_finish_first_allocates_spare_replicas_to_nearest_completion():
    assert evaluation_allocation({1: 126, 2: 293, 3: 403}, 8, "finish_first") == {1: 6, 2: 1, 3: 1}
    assert evaluation_allocation({1: 1, 2: 293, 3: 403}, 8, "finish_first") == {1: 1, 2: 6, 3: 1}
    assert evaluation_allocation({1: 0, 2: 1}, 8, "finish_first") == {2: 1}


@pytest.mark.parametrize("migrate", [False, True])
def test_versions_overlap_and_migration_never_changes_policy(migrate):
    async def scenario():
        servers = [Server() for _ in range(2 if migrate else 4)]
        relay = Relay()
        controller = EvaluationController(
            [{"server": Handle(s), "workers": [Handle(s)], "relay": Handle(relay)} for s in servers],
            Handle(relay),
            batch_quota=2,
            max_running=2,
            check_interval_s=0.02,
            enable_repack=migrate,
        )
        tasks = []
        try:
            await controller.register_version(10)
            tasks = [
                asyncio.create_task(
                    controller.generate(f"old-{i}", [1], {"max_tokens": 2, "_laminar_policy_version": 10})
                )
                for i in range(2)
            ]
            await until(lambda: sum(len(s.requests) for s in servers) == 2)
            if migrate:
                # Both old replicas have reached a telemetry-confirmed ramp-down;
                # their two tails fit safely in one destination.
                for replica in controller.replicas:
                    replica.previous_usage = 0.5
                    replica.ramp_down = True
            await controller.register_version(11)
            tasks.extend(
                asyncio.create_task(
                    controller.generate(f"new-{i}", [5], {"max_tokens": 2, "_laminar_policy_version": 11})
                )
                for i in range(2)
            )
            await until(lambda: {s.version for s in servers if s.requests} == {10, 11})
            assert not tasks[0].done() and not tasks[1].done()
            if migrate:
                assert controller.metrics["migrations"] >= 1
            # An older version remains admissible even after version 11 exists.
            if not migrate:
                tasks.append(
                    asyncio.create_task(
                        controller.generate("older-late", [9], {"max_tokens": 2, "_laminar_policy_version": 10})
                    )
                )
                await until(lambda: sum(len(s.requests) for s in servers) == 5)
            for server in servers:
                server.finish()
            outputs = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            assert [o.extra_fields["global_steps"] for o in outputs] == ([10, 10, 11, 11] + ([] if migrate else [10]))
            assert all(o.extra_fields["min_global_steps"] == o.extra_fields["max_global_steps"] for o in outputs)
            await controller.unregister_version(10)
            await controller.unregister_version(11)
            assert not relay.leases
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await controller.stop()

    asyncio.run(scenario())
