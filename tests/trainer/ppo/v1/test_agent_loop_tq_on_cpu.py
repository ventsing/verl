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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_concurrent_validation_status_and_payload_use_version_partitions(monkeypatch):
    from types import SimpleNamespace

    import torch

    from verl.trainer.ppo.v1 import agent_loop_tq

    worker_class = agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class
    statuses, payloads = [], []

    async def put(**kwargs):
        statuses.append(kwargs)

    async def batch_put(**kwargs):
        payloads.append(kwargs)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_put", put)
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", batch_put)

    async def run():
        worker = SimpleNamespace(
            config=SimpleNamespace(
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=4, val_kwargs=SimpleNamespace(n=1)))
            ),
            _compute_score=noop,
            _compute_teacher_logprobs=noop,
            _compute_multi_modal_inputs=lambda output, ids: {},
            _compute_position_ids=lambda ids, mask, mm: torch.zeros_like(ids),
        )

        async def agent_loop(params, trajectory, trace, session_id, **prompt):
            version = prompt["global_steps"]
            output = SimpleNamespace(
                prompt_ids=[1, 2],
                response_ids=[3],
                reward_score=1.0,
                extra_fields={"min_global_steps": version, "max_global_steps": version},
            )
            output.as_dict = lambda: {
                "prompts": torch.tensor([1, 2]),
                "responses": torch.tensor([3]),
                "response_mask": torch.tensor([1]),
                "extra_fields": output.extra_fields,
            }
            await worker_class._agent_loop_postprocess(worker, output, True, session_id=session_id, **prompt)

        worker._run_agent_loop = agent_loop
        await asyncio.gather(
            *[
                worker_class._run_prompt(
                    worker,
                    {
                        "uid": f"version{version}",
                        "global_steps": version,
                        "__partition_id__": f"val-evaluation-{version}",
                    },
                    {},
                    {"validate": True},
                )
                for version in [1, 2]
            ]
        )

    asyncio.run(run())
    assert {row["partition_id"] for row in statuses} == {"val-evaluation-1", "val-evaluation-2"}
    assert all(row["tag"]["status"] in {"running", "finished"} for row in statuses)
    assert len(payloads) == 2
    for row in payloads:
        version = int(row["partition_id"].rsplit("-", 1)[-1])
        assert row["tags"][0]["min_global_steps"] == version
        assert row["tags"][0]["max_global_steps"] == version
