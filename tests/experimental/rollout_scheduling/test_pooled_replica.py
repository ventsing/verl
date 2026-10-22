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
"""Unit tests for P4b: pooled rollout replicas (design doc §10.4/§10.5).

No Ray cluster / GPUs / plugin install needed — Ray actor semantics are
faked with awaitable refs. Requires the normal verl test env (ray, vllm
importable) because the module under test imports them at module level.

Covered:

- ``check_pooled_prerequisites`` fail-fast matrix (name / dp / determinism /
  disaggregation / LoRA / plugin import / fork socket);
- ``PooledvLLMReplica`` lifecycle ordering: launch -> fill ->
  set_global_steps -> clear_kv_cache; fill failure raises and never tags;
  release path (frontend killed, workers reset, pool.release);
- ``PooledLLMServerManager``: topology math (devices vs actors),
  mandatory-fill validation, worker_group rejection;
- ``LLMServerManager.create`` FQN dispatch socket (config selects subclass);
- ``PooledvLLMHttpServer.run_server`` builds AsyncLLM with
  executor_class=ExternalExecutor + external_actors.
"""

import asyncio
import sys
import types

import pytest

from verl.experimental.rollout_scheduling import pooled_replica as pr


# ---------------------------------------------------------------------------
# Ray-semantics fakes
# ---------------------------------------------------------------------------
class _FakeObjectRef:
    """Awaitable stand-in for a ray ObjectRef."""

    def __init__(self, awaitable):
        self._awaitable = asyncio.ensure_future(awaitable)

    def __await__(self):
        return self._awaitable.__await__()


class _FakeActorMethod:
    def __init__(self, handler, *bound_args):
        self._handler = handler
        self._bound_args = bound_args

    def remote(self, *args, **kwargs):
        coro = self._handler(*self._bound_args, *args, **kwargs)
        return _FakeObjectRef(coro)


class FakeWorkerActor:
    """Fake ExternalWorkerActor handle recording every remote call."""

    def __init__(self, device_id: int, fill_ok: bool = True):
        self.device_id = device_id
        self.fill_ok = fill_ok
        self.calls: list[tuple[str, dict]] = []
        self.state = "IDLE"

    async def _load_model_from_storage(self, *, checkpoint_path, storage_backend, storage_config):
        self.calls.append(("load_model_from_storage", dict(checkpoint_path=checkpoint_path)))
        if not self.fill_ok:
            raise RuntimeError(f"storage pull failed on device {self.device_id}")
        self.state = "RUNNING"
        return True

    async def _reset(self):
        self.calls.append(("reset", {}))
        self.state = "IDLE"
        return True

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        handler = {
            "load_model_from_storage": self._load_model_from_storage,
            "reset": self._reset,
        }.get(name)
        if handler is None:
            raise AttributeError(name)
        return _FakeActorMethod(handler)


class FakeServerHandle:
    """Fake PooledvLLMHttpServer actor handle."""

    def __init__(self):
        self.killed = False
        self.global_steps = None
        self.kv_cache_cleared = 0
        self.remote_calls: list[tuple[str, dict]] = []

    async def _get_master_address(self):
        return ("127.0.0.1", 1234, 5678)

    async def _launch_server(self, **kwargs):
        return None

    async def _get_server_address(self):
        return ("127.0.0.1", 8000)

    async def _set_global_steps(self, global_steps):
        self.global_steps = global_steps

    async def _clear_kv_cache(self):
        self.kv_cache_cleared += 1

    def kill(self):
        self.killed = True

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        handler = {
            "get_master_address": self._get_master_address,
            "launch_server": self._launch_server,
            "get_server_address": self._get_server_address,
            "set_global_steps": self._set_global_steps,
            "clear_kv_cache": self._clear_kv_cache,
        }.get(name)
        if handler is None:
            raise AttributeError(name)
        return _FakeActorMethod(handler)


class FakeServerClass:
    """Stand-in for ``ray.remote(PooledvLLMHttpServer)``: options().remote()."""

    def __init__(self):
        self.instances: list[FakeServerHandle] = []

    def options(self, **kwargs):
        return self

    def remote(self, *args, **kwargs):
        handle = FakeServerHandle()
        self.instances.append(handle)
        return handle


class FakePool:
    """Fake ActorPoolManager: pre_start/acquire/release bookkeeping."""

    def __init__(self, actors=None):
        self.actors = list(actors or [])
        self.states = {i: "IDLE" for i in range(len(self.actors))}
        self.acquired: list[list] = []
        self.released: list[list] = []
        self.pre_started_with = None

    def pre_start(self, **kwargs):
        self.pre_started_with = kwargs

    def acquire(self, tp_size, pp_size):
        world = tp_size * pp_size
        batch = [a for i, a in enumerate(self.actors) if self.states[i] == "IDLE"][:world]
        if len(batch) != world:
            raise RuntimeError("not enough IDLE actors in pool")
        for a in batch:
            self.states[self.actors.index(a)] = "LEASED"
        self.acquired.append(batch)
        return batch

    def release(self, actors):
        self.released.append(list(actors))
        for a in actors:
            self.states[self.actors.index(a)] = "IDLE"


# ---------------------------------------------------------------------------
# Config fakes: plain dicts — omega_conf_to_dataclass turns them into real
# RolloutConfig instances inside RolloutReplica.__init__.
# ---------------------------------------------------------------------------
VALID_FILL = {
    "checkpoint_path": "model/step_100",
    "global_steps": 100,
    "storage_backend": "nfs",
    "storage_config": {"base_path": "/shared/ckpt"},
}


def rollout_cfg(**kw):
    cfg = dict(
        name="vllm",
        mode="async",
        data_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        full_determinism=False,
    )
    cfg.update(kw)
    return cfg


class _AttrDict(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def attr_cfg(**kw):
    """Attribute-access config (for check_pooled_prerequisites, which reads
    name/dp/full_determinism/disaggregation before any dataclass conversion)."""
    return _AttrDict(rollout_cfg(**kw))


# ---------------------------------------------------------------------------
# check_pooled_prerequisites fail-fast matrix
# ---------------------------------------------------------------------------
def test_prereq_rejects_non_vllm():
    with pytest.raises(NotImplementedError, match="vllm"):
        pr.check_pooled_prerequisites(attr_cfg(name="sglang"))


def test_prereq_rejects_dp_gt_1():
    with pytest.raises(NotImplementedError, match="data_parallel_size == 1"):
        pr.check_pooled_prerequisites(attr_cfg(data_parallel_size=2))


def test_prereq_rejects_full_determinism():
    with pytest.raises(NotImplementedError, match="full_determinism"):
        pr.check_pooled_prerequisites(attr_cfg(full_determinism=True))


def test_prereq_rejects_disaggregation():
    disagg = _AttrDict(enabled=True)
    with pytest.raises(NotImplementedError, match="PD-disaggregated"):
        pr.check_pooled_prerequisites(attr_cfg(disaggregation=disagg))


def test_prereq_rejects_lora():
    model_cfg = _AttrDict(lora=_AttrDict(rank=32))
    with pytest.raises(NotImplementedError, match="LoRA"):
        pr.check_pooled_prerequisites(attr_cfg(), model_config=model_cfg)


def test_prereq_plugin_import_error(monkeypatch):
    # None in sys.modules makes `import vllm_external_executor` raise.
    monkeypatch.setitem(sys.modules, "vllm_external_executor", None)
    with pytest.raises(ImportError, match="multi-task-infer plugin"):
        pr.check_pooled_prerequisites(attr_cfg())


def test_prereq_fork_missing_socket(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_external_executor", types.ModuleType("vllm_external_executor"))

    class StockAsyncLLM:  # PyPI vLLM: no external_actors parameter
        def __init__(self, vllm_config=None, log_stats=True):
            pass

    monkeypatch.setattr(pr, "AsyncLLM", StockAsyncLLM)
    with pytest.raises(TypeError, match="feature/external-executor"):
        pr.check_pooled_prerequisites(attr_cfg())


def test_prereq_passes_on_fork(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_external_executor", types.ModuleType("vllm_external_executor"))

    class ForkAsyncLLM:  # fork branch: external_actors socket present
        def __init__(self, vllm_config=None, log_stats=True, external_actors=None):
            pass

    monkeypatch.setattr(pr, "AsyncLLM", ForkAsyncLLM)
    pr.check_pooled_prerequisites(attr_cfg(), model_config=_AttrDict(lora=None))


# ---------------------------------------------------------------------------
# PooledvLLMReplica lifecycle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_init_pooled_ordering():
    """launch -> fill -> tag -> clear KV (red line §10.5: fill mandatory)."""
    replica = pr.PooledvLLMReplica(
        replica_rank=0,
        config=rollout_cfg(tensor_model_parallel_size=2),
        model_config={},
        gpus_per_node=2,
    )
    assert replica.world_size == 2  # tp*dp*pp

    server_cls = FakeServerClass()
    replica.server_class = server_cls
    actors = [FakeWorkerActor(0), FakeWorkerActor(1)]

    await replica.init_pooled(external_actors=actors, fill=dict(VALID_FILL))

    server = server_cls.instances[0]
    # exactly one frontend server actor
    assert len(server_cls.instances) == 1
    assert replica._server_handle is server
    assert replica._server_address == "127.0.0.1:8000"
    # every worker filled from the same checkpoint
    for a in actors:
        assert a.calls == [("load_model_from_storage", {"checkpoint_path": "model/step_100"})]
        assert a.state == "RUNNING"
    # version tagged, then KV cleared (never the reverse)
    assert server.global_steps == 100
    assert server.kv_cache_cleared == 1


@pytest.mark.asyncio
async def test_init_pooled_actor_count_mismatch():
    replica = pr.PooledvLLMReplica(
        replica_rank=0,
        config=rollout_cfg(tensor_model_parallel_size=2),
        model_config={},
        gpus_per_node=2,
    )
    with pytest.raises(ValueError, match="world_size=2"):
        await replica.init_pooled(external_actors=[FakeWorkerActor(0)], fill=dict(VALID_FILL))


@pytest.mark.asyncio
async def test_fill_failure_raises_and_never_tags():
    """allow_dummy_fallback=false: a failed fill raises; global_steps stays
    None so the replica can never be mistaken for weight-tagged."""
    replica = pr.PooledvLLMReplica(
        replica_rank=0,
        config=rollout_cfg(tensor_model_parallel_size=2),
        model_config={},
        gpus_per_node=2,
    )
    replica._server_handle = FakeServerHandle()
    replica.external_actors = [FakeWorkerActor(0, fill_ok=True), FakeWorkerActor(1, fill_ok=False)]

    with pytest.raises(RuntimeError, match="storage pull failed"):
        await replica.fill_weights_from_storage(**VALID_FILL)

    assert replica._server_handle.global_steps is None
    assert replica._server_handle.kv_cache_cleared == 0


@pytest.mark.asyncio
async def test_fill_requires_checkpoint_and_steps():
    replica = pr.PooledvLLMReplica(
        replica_rank=0, config=rollout_cfg(), model_config={}, gpus_per_node=1
    )
    replica._server_handle = FakeServerHandle()
    replica.external_actors = [FakeWorkerActor(0)]

    with pytest.raises(ValueError, match="checkpoint_path"):
        await replica.fill_weights_from_storage(checkpoint_path="", global_steps=5)
    with pytest.raises(ValueError, match="global_steps"):
        await replica.fill_weights_from_storage(checkpoint_path="model/step_1", global_steps="1")


@pytest.mark.asyncio
async def test_release_to_pool(monkeypatch):
    monkeypatch.setattr("ray.kill", lambda handle: handle.kill())
    replica = pr.PooledvLLMReplica(
        replica_rank=0,
        config=rollout_cfg(tensor_model_parallel_size=2),
        model_config={},
        gpus_per_node=2,
    )
    server = FakeServerHandle()
    replica._server_handle = server
    actors = [FakeWorkerActor(0), FakeWorkerActor(1)]
    replica.external_actors = actors
    pool = FakePool(actors)

    await replica.release_to_pool(pool)

    assert server.killed
    assert all(a.state == "IDLE" for a in actors)
    assert pool.released == [actors]
    assert replica.external_actors == []
    assert replica._server_handle is None
    # second release is a no-op (already detached)
    await replica.release_to_pool(pool)
    assert len(pool.released) == 1


# ---------------------------------------------------------------------------
# PooledLLMServerManager wiring
# ---------------------------------------------------------------------------
def _make_manager(pool: FakePool, shared_pool_cfg: dict, rollout_extra: dict | None = None):
    """Bypass __init__ (needs the full hydra config); wire only the state
    _initialize_llm_servers touches."""
    mgr = pr.PooledLLMServerManager.__new__(pr.PooledLLMServerManager)
    mgr.rollout_config = _AttrDict(
        rollout_cfg(
            shared_pool=shared_pool_cfg,
            **(rollout_extra or {}),
        )
    )
    mgr.model_config = _AttrDict(lora=None)
    mgr.start_rank = 0
    mgr.rollout_replica_class = None  # set per test
    mgr.actor_pool = None
    mgr._build_pool = lambda: pool
    return mgr


class _RecordingReplica:
    """Fake replica recording init_pooled args."""

    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kw):
        self.replica_rank = replica_rank
        self.gpus_per_node = gpus_per_node
        self._server_handle = FakeServerHandle()
        self._server_address = "127.0.0.1:8000"
        self.init_calls: list[dict] = []

    async def init_pooled(self, external_actors, fill):
        self.init_calls.append({"actors": list(external_actors), "fill": dict(fill)})

    async def release_to_pool(self, pool):
        self.init_calls.append({"released_to": pool})


@pytest.mark.asyncio
async def test_manager_leases_and_builds_replicas(monkeypatch):
    monkeypatch.setattr(pr, "check_pooled_prerequisites", lambda *a, **k: None)
    pool = FakePool([FakeWorkerActor(i) for i in range(4)])
    mgr = _make_manager(
        pool,
        {"num_replicas": 2, "devices": [0, 1, 2, 3], "fill": VALID_FILL},
        rollout_extra={"tensor_model_parallel_size": 2},
    )
    mgr.rollout_replica_class = _RecordingReplica

    await mgr._initialize_llm_servers()

    # pool pre-started with exactly num_replicas * tp * pp actors
    assert pool.pre_started_with["num_actors"] == 4
    assert pool.pre_started_with["devices_per_node"] == [0, 1, 2, 3]
    assert pool.pre_started_with["warmup_distributed"] is True
    # two leases of 2 actors each, disjoint
    assert len(pool.acquired) == 2
    assert all(len(batch) == 2 for batch in pool.acquired)
    assert not set(id(a) for a in pool.acquired[0]) & set(id(a) for a in pool.acquired[1])
    # each replica got the fill spec and its rank
    replicas = mgr.rollout_replicas
    assert [r.replica_rank for r in replicas] == [0, 1]
    for r in replicas:
        assert r.init_calls[0]["fill"]["checkpoint_path"] == "model/step_100"
        assert r.init_calls[0]["fill"]["global_steps"] == 100
    # LB-facing surface populated
    assert mgr.server_handles == [r._server_handle for r in replicas]
    assert mgr.server_addresses == [r._server_address for r in replicas]


@pytest.mark.asyncio
async def test_manager_rejects_missing_or_partial_fill(monkeypatch):
    monkeypatch.setattr(pr, "check_pooled_prerequisites", lambda *a, **k: None)
    for bad_fill in (None, {}, {"checkpoint_path": "x"}, {"global_steps": 5}):
        pool = FakePool()
        mgr = _make_manager(pool, {"num_replicas": 1, "fill": bad_fill})
        mgr.rollout_replica_class = _RecordingReplica
        with pytest.raises(ValueError, match="fill must define"):
            await mgr._initialize_llm_servers()


@pytest.mark.asyncio
async def test_manager_rejects_num_replicas_and_device_mismatch(monkeypatch):
    monkeypatch.setattr(pr, "check_pooled_prerequisites", lambda *a, **k: None)
    pool = FakePool()
    mgr = _make_manager(pool, {"fill": VALID_FILL})  # num_replicas missing
    mgr.rollout_replica_class = _RecordingReplica
    with pytest.raises(ValueError, match="num_replicas"):
        await mgr._initialize_llm_servers()

    pool2 = FakePool()
    mgr2 = _make_manager(
        pool2,
        {"num_replicas": 2, "devices": [0, 1], "fill": VALID_FILL},
        rollout_extra={"tensor_model_parallel_size": 2},
    )
    mgr2.rollout_replica_class = _RecordingReplica
    with pytest.raises(ValueError, match="devices"):
        await mgr2._initialize_llm_servers()


@pytest.mark.asyncio
async def test_manager_default_devices(monkeypatch):
    """devices defaults to range(num_actors)."""
    monkeypatch.setattr(pr, "check_pooled_prerequisites", lambda *a, **k: None)
    pool = FakePool([FakeWorkerActor(i) for i in range(2)])
    mgr = _make_manager(pool, {"num_replicas": 2, "fill": VALID_FILL})
    mgr.rollout_replica_class = _RecordingReplica

    await mgr._initialize_llm_servers()

    assert pool.pre_started_with["devices_per_node"] == [0, 1]


def test_manager_rejects_worker_group():
    with pytest.raises(NotImplementedError, match="standalone-style"):
        pr.PooledLLMServerManager(config=None, worker_group=object())


# ---------------------------------------------------------------------------
# LLMServerManager.create FQN dispatch socket
# ---------------------------------------------------------------------------
def test_create_dispatch_socket(monkeypatch):
    import verl.utils.import_utils as import_utils
    import verl.workers.rollout.llm_server as ls

    class _Rollout(dict):
        pass

    class _Config:
        def __init__(self):
            self.actor_rollout_ref = types.SimpleNamespace(
                rollout=_Rollout(llm_server_manager_class="fake.fqn.Manager")
            )

    created = {}

    class _FakeManager(ls.LLMServerManager):
        def __init__(self, *a, **kw):
            created["init"] = (a, kw)

        async def _initialize_llm_servers(self, *a, **kw):
            created["servers"] = True

        async def _init_global_load_balancer(self, *a, **kw):
            created["lb"] = True

    monkeypatch.setattr(import_utils, "load_class_from_fqn", lambda fqn, description="class": _FakeManager)

    # auto_await: calling create from sync context runs the coroutine and
    # returns the manager directly (this is how the trainer calls it).
    cfg = _Config()
    result = ls.LLMServerManager.create(config=cfg)

    assert isinstance(result, _FakeManager)
    assert created["init"] == ((), {"config": cfg})
    assert created["servers"] is True
    assert created["lb"] is True


def test_create_dispatch_rejects_non_subclass(monkeypatch):
    import verl.utils.import_utils as import_utils
    import verl.workers.rollout.llm_server as ls

    class _Rollout(dict):
        pass

    class _Config:
        def __init__(self):
            self.actor_rollout_ref = types.SimpleNamespace(
                rollout=_Rollout(llm_server_manager_class="fake.fqn.NotAManager")
            )

    class _Unrelated:
        pass

    monkeypatch.setattr(import_utils, "load_class_from_fqn", lambda fqn, description="class": _Unrelated)

    with pytest.raises(TypeError, match="subclass of LLMServerManager"):
        ls.LLMServerManager.create(config=_Config())


def test_create_without_fqn_uses_base_path(monkeypatch):
    """No llm_server_manager_class -> plain base-class construction flow."""
    import verl.workers.rollout.llm_server as ls

    class _Rollout(dict):
        pass

    class _Config:
        def __init__(self):
            self.actor_rollout_ref = types.SimpleNamespace(rollout=_Rollout())

    created = {}

    class _BaseStub(ls.LLMServerManager):
        def __init__(self, *a, **kw):
            created["init"] = True

        async def _initialize_llm_servers(self, *a, **kw):
            created["servers"] = True

        async def _init_global_load_balancer(self, *a, **kw):
            created["lb"] = True

    # dispatch must NOT trigger: no FQN in config -> cls(*args) path
    def _spy_init(self, *a, **kw):
        created["init"] = True

    monkeypatch.setattr(ls.LLMServerManager, "__init__", _spy_init)
    monkeypatch.setattr(ls.LLMServerManager, "_initialize_llm_servers", _BaseStub._initialize_llm_servers)
    monkeypatch.setattr(ls.LLMServerManager, "_init_global_load_balancer", _BaseStub._init_global_load_balancer)

    result = ls.LLMServerManager.create(config=_Config())
    assert isinstance(result, ls.LLMServerManager)
    assert created == {"init": True, "servers": True, "lb": True}


# ---------------------------------------------------------------------------
# PooledvLLMHttpServer.run_server engine construction
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_server_builds_engine_on_external_actors(monkeypatch):
    """run_server constructs AsyncLLM(executor_class=ExternalExecutor,
    external_actors=...) and skips monkey_patch_model (no worker extension)."""
    captured = {}

    class _ParallelConfig:
        data_parallel_master_port = None
        distributed_executor_backend = "mp"

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _EngineArgs:
        disable_log_stats = False
        enable_log_requests = False

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine_config(self, usage_context=None):
            return _VllmConfig()

    class _FakeLLM:
        def __init__(self, vllm_config=None, executor_class=None, log_stats=True,
                     usage_context=None, log_requests=True, external_actors=None):
            captured.update(
                executor_class=executor_class,
                external_actors=external_actors,
                vllm_config=vllm_config,
            )
            self.model_config = None

        async def reset_mm_cache(self):
            captured["mm_cache_reset"] = True

        async def get_supported_tasks(self):
            return ("generate",)

    class _ExternalExecutor:
        pass

    fake_plugin_mod = types.ModuleType("vllm_external_executor")
    fake_plugin_mod.ExternalExecutor = _ExternalExecutor
    monkeypatch.setitem(sys.modules, "vllm_external_executor", fake_plugin_mod)
    monkeypatch.setattr(pr, "AsyncEngineArgs", _EngineArgs)
    monkeypatch.setattr(pr, "AsyncLLM", _FakeLLM)

    async def _init_app_state(*a, **k):
        captured["app_state"] = True

    async def _run_uvicorn(app, args, address):
        return (8000, None)

    monkeypatch.setattr(pr, "build_app", lambda *a, **k: types.SimpleNamespace(state=object()))
    monkeypatch.setattr(pr, "init_app_state", _init_app_state)
    monkeypatch.setattr(pr, "run_uvicorn", _run_uvicorn)

    # Bypass the heavy vLLMHttpServer.__init__: run_server only needs these.
    server = object.__new__(pr.PooledvLLMHttpServer)
    server._dp_master_port = 1234
    server._external_actors = [FakeWorkerActor(0), FakeWorkerActor(1)]
    server.model_config = _AttrDict(processor=None)
    server.replica_rank = 0

    await server.run_server(types.SimpleNamespace())

    assert captured["executor_class"] is _ExternalExecutor
    assert captured["external_actors"] == server._external_actors
    assert captured["mm_cache_reset"] is True
    assert captured["app_state"] is True


@pytest.mark.asyncio
async def test_run_server_fail_fast_on_stock_vllm(monkeypatch):
    """PyPI vLLM (no external_actors on AsyncLLM) -> TypeError with fork guidance."""

    class _ParallelConfig:
        data_parallel_master_port = None
        distributed_executor_backend = "mp"

    class _VllmConfig:
        parallel_config = _ParallelConfig()

    class _EngineArgs:
        disable_log_stats = False
        enable_log_requests = False

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine_config(self, usage_context=None):
            return _VllmConfig()

    class _StockLLM:
        def __init__(self, vllm_config=None, executor_class=None, log_stats=True,
                     usage_context=None, log_requests=True):
            pass

    monkeypatch.setitem(sys.modules, "vllm_external_executor", types.ModuleType("vllm_external_executor"))
    monkeypatch.setattr(pr, "AsyncEngineArgs", _EngineArgs)
    monkeypatch.setattr(pr, "AsyncLLM", _StockLLM)

    server = object.__new__(pr.PooledvLLMHttpServer)
    server._dp_master_port = 1234
    server._external_actors = [FakeWorkerActor(0)]
    server.model_config = _AttrDict(processor=None)
    server.replica_rank = 0

    with pytest.raises(TypeError, match="feature/external-executor"):
        await server.run_server(types.SimpleNamespace())
