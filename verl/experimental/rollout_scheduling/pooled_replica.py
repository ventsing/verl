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
"""P4b: pooled rollout replicas — verl's first real handshake with multi-task-infer.

Design: rollout_layered_scheduling_design.md §10.4 (integration point 2).

A pooled replica does not create its own vLLM worker group. Instead it leases
pre-started ``ExternalWorkerActor`` handles from an ``ActorPoolManager``
(multi-task-infer, L3b) and builds a single frontend server actor whose
``AsyncLLM`` runs on an ``ExternalExecutor`` bound to those actors:

    pool.pre_start(num_actors=N, devices_per_node=[...])     # once per job
    actors = pool.acquire(tp_size=tp, pp_size=pp)            # per replica
    replica.init_pooled(actors, fill=...)                    # 4th init path
      ├── ONE PooledvLLMHttpServer actor (CPU frontend, no NodeAffinity)
      ├── engine = AsyncLLM(..., executor_class=ExternalExecutor,
      │                    external_actors=actors)           # fork's socket
      └── fill weights from storage  (fail-fast; red line §10.5)
    ... replica lifecycle == lease lifecycle ...
    replica.release_to_pool(pool)                            # reset + IDLE

RL correctness red lines enforced here (design §10.5):

1. ``fill`` is mandatory: a pooled replica enters the load balancer only
   after a successful storage fill tagged with ``global_steps``. A fill
   failure raises (``allow_dummy_fallback: false`` semantics) — dummy
   scaffold weights must never serve traffic.
2. After the fill the KV cache is cleared: scaffold-era KV must not be
   reused with the real weights.
3. ``full_determinism`` is rejected: hash routing requires a stable member
   set, which contradicts lease dynamics.

Known boundaries of this skeleton (documented in the plugin README):

- Per-step trainer→pooled weight sync is NOT wired yet. Options for the
  follow-up (P4b-next): executor-level ``switch_model`` (needs an
  EngineCore-side passthrough in the fork), worker-level
  ``load_model_via_weight_transfer`` via ``collective_rpc``, or per-step
  storage pull. Until then a pooled replica serves the filled checkpoint
  (frozen-policy rollouts / evaluation / warm-pool validation).
- verl's ``vLLMColocateWorkerExtension`` is not applied to pooled workers
  (the plugin builds a plain ``WorkerWrapperBase``): the OOV logits mask
  (``monkey_patch_model``), the MoE weight-loader patch, LoRA hijacking and
  the online-FP8 patches are unavailable → pooled mode is LoRA/quant-model
  restricted in this phase.
- The compile-cache ``CacheManagerActor`` is created with the pool but not
  plumbed into the ``AsyncLLM`` path (the fork's ``EngineCore`` constructs
  the executor without it) — compilation is local per replica for now.
"""

import asyncio
import inspect
import logging

import ray

from verl.plugin.platform import get_platform
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    AsyncEngineArgs,
    AsyncLLM,
    UsageContext,
    build_app,
    init_app_state,
    run_uvicorn,
    vLLMHttpServer,
)

logger = logging.getLogger(__name__)

_FORK_GUIDANCE = (
    "The shared rollout pool requires vLLM with the `external_actors` socket, which "
    "only exists on the fork branch `feature/external-executor` (PyPI vLLM raises "
    "TypeError: unexpected keyword argument 'external_actors'). Install from the fork: "
    "git clone https://github.com/ventsing/vllm && git checkout feature/external-executor "
    "&& pip install -e . --no-build-isolation"
)
_PLUGIN_GUIDANCE = (
    "The shared rollout pool requires the multi-task-infer plugin (vllm_external_executor). "
    "Install it from the multi-task-infer checkout: pip install -e /path/to/vllm/multi-task-infer"
)


def check_pooled_prerequisites(rollout_config, model_config=None) -> None:
    """Fail-fast validation before any pooled replica is created.

    Every check mirrors a deployment constraint from design doc §10.5/§10.6;
    failing here (at manager creation) instead of at first request keeps the
    failure diagnosable.
    """
    if rollout_config.name != "vllm":
        raise NotImplementedError(
            f"The shared rollout pool currently supports rollout.name == 'vllm' only "
            f"(multi-task-infer is a vLLM plugin); got {rollout_config.name!r}."
        )
    if int(rollout_config.data_parallel_size) != 1:
        raise NotImplementedError(
            "The shared rollout pool requires data_parallel_size == 1 per replica; "
            "scale out with more pooled replicas (shared_pool.num_replicas) instead."
        )
    if bool(getattr(rollout_config, "full_determinism", False)):
        raise NotImplementedError(
            "full_determinism is incompatible with the shared rollout pool: hash "
            "routing requires a stable member set, leases change it (design §10.5)."
        )
    disagg = getattr(rollout_config, "disaggregation", None)
    if disagg is not None and bool(getattr(disagg, "enabled", False)):
        raise NotImplementedError("PD-disaggregated rollout is not supported in the shared pool yet.")
    if model_config is not None:
        lora = model_config.get("lora", None) or {}
        lora_rank = lora.get("rank", 0) if isinstance(lora, dict) else getattr(lora, "rank", 0)
        if int(lora_rank or 0) > 0:
            raise NotImplementedError(
                "LoRA is not supported in the shared pool yet (workers are plain "
                "WorkerWrapperBase without verl's extension; design §10.7)."
            )

    # Plugin importable?
    try:
        import vllm_external_executor  # noqa: F401
    except ImportError as e:
        raise ImportError(_PLUGIN_GUIDANCE) from e

    # Fork branch installed? (AsyncLLM accepts external_actors)
    if "external_actors" not in inspect.signature(AsyncLLM.__init__).parameters:
        raise TypeError(_FORK_GUIDANCE)


class PooledvLLMHttpServer(vLLMHttpServer):
    """Frontend HTTP server whose engine runs on pooled external actors.

    Same serving stack as :class:`vLLMHttpServer` (generate / abort / sleep /
    wake / KV management / profiling) — only the engine construction differs:

    - one frontend actor per replica (no per-node servers, no NodeAffinity —
      the pool's placement group owns the GPUs);
    - the engine is built with ``executor_class=ExternalExecutor`` and the
      leased ``external_actors`` instead of spawning workers;
    - the verl worker extension is absent on pooled workers, so the
      ``monkey_patch_model`` startup RPC is skipped.
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        external_actors: list,
        replica_rank: int,
        **kwargs,
    ):
        # The frontend holds no GPUs: engine workers are the external actors.
        kwargs.setdefault("workers", [])
        kwargs.setdefault("node_rank", 0)
        kwargs.setdefault("nnodes", 1)
        kwargs.setdefault("gpus_per_node", len(external_actors))
        kwargs.setdefault("cuda_visible_devices", "")
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            replica_rank=replica_rank,
            **kwargs,
        )
        self._external_actors = external_actors

    async def run_server(self, args):
        """Build the engine on the external actors, then serve as usual.

        The stock path goes through ``AsyncLLM.from_vllm_config``, which has
        no ``external_actors`` parameter (fork limitation) — hence this
        constructor call, mirroring multi-task-infer's own example usage.
        """
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        # Fail-fast socket check (PyPI vLLM lacks the parameter).
        if "external_actors" not in inspect.signature(AsyncLLM.__init__).parameters:
            raise TypeError(_FORK_GUIDANCE)
        from vllm_external_executor import ExternalExecutor

        self.engine = AsyncLLM(
            vllm_config=vllm_config,
            executor_class=ExternalExecutor,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            log_requests=engine_args.enable_log_requests,
            external_actors=self._external_actors,
        )

        # Don't keep the dummy data in memory
        await self.engine.reset_mm_cache()
        # NOTE: the stock server calls collective_rpc("monkey_patch_model")
        # (OOV logits mask + MoE weight-loader patch) — that method lives in
        # verl's vLLMColocateWorkerExtension, which pooled workers do not
        # have. Skip it; vision-placeholder masking is a first-phase loss,
        # documented in the module docstring.
        if self.model_config.processor is not None:
            logger.warning(
                "Pooled replica %s: monkey_patch_model skipped (verl worker extension "
                "not applied to pooled workers); vision placeholder tokens are NOT masked.",
                self.replica_rank,
            )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple = ()
        build_app_kwargs: dict = {}
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await self.engine.get_supported_tasks()
            build_app_kwargs["supported_tasks"] = supported_tasks
        if "model_config" in build_app_sig.parameters:
            build_app_kwargs["model_config"] = self.engine.model_config
        app = build_app(args, **build_app_kwargs)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(self.engine, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(self.engine, app.state, args, supported_tasks)
        else:
            await init_app_state(self.engine, app.state, args)
        if self.replica_rank == 0:
            logger.info("Initializing a pooled V1 LLM engine (external actors) with config: %s", vllm_config)

        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)


class PooledvLLMReplica(RolloutReplica):
    """Rollout replica backed by leased pool actors (4th init path).

    ``init_pooled`` is the counterpart of init_hybrid / init_colocated /
    init_standalone: no resource pool and no worker group are created — the
    engine workers are the leased ``ExternalWorkerActor`` handles, and the
    replica lifecycle equals the lease lifecycle.
    """

    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kwargs):
        super().__init__(replica_rank, config, model_config, gpus_per_node, **kwargs)
        self.server_class = ray.remote(PooledvLLMHttpServer)
        self.external_actors: list = []

    async def init_pooled(self, external_actors: list, fill: dict):
        """Build the frontend on leased actors, then fill weights from storage.

        Args:
            external_actors: leased ``ExternalWorkerActor`` handles;
                ``len(external_actors)`` must equal ``world_size`` (tp*pp).
            fill: mandatory weight-fill spec (design §10.5 red line): keys
                ``checkpoint_path`` and ``global_steps`` are required,
                ``storage_backend`` (default "nfs") and ``storage_config``
                are optional. A failed fill raises — scaffold weights never
                enter the load balancer.
        """
        if len(external_actors) != self.world_size:
            raise ValueError(
                f"Pooled replica needs exactly world_size={self.world_size} external "
                f"actors, got {len(external_actors)}."
            )
        self.rollout_mode = RolloutMode.STANDALONE
        self.external_actors = list(external_actors)
        await self.launch_servers()
        await self.fill_weights_from_storage(**fill)

    async def launch_servers(self):
        """Launch a single frontend server actor (pool owns the GPUs)."""
        env_vars = {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
        }
        name = f"rollout_pooled_server_{self.replica_rank}{self.name_suffix}"
        server = self.server_class.options(
            name=name,
            max_concurrency=self.max_concurrency,
            runtime_env={"env_vars": env_vars},
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            external_actors=self.external_actors,
            replica_rank=self.replica_rank,
        )
        self.servers.append(server)

        master_address, master_port, dp_rpc_port = await server.get_master_address.remote()
        await server.launch_server.remote(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
        )
        server_address, server_port = await server.get_server_address.remote()
        self._server_handle = server
        self._server_address = (
            f"[{server_address}]:{server_port}" if is_valid_ipv6_address(server_address) else f"{server_address}:{server_port}"
        )

    async def fill_weights_from_storage(
        self,
        *,
        checkpoint_path: str,
        global_steps: int,
        storage_backend: str = "nfs",
        storage_config: dict | None = None,
    ):
        """Fill weights on every pooled worker from storage (fail-fast).

        Sequence (design §10.5): storage pull on each actor (any failure
        raises — ``allow_dummy_fallback: false``), then tag the server with
        the checkpoint's ``global_steps``, then clear the KV cache
        (scaffold-era KV must not be reused with the real weights).
        """
        if not checkpoint_path:
            raise ValueError("Pooled replica requires fill.checkpoint_path (allow_dummy_fallback=false).")
        if not isinstance(global_steps, int):
            raise ValueError(
                "Pooled replica requires fill.global_steps (int) so the served weight "
                "version is tagged (design §10.5 red line 2)."
            )
        # Storage pull on every actor; a Ray call re-raises worker exceptions
        # (missing checkpoint / storage down / tensor mismatch) — fail-fast.
        await asyncio.gather(
            *[
                actor.load_model_from_storage.remote(
                    checkpoint_path=checkpoint_path,
                    storage_backend=storage_backend,
                    storage_config=storage_config,
                )
                for actor in self.external_actors
            ]
        )
        # Tag the served version, then invalidate scaffold-era KV.
        await self._server_handle.set_global_steps.remote(global_steps)
        await self._server_handle.clear_kv_cache.remote()
        logger.info(
            "[PooledReplica %s] weights filled from %s (global_steps=%s)",
            self.replica_rank,
            checkpoint_path,
            global_steps,
        )

    async def release_to_pool(self, pool) -> None:
        """Tear the replica down and return its actors to the pool.

        Order: kill the frontend (it holds the engine/executor and its
        message queues), reset each worker (plugin semantics: worker
        shutdown → IDLE), then hand the handles back to the pool registry.
        """
        if self._server_handle is not None:
            ray.kill(self._server_handle)
            self._server_handle = None
            self.servers = []
        if self.external_actors:
            await asyncio.gather(*[actor.reset.remote() for actor in self.external_actors])
            pool.release(self.external_actors)
            self.external_actors = []


class PooledLLMServerManager(LLMServerManager):
    """shared_pool mode: replica lifecycle == lease lifecycle.

    Selected via config (no trainer code changes)::

        actor_rollout_ref:
          rollout:
            llm_server_manager_class: \\
                verl.experimental.rollout_scheduling.pooled_replica.PooledLLMServerManager
            shared_pool:
              num_replicas: 1            # required
              devices: [0, 1]            # optional; default range(num_actors)
              warmup_distributed: true   # NCCL warmup at pre_start
              shared_cache_dir: null     # NFS dir for large compile caches
              fill:                      # mandatory (design §10.5)
                checkpoint_path: model/step_100
                global_steps: 100
                storage_backend: nfs
                storage_config: {base_path: /shared/ckpt}

    The manager owns the pool for the job's lifetime (single-job mode).
    Cross-job sharing moves the pool behind the SharedRolloutBroker in P4c.
    """

    def __init__(self, config, worker_group=None, **kwargs):
        if worker_group is not None:
            raise NotImplementedError(
                "The shared rollout pool is standalone-style; hybrid worker groups are "
                "not pooled in this phase (design §10.6 — a boundary, not a permanent ban)."
            )
        # The base class gates standalone mode on rollout.nnodes > 0; pooled
        # replicas allocate no Ray resources (the pool owns the GPUs), so a
        # missing nnodes is satisfied with 1 rather than rejected.
        rollout = config.actor_rollout_ref.rollout
        if not int(rollout.get("nnodes", 0) or 0):
            rollout.nnodes = 1
        super().__init__(config=config, worker_group=None, **kwargs)
        # Recipe hook from LLMServerManager.__init__ — replace the replica class.
        self.rollout_replica_class = PooledvLLMReplica
        self.actor_pool = None

    async def _initialize_llm_servers(self, start_rank: int = None):
        if start_rank is None:
            start_rank = self.start_rank
        check_pooled_prerequisites(self.rollout_config, self.model_config)
        pool_cfg = self._shared_pool_config()

        tp = int(self.rollout_config.tensor_model_parallel_size)
        pp = int(self.rollout_config.pipeline_model_parallel_size)
        world_size = tp * pp  # dp == 1 asserted by prerequisites
        num_replicas = int(pool_cfg.get("num_replicas") or 0)
        if num_replicas <= 0:
            raise ValueError("shared_pool.num_replicas must be a positive int.")
        num_actors = num_replicas * world_size
        devices = pool_cfg.get("devices") or list(range(num_actors))
        if len(devices) != num_actors:
            raise ValueError(
                f"shared_pool.devices has {len(devices)} entries but "
                f"num_replicas({num_replicas}) * tp*pp({world_size}) = {num_actors} actors are needed."
            )

        fill = pool_cfg.get("fill") or {}
        if not fill.get("checkpoint_path") or not isinstance(fill.get("global_steps"), int):
            raise ValueError(
                "shared_pool.fill must define checkpoint_path and global_steps (int) — "
                "pooled replicas only enter the load balancer with filled, tagged weights "
                "(design §10.5, allow_dummy_fallback=false)."
            )

        # Pool lifecycle: pre-start once, lease per replica.
        self.actor_pool = self._build_pool()
        self.actor_pool.pre_start(
            num_actors=num_actors,
            devices_per_node=list(devices),
            warmup_distributed=bool(pool_cfg.get("warmup_distributed", True)),
            shared_cache_dir=pool_cfg.get("shared_cache_dir"),
            enable_cache_compression=bool(pool_cfg.get("enable_cache_compression", True)),
        )

        self.rollout_replicas = []
        for replica_rank in range(num_replicas):
            actors = self.actor_pool.acquire(tp_size=tp, pp_size=pp)
            replica = self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=world_size,
            )
            await replica.init_pooled(external_actors=actors, fill=dict(fill))
            self.rollout_replicas.append(replica)

        self.server_handles = [replica._server_handle for replica in self.rollout_replicas]
        self.server_addresses = [replica._server_address for replica in self.rollout_replicas]
        logger.info(
            "[PooledLLMServerManager] %d pooled replicas over %d actors (tp=%d, pp=%d)",
            num_replicas,
            num_actors,
            tp,
            pp,
        )

    def _shared_pool_config(self) -> dict:
        """Read ``rollout.shared_pool`` as a plain dict (absent → empty)."""
        from omegaconf import OmegaConf

        pool_cfg = self.rollout_config.get("shared_pool")
        if pool_cfg is None:
            return {}
        if OmegaConf.is_config(pool_cfg):
            return OmegaConf.to_container(pool_cfg, resolve=True)
        return dict(pool_cfg)

    def _build_pool(self):
        """Create the actor pool (seam for tests / future broker hand-off)."""
        from vllm_external_executor import ActorPoolManager

        return ActorPoolManager()

    async def release_replica(self, replica_rank: int) -> None:
        """Release one pooled replica's actors back to the pool (L2/P4c seam)."""
        replica = self.rollout_replicas[replica_rank]
        await replica.release_to_pool(self.actor_pool)
        self.rollout_replicas.pop(replica_rank)
        self.server_handles.pop(replica_rank)
        self.server_addresses.pop(replica_rank)

    async def release_all(self) -> None:
        """Release every pooled replica (job teardown / full cede)."""
        for replica in list(self.rollout_replicas):
            await replica.release_to_pool(self.actor_pool)
        self.rollout_replicas = []
        self.server_handles = []
        self.server_addresses = []
