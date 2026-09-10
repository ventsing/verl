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
import concurrent.futures
import logging
import os
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import checkpoint_engine.distributed as dist
import ray
import torch
from checkpoint_engine.ps import H2DBucket, ParameterMeta, ParameterServer, _gen_h2d_buckets, _to_named_tensor

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.device import get_nccl_backend, get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def ckpt_get_named_tensor_buckets(
    iterable: Generator[tuple[str, torch.Tensor], None, None],
    bucket_bytes: int,
    world_size: int,
    rank_id: int,
    rollout_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = {}
    current_size = 0
    for tensor_idx, (name, tensor) in enumerate(iterable):
        tensor = tensor.to(rollout_dtype)
        if tensor_idx % world_size == rank_id:
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                    current_bucket = {}
                    current_size = 0

            current_bucket[name] = tensor
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


async def receive_tensor(
    self,
    checkpoint_name: str,
    ranks_group: int,
    ranks: list[int] | None = None,
    bucket_size: int = 2 << 30,
    disable_h2d_buffer: bool = False,
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
    assert dist.is_initialized(), "process group is not initialized"
    assert self._p2p_store is not None, "p2p store is not initialized"
    assert ranks, "ranks should be set"

    # first execute a barrier to avoid subsequent device oom
    dist.barrier(group=ranks_group)
    buckets = _gen_h2d_buckets(
        self._current_global_parameter_metas,
        bucket_size,
        self._local_rdma_devices,
        self._remote_rdma_devices,
        ranks,
    )
    h2d_buffer: torch.Tensor | None = (
        None
        if disable_h2d_buffer
        else torch.empty(bucket_size, dtype=torch.uint8, device=self.device_manager.device_type)
    )
    # p2p store need to register h2d_buffer to let other ranks read
    if ranks:
        h2d_buffer_name = "__h2d_buffer__"
        if h2d_buffer is not None and self._p2p_store is not None:
            self._p2p_store.register_named_tensors({h2d_buffer_name: h2d_buffer})
    receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
    for receiver_rank, owner_rank, bucket in buckets:
        if receiver_rank != self._rank:
            continue
        receiver_rank_buckets.append((owner_rank, bucket))
    buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type)
    buckets_by_receiver_rank: dict[int, list[H2DBucket]] = defaultdict(list)

    max_len = 0
    for receiver_rank, _, bucket in buckets:
        buckets_by_receiver_rank[receiver_rank].append(bucket)
        if len(buckets_by_receiver_rank[receiver_rank]) > max_len:
            max_len = len(buckets_by_receiver_rank[receiver_rank])
    gidx = 0
    metadata: list[ParameterMeta]
    try:
        for i in range(max_len):
            if i < len(receiver_rank_buckets) and not disable_h2d_buffer:
                self._copy_to_buffer(
                    checkpoint_name,
                    receiver_rank_buckets[i][1],
                    h2d_buffer,
                    receiver_rank_buckets[i][0] if ranks else None,
                )
            for receiver_rank, _buckets in buckets_by_receiver_rank.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]
                start = gidx % 2 * bucket_size
                buffer_b: torch.Tensor = buffer[start : start + bucket.size]
                if receiver_rank == self._rank:
                    if disable_h2d_buffer:
                        self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                    else:
                        buffer_b.data.copy_(h2d_buffer[: bucket.size])
                broadcast_op = BroadcastOperation(
                    rank=receiver_rank,
                    ranks_group=ranks_group,
                    bucket=buffer_b,
                    metadata=bucket.items,
                )
                if gidx == 0:
                    metadata = await broadcast_op.wait_for_complete()
                    gidx += 1
                    continue
                meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
                for item in meta_list:
                    shape = item["shape"]
                    if isinstance(shape, list | tuple):
                        shape = torch.Size(shape)
                    assert isinstance(shape, torch.Size)
                    dtype, offset = item["dtype"], item["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    yield item["name"], tensor
                metadata = await broadcast_op.wait_for_complete()
                self.device_manager.device_module.synchronize()
                gidx += 1

        meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
        for item in meta_list:
            shape = item["shape"]
            if isinstance(shape, list | tuple):
                shape = torch.Size(shape)
            assert isinstance(shape, torch.Size)
            dtype, offset = item["dtype"], item["offset"]
            size = dtype.itemsize * shape.numel()
            tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
            yield item["name"], tensor

    finally:
        dist.barrier(group=ranks_group)
        if ranks and h2d_buffer is not None:
            self._p2p_store.unregister_named_tensors([h2d_buffer_name])
        self.device_manager.device_module.empty_cache()


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


class BroadcastOperation:
    """Async broadcast operation in separate thread.

    Args:
        rank (int): The rank of the current process.
        ranks_group (int): The process group's value.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (list[ParameterMeta]): The metadata of the tensor.
    """

    def __init__(
        self,
        rank: int,
        ranks_group: int,
        bucket: torch.Tensor,
        metadata: list[ParameterMeta],
    ) -> None:
        self.rank = rank
        self.ranks_group = ranks_group
        self.bucket = bucket
        self.metadata = metadata

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # broadcast tensor
        dist.broadcast(self.bucket, src=self.rank, group=self.ranks_group)

    async def wait_for_complete(self) -> list[ParameterMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            list[ParameterMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("kimi_ckpt_engine")
class KIMICheckpointEngine(CheckpointEngine):
    """kimi checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        rebuild_group (bool): Whether to rebuild the process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.bucket_size = bucket_size
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.initialized = False
        self.checkpoint_name = "kimi_checkpoint_engine"
        # version -> gathered metas snapshot (multi-version pull path)
        self._versioned_metas: dict[int, dict] = {}

    def prepare(self) -> MasterMetadata:
        if self.is_master:
            self.ip = ray.util.get_node_ip_address().strip("[]")
            self.listen_port, _ = get_free_port(self.ip)

        return (
            MasterMetadata(zmq_ip=None, zmq_port=None, dist_ip=self.ip, dist_port=self.listen_port)
            if self.is_master
            else None
        )

    def finalize(self):
        """Destroy the ckpt engine process group if rebuild_group is True."""
        if self.rebuild_group:
            dist.destroy_process_group()
            self.rank = None
            self.world_size = None
            self.initialized = False

    @classmethod
    def build_topology(
        cls,
        actor_wg_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
        replica_partition: list[list[int]] | None = None,
    ):
        """Build per-rank init kwargs; ``replica_partition`` (global engine
        ranks per rollout replica, tiling ``[actor_ws, actor_ws+rollout_ws)``)
        additionally creates one process-group SUBGROUP per replica so pulls
        can be scoped to a single replica (per-replica, any-time pulls)."""
        actor_wg_kwargs = {
            "method": ["init_process_group"] * actor_wg_world_size,
            "rank": list(range(0, actor_wg_world_size)),
            "actor_wg_world_size": [actor_wg_world_size] * actor_wg_world_size,
            "rollout_world_size": [rollout_world_size] * actor_wg_world_size,
            "master_metadata": [metadata[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "method": ["init_process_group"] * rollout_world_size,
            "rank": list(range(actor_wg_world_size, actor_wg_world_size + rollout_world_size)),
            "actor_wg_world_size": [actor_wg_world_size] * rollout_world_size,
            "rollout_world_size": [rollout_world_size] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        if replica_partition is not None:
            # broadcast the same partition to every rank: new_group is a
            # collective over the global group, so ALL ranks must create the
            # subgroups in the same order
            actor_wg_kwargs["replica_partition"] = [replica_partition] * actor_wg_world_size
            rollout_kwargs["replica_partition"] = [replica_partition] * rollout_world_size
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        actor_wg_world_size: int,
        rollout_world_size: int,
        master_metadata: MasterMetadata,
        replica_partition: list[list[int]] | None = None,
    ):
        """Initialize the ckpt engine process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            replica_partition: optional per-replica rank groups (global
                engine ranks, tiling the rollout range) — creates one
                subgroup per replica so ``receive_weights_version`` can
                pull into a SINGLE replica without synchronizing the fleet.
                ``dist.new_group`` is collective, so every rank (actor
                included) creates all subgroups in partition order.
        """
        # idempotence WITH topology checking: a re-init with the SAME
        # partition is a no-op (the relay controller rebuilds the group on
        # failover without changing the fleet); a DIFFERENT partition means
        # the replica set changed (elastic add/remove) — silently keeping
        # the old subgroups would route per-replica pulls over a stale
        # topology, so fail loud instead. True dynamic group rebuild (and
        # the NCCL group-teardown traps that come with it) is a cluster TODO.
        if self.initialized:
            installed = getattr(self, "_installed_partition", None)
            if replica_partition != installed:
                raise RuntimeError(
                    f"checkpoint-engine process group already initialized with "
                    f"replica_partition={installed}; refusing to silently re-init "
                    f"with {replica_partition} (elastic replica changes need an "
                    f"explicit group rebuild, not a re-init)"
                )
            return  # same topology: idempotent no-op

        self.rank = rank
        self.actor_wg_world_size = actor_wg_world_size
        self.rollout_world_size = rollout_world_size
        self.world_size = actor_wg_world_size + rollout_world_size
        # per-replica pull state (None on actor ranks / legacy flat path)
        self.replica_id: int | None = None
        self.replica_group = None
        self.replica_ranks: list[int] | None = None

        self.parameter_server = ParameterServer(
            rank=rank,
            world_size=self.world_size,
            auto_pg=False,
            master_addr=master_metadata.dist_ip,
            master_port=master_metadata.dist_port,
        )
        self.parameter_server.receive_tensor = types.MethodType(receive_tensor, self.parameter_server)

        dist.use_backend(f"vllm_{get_nccl_backend()}")
        self.parameter_server.init_process_group()

        self.rollout_ranks = list(range(self.actor_wg_world_size, self.world_size))
        self.rollout_group = dist.new_group(self.rollout_ranks)

        if replica_partition:
            expected = set(self.rollout_ranks)
            covered: set[int] = set()
            for part in replica_partition:
                covered.update(part)
            if covered != expected or len(covered) != sum(len(p) for p in replica_partition):
                raise ValueError(
                    f"replica_partition {replica_partition} must exactly tile the "
                    f"rollout ranks {self.rollout_ranks} (no overlap, no gap)"
                )
            # every rank creates every subgroup (collective, same order);
            # members receive a usable group, non-members an inert handle
            for replica_id, part in enumerate(replica_partition):
                group = dist.new_group(part)
                if self.rank in part:
                    self.replica_id = replica_id
                    self.replica_group = group
                    self.replica_ranks = list(part)
        self._installed_partition = replica_partition
        self.initialized = True

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """

        def offload_cpu(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
            return name, tensor.to("cpu", non_blocking=True)

        start_time = time.time()
        named_tensors = {}
        for named_tensors_gpu in ckpt_get_named_tensor_buckets(
            weights, self.bucket_size, self.actor_wg_world_size, self.rank, self.rollout_dtype
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(
                        offload_cpu,
                        name,
                        tensor,
                    )
                    for name, tensor in named_tensors_gpu.items()
                ]
            for future in concurrent.futures.as_completed(futures):
                name, tensor_cpu = future.result()
                named_tensors[name] = tensor_cpu

        get_torch_device().synchronize()

        self.parameter_server.register_checkpoint(self.checkpoint_name, named_tensors=named_tensors)
        named_tensors = {}
        get_torch_device().empty_cache()
        logger.info(f"Rank {self.rank} offload and register, time cost: {time.time() - start_time:.2f}s")

        self.parameter_server.gather_metas(self.checkpoint_name)
        dist.barrier()
        self.parameter_server.unregister_checkpoint(self.checkpoint_name)
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        self.parameter_server.gather_metas(self.checkpoint_name)

        start_time = time.time()
        total_bytes, total_params = 0, 0
        async for name, tensor in self.parameter_server.receive_tensor(
            self.checkpoint_name, self.rollout_group, self.rollout_ranks, self.bucket_size
        ):
            total_bytes += tensor.element_size() * tensor.nelement()
            total_params += 1
            yield name, tensor
        dist.barrier()
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

    # ------------------------------------------------- multi-version pull path
    #
    # Laminar-style versioned weight store (consumed by
    # verl/experimental/trajectory_async). ``stage_version`` registers the
    # actor's current shards under a VERSION-SCOPED checkpoint name and — the
    # one behavioral difference from ``send_weights`` — does NOT unregister:
    # the registered CPU/pinned shards remain readable in the P2P store, so
    # rollout replicas can pull the version at any later time via
    # ``receive_weights_version`` (repeatedly; e.g. a restarted or late
    # replica re-pulls). ``unstage_version`` retires a version (driver-side
    # retention policy); frees the pinned CPU memory on every actor rank.
    #
    # ``gather_metas`` is an all_gather_object over the whole engine process
    # group (actor + rollout ranks), so the metadata phase stays synchronized:
    # the driver fires ``stage_weights_version`` on the actor worker group and
    # ``gather_version_metas`` on the rollout worker groups CONCURRENTLY, and
    # every rank snapshots the gathered metas per version. The expensive part
    # (tensor transfer) is what becomes pull-based.
    #
    # Topology note: the stock single ``rollout_group`` spans ALL rollout
    # ranks, and ``receive_tensor`` barriers on it — pulls are then
    # fleet-synchronized (every replica pulls the same version together).
    # Passing a ``replica_partition`` (see ``init_process_group`` /
    # ``build_topology``) additionally installs one subgroup per replica, and
    # ``receive_weights_version(version, replica_id=...)`` pulls over that
    # subgroup alone: the H2D bucket partition and broadcast barriers touch
    # only the replica's ranks, so OTHER replicas keep generating through it.
    # The actor ranks serve reads from their registered CPU shards via the
    # P2P store and need no subgroup membership.

    def _versioned_checkpoint_name(self, version: int) -> str:
        return f"{self.checkpoint_name}:v{version}"

    @torch.no_grad()
    async def stage_version(
        self,
        version: int,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Register this rank's current weights as a pullable version.

        Actor-side counterpart of ``send_weights``: same CPU offload path,
        same registration — but under ``"{checkpoint_name}:v{version}"`` and
        WITHOUT the trailing unregister, and with the gathered metas snapshotted
        per version so later pulls need no further collective.

        The rollout ranks must run ``gather_version_metas(version)`` at the
        same time (gather_metas is collective over the whole group).
        """
        checkpoint_name = self._versioned_checkpoint_name(version)

        def offload_cpu(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
            return name, tensor.to("cpu", non_blocking=True)

        start_time = time.time()
        named_tensors = {}
        for named_tensors_gpu in ckpt_get_named_tensor_buckets(
            weights, self.bucket_size, self.actor_wg_world_size, self.rank, self.rollout_dtype
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [executor.submit(offload_cpu, name, tensor) for name, tensor in named_tensors_gpu.items()]
            for future in concurrent.futures.as_completed(futures):
                name, tensor_cpu = future.result()
                named_tensors[name] = tensor_cpu

        get_torch_device().synchronize()

        # register + KEEP registered: these pinned CPU shards are the relay
        # memory for this version until unstage_version retires it
        staged_bytes = sum(t.element_size() * t.nelement() for t in named_tensors.values())
        staged_params = len(named_tensors)
        self.parameter_server.register_checkpoint(checkpoint_name, named_tensors=named_tensors)
        named_tensors = {}
        get_torch_device().empty_cache()
        logger.info(f"Rank {self.rank} stage v{version}: offload+register, {time.time() - start_time:.2f}s")

        self.parameter_server.gather_metas(checkpoint_name)
        # snapshot the gathered metas per version — receive_tensor reads the
        # single-slot _current_global_parameter_metas, so later pulls restore
        # this snapshot instead of re-gathering (which would be collective)
        self._versioned_metas[version] = self.parameter_server.get_metas()
        logger.info(f"Rank {self.rank} stage v{version} done, {time.time() - start_time:.2f}s")
        # per-rank staged-size metrics: the driver sums these for host-memory
        # quota accounting (relay controller's max_staged_bytes)
        return {"staged_bytes": staged_bytes, "staged_params": staged_params}

    def gather_version_metas(self, version: int):
        """Rollout-side participation in a version's metadata gather.

        Must run CONCURRENTLY with the actor ranks' ``stage_version`` —
        ``gather_metas`` is an all_gather_object over the whole engine group.
        """
        checkpoint_name = self._versioned_checkpoint_name(version)
        self.parameter_server.gather_metas(checkpoint_name)
        self._versioned_metas[version] = self.parameter_server.get_metas()

    @torch.no_grad()
    async def receive_weights_version(
        self,
        version: int,
        replica_id: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Pull a staged version into this rollout rank (anytime, repeatable).

        Same transfer path as ``receive_weights`` but scoped to the version's
        checkpoint name and driven by the snapshotted metas (no re-gather).
        With ``replica_id``: pull over that replica's SUBGROUP — the H2D
        bucket partition and the broadcast barriers involve only the
        replica's ranks, so other replicas keep generating (requires the
        replica partition from ``init_process_group``). Without: the legacy
        fleet-synchronized rollout-group pull.
        """
        checkpoint_name = self._versioned_checkpoint_name(version)
        metas = self._versioned_metas.get(version)
        if metas is None:
            raise LookupError(
                f"version {version} was never staged on this rank "
                f"(known versions: {sorted(self._versioned_metas)})"
            )

        if replica_id is not None:
            if self.replica_id is None:
                raise ValueError(
                    f"per-replica pull requested (replica {replica_id}) but no replica "
                    "partition was installed at init_process_group"
                )
            if self.replica_id != replica_id:
                raise ValueError(
                    f"per-replica pull for replica {replica_id} hit rank {self.rank} "
                    f"of replica {self.replica_id} (driver dispatched to the wrong group)"
                )
            ranks_group, ranks = self.replica_group, self.replica_ranks
        else:
            ranks_group, ranks = self.rollout_group, self.rollout_ranks

        ps = self.parameter_server
        # single-slot restore around the transfer: concurrent pulls of other
        # versions must not observe this version's metas
        previous_metas = ps._current_global_parameter_metas
        ps._current_global_parameter_metas = metas
        try:
            start_time = time.time()
            total_bytes, total_params = 0, 0
            async for name, tensor in ps.receive_tensor(
                checkpoint_name, ranks_group, ranks, self.bucket_size
            ):
                total_bytes += tensor.element_size() * tensor.nelement()
                total_params += 1
                yield name, tensor
            dist.barrier()
        finally:
            ps._current_global_parameter_metas = previous_metas
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive v{version} done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

    def unstage_version(self, version: int):
        """Retire a staged version (actor side): unregister from the P2P store
        and drop the metas snapshot. Frees this rank's pinned CPU shard copy."""
        checkpoint_name = self._versioned_checkpoint_name(version)
        self.parameter_server.unregister_checkpoint(checkpoint_name)
        self._versioned_metas.pop(version, None)

    def drop_version(self, version: int):
        """Drop a retired version's metas snapshot (rollout side)."""
        self._versioned_metas.pop(version, None)
