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
import gc
import logging
import os
import time
from typing import Any, AsyncGenerator, Generator

import ray
import torch
from mooncake.engine import TransferEngine

try:
    from vllm.distributed.utils import StatelessProcessGroup
except ImportError:
    from sglang.srt.distributed.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@CheckpointEngineRegistry.register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    """Mooncake checkpoint engine with p2p communication using TransferEngine

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time.
        device (str): The device to use for the checkpoint engine, "cpu" or "cuda".
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers.
        device_name (str): Mooncake device name filter.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_master: bool = False,
        rebuild_group: bool = False,
    ):
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.rebuild_group = rebuild_group

        rank = int(os.environ["RANK"])
        device_count = get_torch_device().device_count()
        local_rank = rank % device_count
        get_torch_device().set_device(local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "ascend_direct" if self.device == "npu" else "rdma",
            device_name,
        )
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname

        self.buf = torch.empty(2 * self.bucket_size, dtype=torch.uint8, device=self.device)
        self.magic_buf = torch.empty(4 * 1024, dtype=torch.uint8, device=self.device)
        # Separate buffer for receiving magic completion signals (one 4-byte slot per double-buffer)
        # This prevents the next rank's magic write from corrupting data buffer contents.
        self.magic_recv = torch.zeros(8, dtype=torch.uint8, device=self.device)
        ret = self.engine.batch_register_memory(
            [self.buf.data_ptr(), self.magic_buf.data_ptr(), self.magic_recv.data_ptr()],
            [2 * self.bucket_size, 4 * 1024, 8],
        )
        assert ret == 0, f"batch_register_memory failed ret={ret}"
        logger.info(f"__init__ session_id={self.session_id}")

    def prepare(self) -> dict[str, Any]:
        """Prepare send and recv buckets"""
        logger.info(
            f"prepare ptr={self.buf.data_ptr():#x} len={2 * self.bucket_size} "
            f"magic_buf_ptr={self.magic_buf.data_ptr():#x}"
        )
        port, _ = get_free_port(self.hostname)
        return {"addr": self.hostname, "port": port}

    @classmethod
    def build_topology(
        cls,
        actor_wg_world_size: int,
        rollout_world_size: int,
        metadatas: list[dict],
        replica_partition: list[list[int]] | None = None,
    ):
        """``replica_partition`` is accepted but UNUSED: mooncake's versioned
        path uses direct P2P reads from the actor's staged buffer (no rank
        chain, no collective barrier between consumers), so per-replica pulls
        need NO communication-domain subgroups — scoping happens by
        dispatching ``pull_weights_version`` to a single replica's worker
        group. The parameter exists so the versioned driver
        (``build_process_group``) can pass it uniformly across backends."""
        actor_wg_kwargs = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "metadata": [metadatas[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "metadata": [metadatas[0]] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, metadata: dict[str, Any]):
        self.rank = rank
        self.world_size = world_size
        if rank < 0:
            logger.info(f"init_process_group rank={rank}")
            return

        self.store = StatelessProcessGroup.create(
            host=metadata["addr"],
            port=metadata["port"],
            rank=rank,
            world_size=world_size,
        )

        info = {
            "session_id": self.session_id,
            "ptr": self.buf.data_ptr(),
        }

        info_list = self.store.all_gather_obj(info)
        self.buffer_info = None if rank == 0 else info_list[rank - 1]

        logger.info(f"init_process_group rank={rank} world_size={world_size} buffer_info={self.buffer_info}")

    def finalize(self):
        """Cleanup communication and deregister memory"""
        self.store = None
        get_torch_device().empty_cache()
        gc.collect()
        logger.info(f"finalize rank={self.rank}")

    async def wait_for_complete(self, buf: torch.Tensor):
        magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)
        while True:
            if torch.equal(buf[:4], magic):
                buf[:4] = 0  # reset for next use
                break
            await asyncio.sleep(0)

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Send weights using Mooncake TransferEngine"""
        if self.rank < 0:
            for name, weight in weights:
                pass
            logger.info(f"send_weights rank={self.rank}")
            return

        total_bytes = 0
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        should_wait = False
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        magic_slots = [self.magic_recv[:4], self.magic_recv[4:]]
        idx = 0
        current = bufs[idx]

        for name, weight in weights:
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                total_bytes += offset
                get_torch_device().synchronize()
                info = {
                    "bucket_meta": bucket_meta,
                    "ptr": current.data_ptr(),
                    "magic_ptr": magic_slots[idx].data_ptr(),
                    "len": offset,
                    "is_last": False,
                }
                # send to rank 1
                self.store.send_obj(info, 1)

                idx ^= 1
                current = bufs[idx]
                bucket_meta = {}
                offset = 0

                if should_wait:
                    await self.wait_for_complete(magic_slots[idx])
                should_wait = True

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            current[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        get_torch_device().synchronize()
        info = {
            "bucket_meta": bucket_meta,
            "ptr": current.data_ptr(),
            "magic_ptr": magic_slots[idx].data_ptr(),
            "len": offset,
            "is_last": True,
        }
        self.store.send_obj(info, 1)
        await self.wait_for_complete(magic_slots[idx])

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive weights from the previous rank via RDMA.

        Protocol:
        1. Recv metadata from prev rank via TCPStore
        2. RDMA-read data from prev rank's buffer
        3. Forward metadata to next rank (if not last)
        4. Yield weights to consumer
        5. Write magic completion signal to prev rank's magic_ptr
           (a dedicated magic_recv slot, NOT the data buffer, to avoid
           transfer_sync_write local GPU side-effect corruption)
        """
        _magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)

        start_time = time.time()
        total_bytes = 0
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        magic_slots = [self.magic_recv[:4], self.magic_recv[4:]]
        idx = 0
        current = bufs[idx]
        self.magic_buf[:4] = _magic.clone()

        while True:
            info = self.store.recv_obj(self.rank - 1)
            if idx >= 2 and self.rank < self.world_size - 1:
                await self.wait_for_complete(magic_slots[idx % 2])

            prev_ptr = info["ptr"]
            prev_magic_ptr = info.get("magic_ptr")

            ret = self.engine.transfer_sync_read(
                self.buffer_info["session_id"],
                current.data_ptr(),
                prev_ptr,
                info["len"],
            )
            assert ret == 0

            total_bytes += info["len"]

            info["ptr"] = current.data_ptr()
            info["magic_ptr"] = magic_slots[idx % 2].data_ptr()
            if self.rank < self.world_size - 1:
                self.store.send_obj(info, self.rank + 1)

            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = current[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            get_torch_device().synchronize()

            if prev_magic_ptr is not None:
                ret = self.engine.transfer_sync_write(
                    self.buffer_info["session_id"],
                    self.magic_buf.data_ptr(),
                    prev_magic_ptr,
                    4,
                )
                assert ret == 0
            else:
                ret = self.engine.transfer_sync_write(
                    self.buffer_info["session_id"],
                    self.magic_buf.data_ptr(),
                    prev_ptr,
                    4,
                )
                assert ret == 0

            idx += 1
            current = bufs[idx % 2]
            get_torch_device().synchronize()

            if info["is_last"]:
                break

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )

    # ------------------------------------------------- multi-version pull path
    #
    # The versioned path replaces the stock rank CHAIN (0 -> 1 -> ... -> N,
    # each rank reading the previous rank's transient buffer) with direct
    # P2P reads from a per-version STAGED buffer on the actor rank 0:
    #
    # * ``stage_version`` packs the version into its own pinned host buffer
    #   (whole-tensor buckets, the stock packing convention), registers it
    #   with ``batch_register_memory``, and KEEPS it registered — the
    #   registered buffer is the relay memory for this version (the same
    #   placement idea as the kimi engine's per-version CPU shards).
    # * The buffer descriptor (session_id + ptr + bucket map) reaches the
    #   rollout ranks through the SAME store rendezvous the stock topology
    #   uses (``all_gather_obj``), fired concurrently with the actor-side
    #   stage — mirroring kimi's ``gather_metas`` collective.
    # * ``receive_weights_version`` streams bucket by bucket via
    #   ``transfer_sync_read`` DIRECTLY from the staged buffer — no chain,
    #   no barrier, nothing shared between consumers: per-replica pulls are
    #   per-consumer by construction (which is why this engine needs no
    #   replica subgroups — see ``build_topology``).
    #
    # Cluster-validated TODO: the runtime ``batch_register_memory`` /
    # ``unregister_memory`` semantics across mooncake-transfer-engine
    # versions, and the RDMA transport itself (TODO-6 runbook).

    def _versioned_state(self):
        # lazy state so __new__-constructed instances (tests) work too
        if not hasattr(self, "_staged_versions"):
            self._staged_versions: dict[int, dict[str, Any]] = {}  # actor: version -> staged buffer
            self._versioned_descriptors: dict[int, dict[str, Any]] = {}  # rollout: version -> descriptor
        return self._staged_versions, self._versioned_descriptors

    @torch.no_grad()
    async def stage_version(
        self,
        version: int,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        """Register this rank's current weights as a pullable version.

        Actor-side counterpart of ``send_weights`` with the SAME
        single-sender convention: engine rank 0 (the only actor rank in the
        store world) stages the weights it is handed; other actor ranks
        drain their generators (their weights are not part of this engine's
        transfer, exactly as in the stock push path).
        """
        staged, _ = self._versioned_state()
        if self.rank < 0:
            for _ in weights:
                pass
            logger.info(f"stage v{version} rank={self.rank} drained (non-sender actor rank)")
            return {"staged_bytes": 0, "staged_params": 0}

        start_time = time.time()
        tensors: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            tensors.append((name, weight.to(self.rollout_dtype)))
        if not tensors:
            raise ValueError("refusing to stage an empty weight set")

        # pack whole tensors per bucket (stock send_weights convention: a
        # tensor never spans buckets) into ONE pinned host buffer per version
        # — the Laminar relay placement, and what the host-memory quota
        # (relay controller's max_staged_bytes) accounts
        nbytes = sum(t.numel() * t.element_size() for _, t in tensors)
        # PIN when an accelerator is present (the RDMA-preferred host
        # placement); pageable otherwise so CPU-only hosts still run the
        # protocol (tests) — registration pins pages either way
        pin = get_torch_device().is_available()
        buf = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=pin)
        view = buf.view(-1)
        buckets: list[dict[str, Any]] = []
        offset = 0
        cur_len = 0
        cur_tensors: dict[str, tuple[int, Any, Any]] = {}
        per_tensor: dict[str, tuple[int, Any, Any]] = {}
        for name, tensor in tensors:
            flat = tensor.detach().contiguous().view(-1).view(torch.uint8)
            size = flat.numel()
            assert size <= self.bucket_size, (
                f"Weight {name}({tensor.shape}, {tensor.dtype}) is too large to fit in the bucket."
            )
            if cur_len + size > self.bucket_size and cur_len > 0:
                buckets.append({"offset": offset - cur_len, "size": cur_len, "tensors": cur_tensors})
                cur_tensors, cur_len = {}, 0
            view[offset : offset + size].copy_(flat, non_blocking=True)
            cur_tensors[name] = (cur_len, tensor.shape, tensor.dtype)
            per_tensor[name] = (offset, tensor.shape, tensor.dtype)
            offset += size
            cur_len += size
        if cur_len > 0:
            buckets.append({"offset": offset - cur_len, "size": cur_len, "tensors": cur_tensors})
        get_torch_device().synchronize()

        # register + KEEP registered: this pinned buffer is the relay memory
        # for the version until unstage_version retires it
        ret = self.engine.batch_register_memory([buf.data_ptr()], [nbytes])
        assert ret == 0, f"batch_register_memory failed ret={ret} for v{version}"
        staged[version] = {"buf": buf, "ptr": buf.data_ptr(), "nbytes": nbytes}

        # publish the descriptor to every rollout rank: the same store
        # rendezvous the topology init uses — rollout ranks run
        # gather_version_metas(version) concurrently (mirroring kimi's
        # collective gather_metas on both sides of the stage)
        descriptor = {
            "kind": "mooncake",
            "session_id": self.session_id,
            "ptr": buf.data_ptr(),
            "nbytes": nbytes,
            "buckets": buckets,
        }
        info_list = self.store.all_gather_obj(descriptor)
        assert info_list[0] is descriptor or info_list[0] == descriptor

        logger.info(
            f"Rank {self.rank} stage v{version}: {len(per_tensor)} params, {nbytes} bytes, "
            f"{len(buckets)} bucket(s), {time.time() - start_time:.2f}s"
        )
        return {"staged_bytes": nbytes, "staged_params": len(per_tensor)}

    def gather_version_metas(self, version: int):
        """Rollout-side participation in a version's descriptor exchange.

        Must run CONCURRENTLY with the actor rank 0's ``stage_version`` —
        the store ``all_gather_obj`` is the rendezvous (the same collective
        pattern the topology init uses for buffer_info).
        """
        _, descriptors = self._versioned_state()
        info_list = self.store.all_gather_obj(None)
        descriptor = info_list[0]
        if not isinstance(descriptor, dict) or descriptor.get("kind") != "mooncake":
            raise RuntimeError(
                f"gather_version_metas({version}) rendezvoused without a mooncake "
                f"descriptor — did the actor rank 0 run stage_version concurrently?"
            )
        descriptors[version] = descriptor

    @torch.no_grad()
    async def receive_weights_version(
        self,
        version: int,
        replica_id: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Pull a staged version into this rollout rank (anytime, repeatable).

        Streams bucket by bucket via ``transfer_sync_read`` DIRECTLY from the
        actor's registered staging buffer into this rank's registered
        ``buf`` — no rank chain, no collective barrier, nothing shared with
        other consumers. ``replica_id`` is accepted for driver-dispatch
        parity (the driver scopes the pull by dispatching to one replica's
        worker group) and is otherwise unused: this engine has no subgroups
        to scope because per-consumer reads never synchronize the fleet.
        """
        _, descriptors = self._versioned_state()
        descriptor = descriptors.get(version)
        if descriptor is None:
            raise LookupError(
                f"version {version} was never gathered on this rank "
                f"(known versions: {sorted(descriptors)})"
            )

        start_time = time.time()
        total_bytes = 0
        current = self.buf[: self.bucket_size]
        for bucket in descriptor["buckets"]:
            size = bucket["size"]
            if size <= 0:
                continue
            ret = self.engine.transfer_sync_read(
                descriptor["session_id"],
                current.data_ptr(),
                descriptor["ptr"] + bucket["offset"],
                size,
            )
            assert ret == 0, f"transfer_sync_read failed ret={ret} for v{version}"
            total_bytes += size
            for name, (rel_offset, shape, dtype) in bucket["tensors"].items():
                t_size = dtype.itemsize * shape.numel()
                tensor = current[rel_offset : rel_offset + t_size].view(dtype=dtype).view(shape)
                yield name, tensor
            get_torch_device().synchronize()

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024) if time_cost > 0 else 0.0
        logger.info(
            f"Rank {self.rank} receive v{version} done, total bytes: {total_bytes} "
            f"time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    def unstage_version(self, version: int):
        """Retire a staged version (actor side): unregister and free."""
        staged, _ = self._versioned_state()
        entry = staged.pop(version, None)
        if entry is None or self.rank < 0:
            return
        ret = self.engine.unregister_memory(entry["ptr"])
        assert ret == 0, f"unregister_memory failed ret={ret} for v{version}"
        logger.info(f"Rank {self.rank} unstage v{version}: {entry['nbytes']} bytes released")

    def drop_version(self, version: int):
        """Drop a retired version's descriptor snapshot (rollout side)."""
        _, descriptors = self._versioned_state()
        descriptors.pop(version, None)
