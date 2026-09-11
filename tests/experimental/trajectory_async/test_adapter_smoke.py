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
"""Adapter smoke tests: REAL kimi/mooncake backends over duck-typed stub engines.

These are the debugging layer between the CPU orchestration tests and a
cluster run. They execute the actual adapter code paths (tensor packing,
registration calls, chunked reads, reassembly, eviction) against stubs
that reproduce the exact engine surface the adapters were written
against:

* kimi — ``KIMICheckpointEngine.parameter_server``:
  ``register_checkpoint`` / ``gather_metas`` / ``unregister_checkpoint``
  and the patched async-generator ``receive_tensor(checkpoint_name,
  ranks_group, ranks, bucket_size)``;
* mooncake — ``MooncakeCheckpointEngine``: ``engine`` (TransferEngine
  with ``batch_register_memory`` / ``unregister_memory`` /
  ``transfer_sync_read``), ``session_id``, ``bucket_size`` and the
  registered double ``buf``.

A pass here means the adapter CALL SEQUENCES and tensor handling are
correct; what remains unverified for a cluster is transport behavior
only (RDMA, process-group topology). Requires torch — skipped without
it (the rest of the suite is stdlib-only).
"""

import asyncio
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.versioned_weight_store import (  # noqa: E402
    KimiP2PBackend,
    MooncakeP2PBackend,
    VersionedWeightStore,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch required for adapter smoke tests")
class TestKimiAdapterSmoke(unittest.TestCase):
    def test_register_retain_pull_evict(self):
        async def scenario():
            # --- stub: the patched ParameterServer surface
            class StubPS:
                def __init__(self):
                    self.checkpoints = {}
                    self.unregistered = []
                    self.gathered = []

                def register_checkpoint(self, name, named_tensors):
                    self.checkpoints[name] = dict(named_tensors)

                def gather_metas(self, name):
                    self.gathered.append(name)

                def unregister_checkpoint(self, name):
                    self.unregistered.append(name)
                    self.checkpoints.pop(name, None)

                async def receive_tensor(self, checkpoint_name, ranks_group, ranks, bucket_size):
                    assert checkpoint_name in self.checkpoints, (
                        f"read of unregistered checkpoint {checkpoint_name!r}"
                    )
                    for name, tensor in self.checkpoints[checkpoint_name].items():
                        yield name, tensor

            class StubKimiEngine:
                def __init__(self):
                    self.parameter_server = StubPS()
                    self.bucket_size = 1 << 20

            engine = StubKimiEngine()
            backend = KimiP2PBackend(engine, rollout_dtype=None)
            store = VersionedWeightStore(backend, keep_last=5)

            w1 = torch.arange(8, dtype=torch.float32)
            w2 = torch.ones(4, dtype=torch.bfloat16)
            await store.publish(1, [("w.a", w1), ("w.b", w2)])
            await store.publish(2, [("w.a", w1 + 1), ("w.b", w2 + 1)])

            ps = engine.parameter_server
            # multi-version retention: BOTH versions stay registered — the
            # stock kimi send_weights unregisters right after its barrier
            self.assertIn("actor:v1", ps.checkpoints)
            self.assertIn("actor:v2", ps.checkpoints)
            self.assertEqual(ps.unregistered, [])
            self.assertEqual(ps.gathered, ["actor:v1", "actor:v2"])

            # pinned pull of the old version through receive_tensor
            got = {}
            await store.pull(
                "replica-0",
                version=1,
                consumer_ctx={"ranks_group": "rollout-group-0", "ranks": [1, 2]},
                sink=lambda n, t: got.__setitem__(n, t.clone()),
            )
            self.assertTrue(torch.equal(got["w.a"], w1))
            self.assertTrue(torch.equal(got["w.b"], w2))

            # eviction unregisters exactly the evicted version
            await store.release(keep_last=1)
            self.assertEqual(ps.unregistered, ["actor:v1"])
            self.assertNotIn("actor:v1", ps.checkpoints)
            self.assertIn("actor:v2", ps.checkpoints)

        asyncio.run(scenario())

    def test_read_into_runs_on_consumer_engine(self):
        """The receiver's engine drives the pull (stock semantics): the
        constructor-side engine is the STAGER — read_into must execute
        receive_tensor on consumer_ctx["engine"], not on self.engine."""

        async def scenario():
            class StubPS:
                def __init__(self, owner):
                    self.owner = owner
                    self.checkpoints = {}

                def register_checkpoint(self, name, named_tensors):
                    self.checkpoints[name] = dict(named_tensors)

                def gather_metas(self, name):
                    pass

                def unregister_checkpoint(self, name):
                    self.checkpoints.pop(name, None)

                async def receive_tensor(self, checkpoint_name, ranks_group, ranks, bucket_size):
                    # the assertion: the receiver's OWN engine executes this
                    self.owner.receive_calls.append((checkpoint_name, ranks_group, ranks))
                    for name, tensor in self.checkpoints[checkpoint_name].items():
                        yield name, tensor

            class StubEngine:
                def __init__(self):
                    self.receive_calls = []
                    self.parameter_server = StubPS(self)
                    self.bucket_size = 1 << 20

            stager, consumer = StubEngine(), StubEngine()
            backend = KimiP2PBackend(stager, rollout_dtype=None)
            store = VersionedWeightStore(backend, keep_last=2)

            w = torch.arange(4, dtype=torch.float32)
            await store.publish(7, [("w", w)])
            # the consumer's PS mirrors the registration (in reality its
            # metas snapshot points at the stager's memory)
            consumer.parameter_server.checkpoints["actor:v7"] = {"w": w}

            got = {}
            await store.pull(
                "replica-0",
                version=7,
                consumer_ctx={
                    "engine": consumer,  # <- the receiver's engine
                    "ranks_group": "rollout-group-0",
                    "ranks": [3, 4],
                },
                sink=lambda n, t: got.__setitem__(n, t.clone()),
            )
            self.assertTrue(torch.equal(got["w"], w))
            # receive ran on the CONSUMER engine, scoped to its group/ranks
            self.assertEqual(consumer.receive_calls, [("actor:v7", "rollout-group-0", [3, 4])])
            self.assertEqual(stager.receive_calls, [])

        asyncio.run(scenario())


@unittest.skipUnless(HAS_TORCH, "torch required for adapter smoke tests")
class TestMooncakeAdapterSmoke(unittest.TestCase):
    def test_stage_per_version_direct_read_evict(self):
        async def scenario():
            # --- stub: the TransferEngine surface, with REAL copies — reads
            # actually move bytes from the source buffer to the destination,
            # so the per-bucket streaming path is verified end to end
            class StubTransferEngine:
                def __init__(self):
                    self.buffers = {}  # data_ptr -> tensor (registered)
                    self.unregistered = []
                    self.reads = []  # (src_ptr, length)

                def _find(self, ptr):
                    for base, buf in self.buffers.items():
                        if base <= ptr < base + buf.numel():
                            return base, buf
                    raise KeyError(f"ptr {ptr:#x} not registered")

                def batch_register_memory(self, ptrs, sizes):
                    # the caller registers AFTER we learn the tensor — the
                    # test associates buffers via register_buffer below
                    return 0

                def register_buffer(self, buf):
                    self.buffers[buf.data_ptr()] = buf

                def unregister_memory(self, ptr):
                    self.unregistered.append(ptr)
                    self.buffers.pop(ptr, None)
                    return 0

                def transfer_sync_read(self, src_session, dst_ptr, src_ptr, length):
                    src_base, src_buf = self._find(src_ptr)
                    dst_base, dst_buf = self._find(dst_ptr)
                    src_rel = src_ptr - src_base
                    dst_rel = dst_ptr - dst_base
                    dst_buf.view(-1)[dst_rel : dst_rel + length].copy_(
                        src_buf.view(-1)[src_rel : src_rel + length]
                    )
                    self.reads.append((src_ptr, length))
                    return 0

            BUCKET = 1 << 16

            class StubMooncakeEngine:
                def __init__(self, te, session_id):
                    self.engine = te
                    self.session_id = session_id
                    self.bucket_size = BUCKET
                    self.buf = torch.zeros(2 * BUCKET, dtype=torch.uint8)
                    te.register_buffer(self.buf)

            te = StubTransferEngine()
            actor_engine = StubMooncakeEngine(te, "actor-session")
            consumer_engine = StubMooncakeEngine(te, "replica-session")

            # MULTI-BUCKET weights: 3 tensors x 2048B with chunk=5120B ->
            # bucket 1 holds TWO whole tensors (4096B; the third no longer
            # fits: 4096+2048 > 5120), bucket 2 holds the last (2048B) —
            # mixed occupancy exercises per-bucket relative offsets; a
            # tensor never spans buckets
            chunk = 5120
            w1 = torch.arange(512, dtype=torch.float32)  # 2048B
            w2 = torch.full((512,), 7, dtype=torch.float32)  # 2048B
            w3 = torch.arange(1024, dtype=torch.bfloat16)  # 2048B
            weights_v1 = [("w.a", w1), ("w.b", w2), ("w.c", w3)]

            backend = MooncakeP2PBackend(actor_engine, staging_device="cpu", chunk_bytes=chunk)
            store = VersionedWeightStore(backend, keep_last=5)

            m1 = await store.publish(1, weights_v1)
            m2 = await store.publish(2, [(n, t + 1) for n, t in weights_v1])

            # the staging buffers are registered with the (real-copy) engine
            te.register_buffer(backend._buffers[m1.checkpoint_name])
            te.register_buffer(backend._buffers[m2.checkpoint_name])

            # per-version staging buffers: distinct registrations, both alive
            self.assertNotEqual(m1.descriptor["ptr"], m2.descriptor["ptr"])

            # whole-tensor buckets: 2 buckets, none spanning a tensor
            desc = m1.descriptor
            self.assertEqual(len(desc["buckets"]), 2)
            self.assertEqual(desc["buckets"][0]["size"], 4096)
            self.assertEqual(desc["buckets"][1]["size"], 2048)
            self.assertIn("w.a", desc["buckets"][0]["tensors"])
            self.assertIn("w.b", desc["buckets"][0]["tensors"])
            self.assertIn("w.c", desc["buckets"][1]["tensors"])

            # pull v1 end to end: values survive the per-bucket streaming
            # (read into buf[0:size] then yield the bucket's whole tensors)
            got = {}
            await store.pull(
                "replica-0",
                version=1,
                consumer_ctx={"engine": consumer_engine},
                sink=lambda n, t: got.__setitem__(n, t.clone()),
            )
            self.assertTrue(torch.equal(got["w.a"], w1))
            self.assertTrue(torch.equal(got["w.b"], w2))
            self.assertTrue(torch.equal(got["w.c"], w3))

            # repeatable pull (the version stays registered until retired)
            got2 = {}
            await store.pull(
                "replica-0",
                version=1,
                consumer_ctx={"engine": consumer_engine},
                sink=lambda n, t: got2.__setitem__(n, t.clone()),
            )
            self.assertTrue(torch.equal(got2["w.c"], w3))

            # direct read accounting: every read from v1's registered buffer,
            # total bytes == the version's nbytes
            self.assertTrue(te.reads)
            for src_ptr, length in te.reads:
                self.assertGreaterEqual(src_ptr, desc["ptr"])
                self.assertLess(src_ptr, desc["ptr"] + desc["nbytes"])
            self.assertEqual(sum(length for _, length in te.reads), desc["nbytes"])

            # eviction unregisters the evicted version's buffer
            await store.release(keep_last=1)
            self.assertEqual(te.unregistered, [m1.descriptor["ptr"]])
            self.assertNotIn(m1.descriptor["ptr"], te.buffers)
            self.assertIn(m2.descriptor["ptr"], te.buffers)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
