# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import SimpleNamespace

import pytest
import torch

from verl.utils.vllm.ascend_batch_invariant import make_batch_invariant_sum, prepare_ascend_batch_invariant_sum


class DeviceTensor:
    """CPU payload with NPU metadata; no device allocation or operator call."""

    def __init__(self, payload):
        self.payload = payload
        self.dtype, self.ndim = payload.dtype, payload.ndim
        self.device = SimpleNamespace(type="npu")

    def numel(self):
        return self.payload.numel()


def reducer():
    invariant_calls, native_calls = [], []

    def native(value, *args, **kwargs):
        native_calls.append((args, kwargs))
        return torch.sum(value.payload if isinstance(value, DeviceTensor) else value, *args, **kwargs)

    def invariant(value, dim, keepdim):
        assert value.dtype in {torch.float16, torch.bfloat16, torch.float32} and dim == -1
        invariant_calls.append((dim, keepdim))
        return torch.sum(value.payload, dim=dim, keepdim=keepdim)

    metadata_torch = SimpleNamespace(
        Tensor=DeviceTensor, float16=torch.float16, float32=torch.float32, bfloat16=torch.bfloat16
    )
    return make_batch_invariant_sum(metadata_torch, native, invariant), invariant_calls, native_calls


def test_sampler_boolean_rank_count_preserves_int64():
    reduce, invariant, native = reducer()
    values = torch.tensor([[True, False, True], [False, False, True]])
    result = reduce(DeviceTensor(values), -1)
    assert torch.equal(result, torch.tensor([2, 1])) and result.dtype == torch.int64
    assert not invariant and len(native) == 1


def test_visual_nonlast_axis_sum_preserves_shape_and_values():
    reduce, invariant, native = reducer()
    values = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    assert torch.equal(reduce(DeviceTensor(values), dim=0), values.sum(dim=0))
    assert not invariant and len(native) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 1])
def test_supported_floating_last_axis_still_uses_invariant(dtype, dim):
    reduce, invariant, native = reducer()
    values = torch.arange(6, dtype=dtype).reshape(2, 3)
    assert torch.equal(reduce(DeviceTensor(values), dim, True), values.sum(dim, True))
    assert invariant == [(-1, True)] and not native


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float64])
def test_unsupported_dtype_uses_native_semantics(dtype):
    reduce, invariant, native = reducer()
    values = torch.arange(6, dtype=dtype).reshape(2, 3)
    result = reduce(DeviceTensor(values), -1)
    expected = values.sum(-1)
    assert torch.equal(result, expected) and result.dtype == expected.dtype
    assert not invariant and len(native) == 1


def test_explicit_accumulation_dtype_and_out_use_native():
    reduce, invariant, native = reducer()
    values = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    destination = torch.empty(2, dtype=torch.float32)
    assert reduce(DeviceTensor(values), -1, dtype=torch.float32, out=destination) is destination
    assert torch.equal(destination, values.float().sum(-1))
    assert not invariant and len(native) == 1


@pytest.mark.parametrize("kwargs", [{}, {"dim": (0, 1)}, {"dim": -1}])
def test_empty_or_global_or_multi_axis_calls_preserve_native(kwargs):
    reduce, invariant, native = reducer()
    values = torch.empty(2, 0)
    assert torch.equal(reduce(DeviceTensor(values), **kwargs), values.sum(**kwargs))
    assert not invariant and len(native) == 1


def test_invalid_overload_preserves_torch_error():
    reduce, invariant, native = reducer()
    with pytest.raises(TypeError):
        reduce(DeviceTensor(torch.ones(2, 3)), -1, dim=-1)
    assert not invariant and len(native) == 1


def test_cpu_tensor_is_delegated_without_invariant_call():
    calls = []
    reduce = make_batch_invariant_sum(torch, torch.sum, lambda *args: calls.append(args))
    values = torch.tensor([1.0, 2.0])
    assert reduce(values).item() == 3 and not calls


def test_default_worker_installation_is_inert(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    previous_sum, previous_tensor_sum = torch.sum, torch.Tensor.sum
    assert not prepare_ascend_batch_invariant_sum()["installed"]
    assert torch.sum is previous_sum and torch.Tensor.sum is previous_tensor_sum


@pytest.mark.parametrize("dispatch_already_initialized", [False, True])
def test_prepare_and_repeat_respect_dispatch_lifecycle(monkeypatch, dispatch_already_initialized):
    previous = lambda *args, **kwargs: None

    class FakeTensor:
        pass

    FakeTensor.sum = previous
    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        sum=previous,
        float16=torch.float16,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        ops=SimpleNamespace(batch_invariant_ops=SimpleNamespace(npu_reduce_sum_batch_invariant=previous)),
    )
    backend = SimpleNamespace(
        reduce_sum=previous,
        torch_sum=previous,
        HAS_ASCENDC_BATCH_INVARIANT=True,
        _batch_invariant_LIB=object() if dispatch_already_initialized else None,
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm_ascend", SimpleNamespace(batch_invariant=backend))
    assert prepare_ascend_batch_invariant_sum() == {"installed": True, "already_prepared": False}
    wrapper = backend.reduce_sum
    if dispatch_already_initialized:
        assert fake_torch.sum is FakeTensor.sum is wrapper
    else:
        assert fake_torch.sum is FakeTensor.sum is previous
        # Simulate native enable_batch_invariant_mode reading its module global.
        fake_torch.sum = FakeTensor.sum = wrapper
        backend._batch_invariant_LIB = object()
    assert prepare_ascend_batch_invariant_sum() == {"installed": True, "already_prepared": True}
    fake_torch.sum = previous
    with pytest.raises(RuntimeError, match="another owner"):
        prepare_ascend_batch_invariant_sum()


def test_already_initialized_foreign_sum_is_not_overwritten(monkeypatch):
    previous, foreign = lambda *args: None, lambda *args: None

    class FakeTensor:
        pass

    FakeTensor.sum = previous
    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        sum=foreign,
        float16=torch.float16,
        float32=torch.float32,
        bfloat16=torch.bfloat16,
        ops=SimpleNamespace(batch_invariant_ops=SimpleNamespace(npu_reduce_sum_batch_invariant=previous)),
    )
    backend = SimpleNamespace(
        reduce_sum=previous, torch_sum=previous, HAS_ASCENDC_BATCH_INVARIANT=True, _batch_invariant_LIB=object()
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm_ascend", SimpleNamespace(batch_invariant=backend))
    with pytest.raises(RuntimeError, match="independently modified"):
        prepare_ascend_batch_invariant_sum()
    assert backend.reduce_sum is FakeTensor.sum is previous and fake_torch.sum is foreign
