# Copyright 2026 verl contributors
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

"""CPU recurrence/layout checks; these do not establish Ascend kernel parity."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.ascend_consistent_prefill import (
    make_recurrent_prefill_dispatch as make_dispatch,
)
from verl.utils.ascend_consistent_prefill import (
    prepare_ascend_consistent_prefill,
)


def normalized(x):
    return torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype)


def reference(q, k, v, g, beta, initial, scale):
    current = initial.clone()
    head_map = torch.arange(v.shape[1]) // (v.shape[1] // q.shape[1])
    values = []
    for token in range(len(q)):
        query, key = q[token, head_map].float(), k[token, head_map].float()
        current = current * torch.exp(g[token])[:, None, None]
        estimate = torch.einsum("hvk,hk->hv", current, key)
        delta = (v[token].float() - estimate) * beta[token].float()[:, None]
        current = current + delta[:, :, None] * key[:, None, :]
        values.append(torch.einsum("hvk,hk->hv", current, query * scale).to(v.dtype))
    return torch.stack(values) if values else v.clone(), current


def inputs():
    generator = torch.Generator().manual_seed(17)

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    boundaries = (0, 3, 3, 40)
    return dict(
        q=rand(1, 40, 2, 4).bfloat16(),
        k=rand(1, 40, 2, 4).bfloat16(),
        v=rand(1, 40, 4, 3).bfloat16(),
        g=-rand(1, 40, 4).abs(),
        beta=torch.sigmoid(rand(1, 40, 4)).bfloat16(),
        initial_state=rand(3, 4, 3, 4),
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
        scale=0.5,
    ), boundaries


@pytest.mark.parametrize("window", [1, 2, 8, 16])
def test_real_CPU_recurrence_preserves_state_and_sequence_boundaries(window):
    args, boundaries = inputs()
    original = args["initial_state"].clone()
    launches = []

    def kernel(**kw):
        length = len(kw["query"])
        assert kw["actual_seq_lengths"].tolist() == [0, length]
        indices = kw["ssm_state_indices"].tolist()
        assert len(indices) == length and len(set(indices)) == 1
        assert length <= window <= 16
        sequence = indices[0]
        assert kw["state"].data_ptr() != args["initial_state"].data_ptr()
        out, final = reference(
            kw["query"], kw["key"], kw["value"], kw["g"], kw["beta"], kw["state"][sequence], kw["scale"]
        )
        kw["state"][sequence].copy_(final)
        launches.append((sequence, length))
        return out

    state = dict(calls=0, native_kernel_calls=0, max_tokens_per_native_call=0)
    dispatch = make_dispatch(
        torch, normalized, kernel, lambda cu: SimpleNamespace(cu_seqlens_host=boundaries), state, window
    )
    output, final = dispatch(**args)
    expected_output, expected_state = torch.empty_like(output), original.clone()
    for sequence, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        result, updated = reference(
            normalized(args["q"]).squeeze(0)[begin:end],
            normalized(args["k"]).squeeze(0)[begin:end],
            args["v"].squeeze(0)[begin:end],
            args["g"].squeeze(0)[begin:end],
            args["beta"].squeeze(0)[begin:end],
            original[sequence],
            args["scale"],
        )
        expected_output[:, begin:end] = result.unsqueeze(0)
        expected_state[sequence] = updated
    assert torch.equal(output, expected_output) and torch.equal(final, expected_state)
    assert torch.equal(args["initial_state"], original)
    assert final.dtype == torch.float32 and output.dtype == torch.bfloat16
    assert state["native_kernel_calls"] == len(launches) and state["calls"] == 1
    assert 1 not in [sequence for sequence, _ in launches], "Empty sequence state must not be updated"
    assert state["max_tokens_per_native_call"] <= window


@pytest.mark.parametrize("window", [0, 17, True, 1.5])
def test_invalid_native_window_rejected(window):
    with pytest.raises(ValueError):
        make_dispatch(torch, None, None, None, {}, window)


@pytest.mark.parametrize(
    "change", ["bf16_state", "wrong_state_layout", "wrong_boundaries", "fp32_beta", "other_device"]
)
def test_invalid_inputs_fail_before_native_kernel(change):
    args, boundaries = inputs()
    if change == "bf16_state":
        args["initial_state"] = args["initial_state"].bfloat16()
    elif change == "wrong_state_layout":
        args["initial_state"] = args["initial_state"].transpose(-1, -2)
    elif change == "wrong_boundaries":
        boundaries = (0, 3, 3, 41)
    elif change == "fp32_beta":
        args["beta"] = args["beta"].float()
    else:
        args["v"] = args["v"].to("meta")

    def forbidden(**kw):
        raise AssertionError("Malformed metadata reached native kernel")

    dispatch = make_dispatch(
        torch, normalized, forbidden, lambda cu: SimpleNamespace(cu_seqlens_host=boundaries), {}, 1
    )
    with pytest.raises(ValueError):
        dispatch(**args)


def test_disabled_path_never_imports_device_backend(monkeypatch):
    import sys

    monkeypatch.delenv("VERL_ASCEND_CONSISTENT_PREFILL", raising=False)
    before = {name for name in sys.modules if name.startswith(("vllm_ascend", "torch_npu"))}
    assert prepare_ascend_consistent_prefill(None, None)["installed"] is False
    assert {name for name in sys.modules if name.startswith(("vllm_ascend", "torch_npu"))} == before


def test_legacy_dispatcher_conflict_is_rejected_before_backend_import(monkeypatch):
    monkeypatch.setenv("VERL_ASCEND_CONSISTENT_PREFILL", "1")
    monkeypatch.setenv("VERL_ASCEND_GDN_PREFILL_FP32", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_ascend_consistent_prefill(None, None)


def test_invalid_opt_in_is_rejected(monkeypatch):
    monkeypatch.setenv("VERL_ASCEND_CONSISTENT_PREFILL", "unexpected")
    with pytest.raises(ValueError, match="must be0 or1"):
        prepare_ascend_consistent_prefill(None, None)


@pytest.mark.parametrize("unsupported", ["nonEager", "DCP", "SSM"])
def test_unsupported_configuration_fails_before_native_backend(unsupported, monkeypatch):
    monkeypatch.setenv("VERL_ASCEND_CONSISTENT_PREFILL", "1")
    monkeypatch.setenv("VERL_ASCEND_GDN_PREFILL_FP32", "0")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="float32"),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=None,
    )
    if unsupported == "nonEager":
        config.model_config.enforce_eager = False
    elif unsupported == "DCP":
        config.parallel_config.decode_context_parallel_size = 2
    else:
        config.cache_config.mamba_ssm_cache_dtype = "bfloat16"
    with pytest.raises(ValueError):
        prepare_ascend_consistent_prefill(config, object())
