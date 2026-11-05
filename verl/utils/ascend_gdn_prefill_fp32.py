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

"""Opt-in FP32 recurrent-state prefill for Ascend Gated DeltaNet.

The fused chunk wrapper in some Ascend versions rounds incoming and final
recurrent states to BF16 even when the cache is FP32. Use the native alternate
pipeline with its real request metadata to preserve FP32 state boundaries.
Nothing is installed on import, and the default path imports no device backend.
"""

import inspect
import os
import sys


def _validate_config(config):
    cache = getattr(config, "cache_config", None)
    if str(getattr(cache, "mamba_ssm_cache_dtype", None)) not in {"float32", "torch.float32"}:
        raise ValueError("Ascend FP32 GDN prefill requires mamba_ssm_cache_dtype=float32")
    parallel = getattr(config, "parallel_config", None)
    if getattr(parallel, "prefill_context_parallel_size", None) != 1:
        raise ValueError("Ascend FP32 GDN prefill currently requires prefill context parallel size1")


def select_native_chunk_metadata(context_metadata, cu_seqlens):
    if cu_seqlens is None:
        raise ValueError("Ascend FP32 GDN prefill requires native cumulative sequence lengths")
    candidates = context_metadata.values() if isinstance(context_metadata, dict) else [context_metadata]
    matches = {}
    for metadata in candidates:
        if getattr(metadata, "prefill_query_start_loc", None) is not cu_seqlens:
            continue
        chunk = getattr(getattr(metadata, "non_spec_prefill_metadata", None), "chunk", None)
        if chunk is not None:
            matches[id(chunk)] = chunk
    if len(matches) != 1:
        raise RuntimeError(f"Expected one native chunk metadata object for this request tensor, got {len(matches)}")
    result = next(iter(matches.values()))
    if not result.chunk_indices_chunk64_host:
        raise RuntimeError("Native chunk indices missing for a nonempty prefill")
    return result


def make_fp32_prefill_dispatch(torch_module, pipeline, get_context_metadata, state):
    """Keep native Q/K/V and metadata; transpose only recurrent-state layout."""

    def dispatch(q, k, v, g, beta, initial_state, cu_seqlens, scale):
        if initial_state is None or initial_state.dtype != torch_module.float32:
            raise ValueError("Ascend FP32 GDN prefill requires an actual FP32 incoming state")
        if initial_state.ndim != 4 or q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Expected four dimensional GDN state and Q/K/V")
        if tuple(initial_state.shape[-2:]) != (v.shape[-1], k.shape[-1]):
            raise ValueError("Native GDN state must use [N, Hv, Dv, Dk] layout")
        if initial_state.shape[1] != v.shape[2]:
            raise ValueError("GDN recurrent state and value head counts must agree")
        metadata = select_native_chunk_metadata(get_context_metadata(), cu_seqlens)
        triton_state = initial_state.transpose(-1, -2).contiguous()
        output, final_state = pipeline(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=triton_state,
            output_final_state=True, cu_seqlens=cu_seqlens, prebuilt_meta=metadata,
            head_first=False, use_qk_l2norm_in_kernel=True,
        )
        expected_shape = (*initial_state.shape[:-2], initial_state.shape[-1], initial_state.shape[-2])
        if final_state is None or final_state.dtype != torch_module.float32:
            raise RuntimeError("Ascend GDN pipeline did not return an FP32 final state")
        if tuple(final_state.shape) != expected_shape:
            raise RuntimeError("Ascend GDN pipeline returned an unexpected state layout")
        final_state = final_state.transpose(-1, -2).contiguous()
        state.update(calls=state["calls"] + 1, last_initial_dtype=str(initial_state.dtype),
                     last_final_dtype=str(final_state.dtype), exact_native_chunk_metadata_used=True)
        return output, final_state

    dispatch._verl_ascend_gdn_prefill_fp32 = state
    return dispatch


def install_fp32_prefill(config, torch_module, backend, get_context_metadata):
    """Install once after worker initialization, before the first request."""
    _validate_config(config)
    owner = backend.AscendGatedDeltaNetAttention
    descriptor = inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unsupported Ascend GDN prefill wrapper")
    previous = descriptor.__func__
    installed = getattr(previous, "_verl_ascend_gdn_prefill_fp32", None)
    if installed is not None:
        return dict(installed, already_installed=True)
    if getattr(previous, "_verl_prefix_precision_experiment", False):
        raise RuntimeError("Cannot combine production FP32 prefill with an experimental precision dispatcher")
    state = dict(installed=True, calls=0, native_constructor_modified=False,
                 per_stage_synchronization=False, extra_device_probe=False)

    def pipeline(**kwargs):
        # Resolve at call time so existing observation hooks retain their ownership.
        return backend.chunk_gated_delta_rule(**kwargs)

    dispatch = make_fp32_prefill_dispatch(torch_module, pipeline, get_context_metadata, state)
    owner._chunk_gated_delta_rule_fused = staticmethod(dispatch)
    return dict(state, already_installed=False)


def prepare_ascend_fp32_gdn_prefill(config):
    """Enable only when VERL_ASCEND_GDN_PREFILL_FP32=1 is explicitly requested."""
    if os.environ.get("VERL_ASCEND_GDN_PREFILL_FP32", "0") != "1":
        return dict(installed=False, reason="not_requested")
    _validate_config(config)
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ops import gdn

    return install_fp32_prefill(config, torch, gdn, lambda: get_forward_context().attn_metadata)


def get_ascend_gdn_prefill_precision_state():
    """Read only already-loaded metadata, without importing a device backend."""
    backend = sys.modules.get("vllm_ascend.ops.gdn")
    if backend is None:
        return dict(installed=False)
    owner = getattr(backend, "AscendGatedDeltaNetAttention", None)
    method = getattr(owner, "_chunk_gated_delta_rule_fused", None)
    state = getattr(method, "_verl_ascend_gdn_prefill_fp32", None)
    return dict(state) if state is not None else dict(installed=False)
