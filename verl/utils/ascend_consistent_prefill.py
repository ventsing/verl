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

"""Opt-in Ascend prefill using the same recurrent and attention primitives as decode.

Native cache writes and per-query causal KV lengths are preserved. Recurrent
updates retain FP32 state; attention queries use cached one-query sequences.
The default path imports no device backend and installs nothing. Current
hardware validation covers BF16 Qwen3.5-9B/TP2 with FP32 SSM caches and eager
execution, without PCP/DCP, sliding windows, sinks or speculative decoding.
"""

import copy
import functools
import inspect
import os

_STATE = None


def make_recurrent_prefill_dispatch(torch_module, normalize, recurrent, metadata_for, state, chunk_size=1):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 16:
        raise ValueError("Native recurrent candidate windows must be in1..16")

    def dispatch(q, k, v, g, beta, initial_state, cu_seqlens, scale):
        if initial_state is None or initial_state.dtype != torch_module.float32:
            raise ValueError("Candidate requires an actual FP32 incoming recurrent state")
        if any(x.ndim != 4 for x in [q, k, v, initial_state]) or q.shape[0] != 1:
            raise ValueError("Expected flattened four-dimensional GDN prefill tensors")
        if any(x.dtype != torch_module.bfloat16 for x in [q, k, v, beta]):
            raise ValueError("Native recurrent Q/K/V/beta must be BF16")
        if g.dtype != torch_module.float32:
            raise ValueError("Native recurrent decay must be FP32")
        if tuple(q.shape) != tuple(k.shape) or q.shape[1] != v.shape[1] or q.shape[1] == 0:
            raise ValueError("Invalid GDN token or query/key dimensions")
        if any(size <= 0 for size in [initial_state.shape[0], q.shape[2], k.shape[-1], v.shape[2], v.shape[-1]]):
            raise ValueError("GDN state/head dimensions must be positive")
        if any(x.device != q.device for x in [k, v, g, beta, initial_state]):
            raise ValueError("GDN candidate tensors must share one device")
        if tuple(initial_state.shape[1:]) != (v.shape[2], v.shape[-1], k.shape[-1]):
            raise ValueError("State must use native [N,Hv,Dv,Dk] layout")
        if v.shape[2] % q.shape[2] or tuple(g.shape) != tuple(beta.shape) or tuple(g.shape) != v.shape[:3]:
            raise ValueError("Invalid value heads or decay/beta dimensions")
        metadata = metadata_for(cu_seqlens)
        boundaries = tuple(metadata.cu_seqlens_host)
        if len(boundaries) != initial_state.shape[0] + 1 or boundaries[0] != 0 or boundaries[-1] != q.shape[1]:
            raise ValueError("Native host sequence boundaries disagree with state/token counts")
        pairs = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
        if any(type(x) is not int for x in boundaries) or any(a > b for a, b in pairs):
            raise ValueError("Native host sequence boundaries are invalid")
        # Match the decode normalization entry point. Do not use the chunk WY
        # representation, its BF16 solve/W/U intermediates, or transpose state.
        query, key = normalize(q).squeeze(0), normalize(k).squeeze(0)
        if tuple(query.shape) != tuple(q.shape[1:]) or tuple(key.shape) != tuple(k.shape[1:]):
            raise RuntimeError("Unexpected query/key normalization shape")
        if query.dtype != q.dtype or key.dtype != k.dtype or query.device != q.device or key.device != k.device:
            raise RuntimeError("Unexpected query/key normalization dtype/device")
        value, decay, strength = v.squeeze(0), g.squeeze(0), beta.squeeze(0)
        final_state = initial_state.clone()
        output = torch_module.empty_like(v)
        for sequence, (begin, end) in enumerate(pairs):
            # Cache only constant metadata inside this prefill dispatch. State
            # indices address the original sequence, once for every token.
            constants = {}
            for offset in range(begin, end, chunk_size):
                stop = min(offset + chunk_size, end)
                length = stop - offset
                if length not in constants:
                    constants[length] = (
                        torch_module.tensor([0, length], dtype=torch_module.int32, device=q.device),
                        torch_module.full((length,), sequence, dtype=torch_module.int32, device=q.device),
                    )
                lengths, indices = constants[length]
                block = recurrent(
                    query=query[offset:stop],
                    key=key[offset:stop],
                    value=value[offset:stop],
                    g=decay[offset:stop],
                    beta=strength[offset:stop],
                    state=final_state,
                    scale=scale,
                    actual_seq_lengths=lengths,
                    ssm_state_indices=indices,
                )
                if tuple(block.shape) != tuple(value[offset:stop].shape) or block.dtype != v.dtype:
                    raise RuntimeError("Native recurrence returned unexpected output shape/dtype")
                output[:, offset:stop] = block.unsqueeze(0)
                state["native_kernel_calls"] += 1
                state["max_tokens_per_native_call"] = max(state["max_tokens_per_native_call"], length)
        state.update(
            calls=state["calls"] + 1,
            last_initial_dtype=str(initial_state.dtype),
            last_final_dtype=str(final_state.dtype),
            native_host_metadata_used=True,
        )
        return output, final_state

    dispatch._verl_ascend_consistent_prefill = state
    return dispatch


def build_prefill_token_plan(query_ends, kv_lengths):
    if len(query_ends) != len(kv_lengths) or not query_ends:
        raise ValueError("Nonempty native cumulative query lengths and KV lengths must agree")
    if any(type(x) is not int for x in [*query_ends, *kv_lengths]):
        raise ValueError("Native host lengths must contain actual integers")
    previous, requests, contexts, lengths = 0, [], [], []
    for request, (end, total) in enumerate(zip(query_ends, kv_lengths, strict=True)):
        count = end - previous
        if count < 0 or total < count:
            raise ValueError("Invalid cumulative query lengths or insufficient causal history")
        requests.extend([request] * count)
        contexts.extend(range(total - count + 1, total + 1))
        lengths.append(count)
        previous = end
    return requests, contexts, lengths


def make_attention_prefill_dispatch(torch, original, cached_params, fia, settings, state, window=16):
    if type(window) is not int or not 1 <= window <= 64:
        raise ValueError("Candidate query batches must be in1..64")

    @functools.wraps(original)
    def dispatch(query, key, value, metadata, output, kv_cache=None):
        requests, contexts, lengths = build_prefill_token_plan(
            list(metadata.actual_seq_lengths_q), list(metadata.seq_lens_list)
        )
        if not any(length > 1 for length in lengths):
            state["native_decode_bypasses"] += 1
            return original(query, key, value, metadata, output, kv_cache)
        if not metadata.causal:
            raise ValueError("Uniform prefill candidate requires causal attention")
        total = len(requests)
        if query.shape[0] < total or output.shape[0] < total or query.dtype != torch.bfloat16:
            raise ValueError("Invalid actual query count, output size or dtype")
        cached_key, cached_value, block_size, block_table, actual_kv_lengths = cached_params(
            key, value, metadata, kv_cache
        )
        if block_table is None or list(actual_kv_lengths) != list(metadata.seq_lens_list):
            raise ValueError("Native cache view did not preserve actual causal history lengths")
        if block_table.shape[0] < len(lengths) or block_table.device != query.device:
            raise ValueError("Native block-table rows/device do not match requests")
        indices = torch.tensor(requests, dtype=torch.int64, device=query.device)
        token_tables = block_table.index_select(0, indices)
        for offset in range(0, total, window):
            end = min(offset + window, total)
            actual, _ = fia(
                query=query[offset:end],
                key=cached_key,
                value=cached_value,
                atten_mask=metadata.attn_mask,
                block_table=token_tables[offset:end],
                block_size=block_size,
                input_layout="TND",
                actual_seq_lengths=list(range(1, end - offset + 1)),
                actual_seq_lengths_kv=contexts[offset:end],
                sparse_mode=3,
                num_heads=settings["num_heads"],
                num_key_value_heads=settings["num_kv_heads"],
                scale=settings["scale"],
            )
            target = output[offset:end]
            if actual.numel() != target.numel() or actual.dtype != target.dtype:
                raise RuntimeError("FIA returned an unexpected output shape or dtype")
            target.copy_(actual.reshape_as(target))
            state["native_operator_calls"] += 1
        state["prefill_dispatches"] += 1
        state["prefill_tokens"] += total
        state["causal_per_query_KV_lengths_used"] = True
        return output

    return dispatch


def get_ascend_consistent_prefill_state():
    if _STATE is None:
        return {"installed": False}
    return {
        "installed": True,
        "native_constructor_modified": False,
        "actual_attention_operator": _STATE["actual_attention_operator"],
        "recurrent": dict(_STATE["recurrent"]),
        "attention_layers": [dict(row) for row in _STATE["attention_layers"]],
    }


def prepare_ascend_consistent_prefill(config, model):
    """Install only on an explicitly opted-in, fully constructed native worker."""
    global _STATE
    requested = os.environ.get("VERL_ASCEND_CONSISTENT_PREFILL", "0")
    if requested == "0":
        return {"installed": False, "reason": "not_requested"}
    if requested != "1":
        raise ValueError("VERL_ASCEND_CONSISTENT_PREFILL must be0 or1")
    if os.environ.get("VERL_ASCEND_GDN_PREFILL_FP32", "0") != "0":
        raise ValueError("Consistent prefill and legacy VERL_ASCEND_GDN_PREFILL_FP32 are mutually exclusive")
    if _STATE is not None:
        if _STATE["model_id"] != id(model):
            raise RuntimeError("This process already owns a different consistent-prefill model")
        return {"installed": True, "already_prepared": True}

    from verl.utils.ascend_gdn_prefill_fp32 import _validate_config, select_native_chunk_metadata

    _validate_config(config)
    if not config.model_config.enforce_eager or config.speculative_config is not None:
        raise ValueError("Consistent prefill requires eager execution without speculative decoding")
    if config.parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Consistent prefill requires decode context parallel size1")
    text = config.model_config.hf_text_config
    if getattr(text, "linear_key_head_dim", None) != 128 or getattr(text, "linear_value_head_dim", None) != 128:
        raise ValueError("Consistent prefill requires128-dimensional recurrent key/value heads")

    import torch
    import torch_npu
    from vllm.forward_context import get_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.ops import gdn

    fia = torch_npu.npu_fused_infer_attention_score
    if str(fia) != "batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant":
        raise ValueError("Consistent prefill requires the actual Ascend batch-invariant FIA operator")
    owner = gdn.AscendGatedDeltaNetAttention
    descriptor = inspect.getattr_static(owner, "_chunk_gated_delta_rule_fused")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("Unsupported native GDN prefill descriptor")
    for marker in (
        "_verl_ascend_gdn_prefill_fp32",
        "_verl_recurrent_prefill_candidate",
        "_verl_ascend_consistent_prefill",
    ):
        if getattr(descriptor.__func__, marker, None) is not None:
            raise RuntimeError("Another GDN prefill dispatcher is already installed")
    recurrent_op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    layers = [layer for layer in model.modules() if hasattr(layer, "impl") and hasattr(layer, "layer_name")]
    expected = list(getattr(text, "layer_types", [])).count("full_attention")
    if not layers or (expected and len(layers) != expected):
        raise ValueError("Native full-attention layer count does not match the model configuration")
    for layer in layers:
        impl = layer.impl
        if (
            type(impl).__name__ != "AscendAttentionBackendImpl"
            or impl.sinks is not None
            or impl.sliding_window is not None
            or impl.head_size != 256
        ):
            raise ValueError("Unsupported attention backend, sinks, sliding window or head dimension")

    recurrent_state = dict(
        installed=True,
        configured_window=16,
        calls=0,
        native_kernel_calls=0,
        max_tokens_per_native_call=0,
        native_host_metadata_used=False,
    )
    recurrent_dispatch = make_recurrent_prefill_dispatch(
        torch,
        gdn.l2norm_fwd,
        recurrent_op,
        lambda cu: select_native_chunk_metadata(get_forward_context().attn_metadata, cu),
        recurrent_state,
        16,
    )
    replacements, attention_states = [], []
    for layer in layers:
        impl = layer.impl
        state = dict(
            layer=layer.layer_name,
            installed=True,
            window=64,
            native_operator_calls=0,
            prefill_dispatches=0,
            prefill_tokens=0,
            native_decode_bypasses=0,
            causal_per_query_KV_lengths_used=False,
        )

        def cached_params(key, value, metadata, cache, instance=impl):
            native = copy.copy(metadata)
            native.attn_state = AscendAttentionState.DecodeOnly
            return instance._get_fia_params(key, value, native, cache)

        dispatch = make_attention_prefill_dispatch(
            torch,
            impl.forward_fused_infer_attention,
            cached_params,
            fia,
            dict(num_heads=impl.num_heads, num_kv_heads=impl.num_kv_heads, scale=impl.scale),
            state,
            64,
        )
        replacements.append((impl, dispatch))
        attention_states.append(state)

    # All descriptors, operators, shapes and backends have been validated.
    # Native forward() still writes KV caches before calling these dispatchers.
    for impl, dispatch in replacements:
        impl.forward_fused_infer_attention = dispatch
    owner._chunk_gated_delta_rule_fused = staticmethod(recurrent_dispatch)
    # The replacement uses the existing native decode operator, independently
    # of whether the optional BF16 fused chunk operator is available.
    owner._fused_chunk_available = True
    _STATE = dict(
        model_id=id(model),
        recurrent=recurrent_state,
        attention_layers=attention_states,
        actual_attention_operator=str(fia),
    )
    return {"installed": True, "already_prepared": False, "native_constructor_modified": False}
