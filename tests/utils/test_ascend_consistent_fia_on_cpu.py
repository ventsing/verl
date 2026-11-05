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

from types import SimpleNamespace

import pytest
import torch

from verl.utils.ascend_consistent_prefill import (
    build_prefill_token_plan as token_plan,
)
from verl.utils.ascend_consistent_prefill import (
    make_attention_prefill_dispatch as make_dispatch,
)


def state():
    return dict(native_decode_bypasses=0, native_operator_calls=0, prefill_dispatches=0, prefill_tokens=0)


def test_mixed_requests_retain_distinct_causal_histories():
    requests, contexts, counts = token_plan([1, 4, 4, 6], [3, 5, 0, 4])
    assert requests == [0, 1, 1, 1, 3, 3]
    assert contexts == [3, 3, 4, 5, 3, 4]
    assert counts == [1, 3, 0, 2]


@pytest.mark.parametrize("window", [1, 2, 4, 16, 64])
def test_against_independent_causal_attention_oracle_with_padding(window):
    generator = torch.Generator().manual_seed(113)
    q = torch.randn((8, 1, 2), generator=generator).bfloat16()
    key = torch.randn((9, 2, 1, 2), generator=generator).bfloat16()
    value = torch.randn((9, 2, 1, 2), generator=generator).bfloat16()
    tables = torch.tensor([[0, 1, 2], [3, 4, 5], [0, 0, 0], [6, 7, 8]], dtype=torch.int32)
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[1, 4, 4, 6], seq_lens_list=[3, 5, 0, 4], causal=True, attn_mask=object()
    )
    output = torch.full_like(q, 7)
    expected = output.clone()
    start = 0
    for request, end in enumerate(metadata.actual_seq_lengths_q):
        all_keys = key[tables[request].long()].reshape(-1, 2).float()
        all_values = value[tables[request].long()].reshape(-1, 2).float()
        for position in range(start, end):
            limit = metadata.seq_lens_list[request] - (end - start) + (position - start) + 1
            scores = q[position].float() @ all_keys[:limit].T
            expected[position] = (scores.softmax(-1) @ all_values[:limit]).bfloat16()
        start = end

    def forbidden(*args, **kwargs):
        raise AssertionError("A real prefill unexpectedly used the unmodified path")

    def cached(k, v, meta, cache):
        assert meta is metadata
        return key, value, 2, tables, list(meta.seq_lens_list)

    def fia(**kwargs):
        query, table = kwargs["query"], kwargs["block_table"]
        assert kwargs["atten_mask"] is metadata.attn_mask
        assert kwargs["actual_seq_lengths"] == list(range(1, len(query) + 1))
        answers = []
        for row, context in enumerate(kwargs["actual_seq_lengths_kv"]):
            k = kwargs["key"][table[row].long()].reshape(-1, 2)[:context].float()
            v = kwargs["value"][table[row].long()].reshape(-1, 2)[:context].float()
            answers.append(((query[row].float() @ k.T).softmax(-1) @ v).bfloat16())
        return torch.stack(answers), None

    counters = state()
    dispatch = make_dispatch(
        torch, forbidden, cached, fia, dict(num_heads=1, num_kv_heads=1, scale=1.0), counters, window
    )
    assert dispatch(q, None, None, metadata, output) is output
    assert torch.equal(output, expected)
    assert torch.equal(output[6:], torch.full_like(output[6:], 7))
    assert counters["prefill_tokens"] == 6 and counters["causal_per_query_KV_lengths_used"]


def test_native_decode_path_is_preserved():
    calls = []
    original = lambda *args: calls.append(args) or "native_result"
    metadata = SimpleNamespace(actual_seq_lengths_q=[1, 2], seq_lens_list=[5, 9])
    counters = state()
    dispatch = make_dispatch(torch, original, None, None, {}, counters)
    assert dispatch(None, None, None, metadata, None, "cache") == "native_result"
    assert calls == [(None, None, None, metadata, None, "cache")]
    assert counters["native_decode_bypasses"] == 1 and counters["prefill_dispatches"] == 0


@pytest.mark.parametrize("ends,history", [([], []), ([2, 1], [3, 3]), ([3], [2]), ([True], [3]), ([1], [])])
def test_invalid_native_length_metadata_fails_closed(ends, history):
    with pytest.raises(ValueError):
        token_plan(ends, history)


def test_noncausal_request_cannot_use_candidate():
    metadata = SimpleNamespace(actual_seq_lengths_q=[2], seq_lens_list=[2], causal=False)
    dispatch = make_dispatch(torch, None, None, None, {}, state())
    with pytest.raises(ValueError, match="causal"):
        dispatch(None, None, None, metadata, None)
