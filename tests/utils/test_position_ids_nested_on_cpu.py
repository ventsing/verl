# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import io

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize("lengths", [(1118, 1118), (8, 8), (4, 4), (1118, 1009)])
@pytest.mark.parametrize("storage_layout", ["sequence", "coordinates"])
def test_position_ids_survive_normalization_and_minibatch_selection(lengths, storage_layout):
    if storage_layout == "coordinates" and lengths[0] != lengths[1]:
        pytest.skip("coordinate-jagged layout requires equal token dimensions")
    rows = [torch.arange(4 * length).reshape(4, length) + i * 10000 for i, length in enumerate(lengths)]
    if storage_layout == "sequence":
        positions = tu.nested_tensor_from_tensor_list(rows, ragged_idx=2)
    else:
        positions = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(n) for n in lengths], layout=torch.jagged),
            "position_ids": positions,
        },
        batch_size=[2],
    )
    tu.maybe_fix_3d_position_ids(data)
    assert data["position_ids"]._ragged_idx == 2
    assert data["position_ids"].offsets().diff().tolist() == list(lengths)
    selected = tu.index_select_tensor_dict(data, [1, 0, 1])
    for actual, index in zip(selected["position_ids"].unbind(), [1, 0, 1], strict=True):
        torch.testing.assert_close(actual, rows[index])
    # Repeated normalization must be harmless during nested minibatch iteration.
    tu.maybe_fix_3d_position_ids(selected)
    for actual, index in zip(selected["position_ids"].unbind(), [1, 0, 1], strict=True):
        torch.testing.assert_close(actual, rows[index])


def test_constructor_preserves_jagged_axis_through_serialization():
    rows = [torch.arange(4 * n).reshape(4, n) for n in [7, 11]]
    positions = tu.nested_tensor_from_tensor_list(rows, ragged_idx=2)
    buffer = io.BytesIO()
    torch.save(positions, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert restored._ragged_idx == 2
    for actual, expected in zip(restored.unbind(), rows, strict=True):
        torch.testing.assert_close(actual, expected)


def test_normalizes_position_ids_with_reset_ragged_metadata():
    rows = [torch.arange(4 * n).reshape(4, n) for n in [7, 11]]
    positions = tu.nested_tensor_from_tensor_list(rows, ragged_idx=2)
    positions._ragged_idx = 1  # Reproduce affected TensorDict pickle/consolidate paths.
    data = TensorDict({"position_ids": positions}, batch_size=[2])
    tu.maybe_fix_3d_position_ids(data)
    for actual, expected in zip(data["position_ids"].unbind(), rows, strict=True):
        torch.testing.assert_close(actual, expected)
