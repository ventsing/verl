"""Initial-policy replay preserves full groups and original behavior logprobs."""

import os
import time

import pytest
import torch

from verl.experimental.trajectory_async.experience_cache import load_initial_groups


@pytest.fixture
def cache(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"test weights")
    directory = tmp_path / "cache"
    (directory / "train").mkdir(parents=True)
    for uid, lengths in [("short", [3, 5]), ("long", [12, 2]), ("partial", [7])]:
        for session, length in enumerate(lengths):
            key = f"{uid}_{session}_0"
            torch.save(
                {
                    "key": key,
                    "partition": "train",
                    "source_model_dir": str(model),
                    "tag": {
                        "min_global_steps": 0,
                        "max_global_steps": 0,
                        "response_len": length,
                        "trajectory_request_id": key,
                        "migrations": 0,
                    },
                    "data": {
                        "responses": torch.arange(length).unsqueeze(0),
                        "rollout_log_probs": torch.full((1, length), -0.25),
                    },
                },
                directory / "train" / f"{key}.pt",
            )
    return directory, model


def test_selects_complete_longest_group_without_recomputing_logprobs(cache):
    directory, model = cache
    groups = load_initial_groups(directory, model, rollout_n=2, group_count=1)
    assert [row["key"] for row in groups[0]] == ["long_0_0", "long_1_0"]
    assert torch.equal(groups[0][0]["data"]["rollout_log_probs"], torch.full((1, 12), -0.25))
    assert len(load_initial_groups(directory, model, 2, 2)) == 2
    with pytest.raises(ValueError, match="2 complete groups"):
        load_initial_groups(directory, model, 2, 3)


@pytest.mark.parametrize(
    "change,message",
    [
        ("version", "Only initial-policy version 0"),
        ("checkpoint", "different initial checkpoint"),
        ("length", "length mismatch"),
        ("nan", "finite behavior logprobs"),
    ],
)
def test_rejects_incompatible_or_corrupt_cache(cache, change, message):
    directory, model = cache
    path = directory / "train" / "long_0_0.pt"
    row = torch.load(path, weights_only=False)
    if change == "version":
        row["tag"].update(min_global_steps=1, max_global_steps=1)
    elif change == "checkpoint":
        row["source_model_dir"] = str(model / "other")
    elif change == "length":
        row["data"]["rollout_log_probs"] = torch.ones(1, 1)
    else:
        row["data"]["rollout_log_probs"][0, 0] = float("nan")
    torch.save(row, path)
    with pytest.raises(ValueError, match=message):
        load_initial_groups(directory, model, 2, 1)


def test_rejects_checkpoint_modified_since_experience_capture(cache):
    directory, model = cache
    timestamp = time.time_ns() + 10**9
    os.utime(model / "config.json", ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match="Checkpoint files changed"):
        load_initial_groups(directory, model, 2, 1)
