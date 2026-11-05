"""Data preparation must preserve provenance and keep repeated questions disjoint."""

import hashlib
import json

import pyarrow.parquet as pq
import pytest

import prepare_deepscaler as data
import prepare_reasoning_math_data as reasoning


def test_preparation_excludes_empty_conflicting_and_duplicate_questions(monkeypatch, tmp_path):
    rows = [
        {"problem": "A", "answer": "1"},
        {"problem": "A", "answer": "1"},
        {"problem": "B", "answer": "2"},
        {"problem": "C", "answer": "3"},
        {"problem": "C", "answer": "4"},
        {"problem": "empty", "answer": ""},
        {"problem": "D", "answer": "5"},
    ]
    raw = json.dumps(rows).encode()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/deepscaler.json").write_bytes(raw)
    monkeypatch.setattr(data, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    manifest = data.prepare(tmp_path, validation_size=1)
    assert manifest["duplicates_removed"] == 1
    assert manifest["empty_source_indices"] == [5]
    assert manifest["conflicting_rows_excluded"] == 2
    assert manifest["conflicting_problem_hashes"] == [data.problem_sha("C")]
    train = pq.read_table(tmp_path / "train.parquet").to_pylist()
    test = pq.read_table(tmp_path / "test.parquet").to_pylist()
    train_hashes = {r["extra_info"]["problem_sha256"] for r in train}
    test_hashes = {r["extra_info"]["problem_sha256"] for r in test}
    assert len(train) == 2 and len(test) == 1
    assert train_hashes.isdisjoint(test_hashes)
    assert data.prepare(tmp_path, validation_size=1)["splits"] == manifest["splits"]
    with pytest.raises(ValueError, match="separate output"):
        reasoning.prepare(tmp_path, tmp_path)
    copied = reasoning.prepare(tmp_path, tmp_path / "reasoning")
    assert copied["splits"]["test"]["source_sha256"] == copied["splits"]["test"]["output_sha256"]
    # The copy operation must refuse leaking a training problem into validation.
    (tmp_path / "test.parquet").write_bytes((tmp_path / "train.parquet").read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        reasoning.prepare(tmp_path, tmp_path / "other")
