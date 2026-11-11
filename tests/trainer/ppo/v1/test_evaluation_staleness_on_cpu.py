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
"""Delayed evaluation scores must use the matching training version's policy age."""

import csv
import json

import pytest

from examples.trajectory_async.monitor_evaluation_staleness import MEAN, summarize, update


def train(step, age):
    return {"step": step, "data": {"training/global_step": step, MEAN: age}}


def test_delayed_score_joins_exact_version_and_window_without_future_data():
    records = [train(step, step / 10) for step in range(1, 11)]
    records.append(
        {"step": 10, "data": {"evaluation/policy_version": 5, "evaluation/samples": 512, "val-core/math/acc": 0.75}}
    )
    result = summarize(records)
    assert result["latest_training"]["mean_staleness"] == 1.0
    score = result["evaluations"][0]
    assert score["policy_version"] == 5 and score["mean_staleness"] == 0.5
    assert score["window_step_mean_staleness"] == pytest.approx(0.3)
    assert score["window_observed_steps"] == 5
    assert score["scores"] == {"val-core/math/acc": 0.75}


def test_missing_matching_training_step_does_not_borrow_latest_age():
    result = summarize([train(10, 2), {"data": {"evaluation/policy_version": 5}}])
    score = result["evaluations"][0]
    assert score["mean_staleness"] is None
    assert score["window_step_mean_staleness"] is None
    assert score["window_observed_steps"] == 0


def test_window_is_five_actual_step_numbers_not_last_five_records():
    result = summarize([train(1, 100), train(8, 2), train(10, 4)])
    assert result["latest_training"]["window_step_mean_staleness"] == 3
    assert result["latest_training"]["window_observed_steps"] == 2


def test_report_tolerates_partial_metrics_and_writes_aligned_csv(tmp_path):
    records = [train(5, 1.25), {"data": {"evaluation/policy_version": 5, "val-core/math/acc": 0.5}}]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records) + '\n{"data":')
    report = update(tmp_path)
    assert json.loads((tmp_path / "staleness_report.json").read_text()) == report
    with (tmp_path / "evaluation_staleness.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["policy_version"] == "5" and row["mean_staleness"] == "1.25"
    assert row["score"] == "0.5"
