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

from copy import deepcopy

import pytest

from verl.experimental.trajectory_async.staleness_metrics import (
    CompletionRateTracker,
    compute_behavior_policy_metrics,
    compute_replica_policy_metrics,
)

P = "trajectory/train/behavior_policy/"


def test_age_bounds_percentiles_length_weighting_and_padding_without_mutation():
    tags = [{"min_global_steps": 40, "max_global_steps": 48, "response_len": 10},
            {"global_steps": 50, "response_len": 90},
            {"global_steps": 0, "response_len": 1000, "is_padding": True}]
    original = deepcopy(tags)
    result = compute_behavior_policy_metrics(tags, actor_version=50)
    assert result[P + "sample_count"] == 2
    assert result[P + "actor_version"] == 50
    assert result[P + "lag_mean"] == 5 and result[P + "lag_max"] == 10
    assert result[P + "lag_p50"] == 5 and result[P + "lag_p95"] == pytest.approx(9.5)
    assert result[P + "freshest_lag_mean"] == 1 and result[P + "freshest_lag_max"] == 2
    assert result[P + "version_span_mean"] == 5 and result[P + "version_span_max"] == 9
    assert result[P + "multi_version_fraction"] == 0.5
    assert result[P + "response_length_weighted_oldest_lag_mean"] == 1
    assert result[P + "response_length_weighted_freshest_lag_mean"] == 0.2
    assert result[P + "reported_response_tokens"] == 100 and result[P + "lag_ge_10_ratio"] == 0.5
    assert tags == original


def test_resume51_uses_pre_update_actor50_and_fresh_samples_have_zero_age():
    result = compute_behavior_policy_metrics([{"min_global_steps": 50, "max_global_steps": 50}], 51 - 1)
    assert result[P + "lag_mean"] == result[P + "freshest_lag_mean"] == 0
    assert result[P + "version_span_mean"] == 1


def test_unbounded_old_samples_are_measured_without_capping_or_filtering():
    result = compute_behavior_policy_metrics([{"global_steps": 0}], 300)
    assert result[P + "sample_count"] == 1 and result[P + "lag_max"] == 300
    assert result[P + "lag_ge_20_ratio"] == 1


def test_future_or_reversed_versions_are_visible_instead_of_silently_clipped():
    result = compute_behavior_policy_metrics([{"min_global_steps": 52, "max_global_steps": 53}], 50)
    assert result[P + "lag_mean"] == -2 and result[P + "freshest_lag_mean"] == -3
    assert result[P + "future_behavior_count"] == 1
    bad = compute_behavior_policy_metrics([{"min_global_steps": 50, "max_global_steps": 48}], 50)
    assert bad[P + "invalid_version_range_count"] == 1


def test_empty_or_only_padding_has_no_nan_metrics():
    assert compute_behavior_policy_metrics([], 50) == {}
    assert compute_behavior_policy_metrics([{"is_padding": True}], 50) == {}


def test_zero_lengths_do_not_fabricate_token_weighted_age():
    result = compute_behavior_policy_metrics([{"global_steps": 40}], 50)
    assert result[P + "lag_mean"] == 10 and result[P + "reported_response_tokens"] == 0
    assert P + "response_length_weighted_oldest_lag_mean" not in result


def test_replica_ages_are_relative_to_published_version_and_unloaded_are_separate():
    def row(version, active=0, healthy=True, pulling=False):
        return dict(version=version, active=active, reserved=0, healthy=healthy, pulling=pulling, closed=False)

    status = dict(latest=51, replicas=[row(50, 3, pulling=True), row(48, 1), row(-1, healthy=False)])
    original = deepcopy(status)
    result = compute_replica_policy_metrics(status)
    assert result["laminar/serving_policy/lag_mean"] == 2
    assert result["laminar/serving_policy/lag_max"] == 3
    assert result["laminar/replicas/unloaded_count"] == 1
    assert result["laminar/replicas/healthy_count"] == 2
    assert result["laminar/replicas/active_requests"] == 4
    assert result["laminar/replicas/pulling_count"] == 1
    assert status == original


def test_completion_rate_uses_real_interval_and_does_not_invent_first_rate():
    tracker = CompletionRateTracker()
    first = tracker.update(100, 10)
    assert first["laminar/throughput/interval_available"] == 0
    assert "laminar/throughput/completed_requests_per_s" not in first
    second = tracker.update(106, 12)
    assert second["laminar/throughput/interval_s"] == 2
    assert second["laminar/throughput/completed_requests_per_s"] == 3
    assert tracker.update(108, 16)["laminar/throughput/completed_requests_per_s"] == 0.5


def test_counter_reset_and_zero_elapsed_do_not_report_negative_or_infinite_rate():
    tracker = CompletionRateTracker()
    tracker.update(100, 10)
    reset = tracker.update(5, 12)
    assert reset["laminar/throughput/counter_reset"] == 1
    assert "laminar/throughput/completed_requests_per_s" not in reset
    same_time = tracker.update(7, 12)
    assert same_time["laminar/throughput/interval_available"] == 0
    assert tracker.update(9, 14)["laminar/throughput/completed_requests_per_s"] == 1
