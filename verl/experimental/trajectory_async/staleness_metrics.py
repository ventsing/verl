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

"""Read-only version-age and service metrics for trajectory-level async PPO.

Trajectory ages use the actor version before the update consuming the batch.
The newest/oldest behavior versions bound a trajectory's policy age. Their
inclusive span is a version range, not a count of observed distinct versions.
No samples are filtered, dropped, or changed by these diagnostics.
"""

import numpy as np


def _distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {}
    result = {f"{prefix}_mean": float(values.mean()), f"{prefix}_min": float(values.min()),
              f"{prefix}_max": float(values.max()), f"{prefix}_std": float(values.std())}
    for percentile in (50, 90, 95, 99):
        result[f"{prefix}_p{percentile}"] = float(np.percentile(values, percentile))
    return result


def compute_behavior_policy_metrics(tags, actor_version):
    rows = [tag for tag in tags if not tag.get("is_padding", False)]
    if not rows:
        return {}
    oldest = np.array([tag["min_global_steps"] if tag.get("min_global_steps") is not None
                       else tag["global_steps"] for tag in rows], dtype=np.int64)
    newest = np.array([tag["max_global_steps"] if tag.get("max_global_steps") is not None
                       else old for tag, old in zip(rows, oldest, strict=True)], dtype=np.int64)
    oldest_lag = int(actor_version) - oldest
    freshest_lag = int(actor_version) - newest
    span = newest - oldest + 1
    prefix = "trajectory/train/behavior_policy"
    result = {f"{prefix}/sample_count": len(rows), f"{prefix}/actor_version": int(actor_version),
              f"{prefix}/oldest_version_min": int(oldest.min()), f"{prefix}/newest_version_max": int(newest.max()),
              f"{prefix}/multi_version_fraction": float(np.mean(span > 1)),
              f"{prefix}/future_behavior_count": int(np.count_nonzero(newest > actor_version)),
              f"{prefix}/invalid_version_range_count": int(np.count_nonzero(newest < oldest))}
    result.update(_distribution(oldest_lag, prefix + "/lag"))
    result.update(_distribution(freshest_lag, prefix + "/freshest_lag"))
    result.update(_distribution(span, prefix + "/version_span"))
    for threshold in (1, 5, 10, 20):
        result[f"{prefix}/lag_ge_{threshold}_ratio"] = float(np.mean(oldest_lag >= threshold))
    lengths = np.array([tag.get("response_len", 0) for tag in rows], dtype=float)
    result[f"{prefix}/invalid_response_length_count"] = int(np.count_nonzero(lengths < 0))
    weights = np.maximum(lengths, 0)
    result[f"{prefix}/reported_response_tokens"] = int(weights.sum())
    if weights.sum() > 0:
        result[f"{prefix}/response_length_weighted_oldest_lag_mean"] = float(np.average(oldest_lag, weights=weights))
        result[f"{prefix}/response_length_weighted_freshest_lag_mean"] = float(
            np.average(freshest_lag, weights=weights))
    return result


def compute_replica_policy_metrics(status):
    replicas = status.get("replicas", [])
    loaded = [row for row in replicas if row["version"] >= 0]
    prefix = "laminar/replicas"
    result = {f"{prefix}/count": len(replicas), f"{prefix}/unloaded_count": len(replicas) - len(loaded),
              f"{prefix}/healthy_count": sum(bool(row["healthy"]) for row in replicas),
              f"{prefix}/pulling_count": sum(bool(row["pulling"]) for row in replicas),
              f"{prefix}/closed_count": sum(bool(row["closed"]) for row in replicas),
              f"{prefix}/active_requests": sum(row["active"] for row in replicas),
              f"{prefix}/reserved_requests": sum(row["reserved"] for row in replicas),
              "laminar/serving_policy/published_version": int(status["latest"])}
    lags = [status["latest"] - row["version"] for row in loaded]
    result.update(_distribution(lags, "laminar/serving_policy/lag"))
    return result


class CompletionRateTracker:
    """Controller completions/second, including validation and probe requests.

An interval spans two consecutive post-update observations. Counter resets and
nonpositive intervals are reported without fabricating a negative throughput.
"""

    def __init__(self):
        self.previous = None

    def update(self, completed, now):
        prefix = "laminar/throughput"
        result = {f"{prefix}/completed_requests_total": int(completed), f"{prefix}/interval_available": 0}
        if self.previous is not None:
            previous_count, previous_time = self.previous
            elapsed, delta = float(now - previous_time), int(completed - previous_count)
            result[f"{prefix}/counter_reset"] = int(delta < 0)
            if elapsed > 0 and delta >= 0:
                result.update({f"{prefix}/interval_available": 1, f"{prefix}/interval_s": elapsed,
                               f"{prefix}/interval_completed_requests": delta,
                               f"{prefix}/completed_requests_per_s": delta / elapsed})
        self.previous = (int(completed), float(now))
        return result
