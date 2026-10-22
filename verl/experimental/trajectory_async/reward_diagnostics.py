# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Keep raw training reward separate from group windows and length censoring."""

from collections import defaultdict, deque
from statistics import mean, pstdev


class RewardDiagnostics:
    def __init__(self, window_groups=128):
        if window_groups < 1:
            raise ValueError("window_groups must be positive")
        self.window = deque(maxlen=window_groups)

    def update(self, records, response_budget):
        """records are (trajectory key, raw scalar score, response token count)."""
        if response_budget <= 0:
            raise ValueError("response_budget must be positive")
        groups = defaultdict(list)
        for key, score, length in records:
            groups[key.rsplit("_", 2)[0]].append((float(score), int(length)))
        if not groups:
            return {}
        for rows in groups.values():
            self.window.append(mean(score for score, _ in rows))
        scores = [score for rows in groups.values() for score, _ in rows]
        short = [score for rows in groups.values() for score, length in rows if length < response_budget]
        clipped = [score for rows in groups.values() for score, length in rows if length >= response_budget]
        result = {
            "raw_batch_mean": mean(scores),
            "raw_group_mean": mean(mean(score for score, _ in rows) for rows in groups.values()),
            "rolling_group_mean": mean(self.window),
            "rolling_group_count": len(self.window),
            "groups": len(groups),
            "truncated_fraction": len(clipped) / len(scores),
            "complete_count": len(short),
            "truncated_count": len(clipped),
            "mixed_reward_group_fraction": mean(len({score for score, _ in rows}) > 1 for rows in groups.values()),
            "within_group_std_mean": mean(pstdev(score for score, _ in rows) for rows in groups.values()),
        }
        if short:
            result["not_truncated_reward_mean"] = mean(short)
        if clipped:
            result["truncated_reward_mean"] = mean(clipped)
        return {f"laminar/reward/{key}": value for key, value in result.items()}
