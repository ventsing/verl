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
"""Monitor training policy age and join delayed scores to their evaluated version."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

MEAN = "training/off_policy/trajectory_staleness/mean"
WORST = "training/off_policy/trajectory_staleness_worst/mean"
MAXIMUM = "training/off_policy/trajectory_staleness/max"


def summarize(records, window=5):
    """Use pre-update behavior age, without assigning delayed scores to newer weights."""
    if window < 1:
        raise ValueError("window must be positive")
    training = {}
    evaluations = {}
    for record in records:
        data = record.get("data", {})
        if "training/global_step" in data and MEAN in data:
            training[int(data["training/global_step"])] = data
        if "evaluation/policy_version" in data:
            evaluations[int(data["evaluation/policy_version"])] = data

    def context(version):
        point = training.get(version)
        rows = [training[step] for step in sorted(training) if version - window < step <= version]
        return {
            "training_step": version,
            "mean_staleness": point[MEAN] if point else None,
            "mean_worst_staleness": point.get(WORST) if point else None,
            "maximum_staleness": point.get(MAXIMUM) if point else None,
            "window_start_step": max(min(version, 1), version - window + 1),
            "window_end_step": version,
            "window_observed_steps": len(rows),
            "window_step_mean_staleness": sum(row[MEAN] for row in rows) / len(rows) if rows else None,
        }

    scores = []
    for version, data in sorted(evaluations.items()):
        scores.append(
            {
                "policy_version": version,
                **context(version),
                "samples": data.get("evaluation/samples"),
                "evaluation_seconds": data.get("evaluation/seconds"),
                "scores": {key: value for key, value in data.items() if key.startswith("val-core/")},
            }
        )
    return {
        "age_units": "weight versions behind the actor before the consuming update",
        "window_steps": window,
        "latest_training": context(max(training)) if training else None,
        "evaluations": scores,
    }


def update(output, window=5):
    """Write JSON and CSV reports from completed metric records only."""
    records = []
    metrics = output / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A writer may still be completing the final line.
                continue
    report = summarize(records, window)
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = output / "staleness_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output / "staleness_report.json")
    temporary = output / "evaluation_staleness.csv.tmp"
    with temporary.open("w", newline="") as stream:
        fields = [
            "policy_version",
            "mean_staleness",
            "mean_worst_staleness",
            "maximum_staleness",
            "window_step_mean_staleness",
            "window_observed_steps",
            "samples",
            "evaluation_seconds",
            "score_name",
            "score",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for evaluation in report["evaluations"]:
            row = {key: evaluation.get(key) for key in fields[:-2]}
            for name, score in evaluation["scores"].items():
                writer.writerow({**row, "score_name": name, "score": score})
    temporary.replace(output / "evaluation_staleness.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        report = update(args.output, args.window)
        print(
            json.dumps({"updated_at": report["updated_at"], "latest_training": report["latest_training"]}), flush=True
        )
        if args.once:
            return
        status = args.output / "status.json"
        if status.exists():
            try:
                if json.loads(status.read_text()).get("ray_status") in {"FAILED", "STOPPED", "SUCCEEDED"}:
                    return
            except json.JSONDecodeError:
                pass
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
