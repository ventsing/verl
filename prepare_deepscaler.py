# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Prepare reproducible, problem-disjoint DeepScaleR splits for verl."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
SOURCE_SHA256 = "4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4"
SOURCE_URL = (
    f"https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/{SOURCE_REVISION}/deepscaler.json"
)


def problem_sha(problem):
    return hashlib.sha256(problem.strip().encode()).hexdigest()


def prepare(data_dir, validation_size=512, seed=42):
    data_dir = Path(data_dir)
    source = data_dir / "raw/deepscaler.json"
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected DeepScaleR source SHA256: {digest}")
    examples = json.loads(raw)
    by_problem = defaultdict(list)
    empty_rows = []
    for index, example in enumerate(examples):
        if not example["problem"].strip() or not str(example["answer"]).strip():
            empty_rows.append(index)
            continue
        by_problem[problem_sha(example["problem"])].append((index, example))
    conflicts = [key for key, rows in by_problem.items() if len({str(row["answer"]).strip() for _, row in rows}) > 1]
    conflicting_rows = sum(len(by_problem.pop(key)) for key in conflicts)
    if not 0 < validation_size < len(by_problem):
        raise ValueError("validation_size must leave nonempty train and test splits")
    # Hash ranking is independent of source order and Python RNG versions.
    ranked = sorted(by_problem, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    validation = set(ranked[:validation_size])
    records = {"train": [], "test": []}
    for digest, duplicates in by_problem.items():
        split = "test" if digest in validation else "train"
        # Keep one original reference per exact problem, with provenance. The
        # raw download remains available for auditing conflicting duplicates.
        index, example = duplicates[0]
        records[split].append(
            {
                "data_source": "lighteval/MATH",
                "prompt": [
                    {
                        "role": "user",
                        "content": example["problem"].strip()
                        + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "problem_sha256": digest,
                    "problem": example["problem"],
                    "source_revision": SOURCE_REVISION,
                    "duplicate_count": len(duplicates),
                },
            }
        )
    rank = {key: index for index, key in enumerate(ranked)}
    records["test"].sort(key=lambda row: rank[row["extra_info"]["problem_sha256"]])
    manifest = {
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(examples),
        "unique_problems": len(by_problem),
        "empty_source_indices": empty_rows,
        "conflicting_problem_hashes": conflicts,
        "conflicting_rows_excluded": conflicting_rows,
        "duplicates_removed": len(examples) - len(empty_rows) - conflicting_rows - len(by_problem),
        "split_method": "sha256(seed:problem_sha256), first validation_size held out",
        "seed": seed,
        "historical_split_reproduced": False,
        "splits": {},
    }
    for split, rows in records.items():
        destination = data_dir / f"{split}.parquet"
        temporary = destination.with_suffix(".parquet.partial")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(destination)
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.validation_size, args.seed), indent=2))
