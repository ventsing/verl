# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Copy mathematical RL data and apply one audited training-reference repair."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROBLEM_SHA = "00b730999b8fefb8cf9f80976596f86e7779d04f41f4ad5bb91b7e73c6223dfc"
REVISION = "equidistant-line-both-roots-v1"


def correct_reference(row):
    corrected = deepcopy(row)
    info = corrected.get("extra_info") or {}
    reward = corrected.get("reward_model") or {}
    changed = info.get("problem_sha256") == PROBLEM_SHA and reward.get("ground_truth") == "-1"
    if changed:
        reward["ground_truth"] = r"\{-2,-1\}"
        info.update(original_ground_truth="-1", reference_revision=REVISION)
        corrected.update(reward_model=reward, extra_info=info)
    return corrected, changed


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Use a separate output directory to preserve the original references")
    tables = {split: pq.read_table(source / f"{split}.parquet") for split in ("train", "test")}
    hashes = [
        {(row.get("extra_info") or {}).get("problem_sha256") for row in table.to_pylist()} for table in tables.values()
    ]
    if (hashes[0] & hashes[1]) - {None}:
        raise ValueError("Train/test problem overlap")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "reference_revision": REVISION, "splits": {}}
    for split, table in tables.items():
        changed = 0
        if split == "train":
            rows = []
            for row in table.to_pylist():
                corrected, applied = correct_reference(row)
                rows.append(corrected)
                changed += applied
            table = pa.Table.from_pylist(rows)
        destination = output / f"{split}.parquet"
        temporary = destination.with_suffix(".parquet.partial")
        if split == "test":
            temporary.write_bytes((source / f"{split}.parquet").read_bytes())
        else:
            pq.write_table(table, temporary)
        temporary.replace(destination)
        manifest["splits"][split] = {
            "rows": len(table),
            "corrected_references": changed,
            "source_sha256": hashlib.sha256((source / f"{split}.parquet").read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
