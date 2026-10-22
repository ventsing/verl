"""Strict version-zero experience replay for initial-checkpoint diagnostics."""

from collections import defaultdict
from pathlib import Path

import torch


def load_initial_groups(directory, model_dir, rollout_n, group_count):
    """Select complete, longest-first groups from an unchanged initial model.

    This deliberately supports only fresh-policy memory/correctness checks,
    not recovery after actor updates. Original behavior logprobs are retained.
    """
    model_dir = Path(model_dir).resolve()
    checkpoint_files = [
        model_dir / name
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    ] + list(model_dir.glob("*.safetensors"))
    latest_checkpoint_change = max(path.stat().st_mtime_ns for path in checkpoint_files)
    groups = defaultdict(dict)
    for path in sorted((Path(directory) / "train").glob("*.pt")):
        row = torch.load(path, map_location="cpu", weights_only=False)
        if row.get("partition") != "train" or not row.get("source_model_dir"):
            raise ValueError(f"Cache row is not a model-bound training experience: {path}")
        if Path(row["source_model_dir"]).resolve() != model_dir:
            raise ValueError(f"Cached experience uses a different initial checkpoint: {path}")
        if latest_checkpoint_change > path.stat().st_mtime_ns:
            raise ValueError(f"Checkpoint files changed after the cached experience: {path}")
        tag, data = row["tag"], row["data"]
        if tag.get("min_global_steps") != 0 or tag.get("max_global_steps") != 0:
            raise ValueError(f"Only initial-policy version 0 may be replayed: {path}")
        uid, session, index = row["key"].rsplit("_", 2)
        session = int(session)
        if index != "0" or not 0 <= session < rollout_n or session in groups[uid]:
            raise ValueError(f"Invalid or duplicate cached session: {path}")
        response, logprob = data["responses"][0], data["rollout_log_probs"][0]
        if response.numel() != tag["response_len"] or logprob.numel() != response.numel():
            raise ValueError(f"Cached response/logprob length mismatch: {path}")
        if not torch.isfinite(logprob).all() or not tag.get("trajectory_request_id"):
            raise ValueError(f"Cached experience lacks finite behavior logprobs or scheduler lineage: {path}")
        groups[uid][session] = row
    complete = [rows for rows in groups.values() if len(rows) == rollout_n]
    complete.sort(key=lambda rows: (-max(row["tag"]["response_len"] for row in rows.values()), rows[0]["key"]))
    if len(complete) < group_count:
        raise ValueError(f"Cache has {len(complete)} complete groups; {group_count} required")
    return [[rows[index] for index in range(rollout_n)] for rows in complete[:group_count]]
