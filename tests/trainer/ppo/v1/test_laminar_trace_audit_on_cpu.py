"""Independent trace audit catches version drift and corrupted queue lineage."""

import json

import pytest

from audit_laminar_run import TraceAuditError, audit_run


@pytest.fixture
def trace(tmp_path):
    events = [{"event": "published", "version": 0}]
    for slot in (0, 1):
        events.extend({"event": kind, "replica": slot, "version": 0} for kind in ("pull_begin", "pull_end"))
    events.extend(
        [
            {"event": "request_begin", "request": "a", "replica": 0, "version": 0},
            {"event": "request_begin", "request": "b", "replica": 1, "version": 0},
            {"event": "published", "version": 1},
            {"event": "migration", "request": "a", "source": 0, "destination": 1, "version": 0, "prefix_tokens": 20},
            {"event": "pull_begin", "replica": 0, "version": 1},
            {"event": "pull_end", "replica": 0, "version": 1},
            {"event": "request_begin", "request": "c", "replica": 0, "version": 1},
            {"event": "request_end", "request": "c", "version": 1, "tokens": 3, "migrations": 0},
            {"event": "continuation_segment", "request": "a", "version": 0, "tokens": 2},
            {"event": "request_end", "request": "a", "version": 0, "tokens": 6, "migrations": 1},
            {"event": "request_end", "request": "b", "version": 0, "tokens": 7, "migrations": 0},
        ]
    )
    train, val = [], []
    for key, version, tokens, migrations, rows, step in [
        ("a", 0, 6, 1, train, 2),
        ("b", 0, 7, 0, train, 2),
        ("c", 1, 3, 0, val, 1),
    ]:
        rows.extend(
            [
                {
                    "event": "arrived",
                    "key": key,
                    "scheduler_request_id": key,
                    "model_version": version,
                    "response_tokens": tokens,
                    "migrations": migrations,
                },
                {"event": "selected", "key": key, "step": step},
            ]
        )
    return tmp_path, events, train, val


def run_audit(trace):
    root, events, train, val = trace
    directory = root / "trajectories"
    directory.mkdir(exist_ok=True)
    for name, rows in [("scheduler", events), ("train", train), ("val", val)]:
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return audit_run(root, response_budget=8)


def test_migration_and_new_pull_overlap_without_version_drift(trace):
    report = run_audit(trace)
    assert report["consistent"]
    assert report["published_versions"] == [0, 1]
    assert report["migrations"] == {"training": 1}
    assert report["partitions"]["train"] == {"arrived": 2, "selected": 2, "migrated_arrivals": 1}
    assert report["partitions"]["val"]["selected"] == 1
    assert report["requests_without_terminal_record"] == 0


def test_detects_cross_version_migration(trace):
    event = next(e for e in trace[1] if e["event"] == "migration")
    event["version"] = 1
    with pytest.raises(TraceAuditError, match="changed behavior policy"):
        run_audit(trace)


def test_detects_weight_pull_while_request_is_active(trace):
    trace[1].insert(8, {"event": "pull_begin", "replica": 1, "version": 1})
    with pytest.raises(TraceAuditError, match="pulled with active requests"):
        run_audit(trace)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("response_tokens", 5, "lost response tokens"),
        ("model_version", 1, "changed behavior policy"),
        ("migrations", 0, "lost migration history"),
        ("scheduler_request_id", "missing", "no completed scheduler request"),
    ],
)
def test_detects_queue_lineage_corruption(trace, field, value, message):
    trace[2][0][field] = value
    with pytest.raises(TraceAuditError, match=message):
        run_audit(trace)


def test_validation_must_use_requested_update_version(trace):
    trace[3][1]["step"] = 2
    with pytest.raises(TraceAuditError, match="invalid policy version"):
        run_audit(trace)


def test_duplicate_training_selection_is_rejected(trace):
    trace[2].append(dict(trace[2][1]))
    with pytest.raises(TraceAuditError, match="selected twice"):
        run_audit(trace)


def test_imported_experience_is_checked_against_source_run(trace, tmp_path):
    source = tmp_path / "source"
    (source / "trajectories").mkdir(parents=True)
    original = {"event": "request_end", "request": "cached", "version": 0, "tokens": 7, "migrations": 1}
    (source / "trajectories/scheduler.jsonl").write_text(json.dumps(original) + "\n")
    imported = {**original, "event": "request_import", "source_output": str(source)}
    trace[1].insert(5, imported)
    report = run_audit(trace)
    assert report["imported_requests"] == 1
    imported["tokens"] = 6
    with pytest.raises(TraceAuditError, match="differs from its source completion"):
        run_audit(trace)


def test_fault_recovery_keeps_policy_and_allows_old_weight_reload(trace):
    # Replace the first migration with quarantine and a same-version takeover.
    events = trace[1]
    migration = next(i for i, e in enumerate(events) if e["event"] == "migration")
    events[migration : migration + 1] = [
        {"event": "replica_quarantined", "replica": 0, "error": "injected death"},
        {"event": "trajectory_recovered", "request": "a", "source": 0, "destination": 1, "version": 0},
    ]
    # Do not reuse the failed replica in this simplified trace.
    events[:] = [
        e
        for e in events
        if not (
            e.get("request") == "c"
            or e["event"] in ("pull_begin", "pull_end")
            and e["replica"] == 0
            and e["version"] == 1
            or e["event"] == "continuation_segment"
        )
    ]
    next(e for e in events if e["event"] == "request_end" and e["request"] == "a")["migrations"] = 0
    trace[2][0]["migrations"] = 0
    trace[3].clear()
    assert run_audit(trace)["recoveries"] == 1
    next(e for e in events if e["event"] == "trajectory_recovered")["version"] = 1
    with pytest.raises(TraceAuditError, match="changed behavior policy"):
        run_audit(trace)
