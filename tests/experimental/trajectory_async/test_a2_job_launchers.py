# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Job entrypoint smoke tests use fake Ray and cannot touch the real cluster."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def job(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    trace = tmp_path / "calls.jsonl"
    wrapper = f"""#!{sys.executable}
import json, os, pathlib, subprocess, sys
kind, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
source = sys.stdin.read() if kind == 'python-job' and args and args[0] == '-' else ''
with open(os.environ['FAKE_TRACE'], 'a') as f:
 f.write(json.dumps({{'kind':kind, 'args':args, 'source':source[:80]}}) + '\\n')
if kind == 'ray': raise SystemExit(0)
if kind == 'pgrep': raise SystemExit(int(os.environ.get('FAKE_PGREP_EXIT', '1')))
if source and ('import ray' in source or 'import torch' in source): raise SystemExit(0)
if source: raise SystemExit(subprocess.run([sys.executable, *args], input=source, text=True).returncode)
if args[:2] == ['-m', 'verl.trainer.main_ppo']: raise SystemExit(0)
raise SystemExit('unexpected fake Python invocation')
"""
    for name in ("python-job", "ray", "pgrep"):
        path = binaries / name
        path.write_text(wrapper)
        path.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.parquet", "test.parquet"):
        (data / name).write_text("existing test data")
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "PYTHON_BIN": str(binaries / "python-job"),
        "FAKE_TRACE": str(trace),
        "DATA_DIR": str(data),
        "RUN_NAME": "test-job",
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "A2_JOB_LOG_DIR": str(tmp_path / "cluster"),
        "VC_WORKER_HOSTS": ",".join(["127.0.0.1"] * 8),
        "VC_WORKER_NUM": "8",
        "VC_TASK_INDEX": "0",
    }
    for key in (
        "A2_ENV_FILE",
        "OUTPUT_DIR",
        "TRAINER_NNODES",
        "ROLLOUT_NNODES",
        "EVAL_NNODES",
        "NPUS_PER_NODE",
        "RESPONSE_HISTORY_PATH",
        "RAY_NODE_IP",
        "BUFFER_MIN_BATCHES",
        "BUFFER_MAX_BATCHES",
        "BUFFER_INITIAL_BATCHES",
    ):
        env.pop(key, None)

    def run(mode="trajectory_async", args=(), **updates):
        result = subprocess.run(
            ["bash", str(ROOT / f"run_a2_job_{mode}.sh"), *args],
            env={**env, **updates},
            text=True,
            capture_output=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("mode", ["trajectory_async", "sync"])
def test_head_start_and_quoted_scheduler_overrides(job, mode):
    result, calls = job(mode, args=("--dataset_mount=/not-hydra", "trainer.project_name='two words'"))
    assert result.returncode == 0, result.stderr + result.stdout
    ray = [c["args"] for c in calls if c["kind"] == "ray"]
    assert len(ray) == 1 and ray[0][:2] == ["start", "--head"]
    train = [c["args"] for c in calls if c["args"][:2] == ["-m", "verl.trainer.main_ppo"]]
    assert len(train) == 1
    args = train[0]
    assert "trainer.project_name='two words'" in args
    assert not any(arg.startswith("--dataset_mount") for arg in args)
    assert f"trainer.v1.laminar.synchronous={mode == 'sync'}" in args
    assert f"trainer.v1.laminar.adaptive_buffer.enabled={mode != 'sync'}" in args
    assert "trainer.v1.laminar.prompt_buffer_groups=" + ("64" if mode == "sync" else "320") in args
    assert "trainer.v1.laminar.online_capacity.rolling_kv_horizon_s=90.0" in args
    assert "trainer.val_before_train=True" in args


def test_head_reuses_ray(job):
    result, calls = job(FAKE_PGREP_EXIT="0")
    assert result.returncode == 0, result.stderr + result.stdout
    ray = [c["args"] for c in calls if c["kind"] == "ray"]
    assert len(ray) == 1 and ray[0][0] == "status"


def test_worker_joins_without_training(job):
    result, calls = job(VC_TASK_INDEX="1")
    assert result.returncode == 0, result.stderr + result.stdout
    ray = [c["args"] for c in calls if c["kind"] == "ray"]
    assert any(a[0] == "start" and "--block" in a and "--head" not in a for a in ray)
    assert not any(c["args"][:2] == ["-m", "verl.trainer.main_ppo"] for c in calls)


def test_config_only_does_not_connect_to_ray(job):
    result, calls = job("sync", args=("--check-config",), VC_WORKER_HOSTS="", VC_WORKER_NUM="")
    assert result.returncode == 0, result.stderr + result.stdout
    assert all(c["kind"] == "python-job" for c in calls)
    assert not any("import ray" in c["source"] for c in calls)
    train = [c for c in calls if c["args"][:2] == ["-m", "verl.trainer.main_ppo"]]
    assert len(train) == 1 and "--cfg" in train[0]["args"] and "--resolve" in train[0]["args"]


def test_invalid_node_count_fails_before_ray(job):
    result, calls = job(VC_WORKER_NUM="7")
    assert result.returncode == 2
    assert "placement requires 8 nodes" in result.stderr
    assert all(c["kind"] == "python-job" for c in calls)
