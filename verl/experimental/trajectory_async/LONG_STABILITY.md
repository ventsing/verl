# 16 NPU trajectory asynchronous stability run

Run ID: `qwen35_9b_laminar_longstable_20261020_to_20261021_noon`.
Scheduled stop: **2026-10-21 12:00 Asia/Shanghai**, equivalent to
**2026-10-21 04:00 UTC**. This records the run configuration; completion and
stability conclusions must come from the runtime reports after the run ends.

- Qwen3.5-9B, DeepScaleR full training split, thinking enabled, prompt 2048 and
  response 24576 tokens; GRPO, eight prompt groups per update, four responses per group.
- Eight FSDP2 training NPUs and eight inference NPUs, four TP2 replicas.
- `trainer_mode=laminar`: per-trajectory ingestion, complete-group training,
  independent replica weight pulls, exact behavior-policy rollout log probabilities,
  and same-version trajectory migration.
- Native Mooncake CPU relay over TCP on this host; memory-only partial trajectory
  and relay recovery. Checkpoint saving remains disabled.
- Online Roofline and KV/SSM capacity calibration enabled. Initial admission eight
  per replica, engine ceiling 64, actual capacity bounded by engine pages and
  observed calibration. No unobserved hardware knee is assumed.
- `total_training_steps=100000`, `total_epochs=1000` avoid a short smoke-test end.
  The deadline supervisor stops only this named Ray job. Evaluation is disabled
  during the stability run to observe continuous asynchronous training.
- Weight retention remains four versions / 96 GiB per relay. Capacity waiting is
  increased from 600 to 3600 seconds for 24K tails; this does not increase the
  memory budget. Failures are recorded without automatically restarting from
  new weights or discarding evidence.

Runtime directory:
`/tmp/verl-laminar-runs/qwen35_9b_laminar_longstable_20261020_to_20261021_noon`.

Submission is `/tmp/submit_laminar_overnight.py`. The detached supervisor
`/tmp/watch_laminar_overnight.py` polls every 30 seconds, records completed
updates, gradient/reward/length/timing metrics, scheduler and migration events,
capacity/roofline snapshots, errors, disk space, and Ray job status. At the
deadline it calls the named job's stop endpoint; it never stops the shared Ray
cluster or other jobs. `/tmp/start_watch_laminar_overnight.py` starts the watcher;
do not start duplicate watchers or submit the same run again.

Inspect with:

```bash
python3 /tmp/status_laminar_overnight.py
tail -f /tmp/verl-laminar-runs/qwen35_9b_laminar_longstable_20261020_to_20261021_noon/train.log
```

`long_stability_status.json` is the current summary and
`long_stability_history.jsonl` contains the time series. `train.log` and
`trajectories/scheduler.jsonl` retain the underlying evidence. Watcher PID and
errors are in `long_stability_watch.pid` and `long_stability_watch.log`.
A `planned_deadline_stop` result denotes the scheduled end, not proof of a
passed stability test. Check completed updates, nonzero gradient statistics,
policy-version consistency, real training migrations, memory trends, and errors
before reporting acceptance. These files are local `/tmp` runtime artifacts,
not durable model checkpoints or recovery backups.
