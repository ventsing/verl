# Strict synchronous control for the 32K reasoning run

The asynchronous run was stopped at the user's request. Ray confirms STOPPED
for `qwen35_9b_rewardfix_32k_longstable_20261021_to_20261022_1400`; the stop reason
is saved in that run's `user_stop.json`. Its existing metrics/curves are retained.

The replacement submission is **`qwen35_9b_sync_32k_control_20261022`**, Ray job
`54000000`. Its runtime directory is
`/tmp/verl-laminar-runs/qwen35_9b_sync_32k_control_20261022/`.
It starts independently from the original Qwen3.5-9B weights, not the stopped
async run's in-memory policy. It retains the original deadline of
**2026-10-22 14:00 Beijing / 06:00 UTC**.

## Comparable configuration

Both actual Hydra configurations were resolved and compared. The differences
are the synchronous flag and experiment/output paths only. The full leaf diff
is saved as `config_comparison.json` in the replacement runtime directory.
In the older configuration the flag was absent; its effective default is false.

Both runs use the same audited DeepScaleR data copy, full train split and
fixed 16-question greedy validation set, corrected mathematical reward,
32,768 response + 2,048 prompt token limits, thinking mode, GRPO, 16 prompt
groups × 4 samples, learning rate 1e-6, KL coefficient 0.001 and seeds.
Both have eight FSDP2 training NPUs and four TP2 rollout replicas on eight
other NPUs. Actor microbatch, optimizer, model kernels, Mooncake TCP relay,
online KV/Roofline calibration and inference settings are unchanged.
Validation runs before training and every five updates. Checkpoints and
recovery backups remain memory-only / disabled for disk persistence.

## Strict batch and weight barriers

`trainer.v1.laminar.synchronous=True` selects a synchronous control on the
same physical placement and relay implementation:

1. No warmup batches or background prompt feeder are started.
2. Submit exactly one batch of 16 groups. Wait until all 64 scored trajectories
   and their producer terminal markers are available before training.
3. Require every trajectory's behavior-policy version to equal `global_step - 1`.
   Copy old log probabilities from the actual vLLM rollout as before. Fail on
   stale or mixed-version experience; never silently train it.
4. Require no pending prompts or active/reserved requests before publication.
5. Publish the new actor weights, then wait until **every** rollout replica
   finishes loading that exact version. Only then may the next batch begin.

Same-version scheduling/migration within a rollout batch remains available,
matching the asynchronous run's inference settings. There is no generation
overlap with optimizer updates, no prefetch into the next round and no use of
older-policy trajectories. Failures cannot trigger hidden extra-group refill.
This differs from selecting the built-in `trainer_mode=sync`, which colocates
training and inference and would change the hardware layout of this comparison.
The internal trainer remains `laminar`; `laminar/synchronous=1` and zero policy
staleness identify the strict synchronous execution. Weight barrier time is
reported separately as `timing_s/sync_rollout_barrier` and included in step time.

## Run and inspect

```bash
# Use the host's existing Mooncake Python/library environment, with TCP.
MOONCAKE_RELAY_PROTOCOL=tcp \
DATA_DIR=/tmp/verl-reward-fix-20261021/data \
python3 submit_qwen35_laminar.py --reasoning --synchronous \
  --relay-backend mooncake --output-root /tmp/verl-laminar-runs

tail -f /tmp/verl-laminar-runs/qwen35_9b_sync_32k_control_20261022/train.log
```

The detached supervisor PID/status are in `long_stability_watch.pid` and
`long_stability_status.json`. It collects raw numeric metrics to `metrics.json`
and `metrics.csv`, and refreshes `curves.png` / `curves.svg` every five minutes.
`comparison.png` / `comparison.svg` overlay the stopped asynchronous run and
this control by completed update: raw reward, fixed validation accuracy,
actual length-budget hits and mean policy staleness. The renderer is a bounded
child process; only this submission is stopped at the recorded deadline.
Source hashes/archive and a copy of the supervisor are retained in the runtime
directory. No automatic restart from the original model is performed on error.

Training batches differ because async consumption follows completion order
and sync consumption waits for the entire submitted batch. The fixed held-out
validation is the primary comparable learning signal. Validation contains only
16 questions; the current synchronous run also has a shorter wall-clock window
because the user stopped the asynchronous run before starting this control.
Do not claim superiority from different numbers of updates or raw training
reward alone. First-update success and zero staleness must be confirmed in
the actual run before claiming end-to-end synchronous acceptance.

## Verification

The relevant Laminar, strict synchronous, sampler and reward-diagnostic CPU
regressions passed **43 tests**. New tests verify no background submission,
exactly one batch per prepare, refusal of refill, rejection of busy/old/unhealthy
replicas, waiting for the slowest replica, timeout and stale-logprob rejection.
Syntax/Ruff passed for changed Python sources; the launcher config check passed.
Ray confirmed the new real 16-NPU job RUNNING during model/server initialization.
