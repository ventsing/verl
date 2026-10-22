# Reward stability for thinking-enabled mathematical RL

The original 16-NPU run stopped as scheduled at 2026-10-21 12:00 Asia/Shanghai,
after 42 updates. Its 1344 saved responses show a correlation of **-0.938**
between per-update raw reward and the true response-budget hit fraction.
This is evidence of a strong association with truncation, not proof that
truncation is the sole cause or that the policy has improved or regressed.

## Implemented corrections

`reasoning_math_reward.py` is an opt-in dataset-specific scorer. DeepScaleR's
MATH aliases require a closed thinking section and grade only the final boxed
answer following it. A boxed guess inside unfinished reasoning receives no
correctness reward. The opening `<think>` can already be part of the prompt.
Math-Verify compares fractions, intervals (including open/closed endpoints),
expressions and matrices. MATH's percentage-point representation and lists of
alternative functions are normalized explicitly. Other datasets retain their
existing scorer. No global MATH scorer or original dataset was overwritten.

Symbolic work runs in two bounded CPU subprocesses per reward actor, with two
in-flight calls, short parser/verifier alarms and an outer cold-start watchdog.
Unparsed answers, unparsed references and timeouts are recorded separately.
`math_verify` and `latex2sympy2_extended` must be installed on every reward node.

`prepare_reasoning_math_data.py` copies the original disjoint train/test splits
and applies audited reference corrections by exact problem SHA256. One known
DeepScaleR item gives points A(1,-2), B(5,6) equidistant from `ax+y+1=0`.
`|a-1|=|5a+7|` gives **both -2 and -1**; the original reference listed only -1.
The copied training row records the previous reference and revision. Test rows
are unchanged. The manifest records original/output hashes and correction counts.
This is one verified correction, not a claim that every dataset label is clean.

`mixed_ready` sampling keeps the oldest complete group and chooses the other
groups reproducibly from the already-ready pool. It never reads rewards,
splits GRPO groups or waits for an unfinished group. Completion-order selection
remains available for controlled comparisons. Mixing has no effect when only
one exact-sized batch is ready; it cannot eliminate all arrival-time bias.

Online admission can now grow toward a previously measured, ready bound after
the sampling window closes. A change of context still resets admission
conservatively, but a ready bound in that exact new stratum permits recovery.
An unmeasured stratum still cannot explore after sampling ends. This fixes
permanent fallback to batch 8 without restarting telemetry or inventing a knee.

## Raw metrics and fixed evaluation

The actor continues to optimize raw correctness. `critic/rewards/mean` is not
replaced by an EMA, length bonus or normalized curve. Additional diagnostics:

- `laminar/reward/raw_batch_mean` and `raw_group_mean`;
- `rolling_group_mean` with its actual `rolling_group_count` (128-group window);
- `not_truncated_reward_mean`, `truncated_reward_mean`, and their counts;
- `truncated_fraction`, `mixed_reward_group_fraction`, `within_group_std_mean`.

Conditional reward on unfinished/finished lengths is diagnostic, not a fixed
evaluation metric. Use `trajectory/train/length/budget_reached_ratio` or
`laminar/reward/truncated_fraction` for actual budget hits: the generic padded
`response_length/clip_ratio` can mark a batch's maximum length even when that
length is below the configured response budget.

The reasoning preset uses 16 prompts x 4 responses = **64 trajectories/update**
(microbatch 1 per NPU), instead of 8 prompts / 32 responses. This changes the
actual gradient batch, not only the displayed metric. It also enables fixed
held-out greedy evaluation before training and every five updates by default.
Compare those same held-out questions across policy versions before claiming
learning gains. A smooth or increasing training curve alone is insufficient.

## 32K and 48K

`examples/trajectory_async/run_qwen35_9b_16npu_laminar_reasoning.sh` defaults to 32768 response tokens plus
2048 prompt tokens. It retains 8 FSDP2 training NPUs and four TP2 rollout
replicas, exact-version behavior logprobs, migration and memory-only recovery.
`MAX_RESPONSE_LENGTH=49152` selects the 48K configuration, but is **not accepted
for training on the current layout** without additional memory work.

Actual Qwen3.5-9B TP2 prefill plus 256 decode tokens passed at total contexts
34816 and 51200. At 34816 the real pool has 2818 blocks and requests reserve
71 pages; at 51200 requests reserve 103 pages. The native inference probe used
engine ceiling 16; production admission must still use its own online profile.

The full-model 8-NPU FSDP2 forward/backward probe passed 34816 tokens on all
ranks with finite loss/gradients, allocated 37.29 GiB and reserved 49.27 GiB.
At 51200 tokens under the actual jemalloc runtime it **OOMed during backward**.
A prior run without the required jemalloc preload reached both lengths but
aborted on heap cleanup; that run is not clean acceptance evidence.
The synthetic loss probe establishes sequence-memory feasibility, not GRPO
reward quality or a complete optimizer/relay integration result.

## Reproduction and evidence

```bash
python3 prepare_reasoning_math_data.py \
  --source /opt/huawei/dataset/ccb_data/code/package_install/datasets_for_train/deepscaler \
  --output /tmp/verl-reasoning-data

# Export the node's existing Mooncake environment/protocol, then submit:
DATA_DIR=/tmp/verl-reasoning-data python3 submit_qwen35_laminar.py \
  --reasoning --relay-backend mooncake --output-root /tmp/verl-laminar-runs
```

Reasoning submissions without explicit `DATA_DIR` prepare the audited copy under
the run directory. Direct shell launches should set `DATA_DIR` to the prepared
copy explicitly. Keep the node's native Mooncake Python/library paths and use
TCP explicitly on the current host, which has no RDMA interface.

CPU regression: `python3 -m pytest -q -x tests/trainer/ppo/v1
tests/models/test_qwen35_decoder_compat_on_cpu.py` (run as one command).

Offline same-output/same-original-reference replay:
`/tmp/verl-reward-fix-20261021/reward_replay.json` and `replayed_rows.jsonl`.
Mean recorded reward 0.39137 -> regraded 0.41964; 44 responses changed 0->1,
six changed 1->0 (five unfinished reasoning, one disagreement with the
incomplete reference discussed above). The raw per-update standard deviation
**did not decrease** (0.2091 -> 0.2177). This is a scorer counterfactual, not a
new-policy learning result or proof that the wave pattern has disappeared.
The replay deliberately retains original references for a controlled comparison.

Native context probes: `/tmp/verl-reward-fix-20261021/long_context_v2/`.
Integration run: `qwen35_9b_rewardfix_32k_20261021_v1`, output under
`/tmp/verl-laminar-runs/`. It uses the corrected dataset, 32K budget and two
updates, with evaluation disabled only for this integration check. Inspect it
with `python3 /tmp/status_reward_fix.py`. Its status and measured metrics must
be checked before claiming end-to-end acceptance. The previous job's weights
were in memory only; this acceptance run starts from the original model.

## Additional verification

Final CPU regression: **413 passed, 2 skipped**. Syntax and Ruff passed for the
14 changed Python sources/tests. A separate frozen-arrival queue replay is in
`/tmp/verl-reward-fix-20261021/ready_group_replay.json`. With fixed outputs and
an assumed 400 seconds of training per eight groups, completion-order batch 8
has raw batch standard deviation 0.21770; mixed-ready batch 8 has 0.22151;
mixed-ready batch 16 has 0.17574. All consume the same responses and preserve
mean reward 0.41964. This supports increasing the actual gradient batch; mixing
alone did not improve this trace. It is not a live training A/B result.

Five live completed experiences from the new run were read through an independent
job-scoped, read-only TQ client. They had closed thinking, final boxed answers,
score/accuracy 1, and zero unparsed-reference/answer/timeout flags. The summary
is `live_reward_samples.json` in the integration run directory. It samples the
shortest available completions and must not be used as an overall accuracy
estimate. No tensor backups or model checkpoints were written by the inspection.

## Scheduled 32K stability run, October 21–22

The two-update integration job `qwen35_9b_rewardfix_32k_20261021_v1` finished
**SUCCEEDED**. Raw rewards were 0.828125 and 0.46875; actual budget-hit fractions
were 0 and 0.46875, gradient norms 0.06836 and 0.24023, and PPO KL 0.00213 and
0.00415. Maximum reserved training memory reached 52.56 GiB. This establishes
32K optimizer/rollout/relay integration, not elimination of reward fluctuations.
The final reward-specific regression also passed all 30 tests.

The subsequent independent long run is
`qwen35_9b_rewardfix_32k_longstable_20261021_to_20261022_1400`.
It starts from the original Qwen3.5-9B model, uses the corrected dataset copy,
all training rows, 16 prompt groups × 4 samples per update, 32K response budget,
8 FSDP2 training NPUs and four TP2 rollout replicas. Before training and every
five updates it evaluates the same 16 held-out questions with greedy decoding.
The deadline is **2026-10-22 14:00 Beijing / 06:00 UTC**. No model checkpoints or
disk recovery backups are enabled. The supervisor stops only this submission
through the Ray Jobs API and does not stop the shared cluster or other jobs.

Runtime directory:
`/tmp/verl-laminar-runs/qwen35_9b_rewardfix_32k_longstable_20261021_to_20261022_1400/`

- `train.log`: full training/validation output.
- `trajectories/scheduler.jsonl`: scheduler, version, migration and calibration events.
- `metrics.json`, `metrics.csv`: numeric metrics, including NumPy-wrapped scalar values.
- `curves.png`, `curves.svg`: raw/rolling training reward, fixed validation,
  actual budget-hit fraction, response length, gradients/KL, policy staleness
  and per-update timings. Refreshed every five minutes and once at termination.
- `long_stability_status.json`, `long_stability_history.jsonl`: Ray status,
  deadline, completed updates, validation versions and monitor heartbeat.
- `long_stability_watch.pid`, `long_stability_watch.log`, `plot.log`: supervision diagnostics.
- `watch_reward_longstable.py`, `submission.json`, `source_snapshot.zip`,
  `source_sha256.json`: supervisor, submission identity and source provenance.

The detached supervisor was verified reparented to PID 1. It backfilled all
earlier log metrics when started; collection does not require a live terminal.
Rendering runs in a bounded child process so plotting cannot delay the stop
request. A failed job is recorded and is not silently restarted from the base
model. At deadline an active update can be interrupted; only completed updates
appear in the training curve. The no-checkpoint policy means final weights
are not retained after the job stops.

At **2026-10-22 09:13 Beijing**, 24 updates had completed and Ray reported RUNNING.
All recorded gradient/KL values were finite. The fixed validation scores at
versions 0/5/10/15/20 were **0.50 / 0.4375 / 0.4375 / 0.4375 / 0.375**.
The latest raw reward was 0.234375, rolling 128-group mean 0.380859, and budget-hit
fraction 0.640625. Maximum observed trajectory staleness was 10 versions. Thus
this run has **not** demonstrated improved validation accuracy or removed the
length-related reward fluctuations. Sixteen validation items are too few to
make a strong generalization claim. Keep this configuration fixed until the
planned endpoint to obtain an interpretable stability trace; evaluate freshness
control, long-tail admission and larger held-out evaluation as follow-up work.

## Synchronous control requested after early stop

The user stopped the asynchronous run before its deadline and requested a
strict synchronous comparison with the same configuration. Ray confirms STOPPED
for the old job and RUNNING for `qwen35_9b_sync_32k_control_20261022`.
See [SYNC_COMPARISON.md](SYNC_COMPARISON.md) for batch/weight barrier semantics,
the resolved configuration diff, tests, deadline and comparison-curve paths.
The original async 14:00 schedule above is historical; it will not be restarted.
