# Reward stability for thinking-enabled mathematical RL

## October 30: both50-step jobs complete; initial-baseline reliability takes priority

Update at05:20 UTC: the actual rank0 scheduler/model-forward trace was joined
to all128 prompt identities per pool for initial versus same-assignment repeat.
The first scheduled model-forward request/token cohort changed for75/128 and
76/128 questions respectively. In pool0 all24 score changes occurred among
questions with changed first cohorts; in pool1 this was true for11 of14 score
changes. The remaining three demonstrate that unchanged first cohort alone
does not guarantee unchanged output. Actual later decoding schedules and other
state can differ; this is an observational association, not isolated causality.
The evidence and per-question joins are in the v3 phase-audit directory's
`repeat_first_forward_cohort_audit.json`. The fourth phase remains in progress.

Update at05:17 UTC: an isolated full128/32K invariant matrix is prepared and
its CPU controller is live, waiting for the mode0 v3 SDK job to SUCCEED and all32
NPUs to be free. Its immutable plan is in `full128_invariant_matrix_20261030_0518/`,
SHA256 `e035798bff75395c6d87c72aab49c1a62c14d3a34dc6adba9fd91239bb446860`.
The mode1 plan retains the same questions, four phase assignments, sampling,
two loading routes, TP topology and32K budget as mode0. It adds the opt-in sum
compatibility worker and requires actual compatibility metadata on both ranks.
CPU extension method-resolution checks passed; actual full-matrix NPU execution
has not started. A live-SDK preflight rejects the still-running mode0 prerequisite
before allocating NPUs. Controller submission does not release held endpoint
observers or automatically accept reliability/launch36-hour training.

Update at05:15 UTC: the v3 SDK job remains RUNNING in its fourth phase;
the first three complete phases have been independently scored for128 questions
per pool. Initial/repeat/swapped correct counts are pool0 **65/63/59** and pool1
**70/62/64**. Same-assignment repeat changes the binary score on24/128 questions
in pool0 (11 gains,13 losses) and14/128 in pool1 (3 gains,11 losses); only48/128
and57/128 entire response-token sequences respectively remain identical. Of
the score changes,21 and14 respectively involve an unfinished response. This
directly establishes substantial execution variation for full32K generation
with initial parameter hashes unchanged, even when the assigned questions and
submission group size match. Host scheduling/decode batches were not fixed;
this does not attribute the historical13/128 gap quantitatively to one cause.
See `full128_matrix_v3_20261030_0300/independent_phase_audits/`
`first_three_phases_paired_audit.json`. The fourth phase, final matrix scoring,
full-set mode1 verification, model comparison and reliability review remain open.

Update at03:12 UTC: the earlier full128 matrix SDK job failed in its diagnostic
request-identity assertion after preserving one16-response group. vLLM adds an
8-character suffix to internal scheduler IDs while output objects use external
IDs. This was a diagnostic join error, not a new training failure. Its30 pinned
sources and retained raw/trace data are archived with a failure diagnosis.
The corrected join uses unique exact prompt-token sequences on both TP ranks,
keeps both IDs and fails closed on ambiguity. Nine CPU checks include the real
failed-run16-response trace. An actual4-NPU, two-route,64-token smoke job then
passed64 responses across batch16 and two chunks8; it is protocol validation,
not an accuracy measurement. Native decoder parity was rechecked on128 packets.

The corrected full128/32K job `qwen35_full_native128_initial_matrix_v3_20261030`
is RUNNING on32 NPUs in its first phase. Its plan is in
`full128_matrix_v3_20261030_0300/`, SHA256
`cee10ba6db9f4e5e4a61f9be1c1c0d4be4b31fa41c3e1323d7d444915b6fe8ab`.
The driver now observes all already-started replicas before reporting a phase
failure so another diagnostic error does not discard their raw outputs. A live
CPU observer independently audits each completed128-per-pool phase; no complete
phase or new full-set score is available yet. Original539 benchmark pins remain
unchanged; earlier v2 status entries below are historical.

The isolated sum-compatible batch-invariant rank control has SUCCEEDED on real
NPUs with visual profiling enabled. Independent rehashing verified identical
complete TP parameters before/after and against mode0, plus active compatibility
dispatch on both ranks. On the same11 forced-prefix contexts, mode0 changed the
next token for6 alone-versus-batch probes; mode1 changed it for0. All11 mode1
repeat and batch comparisons had identical recorded top scores. This is a
targeted numerical result, not full128/32K accuracy or complete historical-gap
causality. Its evidence is `rank_invariant_sum_compat_priority_20261030_0300/`
`independent_rank_rescue_audit.json`. Full-set mode1 and common-policy evaluation
are still required before accepting alignment or starting36-hour stability.

Update at01:54 UTC: the full32-NPU initial matrix remains SDK RUNNING in
`initial_assignment`; all16 engines passed initial parameter identity and no
full phase has completed yet. The next isolated sum-compatible mode1 rank
control is now concretely queued, not executed, under
`rank_invariant_compat_control_20261030_0155/`. Its immutable plan SHA256 is
`53d8faa0270b201770b716dda34574387f40020000cdf8105449c487d38af4f8`.
Its source-pinned controller waits for the actual full-matrix SDK SUCCEEDED,
all eight128-question score sets, full source/held-observer checks and all32
free NPUs before submitting one deterministic2-NPU SDK job. It never resumes
endpoint observers or launches training/stability.

The next control reuses the exact original rank experiment on all11 actual
mode0 forced prefixes and restores original visual profiling (no skip flag).
Only the extension selects the current-repository opt-in sum guard. Both TP
workers must report actual torch.sum/Tensor.sum/backend ownership before and
after, in addition to existing runtime/cache/full-parameter checks.
Three new CPU checks execute the actual extension source against a fake backend
using vLLM's dynamic-base-class injection order: preparation precedes parent
allocation/native initialization, native dispatch installs the prepared sum,
and late foreign replacement is rejected. They import no torch/use no NPUs.
The earlier21 semantic tests still cover the unchanged guard implementation.
The current worker adapter's source is pinned by the new lifecycle receipt;
the earlier CPU receipt retains the adapter's historical pre-metadata hash.
The actual RUNNING-matrix guard rejects execution before Ray init/output
creation. All539 original pins and three held endpoint observers still pass.
No actual NPU compatibility result or accuracy acceptance exists yet.

Update at01:47 UTC: the original prefix SDK job is SUCCEEDED and its independent
CPU audit passes. All loaded TP parameter records/bytes match before and after;
batch1 initial/repeat/after-batch16 sequences match16/16, including exact8192
chosen logprobs for the first repeat. Batch16 matches5/16. Six independent
forced-identical-prefix alone-versus-mixed-batch probes choose different next
tokens. This establishes actual batch-dependent inference for these controlled
prefixes, not an optimizer update or proof that all historical13/128 score
differences have that cause.

The full128/32K matrix SDK job
`qwen35_full_native128_initial_matrix_v2_20261030` is RUNNING on32 NPUs in
`initial_assignment`. All16 TP2 engines initialize successfully and all initial
full TP parameter fingerprints match across replicas and both loading routes.
Each native-callback TP worker loads760 tensors in9 buckets of up to2048MiB
and performs one post-load processing pass. This verifies controlled current
loading-route parameter parity, not missing historical Async0 resident state.
See the matrix's `initialization_independent_check.json`; scores and repeated
full128 phases remain pending.

The text-profile mode1 control is also FAILED, now during first sampling:
`batched_count_greater_than: (x >= values).sum(-1)` is routed to the floating-only
CANN sum operator, which rejectsDT_BOOL. Actual mode1 runtime and full parameter
identity against mode0 were verified before generation. It produced no rank
result; the earlier visual-dimension failure was avoided but not generally fixed.
Complete SDK logs and both failure diagnoses are retained.

An opt-in current-repository compatibility candidate is now in
`verl/utils/vllm/ascend_batch_invariant.py`, with21 CPU tests passing in
`tests/utils/test_ascend_batch_invariant_sum_on_cpu.py`; Ruff and diff checks pass.
It preserves the supported floating last-axis invariant kernel and delegates
unsupported types/dimensions/overloads to original torch.sum, preserving boolean
rank-count int64 promotion, dtype/out semantics and normal errors. Importing it
does nothing; only an explicitly enabled future worker can prepare its local
dispatch. Ownership/idempotency checks prevent overwriting a foreign sum patch.
Unsupported floating fallback alone does not promise global batch invariance.
The isolated future worker adapter is prepared; no running worker/dependency
file was changed and no NPU compatibility/invariance result is yet available.
The CPU receipt is `initial_baseline_priority_20261029/sum_compat_cpu_validation.json`.
Original539 benchmark pins remain unchanged. Final accuracy/learning/alignment
and36-hour stability acceptance remain false.

Update at01:39 UTC: the unmodified mode1 rank-control SDK job is FAILED before
generation. The actual CANN error is `aclnnReduceSumBatchInvariant` / "Provided
dim only support last dim", from visual profiling's
`Qwen3VL.pos_embed_interpolate_native: embeds.sum(dim=0)`. CPU availability of
the invariant library did not establish runtime compatibility. The complete SDK
log/failure diagnosis are retained in `rank_invariant_control_20261030_0135/`.
No token/rank or accuracy result exists for that attempt.
A separate same11-context control, now RUNNING as
`qwen35_initial_rank_invariant_text_control_20261030`, changes only
`skip_mm_profiling=True` in addition to mode1, avoiding the failing unused visual
profiling for these all-text inputs. Its plan is under
`rank_invariant_control_text_profile_20261030_0139/`, SHA256
`bfbd1918859b7adebb0027fc3c98967a49371b94c425796d056c3eb46cb4b7c5`.
All full TP identity/cache/runtime checks remain required. This is not a fix
for general multimodal invariant execution, and no dependency/core training
source was changed. The original mode0 prefix is still RUNNING.

Update at01:36 UTC: three actual short-prefix phases are now complete. All16
batch1 repeats have identical512-token sequences and all8192 corresponding
chosen-token raw logprobs match exactly. Batch16 preserves only5/16 sequences;
11 diverge, earliest at generated token9 (zero based). Before divergence,2977
of4112 corresponding chosen logprobs differ, maximum absolute difference
0.1491307020. First-divergence candidate rank gaps include0/±0.125/0.25.
These are direct numerical/batch-dependence observations, not full128 score
causality. Final full loaded-parameter unchanged checks, post-batch sequential
phase and forced-prefix probes are still running.

An additional isolated2-NPU job,
`qwen35_initial_rank_invariant_control_20261030`, is now actually RUNNING.
It uses all11 witnessed first-divergence contexts, forces identical prefixes,
and compares alone/mixed-batch16/alone-repeat next tokens with the actual Ascend
batch-invariant mode enabled. It must verify strictHCCL/LCCL1, initialized
invariant dispatch and full TP parameter identity against the mode0 engine,
before comparing generations. Its source/plan are pinned under
`rank_invariant_control_20261030_0135/` (plan SHA256
`4d4fb73153f22a428949cda484c37794f647d252736dc0cbad378dbbf1419aa4`).
This new targeted control explicitly permits its own exclusive device pair
alongside the existing2-NPU prefix job; it never changes the original frozen
prefix plans/inference or runs training. The full32 matrix controller continues
waiting for all32 free, so cannot overlap either diagnostic. No mode1 result
or full-accuracy improvement has yet been observed or accepted.

Update at01:32 UTC: the first unbatched16-question/512-token prefix phase is
complete; the same-assignment repeat is still running. Four new decoded prefixes
match the historical Async0 start in full and seven match Sync0. This is a
short-text comparison with different request/batch settings, not recovered
historical initial token/state identity or a causal explanation. The independent
prefix audit now verifies per-token raw-score consistency, full TP parameter
records and same-forced-prefix next-token rank witnesses once all phases finish.

The initial prepared full128 plan was superseded **before any SDK submission or
NPU execution**: its CPU scorer used `skip_special_tokens=False`, whereas the
actual native naive reward manager usesTrue. The waiting controller3345061
was confirmed waiting with no submission ID and the planned SDK job absent404,
then terminated; all of that plan's pinned sources were archived in
`full128_matrix_20261030_0120/prepared_source_snapshot.zip` with its original
plan and a supersession receipt. Only the new diagnostic scorer/preparation
changed; no original539 benchmark pins, native reward source, running prefix
worker, completed training or logged scores changed.
The corrected decoder was tested on all128 actual Sync50 raw packets:
the fast tokenizer withTrue equals the native tokenizer and logged output for
128/128. False produces different text for98/128 but **zero** different extracted
answer/reason pairs on those128; this preparation correction does not explain
the historical initial13/128 difference. The corrected immutable plan is
`full128_matrix_v2_20261030_0132/evaluation_plan.json`, SHA256
`d6cc412ef90f72bd4d4f25656aee02632076ff1605706570545076595fe07386`.
Its controller3348328 is live waiting on the same prefix SDK job; its new SDK ID
will be `qwen35_full_native128_initial_matrix_v2_20261030`. Actual RUNNING-prefix
guard rejection was reverified before output-directory creation orRay init.
Full128 NPU execution, root review and reliability acceptance remain pending.
The01:20 plan/controller details below are the retained earlier preparation.

At 01:20 UTC the actual Ray SDK reports `reasoning_manifest.json` (Sync)
and `qwen35_9b_async_32npu_fp32_32k_fp32ssm_20261027_060119` SUCCEEDED.
Original-reference fixed128 scores at0/5/.../50 are Sync
73/65/69/76/79/73/85/83/88/86/83 and Async
60/64/65/73/70/75/74/82/85/85/85. These remain provisional:
the initial13/128 difference is unresolved, and own-baseline subtraction
does not establish evaluation reliability or comparable learning.
The final curve is in `fixed_validation_overlay_20261030_0120/` under the
comparison campaign. Both final public BF16 payloads were exported successfully;
the Sync50 export reports760 tensors matching its published manifest.

All50 Sync updates have complete64-response audit receipts. Newly completed
updates42..50 were independently rehashed against all1152 archived input/forward
files and27 native event/metric prefixes. Across50 updates the positive
thinking-close clipped derivative count is1/259 and the end-marker count0/258.
Updates11/16/18/24/36/50 have zero nonzero-advantage policy tokens; this is not
zero optimizer movement, since KL/weight decay can still contribute.
Sync45/50 ending-token audits and complete archive/source provenance pass.
All8 initial-success losses at Sync50 are unfinished32K responses,7 with
strong repeated tails;18 initial failures become correct. This describes
observed outcomes and does not isolate optimizer versus serving causality.

The previous CPU prefix waiting process was confirmed absent and its intended
SDK job absent (404). The same immutable controller plan was restored; it
submitted exactly `qwen35_native_initial_prefix_priority_20261029`, now RUNNING.
No training or cluster was restarted. Three final-comparison CPU observers
(1135896/1497489/2058191) remain SIGSTOP with original argv/start identities;
all539 original evaluation pins remain intact. Their release still requires
full128 initial-baseline reliability/root review, never merely a short probe.

The complete initial128/32K matrix is prepared in
`initial_baseline_priority_20261029/full128_matrix_20261030_0120/`.
Its immutable plan SHA256 is
`0015ec26adeca2ac6d7318d932d5e15e85bf0afa12d1edb6a426119abf9100b5`.
Controller PID3345061 is live waiting for that exact prefix SDK job and all32
free NPUs. Sixteen TP2 engines form two8-replica pools; the experiment repeats
identical assignment, swaps historical Async/Sync assignments, then uses
smaller submission chunks. Every phase covers all128 questions per pool with
no outcome-based selection. Sampling matches inspected native greedy settings,
including request seedNone, detokenizationTrue, chosen-token logprobs0 and
actual model-added EOS capture. It checks full TP parameter identities,
actual request prompt IDs, effective sampling, scheduler/model tensor shapes,
and metadata-only GDN operation input/output dtypes.

Pool0 directly loads canonical initial BF16 weights; pool1 initializes dummy
weights then uses the standard verl BF16 loader callback with the native2048MiB
bucket size and exactly one standard post-load processing pass after all buckets.
This does not replay original transport/publication barriers or recover absent
historical Async0 resident state. Hardware/pool and route effects require a
follow-up if they differ. Submission chunks reproduce controlled groups, not
historical sliding admission, node placement or decode timings.
Ten metadata-wrapper CPU checks and ten assignment/loader CPU checks pass.
The real RUNNING-prefix prerequisite was rejected before execution-directory
creation/Ray initialization. These are preparation/guard results, not actual
32NPU inference verification. Full matrix execution and reliability acceptance
remain pending. Both controllers prohibit automatic final-evaluation resumption
or36-hour stability launch. The36-hour run has not started.

## October 24 environment restoration and diagnosis

The new environment retains the model under
`/opt/huawei/dataset/omni_training/downloaded_models/Qwen3.5-9B`, but not the
historical run logs or prepared dataset. Historical measurements below are
records from the old environment, not new measurements.

`prepare_deepscaler.py` now restores data from a pinned upstream revision and
verifies its SHA256. The 40,315 source rows include six empty-answer rows;
569 rows belong to exact repeated problems with different reference strings.
Those rows are excluded conservatively, without claiming every differing
reference string is mathematically contradictory. Another 783 exact duplicate
rows are removed. The resulting 38,957 questions form 38,445 training questions
and 512 held-out questions, with deterministic seeded hash ranking and no exact
problem-hash overlap. Near-duplicate paraphrases are not detected. The original
JSON and a manifest retain provenance. These are newly reconstructed splits,
so they cannot support a direct numerical comparison with the old validation set.

`prepare_reasoning_math_data.py` produces a separate `deepscaler/reasoning`
copy, repairs the one audited training reference described below, and preserves
the held-out parquet byte-for-byte. The reasoning launcher's default fixed
evaluation subset increases from 16 to 128 questions. This improves evaluation
resolution but does not establish a learning gain.

Per the requested experiment constraint, **no staleness cap or staleness-based
eviction is enabled**. The existing 64-group prompt buffer and sampling policy
are retained. Additional records join producer-ready time, reward computation
duration, behavior version, trainer selection time and actor version. Actual
queue waiting uses the last sibling's producer-ready wall timestamp; it requires
aligned clocks across nodes and includes the final enqueue latency. The separate
`observed_ready_wait_s` measures only time after the trainer first observes a
complete group and must not be interpreted as total queue waiting.

`examples/trajectory_async/run_qwen35_9b_reward_diagnosis.sh` resolves to 50
updates on 32 NPUs (16 trainer + 16 rollout), 32K responses, fixed evaluation before training and every five updates,
and console plus JSONL metrics. It reuses an existing Ray cluster and disables
`verl_omni_ext` loading through both explicit external-module loading and plugin
auto-discovery. The installed `mooncake-transfer-engine-npu==0.3.11.post1` is
present on both nodes, but its device-transport initialization failed inside a
CPU relay actor. The Laminar `mooncake_relay` path transfers CPU snapshots;
the standard `mooncake` checkpoint backend's NPU `ascend_direct` path is a
different implementation. Restoring the former requires an isolated CPU-capable
transfer dependency, not reinstalling the system NPU package.

Two-update, shorter-context attempts are environment integration checks only.
They have not established a new learning result; neither corrected labels nor
additional diagnostics establish that the historical reward regression is fixed.

The isolated CPU transfer engine was rebuilt under `.runtime/mooncake-cpu`,
without replacing the system NPU package. Two CPU actors on different nodes
completed bidirectional TCP transfers of 1,048,583 bytes each with matching
SHA256 digests. The preliminary 16-NPU run published version zero, loaded it
on all four rollout replicas and passed a real same-version migration probe.
It was stopped before any optimizer update when the user selected a sequential
32-NPU comparison; its partial validation is not a complete baseline estimate.

The 32-NPU comparison keeps 16 prompt groups × 4 responses per update, seeds,
32K budget, reward and held-out questions unchanged between modes. Its resolved
configuration comparison permits only the synchronous flag and experiment paths
to differ. The experiments run sequentially because the installed TransferQueue
uses a fixed global controller namespace. The supervisor waits for all 32 NPUs
and the previous controller to be released, starts the synchronous control only
after asynchronous success, and records failures without silently restarting.
Ray remains running throughout. Source hashes, dataset manifests, JSONL metrics,
version/queue traces, validation outputs and periodically refreshed plots are
retained under the repository's `outputs` directory. Learning-quality acceptance
still requires the actual asynchronous and synchronous results.

### Confirmed optimizer precision defect; convergence verification pending

A read-only probe of the live 32-NPU workers found all 9,409,813,744 actor
parameter elements stored in BF16. The optimizer was ordinary
`torch.optim.adamw.AdamW`, receiving BF16 parameters at learning rate `1e-6`,
without FP32 master parameters. This was caused by the base launcher's explicit
`actor.fsdp_config.model_dtype=bfloat16` override; it controls parameter storage,
not merely forward mixed precision. The engine's usual training default is FP32.

Controlled CPU tests on four 128×128 checkpoint slices retained unchanged
fractions of 96.4–97.6% after 50 repeated AdamW steps with fixed gradients.
FP32 master parameters accumulated every update; after conversion for BF16
forward computation, 61.8–85.3% of those elements had changed. A separate scratch
test on an actual assigned training NPU used 1,024 parameters initialized to
0.015625 and gradient 1e-4: BF16 parameters were completely unchanged after 50
steps, while FP32 parameters accumulated an approximately 5e-5 update. Neither
probe modified the live model, its optimizer or RNG state. These are numerical
update tests, not RL learning measurements.

The shared trajectory launcher now keeps actor optimizer parameters in FP32,
while FSDP forward mixed precision remains BF16 and the reference model remains
BF16. Gradient synchronization already occurs on every microbatch in the resolved
configuration, avoiding retained full-model gradients during accumulation. The
old BF16-master trial was stopped before its first optimizer update; the repaired
32-NPU asynchronous/synchronous comparison starts again from the same original
checkpoint with matching settings. This fixes a demonstrated source of lost
small updates. Actual learning gains and asynchronous/synchronous accuracy
alignment still require the complete online results.

The user authorized a separate **36-hour trajectory-asynchronous stability run**
only after corrections pass regression, reward improvement is sustained, and
fixed held-out accuracy aligns with the synchronous control. Diagnosis/control
time is excluded from that 36-hour duration. This conditional run has not yet
been started.

### 32-NPU online result: precision repair alone does not solve convergence

The FP32-master asynchronous job
`qwen35_9b_async_32npu_fp32_32k_20261024_125226` completed all 50 updates.
On the same 128 held-out question/reference pairs, accuracy changed as follows:

| Update | 0 | 5 | 10 | 15 | 25 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 32.81% | 34.38% | 20.31% | 3.91% | 0.78% | 2.34% |

The final paired comparison has 39 regressions and no improvements;
unfinished-thinking fraction rises from 60.94% to 97.66%. Mean raw training
reward over the first ten updates is 0.475, versus 0.01094 over the last ten.
At update 50 all 64 responses reach the 32K budget, and all GRPO advantages
are zero. Actual behavior-version lag is mean 4.0625, maximum 10 at that step;
staleness remains unrestricted as requested. These fixed-set results establish
real generation degradation; completion-order bias alone cannot explain them.
See the run's `regression_analysis.json` and per-question validation dumps.
A key-by-key audit also confirms that all 800 sampled groups contain four
responses to the identical question and a single behavior-policy version;
within-group policy mixing is not observed in this run. See
`version_regression_analysis.json` for the join of saved correctness to actual
behavior versions. Version-zero samples are still selected at update 13.
The FP32 repair remains justified by the independent numerical tests, but
**is insufficient to establish the reward root cause or successful training**.

The subsequent synchronous job
`qwen35_9b_sync_32npu_fp32_32k_20261025_232518` fails before its first optimizer
update. The native error is `aclnnFlashAttentionScoreGrad`, error 207001,
`EL0004 Memory_Allocation_Failure`; this is not a successful synchronous
learning control. Its initial accuracy is 35.94%, also showing run-to-run
variation despite greedy generation. The retry
`qwen35_9b_sync_32npu_fp32_32k_actoffload_20261026_012044` enables activation
offload and saves checkpoints every five updates (retaining two). Learning,
data and sampling settings are otherwise unchanged. This changes the memory
execution path and must be recorded when interpreting the comparison.
The comparison helper now checks the actual resolved Hydra configurations,
allowing only the asynchronous/synchronous flag and run output names/paths to
differ. It reports these three retry differences and does not label the pair
as fully matched, even if both eventually finish 50 updates. Its self-checks
cover matching configurations, changed activation offload, and invalid control
modes; all pass. A successful retry can narrow the cause, but final alignment
requires matching execution settings in both modes.
No Ray cluster restart or external verl extension is involved.

The activation-offload retry also fails at its first backward, with zero
optimizer updates; its initial accuracy is 49/128 (38.28%). Two subsequent
isolated full-model probes use the real FSDP2 actor/reference strategy, FP32
actor masters, non-aligned length 33023, and backward outside autocast. Without
an allocator limit, the baseline fails before its first backward completes:
reserved memory reaches 60.375 GiB and driver-free memory falls to about
30 MiB. The saved native worker log reports `EL0004` while allocating a
runtime memory pool to load the `ReduceMean` kernel. With a 0.9 allocator
fraction, all 16 ranks complete four backward passes each, leaving at least
3.18 GiB driver-free. This demonstrates a remedy for runtime allocation
starvation; it is not a reward improvement or a successful PPO update.
Artifacts: `qwen35_fsdp2_memory_baseline_20261026_025945` and
`qwen35_fsdp2_memory_reserve10pct_20261026_025946` under `outputs`.
Earlier FSDP1/autocast probe runs are explicitly excluded from acceptance in
their `scope_correction.json` files.

FSDP now accepts an optional `memory_fraction` (default `None`, preserving the
device default). It sets the per-process allocator limit before model setup.
The diagnostic launcher sets both actor and reference fractions to 0.9,
enables activation offload, and saves every five updates while retaining two
checkpoints. Other launchers retain their configured behavior. All 42 relevant
CPU configuration/gradient synchronization/temperature tests pass. The subsequent
core-configuration probe `qwen35_fsdp2_memory_config_binding_20261026_031732`
passes a complete backward on all 16 ranks, with at least 5.00 GiB driver-free.
The live synchronous retry
`qwen35_9b_sync_32npu_fp32_32k_reserve10pct_20261026_032256` now uses the new
configuration; its resolved actor/reference limits are both 0.9. A read-only
CPU client archives real behavior logprobs from the existing TransferQueue,
without creating or clearing queues. A separately recorded forward observer
copies actor logprobs for the first three versions, leaving returned outputs
unchanged; it introduces diagnostic CPU synchronization/IO and is not a
throughput measurement. Its source hash and installation ranks are saved in
the run directory. Offline comparison matches unique identical token sequences
and the same policy version, and explicitly checks adjacent token offsets.

The synchronous retry's complete first training batch has 64 uniquely matched
responses and 1,444,710 response tokens. Before any optimizer update, actor
versus true behavior logprobs have RMS difference 0.1020, with 4.23% of token
probability ratios outside [0.8, 1.2]. The correct offset has much smaller RMS
than either neighboring offset (1.1550 and 1.1672), excluding a simple one-token
shift in this batch. Position bins show 0.73% outside the interval in the first
4K response tokens and 7.12% in the final 28K–32K bin. All recorded old logprobs
equal the true behavior logprobs. The 16 groups each contain four responses,
and their recorded GRPO advantages match the observed group reward patterns;
28 responses have nonzero advantages. These observations prioritize long-context
training/inference numerical consistency, but do not establish that this mismatch
causes the learning collapse. See `logprob_alignment_version0.json` and
`initial_batch_audit.json` in the current retry's output directory. The first
optimizer update subsequently succeeds: gradient norm 0.2388, raw batch reward
0.53125, actual behavior lag zero, and peak reserved training memory 55.14 GiB.
The second batch starts from published policy version one. Read-only probes on
ranks zero and eight confirm FP32 AdamW parameters/moments and exactly one
optimizer step. Across eight saved 4,096-element parameter slices, all but one
element change, with mean absolute changes around 0.69–0.93e-6.

A coordinated no-grad replay of the exact first batch, while the next batch is
being generated, verifies the first update's surrogate objective direction.
Recomputed pre-update policy loss 0.08025294 agrees with the training log's
0.08025293. Total policy-plus-KL loss decreases from 0.08025325 to 0.08005881
(change -0.00019445). All 16 ranks remain at optimizer step one before and after
this replay. This supports correct first-step loss aggregation and update
direction; it is not held-out learning evidence and does not establish later
stability. The replay uses evaluation mode versus the recorded training-mode
forward, and adds diagnostic computation. Artifacts:
`first_update_objective.json`, `first_update_objective_records.pt`, and
`first_update_parameter_change.json`.

The second synchronous update also completes, with finite gradient norm 0.1311,
reward 0.4375 and behavior lag zero; the first two rewards are not a learning
trend. Same-version probability comparison for its 64 responses (1,582,480
tokens) gives RMS logprob difference 0.1045 and 4.55% ratios outside [0.8, 1.2].
See `logprob_alignment_version1.json`; the continuously refreshed combined
`logprob_alignment_analysis.json` retains pairs from multiple versions.

Two passive, one-time captures from layer zero on ranks zero and eight retain
three real GDN heads at lengths 32,829 and 26,196. Activation offload uses
reentrant checkpoint recomputation, so the capture's root-counter hint can be
one forward ahead. The analysis rejects that hint and requires a unique full
sequence length at the same rank/version, then joins the corresponding exact
root input hash to the archived training input. Source and timestamps are saved.
CPU fixed-input recurrence tests isolate state rounding from normalization:
BF16 state writes after every decode token introduce relative L2 differences
of 0.00781 and 0.00560 versus FP32 states under FP32 query/key normalization.
FP32 recurrence outputs are closer to the actual training chunk outputs in
these cases. This covers three heads in one layer, not the actual Ascend decode
kernel or full-model generation accuracy. BF16 autocast also affects training
chunk matrix operations; FP32 recurrence is not a complete reproduction of the
training execution path. Reports: `gdn_state_precision_rank0_actor1.json` and
`gdn_state_precision_rank8_actor1.json`. This evidence supports a controlled
precision ablation, but does not identify the reward-collapse root cause.

The subsequent no-grad full-model ablation uses one real 9,013-token response
from behavior version two, while every actor remains at optimizer step two.
All 16 ranks produce identical outputs. Baseline chosen-token logprob RMS versus
the true behavior is 0.05045; FP32 query/key normalization gives 0.05148,
FP32 chunk matrix operations 0.05102, and both together 0.04993. Probability
ratios outside [0.8, 1.2] are respectively 1.331%, 1.442%, 1.343%, and 1.354%.
The candidate normalization is actually invoked 96 times per rank across its
two cases. These closely similar results do not support changing training
normalization or chunk precision as the current remedy. This is one sequence
at 9K, not a full 32K decoding-cache ablation. Every original function is
restored and all optimizers remain at step two; no installed source is edited.
See `gdn_norm_alignment_probe.json`, its saved inputs/outputs and source copy.

The completed asynchronous run's recorded PPO metrics also do not show gross
clipping saturation: PPO KL is about 0.00318 at update 10 and 0.00305 at update
50, while policy clip fraction is 0.00070 and zero respectively. At update 50
all advantages are zero, so zero clipping is not evidence of healthy learning.
Reference KL grows from about 0.00028 at update one to 0.03573 at update 50.
These observations neither rule out sequence-level effects of stale training
nor establish them as the cause. The raw table is preserved in the original
async run's `ppo_drift_diagnostic_table.json`.

A further read-only audit matches the first 128 saved training inputs to their
actual sampler selection records. Actor and behavior versions are kept separate
so the same diagnostic can capture stale asynchronous inputs without imposing
an age limit. The current synchronous records all have age zero. Evaluating
the real vanilla PPO loss with CPU autograd finds zero policy-loss derivatives
on 2.21% and 2.65% of tokens with nonzero advantages in the first two batches;
these exclude KL derivatives and must not be called full parameter-gradient
zeros. Reconstructed policy losses, 0.08025294 and 0.06086902, match their live
training logs. Standard positive/negative-advantage and dual-clipping cases are
checked explicitly. Artifacts: `selected_input_identity_audit.json` and
`selected_policy_drift_analysis.json`. Future asynchronous evidence still needs
actual stale samples; the synchronous audit does not prove async correctness.

The saved first-update replay also shows token-weighted mean chosen-token
logprob changes of +0.00032284 on positive-advantage responses and -0.00003675
on negative-advantage responses. This supports the first update's observed
direction, subject to the recorded training-forward versus no-grad evaluation
scope. It does not establish held-out improvement or stop-token behavior on
unseen prefixes. See `first_update_termination_token_audit.json`.

The third synchronous update completes after the temporary precision ablation,
with finite gradient norm 0.1076, reward 0.546875, and behavior lag zero.
Its 64 responses give reconstructed policy loss 0.02448063, matching the live
training log, and 3.28% zero policy-loss derivatives among nonzero-advantage
tokens. All 192 captured training inputs pass the selected-identity audit.
The first three rewards (0.53125, 0.4375, 0.546875) are different training
batches and do not establish a learning trend. Fixed held-out evaluation is
still only available at the initial policy at this point.

An exact token-diversity audit also checks all 48 four-response prompt groups
from these three batches. Every group has four distinct responses, with mean
common prefixes of 1.25, 2.25 and 4.94 tokens respectively. The falling
mixed-reward group counts (7, 4, 2) therefore are not explained by exact duplicate
generation in these batches. This does not assess mathematical solution
diversity. See `sibling_response_diversity_audit.json`.

The fourth synchronous update completes with finite gradient norm 0.1876,
reward 0.390625, and behavior lag zero. The next batch is generating; the first
post-training fixed validation and checkpoint at update five remain pending.

Migration auditing joins saved request IDs, versions, token counts and counters
in the completed asynchronous run. It records 990 real migrations across 884
requests. At update 50, 34 of the 39 fixed-question regressions occur without
migration in either evaluation. Among the 107 questions never migrated in
both evaluations, correctness falls from 37 to 3 (34 losses, zero wins; exact
paired p=1.16e-10). This rules out direct evaluation-time migration as an
explanation for those regressions, but does not rule out effects of earlier
training migrations. The detailed joined report is the async run's
`migration_regression_analysis.json`.

The fourth synchronous batch supplies a direct observation of long-prefix
recomputation. A passive observer records its actual actor forwards, adding
CPU-copy timing overhead but preserving outputs. All 64 sequences match their
real same-version behavior scores. Four trajectories have six migration events;
three events have complete 1,024-token windows on each side. In these windows,
actor/behavior logprob RMS falls from 0.14822 before migration to 0.03000 after,
and ratios outside [0.8, 1.2] fall from 7.75% to 0.16%. Nonmigrated sibling
windows do not show the same consistent drop. Generated tokens and runtime
batching differ across these windows, so this is not a controlled continuation
counterfactual or accuracy proof. It prioritizes accumulated decoding-state
error; it does not justify disabling migration as an accepted fix. See
`migration_probability_boundaries_version3.json` and the recorded observer source.

Actual native-operator checks then replay all 32K captured response inputs.
The first probe loads the extension after the training process has initialized
CANN: BF16 passes, while FP32 is rejected with `EZ1009`/operator-prototype
matching failure. The installed metadata and binaries nevertheless include
both state dtypes. Fresh isolated processes set custom-op/API-library paths
before importing torch and successfully execute both dtypes on both nodes.
The two nodes produce bitwise identical output tensors. Against the captured
training chunk output, BF16-state relative L2 error is 0.007291, versus
0.003204 for FP32 states. This tests three layer-zero heads tiled into the
production TP2 head dimensions (NK=8, NV=16), initialized from a CPU FP32
prefill state. It is not full-model autoregressive inference or learning proof.
Both parent workers remain healthy at optimizer step four; installed packages
and live model/optimizer parameters are unchanged. The original late-loading
failure must not be interpreted as missing installed FP32 support. See
`native_gdn_precision_two_node_verification.json` and
`native_gdn_probe_scope_correction.json`.

A configuration-only check initially resolves an asynchronous candidate with FP32 SSM
cache, FP32 actor masters, BF16 training forward, the 0.9 allocator limits and
unrestricted staleness. It submits no job and leaves the current baseline
unchanged at that point. Full-model and online asynchronous/synchronous
verification remain required before attributing reward recovery to this
candidate or starting stability.

Two bounded precision checks narrow the investigation without changing the
active source. All 768 values in the checkpoint's 24 FP32 `A_log` vectors
round-trip BF16 exactly, so initial `A_log` export rounding does not explain
the low baseline. On one assigned training NPU, deterministic scratch-only
GDN forward/backward checks at 65 and 257 tokens produce finite gradients.
Against the CPU recurrent reference, the actual BF16-autocast chunk path has
forward relative L2 error about 0.0045–0.0048 and gradient cosine similarity
at least 0.99994 in these cases; FP32 chunk operations are closer. This does
not establish parity at 32K, validate full-model gradients, or identify the
cause of learning collapse. No live weights, optimizer or RNG state were
changed. Reports: `precision_parity_candidate.json` in the comparison directory
and `gdn_gradient_scratch_probe.json` in the synchronous retry directory.

An offline answer-extraction audit leaves official scoring untouched and
allows the last boxed answer even when thinking has not closed. Correct counts
at updates 0/5/10/15/50 are then 48/49/31/9/5 out of 128, versus the official
42/44/26/5/3. Only one of the 39 final paired regressions has a correct last box
inside unfinished thinking. Relaxing the closing-tag rule alone therefore does
not remove the observed regression. This diagnostic cannot establish whether
an unfinished trace would eventually solve the problem with a larger budget.
See `unfinished_answer_audit.json`; these relaxed counts are not acceptance
metrics and must not replace held-out final-answer accuracy.

### Synchronous control also collapses at behavior lag zero (October 27)

The allocator-headroom synchronous retry completes 26 logged PPO updates with
behavior lag zero. On the same 128 held-out questions, correct counts at updates
0/5/10/15/20/25 are 46/29/8/5/2/0. The final paired comparison loses all 46
initial successes with no wins (exact two-sided McNemar p=2.84e-14). Update 26
has raw reward 1/64 and 63/64 responses hit the 32K response budget. This
establishes a shared failure in this training recipe; asynchronous staleness
alone cannot explain it. It does not establish the numerical-cache hypothesis
as the cause, or rule out reward/optimization interactions.

Before stopping this failed control, all 16 model, optimizer and extra-state
shards at checkpoint 25, its FSDP2 world size and data checkpoint were verified
structurally. The SDK then confirms STOPPED and all 32 NPUs are available;
the Ray cluster was not restarted. See `checkpoint25_integrity_manifest.json`
and `regression_stop_decision.json` in the synchronous retry directory.

A separate offline audit of these synchronous outputs allows the last boxed
answer inside unfinished thinking. Correct counts become 52/38/10/8/4/3;
only two of the 46 final paired regressions are rescued. Two earlier unfinished
responses raise the shared legacy box extractor's assertion and are counted
separately (one at update 0, one at update 10); update 25 has no extraction
errors. Thus closing-tag rejection alone does not explain collapse. Official
scores and training rewards remain unchanged; see `unfinished_answer_audit.json`.
Sparse terminal rewards can reduce GRPO learning signal as groups become all
incorrect, but this is not evidence that reward sparsity initiated regression.

The synchronous fixed-set failure is predominantly unfinished, repetitive
generation. Unfinished responses increase from 76/128 initially to 128/128 at
update 25. Using a separate descriptive measure (duplicate 64-character windows
within the final 8192 characters), 57/128 initially versus 121/128 at update 25
have at least half their tail windows duplicated. Of the 46 lost initial
successes, 45 meet that repetition threshold. Manual examples include a cube
volume trace that repeatedly calculates the correct value 8 but never leaves
thinking. These observations do not grade unboxed mathematical content or
identify a causal mechanism; a failure to finish must not be described as proof
that all underlying mathematical ability vanished. The older asynchronous run
also shows unfinished/repetitive generation (125/128 unfinished and 87/128 over
the same repetition threshold at update 50). See each run's
`validation_repetition_analysis.json`; official scoring remains unchanged.

The original model's `text_config.mamba_ssm_dtype` is `float32`, whereas the
previous Qwen3.5 launcher explicitly selected BF16 SSM cache; installed vLLM emits a
warning for this override. The two full-model TP2 probe engines initialize
successfully with BF16 versus FP32 SSM cache: per TP rank, the BF16 case has
64 BF16 cache tensors, while the FP32 case has 40 BF16 and 24 FP32 tensors.
All 36 sampled parameter hashes per matching TP rank agree. The first replay
attempt then fails a diagnostic history guard because vLLM-Ascend asynchronous
scheduling uses a host-side `-1` placeholder. This is not a numerical result.
The corrected probe allows that documented placeholder and still requires
exact equality of all final recorded/generated token IDs and finite raw
pre-processor log probabilities. Both corrected engines pass the first 64-ID
check and report nonzero negative raw scores rather than the forced
distribution's zero scores. Full 32K replay is running as
`qwen35_full_model_cache_parity_20261027_052554`; metadata and this short check
alone do not establish the source of reward collapse.

The current Qwen3.5 launcher now matches the checkpoint's FP32 SSM state
precision and retains BF16 convolution cache. Bash syntax and the actual
diagnostic launcher's resolved Hydra config pass: actor FP32 masters, BF16
forward, allocator limits 0.9, activation offload, 32K response budget and
unrestricted staleness are preserved. This fixes a confirmed configuration
override; numerical benefit and reward convergence remain unverified.

A second numerical check isolates residual normalization. Actual installed
Ascend Gemma RMSNorm with five real checkpoint norm vectors and fixed scratch
BF16 activations differs from HF by relative L2 0.00313–0.00367 in the fused
residual path. Its returned residual is bitwise identical to BF16 addition.
Explicit BF16 addition followed by the native Gemma kernel reduces error to
0–9.94e-6 in these cases. Passing FP32 gamma directly to the fused AddRmsNorm
API is unsupported (input and gamma dtypes must match), so that is not a valid
fix here. This is an operator check, not full-model or learning evidence; see
`native_qwen35_norm_scratch.json` in the comparison directory.

An isolated 8K full-model follow-up applies that separate-add/native-Gemma path
to 81 normalization modules per TP rank, without changing parameters or the
installed packages. It tests both SSM dtypes on the same two recorded prefixes
as the ongoing native-normalization 32K probe, enabling a matched-prefix 2x2
comparison. The first 64-token FP32-cache probability RMS improves from
0.02316 to 0.01115, while the BF16-cache case does not improve (0.01521 to
0.01610). These short-prefix mixed results are not a reason to claim recovery.
Full 8K results are pending as
`qwen35_full_model_norm_cache_parity_20261027_054459`; no production
normalization change or new learning job has been applied at this stage.

### Completed 32K cache ablation and the next learning run (October 27)

Both full-model native-normalization cache cases finish successfully, with all
65,536 forced response IDs matched exactly across the two selected trajectories.
The only paired engine change is SSM cache dtype. Aggregate actor/decoder
log-probability RMS is 0.13100 for BF16 versus 0.02739 for FP32; the fraction of
chosen-token ratios outside [0.8, 1.2] falls from 6.866% to 0.0839%. In the final
4K response positions, BF16 RMS is 0.1637/0.1780 for the two sequences, versus
FP32 0.02851/0.02850. All sampled parameter hashes match between corresponding
TP ranks. See `verified_pair_analysis.json` and raw probability tensors in
`qwen35_full_model_cache_parity_20261027_052554`. These deliberately selected
trajectories establish a numerical effect, not average held-out accuracy or
successful learning.

The completed matched-prefix 2x2 comparison covers 16,384 response tokens.
With native normalization, BF16/FP32 RMS is 0.06357/0.02698; with the diagnostic
HF-aligned normalization it is 0.06267/0.02532. Thus normalization changes add
only 1.42%/6.14% RMS improvement at the respective cache dtypes, while the cache
change produces the larger improvement. The online comparison keeps native
normalization and isolates the FP32 SSM-cache repair. No normalization patch
has been added to the production worker. See `norm_cache_factorial_analysis.json`
in the comparison directory.

All probe actors exit and the existing cluster reports all 32 NPUs available,
with no named actors. A new 50-update asynchronous run is then submitted as
`qwen35_9b_async_32npu_fp32_32k_fp32ssm_20261027_060119`. Its exact command passes
Hydra preflight: FP32 SSM cache, BF16 convolution cache and forward, FP32 actor
masters, 0.9 allocator fractions, activation offload, unrestricted staleness,
32K responses and fixed 128-question validation. The initial CANN custom-op/API
paths are forwarded explicitly before device initialization. No Ray restart,
external extension, scoring change or new staleness limit is involved.
At submission it has zero completed updates; baseline validation, actual PPO
updates, sustained reward recovery and a matching synchronous run are still
required. Passive observation/archive tools now cover actor versions 0–49,
preserving behavior versions independently; diagnostic CPU copies affect timing.
The 36-hour stability run remains gated and has not started.

The live asynchronous run completes initial publication, all eight replica
pulls and its migration preflight. All eight serving engines, across both
nodes, advertise FP32 SSM/BF16 convolution cache through their actual metrics
endpoints (`rollout_runtime_cache_metrics.json`). This is runtime configuration
metadata, not direct tensor inspection. The passive observer is installed on
all 16 training ranks through actor version 49 before the first update.
Initial 128-question validation is now generating; no learning result is
available yet. A continuation process waits for all 50 logged updates, complete
matching initial/final held-out sets, successful job status and resource cleanup
before submitting the authorized synchronous control. Control preflight checks
source/data identity and permits only mode/output-path configuration changes;
an actual-config test accepts that change and rejects a changed learning rate.
Failure or a mismatch leaves a diagnostic record, without retries or any
automatic 36-hour stability launch.

The FP32-cache asynchronous run now completes its initial validation on the
same 128 held-out questions: 60 correct (46.875%), 59 unfinished, and no unparsed
references, unparsed answers or verifier timeouts. The old synchronous BF16-cache
initial result was 46 correct: the paired change has 20 wins and 6 losses
(exact McNemar p=0.00936). Against the old asynchronous initial 42 correct it has
27 wins and 9 losses (p=0.00393). The conservative paired intervals remain wide;
batching/runtime initialization can also differ between separate baseline runs.
Recorded configuration differences versus the old synchronous run are SSM
cache dtype and explicit CANN library/OPP path forwarding, in addition to the
asynchronous/synchronous mode and normalized output paths. These are initial
inference comparisons, not a clean free-generation single-factor experiment or
evidence of training improvement. The fixed-token cache ablation above isolates
the numerical cache effect. See `initial_cache_precision_validation_comparison.json`.
The training feeder is now generating the first batch; no PPO update has yet
completed, and the learning/alignment/stability requirements remain open.

The first real asynchronous update now succeeds. Its 64 exact selected
responses contain 480,226 valid tokens; all behavior/actor versions are zero.
Observed training versus behavior log-probability RMS is 0.02313, with 0.0546%
of ratios outside [0.8, 1.2]. The correct token alignment remains clearly best:
offset -1/+1 RMS is 1.081/1.109. Thus the cache repair also improves numerical
alignment in the live training path, beyond the isolated fixed-token probe.
See `logprob_alignment_analysis.json` and `selected_policy_drift_analysis.json`.

This batch has raw reward 1.0, but all four responses in every one of its 16
groups are correct. All GRPO advantages and the policy-gradient loss are zero;
the finite gradient norm 3.23e-5 comes from the KL regularizer (KL loss 0.000258,
coefficient 0.001). This is not evidence of learning improvement. The exact
selected-batch diversity audit finds four distinct token sequences in all 16
groups, so exact duplicate generation does not explain the zero advantages.
The early ready pool favors completed easy groups; later mixed-reward groups
and fixed validation are required to establish useful learning signal.
See `first_batch_sibling_diversity_audit.json`. Read-only optimizer checks on
ranks 0 and 8 confirm FSDP2, FP32 masters/AdamW moments, step 1 and the 0.9
allocator fraction. No convergence/alignment/stability gate passes from this
single update.

The next two updates complete with nonzero policy gradients. Updates 2/3 have
raw rewards 51/64 and 42/64, respectively, and 1/3 mixed-reward groups out of
16. Their exact forward probability RMS values are 0.02426/0.02468; only
0.0244%/0.0319% of nonzero-advantage token derivatives vanish through policy
clipping. This excludes the KL term and does not mean that the corresponding
parameter gradients vanish. Most responses still come from behavior version 0
(60/64 and 56/64); their declining batch rewards cannot establish policy
regression on a fixed question set. The fourth selected batch also has complete
forward observations, with RMS 0.02627, 3 mixed-reward groups and 47/64 responses
from version 0. It contains one group whose siblings use different behavior
versions; each individual trajectory remains single-version and retains its
own behavior logprobs. Do not assume that trajectory asynchrony guarantees
identical sibling versions. See `learning_signal_by_actor_version` in
`selected_policy_drift_analysis.json`; these remain selected-batch diagnostics,
not held-out learning evidence. Fixed validation after update 5 is pending.

The fifth batch's 64 observed forwards have RMS 0.02493 and four mixed-reward
groups. Its selected reward is 32/64 and five responses reach the 32K budget;
the batch includes behavior versions 0 through 3. Policy version 5 is then
published and checkpoint 5 contains all 16 nonempty model/optimizer/extra-state
shards, FSDP2/world-size-16 metadata and data state. This is structural file
verification, not a restore test. Only four full training metric rows are
logged while the fifth fixed validation remains pending; these observations
do not establish fixed-set improvement. The CPU preparation for all 512
held-out questions uses the production dataset and Qwen VL Continuous Token
builder. All current 128 validation prompt texts/references match its subset;
maximum prompt length is 1,196 tokens. No NPU is initialized and no inference
is performed by this preparation. The prompt artifact and manifest are in
`outputs/comparison_32npu_20261024_125225/full_heldout_evaluation`.

After publication 5, live serving metadata confirms that replicas progressively
drain their older requests and pull version 5. A read-only validation queue
probe at 08:02 UTC finds all 128 prompts submitted, 15 completed responses all
at version 5, and no mixed-version completed trajectory. At 08:03 UTC the
arrival trace contains 18 version-5 validation responses and five replicas
have completed the version-5 pull; no controller failure is recorded. These
partial completion counts are liveness/version evidence only, not partial
accuracy estimates. The full validation dump and logged update-5 metrics are
still pending.

Chosen-token actor/reference RMS for the first five observed batches is
0.02273, 0.02484, 0.02538, 0.02699 and 0.02518. In particular, the initial
actor/reference difference exists before task-policy learning. Resolved actor
masters are FP32 whereas reference model dtype is BF16; this is not an
isolated dtype ablation or a true KL estimate. No reference precision or KL
recipe change is made during this cache-only training comparison.

An additional CPU replay of the actual vanilla PPO derivatives localizes the
old mismatch to an important chosen token. In the old synchronous first batch,
12 of 18 positive-advantage `</think>` tokens have zero task-policy derivative
through clipping (66.7%), before any optimizer update and at behavior/actor
version 0. One such token has actor-minus-behavior logprob 0.8078, giving a
ratio about 2.24 despite identical policy versions. Across its first four
batches, 24/42 positive-advantage thinking-close tokens are masked (57.1%),
versus only 1.72% of all positive-advantage tokens. The current FP32-cache
asynchronous first five batches have 0/26 such thinking-close tokens masked.
Their chosen-token thinking-close RMS is 0.0403 versus 0.5719 in the old four
batches. These are task-loss derivatives with respect to chosen logprobs,
excluding KL; they do not imply zero parameter gradients from other tokens.
This supplies a concrete failure mechanism for promoting successful endings,
but the runs use different questions and it does not isolate the cache effect
at those endings or prove reward recovery. See
`termination_clipping_comparison.json` in the comparison directory.

The original two 32K cache-ablation traces contain no `</think>` token, so their
successful numerical result cannot directly prove restoration of ending-token
gradients. Four successful policy-0 traces from distinct prompt groups are now
preselected for a later matched-token cache comparison: two masked-ending cases
and two longer unmasked controls, with 18,323–31,137 response tokens. The
manifest is `termination_cache_probe_manifest.json`. The private probe and
analysis accept this explicit selection and compute actual PPO derivative
counterfactuals at endings. Their existing two-trajectory numerical replay
still reproduces the recorded RMS values; no new NPU probe has been submitted.

Stop-token semantics are also verified on CPU against the installed tokenizer,
HF generation fallback and actual vLLM SamplingParams update. The tokenizer's
primary EOS is `<|im_end|>` 248046; the model text config supplies additional
`<|endoftext|>` 248044. Both are stopping tokens in the installed path, despite
the absent local generation_config.json. An incorrect 248044-as-primary-EOS
assertion in the unexecuted standalone evaluator is corrected to derive the
primary token from tokenizer metadata and explicitly preserve both stops.
Production training settings are unchanged. See `stop_token_semantics_audit.json`
in the current asynchronous run; no NPU is initialized by this audit.

For a later fixed-weight inference counterfactual, the collapsed old synchronous
checkpoint 25 has been exported on CPU only, without initializing an NPU or
interrupting the current 32-NPU job. The 760 BF16 tensors have keys and shapes
matching the original model; the 15 omitted original keys are all MTP weights.
BF16 export follows the rollout publication dtype. Structural validation is
not parameter-value or inference validation. Artifacts are in
`outputs/old_sync25_cpu_export_20261027_072517`; no inference result from this
checkpoint is available yet.

The paired free-generation evaluator for that export is prepared and its CPU
plan passes against the exact 128-question subset. It fixes standalone greedy
decoding, EOS, native normalization and a 32K budget, and varies only SSM cache
dtype between the paired engines. Actual execution will verify cache tensor
dtypes and sampled parameter hashes, then use the unchanged mathematical
scorer and paired correctness statistics. It also compares its BF16 result to
the saved online checkpoint-25 result to expose serving/batching differences.
Its runtime path has not been tested on NPUs and no inference job is submitted;
the current 32-NPU learning comparison retains its resources. The plan and
source snapshot are under
`outputs/comparison_32npu_20261024_125225/old_sync25_cache_counterfactual_plan`.

The first complete post-repair fixed validation, at update 5, has 64/128 correct
versus its own initial 60/128: 50.0% versus 46.875%, 14 paired wins and 10 losses.
The conservative paired 95% change interval is [-9.25, +15.28] percentage
points and exact McNemar p=0.5413; this is an early positive point estimate,
not established learning improvement. Unfinished thinking falls 59 to 56,
format-valid responses rise 69 to 72, and all parse/verification error flags
remain zero. All 14 wins were previously unfinished responses; all 10 losses
become unfinished, with nine showing strongly repetitive tails. The number of
responses above the descriptive repeated-tail threshold falls 51 to 43.
See `validation5_diagnosis.json`, `validation5_outcome_transition_audit.json`
and `validation_repetition_analysis.json` in the asynchronous output.

The validation runtime is not identical to the initial rollout: no initial
validation response migrated, versus 53/128 at update 5. Eight of the ten
regressions migrated and two did not. Among the 60 initial successes, migrated
cases already had longer initial responses on average (17,754 versus 8,835
tokens). Migration is selected for long-running requests, so this association
does not prove that migration causes the regressions. It also prevents treating
the end-to-end +4 as an isolated SGD effect. The current 50-update asynchronous
recipe remains unchanged, followed by its matched synchronous control.
Checkpoint 5 is additionally exported on CPU into
`outputs/async5_cpu_export_20261027_084551/model` before retention can remove
it: all 760 BF16 tensor keys/shapes match the original, omitted keys are only
the 15 MTP tensors, and no NPU is initialized. This enables later fixed-serving
evaluation; no inference or tensor-value validation is claimed from that export.

Update 6 completes with reward 51/64, one mixed-reward group out of 16 and
gradient norm 0.05373. Its 64 real pre-update forwards have actor/behavior RMS
0.02420; 27/79,353 nonzero-advantage token derivatives vanish through policy
clipping (0.0340%). All three positive-advantage thinking-close tokens remain
active. The maximum behavior lag is five, but those four oldest responses have
zero advantages; this is not evidence about ending gradients at lag five.
The batch was generated by behavior versions 0/2/3/4, so its higher training
reward is not a fixed-question learning result for policy 6.

Chosen thinking-close/end-of-message derivative counts and RMS now refresh
continuously by selected actor version and behavior lag through the existing
read-only archived-forward analyzer. The new aggregates exactly match the
independent first-five-batch replay (26 positive tokens and zero masked tokens
for each ending marker, with the same RMS values). No model/optimizer/RNG or
training configuration is changed, and no additional NPU evaluation is run.
See `continuous_termination_diagnostic_preflight.json` and the versioned source
copy under `diagnostic_postprocessing_sources` in the asynchronous run.

Updates 7/8/9 complete with raw rewards 44/64, 44/64 and 33/64. Their observed
actor/behavior RMS values are 0.02570, 0.02459 and 0.02703. Update 7 has one
masked positive-advantage thinking-close token among eight; it belongs to a
rewarded response that reaches 32K without an EOS, generated at behavior 3 and
selected by actor 6. Its log ratio is 0.22169 at response position 27,709.
This differs from the old same-version, before-update clipping evidence: both
weight changes and numerical error are possible here. Updates 8/9 have zero
masked positive-advantage thinking-close tokens among eight/five, respectively.

The update-7 sample also exposes a separate training-objective weakness. Two
siblings of one question get mathematical score 1 despite reaching 32K without
terminating. One repeats a boxed answer 263 times after closing thinking, with
tail repetition fraction 0.6776; both receive GRPO advantage +0.866. The existing
scorer correctly recognizes the mathematical answer, but gives positive credit
to an unending response. Across the first nine exact selected batches, this
occurs in 2/576 responses, so it cannot by itself establish the cause of the
overall regression. See `update7_rewarded_budget_hit_response_audit.json`.

A candidate patch is prepared, **not applied**, under
`outputs/comparison_32npu_20261024_125225/termination_reward_candidate`. It passes
the observed terminal-token status through V1 TQ reward metadata and optionally
sets training score to mathematical accuracy times that status. Mathematical
`acc` remains separate and unchanged; the reasoning preset would enable the
gate in a subsequent controlled run. CPU replay executes the actual candidate
postprocess, base scoring method, NaiveRewardManager and TQ serialization with
transport/teacher operations stubbed. The two unending rewards become zero,
their mathematical accuracies remain one, and a real completed correct control
retains reward one. Both valid EOS IDs and EOS exactly at the budget are
accepted; missing/malformed metadata is rejected when enabled. Default callback
and non-MATH behavior are preserved. These checks do not establish NPU or
learning behavior of the candidate.

A same-output counterfactual reproduces all recorded GRPO advantages over the
first nine batches before changing only the termination factor: mathematical
correctness stays 386/576, training reward becomes 384/576, mixed groups become
23 instead of 24, and the erroneous positive advantages on 65,536 unending
response tokens disappear. This is not a new-policy reward result. Python/Ruff,
Bash syntax, patch applicability and actual Hydra flag preflight pass. The
current online cache-only recipe stays unchanged; the candidate has not been
applied to either member of the planned asynchronous/synchronous comparison.

The complete update-10 fixed validation has 65/128 correct (50.78125%), versus
60/128 initially and 64/128 at update 5. Relative to initialization there are
15 paired wins and 10 losses (exact McNemar p=0.42436); from update 5 there are
16 wins and 15 losses (p=1.0). These small upward point estimates and substantial
question-level churn do not establish sustained learning. Unfinished thinking
is 55/128, format-valid responses are 73/128, and parse/verification error flags
remain zero. Strongly repetitive tails number 43, unchanged from update 5.
Serving/migration differences still prevent attributing the complete change
to SGD alone. See `validation10_diagnosis.json` and
`validation10_outcome_transition_audit.json` in the asynchronous output.

Update 10 has training reward 48/64 and actor/behavior RMS 0.02654. None of its
four positive-advantage thinking-close tokens has a zero task derivative from
policy clipping. The complete checkpoint 10, including all 16 optimizer and
extra-state shards, is preserved by verified hard links under
`outputs/async_checkpoint10_preserved_20261027_102701` for later diagnosis;
this verifies file preservation, not successful model restoration.

The archived pre-SGD validations also differ across the repair: old synchronous
initialization has 46/128 correct, versus 60/128 for the current asynchronous
initialization, with 20 paired wins and 6 losses (exact McNemar p=0.00936).
Both have zero migrated validation responses and zero parse/verification
error flags; unfinished thinking changes 76 to 59. The question/reference
sets and both data manifests match. Resolved configuration differences also
include early custom-operator loading and mode, so this is an association with
the repair bundle, not an isolated cache effect or SGD improvement. The full
configuration/source comparison and question-level transitions are saved in
`outputs/comparison_32npu_20261024_125225/initial_cache_repair_observation.json`.

CPU comparison of all 760 exported checkpoint-5 tensors (9,409,813,744 elements)
against the original weights cast to BF16 finds 246,667,569 changed elements
(2.6214%), with finite values throughout. Vision weights are unchanged; the
language-model attention/MLP weights and embeddings/output head contain actual
saved changes. This establishes that updates survive the publication dtype in
the saved export, not that those values were verified on live rollout NPUs or
that learning has converged. The comparison separately accounts for rounding
2,620 elements in original FP32 gated-normalization weights; it does not count
that initial cast as learning. See `published_weight_value_comparison.json`
under `outputs/async5_cpu_export_20261027_084551`.

A further CPU audit compares all recorded TP2 `A_log` samples in both earlier
cache-probe cases to original and BF16-rounded values. All 96 hashes match
both: these original FP32 `A_log` values are already on the BF16 grid. Thus
their file dtype alone is not evidence of an actor/probe value mismatch. This
is a sampled-parameter audit, not full-model or accuracy verification; see
`parameter_precision_scope_audit.json` in the full-model cache-probe output.

Updates 11/12 complete with training rewards 40/64 and 39/64 and actor/behavior
RMS 0.02581/0.02664. At update 11, one of 12 positive thinking-close derivatives
is clipped: a completed correct response from behavior version 7 used by
actor 10, with ratio 1.23065. Its chosen end-of-message derivative remains
active. This cannot be identified as same-weight numerical clipping because
the policy versions differ. Update 12 has no clipped positive ending markers
(three of each kind); its rolling 128-group reward mean is 0.64648. Neither
batch reward is a fixed-question learning result. See
`update11_positive_ending_case_audit.json` and the continuous drift analysis.

Update 13 completes with reward 51/64, rolling 128-group reward 0.68359 and
actor/behavior RMS 0.02505. Only one of 16 groups has mixed rewards, so 114,578
of 1,020,601 response tokens have nonzero GRPO advantages. Its three positive
thinking-close and end-of-message task derivatives remain active. A high raw
batch reward is not evidence of a large effective learning batch or improved
fixed-question accuracy. At 10:58 UTC all eight live engines were serving
111 requests at versions 11--13; the oldest-version engines were still making
long responses. No staleness cap or Ray restart is introduced.

The post-training common-serving evaluation is now prepared, with a live
continuation waiting on the exact current asynchronous submission. It requires
both this 50-update job and its matched synchronous 50-update control to
succeed, verifies complete logged updates plus configuration/source/dataset
parity, exports their final checkpoints on CPU, and waits for all 32 NPUs and
the training queue to be released. Only then may it submit a separate 12-NPU
evaluation job: three fixed models, two TP2 replicas per model on the same
16-NPU head node, FP32 SSM cache, greedy decoding, 32K budget, explicit valid
EOS IDs, and no Laminar migration or weight updates. Every model receives
the identical disjoint 256+256 partition of the full 512-question set.
It reports all 512 questions, the monitored 128, and the additional 384
separately, including paired confidence intervals. It never launches stability
training or retries a failed training/evaluation submission automatically.

The initial comparison model is exported on CPU with the same BF16 publication
cast and 15 MTP omissions as the actor exports, under
`outputs/original_publication_bf16_export_20261027_1050/model`. All 760 exported
keys, shapes, dtypes and 128-element prefix samples are verified; full tensor
value replay and NPU inference are not claimed. The actual evaluator preflight
passes using that baseline and existing checkpoint-5/old-checkpoint-25 exports
as schema proxies only; those proxies are not a matched final comparison.
CPU summary checks reject missing/duplicate questions, prompt/reference
mismatches, nonbinary correctness and over-budget responses. Their temporary
384-question score fixtures are explicitly synthetic, deleted after the check,
and provide no held-out accuracy evidence. Source snapshots and the preparation
manifest are under
`outputs/comparison_32npu_20261024_125225/common_evaluation_sources`.
The NPU evaluation path remains unexecuted while the current learning job uses
the cluster; monitor `common_evaluation_continuation_status.json` in that job.

A read-only audit now covers fully observed generated training cohorts for
behavior versions 0--10. All issued requests in each reported cohort ended,
were classified as train/validation, and had complete archived training scores;
all eight engines had moved past those versions at the saved scheduler
snapshot. Probes and validation are excluded. Later incomplete cohorts receive
no partial reward estimate. Event-log prefix byte counts and SHA256 hashes
make the exact observation reproducible despite ongoing log appends.

Within the unchanged behavior-version-0 cohort, all 264 training responses have
mean reward 175/264 = 0.66288. The 64 selected at optimizer update 1 all score
one and average 7,504 tokens. The 47 selected at update 4 score 13/47 = 0.27660
and average 28,346 tokens. All these responses were generated with version 0,
so this within-version drop cannot be read as degradation from SGD between
those optimizer updates. It demonstrates selection/completion-order and batch
composition effects in the training reward trace. It does not explain away
the old fixed-question validation collapse, and differences between these
closed cohorts are not learning evidence because questions and version
assignment differ. See `closed_generation_cohort_analysis.json` and the
PNG/SVG plot `unchanged_policy0_selection_bias` in the current async output.

The complete version-10 generated training cohort contains another rewarded
32K response without an EOS (final token 28). This response was not among
samples selected by the first 14 fully logged updates. It reinforces the
optional termination-gate case without establishing it as the overall cause;
the online scoring recipe remains unchanged. See
`version10_rewarded_budget_hit_termination_audit.json`.

Policy 15 is published and checkpoint 15 has 16 nonempty model, optimizer and
extra-state shards of each kind. This confirms checkpoint presence, not tensor
values or successful restoration. Its complete 64-response pre-update forward
observation has actor/behavior RMS 0.02725, with zero clipped task derivatives
among three positive thinking-close markers and three positive end markers.
The selected input scores average 39/64 = 0.609375. Fixed validation 15 is
running; its 128 prompts are present, but no full accuracy result exists yet.
Training metrics for step 15 are logged after that validation completes, so
the fully logged optimizer-update count can remain 14 while policy 15 is live.

The future common-evaluation preflight exposed a preparation bug in the bare
initial BF16 export: processor assets were missing. Actual local
`AutoProcessor.from_pretrained`, even with the original tokenizer supplied,
failed with an image-processor configuration error. The installed vLLM path
also initializes its temporary processor from the model directory before
injecting the separately configured tokenizer. Seven original tokenizer,
template and image/video-processor files are now copied and hash-checked in
the initial export; its weight files are unchanged. The initial-export helper
includes those assets for future exports.

Common evaluation now requires a separate CPU-only processor preflight before
creating any NPU actors. Initial, checkpoint-5 and old-checkpoint-25 exports
all load as Qwen3VLProcessor/Qwen2Tokenizer and reproduce every prompt token,
decoded prompt, data source and reference for the exact 512-question set using
the production continuous-token builder. This validates loading/input assets,
not model accuracy; the two existing checkpoints remain schema proxies only.
The pinned post-training evaluation watcher is refreshed to include this
helper, baseline metadata/assets and comparison statistics. The training job,
its code/configuration and the existing Ray cluster are unchanged. See
`common_serving_preflight/baseline_processor_before_repair.json` and
`common_serving_preflight/processor_assets_preflight.json` under the comparison
directory, plus its updated `common_evaluation_sources` manifest.

The complete update-15 validation has **73/128 correct (57.03125%)**, giving the
fixed-question sequence 60/64/65/73 at updates 0/5/10/15. Relative to its own
initialization there are 20 wins and 7 losses, a +10.15625 percentage-point
change. Exact McNemar p=0.01916 is nominal and unadjusted for repeated interim
looks; the deliberately conservative paired 95% change interval is
[-2.66, +22.27] percentage points. Against update 10 there are 18 wins and
10 losses (nominal p=0.18493). These are stronger positive end-to-end point
estimates, not automatic sustained-learning, alignment or stability acceptance.

Unfinished thinking falls 59 to 43 and format-valid responses rise 69 to 85;
all parse/verification error flags remain zero. Of the 20 new successes,
19 were initially unfinished; all 7 lost initial successes become unfinished.
Strongly repetitive tails fall 51 to 32, with 6 of the 7 losses still strongly
repetitive. Validation migrations change 0/53/45/30 across the four checkpoints.
Long-running requests are selected for migration, so neither the improvement
nor individual regressions establish a causal SGD/migration effect alone.
All 128 update-15 trajectories use version 15, with the exact same held-out
question/reference set as before. See `validation15_diagnosis.json`,
`validation15_outcome_transition_audit.json` and the refreshed repetition report.

The complete checkpoint 15 is additionally preserved by hard links under
`outputs/async_checkpoint15_preserved_20261027_120947`: all 55 files match
source device, inode and size, including all model/optimizer/extra-state shards.
This is a diagnostic preservation, not successful restoration or final model
selection. After validation the actual trainer logs update 15 with reward
39/64, rolling 128-group reward 0.66211, gradient norm 0.08527 and maximum
behavior lag 8. The same recipe continues toward 50 updates, then its exact
synchronous control and common 512-question evaluation. The conditional
36-hour stability run remains unstarted.

Update 16 completes with reward 49/64, rolling reward 0.671875 and gradient
norm 0.08737. Its 64 real selected forwards have actor/behavior RMS 0.02513;
all nine positive thinking-close and nine positive end-of-message task
derivatives remain active. Four of 16 groups have mixed rewards; maximum
behavior lag is five without introducing an age cap.

The previously archived rewarded nonterminated version-10 response is selected
in update 16. Its three siblings terminate normally and are also correct;
all four original rewards are one, so their actual GRPO advantages are all
zero. Thus this instance does not positively reinforce the unending response,
but the objective also supplies no distinction between it and the completed
correct answers. See `update16_rewarded_nonterminated_group_audit.json`.

The frozen-output termination counterfactual now additionally covers all first
16 updates: 1,024 responses, 256 groups, with every recorded original GRPO
advantage reproduced through the actual implementation. Mathematical labels
remain 697/1,024; gated training rewards would total 694/1,024. Three rewarded
nonterminated responses occur, two receiving positive advantages in update 7
and one having zero advantage in update 16. The optional gate removes the
first erroneous mixed group and creates a useful mixed group for the latter,
leaving the total mixed-group count at 44. This is an objective replay on fixed
outputs, not new-policy or NPU learning evidence. The original nine-update
artifact is retained separately; see `selected_output_counterfactual_first16.json`
under the termination candidate. The candidate remains unapplied to preserve
the current asynchronous/synchronous comparison.

The replicated common-serving preflight passes with the exact same 512 prompt
artifact and unchanged decoding settings. CPU merge checks require every
assigned prompt exactly once, matching references/prompt hashes, consistent
generation-versus-grading records and matching sampled parameter hashes
between model replicas. Missing/duplicate/misassigned responses, incomplete
replicas, worker-rank mismatches and token-length mismatches are rejected.
These checks use explicitly synthetic temporary generation/metadata fixtures,
which are deleted; they provide no accuracy or NPU-runtime acceptance. Actual
exported processors also reproduce all 512 input token sequences. The waiting
post-training evaluator is refreshed with the tested source pins and explicit
`--replicas-per-model 2`; the learning jobs and shared Ray cluster are unchanged.
See `common_serving_replicated_preflight` and the updated evaluation-source
manifest in the comparison directory. The replicated NPU path is unexecuted
until both paired training jobs finish successfully and release their resources.

Policy 20 is now published and checkpoint 20 contains 16 nonempty model,
optimizer and extra-state shards of each kind. The exact 64-response input and
pre-update actor-19 forward observation has raw mathematical reward 50/64,
actor/behavior RMS 0.02502 and four mixed-reward groups. All ten positive
thinking-close and ten positive end-of-message task derivatives are active.
Across the first twenty complete forward batches, 2/118 positive thinking-close
markers and 0/116 positive end markers are clipped. The two thinking-close cases
are the previously audited updates 7 and 11 with different actor/behavior
versions; these totals are not a same-question comparison to the old run.
Fixed validation 20 is running, with no complete accuracy result yet. As at
earlier validation boundaries, step-20 training metrics are logged afterward,
so the last fully logged update remains 19. See
`update20_validation_pending_audit.json`; checkpoint presence is not restoration
proof, and pre-update sample reward is not a fixed-set learning result.

The frozen-output termination-reward replay now covers all twenty complete
batches, reproducing recorded GRPO advantages for all 1,280 responses. There
are still three rewarded nonterminated responses, with only the original two
receiving positive advantages; the four additional batches add none. Original
math labels total 866/1,280 versus 863 with the optional termination multiplier,
and mixed groups remain 58/320 under either objective. The first sixteen batch
summaries and all changed-advantage records are exactly preserved. This
quantifies the observed reward defect without proving its causal contribution
to collapse or excluding problems outside these samples. The candidate is
still unapplied. See `selected_output_counterfactual_first20.json` and
`first20_counterfactual_provenance.json` in the termination candidate directory.

The complete update-20 validation has **70/128 correct (54.6875%)**, giving the
fixed-question sequence 60/64/65/73/70 at updates 0/5/10/15/20. Against
initialization there are 18 wins and 8 losses, +7.8125 percentage points;
nominal exact McNemar p=0.07552 and the conservative paired 95% change interval
is [-4.90, +19.98] percentage points. Against update 15 there are 12 wins and
15 losses, -2.34375 percentage points (nominal p=0.70111). Neither a nonsignificant
comparison nor overlapping intervals establish equivalence or prove that the
decrease is sampling noise. Sustained improvement remains unverified.

All 128 update-20 outputs join by response UID to their actual selected,
arrival, request-begin and request-end events at policy version 20, with exact
response-length agreement. The question/reference set is unchanged and all
parse/verification error flags remain zero. Unfinished outputs rise 43 to 48
from update 15, versus 59 initially; strongly repetitive tails rise 32 to 45,
versus 51 initially. Of 15 newly lost successes, 14 become unfinished and one
becomes finished but incorrect; all 12 new wins were previously unfinished.
Validation migrations rise 30 to 45, so this does not isolate SGD or migration
as the cause. Budget hits total 48, with no correct budget-hit outputs. See
`validation20_diagnosis.json`, `validation20_outcome_transition_audit.json` and
`validation20_serving_version_audit.json` in the active run directory.

The actual step-20 training metrics now log raw reward 50/64, rolling 128-group
reward 0.68945, gradient norm 0.11069 and maximum behavior lag seven. The SDK
confirms RUNNING after validation and training continues toward the paired
50-update comparison. Higher selected-batch reward does not override the
fixed-set ending/repetition regression from update 15. Neither reward scoring
nor source/configuration is changed during this pair, and the common-serving
512-question evaluation remains necessary before accepting accuracy alignment.

A follow-up scorer audit examines the update-15 to update-20 regressions without
changing any official reward. The sole completed but newly incorrect answer is
a real arithmetic error: the pyramid has base area 9*sqrt(3) and maximum height
3*sqrt(3), hence volume 27; the update-20 output instead simplifies the last
product to 81. Independent exact rational recomputation agrees with gold 27.
Of the fourteen newly unfinished failures, relaxing the thinking-close check
finds one correct last box, twelve missing final answers and one overlong
answer. Across all 128 questions, the same relaxed diagnostic gives 76 at
update 15 versus 73 at update 20, retaining the three-question decrease; official
correct counts remain 73 and 70. This is not exhaustive mathematical-content
grading or a replacement acceptance metric. See
`validation20_finished_incorrect_case_audit.json`,
`validation20_regression_unfinished_box_audit.json` and
`validation15_20_relaxed_scoring_comparison.json`.

Update 21 completes with reward 34/64, rolling reward 0.65625, gradient norm
0.05304 and maximum behavior lag seven. Its full 64-response pre-update actor-20
forward observation has actor/behavior RMS 0.02619. Both positive thinking-close
and both positive end-marker task derivatives remain active. Only one of its
sixteen groups has mixed rewards (eight all-correct and seven all-wrong), so
the task-learning signal is concentrated in one question despite the 64-response
batch. This does not establish that low group diversity caused the validation
regression. The matched comparison continues without a scoring, selection or
staleness-control change.

Updates 22/23 complete with rewards 46/64 and 41/64, rolling rewards
0.65820/0.66211, gradient norms 0.09245/0.08728 and maximum behavior lag ten.
Their complete forward batches have actor/behavior RMS 0.02599/0.02577 and
three mixed-reward groups each. One of six and one of nine positive
thinking-close task derivatives are clipped respectively; all six/nine positive
end-of-message derivatives remain active. The update-22 case is a completed
correct 31,851-token response from behavior 16 used by actor 21 (lag five):
its closing-token logprob difference is +0.29965, ratio 1.34938. The update-23
case is a completed correct 24,925-token response from behavior 19 used by
actor 22 (lag three), difference +0.20808, ratio 1.23131. Unlike the old
before-update evidence, these compare different weights and do not establish
false numerical clipping. See `update22_positive_ending_case_audit.json` and
`update23_positive_ending_case_audit.json`.

To retain a usable same-version replay reference, exact publication-21 BF16
weights are exported through the existing CPU relay to
`outputs/async21_live_publication_export_20261027_1418/model`. All 760 tensor
payload SHA256 hashes match the immutable publication manifest before writing
and after a complete safetensors reread (9,409,813,744 elements,
18,819,627,488 weight bytes). The short read lease is released before the
verification pass; catalog status then shows zero leases/capacity waiters and
no failure. This is a published decoder snapshot, not an FP32 actor/optimizer
checkpoint or live-NPU tensor-parity proof. NPU initialization is explicitly
forbidden and remains false. Actual CPU processor/tokenizer loading also
reproduces every one of the 512 held-out prompt token sequences and references.
The private export helper passes Ruff's error checks; production source and
the pending paired-evaluation pins are unchanged.

The update-22 selected input, observed actor-21 forward and a compact replay
packet are copied and hash-checked under
`outputs/comparison_32npu_20261024_125225/update22_ending_drift_probe`.
All 98 prompt IDs and 31,851 response IDs match the actual observed input.
Its original behavior-16 generation includes one migration. A later fixed-token
decoder-21 replay can measure the current train/serve numerical term; its
remaining difference from original behavior 16 still includes old-path and
migration/batching effects as well as policy change. These must not all be
called isolated SGD drift. This NPU replay is **unexecuted**; the current
32-NPU learning job and the authorized async-then-sync order are preserved.

Update 24 completes with reward 48/64, rolling reward 0.66016 and gradient norm
0.00015075. Its actual 64-response pre-update actor-23 observation has twelve
all-correct and four all-wrong groups, zero mixed groups and zero nonzero
advantages across all 811,915 response tokens. The task-policy loss and its
chosen-logprob derivatives are zero despite the high raw reward; the KL term
and optimizer momentum/state can still produce parameter changes. Actor/behavior
RMS remains 0.02654. This is direct evidence that selected-batch reward alone
does not measure the amount of task-learning signal, not proof of the cause of
held-out regression.

Policy 25 is published and its checkpoint contains 16 nonempty model,
optimizer and extra-state shards of each kind. The complete pre-update actor-24
forward batch has input reward 55/64, thirteen all-correct groups, one all-wrong
group and two mixed groups. Actor/behavior RMS is 0.02591. One of three positive
thinking-close task derivatives is clipped, while all three positive end markers
remain active. The audited completed correct response comes from behavior 16,
lag eight, and has closing-token difference +0.18863, ratio 1.20760, just beyond
the PPO upper clip boundary. Different versions and numerical path effects
remain unresolved; this does not establish same-weight false clipping.

The same 128-question validation at policy 25 is running, with no complete
accuracy yet. Step-25 training metrics are recorded after validation, so the
last fully logged update can remain 24 while policy 25 is live. See
`update25_validation_pending_audit.json`, the immutable
`update25_selected_forward_snapshot.json` and
`update25_positive_ending_case_audit.json`. Nonempty checkpoint shards do not
prove a successful restore, and the pre-update input score is not policy-25
held-out accuracy. Source, reward and unrestricted-staleness settings remain
frozen for the asynchronous/synchronous pair.

The complete update-25 validation has **75/128 correct (58.59375%)**, giving
60/64/65/73/70/75 at updates 0/5/10/15/20/25. Relative to initialization there
are 20 wins and 5 losses, +11.71875 percentage points. Exact McNemar p=0.00408
is nominal, unadjusted for repeated interim looks and recipe selection; the
conservative paired 95% change interval is [-0.53, +23.13] percentage points.
Against update 20 there are 17 wins and 12 losses (+3.90625 points, nominal
p=0.45826); against update 15 there are 10 wins and 8 losses (+1.5625 points,
nominal p=0.81453). This is the highest monitored point so far, not a completed
primary learning, alignment or stability result.

All 20 new successes were initially unfinished. All five lost initial successes
are now unfinished and have strongly repetitive tails. Unfinished responses
fall 59 initially / 48 at update 20 to 41 at update 25; strongly repetitive
tails fall 51 / 45 to 31. Against update 20, sixteen unfinished and one finished
incorrect answer become correct, while twelve successes become unfinished.
All parse/verification error flags remain zero. The same 128 question/reference
identities and all actual request-begin/end versions are verified at policy 25,
with exact response-length agreement. Forty-one responses migrate, and 41 hit
the response budget, none correct. Migration/serving effects remain confounded
with learning here. See `validation25_diagnosis.json`,
`validation25_outcome_transition_audit.json` and
`validation25_serving_version_audit.json`.

Checkpoint 25 is preserved under
`outputs/async_checkpoint25_preserved_20261027_153742`: all 55 files retain the
source device, inode and size, including all model/optimizer/extra-state shards,
RNG/data and HF metadata. This is diagnostic retention of the new monitored
peak, not a successful restore or final best-checkpoint selection. The declared
final endpoints remain update 50 and common-serving 512-question evaluation.
After validation, step 25 logs reward 55/64, rolling 128-group reward 0.69727,
gradient norm 0.09670 and maximum behavior lag eight. The SDK confirms RUNNING
and training continues without a source, reward or configuration change.

Updates 26--29 complete with raw rewards 36/64, 48/64, 43/64 and 54/64; the
latest rolling 128-group mean is 0.72461. Gradient norms are finite
(0.10130/0.13570/0.13469/0.09590), and complete actual-forward actor/behavior
RMS remains 0.02652--0.02762. Updates 26/28/29 have no clipped positive
thinking-close markers (8/7/6 positive markers respectively). Update 27 clips
one of twelve: a completed correct response from behavior 21 used by actor 26,
lag five, with closing-token difference +0.23450 and ratio 1.26427. All positive
end-of-message task derivatives remain active across these four batches.
This different-version case is not identified as false numerical clipping.
See `update27_positive_ending_case_audit.json`; the last full held-out result
still belongs to update 25, not these later weights.

The step-27 staleness audit joins all 64 selected inputs to actual generation,
reward-ready and selection events. Its oldest group starts at behavior 16
while the latest published version is also 16. Three versions accrue during
generation/settling siblings and seven after the whole group is ready, giving
actor 26 minus behavior 16 = ten. Generation and settling siblings take about
30.47 minutes, followed by 175.07 minutes from group readiness to selection.
Of that waiting, 10.57 minutes precede complete observation by the sampler and
164.50 follow it. Validation-20 and validation-25 generation/collection windows
overlap 104.04 minutes of the wait; 71.03 minutes lie outside those measured
windows. This overlap is descriptive, not a counterfactual delay estimate.
The ten-version group has no clipped positive ending marker; the actual
step-27 clipped case has lag five.

The frozen `mixed_ready` rule guarantees only one oldest eligible observed
group, then uniformly selects fifteen other groups when the pool exceeds the
batch size. The oldest example's observed-complete ranks at updates 21--27
are 22/16/8/5/3/2/1, and it is selected at 27. The earliest observed-complete
group is selected at every one of these updates, consistent with that rule.
Historical prompt terminal-eligibility snapshots are absent, so reconstructed
observed pools are supersets rather than exact random-draw replays. No ordering
failure is established and neither sampling nor unrestricted staleness is
changed during the pair. See `update27_staleness_timing_audit.json`,
`update27_oldest_group_selection_history.json` and
`update27_staleness_validation_overlap.json`.

Policy 30 is published and its checkpoint contains sixteen nonempty model,
optimizer and extra-state shards of each kind. All 64 pre-update actor-29
forwards are observed, with actor/behavior RMS 0.02600 and input reward 55/64.
Twelve groups are all-correct, one all-wrong and three mixed. All seven positive
thinking-close and seven positive end-marker task derivatives remain active.
The same 128-question validation at policy 30 is now running, with no complete
accuracy yet. As at earlier boundaries, full step-30 training metrics follow
validation, so the last fully logged update remains 29. See
`update30_validation_pending_audit.json` and its immutable
`update30_selected_forward_snapshot.json`. Checkpoint presence is not restoration
proof; input reward is not policy-30 held-out accuracy. The source/configuration
freeze, async-then-sync comparison and common 512-question evaluation remain
in force.

The preserved publication-21 ending case now has an executable fixed-token
replay entry. CPU planning validates its 31,851 response tokens and exact
teacher/behavior log probabilities against the independently frozen case; the
legacy input path also reproduces all 65,536 tokens and log probabilities used
by the already-completed original-policy NPU comparison. Reanalysis of those
actual NPU outputs preserves every aggregate and position-bin metric exactly.
The analyzer records versions/model provenance and separates teacher-minus-replay
from replay-minus-original-behavior, verifying their additive identity. The
second term still includes old generation, migration and batching effects and
must not be called isolated SGD drift. An execution-path check against the live
RUNNING job rejects replay before Ray initialization or any NPU actor creation.
All 506 frozen non-Markdown production sources and 18 common-evaluation pins
remain unchanged. These are CPU preparation/compatibility checks: the new
publication-21 and four original-policy ending-case NPU replays remain
unexecuted and do not establish restoration of ending gradients. See
`update22_ending_drift_probe/replay_preparation_report.json` in the comparison
directory. The authorized paired training and common evaluation retain priority.

The full policy-30 validation now finishes on the unchanged 128 questions:
74 correct (57.8125%), versus 60 initially, 70 at update 20 and 75 at update 25.
The initial comparison has 21 wins/seven losses; its nominal exact McNemar
p=0.01254 is unadjusted for interim looks and recipe selection, and the
conservative 95% change interval remains [-2.04, +23.16] percentage points.
Update 25 to 30 has eleven wins/twelve losses (p=1.0), so the net one-question
decline is not evidence of a new broad collapse or proof of sustained learning.
All twelve regressions are unfinished thinking, eleven with strongly repetitive
tails. Overall unfinished responses are 43 (initially 59, update 25: 41), with
35 strongly repetitive tails (initially 51, update 25: 31). All parsing and
verification-timeout flags are zero. Exact UID joins of all 128 outputs to
selected/arrival/request-begin/request-end events verify version 30 and token
lengths. There are 39 migrated responses and 44 budget hits, none correct;
one budget hit closes thinking but exhausts the budget before a final boxed
answer. Its correct internal candidate does not satisfy the fixed final-answer
objective, and its official zero score is unchanged. Migration and serving
remain confounders, not an isolated SGD/cache estimate.

Step 30 then logs training reward 55/64, rolling 128-group reward 0.74219,
gradient norm 0.09904 and maximum lag eight. The new CPU audit reproduces the
previously independently checked update-25 statistics, all 128 serving joins
and outcome transitions exactly; its update-30 statistics/repetition match the
separate continuous watchers. Event-log prefix hashes, validation hashes,
506 frozen non-Markdown sources and 18 common-evaluation pins are reverified.
See `validation30_diagnosis.json`, `validation30_serving_version_audit.json`,
`validation30_outcome_transition_audit.json`, `validation30_regression_ending_audit.json`
and `validation30_closed_thinking_budget_case_audit.json` in the active run.
The job remains RUNNING. Async50, matched Sync50, the common512 evaluation and
the conditional 36-hour run remain incomplete; no learning/alignment/stability
gate is accepted at this interim boundary.

Update 31 completes and policy 31 is published. Raw training reward is 38/64,
rolling 128-group reward 0.73633, gradient norm 0.10183 and maximum lag ten.
All 64 actual pre-update actor-30 forwards are matched, with actor/behavior
RMS 0.02815. Eight groups are all-correct, five all-wrong and three mixed,
leaving 347,549 nonzero-advantage tokens. All six positive thinking-close and
six positive end-marker task derivatives remain active. The oldest behavior-20
group is all-correct with zero advantages. These are different training
questions and behavior versions, so the batch reward decline from update 30
does not establish held-out regression or isolate staleness effects. See
`update31_training_diagnosis.json` and the immutable
`update31_selected_forward_snapshot.json`; the last fixed validation remains
policy 30 at 74/128, and all final comparison/stability requirements remain open.

Updates 32/33 complete and both policies are published. Their training rewards
are 31/64 and 44/64, rolling 128-group means 0.70313/0.68164, gradient norms
0.08003/0.07018 and maximum lags eleven/seven. Complete pre-update actor-31/32
forward RMS values are 0.02716/0.02645. Update 32 has seven all-correct, six
all-wrong and three mixed groups; update 33 has ten/four/two. Update 33 leaves
all four positive thinking-close and four positive end-marker derivatives
active. Update 32 instead clips two of three positive thinking-close markers,
with all three positive end-marker derivatives active. The clipped correct,
normally ended responses have actor 31 versus behavior 25/28 (lags six/three),
thinking-close deltas 0.21339/0.22399 and ratios 1.23787/1.25105. Their original
generation events verify behavior versions and lengths exactly; the first
migrates once and the second never migrates. Thus migration is not necessary
for this observed clipping, but different-version policy and numerical effects
remain combined. The eleven-version-old group is all-wrong with zero
advantages. No staleness control or recipe change is applied.

An exact live publication-31 CPU export preserves all 760 BF16 tensors
(9,409,813,744 elements), checking their full raw-byte hashes both against the
immutable publication manifest and after safetensors reload. Its temporary
lease is released and the catalog reports zero leases/capacity waiters and no
failure. Both real update-32 cases are independently frozen, totaling 54,535
response tokens. The executable replay input path matches their prompt/token
IDs and teacher/behavior probabilities exactly, and the exported processor
reproduces all 512 common-evaluation prompts and references on CPU. All 506
non-Markdown production sources and 18 evaluation pins remain unchanged.
The attempted behavior-28 export fails before acquiring a lease because version
28 has already been evicted; no exact version-28 weights are preserved. A
subsequent live catalog check confirms zero leaked leases/capacity waiters and
the main job still RUNNING. Consequently these cases still cannot isolate SGD
drift from the original behavior path, even though one never migrated.
Publication-31 same-weight numerical replay remains possible but **NPU
unexecuted**, preserving the primary paired training/common-evaluation order.
See `update32_training_diagnosis.json`, `update33_training_diagnosis.json`,
`update32_positive_ending_case_audit.json`, `update32_clipped_case_generation_audit.json`,
`publication28_export_unavailable_audit.json`, and the comparison directory's
`update32_ending_drift_probe/replay_preparation_report.json`.

Update 34 completes and policy 34 is published with training reward 50/64,
rolling reward 0.70898, gradient norm 0.10639 and maximum lag seven. All 64
actual pre-update actor-33 forwards match, with RMS 0.02595; twelve groups are
all-correct, two all-wrong and two mixed, yielding 196,063 nonzero-advantage
tokens. Both positive thinking-close and both positive end-marker derivatives
remain active. See `update34_training_diagnosis.json` and the immutable
`update34_selected_forward_snapshot.json`. The job remains RUNNING, the last
complete fixed validation remains step 30 at 74/128, and the new read-only
`validation35_live_wait.jsonl` observation is tracking the next full validation
without computing completion-biased partial accuracy. No training, scoring or
sampling configuration change is made.

Policy 35 is published, checkpoint 35 has sixteen nonempty model, optimizer
and extra-state shards each, and the fixed 128-question validation is running.
The last complete training log remains update 34 because update-35 metrics
follow validation. All 64 pre-update actor-34 forwards are matched: input
reward 46/64, RMS 0.02669, ten all-correct/three all-wrong/three mixed groups
and 343,848 nonzero-advantage tokens. One of six positive thinking-close
derivatives is clipped; all five positive end-marker derivatives remain active.
The differing marker counts correspond to a real correct-labelled 32K response
without either valid terminal token. Its behavior is 28 versus actor 34, with
one migration, actual last token 279, advantage +1.499997 and thinking-close
delta 0.45881 (ratio 1.58219). Original begin/end events verify behavior 28 and
32,768 tokens. This is a termination-reward boundary defect and a
different-version clipping observation, not an isolated numerical-error proof.
See `update35_validation_pending_audit.json`,
`update35_selected_forward_snapshot.json`, `update35_positive_ending_case_audit.json`,
`update35_rewarded_budget_hit_response_audit.json` and
`update35_rewarded_case_generation_audit.json`.

The guarded CPU termination-reward counterfactual now reproduces actual GRPO
advantages for all 2,240 responses/560 groups from the first 35 updates. There
are four rewarded nonterminated responses; three have positive advantages,
covering 98,304 response tokens. Mathematical labels total 1,535; the optional
termination-gated training reward would total 1,531. Mixed groups change from
99 to 98 and nonzero-advantage tokens from 9,652,017 to 9,468,501. The new
update-35 affected group becomes all-zero reward/advantage, removing its 128,405
originally active tokens. All previous twenty-batch summaries and changed
advantage records remain exactly equal to the earlier independent replay.
This quantifies the defect, not its causal role in collapse or the learning
effect of the candidate. The candidate remains **unapplied**. See
`termination_reward_candidate/selected_output_counterfactual_first35.json` and
`first35_counterfactual_provenance.json` in the comparison directory. All 506
frozen non-Markdown source hashes and 18 evaluation pins still match. No
policy-35 accuracy, restoration, alignment or stability acceptance is claimed.

The complete policy-35 validation now scores 82/128 (64.0625%), versus 60
initially, 75 at update 25 and 74 at update 30. Against the initial outputs
there are 29 wins/seven losses; 28 wins were previously unfinished and one
previously completed but incorrect. The nominal exact McNemar p=0.00031255
and conservative nominal 95% change interval [+3.08, +30.15] percentage points
remain unadjusted for repeated interim looks and recipe selection. Update 30
to 35 has fifteen wins/seven losses (p=0.1338), all fifteen wins previously
unfinished and all seven regressions now unfinished with strongly repetitive
tails. Update 25 to 35 has thirteen wins/six losses (p=0.1671). Thus the observed
gain is not an isolated SGD/cache causal estimate or final learning/alignment
acceptance. Serving and migration remain confounders.

Unfinished responses decline from 59 initially/43 at update 30 to 35; strongly
repetitive tails decline from 51/35 to 29. All parsing and verifier-timeout
flags remain zero. Exact UID joins of all 128 outputs to selected, arrival and
request-begin/end events verify version 35 and token lengths. There are 39
migrated responses and 35 budget hits, none correct. The continuous paired
statistics and repetition watchers independently match the complete CPU audit;
event-log prefix hashes, validation hashes, all 506 non-Markdown source hashes
and 18 common-evaluation pins are reverified. Full checkpoint 35 is retained
because this is the highest monitored point: all 55 files have identical
device/inode/size after hard linking, including sixteen model, optimizer and
extra-state shards each. This is retention, not a restoration test or a change
to the predeclared update-50 endpoints. See `validation35_diagnosis.json`,
`validation35_outcome_transition_audit.json`, `validation35_serving_version_audit.json`,
`validation35_regression_ending_audit.json`, `validation35_audit_provenance.json`
and `checkpoint35_preservation.json` in the active run.

After validation, step 35 logs training reward 46/64, rolling reward 0.70508,
gradient norm 0.11058 and maximum lag eight. The job remains RUNNING and the
source, scoring and unrestricted-staleness recipe remain unchanged. Async50,
matched Sync50, common512 and conditional 36-hour stability remain incomplete.

Update 36 completes and policy 36 is published: training reward 46/64, rolling
reward 0.71094, gradient norm 0.06288 and maximum lag eight. Complete pre-update
actor-35 forwards have RMS 0.02638, with both positive thinking-close and both
positive end-marker derivatives active. Eleven groups are all-correct, four
all-wrong and only one mixed, leaving 120,383 nonzero-advantage tokens. Thus
high input reward alone still does not imply a large effective task-gradient
batch or post-update accuracy. See `update36_training_diagnosis.json` and the
immutable `update36_selected_forward_snapshot.json`. The read-only observer
now follows `validation40_live_wait.jsonl`; primary comparison and stability
requirements remain unchanged and unaccepted.

Update 37 completes and policy 37 is published: training reward 55/64, rolling
reward 0.71289, gradient norm 0.07558 and maximum lag ten. Complete actor-36
forwards have RMS 0.02655, with all three positive thinking-close and three
positive end-marker derivatives active. Thirteen groups are all-correct, one
all-wrong and two mixed; nonzero-advantage tokens total 178,006. One group has
three behavior-33 siblings and one behavior-34 sibling, all correct with zero
task advantages. This mixed-version group therefore provides no direct task
policy-gradient signal and does not establish a regression cause. Neither
sampling nor staleness control changes. See `update37_training_diagnosis.json`
and `update37_selected_forward_snapshot.json`; the latest fixed-set result
remains policy 35 at 82/128 and the next full validation is policy 40.

Update 38 completes and policy 38 is published: training reward 51/64, rolling
reward 0.70508, gradient norm 0.07878 and maximum lag eight. All 64 actual
pre-update actor-37 forwards match, with RMS 0.02657 and all three positive
thinking-close and three positive end-marker derivatives active. Twelve groups
are all-correct, three all-wrong and only one mixed, leaving 103,557 active
advantage tokens. See `update38_training_diagnosis.json` and
`update38_selected_forward_snapshot.json`. These input/optimizer observations
do not replace fixed-set policy accuracy; the job and primary comparison
continue without a recipe change or stability acceptance.

Update 39 completes with training reward 54/64, rolling reward 0.73633,
gradient norm 0.09383 and maximum lag nine. All 64 pre-update actor-38
forwards match, with RMS 0.02595, twelve all-correct/one all-wrong/three mixed
groups and 158,453 nonzero-advantage tokens. One of six positive thinking-close
derivatives is clipped; all six positive end-marker derivatives remain active.
Its successful 27,313-token response was generated entirely at behavior 30,
without migration, and has a thinking-close ratio of 1.40067 at actor 38.
This lag-eight example does not distinguish weight drift from numerical error.
See `update39_training_diagnosis.json` and the corresponding selected-forward,
ending-case and generation audits.

Policy 40 is published and all 55 checkpoint files are present, but its fixed
128-question validation and complete training metric row are still pending.
The 64 pre-update actor-39 forwards have RMS 0.02790, nine all-correct/four
all-wrong/three mixed groups and 353,796 nonzero-advantage tokens. Two of seven
positive thinking-close derivatives are clipped; both use behavior 35, lag
four, while all seven positive end-marker derivatives remain active. These
are selected-input observations, not validation-40 accuracy. See
`update40_validation_pending_audit.json` and its immutable forward snapshot.

The exact live BF16 publications 38 and 40 are exported on CPU, with all 760
tensor hashes verified against their immutable publication manifests before
and after serialization. Checkpoint 30 is preserved before retention and
reconstructed on CPU for the behavior-30 control above. Independently, the
same CPU reconstruction route at checkpoint 40 reproduces every byte of all
760 live-publication tensors (9,409,813,744 values, zero mismatches). This
validates model export at 40, not optimizer restoration or NPU inference.
There is no captured live-publication-30 manifest, so its direct byte identity
remains unverified. All 512 processor cases pass for the publication-38 and
reconstructed-30/40 models; full vocabulary maps, EOS and banned vision-token
masks also agree. Forced-token NPU drift decomposition remains unexecuted.
All 506 frozen production sources and 18 common-evaluation pins still match.

The frozen-output termination-reward audit also covers the old synchronous
run's four completely archived batches: 256 responses, 64 groups and 122
mathematically correct responses. All 122 correct responses end normally.
Reproducing every actual GRPO advantage and applying the termination
multiplier therefore changes no rewards, no advantages, none of the 19 mixed
groups, and none of the 1,914,830 nonzero-advantage tokens. Independently checked
unpadded masks and actual final tokens confirm all 256 termination decisions;
all archived inputs have recorded hashes. This excludes this particular reward
boundary defect from those four batches, while not ruling out later batches
or other reward defects. It does not prove numerical error is the sole cause
of the synchronous collapse. See `termination_reward_candidate/old_sync_first4_counterfactual.json`
and its provenance in the comparison directory. The private CPU helper now
accepts explicit run/output paths; replaying the first 35 current asynchronous
batches reproduces every field of the original counterfactual report. No
online scorer, training source, NPU worker or recipe changes.

Complete validation 40 now has 85/128 correct (66.40625%), compared with
60/128 initially and 82/128 at 35. Against the initial fixed set it has 27
wins, two losses and 99 unchanged results, a 19.53125-point increase; the
nominal exact paired p-value is 1.62423e-6 and conservative 95% change interval
is [+7.38,+30.23] points. These statistics remain unadjusted for repeated
interim looks and recipe selection. Against 35 it has nine wins and six
losses, only +2.34375 points (nominal p=0.60724); that small increment is not
established improvement on its own. Initial-to-40 wins comprise 26 previously
unfinished answers and one finished incorrect answer. Both initial
regressions are unfinished with strongly repetitive tails; all six 35-to-40
regressions are unfinished, four with strongly repetitive tails.

Format-valid responses rise from 69 initially to 98, unfinished thinking
falls 59 to 30, and strongly repetitive tails fall 51 to 23. All parsing and
verification error flags remain zero. All 128 UID joins verify selected,
arrival and actual begin/end version 40 with exact lengths; 38 responses
migrate and 30 hit the budget, with no correct budget-hit response. End-to-end
improvement therefore remains confounded by serving/migration and does not
isolate SGD or cache causality. Independent checks reproduce canonical paired
and repetition summaries and reverify every validation hash and event-log
prefix hash. See `validation40_diagnosis.json`, `validation40_audit_provenance.json`
and the serving, transition and regression audits.

Step 40 logs raw training reward 43/64, rolling reward 0.75977, gradient norm
0.10997 and maximum lag eight. Its lower raw batch reward than step 39 is not
a fixed-question regression; complete fixed-set accuracy increases instead.
The new highest monitored checkpoint 40 is retained with all 55 files verified
by device/inode/size, for diagnostic use only. Neither best-model selection nor
optimizer restoration is claimed. All 506 frozen production sources and 18
common-evaluation pins remain unchanged. The asynchronous job continues toward
50, followed by the exact synchronous control and common512; alignment and
36-hour stability have not been accepted.

Update 41 completes and policy 41 is published. Training reward is 43/64,
rolling reward 0.75781, gradient norm 0.03431 and maximum lag nine. All 64
pre-update actor-40 forwards match, with actor/behavior RMS 0.02577 and all
three positive thinking-close and three positive end-marker derivatives
active. Ten groups are all-correct, five all-wrong and only one mixed;
nonzero-advantage tokens total 46,151. The four behavior-31 responses with
lag nine are all-correct with zero advantages, supplying no direct task
policy-gradient signal. This is not an isolated staleness or convergence
result. See `update41_training_diagnosis.json` and its immutable selected-forward
snapshot. The latest complete held-out result remains validation 40 at 85/128.
Training continues toward 45/50 and the paired control, without changing the
recipe; all 506 production sources and 18 evaluation pins still match.

Update 42 completes and policy 42 is published: training reward 40/64,
rolling reward 0.73828, gradient norm 0.09776 and maximum lag eight. Its 64
complete pre-update actor-41 forwards have RMS 0.02690; all four positive
thinking-close and four positive end-marker derivatives remain active.
Nine groups are all-correct, four all-wrong and three mixed, with 324,114
nonzero-advantage tokens. Nineteen responses hit the length budget, all with
zero correctness reward. This batch's lower raw reward does not establish
fixed-question regression; the latest complete validation remains 40 at
85/128, with the next at 45. See `update42_training_diagnosis.json` and its
immutable selected-forward snapshot. Frozen source/evaluation hashes match;
the paired learning and stability requirements remain open.

Update 43 completes and policy 43 is published: training reward 50/64,
rolling reward 0.74609, gradient norm 0.09070 and maximum lag ten. All 64
pre-update actor-42 forwards match, with RMS 0.02674 and all six positive
thinking-close and six positive end-marker derivatives active. Eleven groups
are all-correct, two all-wrong and three mixed, with 136,883 nonzero-advantage
tokens; four responses hit the budget. The four behavior-32 responses at lag
ten are all-wrong with zero advantages, so they provide no direct task-policy
gradient. Neither this higher raw reward nor its lag establishes a fixed-set
learning or staleness effect. See `update43_training_diagnosis.json` and its
immutable selected-forward snapshot. The latest complete validation remains
40 at 85/128 and all production/evaluation pins match.

A CPU observer now waits for the final asynchronous publication 50, to preserve
its exact live BF16 bytes before job cleanup. Its exporter is the unchanged
route already verified at 38/40; it starts no NPU actors or training jobs.
Live capture at 50 is still unexecuted. This provides a later check against
the actual final checkpoint export used by common512; the existing full-byte
identity proof at 40 does not assert final-50 identity. Observer state is
`final_publication_export_watch_status.json`; neither source/scoring changes
nor automatic stability acceptance are introduced.

Update 44 completes and policy 44 is published: training reward 38/64,
rolling reward 0.73047, gradient norm 0.07815 and maximum lag ten. All 64
pre-update actor-43 forwards match, with RMS 0.02777 and both positive
thinking-close and both positive end-marker derivatives active. Nine groups
are all-correct, six all-wrong and one mixed, with 84,784 nonzero-advantage
tokens; eleven responses hit the budget. One all-correct group has three
behavior-34 siblings and one behavior-33 sibling, all with zero task
advantages. This supplies no direct task-policy gradient and does not
establish a regression cause. The batch score is not a fixed-set policy
comparison. See `update44_training_diagnosis.json` and its immutable forward
snapshot. Frozen source/evaluation hashes match and validation 45 is next;
no recipe, staleness or scoring change is introduced.

Policy 45 is published, all 55 checkpoint files are present, and its fixed
128-question validation is generating. Complete training metrics for step45
remain pending until validation finishes; the latest complete validation is
still 40 at 85/128. Its 64 pre-update actor-44 forwards have RMS 0.02805, six
all-correct/seven all-wrong/three mixed groups and 220,194 nonzero-advantage
tokens. Selected inputs have reward 31/64 and sixteen budget hits. One of
seven positive thinking-close derivatives is clipped, while all seven positive
end-marker derivatives remain active. The successful, normally ended
21,038-token case uses behavior35, lag nine: its thinking-close ratio is
1.20832, just above the 1.2 clip threshold, and its EOS ratio is 1.00035.
Policy differences and numerical effects remain combined; this is not a
same-weight numerical-failure result. See `update45_validation_pending_audit.json`,
its immutable forward snapshot and `update45_positive_ending_case_audit.json`.
Frozen production/evaluation inputs still match; no recipe change or gate
acceptance follows from this batch.

Complete validation 45 remains at 85/128 (66.40625%), matching 40. Their
paired transition has five wins/five losses/118 unchanged outcomes, with a
nominal conservative 95% difference interval of [-8.61,+8.61] points; equal
point scores or p=1 do not establish equivalence or convergence. Against the
initial 60/128 it has 27 wins and two losses, +19.53125 points, nominal paired
p=1.62423e-6 and a conservative interval of [+7.38,+30.23] points. These
interim statistics remain unadjusted for repeated looks and recipe selection.
All five 40-to-45 regressions are unfinished, three with strongly repetitive
tails; both initial regressions are unfinished with strongly repetitive tails.

Format-valid answers rise from 98 at40 to99 at45, unfinished thinking falls
30 to29, and strongly repetitive tails remain23 (initially51). Parsing and
verification error flags remain zero. All128 UID joins verify actual selected,
arrival and begin/end version45 with exact lengths;53 responses migrate and
29 hit the budget, with no correct budget-hit response. Serving/migration
continues to confound isolated SGD/cache attribution. Independent checks
exactly match canonical paired/repetition summaries and reverify all raw
validation hashes and serving-event prefix hashes. See `validation45_diagnosis.json`,
`validation45_audit_provenance.json` and its serving/transition/regression audits.

Step45 logs raw reward31/64, rolling reward0.68359, gradient norm0.08892 and
maximum lag nine. Its lower raw training score accompanies an unchanged
complete fixed-set score, reinforcing the need to keep those measures distinct.
See `update45_training_diagnosis.json`; its earlier pending report is a
historical snapshot. No extra full-checkpoint preservation is needed for this
tied monitored score; the retained40 remains diagnostic and the fixed primary
endpoint stays50. All506 frozen production sources and18 common-evaluation
pins match. Asynchronous training continues toward50, then its exact
synchronous control and common512; no learning/alignment/stability gate is
accepted yet.

Update46 completes and policy46 is published: training reward49/64, rolling
reward0.67969, gradient norm0.08780 and maximum lag nine. All64 pre-update
actor45 forwards match, with RMS0.02580 and all nine positive thinking-close
and nine positive end-marker derivatives active. Ten groups are all-correct,
two all-wrong and four mixed, with245,976 nonzero-advantage tokens; eleven
responses hit the budget. The four behavior36 responses at lag nine are
all-correct with zero advantages, so supply no direct task-policy gradient.
This raw-batch rebound does not establish fixed-set learning improvement;
the latest full validation remains45 at85/128. See `update46_training_diagnosis.json`
and its immutable forward snapshot. All506 production and18 evaluation pins
remain unchanged while the primary50-update comparison continues.

The private live-generation probe now states its actual scope explicitly:
it reads each engine detokenizer's output counters across all active request
partitions, without train/validation identity joins or a cumulative trajectory
length guarantee across migration. Its global active-request count must not
be identified with remaining held-out questions merely because counts match.
Per-validation liveness uses queue metadata; the complete validation audit
uses actual UID/request/version/length joins. This clarification changes only
diagnostic metadata, with no production or pinned evaluation changes.

Update 47 completes and policy 47 is published: training reward 41/64,
rolling reward 0.65430, gradient norm 0.12834 and maximum lag seven. Its 64
pre-update actor-46 forwards have RMS 0.02640, nine all-correct/four all-wrong/
three mixed groups and 275,636 nonzero-advantage tokens. One of five positive
thinking-close derivatives is clipped; all five positive end-marker derivatives
remain active. The successful normally ended 9,000-token case uses behavior
40 at lag six, with closing ratio 1.45800 and active EOS ratio 1.00031.
Actual original generation begins and ends at 40 with zero migrations.
See `update47_training_diagnosis.json`, its immutable selected-forward snapshot,
ending-case audit and `update47_clipped_case_generation_audit.json`.

This case supplies a stronger later weight-drift control: exact original
publication 40 was already retained, and exact live publication 46 is now
exported on CPU with all 760 tensor hashes verified before/after serialization,
lease released, no capacity waiters/failure and no NPU initialization. Both
versions have byte-identical nonweight assets. The 76 prompt tokens, all 9,000
response tokens, teacher46 and original behavior40 probabilities are frozen
independently and exactly match the shared replay loader. Both actual model
paths pass all 512 processor/prompt/reference comparisons; the new46 same-weight
plan-only path also passes. Artifacts are under
`update47_exact40_46_drift_probe` in the comparison directory, including
`replay_preparation_report.json`. No NPU replay or three-term drift decomposition
has executed. The existing driver still requires replay weight version equal
to source actor version, so old40 versus new46 requires explicitly labelled
weight versions and the paired50/common512 resource guards before execution.
Unlike the checkpoint-derived old30 control, both weights here come directly
from verified live publications; this remains diagnostic preparation, not a
learning/alignment/stability result. Primary Async50/Sync50/common512 retains
priority and the training/scoring recipe remains frozen.

Update 48 completes and policy 48 is published: training reward 47/64,
rolling reward 0.66211, gradient norm 0.09746 and maximum lag eight. All 64
pre-update actor-47 forwards match, with RMS 0.02701 and all seven positive
thinking-close and seven positive end-marker derivatives active. Ten groups
are all-correct, three all-wrong and three mixed, with 335,671 nonzero-advantage
tokens; nine responses hit the budget. See `update48_training_diagnosis.json`
and its immutable selected-forward snapshot. The latest full fixed validation
remains 45 at 85/128; these batch scores do not replace the final50 validation.
All 506 frozen production sources and 18 evaluation pins still match, and
the exact-pair alignment and conditional stability requirements remain open.

Update 49 completes and policy 49 is published: training reward 56/64,
rolling reward 0.68750, gradient norm 0.10137 and maximum lag eleven. All 64
pre-update actor-48 forwards match, with RMS 0.02732, thirteen all-correct/one
all-wrong/two mixed groups and 247,622 nonzero-advantage tokens. Eight responses
hit the budget, all with zero correctness reward; all 56 finished responses
are correct in this selected batch. One of four positive thinking-close
derivatives is clipped, while all four positive end-marker derivatives remain
active. The normally ended 31,142-token successful case uses behavior37 at
lag eleven, with closing ratio 1.22435 and active EOS ratio 1.00062. Different
policy versions and numerical effects remain combined, so this is not an
isolated numerical-failure or staleness-causality result. See
`update49_training_diagnosis.json`, its immutable forward snapshot and
`update49_positive_ending_case_audit.json`. The primary final50 validation is
still required; this high selected-batch reward is not a fixed-set accuracy
result. All 506 production sources and 18 evaluation pins remain unchanged.

Policy 50 is published and all 55 final checkpoint files are present. All
50 pre-update selected-forward batches (3,200 responses, actor versions0–49)
are observed completely, but the full final128 validation and logged step50
metrics are still pending. The actor49 batch has RMS0.02716, nine all-correct/
five all-wrong/two mixed groups,177,353 nonzero-advantage tokens and all four
positive thinking-close/four positive end-marker derivatives active. Selected
inputs have reward40/64 and six budget hits. See
`update50_validation_pending_audit.json` and its immutable forward snapshot.
The CPU observer begins preserving the final live50 publication; its result
must be checked separately and does not itself prove checkpoint50 export
identity or inference accuracy. All506 production sources and18 evaluation
pins match. Matched synchronous training still requires full asynchronous job
success and resource cleanup; common512 and conditional stability remain open.

The final live50 publication is now preserved successfully by the CPU observer:
all 760 BF16 tensor hashes match the publication manifest before and after
serialization, with its lease released and no NPU initialization. The frozen
CPU merger also reconstructs the actual final checkpoint50. A complete byte
comparison verifies all 760 tensors/9,409,813,744 values exactly equal to the
captured publication, zero mismatches and all finite. This final-specific
check supplements the earlier route control at40. All512 actual processor,
prompt-token and reference cases pass for the reconstructed50 model. See
`final_checkpoint_publication_identity_control.json` and
`outputs/async50_cpu_export_publication_identity_20261027_230156/`.
This verifies model-export bytes, not NPU inference, optimizer restoration,
learning improvement, async/sync alignment or stability.

Verified publication manifests also show 340 tensors changing from40 to46,
337 from46 to50 and346 from40 to50, including339/336/345 language-model
tensors respectively and no vision tensors. Thus published BF16 weights are
actually changing; the original all-updates-rounded-away failure is not
reproduced across these intervals. Counts do not establish update magnitude,
gradient correctness or improved inference. See
`publication40_46_50_parameter_update_audit.json`. The primary final128
validation remains pending, the matched synchronous job has not yet started,
and the common512 evaluator will still export/verify its actual inputs after
both50 jobs succeed. Production/evaluation source pins remain unchanged.

The repaired primary asynchronous job now finishes all 50 updates with SDK
`SUCCEEDED` at 2026-10-27 23:42 UTC. Final fixed validation is 85/128 (66.40625%)
versus 60/128 initially, with 31 paired wins, six losses and 91 unchanged.
The nominal exact McNemar p is 0.00004126 and conservative 95% change interval
is [+5.46, +32.30] percentage points; these statistics are not adjusted for
repeated looks or recipe selection. Steps40/45/50 all score85, but45-to50 has
nine wins/nine losses, so equal totals do not establish equivalent outputs,
convergence or stability. All six initial regressions are unfinished; four
have strong repetitive tails. All nine45-to50 regressions are unfinished,
eight with repetitive tails. Final50 has32 unfinished/budget-hit responses,
23 repetitive tails,46 migrated responses and no parser/verifier failures.
All128 actual begin/end serving versions are50. Serving/migration differences
still prevent isolated attribution to SGD or cache precision.

The final selected actor49 batch has reward40/64, rolling reward0.6875,
gradient norm0.08892 and maximum lag nine. All3,200 selected forwards across
50 batches are archived and matched. Canonical paired/repetition results,
raw validation hashes, event-prefix hashes,506 production sources and18
evaluation pins pass independent checks. See `validation50_diagnosis.json`,
`validation50_audit_provenance.json`, `validation50_regression_ending_audit.json`
and `update50_training_diagnosis.json`. Final50 remains the primary checkpoint;
its full55-file checkpoint and760-tensor publication identity are preserved.

The matched synchronous50 control is submitted on the existing cluster after
all32 NPUs become free and the old transfer queue disappears. Its experiment
directory is `qwen35_9b_sync_32npu_fp32_32k_fp32ssm_20261027_234300`, but its actual
Ray submission ID is `reasoning_manifest.json`: the private submission helper
accidentally reused the run-name variable in the reference-manifest loop.
The live SDK confirms this actual task is running with the intended command;
it is not resubmitted or renamed. The helper now keeps `run_name` separate
from manifest filenames. A fake-SDK regression exercises the real source,
dataset and configuration checks, reproduces the old ID collision and verifies
the corrected ID without submitting jobs or changing the active pointer.
Two initial harness attempts were rejected for changed PATH/double-prefixed
PYTHONPATH before the original launch environment was reconstructed; those
preflight failures are retained. See the synchronous run's
`diagnostic_tools/submission_name_regression.json`. Its actual resolved trainer
configuration is copied by experiment name rather than SDK ID and independently
matches the async configuration except mode/output paths, with identical506
non-Markdown sources and dataset manifests. This private helper repair changes
no frozen training/evaluation source, online reward or active training process.
The paired synchronous endpoint, common512 evaluation and36-hour run remain open.

The synchronous control publishes initial policy0 and all eight live serving
engines report SSM cache `float32` and convolution cache `bfloat16` through their
runtime configuration metrics. This is configuration metadata, not direct
state-tensor inspection or learning evidence. The passive actor-logprob observer
installs successfully. A CPU observer waits to preserve its exact final live50
publication, and a separate observer waits for the complete logged validation50
before running the unchanged auditor. See `startup_runtime_evidence.json` in
the synchronous output. No extra NPU diagnostic actors or training jobs are
created; initial synchronous validation remains pending.

The complete synchronous initial validation subsequently scores73/128 (57.03125%),
versus the asynchronous initial60/128 (46.875%), before any logged optimizer
update in either run. The paired pre-training difference is19 wins/six losses,
with18 wins from previously unfinished answers and one from a finished wrong
answer; all six losses are unfinished. Both complete128-question sets have
identical question/reference identities, actual begin/end version0, zero
migrations,16 requests starting on each of eight replicas, and no parser or
verifier failures. Unfinished/budget hits are59 async versus46 sync. However,
96 questions start on different numbered replicas and only two output strings
match exactly (median common prefix755.5 characters). Replica placement is
not actual kernel batch shape, and these observations do not isolate a cause.
The initial+10.15625 percentage-point difference must not be counted as a
synchronous training gain or explained by stale weights/migration in these
initial responses. See the synchronous run's
`initial_validation_serving_comparison.json` and
`initial_validation_audit_provenance.json`. The two raw initial scores remain
unchanged; each learning trajectory retains its own initial reference.

The actual synchronous live publication0 is now preserved on CPU, with all760
tensor hashes verified against the immutable relay manifest and its lease
released. A full-value CPU comparison proves all760 tensors/9,409,813,744
values exactly match both the original model cast to BF16 and the canonical
common-evaluation baseline, zero mismatches, all finite. All eight copied
nonweight assets are byte-identical as well. See
`outputs/sync0_live_publication_export_20261028_0025/sync0_vs_original_and_canonical_bytes.json`.
This establishes the sync initial model and strengthens the canonical baseline's
earlier prefix-only weight validation. No complete live async0 manifest was
captured, so direct async0-versus-sync0 live byte identity remains unavailable.
Relay-byte identity is not decoder-state equivalence or isolated batching
causality. All506 training source and18 common-evaluation pins still match.
The initial serving discrepancy reinforces the need for the unchanged common512
comparison after both50-update jobs; online raw validation differences alone
must not be accepted as learning/alignment/stability. The synchronous task
continues generating its first64 training responses without resubmission or
a Ray-cluster restart.

The repaired synchronous control completes its first optimizer update and
publishes policy1 at2026-10-28 01:05 UTC. All64 selected actor0/behavior0
responses match their archived forwards, with1,136,285 valid tokens and
actor/behavior logprob RMS0.0239751. Correct offset0 RMS matches the independent
alignment report; offsets-1/+1 instead have RMS1.1027/1.1154. All-token ratios
outside the PPO band are0.05562%. Of249,660 nonzero-advantage tokens,74 have
zero chosen-logprob task derivative (0.02964%). All eight positive-advantage
thinking-close and eight positive end-marker derivatives remain active;
thinking-close positive RMS is0.02985. This batch does not reproduce the old
first synchronous batch's12/18 masked positive thinking-close tokens, but the
questions differ, so this is not an isolated matched-ending cache ablation.

The full batch has44/64 correctness reward, nine all-correct/four all-wrong/
three mixed groups and eight response-budget hits. Logged gradient norm is
finite0.105641, behavior lag is zero, and independently reconstructed task
loss0.0653673313 matches logged0.0653673271. Actual request begin/end joins
verify all64 serving versions0. Read-only optimizer probes on worker ranks0
and8 report FP32 masters and AdamW moments, BF16 forward computation and all
760 inspected optimizer-state step counters at1. This is inspection of those
two workers, not all16 optimizer shards or a parameter-delta test. See the
synchronous run's `update1_training_diagnosis.json`, immutable
`update1_selected_forward_snapshot.json` and `update1_audit_provenance.json`.
All128 raw input/forward files are hashed, and506 production/18 evaluation
pins match. First-batch reward, numerical agreement, active derivatives and a
successful optimizer update are not post-update held-out accuracy. The next
scheduled fixed validation is5; Sync50/common512/stability remain pending.

Synchronous update2 also completes and publishes policy2. All64 selected
actor1/behavior1 responses match their forwards and actual request begin/end
versions1. Training reward is43/64, rolling reward0.6796875, gradient norm
0.0827794 and behavior lag zero. Actor/behavior RMS is0.0245644; the batch has
nine all-correct/four all-wrong/three mixed groups,305,680 nonzero-advantage
tokens and15 budget hits. All seven positive thinking-close/seven positive
end-marker task derivatives remain active. Across the two complete batches,
all128 selected forwards are matched and none of the15 positive thinking-close
or15 positive end-marker tokens is masked. Different questions still prevent
an isolated cache-ending causal claim. CPU reconstruction matches logged PPO
loss, all128 second-batch raw input/forward files are hashed, and506 production/
18 evaluation pins match. See `update2_training_diagnosis.json`, its immutable
forward snapshot and `update2_audit_provenance.json`. These are successful
numerical/optimizer updates, not a held-out learning trend; validation5 and the
full Sync50/common512 comparison remain pending.

Synchronous update3 completes and publishes policy3. Its full64-response batch
has38/64 training correctness reward (down from44/64 and43/64), rolling
reward0.6510417, finite gradient norm0.0914841 and behavior lag zero. Actual
actor2/behavior2 request joins match all64 responses. Actor/behavior RMS is
0.0249554 and reconstructed task loss0.0073806477 matches logged0.0073806434.
Nine groups are all-correct, five all-wrong and two mixed;181,192 tokens have
nonzero advantage and13 responses hit the budget. Both positive thinking-close
and both positive end-marker derivatives remain active. Across the first three
complete batches, all192 forwards match and no positive ending is masked
(17 thinking-close and17 end markers). Different training questions and this
small number of updates do not establish whether held-out precision is rising
or falling. See `update3_training_diagnosis.json` and its snapshot/provenance.

The private CPU-only `audit_complete_training_update.py` now produces these
immutable per-update snapshots, request joins, raw-file hashes and source/pin
checks. Its core results exactly reproduce the independently saved Sync1/2
and stale Async50 diagnoses in read-only dry runs; no earlier report is
overwritten. The actual Sync3 audit passes as well, with506 production and18
evaluation pins unchanged. Source and regression receipts are archived under
the synchronous run's `diagnostic_tools/`. A separate read-only observer is
attached to audit the complete validation5 once logged, alongside the existing
final50 auditor and final-publication observer. The main task continues; no
training configuration/reward changes, extra NPU actors or restart are made.

The detached per-update CPU observer is now live and has reverified Sync1/2/3,
including immutable diagnosis/snapshot hashes and all128 raw input/forward
hashes per batch. Its first attempt correctly fails closed because the early
Sync1 provenance included the refreshing `logprob_alignment_analysis.json` as
an artifact. That summary changed when more batches arrived; the original
diagnosis, selected snapshot, optimizer probe and raw packets still match.
The original alignment analyzer is replayed on only the preserved version0
packets/actor0 forwards, without writing the live summary. It recovers the
entire historical JSON byte-for-byte with the exact originally recorded SHA256
`9d2d0ddbf62e1a82c9a997db4a652e3cf3fce3a219ab0d2ac7b4b2a90809e3da`.
The versioned `update1_logprob_alignment_snapshot.json` and
`update1_recovered_artifacts.json` preserve that proof. The initial observer's
source, failure status and log are retained; only that CPU observer is replaced.
Its verifier now accepts a recovered artifact only when its bytes match the
original expected hash, and records the recovery receipt hash. The original
provenance and live canonical summary remain unchanged. The corrected observer
reverifies all three existing updates and waits for the next complete update;
it never changes training or accepts a quality gate. See
`training_update_audit_watch_status.json` in the synchronous run.

Synchronous update4 completes and the passive observer successfully audits it:
all64 actor3/behavior3 forwards and actual serving-version joins match. Raw
reward is38/64, rolling reward0.63671875, gradient norm0.0933065 and actor/
behavior RMS0.0254376. Seven groups are all-correct, five all-wrong and four
mixed, with362,013 active-advantage tokens. There are21 budget hits and18
migrated trajectories. All ten positive thinking-close/nine positive end-marker
derivatives remain active; cumulative positive endings masked across the four
batches are0/27 thinking-close and0/26 end markers. CPU task loss0.0466795280
matches logged0.0466795294. All506 production/18 evaluation pins match. See
`update4_training_diagnosis.json` and the automatically generated provenance.
Batch reward has not established a held-out learning trend; validation5 remains
the next scheduled complete fixed-set observation.

This fourth batch also supplies a concrete reward-boundary example, independent
of staleness/migration: `eb02b767-7fe0-4dfa-ad61-cffde352b2b1_0_0` is actor3/
behavior3, actual begin/end3, migration0. It closes thinking at response token
9513, contains135 `\boxed` occurrences and repeatedly plans its final answer, reaches
32768 tokens, and ends on token11 (a comma), without a normal final EOS.
The last8192 characters have only0.001968 repeated64-character-window fraction;
that character-window diagnostic does not detect this semantic planning loop.
The mathematically correct last box earns reward1 and advantage+0.8660239 over
the full unpadded response. The positive thinking-close derivative is active.
The labels follow the existing mathematical-correctness objective; this is not
proof that the mathematical verifier assigned the wrong answer label.

Replaying the optional completion-only reward on all256 first-four-batch
responses first reproduces every original actual GRPO advantage. All256
unpadded masks/final tokens are independently checked and source files hashed.
There is exactly one mathematically correct nonterminated response: this case.
Correct/completed-correct totals are163/162; mixed groups12 and active tokens
1,098,545 remain unchanged. In its four-response group, the truncated sample's
advantage changes+0.8660239 to-0.4999990, the other normally ended correct sample
changes+0.8660239 to+1.4999970, and the two normally ended wrong samples change
-0.8660239 to-0.4999990. Thus the frozen objective can change the learning signal
without changing mathematical labels. It does not demonstrate online reward,
held-out accuracy or stability benefits. See the synchronous run's
`diagnostic_tools/new_sync_first4_termination_counterfactual.json`, its provenance,
and `diagnostic_tools/update4_nonterminated_reward_case_audit.json`.
No online reward or frozen training source is changed. Unlike this new run,
the old failed synchronous run's first four batches had no rewarded nonterminated
response, so this new example cannot explain that old early collapse on its own.

At 2026-10-28 04:31 UTC, the actual synchronous SDK job remains RUNNING,
policy5 is published, and validation5 is still incomplete. A read-only CPU
observer preserves its original prompt/response token IDs, masks, positions and
rollout log-probabilities from the existing validation queue without consuming
entries. An independent audit checks the first40 preserved responses (308,049
response tokens): input IDs equal the concatenated prompt and response, tag
versions are5/5, response masks are unpadded, and all log-probabilities are finite
and match response lengths. Every checked packet is hashed. This is input
integrity evidence only; no partial accuracy is computed, and the prefix is not
claimed to be the complete128-question validation. The observer reports any
partial archive explicitly. See the synchronous run's
`validation5_input_archive_status.json` and
`diagnostic_tools/validation5_packet_prefix_integrity.json`.
Training/reward configuration and NPU actors remain unchanged. Full validation5
and the subsequent complete update5 audit remain pending.

The existing CPU validation-audit waiter is also started for intermediate
steps10/15/20/25/30/35/40/45. All eight detached processes are independently
verified alive and waiting on the actual SDK submission `reasoning_manifest.json`.
The existing step5/50 observers remain in place. Each waiter checks only a fully
logged128-question validation, joins actual serving versions and token lengths,
and records paired outcomes and repetition; none accepts a learning/stability
gate or creates NPU actors. The unchanged auditor/wrapper/statistics sources are
pinned, and all506 production hashes/18 common-evaluation pins still match.
Commands, process identities and initial verification are retained in the
synchronous run's `intermediate_validation_audit_observers.json`.

The first complete post-update synchronous validation is now available at
2026-10-28 05:16 UTC:73/128 to65/128 (57.03125% to50.78125%), ten wins,
18 losses and100 unchanged. Sixteen lost successes are unfinished and13 have
strong repetitive tails. Overall unfinished outputs increase46 to50, format
passes decrease82 to77, and strong repetitive tails decrease39 to38. All50
budget hits are incorrect. One answer is unparsed. The two normally ended lost
successes contain, respectively, two thinking-close markers (rejected by the
existing format contract), and a boxed parametrization plus the correct-looking
distance value that the parser does not accept. These observations do not
change official scores or establish a replacement scoring rule.
The nominal paired p-value is0.18493 and the conservative change interval is
[-18.9803,+6.9058] percentage points; this uncertainty does not establish
non-regression or equivalence. The observed six-and-a-quarter-point decrease
remains an open learning concern. See `validation5_diagnosis.json`, the
independently checked `validation5_audit_provenance.json` and
`validation5_regression_ending_audit.json` in the synchronous run.

Actual validation5 begin/end versions are5 for all128 responses. Forty-eight
responses migrate: their initial/current correct totals are21/13, while the80
nonmigrating responses have52/52. Migration membership is selected by actual
generation outcomes and duration; this grouping does not identify a causal
migration effect. Nine of the18 lost successes were initial Sync0-only successes
relative to Async0, and nine were correct under both initial runs. Neither
initial baseline is replaced. These complete descriptive overlaps are retained
in `validation5_serving_association_audit.json`; common512 remains required.

Update5 has training reward37/64, rolling reward0.625, finite gradient
norm0.1066180, lag zero and actor/behavior RMS0.0256036. Its64 complete matched
forwards bring the cumulative total to320. Positive thinking-close/end-marker
derivatives remain active for all36/35 checked tokens. The independent proof
rechecks all128 raw input/forward hashes for update5, complete validation
statistics, immutable event-log prefixes,506 production hashes and18 evaluation
pins. All128 validation packets are preserved with finite behavior log-probs;
their2,450,177 response tokens decode exactly to the scored texts, and archived
rewards/versions/lengths/migration counts match the complete logged results.
See `validation5_complete_archive_audit.json`.

All55 native checkpoint5 files are preserved through hardlinks, and exact live
publication5 is CPU-exported to
`outputs/sync5_live_publication_export_20261028_051957/model` with all760 raw
hashes verified and its relay lease released. Comparing the verified live0/5
manifests finds350 changed tensors (349 language, zero vision); eight nonweight
assets are identical. Thus updates survive BF16 publication, without proving
their magnitude, gradient correctness, direct all-engine NPU byte identity or
learning benefit. Receipts are under the synchronous run's `diagnostic_tools/`.
The primary synchronous endpoint remains50 and the job continues unchanged.

The completion-only objective counterfactual is additionally replayed over all
3200 responses/800 groups from the completed repaired Async50 run. Every actual
original GRPO advantage is reproduced, all unpadded masks/lengths are checked,
and all3200 source hashes are independently reverified. There are2219
mathematically correct responses,2215 normally ended correct responses, and
four correct but nonterminated32K outputs. Three have positive advantages over
98,304 tokens (0.77646% of all12,660,596 nonzero-advantage tokens), with98,252
actual nonzero chosen-logprob task derivatives; the fourth originally has zero
advantage. The three positive cases belong to groups without a normally ended
correct sibling, so the candidate zeros their entire group advantages rather
than producing a negative advantage on those cases. The fourth receives
advantage-1.499997 under the candidate. Mixed groups change134 to133 and active
tokens12,660,596 to12,477,080. This quantifies a real but infrequent objective
gap in this repaired run; it neither proves online benefit nor explains the
old synchronous collapse. The candidate remains unapplied. Full report,
provenance and independent inspection are in the Async50 run's
`diagnostic_tools/async50_termination_objective_counterfactual*.json` and
`async50_termination_objective_independent_inspection.json`.

Synchronous update6 completes and publication6 is installed. All64 actor5/
behavior5 forwards and actual serving-version joins pass the automatic audit.
Training reward is29/64 (0.453125), rolling reward0.5963542, gradient
norm0.1111792, actor/behavior RMS0.0252288 and lag zero. Four groups are
all-correct, seven all-wrong and five mixed, with392,165 active-advantage tokens;
14 responses hit the budget and19 migrate. All13 positive thinking-close and13
positive end-marker task derivatives remain active. Cumulative positive ending
derivatives masked across the six updates are0/49 thinking-close and0/48 end
markers, over384 complete matched forwards. CPU task loss0.0849436934 matches
logged0.0849437094 within the declared tolerance. All506 production hashes and18
evaluation pins still match. Raw training reward continues to decline, while
the next scheduled full fixed-set validation remains10; no learning/alignment
or stability gate is accepted.

A historical same-question comparison now separates some question-composition
variation from the training curve. The completed Async run has264 archived
behavior0 training samples:65 prompt/reference groups have exactly four
samples, while two groups without four version0 samples are excluded. Matching
by exact prompt-token hash and reference, before inspecting outcomes, fully
covers each of the first four synchronous batches. Their historical behavior0
versus synchronous correct totals are44/44,43/43,38/38 and45/38 (each out of64).
The fifth batch matches only one of16 groups, with4/4 correct under each run;
the sixth matches no complete baseline group. No full-batch baseline estimate
is made for those last two batches. Independent inspection verifies all648
source-job/file hashes and exact prompt/reference identities, matches all six
synchronous full-batch rewards to logged metrics, and freezes their384 selected
identities. None of the260 matched response-token sequences is identical
between the two runs. Within the fourth batch, seven prompt groups have fewer
correct samples, one has more and eight are unchanged. Different sampling draws,
serving histories and version-assignment timing remain confounded. The first
three falling batch means therefore do not themselves identify learning damage,
while the fourth-batch gap and fixed validation5 regression remain open concerns.
See the synchronous run's `diagnostic_tools/sync_first6_matched_async0_training_questions.json`,
its provenance, and `sync_first6_matched_async0_token_inspection.json`.

Synchronous update7 completes with reward44/64 (a rebound from29/64), rolling
reward0.609375, finite gradient norm0.1613872, lag zero and actor/behavior
RMS0.0241562. All64 actor6/behavior6 forwards and actual serving-version joins
pass the automatic audit. Ten groups are all-correct, four all-wrong and two
mixed; five responses hit the budget and21 migrate. All four positive
thinking-close/four positive end-marker task derivatives remain active. The
seven-update total is448 matched forwards and0/53 thinking-close,0/52 end-marker
positive derivatives masked. This changing-question training rebound does not
resolve fixed validation5's73-to65 decrease; the next fixed validation is10.

A CPU audit now measures actual publication0/5 values, rather than only counting
changed tensor hashes. All760 tensors (9,409,813,744 elements) are finite and
their actual raw hashes match the preserved live manifests. Exactly350 tensors/
364,824,817 elements change, or3.877067% of elements. Global relative L2 change
is5.875056e-5, with the unchanged vision weights included in that denominator.
The maximum absolute change is0.0078125 and occurs in gated normalization,
whereas the other changed parameter categories peak at7.629395e-6. Gated norm
weights account for45.3238% of total squared published-weight change. See
`diagnostic_tools/sync0_to5_publication_delta_magnitude.json`.

All16 retained checkpoint5 model shards are read on CPU with mmap and hashed.
Only72 small SSM tensors are reconstructed:24 gated norms,24 A_log and24
dt_bias. Their FP32-master casts match actual publication5, and the original
checkpoint casts match actual publication0. The gated norms' maximum FP32
change from the original checkpoint is5.066395e-6, but247 of3072 norm elements
cross one BF16 representation interval. All247 original values are exact BF16
midpoints. An independent raw-safetensor/IEEE-bit implementation verifies the
original values, actual publication bits, nearest-even casts and adjacent-value
crossings. On these flipped elements, published/master-change amplitude ratios
range809.09 to65536 (median1820.44). The original24 gated-norm tensors are F32.
Meanwhile756 A_log and745 dt_bias master elements change slightly, but their
BF16 published values remain unchanged. This establishes quantization effects,
not their effect on reward or correctness of the parameter gradients. The
comparison uses original-checkpoint FP32 values, not a directly captured
initial-master0 snapshot.

The precision evidence and exact source copies are retained in
`diagnostic_tools/sync5_ssm_fp32_master_rounding_audit.json`,
`sync5_rounding_midpoint_independent_inspection.json` and
`sync5_parameter_precision_audit_provenance.json`. A selective FP32 gated-norm
candidate would require coordinated learner-forward, publication and serving
precision plus an isolated NPU/learning control. It must not be inferred from
the earlier native-versus-fused norm-operator comparison, which did not isolate
parameter precision. No such NPU counterfactual or precision change is applied
to the current frozen pair. Primary Sync50/common512 and the conditional
separate36-hour stability run remain outstanding.

To preserve transient evidence at the remaining fixed validations, a read-only
CPU archive supervisor is started for versions10/15/20/25/30/35/40/45/50. Its
actual process and SDK status are verified live, waiting for the real
publication10 event; no future raw archive is claimed complete. It launches
one child at a time using a separately archived derivative of the successful
validation5 packet archiver. The derivative keeps the CPU/NPU-init guards,
existing-queue-only attachment, finite log-probability checks, atomic packet
writes and explicit partial-archive reporting. Its default observation window
has no two-hour expiry, and transient SDK/queue reads are retried without
restarting or changing training. Complete scored validations remain governed
by the independent validation auditors. The original validation5 archiver
source is unchanged. Source hashes, command and PID are retained in
`diagnostic_tools/remaining_validation_input_archiver_launch.json`; live state
is in `remaining_validation_input_archiver_status.json`.

Synchronous update8 completes and its automatic audit passes all64 actor7/
behavior7 forwards and actual serving joins. Training reward is48/64 (0.75),
rolling reward0.626953125, gradient norm0.0906278, lag zero and actor/behavior
RMS0.0242859. Ten groups are all-correct, three all-wrong and three mixed,
with259,820 active-advantage tokens; ten responses hit the budget and18 migrate.
All eight positive thinking-close/eight positive end-marker task derivatives
remain active. Across eight updates, all512 selected forwards match and positive
ending derivatives masked remain0/61 thinking-close and0/60 end markers.
The reward rebounds in updates7/8 concern different training questions; they
do not establish fixed-set learning or resolve validation5's73-to65 decrease.
Publication8 is installed and the next batch starts; full validation10 remains
the next scheduled held-out observation. Source/evaluation pins and quality
acceptance remain unchanged.

Synchronous update9 completes with reward50/64 (0.78125), rolling128-group
reward0.638671875, finite gradient norm0.0936348, lag zero and actor/behavior
RMS0.0243479. All64 actor8/behavior8 forwards and actual serving-version joins
pass the automatic audit. Twelve groups are all-correct, three all-wrong and
one mixed, with121,268 active-advantage tokens; seven responses hit the budget
and19 migrate. Both positive thinking-close/two positive end-marker task
derivatives remain active. Across nine updates, all576 selected forwards match
and positive ending derivatives masked remain0/63 thinking-close and0/62 end
markers. Recent batch rewards44/64,48/64,50/64 cover different questions and
cannot establish held-out improvement. Publication9 is installed and update10
sampling starts. Validation10 and its raw-packet observer remain pending.

At 2026-10-28 08:52:39 UTC, the native scheduler publishes synchronous
version10 after all64 actor9 forward records are captured. The fixed128-question
validation starts; its complete score and logged update10 row are still pending.
The remaining-validation archiver automatically starts the version10 child,
PID1064582. At 08:56 UTC, eight responses are preserved with behavior log-probs,
zero transient read errors and no NPU initialization. These partial packets
are an archive-progress observation, not a validation accuracy estimate.
The native publication event and its scheduler-prefix hash, together with
observer startup snapshots, are retained in the synchronous run's
`diagnostic_tools/validation10_publication_archive_startup.json`.

A separate read-only CPU observer, PID1069402, now waits for each remaining
complete scored-validation audit and all128 archived packets before invoking
the unchanged complete-archive checker already used for validation5. It checks
decoded prompt/output identity, archived scores, actual serving versions,
lengths and migration counts. An incomplete archive is reported as unavailable
for full identity checking and is not treated as a learning failure. Source
hashes and launch provenance are recorded in
`diagnostic_tools/validation_archive_audit_watch_launch.json`; live state is
`validation_archive_audit_watch_status.json`. Syntax compilation, the existing
validation5 checker-proof identity and startup waiting state are verified;
future complete checks have not yet run. Production sources, the active
training configuration and common-evaluation pins remain unchanged.

The common-evaluation continuation is strengthened before either evaluation
job starts: the actual initial/Async50/Sync50 evaluation model payloads must
match their retained live-publication tensor manifests. The previous CPU
continuation only checked successful CPU exports and did not perform this
full comparison on the newly created evaluation artifacts. The new checker
validates all760 BF16 tensor hashes, shapes, finiteness and actual safetensor
file/index mappings; tokenizer/processor semantics remain covered separately
by the unchanged common-serving preflight. Both indexed baseline and
single-file Async50 exports pass all9,409,813,744 values, and their hashes
independently match the previous Torch-based comparisons. A deliberately
incorrect manifest is rejected without creating a success proof. The real
512-question/two-model processor preflight also passes without NPU inference.

Only the waiting CPU continuation is replaced: old PID176250 is confirmed
terminated, and PID1135896 is verified live waiting on the actual synchronous
submission `reasoning_manifest.json`. The replacement retains the exact18
evaluation input pins, all506 production source hashes, original decoding
protocol and 50-step endpoints. It waits for the Sync50 live-export receipt,
checks the actual three evaluation models, and refuses evaluation submission
on a byte mismatch. Provenance, original observer state, source copies and
CPU preflights are retained under
`outputs/common_evaluation_identity_preflight_20261028_0949/`, including
`observer_replacement_launch.json`. No training job or Ray cluster is restarted,
and no model-quality, alignment or stability acceptance follows from these
identity checks. At 09:51 UTC, Sync10 validation still has14 outstanding
responses and no complete score.

At 2026-10-28 09:58 UTC, synchronous validation10 completes at69/128
(0.5390625), compared with73/128 initially and65/128 at5. Relative to the
initial outputs there are15 wins,19 losses and94 unchanged; relative to5
there are16 wins and12 losses. The nominal paired p-value0.60759 and
conservative change interval[-0.17136,0.11091] do not establish non-regression
or equivalence. There are46 unclosed thinking sections,81 format-valid
answers and37 strongly repetitive tails, versus46/82/39 initially and50/77/38
at5. All48 budget-hit responses are wrong; parser/verifier failures are zero.
Actual serving-event joins verify all128 responses begin and end at version10.

All19 initial successes lost at10 hit the32,768-token budget and their preserved
raw response tensors lack a final stop token. Seventeen leave thinking open;
the other two close thinking but still truncate the final response. They must
not be classified as normally terminated answers based only on
`unfinished_thinking=0`. Eleven of the19 have strongly repetitive tails.
The60 migrated responses change from27 correct initially to18, while the68
nonmigrated responses change46 to51; migration membership depends on outcome
duration, so these are descriptive strata and do not identify a migration
effect. Five lost successes occur without migration. The original held-out
labels and online rewards remain unchanged.

The full raw-token/archive check passes for all128 scored outputs. Independent
provenance also rechecks native event-prefix hashes, all128 validation packet
hashes, all128 selected-input/actor-forward packet hashes for update10, the506
production sources and18 evaluation pins. See `validation10_diagnosis.json`,
`validation10_complete_archive_audit.json`, `validation10_regression_ending_audit.json`,
`validation10_serving_association_audit.json` and `validation10_audit_provenance.json`
in the synchronous run, with producer
`diagnostic_tools/validation10_independent_provenance.py`.

Update10 training itself scores48/64, rolling128-group reward0.6484375,
gradient norm0.1062459, actor/behavior RMS0.0247693 and lag zero. All64
actor9/behavior9 forwards match. Ten groups are all-correct, three all-wrong
and three mixed; eight responses hit the budget and15 migrate. All eight
positive thinking-close/eight positive end-marker task derivatives remain
active. Across10 updates, all640 selected forwards match and positive ending
derivatives masked remain0/71 thinking-close and0/70 end markers. The native
job continues sampling update11 unchanged. This is partial recovery from5,
not proof of improvement over the synchronous initial policy or final
asynchronous/synchronous accuracy alignment; Sync50/common512 remain pending.

A CPU-only configuration preflight prepares, but does not submit, a separate
conditional36-hour asynchronous run. After replaying the verified training
parent environment and accounting for the launcher's own PYTHONPATH prefix,
the normalized configuration differs from actual Async50 only in
`trainer.total_training_steps=100000` and `trainer.total_epochs=100`, preventing
reuse of the50-step comparison limit. The constant1e-6 learning rate and zero
warmup are unchanged; model/data, FP32 masters/SSM cache, native normalization,
32-NPU layout, no staleness cap, reward and fixed evaluation settings all match.
The506 production hashes are reverified. Initial environment differences and
all successive config-only preflights remain recorded under
`outputs/conditional_36h_config_preflight_20261028_1010/`; the verified comparison
is `launcher_input_env_replay/configuration_comparison.json`.
No training submission, NPU actor, completed stability interval or recipe
acceptance follows from this preflight. A real129,600-second supervisor remains
to be connected after complete quality/alignment review; diagnosis/control
time is excluded and the stability authorization remains gated.

The separate-run36-hour timer is now implemented, but not launched, in
`.runtime/mooncake-cpu/watch_accepted_36h_stability.py`. Its entrypoint requires
a receipt of completed regression/reward/alignment review, hashes of the
reviewed common-evaluation result, matching production sources/dataset
manifests and the exact newly submitted36h asynchronous job. Timing begins
at that separate run's native publication1; initialization/initial validation
and all diagnosis/control time are excluded. It never submits/restarts jobs
or restarts Ray. At129,600 seconds it requests stopping only that exact job,
then waits for actual SDK termination; timer completion does not accept
post-run learning or stability quality.

Seven simulated checks pass for the real duration boundary, SDK read/stop
transport retries, observer restart, early training failure, missing initial
publication, wrong SDK identity and rejection of the current synchronous
comparison. Real read-only checks also confirm SDK entrypoint identity and
native publication1/complete-update parsing against existing Sync audits.
Source copies, test source/results and the read-only interface receipt are
under `outputs/conditional_36h_timer_checks_20261028_1018/`. No real stop request,
training submission or36-hour timer has been started by these checks. The
actual quality/alignment condition and the long run remain unfulfilled.

Synchronous update11 completes and its automatic audit verifies all64
actor10/behavior10 inputs and actual serving joins. Reward is48/64, rolling
128-group reward0.66796875, lag zero, gradient norm0.0001156614 and actor/
behavior RMS0.0260225. Twelve prompt groups are all-correct and four all-wrong;
there are no mixed-reward groups. A separate CPU inspection verifies that
all944,022 valid tokens in the64 archived forward-input packets have exactly
zero advantages. Each of the16 groups has four distinct output strings.
Logged task policy loss is zero, while total loss5.02694e-7 equals the
KL loss0.000502694 multiplied by0.001. This is consistent with group-mean
subtraction under uniform binary rewards; optimizer momentum and weight
decay can still move parameters despite zero task advantages.

Mixed groups are30/176 in Sync1–11,33/176 in Async1–11 and134/800 in Async1–50.
The asynchronous run also has zero-mixed batches at1 and24. All observed groups
in these cohorts have four distinct outputs. These different-question cohort
statistics do not isolate a mode effect, prove independent random draws or
explain the fixed-set regression. See `update11_training_diagnosis.json`,
`update11_audit_provenance.json`, and
`update11_group_signal_independent_inspection.json`, whose producer is
`diagnostic_tools/update11_group_signal_inspection.py`. The latter rechecks
all128 selected-input/forward packet hashes and the original rollout JSONL
hashes, and never initializes an NPU. Across11 updates all704 selected forwards
match; positive ending derivative counts remain0/71 thinking-close and0/70
end markers because update11 supplies no positive-advantage endings.
The unchanged native job publishes11 and starts update12 sampling. The latest
complete fixed validation remains69/128 at10; validation15, Sync50/common512
and the conditional36-hour stability run remain incomplete.

Synchronous update12 completes at46/64 reward, rolling128-group reward
0.68359375, finite gradient norm0.1408722 and lag zero. All64 actor11/behavior11
forwards and serving joins pass. Eight groups are all-correct, one all-wrong
and seven mixed, with833,541 active-advantage tokens;12 responses hit the budget
and15 migrate. The actor/behavior RMS is0.0246714. One of14 positive
thinking-close task derivatives is clipped; all14 positive end-marker
derivatives remain active. Across12 updates, all768 selected forwards match;
positive ending derivatives masked are now1/85 thinking-close and0/84 end
markers. This exception is retained for diagnosis rather than treating the
previous zero-clipping observation as permanent.

The affected case, `6e6aa4f1-91ae-402b-aebc-d15af99240fb_1_0`, is a correct
32,224-token answer ending normally. Its thinking-close ratio is1.307603
(delta0.268196, advantage0.866024); EOS ratio1.125919 has an active task
derivative. All begin/end/selected versions are11. Two recompute migrations
occur at response prefixes27,301 and31,368, before and after the thinking-close
index28,215 respectively. These facts do not isolate migration, batching,
parameter precision or loading as a cause. Exact publication11 is exported
on CPU to `outputs/sync11_live_publication_export_20261028_1132/model`, with all760
tensor hashes verified and its lease released; FP32 master11 is not preserved.
Complete raw inputs, observed forwards, native event-prefix hashes, ending
derivatives and the CPU-only replay plan are linked in
`update12_clipped_case_cpu_replay_preparation.json`. Actual NPU replay remains
unexecuted. The existing generic execution branch assumes an asynchronous
source and requires a correctly labelled sync-source paired-job guard before
future execution; neither primary50 run nor common512 priority is bypassed.

At the user's request, storage cleanup removes162 obsolete payload files:
model/optimizer shards from retained Async10/15/25/30/35 checkpoints and the
superseded actor38/behavior30 drift-control model exports. Logs, metrics,
configuration, sources and compact metadata remain, with explicit
`PAYLOADS_REMOVED.json` markers. Removed allocated space is602,334,568,448 bytes
(560.968GiB); `du -sh outputs` is983G afterwards. The old30/38 optional replay
payloads are retired; exact40/46 and the new same-version11 controls remain.
The current Sync run, Async50/native final checkpoints, canonical baseline,
required diagnostic weights and18 common-evaluation pins are preserved.
The plan and completed removal receipt are under
`outputs/storage_cleanup_20261028_1138/`. Historical checkpoint-preservation
records remain historical and do not imply those retired payloads are still
available. The live training job is rechecked RUNNING after cleanup; Ray is
not restarted.

The same-version11 replay now has a separate prepared driver,
`.runtime/mooncake-cpu/probe_paired_full_model_cache_parity.py`, accepting either
an asynchronous or synchronous source with explicitly matching counterpart
submissions. Publication job identity must match the source, and execution
requires both full50 training SDK statuses and the corresponding common512
job/result to succeed before checking genuinely free4 head NPUs and absence
of a training queue. It therefore does not preempt the primary comparison.
The engine implementation and CPU input-loader AST are unchanged from the
tested original driver. Ten mocked eligibility checks, the real case's CPU
input/publication preflight and a read-only rejection against the currently
RUNNING Sync job all pass. Sources, results and hashes are retained in
`diagnostic_tools/update12_paired_replay_checks/` and linked by
`update12_paired_replay_preparation.json`. This supersedes the old sync-source
execution-guard limitation; actual NPU replay and isolated causal results
remain unexecuted. All primary training/evaluation sources remain unchanged.

Synchronous update13 completes at42/64 reward (0.65625), rolling128-group
reward0.693359375, finite gradient norm0.1291721 and lag zero. All64
actor12/behavior12 selected forwards and actual serving joins pass. Eight
groups are all-correct, three all-wrong and five mixed, with461,456
active-advantage tokens;12 responses hit the budget and20 migrate. Actor/
behavior RMS is0.0256160, and all ten positive thinking-close/ten positive
end-marker task derivatives remain active. Across13 updates all832 selected
forwards match; cumulative positive ending derivatives masked are1/95
thinking-close and0/94 end markers. A lower raw batch reward and a higher
rolling mean concern changing question cohorts/window membership, not a new
fixed-set learning result. The unchanged job publishes13 and starts update14;
complete validation15 remains the next held-out observation.

Synchronous update14 completes at46/64 reward (0.71875), rolling128-group
reward0.7265625, finite gradient norm0.0355810 and lag zero. All64 actor13/
behavior13 forwards and actual serving joins pass. Eleven prompt groups are
all-correct, four all-wrong and one mixed, with57,402 active-advantage tokens;
12 responses hit the budget and19 migrate. Actor/behavior RMS is0.0247850.
Both positive thinking-close/two positive end-marker task derivatives remain
active. Across14 updates all896 selected forwards match, with cumulative
positive ending derivatives masked1/97 thinking-close and0/96 end markers.
The rolling mean also reflects the low-reward update6 leaving the eight-update
window; its increase does not establish fixed-set learning. See
`update14_training_diagnosis.json` and `update14_audit_provenance.json`.
Publication14 is installed and update15 sampling starts unchanged. Complete
validation15, both50-step endpoint comparison/common512 and conditional36-hour
stability remain pending.

The update12 clipped case's native migration events are also joined to the
retained startup engine endpoints: replica2 on172.16.9.13 moves to replica5
on172.16.18.136 at response prefix27,301, then to replica3 on the same
172.16.18.136 host at31,368. Thinking-close and EOS therefore correspond to
engines whose reported startup hosts are the same; the first move is
cross-node and the second is within-node. Update12 logs zero recoveries and
zero quarantined replicas. This endpoint association is not live physical
cache/parameter inspection, a learner-rank host inference or an isolated
hardware/migration cause. It is retained in
`update12_clipped_case_startup_engine_mapping.json` for interpreting the
planned head-node replay; no new NPU experiment is started.

At 2026-10-28 13:41:52 UTC, the native scheduler publishes synchronous
version15 after all64 actor14 forward records are captured. The fixed128-question
validation starts and the existing raw-archive parent launches its version15
child, PID1350553. The startup receipt verifies six packets with behavior
log-probs, zero transient read errors and no NPU initialization; seven are
observed shortly afterwards. This is archive progress only, with no partial
accuracy estimate. The publication event/scheduler-prefix hash and raw observer
snapshot are retained in
`diagnostic_tools/validation15_publication_archive_startup.json`. All506
non-Markdown production source hashes and18 common-evaluation input pins are
reverified unchanged. The complete validation15 score, logged update15 metrics
and complete-output audits remain pending; initial73/128, validation5's65/128
and validation10's69/128 remain the latest complete fixed-set observations.

A CPU-only position profile of the update12 same-version11 clipped case
verifies all32,224 archived response log-probs and exact source hashes. Of13
positive-advantage token task derivatives clipped to zero, seven precede the
first migration, five lie between the two migrations and one follows the
second. Whole-segment actor/behavior RMS values are0.0279882,0.0314197 and
0.0310204. The256-token windows immediately before/after the first move have
RMS0.0304418/0.0280645; the second has0.0346547/0.0288205. Thus this observation
does not show a universal error jump at migration boundaries, and clipping
already exists before either move. The clipped thinking-close token remains
an individual outlier (delta0.268196), with the EOS derivative active. This
does not rule out a migration effect on that particular marker or isolate
precision, batching or hardware causes; token content and serving context
co-vary. `update12_migration_logprob_profile.json` retains all segments,
64/256/1024-token boundary windows and512-token position bins. An independent
NumPy/stdlib reader verifies81 summaries using `math.fsum`, every clipped
index and all source hashes; both analyses leave NPU uninitialized and the
primary job unchanged. Actual controlled NPU replay remains pending after the
primary comparison and common512 evaluation.

At 2026-10-28 14:47 UTC, the complete synchronous validation15 is76/128
(0.59375), versus initial73/128, validation5's65/128 and validation10's69/128.
The initial-to15 pair has13 gains,10 losses and105 unchanged questions; versus
10 there are18 gains and11 losses. The initial-relative change is+2.34375
percentage points, with a conservative95% interval[-9.8082,+14.3300] points
and nominal McNemar p0.677639 (unadjusted for repeated looks). This is an
observed recovery above its own initial score, not established sustained
learning, equivalence or async/sync accuracy alignment. All13 initial-relative
gains come from formerly unfinished responses; all10 lost initial successes
hit32,768 tokens without a final stop token and retain unfinished thinking,
with9 strongly repetitive tails. Whole-set unfinished counts fall46→41,
strong-repeat counts39→32 and format-valid counts rise82→86. All42 budget
hits are incorrect;49 responses migrate. Gold/answer parse failures and
verifier timeouts are zero. These associations do not isolate an SGD or
migration effect.

All128 validation15 native begin/end versions are15 and raw packet prompts,
outputs, scores, lengths, migration counts and finite behavior log-probs match
the scored records. A reusable CPU ending-token auditor verifies actual EOS
rather than inferring normal termination from closed thinking. Its historical
validation10 cross-check reproduces19 budget-truncated regressions, including
the2 with closed thinking. The new independent provenance check rehashes all
128 validation packets,128 update15 selected/teacher files, native event-log
prefixes,506 non-Markdown production sources and18 common-evaluation pins.
See `validation15_diagnosis.json`, `validation15_ending_token_audit.json` and
`validation15_audit_provenance.json`; all checks pass without model inference
or NPU initialization. The producer sources and historical cross-check are
retained under the run's `diagnostic_tools` directory.

The logged synchronous update15 is46/64 reward, rolling128-group mean0.73046875,
finite gradient norm0.0871804, actor14/behavior14 and lag zero. Ten groups are
all-correct, three all-wrong and three mixed;9 responses hit the budget and16
migrate. Actor/behavior RMS is0.0244387, with all6 positive thinking-close and
6 positive end-marker task derivatives active. Across15 updates all960
selected forwards match; cumulative positive ending derivatives masked are
1/103 thinking-close and0/102 end markers. The same job proceeds to update16
sampling. The production recipe remains frozen for Sync50/common512; the
separate36-hour run remains conditional and has not started.

### Additional confirmed reference/prompt defects, October28 15:00–15:20 UTC

Review of all11 Sync15 responses with score0 but closed thinking finds seven
mathematically incorrect scalar references and one incomplete reference. These
are joined by exact problem hash/source index to the pinned original JSON,
prepared parquet and scored output; the preprocessing did not mismatch rows.
Examples include a rectangle-area reference24 instead of48, an ellipse's
between-focus distance4sqrt2 instead of8sqrt2, a triangle-side reference
sqrt19 despite the other sides being3 and1, and a positive sqrt3 reference for
(x+y)/(x-y) when y>x>0. Exact branch checks, geometry, recurrence calculation
and exhaustive permutation enumeration certify the seven specific defects.
Changing only those seven references in the diagnostic verifier accepts the
corresponding model answers; official scores and frozen data are unchanged.
The other selected cases are an incomplete matrix-parameter reference, an
inconsistent diagram requiring review, a truncated final answer, and a
mathematically valid left-shift answer rejected when prose is inside the box.
This outcome-selected review does not estimate the corpus-wide error rate.

An independent arithmetic audit uses rational/decimal geometry, direct
log-recurrence iteration and subset-DP permutation counting. A separate
deterministic12-row training sample, selected by a fixed hash before using
model outputs, contains two incorrect scalar references and one inconsistent
problem: seven-point hexagon/center equilateral triangles number8 rather
than12; the harmonic-mean problem gives488959/3465 rather than10/11; and
3+5+... over12 rows totals168, not the stated169. Missing-diagram, ambiguous
wording and last-subquestion-only references remain separate review categories.
Do not extrapolate a global error rate from this small sample.

The actually consumed update11 group `bfae52ac-fd6c-4795-8d74-4f566f5c8ac4`
(source18660) supplies a concrete answer-type mismatch: the prompt asserts
that -pi/3 and pi/3 have the same terminal side, while its reference is5pi/3.
All four responses correctly reject the false statement with a final `False`,
receive score0 and have exactly zero valid-token advantages, verified against
the actual actor10/behavior10 input/forward hashes. The reference is a valid
coterminal angle for a different find-an-angle task. This proves mis-scoring
in a real consumed group, but correcting all four rewards from0 to1 would
still leave group-relative advantages at zero. It does not prove that the
reference mismatch caused the zero task gradient, harmful parameter updates
or the whole reward curve. All proofs, source joins and unapplied
correction/exclusion candidates are in
`outputs/reward_reference_audit_20261028_1502/`.

Accordingly, existing fixed-set "accuracy" values must be read as agreement
with the frozen references until independently audited. They alone cannot
establish true mathematical improvement or authorize36-hour stability.
The previously demonstrated FP32-master and FP32-SSM numerical defects remain
separate confirmed findings; label defects do not by themselves explain the
old near-zero collapse. Current Async50/Sync50 inputs and18 common-evaluation
pins remain frozen to preserve the numerical comparison. The independent
reference-quality follow-up is explicitly unresolved in the campaign and
conditional stability authorization.

For supplementary evaluation, official `HuggingFaceH4/MATH-500` revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` is downloaded under
`/opt/huawei/dataset/omni_training/datasets/MATH-500`. All500 rows have worked
solutions. One exact and four whitespace/LaTeX-delimiter-normalized overlaps
with current training are found; excluding the four normalized overlaps
leaves496 candidate questions, with zero such overlap with the current test
split. Both full500 and filtered496 parquet files retain source hashes and
provenance. This is preparation only: no model inference has run, worked
solutions are not proof that every reference is correct, and semantic
paraphrase overlap/pretraining contamination remain unverified. It does not
silently replace the frozen common512 protocol.

The official MATH-500 source-solution verifier preflight then passes500/500
worked-solution final answers against their references using two bounded CPU
workers and the unchanged current scorer. This checks source/parser
consistency, not every solution's mathematical validity or arbitrary generated
boxed-prose handling. Its report is
`datasets/MATH-500/source_solution_verifier_preflight.json` under the shared
dataset root; no model inference is executed.

A separately versioned candidate dataset is prepared at
`datasets/deepscaler/reasoning_known_reference_fixes_v2/`: two exactly audited
training references are corrected, three known inconsistent/missing-diagram/
prompt-reference-mismatch training questions are excluded, and the seven
scalar plus one incomplete held-out references are corrected. It contains
38,442 training and512 test questions, retains exact train/test disjointness,
and records every change and output hash. Source parquet hashes remain
unchanged; this candidate is not connected to the running pair, accepted for
36-hour training or claimed to be a complete corpus-wide audit. Other
ambiguous/multipart cases and boxed-prose scoring remain unresolved. The
producer, candidate manifest and source-solution preflight are linked in
`campaign.json`'s `reward_reference_quality_followup`.

Synchronous update16 completes at44/64 reward, rolling128-group mean0.72265625,
finite gradient norm0.000181330 and zero lag; actor15/behavior15 RMS is0.0256159.
All1,024 selected forwards across16 updates match. This batch has11 all-correct,
5 all-wrong and0 mixed groups, with all888,520 valid-token advantages exactly
zero and task policy loss0. All64 decoded outputs are distinct within their
four-response groups, so this is not exact output duplication. The remaining
gradient can come from the KL term; optimizer momentum/decay may still change
parameters. A scratch CPU call to the actual configured GRPO estimator shows
both[0,0,0,0] and[1,1,1,1] produce zero advantages, while[0,0,1,1] produces
nonzero contrast. Thus uniformly correcting scores cannot alone restore task
gradient in this batch.

All five zero-score update16 groups also contain confirmed reference/schema
defects: a multi-part arithmetic-sequence answer is compared only with26;
the60-foot pen reference450 exceeds even the225-square-foot upper bound
without its divider (the stated divided-square construction gives7200/49);
the potato problem yields19 rather than23; the two-line probability rounds
to0.19 rather than0.16; and the alternating factorial sum is negative despite
a positive reference. The pen outputs round their final answer and the four
factorial outputs truncate, so reference correction is not asserted to make
all these actual outputs correct. Exact rational witnesses, original-source
joins and20 real input/forward packet hashes are retained in
`update16_zero_signal_and_reference_audit.json` under the reference audit
directory. These four newly confirmed scalar defects are not in candidatev2;
it therefore remains a partial, unaccepted repair. All506 production-source
and18 common-evaluation hashes are reverified unchanged, and the original
job continues with update17 sampling.

### Supplementary MATH-500 evaluation prepared without changing the pair

All496 supplementary prompts are built on CPU through the production
RLHFDataset and QwenVLContinuousTokenBuilder; none exceeds857 tokens and no
question is filtered by the2048-token limit. Both currently available actual
exports (initial and Async50) reproduce all496 prompt IDs, references and
decoded inputs using their own processors/tokenizers. The Sync50 export
does not yet exist and its preflight remains required after training.

The separate supplementary driver imports the exact unchanged `Evaluation`
class from the pinned original evaluator. It retains greedy32K decoding,
TP2, two strided replicas per model, FP32 SSM/BF16 convolution caches, native
normalization, memory0.55 and the same stop/banned-token handling. The final
three models reuse the original common512 case file and fresh CPU checkpoint
exports, and their760 tensor payloads must again match the retained live
publication manifests before any new NPU actor is created. Results contain
all496 questions and paired initial/Async50/Sync50 comparisons with intervals;
they supplement, rather than silently replace, the original512 outputs.

Twenty protocol checks pass, covering full1/2/4-replica assignment, synthetic
three-case merge/statistics, tampered/missing/duplicate data, mismatched
parameter samples, job identity and live-training rejection. Four mocked
observer-recovery checks confirm same-job continuation after success, SDK
retry, failed job and an uncertain/missing job handle, with zero resubmissions.
The synthetic fixture scores are explicitly not model results. A real second
observer start is rejected by the process lock without altering the active
observer or submitting an evaluation. All537 pinned files (including506
production sources and the original18 evaluation inputs) are reverified.

The actual detached supplementary observer is PID1497489, currently waiting
for both original50-step training jobs to succeed. It waits until the original
head-node512 evaluation has passed its all32-free check and reached execution
before permitting a separate12-NPU evaluation on172.16.9.13; the original
evaluation uses12 NPUs on172.16.18.136. Execution also requires no training TQ
and at least12 NPUs/48 CPUs available on the peer node. No new inference job
or NPU actor has been launched at preparation time. Source copies, exact
prompts, final CPU preflight, protocol/recovery checks, immutable input pins
and launch evidence are retained in
`outputs/comparison_32npu_20261024_125225/supplementary_math500_evaluation/`.
Its `continuation_status.json` and existing Ray job SDK status remain
authoritative. Neither this preparation nor later job success automatically
accepts learning, accuracy alignment, reference quality or36-hour stability.

### Update17, answer-schema task penalties and source-backed candidates

Synchronous update17 completes at56/64 reward, rolling128-group mean0.734375,
finite gradient norm0.1258425 and zero lag. Actor16/behavior16 RMS is0.0234139;
there are11 all-correct,1 all-wrong and4 mixed groups with327,074 active-
advantage tokens. Only2 responses hit the budget,19 migrate, and all12
positive thinking-close/12 positive end-marker task derivatives remain
active. Across17 updates all1,088 selected forwards match; positive markers
clipped remain1/115 thinking-close and0/114 end markers. This batch has
effective task contrast; its high raw score is not itself a fixed-set learning
or accuracy-alignment result. The original job proceeds to update18.

An actual mixed-group regression answer
`a2140ffa-896d-4dbc-9a32-6cb50330510e_0_0` gives all requested mathematical
results correctly: slope3/10, intercept-2/5, positive correlation and the
prediction17/10. Its complete multi-part final box is compared with the
last-result-only reference1.7 and scored0 while three sibling answers score1.
The actual advantage is-1.499997 and the locally normalized chosen-EOS
task-loss derivative is+0.000165173, a local gradient-descent penalty.
Recognizing all four mathematically correct contents in the offline label
counterfactual yields homogeneous rewards and zero task penalty; it does not
create positive contrast or prove full-parameter/optimizer harm or future
learning improvement. A separate geometry answer in the same batch really
contains backspace token196 followed by `oxed{24}`: preserved raw tokens
decode exactly to the logged text. Its missing valid box is a format failure,
not a proven bad reference or serialization bug. Evidence is retained in
`update17_mixed_answer_schema_audit.json` and
`safe_unboxing_candidate_check.json` under the reference audit directory.

All14 confirmed numeric-reference/constraint issues audited so far lie in
source rows with no worked solution; this selected observation is not a
corpus-wide error-rate estimate and does not invalidate all such rows.
A two-worker CPU consistency screen of all7,025 training rows with source
solutions finds2,184 direct reference matches,1,848 disagreements requiring
review,118 parser-unsupported cases,2,874 sources without a usable final box,
and1 valid `\\fbox{363}` that triggers the legacy unboxing assertion. Many
nonmatches are source-format or choice-letter cases, not demonstrated
mathematical errors. The initial run aborts on this final assertion; its
status/source are preserved. The format-guarded rerun records the case and
completes all7,025 rows at
`outputs/worked_reference_screen_20261028_1610_format_guard/`.

An unapplied minimal extractor patch supports the already-recognized fbox
wrapper and turns malformed unboxing assertions into a format diagnostic.
It preserves malformed-control-character rejection and changes no other
extraction across13,233 source/archived-output examples:13,232 old
nonexception results match, and the one old fbox exception becomes363.
Fourteen boundary checks and all30 existing reward regressions pass against
the proposed function in an isolated CPU test process. The first existing-
suite attempt uses a stdin entry and fails two spawn-dependent cases; the
proper-file rerun passes all30 and both sets of artifacts are retained.
The production source remains unchanged under the original pair and537
supplementary pins. The reviewable patch is
`outputs/reward_reference_audit_20261028_1502/safe_unboxing_candidate.patch`.
This fixes an observed format exception, not incorrect numeric references or
the incomplete multi-part answer schema.

A second, unaccepted data candidate is prepared at
`datasets/deepscaler/reasoning_source_worked_candidate_v1/`:7,021 training
questions retain their original prompts/references and worked-solution hashes,
while4 normalized overlaps with official MATH-500 are excluded. Its500 test
questions preserve official prompts/references. Selection uses source
solution availability and train/test overlap only, not target-model outcomes
or parser-pass screening: all7,021 source-backed rows are retained, including
the reviewed format categories. All currently confirmed numeric/constraint
defects and the observed statement/multipart-mismatch source rows are absent.
This is not a mathematical correctness certificate or a substitute result for
the original full-data pair. New-recipe configuration, data/runtime checks,
actual learning/accuracy validation and stability acceptance remain pending;
current training inputs, source files and supplementary execution pins are
unchanged. Both this candidate and the earlier partial v2 candidate remain
explicitly unaccepted for36-hour training.

At **October28 17:08 UTC**, synchronous update18 is complete and independently
audited:52/64 legacy-reference successes, rolling128-group reward0.7421875,
actor/behavior17/17, zero lag and logprob RMS0.02421997498. Its13 all-correct
and3 all-wrong groups contain no mixed group, so all996,813 valid-token
advantages and the task policy loss are zero. This is another uninformative
GRPO task batch; neither changing all four failures into successes nor the
high raw reward alone establishes a learning signal. The small total gradient
norm0.00016551385 can include the separately configured regularization. Across
18 updates,1,152 selected forwards are matched; positive thinking-close clipping
remains1/115 and positive end-marker clipping0/114. The same SDK job remains
RUNNING and version18 sampling is in progress, without a cluster restart.

The new candidate now passes the actual launcher/configuration comparison:
relative to the completed asynchronous run, only the two data paths differ;
the corresponding candidate synchronous configuration differs only in mode.
All numerical settings,50-step horizon and uncapped staleness are retained.
Full CPU production `RLHFDataset` and `QwenVLContinuousTokenBuilder` checks load
**7,020/7,021 training rows**:one3,321-token prompt is excluded by the unchanged
2,048-token prompt budget. All500 validation questions load, with maximum857
prompt tokens. Retained prompts, references, token IDs and ordering are checked
against the candidate parquet and builder selection. Reports are under
`outputs/source_worked_candidate_config_preflight_20261028_1650/` and
`outputs/source_worked_candidate_dataset_preflight_20261028_1710/`. No NPU was
initialized and no new training was submitted. This validates loading, not
mathematical reference quality or learning. A source-only review also finds
candidate source2467 depends on an absent octagon diagram; source-solution
presence therefore does not resolve every prompt completeness issue. Original
MATH source comparison is the next data-quality check. All537 production and
evaluation input hashes remain unchanged after these CPU checks.

### Original MATH candidate preparation, October28 17:19 UTC

The pinned `DigitalLearningGmbH/MATH-lighteval` dataset revision
`0530c78699ea5e8eb5530600900e1f328b48acad` is downloaded to
`datasets/MATH-lighteval/`, with source URLs, byte counts and SHA256 receipts.
Its7,500 training questions preserve original problem text, including707
Asymptote diagrams. A source-only join also shows why worked-solution presence
was an inadequate provenance filter: only325 of DeepScaleR's7,391 raw worked
rows match this MATH source under whitespace/left-right delimiter normalization,
versus7,640 matches across all40,315 raw DeepScaleR rows. This is a surface-text
join, not a mathematical quality ranking or semantic-overlap analysis.

A separate unaccepted recipe is prepared at
`datasets/MATH-lighteval/reasoning_candidate_v1/`. It retains7,498 unique training
questions and all500 official MATH-500 validation questions, with no overlap
under that normalization. The two removed duplicate questions have independently
verified equal answers: completing squares gives x+y=2, and log_8(2)=1/3.
Two original empty final boxes, at source indices5341 and5343, are explicitly
repaired to0. Both ask for prime counts over sets of composite integers;
the manifest records the divisor proof and independent finite integer checks.
All other training references come from the original source final box. No model
outcome or verifier-pass filtering is used, and no diagram text is stripped.

Real-launcher candidate configurations pass the same comparison as above:
only train/validation paths differ from the completed Async50 configuration;
candidate asynchronous/synchronous configurations differ only in mode. Full CPU
production dataset/token-builder checks load all7,498 training and500 validation
questions without length filtering. Maximum prompt lengths are1,782 and857
tokens, below the unchanged2,048 limit. Prompts/references/token IDs/order match
the prepared source rows. No NPU is initialized. A fixed12-row sample, selected
by SHA256 before solving and independently of model outputs, passes exact
arithmetic, algebra, combinatorial enumeration, geometric and retained-code
witnesses without using source solutions or the production symbolic verifier.
This is evidence for those12 references, not certification of all annotations.

Configuration and loader/sample reports are under
`outputs/canonical_math_candidate_config_preflight_20261028_1715/` and
`outputs/canonical_math_candidate_dataset_preflight_20261028_1716/`, with producer
source copies. This candidate has not been trained; it establishes neither
reward improvement nor asynchronous/synchronous alignment. The scorer unboxing
patch remains unapplied, and the frozen original pair and both queued common
evaluations retain all537 pinned inputs. A fresh SDK check confirms the same
main synchronous job RUNNING, with update19 still collecting behavior18 outputs.

### Post-hoc sensitivity to seven scalar reference errors, October28 17:26 UTC

An offline CPU sensitivity calculation keeps all complete fixed128 outputs and
the unchanged thinking/final-box/verifier policy, replacing only the seven
independently audited wrong scalar references. The incomplete matrix-reference
schema is left unchanged. All105 affected original scores are reproduced
exactly before changing their references, and all15 complete scored validation
files are hashed before/after. Original scores, datasets and running jobs are
unchanged. The seven reference cases were selected after Sync15 failures, so
this is a **post-hoc, partially repaired sensitivity analysis**, not a new primary
endpoint or certification of all128 mathematical answers.

Async corrected counts at0/5/10/15/20/25/30/35/40/45/50 are
**67/69/70/80/76/82/80/89/91/92/90** out of128. Its endpoint change from its own
initial value is+23/128 (+17.96875pp),31 paired gains and8 losses. The conservative
nominal95% interval is[+3.3798,+31.3776]pp, without correction for the post-hoc
selection or interim looks. Sync corrected counts at0/5/10/15 are
**79/71/75/83**, a+4/128 (+3.125pp) change with14 gains and10 losses; its nominal
interval includes declines and gains. Baselines remain different and Sync50 is
pending, so these results do not establish asynchronous/synchronous alignment.

Reference repair exposes two additional Async50 regressions that the old labels
hid: source13448 (correct area48) and12956 (correct conditional permutation730)
were solved initially but end with unfinished repetitive thinking at50.
Previously checked actual serving joins report32,768 tokens, versions50/50 and
one migration for each; decoded-tail repeat fractions are0.98954 and0.97724.
This followup does not inspect raw token packets and does not isolate migration
or SGD causality. These failures show that wrong references do not explain all
remaining truncation/repetition failures after the precision repair.

Reports and reproducible curve artifacts are
`outputs/reward_reference_audit_20261028_1502/known7_scalar_fixed_validation_sensitivity.json`
and `known7_additional_async50_regressions.json`, with producer copies and a
two-panel PNG/SVG curve. All537 frozen execution inputs remain unchanged.

At **October28 17:40 UTC**, synchronous update19 is complete and fully audited:
40/64 legacy-reference successes (0.625), rolling128-group reward0.7265625,
grad norm0.08455751, actor/behavior18/18, zero lag and logprob RMS0.02597965328.
The16 groups consist of9 all-correct,5 all-wrong and2 mixed groups. There are
237,852 nonzero-advantage tokens among1,133,632 valid tokens, with task policy
loss0.01854998. All4 positive-advantage thinking-close markers and all4 positive
end markers retain task gradients. Across19 updates,1,216 selected forwards are
matched, with cumulative positive clipping1/119 for thinking close and0/118 for
the end marker. The complete CPU task loss matches the logged value; all64 actual
serving versions match recorded behavior versions. There are12 budget hits and
18 migrated responses in this batch. These are complete execution/numerical
checks, not post-update held-out improvement or alignment acceptance.
All8 rollout replicas are now on version19 and sampling the next batch. The
30-minute foreground observer expired normally while the separate auditor
finished19; actual SDK status remains RUNNING, and no training is resubmitted.
The next fixed128 evaluation remains scheduled at20. All537 frozen inputs are
rechecked unchanged after the19 audit and reference-sensitivity analysis.

### Candidate submission provenance and SDK checks, October28 17:50 UTC

The private submission helper now accepts an explicit dataset directory and
validates its train/test parquet hashes against the actual manifest before and
after launcher resolution. It archives that manifest separately from historical
DeepScaleR preparation receipts and asserts both resolved data paths match the
declared directory. Matched controls compare the actual dataset manifest as well
as the existing source/configuration checks. `--prepare-only` creates a source
snapshot and launch plan, exits before SDK client construction and leaves the
active-submission file unchanged. This private helper is outside the537 frozen
inputs and does not modify the running pair or either queued evaluation.

A real preparation against the7,498/500 MATH candidate passes. Its planned job
ID is independently absent from the actual Ray Jobs service. A separate guarded
CPU check replaces SDK construction with an immediate failure for preparation;
no client is constructed. A synthetic matched-control fixture then uses a fake
SDK, redirects active-submission writes and verifies source/config/actual-data
parity, canonical dataset selection and the full run-name job identifier. The
real active-submission file is byte-identical before/after. These are submission
boundary checks, **not model execution or learning results**. The fake SDK issues
no real jobs, allocates no NPUs and changes no cluster state. Receipts and before/
after producer copies are under
`outputs/canonical_math_submission_preflight_20261028_1745/`.

### Native publication20 and fixed validation in progress, October28 18:23 UTC

The existing synchronous SDK job remains RUNNING. All64 behavior19 training
responses and all64 actor19 forwards are captured, and native policy20 is
published. The scheduler reports all8 rollout replicas at version20. The existing
CPU archive supervisor starts its version20 child, PID1629630; its actual process
command and source hash are checked. Six of the expected128 actual validation
packets, including behavior logprobs, are already archived without read errors.
No partial validation accuracy is calculated. The publication/archive startup
receipt is `diagnostic_tools/validation20_publication_archive_startup.json`
inside the active synchronous output, and campaign tracking records20 pending.
All537 frozen execution input hashes are reverified unchanged.

The completed-update metric counter still reports19 at this point. Source
inspection confirms `trainer_base.py` performs checkpoint saving, fixed
validation and metric calculation before its final `logger.log` call. Thus the
native optimizer/publication and the complete logged/audited update are distinct
milestones on validation steps. The update20 audit correctly waits for the
complete logged metrics and all128 scored validation outputs. This observation
is not a failed step or a reason to resubmit training. Original SDK identity,
cluster and training configuration are unchanged; no new NPU job is created.

### Complete Sync20/25 followup, October29 01:06–01:27 UTC

The same synchronous job has26 fully audited updates and is sampling27. Complete
fixed128 legacy-reference counts at0/5/10/15/20/25 are
**73/65/69/76/79/73**. At20,17 initial failures become successes and11 initial
successes are lost; the net+6/128 change has a conservative nominal95% interval
[-8.483,+17.540]pp. At25,14 gains and14 losses return the count to its initial
value; its interval is[-13.097,+13.097]pp. This does not establish sustained
synchronous learning or alignment. Every raw packet, decoded prompt/response,
scored output, serving-event version/length and terminal token is checked for
both20 and25; their provenance reports also reverify all128 selected input/
forward hashes and all506 production/18 original evaluation pins. All537 full
execution pins are independently unchanged.

All17 gains at20 and all14 gains at25 were originally unfinished responses.
All14 regressions from20 to25 hit32,768 tokens without a final stop token;
13 remain in thinking and13 have strongly repetitive tails. The remaining
closed-thinking regression runs out of budget while writing its final binomial
expression. Relative to initial0, all14 losses at25 are unfinished/budget-hit,
with12 strongly repetitive tails. This identifies concrete termination/length
failures, without isolating SGD versus serving/migration causality.

The sole closed/EOS initial-to20 legacy regression, source23287, is independently
verified as a two-part reference/schema defect. Its correct answers are
k in(1/12,1/3) and k_max=1/6. For the second part, h=f-g has h(0)=h'(0)=0;
at k=1/6, h''=x(exp(x)-1)>=0, while for k>1/6,
lim[x->0+] h/x^3=1/6-k<0. Exact symbolic identities and the open critical-point
interval are checked. Versions0/5/15/20 contain both correct answers in their
final sections, but20 combines them in the last box and gets0 against the
scalar-only1/6 reference. The other three finish with only that scalar box and
get1. All six archived version scores are reproduced. This is an evaluation
artifact, not a mathematically wrong20 answer or proof of training harm.
The reference/scorer and original scores remain unchanged. The proof is
`outputs/reward_reference_audit_20261028_1502/sync20_closed_multipart_calculus_audit.json`.

The same seven-scalar-reference sensitivity calculation now gives synchronous
counts **79/71/75/83/84/80**; its initial-to25 change is only+1/128 with15 gains
and14 losses. This partial post-hoc correction still does not establish learning;
it deliberately leaves multipart/schema defects unresolved. The reproducible
report/PNG/SVG end in `known7_scalar_fixed_validation_sensitivity_through_sync25`.

Across26 updates,1,664 forwards and416 groups are matched:253 all-correct,
95 all-wrong and68 mixed. Updates11/16/18/24 have no task-advantage contrast;
6,369,260 of27,571,568 valid tokens have nonzero advantage. The mixed-group
fraction68/416 is close to Async50's134/800, so this count alone does not explain
their different native validation traces. Positive thinking-close clipping
remains1/161 and positive end-marker clipping0/160. Same-version actor/behavior
logprob RMS for20–26 stays approximately0.0236–0.0257. These numerical checks
remain healthy while native validation still fluctuates.

### Same-weight baseline repeat prepared and observing the original pair

To measure an actual serving-repeat control before interpreting final common
scores, an extra496-question canonical-initial-model repeat is prepared. It uses
the exact frozen `Evaluation` class,496 prompt IDs, greedy32K settings, two TP2
replicas and the same FP32 SSM/BF16 convolution cache path as the original
supplementary evaluation. Both50-step training SDK jobs must succeed with all50
metrics rows, the original head evaluation must have passed its all32-free gate,
and the supplementary peer evaluation must have passed its execution/resource
gate. Only then can the repeat use4 remaining peer NPUs and16 CPUs. Planned total
evaluation use is12 head+12 peer+4 peer=28 NPUs, within the authorized32.
Before execution, all760 baseline payloads are freshly checked against its
retained publication. Per-replica physical cache dtype checks are inherited from
the unchanged implementation. The completed repeat will be compared with the
original supplementary baseline by exact question/token hashes, paired outcome
flips, unfinished status and identical response-token counts. One repeat is not
the full execution-variance distribution and will not automatically accept
learning/alignment or start stability.

The496-input exported-processor CPU preflight and13 protocol checks pass. Checks
cover the same inference class, balanced-score question flips, duplicate/changed
inputs, changed sampled weights, actual active-primary rejection, reference-job
identity/resource-gate states and no repeated submission after an uncertain SDK
response. Fixtures are explicitly synthetic, not model results. Actual duplicate
observer startup is rejected by its exclusive lock before SDK use. CPU observer
PID2058191 is live in `waiting_for_primary_pair_completion`; **no extra NPU job
is submitted yet**. Its artifacts are under
`outputs/comparison_32npu_20261024_125225/baseline_repeat_evaluation/`.
The original537 pins remain unchanged; two extra repeat sources are separately
pinned. Preserve the shared sources until the original common512, supplementary
496 **and this baseline repeat** finish. Neither original evaluation is replaced
or resubmitted, and the current training/cluster remain unchanged.

At **October29 01:35 UTC**, update27 is fully audited:40/64 legacy-reference
successes, rolling128-group reward0.73046875, grad norm0.07537188, actor/behavior
26/26 with zero lag and logprob RMS0.02498807443. Its9 all-correct,4 all-wrong
and3 mixed groups contain175,570 nonzero-advantage tokens among1,099,615 valid
tokens. All4 positive thinking-close and4 positive EOS markers retain task
gradients. Across27 updates,1,728 forwards are matched; cumulative positive
clipping is1/165 for thinking close and0/164 for EOS. There are10 budget hits and
19 migrations in this batch. The next fixed validation remains30; the latest
complete validation is25 and has not established synchronous learning.
All537 original and two extra repeat-source hashes are unchanged. The three
common/supplementary/repeat CPU observers are alive and waiting for the primary
pair; no additional evaluation NPU job is running yet.

The private async-to-sync continuation also now verifies and inherits a parent
run's actual dataset manifest/parquet hashes, dataset directory and non-mode
overrides. This completes the custom-dataset submission path for a future clean
candidate pair. Its syntax and reviewable diff are recorded alongside the prior
submission preflight; no new training is launched. Existing historical records
without the new actual-dataset metadata retain the original recipe path.

### Unequal step0 baselines remain an open comparison issue

The user explicitly challenges the60/128 versus73/128 pre-training scores.
This is a10.15625pp discrepancy before any SGD update and must not be dismissed
as ordinary noise or attributed to asynchronous learning/staleness. The128
decoded prompt/reference pairs and declared greedy parameters agree; both
initial evaluations have zero migrations and actual begin/end version0. There
are25 outcome flips (19 favor Sync,6 favor Async),24 involving a change in
unfinished status. Only2 complete decoded outputs are identical. Thus the
observed difference is primarily in generation/termination, not merely the
aggregate score calculation. Matching decoded strings does not independently
establish matching actual prompt token arrays.

Sync0 publication payloads are checked against the original/canonical baseline;
a full actual Async0 loaded-weight fingerprint was not captured. The expected
initial raw-token archive directories also do not provide a pair of historical
prompt-token traces. Consequently, current configuration identity and native
version labels do not prove historical backend-state identity. Do not treat
subtracting each run's initial score as a repair of this comparability gap, and
do not infer causal mode superiority/alignment from those native curves alone.

The diagnostic order is actual loaded-parameter fingerprints by TP rank and
exact request token/config capture, followed by repeated zero-update greedy
generation on one fixed TP2 replica. Then locate the first actual output-token
divergence, teacher-force the same prefix and compare logits/top-two margins,
changing batch layout, replica placement and cold/warm prefill/SSM state one
factor at a time. Cross-grade identical outputs to separate scoring from
generation differences. The prepared496 baseline-repeat control measures an
observed repeat under common serving; it does not by itself recover or prove
the missing historical Async0 state. These limits and next diagnostics are
recorded in `initial_validation_reproducibility_followup.json` and campaign
tracking. The current training job and all frozen execution inputs are unchanged.

### Sync30 validation audited; Sync32 training remains active

At 2026-10-29 06:29 UTC, the actual SDK job `reasoning_manifest.json` remains
RUNNING with32 logged and audited updates. Sync32 training reward is47/64
(0.734375), rolling128-group reward0.736328125 and gradient norm0.0434567;
actor/behavior versions31/31 agree. The next training batch is being sampled.

The complete fixed128 validation at30 is85/128 (0.6640625), compared with
73/128 initially and73/128 at25. Relative to initial,19 questions improve and7
regress. All7 regressions reach the32K response budget without a stop token
and without closing thinking;3 have strongly repeated tails. Overall unfinished
responses fall from46 initially to29, with21 strongly repeated tails. These
are original-reference scores, with no retrospective reference corrections.

All128 raw validation packets match the scored outputs. CPU ending-token and
provenance checks additionally reverify actual terminal tokens, all128 training
input/forward hashes, event-log prefixes,506 frozen sources and18 original
evaluation pins. Reports are `validation30_ending_token_audit.json` and
`validation30_audit_provenance.json` in the Sync output. This interim rebound
does not resolve the initial-baseline discrepancy or establish causal mode
alignment or sustained learning. The common512, supplementary496 and initial
weight repeat CPU observers are live and still waiting for the primary pair;
no extra evaluation NPU job was submitted. Current training is unchanged.

### Actual archived prompt tensors checked across Sync5 through30

The CPU diagnostic `check_native_validation_prompt_identity.py` verifies768
actual packets across complete validations5/10/15/20/25/30. All128 questions
have exactly equal prompt token arrays, concatenated input prompt prefixes and
position-ID prompt prefixes across these steps, and match the prepared canonical
512-question input token arrays. Raw packet and scored-output hashes are checked
against each complete archive audit. The archived `loss_mask` is response-only;
these packets do not contain a prompt attention mask. The initial producer's
mislabeling of a response-mask slice is explicitly superseded, retained in
`native_prompt_identity_20261029_0635/SUPERSEDED.json`, and absent from the
corrected full rerun at `native_prompt_identity_20261029_0637`.

The corrected report and producer copy are in that latter comparison directory,
along with all128 matched token inputs exported as `native128_replay_inputs.jsonl`
for a future zero-update reproduction. Original and extra repeat execution pins
remain verified. This establishes post-update Sync input consistency, not the
missing historical Async0/Sync0 request-token identity or initial loaded-weight
identity. No inference, training change or NPU allocation was performed.

### Sync30 completion and reference sensitivity follow-up

Joining complete outputs by decoded input/reference, all19 gains from initial
to Sync30 were initially unfinished. This is evidence that improved completion
contributes to the observed score rebound; it does not isolate SGD from serving.
The pre-existing seven independently audited scalar-reference replacements give
79/128 initially and92/128 at30, versus original73/128 and85/128. All126 affected
archived scores across the two runs reproduce with the unchanged original
scorer before the offline replacements. These seven cases were selected after
earlier failures, so this is a partial post-hoc sensitivity analysis, not a
corrected independent benchmark or a replacement for the primary endpoints.
The new immutable report is
`known7_scalar_fixed_validation_sensitivity_through_sync30.json` under the
reference audit directory. No live references or scores are changed.

The one closed regression versus Sync25 is a genuine arithmetic error, rather
than another reference-schema defect: trial division independently enumerates
168 primes below1000 with sum76127, matching the retained reference, while
Sync30 closes thinking and answers51174. The complete per-step case and exact
integer witness list are retained in
`validation30_completion_and_reference_followup.json` in the Sync run. This
question was unfinished initially, correct at10/15/25 and unfinished at20;
it is not one of the seven initial-to30 regressions. Completion failure does
not explain every intermediate score loss.

### Sync33 completes without a new numerical or freshness failure

At 2026-10-29 06:44 UTC, update33 is logged and audited and the same SDK job
remains RUNNING while sampling34. Reward48/64=0.75, rolling128-group mean
0.73828125, gradient norm0.0992357 and actor/behavior log-probability RMS0.024072.
All64 actual serving versions match recorded behavior32, with trajectory lag0.
The16 groups comprise10 all-correct,3 all-wrong and3 mixed;195,767 of887,888
valid tokens have nonzero advantage. All8 positive thinking-close and8 positive
EOS markers retain task gradients. There are5 response-budget hits and18
migrations. Cumulative33-update coverage is2,112 actual forwards, with positive
clipping1/203 for thinking close and0/202 for EOS. This is continued numerical
health, not an additional held-out endpoint or alignment acceptance. The latest
complete fixed validation remains30 and the initial-baseline issue stays open.

### Full loaded-parameter fingerprint RPC prepared for isolated replay

The private `reproducibility_worker.ReproducibilityWorkerExtension` adds a named
RPC `full_loaded_parameter_fingerprints`. It snapshots every named loaded
parameter including aliases, with shape, original dtype and exact logical-byte
SHA256, plus a stable manifest digest for each TP rank. It inherits the existing
cache metadata/model setup extension and performs no optimizer update. Unlike
the existing sampled-parameter metadata, this covers complete named parameter
payloads. It still requires matching TP topology and names: it is not a direct
canonical-checkpoint-to-TP mapping proof, cache-state fingerprint or physical
storage-layout dump.

Nine CPU protocol checks pass, including independent IEEE F32/BF16 byte witnesses,
single-bit BF16 mutation detection, noncontiguous logical-value equality,
scalar/empty tensors, stable and mutation-sensitive full manifests, aliases and
worker-source parsing without importing an engine. Producer copies and results
are in `loaded_parameter_fingerprint_preflight_20261029_0647`. The actual NPU
worker RPC has not been executed and is not installed into current actors;
this neither recovers historical Async0 state nor establishes weight identity
for an unexecuted replay. All original and repeat execution pins still match.
Collect and compare full loaded manifests before and after a future isolated
zero-update replay; do not change the pinned common benchmark implementation.

### Executable bounded prefix-reproducibility diagnostic prepared

`diagnose_native_prefix_reproducibility.py` now prepares and can execute an
isolated initial-weight TP2 probe after both primary50-step jobs and the three
frozen evaluations succeed, all32 NPUs are free, and no training queue remains.
Its actual execution guard was tested against the RUNNING primary job: the
entrypoint rejects in the primary-success check before creating an execution
directory, calling `ray.init` or creating an NPU actor. This is an expected
diagnostic refusal, not a failure of the training job or an approval rejection.
No additional NPU job is submitted or automatically scheduled.

The planned probe takes the lowest16 original source indices from the matched
native128 token inputs, without reading outputs or rewards. One fixed engine
runs batch1 twice, batch16 once, then batch1 again, with greedy512-token prefixes,
FP32 SSM and BF16 convolution caches. Full loaded TP parameter manifests are
collected before/after. For each question's first observed divergence, it forces
the same exact prefix and generates one step alone and within a16-request batch,
retaining actual token IDs, raw normalized top-logprobs and top-two margins.
The canonical model's760 CPU payloads must be freshly reverified before actors
start; only its own actors are cleaned up. Frozen benchmark implementation and
current training stay unchanged.

The actual128-input preparation, source copies, eight CPU protocol checks,
plan hash and live guard rejection are retained in
`native_prefix_replay_preflight_20261029_0658`. The tests verify token divergence
boundaries, score serialization, real-input outcome-blind selection, mutation
rejection and the incomplete-primary guard. No engine or loaded-NPU RPC has yet
run. This bounded instrumented probe cannot establish full32K accuracy, recover
historical Async0 state, or by itself isolate a kernel or cache-state cause.

### Fixed-validation overlay refreshed through Sync30

`fixed_validation_overlay_20261029_0700` contains the refreshed PNG/SVG/CSV and
hashed complete-output/metric-prefix provenance. Original-reference values are
unchanged: Async30=74/128, Sync30=85/128, Async50=85/128. The descriptive paired
Async30-to-Sync30 comparison has18 gains and7 losses, a net11/128; historical
baseline-serving differences remain unresolved. Async50-to-Sync30 has9 gains
and9 losses, with76 jointly correct and34 jointly wrong. Its zero point estimate
has a conservative95% paired change interval of approximately[-10.94,+10.94]pp.
The equal85/128 counts, unequal training steps and these interim observations
do not establish equivalence or causal mode alignment. The separate right-hand
own-baseline delta plot does not repair the unequal initial baselines. Exact
paired results and their limits are retained in `interim_paired_comparison.json`.

### Completed Async50 residual thinking-close clipping independently checked

The full selected-forward analysis covers all50 actor versions and3,200
responses. Cumulative positive EOS task-gradient clipping is0/268; positive
thinking-close clipping is15/271. All15 thinking-close cases are independently
recomputed on CPU from actual archived inputs and pre-update actor forwards,
with prompt/response indexing, input identity, selected behavior versions,
advantages and real vanilla PPO autograd checked. Every case has positive
advantage and a chosen-token actor/behavior probability ratio above the
configured1.2 upper clipping threshold, and every case has behavior lag>=3.
This matches the configured surrogate; lag association does not establish a
causal accuracy effect, and no staleness cap or clipping exception is added.

Across all50 updates the aggregate actor/behavior RMS ranges0.02313–0.02815.
First10 and last10 selected-batch training rewards are0.678125 and0.6796875,
respectively, so this comparison does not establish a sustained training-reward
rise. Original fixed128 validation increases60->85, with31 gains all initially
unfinished and6 losses all finally unfinished, but baseline-serving
comparability and independent common-serving learning remain unresolved.
The report `async50_residual_termination_clipping_audit.json` and producer copy
are retained in the Async run; all frozen execution inputs still match and the
live Sync job is unchanged. Do not interpret absence of EOS clipping, normal
PPO clipping, or these native scores as full goal acceptance.

### Sync34 completes; next fixed validation remains35

At 2026-10-29 07:24 UTC, the same SDK job is RUNNING with34 logged and audited
updates. Update34 reward is52/64=0.8125 and the rolling128-group mean0.73046875;
gradient norm0.1070755 and actor/behavior RMS0.0243898 remain finite. All64
serving versions match behavior33 with lag0. The16 groups comprise12 all-correct,
2 all-wrong and2 mixed;235,500 of985,574 valid tokens have nonzero advantage.
All4 positive thinking-close and4 positive EOS markers retain task gradients.
There are4 budget hits and19 migrated training responses. The full128 archived
input/forward hashes are reverified after the automatic audit.

Cumulative34-update coverage is2,176 forwards, with positive clipping1/207 for
thinking close and0/206 for EOS. These numerical observations do not supply a
new validation endpoint. The latest complete fixed128 result remains Sync30=85;
validation35, Sync50/common-serving alignment and the initial-baseline root
cause remain open. Training has not been interrupted and Ray has not restarted.

### Startup-safe owner-bound initial token archive prepared for future runs

The existing validation archiver assumes `metrics.jsonl` already exists; its
direct read raises before a fresh run has written any metrics. A new independent
`archive_initial_validation_inputs.py` tolerates the absent/incomplete startup
file, waits for a RUNNING SDK driver and a live TransferQueue controller owned
by that exact Ray driver job ID, and attaches only to the existing queue. It
copies complete version0 CPU packets without deleting entries or creating a
queue. Controller identity changes stop further reads; partial archives remain
explicit, and invariant failures do not become successful captures. It does
not patch the old helper or current archive workers.

Eight CPU boundary checks pass. A synthetic complete-loop test also verifies
that an unrelated live controller is never read, one partial batch can be
archived while metrics remain absent, and all final toy-packet hashes match.
Its two packets and fake driver are conspicuously marked CPU_PROTOCOL_ONLY;
they are not model evidence. A separate read-only live query confirms the
current SDK driver and TransferQueue controller owner agree, without attaching
an archiver or calling `ray.init`. Results and producer copies are in
`initial_archive_startup_preflight_20261029_0745`. No actual initial capture is
started, no historical initial tokens are recovered and all execution pins
remain unchanged.

For any fresh native training pair, launch this private collector with
`--submission <new-run>/submission.json` immediately after SDK submission so it
can wait through startup and retain initial token IDs, positions and behavior
logprobs before validation cleanup. It is not automatically wired into the
submission helper. Future orchestration must explicitly launch and verify it;
do not assume a prepared helper has captured a real baseline. Loaded-parameter
identity requires its separate NPU diagnostic evidence.

### Sync publication35 and fixed128 validation now active

At 2026-10-29 07:57 UTC, the actual SDK job remains RUNNING, publication35 is
observed, and the existing initial-to-final archive parent has started its
version35 child (PID2438060). At least one actual validation35 packet is already
archived with behavior logprobs, and the complete-result auditor is alive. The
checkpoint35 directory is present; this is not a completed integrity receipt.
The latest logged/audited training update remains34 because the trainer saves
and validates before logging the corresponding training metrics. Do not restart
or call this a stalled/failed35 update because that counter remains34 during
the ongoing validation. No partial validation score is calculated. Latest
complete fixed128 remains Sync30=85/128, with baseline comparability and the
final/common-serving endpoints still outstanding.

### Installed Ascend batch-invariance capability and current node source parity

The installed Ascend backend has its own `VLLM_BATCH_INVARIANT` implementation.
A guarded CPU import succeeds with HAS_TRITON and HAS_ASCENDC_BATCH_INVARIANT
both true, without initializing an NPU or registering the invariant dispatch
library. Its source initializes the mode in worker distributed setup and can
replace matmul, add/RMS normalization and attention operations while setting
deterministic collective options. Actual startup log prefixes confirm that both
primary runs recorded VLLM_BATCH_INVARIANT=0 and VERL_FULL_DETERMINISM=0. This is
available for a later isolated ablation, not a current settings change or proof
that all Qwen3.5 operators become invariant.

Qwen3.5 uses dedicated GDN/causal-convolution operators. The inspected fused
GDN prefill helper explicitly casts incoming state to BF16 and is selected only
when its device smoke probe succeeds and PCP world size is1; the alternate
Triton path preserves the configured state dtype. The actual current fused-path
decision has not been observed. No operator probe or branch change is executed,
and this source fact is not asserted as the initial-score or learning cause.
Future runtime-path inspection must read the existing decision without invoking
an additional device smoke probe during live training. The separate training
consistency flag found in source gates FA3 selection and is not evidence of a
general GDN precision repair.

One pure file-hashing CPU task on each live Ray node compares4,599 Python files
(vLLM2,239; Ascend616; batch-invariant ops13; torch_npu1,559; Triton172), the
targeted `libcust_opapi.so` and distribution-version metadata. All compared
payloads match; neither task imports Torch, loads a model or requests NPU
resources. This is current targeted source/library parity, not every CANN/driver
binary, historical cross-run parity or actual loaded-worker identity.
Reports and source snapshots are retained in
`ascend_batch_invariant_capability_20261029_0840` and
`runtime_source_parity_20261029_0845`. All frozen execution pins still match.

### Sync35 complete fixed validation and training audit

Sync35 original-reference score is83/128 (0.6484375), versus73 initially and85
at30. It has22 gains and12 losses against initial, net+10; all22 gains were
initially unfinished. Against30 there are7 gains and9 losses, net-2. Overall32
responses remain unfinished and reach32K, none accepted;22 have strongly repeated
tails and51 responses migrate. Raw/scored output identity, all128 packet hashes,
all128 training input/forward hashes, native event prefixes,506 frozen source
files and18 original evaluation pins pass the complete provenance check.
Reports are `validation35_ending_token_audit.json` and
`validation35_audit_provenance.json`. This remains an interim observation with
unresolved baseline comparability and no final50/common-serving acceptance.

Of the12 initial regressions,11 exhaust32K without closing thinking and one is
a new multipart reference-schema case, DeepScaleR source23128. An independent
exact witness gives p=-4, parabola y²=-8x, directrix x=2, and focal chord32/3 for
the intended part2 reading. Sync35 supplies all requested answers in its final
box, whereas the scalar reference contains only32/3; its recorded zero score
reproduces, and comparing the correct scalar alone passes. The prompt reuses
the endpoint label A between parts; the mathematical witness explicitly states
the conventional intended focal-chord reading. This is a held-out case, not a
training-gradient harm proof. The report and producer copy are
`sync35_closed_multipart_parabola_audit.json` in the reference audit directory.
No official score, live reference or generic scorer is modified.

The pre-existing known-seven scalar sensitivity extends through35 with133
affected old scores exactly reproduced. Its Sync counts are79 initially,92 at30
and89 at35; the new multipart case is left outside these seven corrections.
This is partial post-hoc sensitivity rather than an independent primary score.
The original-reference fixed plot is refreshed in
`fixed_validation_overlay_20261029_0853`; native matched35 counts are Async82
and Sync83, which alone does not establish mode equivalence or learning.

Update35 training reward is46/64=0.71875, rolling128-group mean0.7421875,
gradient norm0.0756592 and actor/behavior RMS0.0246455. All64 serving versions
match behavior34 with lag0;11 groups are all-correct,3 all-wrong and2 mixed.
There are88,398 nonzero-advantage tokens among638,404, no response-budget hits,
and21 migrated training responses. Both positive thinking-close and both
positive EOS markers retain gradients. Cumulative35-update coverage is2,240
forwards, with positive clipping1/209 for thinking close and0/208 for EOS.
The actual SDK job continues RUNNING, sampling36. The read-only observation
for35 finishes successfully; no training restart or Ray restart occurs.

### Prefix diagnostic version2 verifies actual worker mode and cached paths

The private prefix diagnostic now has explicit baseline0 and optional
batch-invariant1 plans. The active immutable plans are
`native_prefix_replay_v2_baseline_final_20261029_0920/evaluation_plan.json` and
`native_prefix_replay_v2_batch_invariant_final_20261029_0920/evaluation_plan.json`.
Their16 input records, canonical weights, code/source pins and other settings
are equal; only requested mode, preparation time and execution output differ.
Run baseline0 first after the primary pair and all three frozen evaluations
finish; use mode1 only if needed to investigate observed variation. Neither has
been submitted or executed on NPUs. Version1 and intermediate preparation
directories carry explicit SUPERSEDED pointers; their historical CPU evidence
and source copies are retained.

Mode is set before vLLM imports and inherited by spawned TP workers. Named
runtime metadata RPCs run before and after the diagnostic, retaining visible
device IDs, local ranks, actual invariant dispatch registration, deterministic
collective settings and GDN's existing cached fused-op availability decision.
They read already-loaded modules and do not call a device availability probe.
The actual metadata is saved before validation, so an ignored/unsupported mode
cannot silently pass as an enabled experiment. Six CPU protocol checks verify
all cached GDN states without invoking its probe, both-mode validation,
rejection of inactive/misconfigured runtime state, real-input plan parity and
all new source pins. Actual entrypoint tests for both plans reject against the
RUNNING primary job before `ray.init` or NPU actors. Original537 plus two repeat
source pins remain unchanged.

Full per-TP parameter manifests are still collected before/after each real
execution. Compare them across mode0 and mode1 as well before attributing any
observed difference to execution mode. Cached GDN availability is not an
operator-call trace, and configured deterministic mode is not itself proof of
full32K Qwen3.5 reproducibility or accuracy. Current training and original
common benchmark implementations are unchanged.

### Sync36 has zero GRPO task advantage despite distinct sibling responses

Update36 completes and is audited while the same SDK job continues RUNNING
into sampling37. Training reward44/64=0.6875, rolling128-group mean0.7421875,
gradient norm0.000341066 and actor/behavior RMS0.025778 remain finite. All64
actual serving versions match behavior35 with lag0. There are11 budget hits
and20 migrated training responses. All1,016,639 valid tokens have zero
advantage: the16 groups comprise11 all-correct and5 all-wrong, with no mixed
group. The recorded policy loss is0; this update does not provide a GRPO task
policy gradient and must not be counted as evidence of task learning.

A CPU recheck of all64 actual selected packets confirms exactly16 distinct
prompts, four identical prompt token arrays within each group, and four distinct
response token arrays within each group. Each group's four binary rewards are
constant and all valid advantages are exactly zero. All128 original archived
input/forward hashes are reverified. This matches the group-relative objective
and is not attributed to duplicated sibling sampling or a new gradient-precision
failure. The immutable report is `update36_zero_advantage_group_audit.json`.
Zero-advantage Sync updates through36 are11/16/18/24/36; the completed Async50
has such updates1 and24. These use different selected questions and are not an
isolated causal comparison of modes.

Cumulative Sync coverage is2,304 forwards. Positive ending-token clipping
counts remain1/209 thinking-close and0/208 EOS because this batch has no
nonzero-advantage tokens. Latest full fixed128 remains83/128 at35; full Sync50,
uniform-serving comparison, initial-state diagnosis and the conditional36-hour
run are still outstanding. No training or Ray restart occurs.

### Completed Async50: full behavior cohorts versus consumed training batches

The ended asynchronous SDK job is rechecked SUCCEEDED. A streaming CPU rerun
of the original complete-cohort analysis preserves the earlier report and its
event-prefix proofs. All960 previously reported response records match their
request IDs, original rewards, lengths and migration counts; existing selected
steps remain identical. New results and source copies are under
`outputs/comparison_32npu_20261024_125225/async50_closed_behavior_cohorts_20261029_1006/`.

All issued training requests at behavior versions0–49 have ended, have an
identified partition and have original scores archived; all eight engines have
moved beyond these versions. The complete cohorts contain3,456 responses,
2,404 correct (0.695602), including584 response-budget hits. Their first ten
behavior versions contain565/855 correct (0.660819), and their last ten
contain456/640 (0.7125). These are response-weighted descriptive means on
different questions, with correlated siblings; they do not establish a causal
learning effect or sustained improvement. The128 requests at version50 are
excluded under the original minimum-serving-version closure rule.

An independent cross-check matches every one of the3,200 actual consumed
responses to the complete-cohort record by key, behavior version, actor update,
original reward and budget status. Exactly256 completed responses were not
selected for training; all256 are in the last ten behavior versions. Thus the
complete-generation means must not be presented as consumed-training reward
or as an independent held-out benchmark. The previously reported near-flat
first/last-ten *actor-update* training means remain unchanged. All3,456 packet
files (4,250,810,316 bytes) now have SHA256 receipts in
`raw_archive_provenance.json`. No rescoring, NPU inference or source/config
change is involved. The updated statistics reduce completion censoring and
clarify the reward-curve interpretation, but do not close initial-version
reproducibility, final uniform-serving alignment or the36-hour acceptance gate.

The inspected PNG/SVG `consumed_vs_complete_behavior_reward` uses separate
actor-update and behavior-version axes and includes the original unsmoothed
points, an8-index response-weighted rolling mean and exact CSV/source receipts.
It labels the256 unused responses and the different-question/correlated-sibling
limitation explicitly. Consumed first/last-ten means remain0.678125/0.6796875;
the complete-generation curve does not replace that flat training statistic.

The complete train event log contains exactly3,200 selection events at steps1–50
and no selections at later steps; all256 unused keys are absent from those
events. Within behavior versions40–49 the384 consumed responses score271/384
(0.705729), and the256 unconsumed responses score185/256 (0.722656). These are
behavior-cohort subsets, not the last ten actor-update batches. Receipts are in
`cohort_consumption_summary.json`. After this CPU-only work, all539 frozen
benchmark/repeat input hashes and both active prefix plans'14 source pins
remain unchanged (`input_pin_integrity_after_cpu_analysis.json`).

### Sync37 restores a small mixed-group task signal; main job continues

The same actual SDK job remains RUNNING, publishes37 and starts sampling38.
Its37th completed update scores39/64=0.609375, with rolling128-group reward
0.71484375 and gradient norm0.029349817. The16 groups contain9 all-correct,
6 all-wrong and1 mixed group (three correct siblings and one incorrect).
There are38,109 nonzero-advantage tokens among1,029,602 valid tokens, so this
update provides a task signal unlike the preceding all-zero-advantage batch.
Seven active-token policy derivatives are zero; all three positive-advantage
thinking-close tokens and all three positive EOS tokens retain their gradients.

All64 actual request-begin/end versions match recorded behavior36 and actor36,
with lag0. Actor/behavior RMS is0.02511135; there are11 budget hits and17
migrated responses. Two immutable diagnosis/snapshot artifacts, all128 archived
input/forward file SHAs and all three event/metrics prefix hashes are separately
reverified after the automatic audit. Cumulative selected-forward coverage is
2,368; positive ending-token gradient-zero counts are1/212 thinking-close and
0/211 EOS. The latest full fixed128 validation remains83/128 at35; update37
has no scheduled fixed validation. These checks do not establish sustained
reward improvement, final mode alignment or explain the initial score gap.
No source/config change, training interruption or Ray restart occurs.

### Current job's mapped Ascend library names and filesystem bytes agree

A read-only inspection binds to the actual RUNNING SDK job's driver31010000,
its16 live training actors and the16 TP worker descendants of its eight live
vLLM servers. Process start identities remain stable throughout. Each node
uses one CPU-only task, has no NPU/GPU resources assigned and does not import
Torch. There is no worker RPC injection, model inference, cluster operation
or environment change. Reports and all32 targeted process-map snapshots are in
`outputs/comparison_32npu_20261024_125225/loaded_ascend_library_parity_v2_20261029_1050/`.

The union of mapped Ascend/torch_npu/vllm_ascend/custom_transformer shared-library
names is the same136 paths on both nodes. Their current filesystem bytes total
1,039,991,445 bytes per node, and all136 SHA256 values agree. There are no
node-only paths. All17,536 retained mapping rows have the same inode as the
corresponding current path. Saved source and all32 map hashes are separately
reverified. This adds actual-process path evidence to the earlier Python
source parity check and finds no current installed-file mismatch in this scope.

The initial diagnostic mistakenly required the mapping device number to equal
`stat(path).st_dev`. A head control shows backing devicefd:00 versus container
path device00:de with the same inode; the initial rejection is retained under
`loaded_ascend_library_parity_20261029_1045/FAILED_PRECHECK.json`. The corrected
inspection records device differences explicitly (8,672 rows on each node)
rather than calling them a training failure. Both nodes' direct `map_files`
readability probes return EPERM. Consequently the SHA comparison covers current
filesystem-path bytes, not actual mapped backing-file or in-memory bytes.
This is not complete driver/hardware equality, a device-kernel call trace,
historical Async0 weight/runtime identity or an explanation of the initial
score gap. The main training remains RUNNING; all acceptance gates stay false.

### Sync38 completes with two mixed groups; no convergence claim from a high batch

The same primary SDK job completes38, publishes version38 and continues sampling39.
Reward58/64=0.90625 follows update37's39/64; rolling128-group mean is0.7421875
and gradient norm0.08291403. The16 groups comprise13 all-correct,1 all-wrong
and2 mixed. There are108,785 nonzero-advantage tokens among784,860 valid tokens.
All six positive-advantage thinking-close tokens and all six positive EOS tokens
retain their policy gradients.36 other active-token derivatives are zero
(0.00033093 of active tokens). The high batch reward uses different selected
questions and is not proof of convergence or an isolated effect of this update.

All64 actual serving versions match behavior37 and preupdate actor37 with lag0.
Actor/behavior RMS is0.02406532. There are5 budget hits and20 migrated responses.
After the automatic audit, both static diagnosis/snapshot artifacts, all128
archived input/forward file hashes and all three event/metrics prefix hashes
are independently reverified. Cumulative selected-forward coverage is2,432;
positive ending-token gradient-zero counts are1/218 thinking-close and0/217 EOS.
Latest complete fixed128 validation remains83/128 at35; fixed40 is still
pending. Final Sync50/common-serving comparison, the initial score discrepancy
and conditional36-hour acceptance remain unresolved. No training/config change
or Ray restart is made.

### Sync39: one active group, with an independent exact reward witness

The same primary job completes39, publishes39 and starts sampling40. The
batch scores43/64=0.671875, rolling128-group reward0.736328125 and gradient
norm0.07567284. Groups are10 all-correct,5 all-wrong and1 mixed. All64 actual
serving versions match behavior38/preupdate actor38 at lag0. Actor/behavior
RMS is0.02503788, with12 budget hits and19 migrations. All three positive
thinking-close and all three positive EOS tokens retain gradients.41 active
token derivatives are zero among104,705 active tokens out of960,909 valid
tokens. Both static diagnosis/snapshot SHAs, all128 archived input/forward SHAs
and all three event/metrics prefixes are separately reverified. Cumulative
coverage is2,496 forwards, with positive ending-token gradient-zero counts
1/221 thinking-close and0/220 EOS. Fixed40 remains pending; a high or low
selected training batch is not taken as convergence evidence.

The sole mixed group accounts for every one of this update's104,705 task-
advantage tokens. Its four actual prompts are identical and match raw source
index13832: a cyclic quadrilateral pyramid with base area36, successive side
face areas9/9/27/27 and equal base-edge dihedral angles. The *actual training
parquet*, verified against its launch reasoning manifest, also uses reference54.
An independent exact SymPy coordinate construction satisfies all four side
areas, cyclicity, base area and equal interior dihedral cosines1/2, and gives
height9/2 and volume54. Uniqueness follows from common slant height giving base
sidesx/x/3x/3x; SSS and cyclicity make two opposite angles right, so base
area3x²=36, r=36/(4x), l=18/x and h=9/2.

The original extraction/verifier functions reproduce all four archived scores:
siblings0/1/3 finish with54 and score1; sibling2 reaches32,768 tokens with no
thinking close and scores0 as unfinished. Their valid advantages are constant
approximately+0.499999 for the three correct responses and−1.499997 for the
unfinished response. All eight corresponding input/forward file SHAs match
the original update provenance. This is a case where the active task gradient
has valid reference and completion feedback; it is not another closed-correct
false penalty. The source happens to have no worked solution, so that absence
alone must not be treated as evidence that its reference is wrong.

The immutable CPU-only witness is
`outputs/reward_reference_audit_20261028_1502/sync39_active_group_pyramid_audit.json`,
with its executed producer source alongside it. It does not certify other
training references, undo the previously proved false penalties, establish
full asynchronous callback timing, isolate learning causality or accept final
mode alignment. Original reward/reference data and training configuration are
unchanged; no Ray restart occurs.

The companion `sync39_active_group_pyramid_audit_provenance.json` rechecks all
eight case-input hashes, all eight input/forward hashes, the unchanged frozen
production reward source, all539 benchmark/repeat pins and both active prefix
plans' source pins. The canonical training parquet hash matches the launch
reasoning manifest rather than the older uncorrected source-data manifest.

### Sync40 complete observation; baseline reliability remains unaccepted

The complete original-reference fixed128 score at40 is88/128 (0.6875),
compared with73/128 at this run's initial evaluation. There are20 gains and5
losses; all20 gains were initially unfinished. All five lost initial successes
reach32K without finishing, and three have strongly repetitive tails. Against
35 there are14 gains/9 losses, a net5/128. All nine regressions reach32K and
remain unfinished, seven with strongly repetitive tails. Overall unfinished
responses fall from32 at35 to27 at40; all27 budget hits score0. There are57
migrated validation responses, no gold parse failures or verifier timeouts,
and one answer parse failure. All128 actual serving joins are version40.

Scored/raw prompt and response identity, archived score identity, all128 raw
packet SHAs/ending tokens, all128 training-input/forward SHAs, event prefixes,
the506 non-Markdown production sources and18 original evaluation pins pass
the complete provenance check. The retained reports are
`validation40_ending_token_audit.json` and `validation40_audit_provenance.json`.
Training40 scores51/64, rolling0.744140625, gradient norm0.06110062 and
actor/behavior RMS0.02511760, with12 all-correct/3 all-wrong/1 mixed groups.
All three positive thinking-close/EOS gradients are retained.

The corresponding asynchronous40 score is85/128. Its paired comparison to
Sync40 has12 wins/9 losses, net3/128, conservative95% interval
[-0.0936591,0.1388385] and nominal McNemar p0.663624. This is an interim
original-reference observation, not equivalence or accepted alignment. The
new inspected plot in `fixed_validation_overlay_20261029_1343/` explicitly
marks the curves PROVISIONAL and warns that own-baseline subtraction does not
resolve the initial mismatch. No official scores or references are changed.

Update41 subsequently completes:60/64 reward, rolling0.767578125, gradient
norm0.14156428, actor/behavior RMS0.02507995,14 all-correct/2 mixed groups,
180,612 active tokens out of651,514 valid,3 budget hits and22 migrations.
All four positive thinking-close and all four positive EOS gradients remain.
Both static artifacts,128 input/forward hashes and three event prefixes are
independently reverified. Cumulative coverage through41 is2,624 forwards,
with positive ending-token gradient-zero counts1/228 thinking-close,0/227 EOS.
This numerical observation is also not a learning/mode acceptance claim.

### User priority: diagnose the initial60/128 versus73/128 before endpoint acceptance

The user explicitly requires the initial discrepancy to be investigated after
Sync50, warning that it can invalidate later comparisons. Current native
curves and all future endpoint interpretations therefore remain unaccepted.
At the request, the three still-waiting CPU continuation processes1135896,
1497489 and2058191 are held with SIGSTOP after verifying their exact script
arguments, process start identities, waiting stages and absence of submitted
evaluation jobs. The primary SDK job remains RUNNING. No NPU process, training
job, final-publication export observer or Ray cluster is stopped. The frozen
benchmark inputs and implementations are preserved unchanged.

The immutable hold receipt is
`initial_baseline_priority_20261029/held_evaluation_observers.json`.
**Do not resume these final-evaluation observers merely because training50
or a short prefix diagnostic succeeds.** A complete128-question/32K initial
baseline repeat and root-cause/reliability review are mandatory before release.
If diagnosis requires a corrected evaluation runtime, preserve the old protocol
as superseded and establish a consistent corrected protocol for all models;
do not resume an evaluation protocol shown to be unreliable.

An independent CPU controller now waits on both actual50-step SDK successes,
exact entrypoints and50 logged optimizer updates, unchanged frozen inputs,
all32 NPUs free and absence of the training queue. It then submits one
same-ID-recoverable initial prefix diagnostic before the held evaluations.
The unchanged v2 inference implementation is reused with an explicit new
campaign-order guard in `run_priority_initial_prefix_probe.py`; no old source
or plan is overwritten. Actual entrypoint testing rejects while Sync is
RUNNING before creating an execution directory or reserving NPUs. Four CPU
protocol checks reject wrong process start identity, modified arguments,
removal of the full128 requirement and a false hold stage.

Plans, copied sources, guard evidence and controller status are under
`initial_baseline_priority_20261029/`. The preliminary probe remains16 inputs
and512 tokens, with full loaded-parameter fingerprints and same-prefix
next-token probes; it is only a localization step. The controller cannot
resume the held final evaluations or launch stability training, and ends by
requiring the full native128 repeated evaluation and root review. That full128
execution phase still needs to be implemented/prepared before the preliminary
probe is used to claim any baseline reliability. Historical Async0 full
resident weights/prompt tensors were not preserved; matching model paths or
current library files alone must not be presented as historical identity proof.

The 36-hour stability gate therefore **fails** at this stage; the conditional
long run has not started. The full-model cache comparison is complete; next,
validate the repair through actual asynchronous and synchronous learning and
their fixed held-out evaluations. Do not infer
asynchronous/synchronous accuracy alignment from job success or a bounded
numerical test.

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
