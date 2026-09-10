# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loss-side staleness correction for trajectory-level async RL.

Positioning vs the literature (kept honest — see README "Staleness
correction"):

* The Laminar paper (arXiv 2510.12633) itself derives NO importance-sampling
  bias or convergence bound — its Appendix D analyzes chain-broadcast
  latency, and §8.2 is an empirical comparison. Its recipe is: trajectory
  version-atomicity (one version per trajectory, unlike partial-rollout
  systems that mix versions WITHIN a trajectory), a bounded observed
  staleness (max 4 in their runs) and a larger mini-batch (2048) to
  stabilize off-policy training.
* What the async path ALREADY has: with
  ``algorithm.rollout_correction.bypass_mode=True`` (the fully-async
  default), ``old_log_probs := rollout_log_probs``, so the PPO ratio is
  π_current/π_{v_i} — the per-token cross-version importance ratio, and
  its clipping IS truncated importance sampling (TIS). The correction
  exists; it is version-blind.
* What this module adds — the version-aware layer, grounded in the
  staleness-aware training literature (gap-aware gradient-staleness
  mitigation; SAPipe-style staleness-aware reweighting; truncated-IS
  variance control à la V-trace / the codebase's own
  ``rollout_corr_helper``):

  1. **Staleness reweighting** (default ON when the config block is set):
     per-trajectory weight ``w(age) = 1/(1+λ·age)`` (or ``exp(-λ·age)``),
     self-normalized to mean 1 so the gradient scale is preserved. This
     reweights REPRESENTATION (how much each version cohort contributes
     to the update); it composes with — never double-counts — the
     π-ratio IS inside the PPO clip, because it is a function of version
     distance, not of the ratio.
  2. **Adaptive clipping** (default OFF): per-trajectory clip scaling
     ``ε_i = ε · clamp(1+γ·age, min, max)``. γ>0 widens the trust region
     for older trajectories (counteracts clip saturation silently
     shrinking their gradient); γ<0 tightens it (conservative updates
     from stale data). Both directions appear in the literature; the
     default stays at the stock fixed ε.
  3. **Version-cohort group baseline** (default OFF, experimental): for
     GRPO groups with version_span>0, normalize advantages within
     same-version cohorts instead of across the mixed group. Any
     baseline keeps the IS-weighted estimator unbiased; a cohort
     baseline removes between-version reward drift from the control
     variate — at the cost of discarding the between-cohort comparative
     signal and shrinking the cohort sample size. Empirical tradeoff,
     hence opt-in.
  4. **Cohort diagnostics**: per-age weight/reward statistics and
     version-span summaries (``trajectory_async/stale_*``).

Wiring: :func:`apply_staleness_correction` runs in the trainer's
advantage step (see ``async_trainer.py``), reading per-row
``model_version`` stamped by the trajectory-level producer and writing
two batch columns consumed by the loss (``staleness_weights``,
``cliprange_scale``) plus, when enabled, adjusting ``advantages``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class StalenessCorrectionConfig:
    """Config block ``async_training.staleness_correction``.

    Attributes:
        mode: staleness reweighting family — "none", "decay" (1/(1+λ·age))
            or "exp" (exp(-λ·age)).
        lam: decay rate for the reweighting family.
        normalize: self-normalize weights to mean 1 within the batch
            (preserves the gradient scale; the RELATIVE weights carry the
            correction).
        adaptive_clip: enable per-trajectory clip scaling (experimental).
        adaptive_clip_gamma: clip growth per version of age; sign picks the
            direction (positive widens the trust region with age, negative
            tightens it).
        adaptive_clip_max_scale: upper clamp for the clip scale.
        adaptive_clip_min_scale: lower clamp for the clip scale.
        cohort_baseline: normalize GRPO advantages within same-version
            cohorts for mixed-version groups (experimental; see module
            docstring for the variance-vs-signal tradeoff).
    """

    mode: str = "decay"
    lam: float = 0.5
    normalize: bool = True
    adaptive_clip: bool = False
    adaptive_clip_gamma: float = 0.25
    adaptive_clip_max_scale: float = 2.0
    adaptive_clip_min_scale: float = 0.5
    cohort_baseline: bool = False

    @classmethod
    def from_config(cls, cfg: Any) -> "StalenessCorrectionConfig | None":
        """Build from an OmegaConf/dict block; None/empty -> None (disabled)."""
        if cfg is None or isinstance(cfg, cls):
            return cfg if isinstance(cfg, cls) else None
        kwargs = dict(cfg) if hasattr(cfg, "keys") else {}
        if not kwargs:
            return None
        mode = str(kwargs.get("mode", "decay"))
        if mode not in ("none", "decay", "exp"):
            raise ValueError(f"staleness_correction.mode must be none|decay|exp, got {mode!r}")
        clean = {f: kwargs[f] for f in cls.__dataclass_fields__ if f in kwargs}
        return cls(**clean)


# ------------------------------------------------------------- pure math


def staleness_weight(age: int, mode: str = "decay", lam: float = 0.5) -> float:
    """Weight of one trajectory as a function of its version age."""
    if age <= 0:
        return 1.0
    if mode == "none":
        return 1.0
    if mode == "decay":
        return 1.0 / (1.0 + lam * age)
    if mode == "exp":
        return math.exp(-lam * age)
    raise ValueError(f"unknown staleness weight mode {mode!r}")


def staleness_weights(
    ages: Sequence[int],
    mode: str = "decay",
    lam: float = 0.5,
    normalize: bool = True,
) -> list[float]:
    """Per-trajectory staleness weights, optionally self-normalized to
    mean 1 (the relative weights carry the correction; the gradient scale
    is preserved)."""
    raw = [staleness_weight(int(a), mode=mode, lam=lam) for a in ages]
    if not normalize or not raw:
        return raw
    total = sum(raw)
    if total <= 0.0:
        return [1.0] * len(raw)
    return [w * len(raw) / total for w in raw]


def adaptive_clip_scale(
    age: int,
    gamma: float = 0.25,
    max_scale: float = 2.0,
    min_scale: float = 0.5,
) -> float:
    """Per-trajectory PPO clip-range scale. γ>0 widens the trust region
    with age (counteracts clip saturation on stale ratios); γ<0 tightens
    it (conservative updates from stale data)."""
    if age <= 0 or gamma == 0.0:
        return 1.0
    return min(max(1.0 + gamma * age, min_scale), max_scale)


def cohort_advantages(
    versions: Sequence[int],
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> list[float]:
    """GRPO group advantages with version-cohort baselines.

    For a single-version group this is exactly whole-group GRPO
    normalization. For a mixed-version group (version_span > 0), each
    same-version cohort with ≥2 members is normalized by its OWN
    mean/std; singleton trajectories fall back to the whole-group
    statistics (a singleton has no cohort baseline — the group statistic
    is the best available control variate).

    Unbiasedness: any baseline (group, cohort, or otherwise) keeps the
    IS-weighted estimator unbiased — this only changes the variance
    profile (removes between-version reward drift from the control
    variate, at the cost of the between-cohort comparative signal and a
    smaller cohort sample).
    """
    n = len(rewards)
    if n == 0:
        return []
    g_mean = sum(rewards) / n
    g_std = math.sqrt(sum((r - g_mean) ** 2 for r in rewards) / n)
    if g_std < eps:
        g_std = None  # degenerate group

    cohorts: dict[int, list[int]] = {}
    for i, v in enumerate(versions):
        cohorts.setdefault(int(v), []).append(i)

    out = [0.0] * n
    for members in cohorts.values():
        if len(members) >= 2:
            c_rewards = [rewards[i] for i in members]
            c_mean = sum(c_rewards) / len(c_rewards)
            c_std = math.sqrt(sum((r - c_mean) ** 2 for r in c_rewards) / len(c_rewards))
            if c_std >= eps:
                for i in members:
                    out[i] = (rewards[i] - c_mean) / c_std
                continue
        # singleton or degenerate cohort -> whole-group baseline
        for i in members:
            out[i] = 0.0 if g_std is None else (rewards[i] - g_mean) / g_std
    return out


def cohort_stats(ages: Sequence[int], weights: Sequence[float]) -> dict[str, float]:
    """Per-batch version-cohort diagnostics (trajectory_async/stale_*)."""
    stats: dict[str, float] = {}
    if not ages:
        return stats
    by_age: dict[int, int] = {}
    for a in ages:
        by_age[int(a)] = by_age.get(int(a), 0) + 1
    stats["trajectory_async/stale_age_mean"] = sum(ages) / len(ages)
    stats["trajectory_async/stale_age_max"] = max(ages)
    stats["trajectory_async/stale_age_cohorts"] = len(by_age)
    stats["trajectory_async/stale_fresh_frac"] = by_age.get(0, 0) / len(ages)
    if weights:
        stats["trajectory_async/stale_weight_min"] = min(weights)
        stats["trajectory_async/stale_weight_max"] = max(weights)
    return stats


# ------------------------------------------------------ batch application


def apply_staleness_correction(
    batch: Any,
    current_version: int,
    config: StalenessCorrectionConfig,
    metrics_out: dict[str, Any] | None = None,
) -> Any:
    """Attach staleness-correction columns to a training batch (torch path).

    Reads per-row ``model_version`` from ``batch.non_tensor_batch``
    (stamped by the trajectory-level producer; batches without it are
    returned untouched). Writes:

    * ``batch.batch["staleness_weights"]`` — per-token weight column
      (row weight broadcast over the response, masked to 0 on padding),
      consumed by the policy loss after clipping (representation
      reweighting; never double-counts the π-ratio IS);
    * ``batch.batch["cliprange_scale"]`` — per-row scale for the PPO clip
      bounds (only when ``adaptive_clip`` is on);
    * ``batch.batch["advantages"]`` — replaced by cohort-baseline
      advantages when ``cohort_baseline`` is on (mixed-version groups
      only).

    Age is measured against ``current_version`` — the version the actor
    embodies during this update (the trainer publishes
    ``current_param_version + 1`` at the end of the step, so pass
    ``current_param_version + 1``).
    """
    ntb = getattr(batch, "non_tensor_batch", None) or {}
    if "model_version" not in ntb or len(ntb["model_version"]) == 0:
        return batch  # not a version-stamped (trajectory-level) batch

    ages = [max(0, int(current_version) - int(v)) for v in ntb["model_version"]]
    n = len(ages)
    if n == 0:
        return batch

    weights = staleness_weights(ages, mode=config.mode, lam=config.lam, normalize=config.normalize)

    import torch  # lazy: cluster-only path

    response_mask = batch.batch["response_mask"]
    weight_col = torch.tensor(weights, dtype=response_mask.dtype).unsqueeze(-1)
    # broadcast per-row weight over the response positions; mask padding so
    # the column never injects weight where the loss is masked anyway
    staleness_col = (weight_col.expand_as(response_mask) * response_mask).to(response_mask.dtype)
    batch.batch["staleness_weights"] = staleness_col

    if config.adaptive_clip:
        scales = [adaptive_clip_scale(a, config.adaptive_clip_gamma, config.adaptive_clip_max_scale, config.adaptive_clip_min_scale) for a in ages]
        batch.batch["cliprange_scale"] = torch.tensor(scales, dtype=response_mask.dtype).unsqueeze(-1)

    if config.cohort_baseline and "uid" in ntb:
        _apply_cohort_baseline(batch, ntb, eps=1e-6)

    if metrics_out is not None:
        metrics_out.update(cohort_stats(ages, weights))
        if config.adaptive_clip:
            metrics_out["trajectory_async/stale_clip_scale_mean"] = sum(scales) / len(scales)

    return batch


def _apply_cohort_baseline(batch: Any, ntb: dict, eps: float = 1e-6) -> None:
    """Replace advantages with version-cohort-baseline advantages per uid
    group (mixed-version groups only; single-version groups are already
    cohort-normalized by whole-group GRPO)."""
    import torch

    uids = [str(u) for u in ntb["uid"]]
    versions = [int(v) for v in ntb["model_version"]]
    rewards = batch.batch["token_level_rewards"]
    response_mask = batch.batch["response_mask"]
    # sequence-level outcome reward: masked token-reward sum per row
    seq_rewards = (rewards * response_mask).sum(dim=-1).tolist()

    groups: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        groups.setdefault(u, []).append(i)

    new_adv = batch.batch["advantages"].clone()
    adjusted_groups = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_versions = [versions[i] for i in members]
        if max(group_versions) == min(group_versions):
            continue  # single-version group: whole-group GRPO is already a cohort baseline
        cohort_adv = cohort_advantages(group_versions, [seq_rewards[i] for i in members], eps=eps)
        # broadcast the scalar advantage over each row's response positions
        for i, adv in zip(members, cohort_adv):
            new_adv[i] = adv * response_mask[i]
        adjusted_groups += 1

    if adjusted_groups:
        batch.batch["advantages"] = new_adv
