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
"""CPU tests for the loss-side staleness correction: the pure math
(reweighting, adaptive clip, cohort baselines, diagnostics) plus the
torch-gated batch application (skips without torch)."""

import sys
import unittest

try:
    from tests.experimental.trajectory_async import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from verl.experimental.trajectory_async.staleness_correction import (
    StalenessCorrectionConfig,
    adaptive_clip_scale,
    cohort_advantages,
    cohort_stats,
    staleness_weight,
    staleness_weights,
)


class TestStalenessWeights(unittest.TestCase):
    def test_age_zero_is_fresh(self):
        for mode in ("none", "decay", "exp"):
            self.assertEqual(staleness_weight(0, mode=mode, lam=0.5), 1.0)

    def test_decay_family(self):
        self.assertAlmostEqual(staleness_weight(1, mode="decay", lam=0.5), 1 / 1.5)
        self.assertAlmostEqual(staleness_weight(4, mode="decay", lam=0.5), 1 / 3.0)

    def test_exp_family(self):
        import math

        self.assertAlmostEqual(staleness_weight(2, mode="exp", lam=0.25), math.exp(-0.5))

    def test_monotone_decreasing_with_age(self):
        for mode in ("decay", "exp"):
            prev = staleness_weight(0, mode=mode, lam=0.5)
            for age in range(1, 8):
                w = staleness_weight(age, mode=mode, lam=0.5)
                self.assertLessEqual(w, prev + 1e-12)
                prev = w

    def test_none_mode_is_identity(self):
        self.assertEqual(staleness_weights([0, 3, 7], mode="none"), [1.0, 1.0, 1.0])

    def test_normalization_preserves_scale_and_relative_order(self):
        ages = [0, 1, 4]
        w = staleness_weights(ages, mode="decay", lam=0.5, normalize=True)
        self.assertAlmostEqual(sum(w) / len(w), 1.0, places=9)  # mean 1
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])  # relative order preserved

    def test_unnormalized_raw(self):
        w = staleness_weights([0, 1], mode="decay", lam=1.0, normalize=False)
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 0.5)

    def test_all_stale_normalization_stays_finite(self):
        # extreme ages -> tiny raw weights; normalization must stay well-behaved
        w = staleness_weights([50, 60, 70], mode="exp", lam=0.5, normalize=True)
        self.assertAlmostEqual(sum(w) / len(w), 1.0, places=9)
        self.assertTrue(all(x > 0 for x in w))


class TestAdaptiveClipScale(unittest.TestCase):
    def test_fresh_or_zero_gamma_is_neutral(self):
        self.assertEqual(adaptive_clip_scale(0, gamma=0.25), 1.0)
        self.assertEqual(adaptive_clip_scale(4, gamma=0.0), 1.0)

    def test_positive_gamma_widens_with_age_capped(self):
        self.assertAlmostEqual(adaptive_clip_scale(2, gamma=0.25), 1.5)
        self.assertEqual(adaptive_clip_scale(100, gamma=0.25, max_scale=2.0), 2.0)

    def test_negative_gamma_tightens_floored(self):
        self.assertAlmostEqual(adaptive_clip_scale(2, gamma=-0.25), 0.5)
        self.assertEqual(adaptive_clip_scale(100, gamma=-0.25, min_scale=0.5), 0.5)


class TestCohortAdvantages(unittest.TestCase):
    def test_single_version_group_is_whole_group_grpo(self):
        versions = [3, 3, 3, 3]
        rewards = [1.0, 0.0, 0.0, 1.0]
        adv = cohort_advantages(versions, rewards)
        mean = sum(rewards) / 4
        import math

        std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 4)
        for r, a in zip(rewards, adv):
            self.assertAlmostEqual(a, (r - mean) / std, places=9)

    def test_mixed_versions_use_cohort_baselines(self):
        # two cohorts of 2; each normalized by its own mean/std
        versions = [1, 1, 2, 2]
        rewards = [1.0, 0.0, 10.0, 0.0]
        adv = cohort_advantages(versions, rewards)
        # cohort v1: mean .5 std .5 -> +/-1; cohort v2: mean 5 std 5 -> +/-1
        self.assertAlmostEqual(adv[0], 1.0, places=9)
        self.assertAlmostEqual(adv[1], -1.0, places=9)
        self.assertAlmostEqual(adv[2], 1.0, places=9)
        self.assertAlmostEqual(adv[3], -1.0, places=9)
        # the whole-group normalization would have given DIFFERENT values
        # (cohort v2's raw rewards dominate the group mean) — the point of
        # the cohort baseline is removing that between-version drift

    def test_singleton_cohort_falls_back_to_group_baseline(self):
        versions = [1, 1, 2]
        rewards = [1.0, 0.0, 8.0]
        adv = cohort_advantages(versions, rewards)
        # cohort {v1: [1.0, 0.0]} normalized by its own stats
        self.assertAlmostEqual(adv[0], 1.0, places=9)
        self.assertAlmostEqual(adv[1], -1.0, places=9)
        # the v2 singleton has no cohort baseline -> whole-group stats
        mean = 9.0 / 3
        import math

        std = math.sqrt(((1 - mean) ** 2 + (0 - mean) ** 2 + (8 - mean) ** 2) / 3)
        self.assertAlmostEqual(adv[2], (8.0 - mean) / std, places=9)

    def test_degenerate_rewards_zero_advantage(self):
        self.assertEqual(cohort_advantages([1, 2], [0.5, 0.5]), [0.0, 0.0])
        self.assertEqual(cohort_advantages([1, 1, 1], [0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])

    def test_empty(self):
        self.assertEqual(cohort_advantages([], []), [])


class TestCohortStats(unittest.TestCase):
    def test_stats_fields(self):
        stats = cohort_stats([0, 0, 1, 3], weights=[1.2, 1.2, 0.8, 0.4])
        self.assertAlmostEqual(stats["trajectory_async/stale_age_mean"], 1.0)
        self.assertEqual(stats["trajectory_async/stale_age_max"], 3)
        self.assertEqual(stats["trajectory_async/stale_age_cohorts"], 3)
        self.assertAlmostEqual(stats["trajectory_async/stale_fresh_frac"], 0.5)
        self.assertAlmostEqual(stats["trajectory_async/stale_weight_min"], 0.4)
        self.assertEqual(stats["trajectory_async/stale_weight_max"], 1.2)

    def test_empty_stats(self):
        self.assertEqual(cohort_stats([], []), {})


class TestConfig(unittest.TestCase):
    def test_from_config_none_and_empty(self):
        self.assertIsNone(StalenessCorrectionConfig.from_config(None))
        self.assertIsNone(StalenessCorrectionConfig.from_config({}))

    def test_from_config_partial_uses_defaults(self):
        cfg = StalenessCorrectionConfig.from_config({"mode": "exp", "lam": 0.25})
        self.assertEqual(cfg.mode, "exp")
        self.assertEqual(cfg.lam, 0.25)
        self.assertTrue(cfg.normalize)
        self.assertFalse(cfg.adaptive_clip)
        self.assertFalse(cfg.cohort_baseline)

    def test_from_config_rejects_bad_mode(self):
        with self.assertRaises(ValueError):
            StalenessCorrectionConfig.from_config({"mode": "quadratic"})

    def test_from_config_passthrough(self):
        cfg = StalenessCorrectionConfig(mode="decay")
        self.assertIs(StalenessCorrectionConfig.from_config(cfg), cfg)


class TestApplyStalenessCorrectionTorch(unittest.TestCase):
    """The torch batch-application path (skips without torch)."""

    def test_columns_and_metrics(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch required")

        from verl.experimental.trajectory_async.staleness_correction import apply_staleness_correction

        class FakeBatch:
            pass

        batch = FakeBatch()
        # 4 rows, 6 response positions; row ages vs version 10: 0,0,1,4
        batch.non_tensor_batch = {
            "model_version": [10, 10, 9, 6],
            "uid": ["u", "u", "u", "u"],
        }
        response_mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
            dtype=torch.float32,
        )
        batch.batch = {
            "response_mask": response_mask,
            "token_level_rewards": torch.rand(4, 6),
            "advantages": torch.randn(4, 6),
        }
        metrics = {}
        cfg = StalenessCorrectionConfig(mode="decay", lam=0.5, normalize=True, adaptive_clip=True, adaptive_clip_gamma=0.25)
        out = apply_staleness_correction(batch, current_version=10, config=cfg, metrics_out=metrics)

        w = out.batch["staleness_weights"]
        self.assertEqual(w.shape, response_mask.shape)
        # padding positions carry zero weight
        self.assertEqual(w[0, 3:].sum().item(), 0.0)
        # per-row constant weight on response positions
        for i in range(4):
            active = response_mask[i].bool()
            row_w = w[i][active]
            self.assertAlmostEqual(row_w.max().item() - row_w.min().item(), 0.0, places=6)
        # normalized: mean over ALL active positions is 1
        active_total = response_mask.sum().item()
        self.assertAlmostEqual(w.sum().item() / active_total, 1.0, places=5)
        # relative order: fresh rows heavier than stale
        self.assertGreater(w[0, 0].item(), w[3, 0].item())

        # adaptive clip column: per-row, fresh = 1.0
        scale = out.batch["cliprange_scale"]
        self.assertEqual(scale.shape, (4, 1))
        self.assertEqual(scale[0].item(), 1.0)
        self.assertAlmostEqual(scale[1].item(), 1.0)
        self.assertAlmostEqual(scale[2].item(), 1.25)
        self.assertAlmostEqual(scale[3].item(), 2.0)  # capped

        # diagnostics
        self.assertAlmostEqual(metrics["trajectory_async/stale_age_mean"], (0 + 0 + 1 + 4) / 4)
        self.assertEqual(metrics["trajectory_async/stale_age_max"], 4)

    def test_unstamped_batch_untouched(self):
        class FakeBatch:
            pass

        batch = FakeBatch()
        batch.non_tensor_batch = {"uid": ["u"]}
        batch.batch = {"response_mask": None}
        from verl.experimental.trajectory_async.staleness_correction import apply_staleness_correction

        out = apply_staleness_correction(batch, current_version=5, config=StalenessCorrectionConfig())
        self.assertIs(out, batch)
        self.assertNotIn("staleness_weights", out.batch)

    def test_cohort_baseline_rewrites_advantages(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch required")

        from verl.experimental.trajectory_async.staleness_correction import apply_staleness_correction

        class FakeBatch:
            pass

        batch = FakeBatch()
        # mixed-version group: v1 pair, v2 pair — sequence rewards 1,0,10,0
        batch.non_tensor_batch = {
            "model_version": [9, 9, 8, 8],
            "uid": ["u", "u", "u", "u"],
        }
        response_mask = torch.ones(4, 4)
        rewards = torch.zeros(4, 4)
        rewards[0, :] = 0.25  # seq reward 1.0
        rewards[1, :] = 0.0
        rewards[2, :] = 2.5  # seq reward 10.0
        rewards[3, :] = 0.0
        batch.batch = {
            "response_mask": response_mask,
            "token_level_rewards": rewards,
            "advantages": torch.zeros(4, 4),
        }
        cfg = StalenessCorrectionConfig(cohort_baseline=True)
        out = apply_staleness_correction(batch, current_version=9, config=cfg)
        adv = out.batch["advantages"]
        # cohort baselines: +/- 1 per cohort (see TestCohortAdvantages)
        self.assertAlmostEqual(adv[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), -1.0, places=5)
        self.assertAlmostEqual(adv[2, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(adv[3, 0].item(), -1.0, places=5)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], "-v"])
