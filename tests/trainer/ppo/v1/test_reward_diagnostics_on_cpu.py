from verl.experimental.trajectory_async.reward_diagnostics import RewardDiagnostics


def test_raw_reward_is_unchanged_and_length_censoring_is_separate():
    diagnostics = RewardDiagnostics(window_groups=2)
    a = diagnostics.update([("a_0_0", 1, 100), ("a_1_0", 0, 32768)], 32768)
    assert a["laminar/reward/raw_batch_mean"] == 0.5
    assert a["laminar/reward/not_truncated_reward_mean"] == 1
    assert a["laminar/reward/truncated_reward_mean"] == 0
    assert a["laminar/reward/mixed_reward_group_fraction"] == 1
    b = diagnostics.update([("b_0_0", 0, 32768), ("b_1_0", 0, 32768)], 32768)
    assert b["laminar/reward/raw_batch_mean"] == 0
    assert b["laminar/reward/rolling_group_mean"] == 0.25
    assert "laminar/reward/not_truncated_reward_mean" not in b
    c = diagnostics.update([("c_0_0", 1, 100), ("c_1_0", 1, 200)], 32768)
    assert c["laminar/reward/rolling_group_count"] == 2
    assert c["laminar/reward/rolling_group_mean"] == 0.5
