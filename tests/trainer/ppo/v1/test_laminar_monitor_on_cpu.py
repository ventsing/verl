"""Monitor completed training steps even when validation exports are absent."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "laminar_monitor", Path(__file__).resolve().parents[4] / "monitor_qwen35_longtail.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_updates_use_metrics_and_deduplicate(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(
        "step:1 - actor/loss:0.5\n"
        "step:1 - training/global_step:1 - timing_s/step:12.3 - actor/grad_norm:2.1 - actor/loss:-1e-3\n"
        "step:1 - training/global_step:1 - timing_s/step:12.3 - actor/grad_norm:2.1 - actor/loss:-1e-3\n"
        "\x1b[36m(TaskRunner)\x1b[0m step:2 - training/global_step:2 - timing_s/step:14 "
        "- actor/grad_norm:0.8 - unrelated:np.float64(2)\n"
    )
    result = monitor.read_training_updates(p)
    assert [r["step"] for r in result] == [1, 2]
    assert result[0]["metrics"]["actor/loss"] == -0.001
    assert result[1]["metrics"]["actor/grad_norm"] == 0.8


@pytest.mark.parametrize(
    "fields",
    [
        "training/global_step:1",
        "training/global_step:2 - timing_s/step:4",
        "training/global_step:1 - timing_s/step:nan",
        "training/global_step:1 - timing_s/step:0",
    ],
)
def test_partial_or_inconsistent_steps_are_not_completed(tmp_path, fields):
    p = tmp_path / "train.log"
    p.write_text("step:1 - " + fields + "\n")
    assert monitor.read_training_updates(p) == []


def test_missing_log_has_no_updates(tmp_path):
    assert monitor.read_training_updates(tmp_path / "missing.log") == []
