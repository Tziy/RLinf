import json
from types import SimpleNamespace

import pytest

from rlinf.runners.embodied_runner import EmbodiedRunner


def _runner(tmp_path):
    runner = object.__new__(EmbodiedRunner)
    runner.success_early_stop_enabled = True
    runner.success_early_stop_metric = "eval/success_once"
    runner.success_early_stop_threshold = 0.75
    runner.success_early_stop_band_low = 0.75
    runner.success_early_stop_band_high = 0.85
    runner.success_early_stop_protocol_label = "weak1000_target80"
    runner._last_saved_checkpoint_step = None
    runner._last_saved_checkpoint_dir = None
    runner.global_step = 70
    runner.max_steps = 150
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(
            logger=SimpleNamespace(log_path=str(tmp_path)), val_check_interval=10
        )
    )
    runner.metric_logger = SimpleNamespace(log=lambda **_kwargs: None)
    runner.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    saved = []

    def save_checkpoint():
        saved.append(runner.global_step)
        return str(tmp_path / "checkpoint")

    runner._save_checkpoint = save_checkpoint
    return runner, saved


def test_success_early_stop_ignores_evaluation_below_threshold(tmp_path):
    runner, saved = _runner(tmp_path)

    assert not runner._maybe_request_success_early_stop(
        {"eval/success_once": 0.749}
    )
    assert saved == []
    assert not (tmp_path / "development_success_early_stop.json").exists()


def test_success_early_stop_saves_first_crossing_and_records_band(tmp_path):
    runner, saved = _runner(tmp_path)
    metrics = {"eval/success_once": 0.80}

    assert runner._maybe_request_success_early_stop(metrics)
    assert saved == [70]
    assert metrics["runner/success_early_stop_triggered"] == 1.0
    assert metrics["runner/success_early_stop_within_desired_band"] == 1.0
    assert metrics["runner/success_early_stop_overshoot"] == 0.0

    record = json.loads(
        (tmp_path / "development_success_early_stop.json").read_text()
    )
    assert record["global_step"] == 70
    assert record["metric_value"] == 0.80
    assert record["desired_band"] == [0.75, 0.85]
    assert record["within_desired_band"] is True
    assert record["overshoot"] is False
    assert record["selection_role"] == "development_only"


def test_success_early_stop_reports_overshoot_without_rewinding(tmp_path):
    runner, saved = _runner(tmp_path)
    runner._last_saved_checkpoint_step = 70
    runner._last_saved_checkpoint_dir = str(tmp_path / "already_saved")

    assert runner._maybe_request_success_early_stop(
        {"eval/success_once": 0.90}
    )
    assert saved == []
    record = json.loads(
        (tmp_path / "development_success_early_stop.json").read_text()
    )
    assert record["checkpoint_dir"] == str(tmp_path / "already_saved")
    assert record["within_desired_band"] is False
    assert record["overshoot"] is True


def test_success_early_stop_rejects_nonfinite_metric(tmp_path):
    runner, _ = _runner(tmp_path)

    with pytest.raises(ValueError, match="not finite"):
        runner._maybe_request_success_early_stop(
            {"eval/success_once": float("nan")}
        )
