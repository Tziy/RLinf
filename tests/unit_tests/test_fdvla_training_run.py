import math
from types import SimpleNamespace

import pytest

from examples.analysis.summarize_fdvla_training_run import (
    _load_posthoc_enabled,
    _run_summary,
    _stable_timing_summary,
    _summarize_events,
)


def _event(step, value, wall_time):
    return SimpleNamespace(step=step, value=value, wall_time=wall_time)


def test_training_run_summary_uses_actual_frames_and_step0_wallclock():
    scalars = {
        "eval/success_once": [_event(0, 0.5, 100.0)],
        "env/success_once": [
            _event(0, 0.5, 120.0),
            _event(1, 0.75, 140.0),
        ],
        "env/num_trajectories": [_event(0, 10, 120.0), _event(1, 20, 140.0)],
        "time/rollout/profile/control_frame_count": [
            _event(0, 1000, 120.0),
            _event(1, 1200, 140.0),
        ],
        "time/rollout/profile/local_vlm_forward_count": [
            _event(0, 0, 120.0),
            _event(1, 0, 140.0),
        ],
        "time/rollout/profile/cross_episode_packet_mismatch_count": [
            _event(0, 0, 120.0),
            _event(1, 0, 140.0),
        ],
        "time/rollout/profile/peak_gpu_memory_mib": [
            _event(0, 3000, 120.0),
            _event(1, 3200, 140.0),
        ],
    }

    summary = _run_summary(
        scalars,
        "Reward Worker disabled; reward-model invocation count: 0; "
        "VLM trainable parameter count: 0",
    )

    assert summary["completed_updates"] == 2
    assert summary["actual_control_frames"] == 2200
    assert summary["completed_episodes"] == 30
    assert summary["successful_episodes"] == 20
    assert summary["failed_episodes"] == 10
    assert summary["wallclock_from_step0_eval_s"] == 40.0
    assert summary["aggregate_control_frames_per_s"] == 55.0
    assert summary["completed_episodes_per_hour"] == 2700.0
    assert summary["local_vlm_forward_count"] == 0
    assert summary["cross_episode_packet_mismatch_count"] == 0
    assert summary["peak_rollout_gpu_memory_mib"] == 3200
    assert summary["reward_model_invocation_count"] == 0
    assert summary["trainable_vlm_parameter_count"] == 0


def test_training_run_summary_separates_fixed_eval_profile_counts():
    scalars = {
        "eval/success_once": [
            _event(0, 0.25, 1.0),
            _event(10, 0.5, 100.0),
        ],
        "env/success_once": [_event(step, 0.0, 10.0 + step) for step in range(10)],
        "env/num_trajectories": [_event(step, 1, 10.0 + step) for step in range(10)],
        "time/rollout/profile/control_frame_count": [
            _event(step, 22080 if step in {0, 9} else 16320, 10.0 + step)
            for step in range(10)
        ],
        "time/rollout/profile/local_vlm_forward_count": [
            _event(step, 308 if step in {0, 9} else 68, 10.0 + step)
            for step in range(10)
        ],
    }

    summary = _run_summary(scalars, training_environment_frames_per_update=61440)

    assert summary["actual_control_frames"] == 614400
    assert summary["environment_frame_source"] == "resolved_config_fixed_budget"
    assert summary["training_environment_frames_per_update"] == 61440
    assert summary["rollout_profile_control_frame_slots_including_eval"] == 174720
    assert summary["rollout_profile_eval_control_frame_slots"] == 11520
    assert summary["rollout_profile_frame_correction_applied"] is True
    assert summary["rollout_profile_training_control_frame_slots_per_update"] == 16320
    assert summary["local_vlm_forward_count"] == 680
    assert summary["local_vlm_forward_count_including_eval"] == 1160
    assert summary["eval_local_vlm_forward_count"] == 480
    assert summary["local_vlm_forward_correction_applied"] is True


def test_training_run_summary_accepts_decoupled_no_local_vlm_marker():
    summary = _run_summary({}, "DiT-only worker contains no local VLM parameters")

    assert summary["trainable_vlm_parameter_count"] == 0


def test_training_run_summary_detects_nonfinite_and_mismatch_log():
    scalars = {"train/actor/ratio": [_event(0, math.nan, 1.0)]}

    summary = _run_summary(
        scalars, "PPO semantic replay fingerprint mismatch: expected=a actual=b"
    )

    assert summary["nonfinite_scalar_count"] == 1
    assert summary["semantic_replay_fingerprint_mismatch_log_count"] == 1


def test_training_run_summary_promotes_posthoc_contract_metrics():
    scalars = {
        "train/posthoc/replay_selected": [
            _event(0, 0.125, 1.0),
            _event(1, 0.125, 2.0),
        ],
        "train/posthoc/ppo_eligible_count": [_event(0, 0.0, 1.0)],
        "train/posthoc/nonfinite_logprob_count": [_event(0, 0.0, 1.0)],
        "train/posthoc/weighted_aux_loss": [_event(0, 0.0002, 1.0)],
    }

    contract = _run_summary(scalars)["posthoc_contract"]

    assert contract["enabled"] is True
    assert contract["enabled_source"] == "metric_inference"
    assert contract["replay_fraction_mean"] == pytest.approx(0.125)
    assert contract["ppo_eligible_count_max"] == 0.0
    assert contract["nonfinite_logprob_count_max"] == 0.0
    assert contract["metrics"]["train/posthoc/weighted_aux_loss"][
        "mean"
    ] == pytest.approx(0.0002)


def test_training_run_summary_resolved_config_disables_placeholder_posthoc_metric():
    scalars = {
        "train/posthoc/weighted_aux_loss": [_event(0, 0.0, 1.0)],
    }

    contract = _run_summary(scalars, posthoc_enabled=False)["posthoc_contract"]

    assert contract["enabled"] is False
    assert contract["enabled_source"] == "resolved_config"


def test_load_posthoc_enabled_from_resolved_config(tmp_path):
    metadata = tmp_path / "fdvla_metadata_D-PPO"
    metadata.mkdir()
    (metadata / "resolved_config.yaml").write_text(
        "algorithm:\n  posthoc_semantic_delay_augmentation:\n    enabled: true\n"
    )

    assert _load_posthoc_enabled(tmp_path) is True


def test_training_run_summary_exports_system_profile_scalars():
    scalars = {
        "time/rollout/profile/action_boundary_blocking_ms_p95": [
            _event(0, 20.0, 1.0),
            _event(1, 30.0, 2.0),
        ],
        "time/rollout/profile/dit_forwards_per_unique_semantic_packet": [
            _event(0, 4.0, 1.0),
        ],
    }

    profile = _run_summary(scalars)["system_profile"]

    assert profile["aggregation"] == "summary_of_per-update_profile_scalars"
    assert profile["metrics"][
        "time/rollout/profile/action_boundary_blocking_ms_p95"
    ]["mean"] == pytest.approx(25.0)
    assert profile["metrics"][
        "time/rollout/profile/dit_forwards_per_unique_semantic_packet"
    ]["last"] == 4.0


def test_scalar_summary_reports_first_last_and_finite_range():
    summary = _summarize_events(
        [_event(0, 2.0, 1.0), _event(1, 4.0, 2.0), _event(2, math.inf, 3.0)]
    )

    assert summary == {
        "count": 3,
        "finite_count": 2,
        "nonfinite_count": 1,
        "first": 2.0,
        "last": math.inf,
        "min": 2.0,
        "max": 4.0,
        "mean": pytest.approx(3.0),
    }


def test_stable_timing_excludes_warmup_and_evaluation_enclosing_step():
    scalars = {
        "eval/success_once": [
            _event(0, 0.25, 1.0),
            _event(10, 0.5, 100.0),
        ],
        "time/step": [
            _event(0, 100.0, 10.0),
            _event(1, 80.0, 20.0),
            _event(2, 10.0, 30.0),
            _event(3, 12.0, 40.0),
            _event(9, 200.0, 100.0),
        ],
    }

    summary = _stable_timing_summary(scalars)

    assert summary["warmup_update_count"] == 2
    assert summary["evaluation_global_steps"] == [10]
    assert summary["excluded_training_event_steps"] == [9]
    step = summary["metrics"]["time/step"]
    assert step["count"] == 2
    assert step["first"] == 10.0
    assert step["last"] == 12.0
    assert step["mean"] == pytest.approx(11.0)
    assert step["median"] == pytest.approx(11.0)
