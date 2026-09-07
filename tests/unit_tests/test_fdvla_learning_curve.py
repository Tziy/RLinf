import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from examples.analysis.collect_fdvla_learning_curve import (
    _build_curve_rows,
    _events_by_step,
    _fixed_budget_profile_accounting,
    _pairing_audit,
    _parse_run,
    _training_wallclock_accounting,
)


def _trial(trial_id: int, success: bool):
    return {
        "task_id": 0,
        "trial_id": trial_id,
        "semantic_age": 4,
        "policy_noise_seed": 2026,
        "success": success,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_learning_curve_uses_fixed_jsonl_and_actual_cumulative_frames(tmp_path):
    _write_jsonl(
        tmp_path / "eval_trials_step_0.jsonl",
        [_trial(0, True), _trial(1, False)],
    )
    _write_jsonl(
        tmp_path / "eval_trials_step_10.jsonl",
        [_trial(0, True), _trial(1, True)],
    )
    eval_events = [
        SimpleNamespace(step=0, value=0.5, wall_time=100.0),
        SimpleNamespace(step=10, value=1.0, wall_time=160.0),
    ]
    frame_events = [
        SimpleNamespace(step=0, value=1000, wall_time=110.0),
        SimpleNamespace(step=1, value=900, wall_time=120.0),
        SimpleNamespace(step=9, value=1100, wall_time=150.0),
        SimpleNamespace(step=10, value=500, wall_time=170.0),
    ]

    rows = _build_curve_rows(
        run_dir=tmp_path,
        method="D-PPO",
        seed=0,
        eval_events=eval_events,
        frame_events=frame_events,
    )

    assert [(row["global_step"], row["environment_frames"]) for row in rows] == [
        (0, 0),
        (10, 3000),
    ]
    assert [row["wallclock_s"] for row in rows] == [0.0, 60.0]
    assert [row["success_rate"] for row in rows] == [0.5, 1.0]


def test_learning_curve_uses_resolved_config_simulator_frame_budget(tmp_path):
    metadata_dir = tmp_path / "fdvla_metadata"
    metadata_dir.mkdir()
    (metadata_dir / "resolved_config.yaml").write_text(
        "env:\n  train:\n    total_num_envs: 2\n"
        "    max_steps_per_rollout_epoch: 100\n"
        "algorithm:\n  rollout_epoch: 3\n"
    )
    _write_jsonl(tmp_path / "eval_trials_step_0.jsonl", [_trial(0, False)])
    _write_jsonl(tmp_path / "eval_trials_step_10.jsonl", [_trial(0, True)])

    rows = _build_curve_rows(
        run_dir=tmp_path,
        method="D-PPO",
        seed=0,
        eval_events=[
            SimpleNamespace(step=0, value=0.0, wall_time=1.0),
            SimpleNamespace(step=10, value=1.0, wall_time=2.0),
        ],
        frame_events=[],
    )

    assert rows[1]["environment_frames"] == 6000
    assert rows[1]["training_environment_frames_per_update"] == 600
    assert rows[1]["environment_frame_source"] == "resolved_config_fixed_budget"


def test_fixed_budget_profile_accounting_removes_enclosing_eval_counts():
    eval_events = [
        SimpleNamespace(step=0, value=0.25, wall_time=1.0),
        SimpleNamespace(step=10, value=0.5, wall_time=100.0),
    ]
    counter_events = [
        SimpleNamespace(
            step=step,
            value=22080 if step in {0, 9} else 16320,
            wall_time=step,
        )
        for step in range(10)
    ]

    accounting = _fixed_budget_profile_accounting(counter_events, eval_events)

    assert accounting["correction_applied"] is True
    assert accounting["training_counter_per_update"] == 16320
    assert accounting["profile_counter_including_eval"] == 174720
    assert accounting["training_counter"] == 163200
    assert accounting["eval_counter"] == 11520
    assert accounting["contaminated_event_steps"] == [0, 9]


def test_training_wallclock_removes_periodic_but_not_pretraining_evaluation():
    accounting = _training_wallclock_accounting(
        step_time_events=[
            SimpleNamespace(step=0, value=100.0, wall_time=10.0),
            SimpleNamespace(step=1, value=120.0, wall_time=20.0),
            SimpleNamespace(step=2, value=105.0, wall_time=30.0),
        ],
        evaluation_time_events=[
            SimpleNamespace(step=0, value=30.0, wall_time=1.0),
            SimpleNamespace(step=1, value=20.0, wall_time=19.0),
        ],
        eval_events=[
            SimpleNamespace(step=0, value=0.25, wall_time=1.0),
            SimpleNamespace(step=2, value=0.50, wall_time=19.0),
        ],
    )

    assert accounting["training_time_by_step"] == {0: 100.0, 1: 100.0, 2: 105.0}
    assert accounting["subtracted_evaluation_time_by_step"] == {1: 20.0}
    assert accounting["periodic_eval_update_steps"] == [1]


def test_learning_curve_exports_training_only_wallclock(tmp_path):
    _write_jsonl(tmp_path / "eval_trials_step_0.jsonl", [_trial(0, False)])
    _write_jsonl(tmp_path / "eval_trials_step_2.jsonl", [_trial(0, True)])

    rows = _build_curve_rows(
        run_dir=tmp_path,
        method="D-PPO",
        seed=0,
        eval_events=[
            SimpleNamespace(step=0, value=0.0, wall_time=1.0),
            SimpleNamespace(step=2, value=1.0, wall_time=230.0),
        ],
        frame_events=[],
        step_time_events=[
            SimpleNamespace(step=0, value=100.0, wall_time=100.0),
            SimpleNamespace(step=1, value=120.0, wall_time=220.0),
        ],
        evaluation_time_events=[
            SimpleNamespace(step=0, value=30.0, wall_time=1.0),
            SimpleNamespace(step=1, value=20.0, wall_time=219.0),
        ],
    )

    assert rows[1]["wallclock_s"] == 229.0
    assert rows[1]["update_wallclock_s"] == 220.0
    assert rows[1]["training_wallclock_s"] == 200.0
    assert rows[1]["periodic_evaluation_wallclock_s"] == 20.0


def test_learning_curve_rejects_missing_eval_event(tmp_path):
    _write_jsonl(tmp_path / "eval_trials_step_0.jsonl", [_trial(0, True)])
    _write_jsonl(tmp_path / "eval_trials_step_10.jsonl", [_trial(0, True)])

    with pytest.raises(ValueError, match="no matching eval event"):
        _build_curve_rows(
            run_dir=tmp_path,
            method="D-PPO",
            seed=0,
            eval_events=[SimpleNamespace(step=0, value=1.0, wall_time=100.0)],
            frame_events=[],
        )


def test_learning_curve_explicitly_marks_legacy_eval_step_minus_one(tmp_path):
    _write_jsonl(tmp_path / "eval_trials_step_0.jsonl", [_trial(0, True)])
    _write_jsonl(tmp_path / "eval_trials_step_10.jsonl", [_trial(0, True)])

    rows = _build_curve_rows(
        run_dir=tmp_path,
        method="D-PPO",
        seed=0,
        eval_events=[
            SimpleNamespace(step=0, value=1.0, wall_time=100.0),
            SimpleNamespace(step=9, value=1.0, wall_time=160.0),
        ],
        frame_events=[],
    )

    assert rows[1]["eval_event_step"] == 9
    assert rows[1]["eval_step_alignment"] == "legacy_step_minus_one"
    assert rows[1]["wallclock_s"] == 60.0


def test_events_by_step_keeps_latest_resume_event():
    events = [
        SimpleNamespace(step=1, value=10, wall_time=100.0),
        SimpleNamespace(step=1, value=20, wall_time=200.0),
    ]

    assert _events_by_step(events)[1].value == 20


def test_parse_run_requires_method_seed_and_path():
    method, seed, path = _parse_run("D-PPO:2=/tmp/run")
    assert (method, seed, str(path)) == ("D-PPO", 2, "/tmp/run")
    with pytest.raises(Exception, match="METHOD:SEED=RUN_DIR"):
        _parse_run("D-PPO=/tmp/run")


def test_learning_curve_pairing_audit_rejects_random_age_drift(tmp_path):
    baseline = [
        {
            **_trial(0, False),
            "requested_semantic_age": 2,
            "action_execution_horizon": 16,
        }
    ]
    candidate = [
        {
            **_trial(0, True),
            "requested_semantic_age": 4,
            "action_execution_horizon": 16,
        }
    ]
    _write_jsonl(tmp_path / "eval_trials_step_0.jsonl", baseline)
    _write_jsonl(tmp_path / "eval_trials_step_10.jsonl", candidate)

    audit = _pairing_audit(tmp_path)

    assert audit["formal_pairing"] is False
    assert audit["paired_condition_identity_complete"] is True
    assert audit["paired_condition_identity_count"] == 1
    assert len(audit["paired_condition_identity_sha256"]) == 64
    assert audit["comparisons"]["eval_trials_step_10.jsonl"] == {
        "formal_pairing": False,
        "operational_trial_pairing": True,
        "missing_candidate_trial_identities": 0,
        "missing_baseline_trial_identities": 0,
        "requested_semantic_age_mismatch_count": 1,
        "actual_semantic_age_mismatch_count": 0,
        "action_execution_horizon_mismatch_count": 0,
        "paired_condition_identity_complete": True,
        "paired_condition_identity_count": 1,
        "paired_condition_identity_sha256": audit["comparisons"][
            "eval_trials_step_10.jsonl"
        ]["paired_condition_identity_sha256"],
        "paired_condition_missing_fields": [],
    }


def test_learning_curve_pairing_audit_requires_complete_condition_identity(tmp_path):
    _write_jsonl(
        tmp_path / "eval_trials_step_0.jsonl",
        [{**_trial(0, False), "requested_semantic_age": 4}],
    )

    audit = _pairing_audit(tmp_path)

    assert audit["formal_pairing"] is False
    assert audit["paired_condition_identity_complete"] is False
    assert audit["paired_condition_identity_sha256"] is None
    assert audit["paired_condition_missing_fields"] == ["action_execution_horizon"]


def test_runner_logs_eval_at_completed_update_step(monkeypatch):
    from rlinf.runners import embodied_runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "check_progress",
        lambda *_args, **_kwargs: (True, False, False),
    )
    runner = object.__new__(runner_module.EmbodiedRunner)
    runner.global_step = 10
    runner.max_steps = 50
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(val_check_interval=10, save_interval=50)
    )
    runner.timer = lambda _name: nullcontext()
    runner.update_rollout_weights = lambda: None
    runner.evaluate = lambda: {"success_once": 0.875}
    runner.update_acceptance_enabled = False
    runner.save_best_only = False
    logged = []
    runner.metric_logger = SimpleNamespace(log=lambda **kwargs: logged.append(kwargs))

    metrics = runner._maybe_eval_and_checkpoint()

    assert metrics == {"eval/success_once": 0.875}
    assert logged == [{"data": {"eval/success_once": 0.875}, "step": 10}]
