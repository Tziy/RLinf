import json

import pytest

from examples.analysis.fdvla_statistics import (
    exact_mcnemar_p,
    paired_binary_summary,
    trial_key,
    wilson_interval,
)
from examples.analysis.summarize_fdvla_results import (
    FORMAL_MIN_FINAL_TRIALS,
    _bootstrap_seed_mean,
    _curve_arrays,
    _curve_pairing_metadata_audit,
    _final_eval_audit,
    _paired_curve_auc,
    _parse_eval,
)


def _row(trial_id: int, success: bool, *, age: int = 4, noise: int = 99):
    return {
        "task_id": 0,
        "trial_id": trial_id,
        "semantic_age": age,
        "policy_noise_seed": noise,
        "success": success,
    }


def test_trial_key_includes_task_trial_age_and_noise():
    assert trial_key(_row(3, True)) == (0, 3, 4, 99)


def test_pairing_rejects_age_or_noise_mismatch():
    with pytest.raises(ValueError, match="Paired trial sets differ"):
        paired_binary_summary([_row(0, False, age=2)], [_row(0, True, age=4)])


def test_pairing_rejects_action_horizon_mismatch():
    reference = [{**_row(0, False), "action_execution_horizon": 16}]
    candidate = [{**_row(0, True), "action_execution_horizon": 8}]

    with pytest.raises(ValueError, match="action-execution horizons differ"):
        paired_binary_summary(reference, candidate)


def test_pairing_counts_transitions_and_exact_mcnemar():
    reference = [_row(0, False), _row(1, True), _row(2, False), _row(3, True)]
    candidate = [_row(0, True), _row(1, False), _row(2, True), _row(3, True)]

    summary = paired_binary_summary(reference, candidate)

    assert summary["successes"] == 3
    assert summary["failure_to_success"] == 2
    assert summary["success_to_failure"] == 1
    assert summary["exact_mcnemar_p"] == exact_mcnemar_p(2, 1)
    assert summary["formal_pairing"] is True


def test_development_pairing_reports_condition_mismatches():
    reference = [
        {
            **_row(0, False, age=2),
            "requested_semantic_age": 2,
            "action_execution_horizon": 16,
        }
    ]
    candidate = [
        {
            **_row(0, True, age=4),
            "requested_semantic_age": 4,
            "action_execution_horizon": 8,
        }
    ]

    summary = paired_binary_summary(reference, candidate, require_condition_match=False)

    assert summary["formal_pairing"] is False
    assert summary["failure_to_success"] == 1
    assert summary["requested_semantic_age_mismatch_count"] == 1
    assert summary["actual_semantic_age_mismatch_count"] == 1
    assert summary["action_execution_horizon_mismatch_count"] == 1


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(30, 48)

    assert low < 30 / 48 < high


def test_formal_seed_summary_requires_exactly_three_seeds():
    with pytest.raises(ValueError, match="exactly three seeds"):
        _bootstrap_seed_mean([0.1, 0.2])

    summary = _bootstrap_seed_mean([0.1, 0.2, 0.3], samples=100)
    assert summary["mean"] == pytest.approx(0.2)


def test_parse_final_eval_distinguishes_shared_and_training_seed():
    assert _parse_eval("D-SFT=/tmp/sft.jsonl").seed is None
    parsed = _parse_eval("D-PPO:2=/tmp/ppo.jsonl")
    assert (parsed.method, parsed.seed, str(parsed.path)) == (
        "D-PPO",
        2,
        "/tmp/ppo.jsonl",
    )


def test_formal_final_eval_audit_requires_400_trials_and_three_matching_seeds():
    def rows(count):
        return [
            {
                **_row(index, index % 2 == 0, noise=2026 + index // 50),
                "trial_id": index % 50,
                "requested_semantic_age": 4,
                "action_execution_horizon": 16,
            }
            for index in range(count)
        ]

    evaluations = {("D-SFT", None): rows(FORMAL_MIN_FINAL_TRIALS)}
    for method in ("D-PPO", "C-PPO"):
        for seed in (0, 1, 2):
            evaluations[(method, seed)] = rows(FORMAL_MIN_FINAL_TRIALS)

    audit = _final_eval_audit(
        evaluations,
        required_unseeded_methods=("D-SFT",),
        required_seeded_methods=("D-PPO", "C-PPO"),
    )

    assert audit["complete"] is True
    evaluations[("D-PPO", 2)] = rows(48)
    audit = _final_eval_audit(
        evaluations,
        required_unseeded_methods=("D-SFT",),
        required_seeded_methods=("D-PPO", "C-PPO"),
    )
    assert audit["complete"] is False
    assert audit["underfilled_evaluations"] == [
        {"method": "D-PPO", "seed": 2, "trials": 48}
    ]


def test_curve_arrays_rejects_missing_training_wallclock_column():
    with pytest.raises(ValueError, match="training_wallclock_s"):
        _curve_arrays(
            [{"wallclock_s": 0, "success_rate": 0.2}],
            "training_wallclock_s",
        )


def test_paired_auc_uses_only_shared_domain_and_reports_normalized_delta():
    reference = [
        {"wallclock_s": 0, "success_rate": 0.2},
        {"wallclock_s": 10, "success_rate": 0.4},
        {"wallclock_s": 20, "success_rate": 0.6},
    ]
    candidate = [
        {"wallclock_s": 0, "success_rate": 0.3},
        {"wallclock_s": 10, "success_rate": 0.5},
        {"wallclock_s": 30, "success_rate": 0.9},
    ]

    summary = _paired_curve_auc(reference, candidate, "wallclock_s")

    assert summary["shared_domain_max"] == 20
    assert summary["reference_auc"] == pytest.approx(8.0)
    assert summary["candidate_auc"] == pytest.approx(10.0)
    assert summary["auc_delta"] == pytest.approx(2.0)
    assert summary["normalized_auc_delta"] == pytest.approx(0.1)


def test_curve_metadata_requires_same_condition_fingerprint_across_runs(tmp_path):
    required = {
        f"{method}/seed_{seed}" for method in ("C-PPO", "D-PPO") for seed in (0, 1, 2)
    }
    audits = {
        key: {
            "formal_pairing": True,
            "paired_condition_identity_complete": True,
            "paired_condition_identity_count": 400,
            "paired_condition_identity_sha256": "a" * 64,
        }
        for key in required
    }
    metadata_path = tmp_path / "curves.csv.metadata.json"
    metadata_path.write_text(json.dumps({"pairing_audit": audits}))

    audit = _curve_pairing_metadata_audit(metadata_path, required)
    assert audit["complete"] is True
    assert audit["reference_condition_identity"]["count"] == 400

    audits["D-PPO/seed_2"]["paired_condition_identity_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps({"pairing_audit": audits}))
    audit = _curve_pairing_metadata_audit(metadata_path, required)
    assert audit["complete"] is False
    assert audit["condition_fingerprint_mismatches"] == [
        {
            "reference_run": "C-PPO/seed_0",
            "reference_count": 400,
            "reference_sha256": "a" * 64,
            "run": "D-PPO/seed_2",
            "count": 400,
            "sha256": "b" * 64,
        }
    ]


def test_eval_audit_metadata_selects_only_completed_rows():
    import torch

    from rlinf.workers.env.env_worker import _extract_eval_audit_metadata

    selected = _extract_eval_audit_metadata(
        {
            "rollout_semantic_actual_age_frames": torch.tensor([0, 4, 6]),
            "rollout_semantic_requested_age_frames": torch.tensor([2, 4, 8]),
            "rollout_semantic_bootstrap_clipped": torch.tensor([True, False, True]),
            "rollout_policy_noise_seeds": torch.tensor([99, 99, 99]),
            "rollout_action_execution_horizon": torch.tensor([4, 4, 4]),
        },
        torch.tensor([False, True, True]),
    )

    assert selected["semantic_age"].tolist() == [4, 6]
    assert selected["requested_semantic_age"].tolist() == [4, 8]
    assert selected["semantic_age_bootstrap_clipped"].tolist() == [False, True]
    assert selected["policy_noise_seed"].tolist() == [99, 99]
    assert selected["action_execution_horizon"].tolist() == [4, 4]


def test_eval_semantic_episode_audit_accumulates_bootstrap_boundaries():
    import torch

    from rlinf.workers.env.env_worker import (
        _accumulate_eval_semantic_episode_audit,
        _new_eval_semantic_episode_audit,
        _select_eval_semantic_episode_audit,
    )

    audit = _new_eval_semantic_episode_audit(2)
    _accumulate_eval_semantic_episode_audit(
        audit,
        {
            "rollout_semantic_actual_age_frames": torch.tensor([0, 2]),
            "rollout_semantic_requested_age_frames": torch.tensor([4, 4]),
            "rollout_semantic_bootstrap_clipped": torch.tensor([True, True]),
        },
        torch.tensor([True, True]),
    )
    _accumulate_eval_semantic_episode_audit(
        audit,
        {
            "rollout_semantic_actual_age_frames": torch.tensor([4, 4]),
            "rollout_semantic_requested_age_frames": torch.tensor([4, 4]),
            "rollout_semantic_bootstrap_clipped": torch.tensor([False, False]),
        },
        torch.tensor([True, False]),
    )

    selected = _select_eval_semantic_episode_audit(audit, torch.tensor([True, True]))
    assert selected["semantic_boundary_count"].tolist() == [2, 1]
    assert selected["semantic_bootstrap_clipped_boundary_count"].tolist() == [1, 1]
    assert selected[
        "semantic_requested_actual_age_mismatch_boundary_count"
    ].tolist() == [1, 1]
    assert selected["semantic_actual_age_mean"].tolist() == [2.0, 2.0]
    assert selected["semantic_actual_age_min"].tolist() == [0, 2]
    assert selected["semantic_actual_age_max"].tolist() == [4, 2]


def test_trial_jsonl_persists_pairing_audit_fields(tmp_path):
    import json

    import torch

    from rlinf.utils.metric_utils import write_evaluate_trials

    output_path = tmp_path / "eval_trials_step_0.jsonl"
    count = write_evaluate_trials(
        [
            {
                "task_id": torch.tensor([0, 0]),
                "trial_id": torch.tensor([3, 4]),
                "success_once": torch.tensor([False, True]),
                "semantic_age": torch.tensor([2, 4]),
                "requested_semantic_age": torch.tensor([4, 4]),
                "semantic_age_bootstrap_clipped": torch.tensor([True, False]),
                "semantic_boundary_count": torch.tensor([2, 3]),
                "semantic_bootstrap_clipped_boundary_count": torch.tensor([1, 0]),
                "semantic_requested_actual_age_mismatch_boundary_count": torch.tensor(
                    [1, 0]
                ),
                "semantic_actual_age_mean": torch.tensor([1.0, 4.0]),
                "semantic_actual_age_min": torch.tensor([0, 4]),
                "semantic_actual_age_max": torch.tensor([2, 4]),
                "policy_noise_seed": torch.tensor([1234, 1234]),
                "action_execution_horizon": torch.tensor([4, 4]),
            }
        ],
        str(output_path),
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 2
    assert rows[0] == {
        "action_execution_horizon": 4,
        "policy_noise_seed": 1234,
        "requested_semantic_age": 4,
        "semantic_age": 2,
        "semantic_age_bootstrap_clipped": True,
        "semantic_actual_age_max": 2,
        "semantic_actual_age_mean": 1.0,
        "semantic_actual_age_min": 0,
        "semantic_bootstrap_clipped_boundary_count": 1,
        "semantic_boundary_count": 2,
        "semantic_requested_actual_age_mismatch_boundary_count": 1,
        "success": False,
        "task_id": 0,
        "trial_id": 3,
    }


def test_fixed_age_controls_exact_publish_frame():
    from types import SimpleNamespace

    from rlinf.workers.env.env_worker import (
        EnvWorker,
        _should_publish_eval_semantic,
    )

    worker = object.__new__(EnvWorker)
    worker._semantic_eval_random_age_max_frames = -1
    worker._semantic_eval_fixed_age_frames = 0
    worker._semantic_eval_rollout_step = [0]
    worker._semantic_eval_target_age = [0]
    worker.model_cfg = SimpleNamespace(num_action_chunks=16)
    worker.eval_execution_horizon = 16

    assert worker._semantic_eval_publish_frame_for_next_boundary(0) == 16
    assert worker._semantic_eval_target_age[0] == 0

    worker._semantic_eval_fixed_age_frames = 12
    assert worker._semantic_eval_publish_frame_for_next_boundary(0) == 4
    assert worker._semantic_eval_target_age[0] == 12
    assert not _should_publish_eval_semantic(False, -1, -1)
    assert _should_publish_eval_semantic(True, -1, -1)
    assert _should_publish_eval_semantic(False, 0, -1)
    assert _should_publish_eval_semantic(False, 12, -1)
    assert _should_publish_eval_semantic(False, -1, 6)


@pytest.mark.parametrize(
    ("fixed_age", "execution_horizon", "expected_publish_frame"),
    [
        (0, 1, 1),
        (1, 1, 1),
        (2, 4, 2),
        (6, 4, 2),
        (8, 4, 4),
        (12, 8, 4),
        (16, 16, 16),
    ],
)
def test_fixed_age_publish_phase_supports_cross_chunk_history(
    fixed_age, execution_horizon, expected_publish_frame
):
    from rlinf.workers.env.env_worker import _fixed_age_publish_frame

    assert (
        _fixed_age_publish_frame(fixed_age, execution_horizon) == expected_publish_frame
    )


def test_eval_execution_horizon_is_separate_from_prediction_horizon():
    from rlinf.workers.env.env_worker import _resolve_eval_execution_horizon

    assert _resolve_eval_execution_horizon(16, -1) == 16
    assert _resolve_eval_execution_horizon(16, 1) == 1
    assert _resolve_eval_execution_horizon(16, 8) == 8
    with pytest.raises(ValueError, match="eval_execution_horizon"):
        _resolve_eval_execution_horizon(16, 17)


def test_train_execution_horizon_is_separate_from_prediction_horizon():
    from rlinf.workers.env.env_worker import _resolve_train_execution_horizon

    assert _resolve_train_execution_horizon(16, -1) == 16
    assert _resolve_train_execution_horizon(16, 2) == 2
    with pytest.raises(ValueError, match="train_execution_horizon"):
        _resolve_train_execution_horizon(16, 17)


def test_fixed_age_is_forwarded_as_requested_age_metadata():
    import torch

    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker._semantic_env_bootstrap_publish = False
    worker._semantic_raw_publisher = None
    worker._semantic_eval_fixed_age_frames = 12
    worker._semantic_eval_random_age_max_frames = -1
    worker.enable_rlt = False
    worker._build_semantic_metadata = lambda *_args, **_kwargs: {
        "env_ids": torch.tensor([10, 11]),
        "frame_ids": torch.tensor([0, 4]),
        "episode_generations": torch.tensor([0, 0]),
        "observation_wallclock_s": torch.tensor([1.0, 1.0]),
    }

    data = worker._build_rollout_input_data(
        {"obs": {"states": torch.zeros(2, 3)}, "final_obs": None},
        stage_id=0,
        mode="eval",
    )

    assert data["obs"]["__rlinf_semantic_target_age_frames"].tolist() == [12, 12]
