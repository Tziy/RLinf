import pytest

from examples.analysis.summarize_fdvla_posthoc_pair import (
    _config_differences,
    _parameter_fingerprint_identity,
    _posthoc_health_audit,
)


def test_posthoc_pair_config_diff_is_path_precise():
    control = {
        "runner": {"logger": {"experiment_name": "control"}},
        "algorithm": {
            "posthoc_semantic_delay_augmentation": {
                "consistency_loss_weight": 0.0,
                "force_replay_forward": True,
            }
        },
    }
    augmentation = {
        "runner": {"logger": {"experiment_name": "augmentation"}},
        "algorithm": {
            "posthoc_semantic_delay_augmentation": {
                "consistency_loss_weight": 0.01,
                "force_replay_forward": False,
            }
        },
    }

    assert set(_config_differences(control, augmentation)) == {
        "runner.logger.experiment_name",
        "algorithm.posthoc_semantic_delay_augmentation.consistency_loss_weight",
        "algorithm.posthoc_semantic_delay_augmentation.force_replay_forward",
    }


@pytest.mark.parametrize(
    ("arm", "weighted", "expected"),
    [
        ("D-PPO-PosthocControl", 0.0, True),
        ("D-PPO-PosthocControl", 0.1, False),
        ("D-PPO-PosthocAug", 0.001, True),
        ("D-PPO-PosthocAug", 0.0, False),
    ],
)
def test_posthoc_health_audit_checks_replay_and_weighted_signal(
    arm, weighted, expected
):
    health = {
        "completed_updates": 10,
        "posthoc_contract": {
            "enabled": True,
            "replay_fraction_mean": 0.125,
            "ppo_eligible_count_max": 0,
            "nonfinite_logprob_count_max": 0,
            "metrics": {"train/posthoc/weighted_aux_loss": {"max": weighted}},
        },
        "nonfinite_scalar_count": 0,
        "semantic_replay_fingerprint_mismatch_log_count": 0,
        "cross_episode_packet_mismatch_count": 0,
        "reward_model_invocation_count": 0,
        "trainable_vlm_parameter_count": 0,
    }

    assert _posthoc_health_audit(health, 0.125, arm)["complete"] is expected


def test_parameter_fingerprint_identity_requires_all_three_scopes_to_match():
    fingerprints = {
        "dit": "a" * 64,
        "value": "b" * 64,
        "all": "c" * 64,
    }
    log = (
        f"DiT initialization fingerprint (trainable sampled SHA256): {fingerprints['dit']}\n"
        f"Value head initialization fingerprint (trainable sampled SHA256): {fingerprints['value']}\n"
        f"All trainable initialization fingerprint (sampled SHA256): {fingerprints['all']}\n"
    )
    audit = _parameter_fingerprint_identity(
        {
            "D-PPO-PosthocControl": log,
            "D-PPO-PosthocAug": log,
        }
    )
    assert all(item["matched"] for item in audit.values())

    mismatch = log.replace(
        fingerprints["value"],
        "d" * 64,
    )
    audit = _parameter_fingerprint_identity(
        {"D-PPO-PosthocControl": log, "D-PPO-PosthocAug": mismatch}
    )
    assert not audit["value_head_trainable"]["matched"]
    assert audit["dit_trainable"]["matched"]
    assert audit["all_trainable"]["matched"]
