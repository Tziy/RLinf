import hashlib
import json
from pathlib import Path

from examples.analysis.audit_fdvla_development_closure import (
    audit_development_closure,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    step0 = run_dir / "eval_trials_step_0.jsonl"
    step50 = run_dir / "eval_trials_step_50.jsonl"
    step0.write_text('{"success": false}\n', encoding="utf-8")
    step50.write_text('{"success": true}\n', encoding="utf-8")
    dsft_fresh = tmp_path / "dsft.jsonl"
    dppo_fresh = tmp_path / "dppo.jsonl"
    dsft_fresh.write_bytes(step0.read_bytes())
    dppo_fresh.write_bytes(step50.read_bytes())

    checkpoint = tmp_path / "full_weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    health = tmp_path / "health.json"
    _write_json(
        health,
        {
            "completed_updates": 50,
            "through_global_step": 50,
            "nonfinite_scalar_count": 0,
            "nonfinite_scalars": [],
            "missing_required_tags": [],
            "health": {
                "env/return": {
                    "count": 50,
                    "finite_count": 50,
                    "first": 0.2,
                    "last": 0.3,
                    "min": 0.1,
                    "max": 0.4,
                    "mean": 0.25,
                    "nonfinite_count": 0,
                },
                "env/success_once": {
                    "count": 50,
                    "finite_count": 50,
                    "first": 0.2,
                    "last": 0.3,
                    "min": 0.1,
                    "max": 0.4,
                    "mean": 0.25,
                    "nonfinite_count": 0,
                },
                "train/actor/policy_loss": {"nonfinite_count": 0},
            },
            "completed_episodes": 10,
            "successful_episodes": 4,
            "failed_episodes": 6,
            "cross_episode_packet_mismatch_count": 0,
            "semantic_replay_fingerprint_mismatch_log_count": 0,
            "local_vlm_forward_count": 0,
            "reward_model_invocation_count": 0,
        },
    )
    curve = tmp_path / "curve.json"
    _write_json(
        curve,
        {
            "pairing_audit": {
                "D-PPO/seed_0": {
                    "formal_pairing": True,
                    "paired_condition_identity_complete": True,
                    "paired_condition_identity_count": 48,
                }
            }
        },
    )
    surface = tmp_path / "surface.json"
    _write_json(
        surface,
        {
            "cells": [
                {
                    "method": "D-SFT",
                    "trials": 48,
                    "cross_episode_packet_mismatch_count": 0,
                    "local_vlm_forward_count": 0,
                },
                {
                    "method": "D-PPO",
                    "trials": 48,
                    "cross_episode_packet_mismatch_count": 0,
                    "local_vlm_forward_count": 0,
                },
            ],
            "development_mode": True,
            "formal_matrix_complete": False,
            "formal_trial_audit": {"complete": False},
            "manifest_provenance_audit": {
                "complete": True,
                "mismatches": [],
                "manifests": [
                    {
                        "identity": {
                            "method": "D-PPO",
                            "policy_checkpoint_path": str(checkpoint.resolve()),
                            "policy_checkpoint_sha256": checkpoint_sha,
                        }
                    }
                ],
            },
            "trial_identity_audit": {"complete": True},
            "paired_d_ppo_vs_d_sft": {
                "age_6_ka_8": {"actual_age_contract_violation_count": 0}
            },
        },
    )
    return {
        "run_dir": run_dir,
        "health": health,
        "curve": curve,
        "surface": surface,
        "dsft": dsft_fresh,
        "dppo": dppo_fresh,
        "checkpoint": checkpoint,
    }


def _audit(paths: dict[str, Path]) -> dict:
    return audit_development_closure(
        run_dir=paths["run_dir"],
        training_health_path=paths["health"],
        learning_curve_metadata_path=paths["curve"],
        surface_summary_path=paths["surface"],
        dsft_fresh_jsonl=paths["dsft"],
        dppo_fresh_jsonl=paths["dppo"],
        checkpoint_path=paths["checkpoint"],
        expected_updates=50,
        expected_trials=48,
    )


def test_development_closure_accepts_complete_nonformal_pilot(tmp_path):
    report = _audit(_fixture(tmp_path))

    assert report["development_closure_complete"] is True
    assert report["scientific_result_formal"] is False
    assert all(gate["passed"] for gate in report["gates"].values())


def test_development_closure_rejects_fresh_process_outcome_drift(tmp_path):
    paths = _fixture(tmp_path)
    paths["dppo"].write_text('{"success": false}\n', encoding="utf-8")

    report = _audit(paths)

    assert report["development_closure_complete"] is False
    assert report["gates"]["fresh_process_trial_reproduction"]["passed"] is False
