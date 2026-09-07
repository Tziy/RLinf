"""Tests for initial-state matching without selecting PPO outcomes."""

import json

import pytest

from examples.analysis.run_fdvla_main_experiment import read_trials, select_pair


def row(successes, identities=None):
    return {"successes": successes, "identities": identities or [(0, 1, 36026)]}


def test_select_strongest_matched_start_pair():
    candidates = {
        "C": {"low": row(10), "high": row(20)},
        "D": {"low": row(12), "high": row(23)},
    }
    selected = select_pair(candidates, max_gap=4, min_successes=8)
    assert selected["C"] == selected["D"] == "high"
    assert selected["gap_successes"] == 3


def test_reject_unmatched_starts_and_different_trial_pools():
    with pytest.raises(ValueError, match="No eligible"):
        select_pair({"C": {"c": row(10)}, "D": {"d": row(20)}}, 4, 8)
    with pytest.raises(ValueError, match="identities"):
        select_pair({"C": {"c": row(10)}, "D": {"d": row(10, [(0, 2, 36026)])}}, 4, 8)


def test_trial_pairing_rejects_duplicates_but_does_not_condition_on_age(tmp_path):
    path = tmp_path / "trials.jsonl"
    trial = {
        "task_id": 0,
        "trial_id": 1,
        "policy_noise_seed": 36026,
        "semantic_age": 8,
        "success": True,
    }
    path.write_text(json.dumps(trial) + "\n")
    assert read_trials(path)["successes"] == 1
    trial["semantic_age"] = 0
    path.write_text(path.read_text() + json.dumps(trial) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_trials(path)
