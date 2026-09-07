import json
from pathlib import Path

import pytest

from examples.analysis import select_fdvla_initial_checkpoint as selector
from examples.analysis import select_fdvla_paired_starts as paired_selector


def _protocol():
    return {
        "candidates": {
            "weak1000": {},
            "weak1500": {},
            "weak1000plus500": {},
        },
        "paired_condition": {"policy_noise_seed": 12026},
        "selection_rule": {
            "tie_break_order": ["weak1500", "weak1000plus500", "weak1000"]
        },
        "isolation": {
            "formal_eval_policy_noise_seeds": list(range(2026, 2034)),
        },
    }


def _result(name, distance):
    return {
        "name": name,
        "absolute_distance_to_target": distance,
        "checkpoint_path": f"/checkpoints/{name}",
        "checkpoint_sha256": name.ljust(64, "0"),
        "git_sha": "git-sha",
        "worktree_sha256": "worktree-sha",
        "backbone_path": "/backbone",
        "backbone_sha256": "backbone-sha",
    }


def test_selector_uses_registered_distance_then_tie_break(monkeypatch, tmp_path):
    protocol = _protocol()
    distances = {"weak1000": 0.15, "weak1500": 0.35, "weak1000plus500": 0.15}
    monkeypatch.setattr(selector, "_load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        selector,
        "_candidate_result",
        lambda name, _path, _protocol: (
            _result(name, distances[name]),
            {(0, trial, 12026) for trial in range(50)},
        ),
    )

    report = selector.select_candidates(
        tmp_path / "protocol.yaml",
        {name: Path(f"{name}.tsv") for name in protocol["candidates"]},
    )

    assert report["ranking"] == ["weak1000plus500", "weak1000", "weak1500"]
    assert report["selected_candidate"] == "weak1000plus500"
    assert report["paired_trial_identity_count"] == 50
    assert report["development_only"] is True


def test_selector_rejects_missing_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(selector, "_load_protocol", lambda _path: _protocol())

    with pytest.raises(ValueError, match="exactly match registration"):
        selector.select_candidates(
            tmp_path / "protocol.yaml", {"weak1000": Path("weak1000.tsv")}
        )


def test_selector_rejects_unpaired_candidate_trials(monkeypatch, tmp_path):
    protocol = _protocol()
    monkeypatch.setattr(selector, "_load_protocol", lambda _path: protocol)

    def candidate_result(name, _path, _protocol):
        trial_offset = 1 if name == "weak1500" else 0
        return _result(name, 0.1), {
            (0, trial + trial_offset, 12026) for trial in range(50)
        }

    monkeypatch.setattr(selector, "_candidate_result", candidate_result)

    with pytest.raises(ValueError, match="does not share paired trial identities"):
        selector.select_candidates(
            tmp_path / "protocol.yaml",
            {name: Path(f"{name}.tsv") for name in protocol["candidates"]},
        )


def test_selector_rejects_selection_noise_overlap(monkeypatch, tmp_path):
    protocol = _protocol()
    protocol["isolation"]["formal_eval_policy_noise_seeds"].append(12026)
    monkeypatch.setattr(selector, "_load_protocol", lambda _path: protocol)

    with pytest.raises(ValueError, match="overlaps formal evaluation"):
        selector.select_candidates(
            tmp_path / "protocol.yaml",
            {name: Path(f"{name}.tsv") for name in protocol["candidates"]},
        )


def test_selector_rejects_mixed_runtime_provenance(monkeypatch, tmp_path):
    protocol = _protocol()
    monkeypatch.setattr(selector, "_load_protocol", lambda _path: protocol)

    def candidate_result(name, _path, _protocol):
        result = _result(name, 0.1)
        if name == "weak1500":
            result["worktree_sha256"] = "different-worktree"
        return result, {(0, trial, 12026) for trial in range(50)}

    monkeypatch.setattr(selector, "_candidate_result", candidate_result)

    with pytest.raises(ValueError, match="does not share runtime provenance"):
        selector.select_candidates(
            tmp_path / "protocol.yaml",
            {name: Path(f"{name}.tsv") for name in protocol["candidates"]},
        )


def _paired_start_row(tmp_path, *, requested_age=0, mismatch=0, bootstrap=0):
    jsonl = tmp_path / f"age_{requested_age}_mismatch_{mismatch}.jsonl"
    rows = [
        {
            "task_id": 0,
            "trial_id": trial_id,
            "policy_noise_seed": 14026,
            "action_execution_horizon": 2,
            "requested_semantic_age": requested_age,
            "semantic_age": requested_age,
            "semantic_requested_actual_age_mismatch_boundary_count": mismatch,
            "semantic_bootstrap_clipped_boundary_count": bootstrap,
            "success": trial_id % 2 == 0,
        }
        for trial_id in range(48)
    ]
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"jsonl": str(jsonl), "successes": "24"}


def test_paired_start_validator_requires_exact_age_zero(tmp_path):
    row = _paired_start_row(tmp_path)
    identities = paired_selector.validate_trial_file(row)
    assert len(identities) == 48

    natural_row = _paired_start_row(tmp_path, requested_age=4)
    with pytest.raises(SystemExit, match="Expected exact semantic age 0"):
        paired_selector.validate_trial_file(natural_row)


def test_paired_start_validator_rejects_nonbootstrap_age_mismatch(tmp_path):
    row = _paired_start_row(tmp_path, mismatch=2, bootstrap=1)
    with pytest.raises(SystemExit, match="non-bootstrap mismatches"):
        paired_selector.validate_trial_file(row)


def test_paired_selector_chooses_best_jointly_eligible_pair():
    c_rows = [
        {"successes": "13", "step": "1", "source_tsv": "c.tsv"},
        {"successes": "10", "step": "2", "source_tsv": "c.tsv"},
    ]
    d_rows = [
        {"successes": "16", "step": "10", "source_tsv": "d.tsv"},
        {"successes": "9", "step": "20", "source_tsv": "d.tsv"},
    ]

    c_row, d_row = paired_selector.choose_pair(
        c_rows, d_rows, target=13, max_deviation=4, max_gap=2
    )

    assert c_row["step"] == "2"
    assert d_row["step"] == "20"


def test_paired_selector_rejects_when_no_joint_pair_is_eligible():
    c_rows = [{"successes": "13", "step": "1", "source_tsv": "c.tsv"}]
    d_rows = [{"successes": "18", "step": "10", "source_tsv": "d.tsv"}]

    with pytest.raises(SystemExit, match="No eligible matched-start pair"):
        paired_selector.choose_pair(
            c_rows, d_rows, target=13, max_deviation=4, max_gap=4
        )
