import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from examples.analysis.summarize_fdvla_realtime_surface import (
    FORMAL_MIN_TRIALS_PER_CELL,
    CellSpec,
    _deadline_summary,
    _formal_trial_audit,
    _manifest_provenance_audit,
    _matrix_audit,
    _paired_transition,
    _pareto_summary,
    _read_manifest,
    _read_manifest_with_provenance,
    _surface_timing,
    _trial_identity_audit,
    _validate_cell,
)

ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / "examples" / "embodiment" / "fdvla_binary_common.sh"
ARTIFACT_HASH_ALGORITHM = "file-raw+dir-logical-deref-sha256-v1"


def _row(
    trial_id,
    success,
    *,
    age=4,
    horizon=2,
    clipped=False,
    boundary_count=None,
    clipped_boundary_count=None,
):
    row = {
        "task_id": 0,
        "trial_id": trial_id,
        "semantic_age": age,
        "requested_semantic_age": 4,
        "semantic_age_bootstrap_clipped": clipped,
        "action_execution_horizon": horizon,
        "policy_noise_seed": 2026,
        "success": success,
    }
    if boundary_count is not None:
        row.update(
            {
                "semantic_boundary_count": boundary_count,
                "semantic_bootstrap_clipped_boundary_count": clipped_boundary_count,
                "semantic_requested_actual_age_mismatch_boundary_count": (
                    clipped_boundary_count
                ),
                "semantic_actual_age_mean": age,
                "semantic_actual_age_min": age,
                "semantic_actual_age_max": age,
            }
        )
    return row


def test_surface_cell_records_wilson_and_bootstrap_audit():
    spec = CellSpec("D-SFT", 4, 2, Path("unused.jsonl"))
    first_row = _row(0, True, boundary_count=3, clipped_boundary_count=1)
    first_row.update(
        {
            "semantic_actual_age_mean": 2.0,
            "semantic_actual_age_min": 0,
            "semantic_actual_age_max": 4,
        }
    )
    summary = _validate_cell(
        spec,
        [
            first_row,
            _row(
                1,
                False,
                age=2,
                clipped=True,
                boundary_count=2,
                clipped_boundary_count=1,
            ),
        ],
    )

    assert summary["success_rate"] == 0.5
    assert summary["wilson_95_low"] < 0.5 < summary["wilson_95_high"]
    assert summary["bootstrap_clipped_trials"] == 2
    assert summary["terminal_boundary_bootstrap_clipped_trials"] == 1
    assert summary["semantic_boundaries"] == 5
    assert summary["bootstrap_clipped_boundaries"] == 2
    assert summary["episodes_with_bootstrap_clipping"] == 2
    assert summary["actual_age_min"] == 0
    assert summary["actual_age_max"] == 4
    assert summary["actual_age_mean"] == 2.0


def test_surface_cell_rejects_wrong_action_horizon():
    spec = CellSpec("D-SFT", 4, 4, Path("unused.jsonl"))
    with pytest.raises(ValueError, match="action horizon"):
        _validate_cell(spec, [_row(0, True, horizon=2)])


def test_surface_cell_enforces_formal_trial_floor():
    spec = CellSpec("D-SFT", 4, 2, Path("unused.jsonl"))
    with pytest.raises(ValueError, match="required 400"):
        _validate_cell(spec, [_row(0, True)], min_trials=400)


def test_surface_timing_is_explicitly_unavailable_without_event_files(tmp_path):
    spec = CellSpec("D-SFT", 4, 2, tmp_path / "eval_trials_combined.jsonl")

    timing = _surface_timing(spec, successes=1, trials=2)

    assert timing["timing_available"] is False
    assert timing["timing_run_count"] == 0
    assert timing["eval_wall_clock_s"] is None
    assert timing["eval_wall_clock_per_trial_s"] is None
    assert timing["successful_trials_per_hour"] is None


def test_success_wall_clock_pareto_marks_dominated_horizon():
    def cell(horizon, success, seconds):
        return {
            "method": "D-PPO",
            "semantic_age_frames": 6,
            "action_execution_horizon": horizon,
            "success_rate": success,
            "eval_wall_clock_s": 50 * seconds,
            "eval_wall_clock_per_trial_s": seconds,
            "successful_trials_per_hour": 3600 * success / seconds,
        }

    summary = _pareto_summary(
        [cell(4, 0.70, 2.0), cell(8, 0.78, 1.2), cell(16, 0.52, 1.0)]
    )["D-PPO:age_6"]

    assert summary["available"] is True
    assert summary["frontier_action_horizons"] == [8, 16]


def test_surface_runner_uses_distinct_noise_replicates_for_400_trials():
    script = Path("examples/embodiment/eval_fdvla_realtime_surface.sh").read_text()

    assert (
        "SURFACE_POLICY_NOISE_SEEDS:-2026 2027 2028 2029 2030 2031 2032 2033" in script
    )
    assert 'EVAL_NOISE_SEED="${noise_seed}"' in script
    assert "SURFACE_TASK_ID_FILTER:-${EVAL_TASK_ID_FILTER:-'[0]'}" in script
    assert 'EVAL_TASK_ID_FILTER="${SURFACE_TASK_ID_FILTER}"' in script
    assert "eval_trials_combined.jsonl" in script
    assert 'cat "${seed_jsonl}" >> "${jsonl_path}"' in script
    assert "SURFACE_EVAL_ROLLOUT_EPOCH=${EVAL_ROLLOUT_EPOCH:-1}" in script
    assert "SURFACE_EVAL_NUM_ENVS=${SURFACE_EVAL_NUM_ENVS:-52}" in script
    assert "C-PPO | D-PPO)" in script
    assert "EVAL_LAUNCHER=${SCRIPT_DIR}/run_coupled_n1d7_binary_ppo.sh" in script
    assert 'bash "${EVAL_LAUNCHER}"' in script
    assert "artifact_hash_algorithm\\tmethod" in script
    assert "policy_checkpoint_path\\tpolicy_checkpoint_sha256" in script
    assert "git_sha\\tworktree_sha256\\taction_prediction_horizon" in script
    assert 'PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"' in script
    assert '--min-trials-per-cell "${SURFACE_MIN_TRIALS}"' in script
    assert "--allow-incomplete" in script
    assert "RESUME_SURFACE=${RESUME_SURFACE:-false}" in script
    assert "Resume validated completed cell" in script
    assert "Resume signature mismatch" in script
    assert (
        "SURFACE_POLICY_CHECKPOINT_SHA256=${SURFACE_INITIAL_CHECKPOINT_SHA256}"
        in script
    )
    assert (
        'FDVLA_RESOLVED_MODEL_SHA256="${SURFACE_INITIAL_CHECKPOINT_SHA256}"' in script
    )
    assert 'FDVLA_RESOLVED_BACKBONE_SHA256="${SURFACE_BACKBONE_SHA256}"' in script


def test_run_metadata_records_checkpoint_backbone_and_hash_algorithm():
    common = COMMON_SCRIPT.read_text()

    assert "fdvla_write_resolved_sha256()" in common
    assert '"${FDVLA_RESOLVED_MODEL_SHA256:-}"' in common
    assert '"${FDVLA_RESOLVED_BACKBONE_SHA256:-}"' in common
    assert '"${FDVLA_RESOLVED_PPO_SHA256:-}"' in common
    assert '"${metadata_dir}/backbone_sha256.txt"' in common
    assert '"${metadata_dir}/artifact_hash_algorithm.txt"' in common


def test_precomputed_metadata_hash_is_validated_and_written_without_rehash(tmp_path):
    output = tmp_path / "artifact.sha256"
    artifact = tmp_path / "missing-artifact-is-not-read"
    digest = "a" * 64
    command = (
        'source "$FDVLA_COMMON"; '
        'fdvla_write_resolved_sha256 "$OUTPUT" "$ARTIFACT" "$DIGEST"'
    )
    subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "FDVLA_COMMON": str(COMMON_SCRIPT),
            "OUTPUT": str(output),
            "ARTIFACT": str(artifact),
            "DIGEST": digest,
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert output.read_text().strip() == digest

    invalid = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "FDVLA_COMMON": str(COMMON_SCRIPT),
            "OUTPUT": str(output),
            "ARTIFACT": str(artifact),
            "DIGEST": "not-a-sha",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "Invalid precomputed SHA256" in invalid.stderr


def test_surface_resume_binds_policy_checkpoint_content_before_launch(tmp_path):
    script = Path("examples/embodiment/eval_fdvla_realtime_surface.sh").resolve()
    common = Path("examples/embodiment/fdvla_binary_common.sh").resolve()
    model = tmp_path / "initial.bin"
    backbone = tmp_path / "backbone.bin"
    model.write_bytes(b"initial checkpoint")
    backbone.write_bytes(b"frozen backbone")

    fingerprint = subprocess.run(
        ["bash", "-c", 'source "$FDVLA_COMMON"; fdvla_worktree_sha256'],
        env={**os.environ, "FDVLA_COMMON": str(common)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    backbone_sha = hashlib.sha256(backbone.read_bytes()).hexdigest()

    surface_root = tmp_path / "surface"
    surface_root.mkdir()
    jsonl = surface_root / "age_4_ka_2" / "eval_trials_combined.jsonl"
    header = (
        "schema_version\tartifact_hash_algorithm\tmethod\tsemantic_age\t"
        "action_horizon\t"
        "task_id_filter\tpolicy_noise_seeds\tpolicy_checkpoint_path\t"
        "policy_checkpoint_sha256\tinitial_checkpoint_path\t"
        "initial_checkpoint_sha256\tbackbone_path\tbackbone_sha256\t"
        "git_sha\tworktree_sha256\taction_prediction_horizon\tcontrol_hz\tjsonl"
    )

    def manifest_row(policy_sha: str) -> str:
        return "\t".join(
            (
                "3",
                ARTIFACT_HASH_ALGORITHM,
                "D-SFT",
                "4",
                "2",
                "[0]",
                "2026",
                str(model),
                policy_sha,
                str(model),
                model_sha,
                str(backbone),
                backbone_sha,
                git_sha,
                fingerprint,
                "16",
                "20",
                str(jsonl),
            )
        )

    manifest = surface_root / "surface_cells.tsv"
    manifest.write_text(f"{header}\n{manifest_row('0' * 64)}\n", encoding="utf-8")
    environment = {
        **os.environ,
        "ACTION_CHUNK_SIZE": "16",
        "DRY_RUN": "false",
        "EVAL_ROLLOUT_EPOCH": "1",
        "FDVLA_CONTROL_HZ": "20",
        "FDVLA_SURFACE_ROOT": str(surface_root),
        "GR00T_BACKBONE_PATH": str(backbone),
        "GR00T_MODEL_PATH": str(model),
        "POLICY_METHOD": "D-SFT",
        "RESUME_SURFACE": "true",
        "SEMANTIC_AGES": "4",
        "ACTION_HORIZONS": "2",
        "SURFACE_EVAL_NUM_ENVS": "1",
        "SURFACE_MIN_TRIALS": "1",
        "SURFACE_POLICY_NOISE_SEEDS": "2026",
    }

    mismatch = subprocess.run(
        ["bash", str(script)],
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatch.returncode == 2
    assert "Resume signature mismatch" in mismatch.stderr

    manifest.write_text(f"{header}\n{manifest_row(model_sha)}\n", encoding="utf-8")
    accepted_identity = subprocess.run(
        ["bash", str(script)],
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted_identity.returncode == 2
    assert "Resume result is missing or empty" in accepted_identity.stderr


def _artifact_hash(path: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$FDVLA_COMMON"; fdvla_checkpoint_sha256 "$CHECKPOINT"',
        ],
        env={
            **os.environ,
            "CHECKPOINT": str(path),
            "FDVLA_COMMON": str(COMMON_SCRIPT),
        },
        capture_output=True,
        text=True,
        check=check,
    )


def test_artifact_hash_dereferences_symlinks_and_is_relocation_stable(tmp_path):
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"model-v1")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").symlink_to(blob)

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    (relocated / "model.bin").write_bytes(b"model-v1")

    first = _artifact_hash(snapshot).stdout.strip()
    assert first == _artifact_hash(relocated).stdout.strip()
    assert len(first) == 64

    blob.write_bytes(b"model-v2")
    assert _artifact_hash(snapshot).stdout.strip() != first


@pytest.mark.parametrize("kind", ["empty", "broken-symlink", "symlink-loop"])
def test_artifact_hash_rejects_unresolved_checkpoint_directories(tmp_path, kind):
    checkpoint = tmp_path / kind
    checkpoint.mkdir()
    if kind == "broken-symlink":
        (checkpoint / "model.bin").symlink_to(tmp_path / "missing.bin")
    elif kind == "symlink-loop":
        (checkpoint / "loop").symlink_to(".", target_is_directory=True)

    result = _artifact_hash(checkpoint, check=False)

    assert result.returncode == 2
    assert "Checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"SEMANTIC_AGES": "0 0"}, "Duplicate semantic age"),
        ({"ACTION_HORIZONS": "1 17"}, "exceeds prediction horizon"),
        (
            {"ACTION_CHUNK_SIZE": "8", "ACTION_HORIZONS": "1 2 4 8"},
            "16-step action prediction horizon",
        ),
        ({"EVAL_ROLLOUT_EPOCH": "2"}, "EVAL_ROLLOUT_EPOCH=1"),
        ({"SURFACE_MIN_TRIALS": "not-an-int"}, "must be positive integers"),
    ],
)
def test_surface_runner_rejects_invalid_matrix_before_launch(override, message):
    script = Path("examples/embodiment/eval_fdvla_realtime_surface.sh").resolve()
    environment = {
        **os.environ,
        "DRY_RUN": "true",
        "POLICY_METHOD": "D-SFT",
        **override,
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd="/tmp",
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_surface_runner_deadline_axes_dry_run_emits_registered_union_once():
    script = Path("examples/embodiment/eval_fdvla_realtime_surface.sh").resolve()
    result = subprocess.run(
        ["bash", str(script)],
        cwd="/tmp",
        env={
            **os.environ,
            "DRY_RUN": "true",
            "POLICY_METHOD": "D-SFT",
            "SURFACE_SWEEP_MODE": "deadline_axes",
        },
        capture_output=True,
        text=True,
        check=True,
    )

    cell_lines = [
        line for line in result.stdout.splitlines() if line.startswith("age_")
    ]
    expected = {
        *(f"age_{age}_ka_1" for age in (0, 1, 2, 4, 6, 8, 12, 16)),
        *(f"age_0_ka_{horizon}" for horizon in (2, 4, 8, 16)),
    }
    assert len(cell_lines) == 12
    assert {line.split()[0] for line in cell_lines} == expected
    assert "surface_sweep_mode=deadline_axes" in result.stdout


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"SURFACE_SWEEP_MODE": "deadline_axes", "SEMANTIC_AGES": "1 2"},
            "requires semantic age 0 and action horizon 1",
        ),
        (
            {"SURFACE_SWEEP_MODE": "deadline_axes", "ACTION_HORIZONS": "2 4"},
            "requires semantic age 0 and action horizon 1",
        ),
        ({"SURFACE_SWEEP_MODE": "unknown"}, "must be cartesian or deadline_axes"),
    ],
)
def test_surface_runner_rejects_invalid_deadline_axes(override, message):
    script = Path("examples/embodiment/eval_fdvla_realtime_surface.sh").resolve()
    result = subprocess.run(
        ["bash", str(script)],
        cwd="/tmp",
        env={
            **os.environ,
            "DRY_RUN": "true",
            "POLICY_METHOD": "D-SFT",
            **override,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_surface_cell_accepts_50_states_by_8_noise_replicates():
    spec = CellSpec("D-SFT", 4, 2, Path("unused.jsonl"))
    rows = []
    for noise_seed in range(2026, 2034):
        for trial_id in range(50):
            row = _row(trial_id, trial_id % 2 == 0)
            row["policy_noise_seed"] = noise_seed
            rows.append(row)

    summary = _validate_cell(spec, rows, min_trials=400)

    assert summary["trials"] == 400
    assert summary["successes"] == 200


def test_surface_trial_identity_audit_requires_same_trials_across_cells_and_methods():
    paired_rows = [_row(0, True), _row(1, False)]
    raw_cells = {
        ("D-SFT", 4, 2): paired_rows,
        ("D-SFT", 6, 2): [
            {**row, "requested_semantic_age": 6, "semantic_age": 6}
            for row in paired_rows
        ],
        ("D-PPO", 4, 2): paired_rows,
    }

    audit = _trial_identity_audit(raw_cells)

    assert audit["complete"] is True
    assert audit["cross_method"]["complete"] is True


def test_surface_trial_identity_audit_rejects_different_trial_set():
    raw_cells = {
        ("D-SFT", 4, 2): [_row(0, True), _row(1, False)],
        ("D-SFT", 6, 2): [
            {**_row(0, True), "requested_semantic_age": 6},
            {**_row(2, False), "requested_semantic_age": 6},
        ],
    }

    audit = _trial_identity_audit(raw_cells)

    assert audit["complete"] is False
    assert audit["within_method"]["D-SFT"]["cells"]["age_6_ka_2"] == {
        "matches_reference": False,
        "trial_identities": 2,
        "missing_identity_count": 1,
        "unexpected_identity_count": 1,
    }


def test_deadline_summary_uses_age_axis_at_ka1_and_action_axis_at_age0():
    cells = [
        {
            "semantic_age_frames": age,
            "action_execution_horizon": horizon,
            "success_rate": rate,
            "wilson_95_low": low,
        }
        for age, horizon, rate, low in (
            (0, 1, 1.00, 0.95),
            (2, 1, 0.95, 0.91),
            (4, 1, 0.89, 0.85),
            (0, 2, 0.92, 0.90),
            (0, 4, 0.85, 0.80),
        )
    ]

    summary = _deadline_summary(
        cells,
        control_hz=20.0,
        expected_semantic_ages=(0, 2, 4),
        expected_action_horizons=(1, 2, 4),
    )

    assert summary["available"] is True
    assert summary["reason"] is None
    assert summary["levels"]["90"]["point_estimate"] == {
        "semantic_deadline_s": 0.1,
        "action_deadline_s": 0.1,
        "semantic_min_update_hz": 10.0,
        "action_min_replan_hz": 10.0,
        "asymmetry_ratio": 1.0,
    }
    assert summary["levels"]["90"]["wilson_lower_bound"] == {
        "semantic_deadline_s": 0.1,
        "action_deadline_s": 0.1,
        "semantic_min_update_hz": 10.0,
        "action_min_replan_hz": 10.0,
        "asymmetry_ratio": 1.0,
    }


def test_deadline_uses_registered_max_definition_and_reports_contiguous_sensitivity():
    cells = [
        {
            "semantic_age_frames": age,
            "action_execution_horizon": horizon,
            "success_rate": rate,
            "wilson_95_low": rate,
        }
        for age, horizon, rate in (
            (0, 1, 1.00),
            (2, 1, 0.80),
            (4, 1, 0.99),
            (0, 2, 0.80),
            (0, 4, 0.99),
        )
    ]

    summary = _deadline_summary(
        cells,
        control_hz=20.0,
        expected_semantic_ages=(0, 2, 4),
        expected_action_horizons=(1, 2, 4),
    )

    assert summary["levels"]["90"]["point_estimate"] == {
        "semantic_deadline_s": 0.2,
        "action_deadline_s": 0.2,
        "semantic_min_update_hz": 5.0,
        "action_min_replan_hz": 5.0,
        "asymmetry_ratio": 1.0,
    }
    assert summary["levels"]["90"]["contiguous_sensitivity"]["point_estimate"] == {
        "semantic_deadline_s": 0.0,
        "action_deadline_s": 0.05,
        "semantic_min_update_hz": None,
        "action_min_replan_hz": 20.0,
        "asymmetry_ratio": 0.0,
    }


def test_deadline_summary_marks_nonbaseline_development_cell_unavailable():
    summary = _deadline_summary(
        [
            {
                "semantic_age_frames": 4,
                "action_execution_horizon": 4,
                "success_rate": 1.0,
                "wilson_95_low": 0.5,
            }
        ],
        control_hz=20.0,
    )

    assert summary == {
        "available": False,
        "reason": "deadline requires the P(0,1) baseline cell",
        "baseline_success_rate": None,
        "levels": {},
        "axis_audit": {
            "complete": False,
            "semantic_axis": {
                "complete": False,
                "expected_ages": [0, 1, 2, 4, 6, 8, 12, 16],
                "observed_ages": [],
                "missing_ages": [0, 1, 2, 4, 6, 8, 12, 16],
            },
            "action_axis": {
                "complete": False,
                "expected_horizons": [1, 2, 4, 8, 16],
                "observed_horizons": [],
                "missing_horizons": [1, 2, 4, 8, 16],
            },
        },
    }


def test_deadline_summary_rejects_baseline_with_incomplete_registered_axes():
    summary = _deadline_summary(
        [
            {
                "semantic_age_frames": 0,
                "action_execution_horizon": 1,
                "success_rate": 1.0,
                "wilson_95_low": 0.9,
            },
            {
                "semantic_age_frames": 2,
                "action_execution_horizon": 1,
                "success_rate": 0.8,
                "wilson_95_low": 0.7,
            },
        ],
        control_hz=20.0,
        expected_semantic_ages=(0, 2, 4),
        expected_action_horizons=(1, 2),
    )

    assert summary["available"] is False
    assert summary["reason"] == (
        "deadline requires complete registered semantic and action axes"
    )
    assert summary["baseline_success_rate"] == 1.0
    assert summary["levels"] == {}
    assert summary["axis_audit"] == {
        "complete": False,
        "semantic_axis": {
            "complete": False,
            "expected_ages": [0, 2, 4],
            "observed_ages": [0, 2],
            "missing_ages": [4],
        },
        "action_axis": {
            "complete": False,
            "expected_horizons": [1, 2],
            "observed_horizons": [1],
            "missing_horizons": [2],
        },
    }


def test_matrix_audit_requires_every_registered_cell_and_method():
    summaries = [
        {
            "method": "D-SFT",
            "semantic_age_frames": age,
            "action_execution_horizon": horizon,
        }
        for age in (0, 2)
        for horizon in (1, 4)
    ]
    summaries.pop()

    audit = _matrix_audit(summaries, (0, 2), (1, 4), ("D-SFT", "D-PPO"))

    assert audit["D-SFT"] == {
        "complete": False,
        "missing_method": False,
        "unexpected_method": False,
        "missing_cells": [[2, 4]],
        "unexpected_cells": [],
    }
    assert audit["D-PPO"] == {
        "complete": False,
        "missing_method": True,
        "unexpected_method": False,
        "missing_cells": [[0, 1], [0, 4], [2, 1], [2, 4]],
        "unexpected_cells": [],
    }


def test_formal_trial_audit_cannot_be_weakened_by_cli_validation_floor():
    summaries = [
        {
            "method": "D-SFT",
            "semantic_age_frames": 0,
            "action_execution_horizon": 1,
            "trials": FORMAL_MIN_TRIALS_PER_CELL,
        },
        {
            "method": "D-PPO",
            "semantic_age_frames": 0,
            "action_execution_horizon": 1,
            "trials": 48,
        },
    ]

    audit = _formal_trial_audit(summaries)

    assert audit == {
        "complete": False,
        "required_trials_per_cell": 400,
        "minimum_observed_trials": 48,
        "underfilled_cells": [
            {
                "method": "D-PPO",
                "semantic_age_frames": 0,
                "action_execution_horizon": 1,
                "trials": 48,
            }
        ],
    }


def test_formal_trial_audit_accepts_registered_floor():
    audit = _formal_trial_audit(
        [
            {
                "method": "D-SFT",
                "semantic_age_frames": 0,
                "action_execution_horizon": 1,
                "trials": 400,
            },
            {
                "method": "D-PPO",
                "semantic_age_frames": 16,
                "action_execution_horizon": 16,
                "trials": 800,
            },
        ]
    )

    assert audit["complete"] is True
    assert audit["minimum_observed_trials"] == 400
    assert audit["underfilled_cells"] == []


def test_read_surface_manifest(tmp_path):
    manifest = tmp_path / "surface_cells.tsv"
    manifest.write_text(
        "method\tsemantic_age\taction_horizon\tjsonl\n"
        "D-PPO\t6\t4\t/results/age_6_ka_4/eval_trials.jsonl\n"
    )

    assert _read_manifest(manifest) == [
        CellSpec("D-PPO", 6, 4, Path("/results/age_6_ka_4/eval_trials.jsonl"))
    ]
    _, provenance = _read_manifest_with_provenance(manifest)
    assert provenance["complete"] is False
    assert "legacy manifest" in provenance["reason"]


def _provenance_record(method: str, policy: str, *, initial_sha: str = "initial"):
    return {
        "path": f"/{method}.tsv",
        "complete": True,
        "reason": None,
        "identity": {
            "schema_version": 3,
            "artifact_hash_algorithm": ARTIFACT_HASH_ALGORITHM,
            "method": method,
            "task_id_filter": "[0]",
            "policy_noise_seeds": "2026 2027",
            "policy_checkpoint_path": policy,
            "policy_checkpoint_sha256": f"sha-{policy}",
            "initial_checkpoint_path": "/initial",
            "initial_checkpoint_sha256": initial_sha,
            "backbone_path": "/backbone",
            "backbone_sha256": "backbone",
            "git_sha": "git",
            "worktree_sha256": "worktree",
            "action_prediction_horizon": 16,
            "control_hz": 20.0,
        },
    }


def test_manifest_provenance_allows_only_method_policy_checkpoint_to_differ():
    sft = _provenance_record("D-SFT", "/initial")
    sft["identity"]["policy_checkpoint_sha256"] = "initial"
    ppo = _provenance_record("D-PPO", "/ppo")

    audit = _manifest_provenance_audit(
        [sft, ppo], explicit_cell_count=0, control_hz=20.0
    )

    assert audit["complete"] is True
    assert audit["mismatches"] == []


def test_manifest_provenance_rejects_cross_method_initial_checkpoint_mismatch():
    sft = _provenance_record("D-SFT", "/initial")
    sft["identity"]["policy_checkpoint_sha256"] = "initial"
    ppo = _provenance_record("D-PPO", "/ppo", initial_sha="different")

    audit = _manifest_provenance_audit(
        [sft, ppo], explicit_cell_count=0, control_hz=20.0
    )

    assert audit["complete"] is False
    assert {item["field"] for item in audit["mismatches"]} == {
        "initial_checkpoint_sha256"
    }


def test_paired_surface_transition_includes_action_horizon_and_actual_age():
    reference = [_row(0, False), _row(1, True)]
    candidate = [_row(0, True, age=2, clipped=True), _row(1, False)]

    summary = _paired_transition(reference, candidate)

    assert summary["failure_to_success"] == 1
    assert summary["success_to_failure"] == 1
    assert summary["actual_age_contract_violation_count"] == 0
    assert summary["actual_age_mismatch_count"] == 0
    assert summary["trajectory_dependent_actual_age_mean_difference_count"] == 1
    assert summary["bootstrap_clip_mismatch_count"] == 1
    with pytest.raises(ValueError, match="sets differ"):
        _paired_transition(reference, [_row(0, True, horizon=4), _row(1, False)])


def test_paired_surface_transition_flags_nonbootstrap_actual_age_violation():
    reference = [_row(0, False)]
    candidate = [_row(0, True, age=2, clipped=False)]

    summary = _paired_transition(reference, candidate)

    assert summary["actual_age_contract_violation_count"] == 1
    assert summary["trajectory_dependent_actual_age_mean_difference_count"] == 1
