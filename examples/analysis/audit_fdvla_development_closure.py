#!/usr/bin/env python3
"""Verify the recorded FDVLA development pilot from its immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"passed": bool(passed), **evidence}


def audit_development_closure(
    *,
    run_dir: Path,
    training_health_path: Path,
    learning_curve_metadata_path: Path,
    surface_summary_path: Path,
    dsft_fresh_jsonl: Path,
    dppo_fresh_jsonl: Path,
    checkpoint_path: Path,
    expected_updates: int,
    expected_trials: int,
) -> dict[str, Any]:
    health = _load_json(training_health_path)
    curve_metadata = _load_json(learning_curve_metadata_path)
    surface = _load_json(surface_summary_path)

    step0_jsonl = run_dir / "eval_trials_step_0.jsonl"
    final_jsonl = run_dir / f"eval_trials_step_{expected_updates}.jsonl"
    required_paths = (
        step0_jsonl,
        final_jsonl,
        dsft_fresh_jsonl,
        dppo_fresh_jsonl,
        checkpoint_path,
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing closure artifacts: {missing_paths}")

    health_series = health.get("health", {})
    health_nonfinite = {
        name: record.get("nonfinite_count")
        for name, record in health_series.items()
        if record.get("nonfinite_count") != 0
    }
    gates: dict[str, dict[str, Any]] = {}
    gates["training_updates"] = _gate(
        health.get("completed_updates") == expected_updates
        and health.get("through_global_step") == expected_updates,
        expected=expected_updates,
        completed=health.get("completed_updates"),
        through_global_step=health.get("through_global_step"),
    )
    gates["finite_training_scalars"] = _gate(
        health.get("nonfinite_scalar_count") == 0
        and not health.get("nonfinite_scalars")
        and not health.get("missing_required_tags")
        and not health_nonfinite,
        aggregate_nonfinite_count=health.get("nonfinite_scalar_count"),
        per_series_nonfinite=health_nonfinite,
        missing_required_tags=health.get("missing_required_tags"),
    )
    gates["binary_decoupled_correctness_counters"] = _gate(
        health.get("cross_episode_packet_mismatch_count") == 0
        and health.get("semantic_replay_fingerprint_mismatch_log_count") == 0
        and health.get("local_vlm_forward_count") == 0
        and health.get("reward_model_invocation_count") == 0,
        cross_episode_packet_mismatch_count=health.get(
            "cross_episode_packet_mismatch_count"
        ),
        semantic_replay_fingerprint_mismatch_log_count=health.get(
            "semantic_replay_fingerprint_mismatch_log_count"
        ),
        local_vlm_forward_count=health.get("local_vlm_forward_count"),
        reward_model_invocation_count=health.get("reward_model_invocation_count"),
    )

    return_health = health_series.get("env/return", {})
    success_health = health_series.get("env/success_once", {})
    binary_summary_fields = (
        "count",
        "finite_count",
        "first",
        "last",
        "min",
        "max",
        "mean",
    )
    binary_summaries_match = bool(return_health) and all(
        return_health.get(field) == success_health.get(field)
        for field in binary_summary_fields
    )
    gates["binary_episode_return_contract"] = _gate(
        binary_summaries_match
        and 0.0 <= return_health.get("min", -1.0)
        and return_health.get("max", 2.0) <= 1.0
        and health.get("successful_episodes", 0) + health.get("failed_episodes", 0)
        == health.get("completed_episodes"),
        compared_fields=list(binary_summary_fields),
        return_summary={
            field: return_health.get(field) for field in binary_summary_fields
        },
        success_once_summary={
            field: success_health.get(field) for field in binary_summary_fields
        },
        completed_episodes=health.get("completed_episodes"),
        successful_episodes=health.get("successful_episodes"),
        failed_episodes=health.get("failed_episodes"),
    )

    pairing_audits = curve_metadata.get("pairing_audit", {})
    pairing_records = list(pairing_audits.values())
    gates["learning_curve_pairing"] = _gate(
        len(pairing_records) == 1
        and pairing_records[0].get("formal_pairing") is True
        and pairing_records[0].get("paired_condition_identity_complete") is True
        and pairing_records[0].get("paired_condition_identity_count")
        == expected_trials,
        runs=sorted(pairing_audits),
        expected_trials=expected_trials,
        observed_trials=(
            pairing_records[0].get("paired_condition_identity_count")
            if len(pairing_records) == 1
            else None
        ),
    )

    manifest_audit = surface.get("manifest_provenance_audit", {})
    trial_audit = surface.get("trial_identity_audit", {})
    cells = surface.get("cells", [])
    cells_by_method = {cell.get("method"): cell for cell in cells}
    paired_records = surface.get("paired_d_ppo_vs_d_sft", {})
    paired_record = next(iter(paired_records.values()), {})
    gates["fresh_surface_pairing"] = _gate(
        manifest_audit.get("complete") is True
        and not manifest_audit.get("mismatches")
        and trial_audit.get("complete") is True
        and set(cells_by_method) == {"D-SFT", "D-PPO"}
        and all(
            cell.get("trials") == expected_trials for cell in cells_by_method.values()
        )
        and all(
            cell.get("cross_episode_packet_mismatch_count") == 0
            and cell.get("local_vlm_forward_count") == 0
            for cell in cells_by_method.values()
        )
        and len(paired_records) == 1
        and paired_record.get("actual_age_contract_violation_count") == 0,
        methods=sorted(cells_by_method),
        trials={method: cell.get("trials") for method, cell in cells_by_method.items()},
        provenance_mismatches=manifest_audit.get("mismatches"),
        actual_age_contract_violation_count=paired_record.get(
            "actual_age_contract_violation_count"
        ),
        correctness_counters={
            method: {
                "cross_episode_packet_mismatch_count": cell.get(
                    "cross_episode_packet_mismatch_count"
                ),
                "local_vlm_forward_count": cell.get("local_vlm_forward_count"),
            }
            for method, cell in cells_by_method.items()
        },
    )
    gates["development_result_not_mislabeled_formal"] = _gate(
        surface.get("development_mode") is True
        and surface.get("formal_matrix_complete") is False
        and surface.get("formal_trial_audit", {}).get("complete") is False,
        development_mode=surface.get("development_mode"),
        formal_matrix_complete=surface.get("formal_matrix_complete"),
        formal_trial_audit_complete=surface.get("formal_trial_audit", {}).get(
            "complete"
        ),
    )

    step0_sha = _sha256(step0_jsonl)
    final_sha = _sha256(final_jsonl)
    dsft_fresh_sha = _sha256(dsft_fresh_jsonl)
    dppo_fresh_sha = _sha256(dppo_fresh_jsonl)
    gates["fresh_process_trial_reproduction"] = _gate(
        step0_sha == dsft_fresh_sha and final_sha == dppo_fresh_sha,
        training_step0_sha256=step0_sha,
        fresh_dsft_sha256=dsft_fresh_sha,
        training_final_sha256=final_sha,
        fresh_dppo_sha256=dppo_fresh_sha,
    )

    manifests = manifest_audit.get("manifests", [])
    dppo_manifest = next(
        (
            record
            for record in manifests
            if record.get("identity", {}).get("method") == "D-PPO"
        ),
        None,
    )
    checkpoint_sha = _sha256(checkpoint_path)
    recorded_checkpoint_path = (
        dppo_manifest.get("identity", {}).get("policy_checkpoint_path")
        if dppo_manifest
        else None
    )
    recorded_checkpoint_sha = (
        dppo_manifest.get("identity", {}).get("policy_checkpoint_sha256")
        if dppo_manifest
        else None
    )
    recorded_checkpoint_matches = isinstance(recorded_checkpoint_path, str) and (
        Path(recorded_checkpoint_path).resolve() == checkpoint_path.resolve()
    )
    gates["final_checkpoint_identity"] = _gate(
        dppo_manifest is not None
        and recorded_checkpoint_matches
        and recorded_checkpoint_sha == checkpoint_sha,
        path=str(checkpoint_path.resolve()),
        recorded_path=recorded_checkpoint_path,
        sha256=checkpoint_sha,
        recorded_sha256=recorded_checkpoint_sha,
    )

    complete = all(record["passed"] for record in gates.values())
    return {
        "schema_version": 1,
        "development_closure_complete": complete,
        "scientific_result_formal": False,
        "expected_updates": expected_updates,
        "expected_trials": expected_trials,
        "gates": gates,
        "remaining_formal_requirements": [
            "three paired training seeds",
            "at least 400 final fixed trials per evaluation",
            "paired C-PPO learning curves",
            "complete pre-registered semantic-age x action-horizon surface",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-health", type=Path, required=True)
    parser.add_argument("--learning-curve-metadata", type=Path, required=True)
    parser.add_argument("--surface-summary", type=Path, required=True)
    parser.add_argument("--dsft-fresh-jsonl", type=Path, required=True)
    parser.add_argument("--dppo-fresh-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, default=50)
    parser.add_argument("--expected-trials", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_development_closure(
        run_dir=args.run_dir.resolve(),
        training_health_path=args.training_health.resolve(),
        learning_curve_metadata_path=args.learning_curve_metadata.resolve(),
        surface_summary_path=args.surface_summary.resolve(),
        dsft_fresh_jsonl=args.dsft_fresh_jsonl.resolve(),
        dppo_fresh_jsonl=args.dppo_fresh_jsonl.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        expected_updates=args.expected_updates,
        expected_trials=args.expected_trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    if not report["development_closure_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
