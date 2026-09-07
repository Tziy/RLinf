#!/usr/bin/env python3
"""Audit and summarize a frozen-worktree FDVLA post-hoc paired pilot."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from examples.analysis.fdvla_statistics import (
    paired_binary_summary,
    wilson_interval,
)

ARMS = ("D-PPO-PosthocControl", "D-PPO-PosthocAug")
IDENTITY_FILES = (
    "git_sha.txt",
    "worktree_sha256.txt",
    "checkpoint_sha256.txt",
    "backbone_sha256.txt",
)
ALLOWED_CONFIG_DIFFERENCES = {
    "runner.logger.log_path",
    "runner.logger.experiment_name",
    "algorithm.posthoc_semantic_delay_augmentation.consistency_loss_weight",
    "algorithm.posthoc_semantic_delay_augmentation.force_replay_forward",
}
PARAMETER_FINGERPRINT_LABELS = {
    "dit_trainable": "DiT initialization fingerprint (trainable sampled SHA256)",
    "value_head_trainable": (
        "Value head initialization fingerprint (trainable sampled SHA256)"
    ),
    "all_trainable": "All trainable initialization fingerprint (sampled SHA256)",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            key: item
            for name, child in value.items()
            for key, item in _flatten(
                child, f"{prefix}.{name}" if prefix else str(name)
            ).items()
        }
    return {prefix: value}


def _config_differences(
    control: dict[str, Any], augmentation: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    left = _flatten(control)
    right = _flatten(augmentation)
    return {
        key: {"control": left.get(key), "augmentation": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _metadata_dir(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("fdvla_metadata_*"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one metadata directory in {run_dir}, got {matches}"
        )
    return matches[0]


def _evaluation_steps(run_dir: Path) -> dict[int, Path]:
    pattern = re.compile(r"eval_trials_step_(\d+)\.jsonl$")
    result = {}
    for path in run_dir.glob("eval_trials_step_*.jsonl"):
        match = pattern.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def _success_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "successes": successes,
        "trials": len(rows),
        "success_rate": successes / len(rows),
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def _actual_age_contract_violation_count(rows: list[dict[str, Any]]) -> int:
    violations = 0
    for row in rows:
        mismatches = int(
            row.get("semantic_requested_actual_age_mismatch_boundary_count", 0)
        )
        bootstrap = int(row.get("semantic_bootstrap_clipped_boundary_count", 0))
        violations += max(0, mismatches - bootstrap)
    return violations


def _posthoc_health_audit(
    health: dict[str, Any], expected_replay_fraction: float, arm: str
) -> dict[str, Any]:
    contract = health.get("posthoc_contract", {})
    replay_fraction = contract.get("replay_fraction_mean")
    checks = {
        "completed_updates": int(health.get("completed_updates", -1)) == 10,
        "posthoc_enabled": contract.get("enabled") is True,
        "replay_fraction": (
            replay_fraction is not None
            and math.isclose(
                float(replay_fraction),
                expected_replay_fraction,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ),
        "ppo_rows_excluded": contract.get("ppo_eligible_count_max") == 0,
        "finite_posthoc_logprobs": contract.get("nonfinite_logprob_count_max") == 0,
        "finite_all_scalars": health.get("nonfinite_scalar_count") == 0,
        "semantic_replay_matches": health.get(
            "semantic_replay_fingerprint_mismatch_log_count"
        )
        == 0,
        "cross_episode_packets_match": health.get("cross_episode_packet_mismatch_count")
        == 0,
        "reward_model_disabled": health.get("reward_model_invocation_count") == 0,
        "vlm_trainable_zero": health.get("trainable_vlm_parameter_count") == 0,
    }
    weighted = (
        contract.get("metrics", {})
        .get("train/posthoc/weighted_aux_loss", {})
        .get("max")
    )
    checks["weighted_auxiliary_signal"] = (
        weighted == 0
        if arm == "D-PPO-PosthocControl"
        else weighted is not None and float(weighted) > 0
    )
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "weighted_aux_loss_max": weighted,
        "replay_fraction_mean": replay_fraction,
    }


def _parameter_fingerprint_identity(
    run_logs: dict[str, str],
) -> dict[str, dict[str, Any]]:
    identity = {}
    for key, label in PARAMETER_FINGERPRINT_LABELS.items():
        pattern = re.compile(rf"{re.escape(label)}: ([0-9a-f]{{64}})")
        values = {
            arm: sorted(set(pattern.findall(run_logs.get(arm, "")))) for arm in ARMS
        }
        singletons = {
            arm: arm_values[0]
            for arm, arm_values in values.items()
            if len(arm_values) == 1
        }
        matched = len(singletons) == len(ARMS) and len(set(singletons.values())) == 1
        identity[key] = {
            "matched": matched,
            "value": next(iter(singletons.values())) if matched else values,
            "per_arm": values,
        }
    return identity


def summarize_pair(
    run_root: Path, *, expected_replay_fraction: float = 0.125
) -> dict[str, Any]:
    pair_sha = (run_root / "pair_worktree_sha256.txt").read_text().strip()
    arm_data: dict[str, Any] = {}
    metadata: dict[str, Path] = {}
    configs: dict[str, dict[str, Any]] = {}
    run_logs: dict[str, str] = {}

    for arm in ARMS:
        run_dir = run_root / arm
        metadata[arm] = _metadata_dir(run_dir)
        configs[arm] = yaml.safe_load(
            (metadata[arm] / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        run_logs[arm] = (run_dir / "launcher.log").read_text(encoding="utf-8")
        steps = _evaluation_steps(run_dir)
        if 0 not in steps or not steps:
            raise ValueError(f"{arm} lacks step-0/final fixed evaluation")
        final_step = max(steps)
        if final_step != 10:
            raise ValueError(
                f"{arm} final evaluation is step {final_step}, expected 10"
            )
        initial_rows = _load_jsonl(steps[0])
        final_rows = _load_jsonl(steps[final_step])
        health = _load_json(run_dir / "summary" / "fdvla_training_health.json")
        arm_data[arm] = {
            "initial": _success_summary(initial_rows),
            "final": _success_summary(final_rows),
            "initial_to_final": paired_binary_summary(initial_rows, final_rows),
            "actual_age_contract_violation_count": (
                _actual_age_contract_violation_count(initial_rows)
                + _actual_age_contract_violation_count(final_rows)
            ),
            "health": _posthoc_health_audit(health, expected_replay_fraction, arm),
            "process_elapsed_seconds": float(
                (run_dir / "process_time.txt").read_text().strip().split("=", 1)[1]
            ),
        }

    identity = {}
    for name in IDENTITY_FILES:
        values = {arm: (metadata[arm] / name).read_text().strip() for arm in ARMS}
        identity[name] = {
            "matched": len(set(values.values())) == 1,
            "value": values[ARMS[0]] if len(set(values.values())) == 1 else values,
        }
    identity["pair_worktree_sha256"] = {
        "matched": all(
            identity["worktree_sha256.txt"]["matched"]
            and (metadata[arm] / "worktree_sha256.txt").read_text().strip() == pair_sha
            for arm in ARMS
        ),
        "value": pair_sha,
    }
    identity.update(_parameter_fingerprint_identity(run_logs))

    differences = _config_differences(configs[ARMS[0]], configs[ARMS[1]])
    unexpected = sorted(set(differences) - ALLOWED_CONFIG_DIFFERENCES)
    missing_expected = sorted(ALLOWED_CONFIG_DIFFERENCES - set(differences))

    control_rows = _load_jsonl(run_root / ARMS[0] / "eval_trials_step_10.jsonl")
    augmentation_rows = _load_jsonl(run_root / ARMS[1] / "eval_trials_step_10.jsonl")
    final_pair = paired_binary_summary(control_rows, augmentation_rows)
    actual_age_violations = sum(
        _actual_age_contract_violation_count(rows)
        for rows in (control_rows, augmentation_rows)
    )
    complete = (
        all(item["matched"] for item in identity.values())
        and not unexpected
        and not missing_expected
        and all(arm_data[arm]["health"]["complete"] for arm in ARMS)
        and all(
            arm_data[arm]["actual_age_contract_violation_count"] == 0 for arm in ARMS
        )
        and actual_age_violations == 0
    )
    return {
        "complete": complete,
        "protocol": {
            "expected_updates": 10,
            "expected_replay_fraction": expected_replay_fraction,
            "arms": list(ARMS),
        },
        "identity": identity,
        "config_differences": differences,
        "unexpected_config_differences": unexpected,
        "missing_expected_config_differences": missing_expected,
        "arms": arm_data,
        "paired_final_augmentation_vs_control": final_pair,
        "paired_final_actual_age_contract_violation_count": actual_age_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-replay-fraction", type=float, default=0.125)
    args = parser.parse_args()

    summary = summarize_pair(
        args.run_root,
        expected_replay_fraction=args.expected_replay_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    if not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
