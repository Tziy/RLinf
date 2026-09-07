#!/usr/bin/env python3
"""Paired binary-outcome statistics for FDVLA fixed-trial evaluations."""

from __future__ import annotations

import math
from collections.abc import Iterable

PAIR_FIELDS = ("task_id", "trial_id", "semantic_age", "policy_noise_seed")
TRIAL_IDENTITY_FIELDS = ("task_id", "trial_id", "policy_noise_seed")


def trial_key(row: dict) -> tuple[int, int, int, int]:
    missing = [
        field
        for field in ("task_id", "trial_id", "policy_noise_seed")
        if field not in row
    ]
    if "requested_semantic_age" not in row and "semantic_age" not in row:
        missing.append("requested_semantic_age")
    if missing:
        raise ValueError(f"Trial row is missing pairing fields: {missing}")
    requested_age = row.get("requested_semantic_age", row["semantic_age"])
    return (
        int(row["task_id"]),
        int(row["trial_id"]),
        int(requested_age),
        int(row["policy_noise_seed"]),
    )


def index_unique_trials(rows: Iterable[dict]) -> dict[tuple[int, int, int, int], dict]:
    indexed = {}
    for row in rows:
        key = trial_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate paired trial key: {key}")
        indexed[key] = row
    return indexed


def trial_identity_key(row: dict) -> tuple[int, int, int]:
    missing = [field for field in TRIAL_IDENTITY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Trial row is missing identity fields: {missing}")
    return tuple(int(row[field]) for field in TRIAL_IDENTITY_FIELDS)


def index_unique_trial_identities(
    rows: Iterable[dict],
) -> dict[tuple[int, int, int], dict]:
    indexed = {}
    for row in rows:
        key = trial_identity_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate trial identity key: {key}")
        indexed[key] = row
    return indexed


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - radius, center + radius


def exact_mcnemar_p(failure_to_success: int, success_to_failure: int) -> float:
    discordant = failure_to_success + success_to_failure
    if discordant == 0:
        return 1.0
    tail = min(failure_to_success, success_to_failure)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    probability /= 2**discordant
    return min(1.0, 2.0 * probability)


def paired_binary_summary(
    reference_rows: Iterable[dict],
    candidate_rows: Iterable[dict],
    *,
    require_condition_match: bool = True,
):
    if require_condition_match:
        reference = index_unique_trials(reference_rows)
        candidate = index_unique_trials(candidate_rows)
    else:
        reference = index_unique_trial_identities(reference_rows)
        candidate = index_unique_trial_identities(candidate_rows)
    if set(reference) != set(candidate):
        missing_candidate = sorted(set(reference) - set(candidate))
        missing_reference = sorted(set(candidate) - set(reference))
        raise ValueError(
            "Paired trial sets differ: "
            f"missing_candidate={missing_candidate[:5]} "
            f"missing_reference={missing_reference[:5]}"
        )

    failure_to_success = 0
    success_to_failure = 0
    successes = 0
    requested_age_mismatches = 0
    actual_age_mismatches = 0
    action_horizon_mismatches = 0
    for key in sorted(reference):
        before = bool(reference[key]["success"])
        after = bool(candidate[key]["success"])
        successes += int(after)
        failure_to_success += int(not before and after)
        success_to_failure += int(before and not after)
        requested_age_mismatches += int(
            reference[key].get("requested_semantic_age")
            != candidate[key].get("requested_semantic_age")
        )
        actual_age_mismatches += int(
            reference[key].get("semantic_age") != candidate[key].get("semantic_age")
        )
        action_horizon_mismatches += int(
            reference[key].get("action_execution_horizon")
            != candidate[key].get("action_execution_horizon")
        )

    if require_condition_match and action_horizon_mismatches:
        raise ValueError(
            "Paired trial action-execution horizons differ: "
            f"mismatch_count={action_horizon_mismatches}"
        )

    trials = len(candidate)
    low, high = wilson_interval(successes, trials)
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
        "exact_mcnemar_p": exact_mcnemar_p(failure_to_success, success_to_failure),
        "formal_pairing": require_condition_match,
        "requested_semantic_age_mismatch_count": requested_age_mismatches,
        "actual_semantic_age_mismatch_count": actual_age_mismatches,
        "action_execution_horizon_mismatch_count": action_horizon_mismatches,
    }
