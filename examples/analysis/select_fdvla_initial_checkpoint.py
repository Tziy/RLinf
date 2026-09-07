#!/usr/bin/env python3
"""Select an FDVLA development initialization by a pre-registered rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from examples.analysis.summarize_fdvla_realtime_surface import (
    _read_jsonl,
    _read_manifest_with_provenance,
    _surface_timing,
    _trial_identity,
    _validate_cell,
)


def _parse_candidate(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid candidate {value!r}; expected NAME=SURFACE_MANIFEST"
        ) from error
    if not name or not path:
        raise argparse.ArgumentTypeError("Candidate name and path must not be empty")
    return name, Path(path)


def _load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        manifest = yaml.safe_load(source)
    return manifest["development_initial_checkpoint_screen"]


def _candidate_result(name: str, path: Path, protocol: dict) -> tuple[dict, set]:
    specs, provenance = _read_manifest_with_provenance(path)
    if len(specs) != 1:
        raise ValueError(f"Candidate {name} must contain exactly one surface cell")
    if not provenance["complete"]:
        raise ValueError(f"Candidate {name} has incomplete provenance: {provenance}")

    spec = specs[0]
    condition = protocol["paired_condition"]
    if spec.method != "D-SFT":
        raise ValueError(f"Candidate {name} must use D-SFT, got {spec.method}")
    if (spec.semantic_age, spec.action_horizon) != (
        condition["semantic_age_frames"],
        condition["action_execution_horizon"],
    ):
        raise ValueError(f"Candidate {name} has the wrong realtime condition")

    rows = _read_jsonl(spec.path)
    summary = _validate_cell(spec, rows, min_trials=condition["unique_trials"])
    if summary["trials"] != condition["unique_trials"]:
        raise ValueError(
            f"Candidate {name} must have exactly the registered trial count"
        )
    identities = {_trial_identity(row) for row in rows}
    if len(identities) != len(rows):
        raise ValueError(f"Candidate {name} contains duplicate trial identities")
    expected_noise = condition["policy_noise_seed"]
    if {identity[2] for identity in identities} != {expected_noise}:
        raise ValueError(f"Candidate {name} uses the wrong policy-noise seed")
    if {identity[0] for identity in identities} != {condition["task_id"]}:
        raise ValueError(f"Candidate {name} uses the wrong task")

    timing = _surface_timing(
        spec,
        successes=summary["successes"],
        trials=summary["trials"],
    )
    nonbootstrap_mismatches = (
        summary["requested_actual_age_mismatch_boundaries"]
        - summary["bootstrap_clipped_boundaries"]
    )
    correctness = protocol["selection_rule"]["correctness_requirements"]
    observed_correctness = {
        "nonbootstrap_requested_actual_age_mismatches": nonbootstrap_mismatches,
        "cross_episode_packet_mismatches": timing[
            "cross_episode_packet_mismatch_count"
        ],
        "local_vlm_forward_count": timing["local_vlm_forward_count"],
    }
    if observed_correctness != correctness:
        raise ValueError(
            f"Candidate {name} violates correctness requirements: "
            f"observed={observed_correctness}, required={correctness}"
        )

    identity = provenance["identity"]
    if identity["policy_noise_seeds"] != str(expected_noise):
        raise ValueError(f"Candidate {name} manifest records the wrong noise seed")
    if identity["action_prediction_horizon"] != condition["action_prediction_horizon"]:
        raise ValueError(f"Candidate {name} uses the wrong prediction horizon")
    checkpoint_path = Path(identity["initial_checkpoint_path"])
    if not checkpoint_path.is_absolute():
        raise ValueError(f"Candidate {name} checkpoint path is not absolute")
    if identity["initial_checkpoint_path"] != identity["policy_checkpoint_path"]:
        raise ValueError(f"Candidate {name} policy and initial checkpoint differ")
    if identity["initial_checkpoint_sha256"] != identity["policy_checkpoint_sha256"]:
        raise ValueError(f"Candidate {name} checkpoint hashes differ")

    target = protocol["selection_rule"]["target_success_rate"]
    preferred_low, preferred_high = protocol["selection_rule"]["preferred_interval"]
    return (
        {
            "name": name,
            "surface_manifest": str(path.resolve()),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": identity["initial_checkpoint_sha256"],
            "trials": summary["trials"],
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "wilson_95_low": summary["wilson_95_low"],
            "wilson_95_high": summary["wilson_95_high"],
            "absolute_distance_to_target": abs(summary["success_rate"] - target),
            "inside_preferred_interval": (
                preferred_low <= summary["success_rate"] <= preferred_high
            ),
            "correctness": observed_correctness,
            "timing": timing,
            "git_sha": identity["git_sha"],
            "worktree_sha256": identity["worktree_sha256"],
            "backbone_path": identity["backbone_path"],
            "backbone_sha256": identity["backbone_sha256"],
        },
        identities,
    )


def select_candidates(protocol_path: Path, candidates: dict[str, Path]) -> dict:
    protocol = _load_protocol(protocol_path)
    expected = list(protocol["candidates"])
    if set(candidates) != set(expected):
        raise ValueError(
            f"Candidate set must exactly match registration; "
            f"expected={expected}, observed={sorted(candidates)}"
        )
    formal_noise_seeds = set(protocol["isolation"]["formal_eval_policy_noise_seeds"])
    selection_noise_seed = protocol["paired_condition"]["policy_noise_seed"]
    if selection_noise_seed in formal_noise_seeds:
        raise ValueError("Selection policy-noise seed overlaps formal evaluation")

    results = []
    reference_identities = None
    reference_provenance = None
    for name in expected:
        result, identities = _candidate_result(name, candidates[name], protocol)
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError(f"Candidate {name} does not share paired trial identities")
        provenance = tuple(
            result[field]
            for field in (
                "git_sha",
                "worktree_sha256",
                "backbone_path",
                "backbone_sha256",
            )
        )
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ValueError(f"Candidate {name} does not share runtime provenance")
        results.append(result)

    tie_order = protocol["selection_rule"]["tie_break_order"]
    tie_rank = {name: rank for rank, name in enumerate(tie_order)}
    ordered = sorted(
        results,
        key=lambda item: (
            item["absolute_distance_to_target"],
            tie_rank[item["name"]],
        ),
    )
    selected = ordered[0]
    return {
        "schema_version": 1,
        "development_only": True,
        "protocol_path": str(protocol_path.resolve()),
        "paired_condition": protocol["paired_condition"],
        "selection_rule": protocol["selection_rule"],
        "formal_eval_policy_noise_seeds": sorted(formal_noise_seeds),
        "paired_trial_identity_count": len(reference_identities or ()),
        "candidates": results,
        "ranking": [item["name"] for item in ordered],
        "selected_candidate": selected["name"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        type=_parse_candidate,
        metavar="NAME=SURFACE_MANIFEST",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("Duplicate candidate name")
    report = select_candidates(args.protocol, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
