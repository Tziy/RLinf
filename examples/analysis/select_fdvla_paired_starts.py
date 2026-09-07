#!/usr/bin/env python3
"""Select paired C/D starts from pre-declared fixed-trial candidate tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-tsv", action="append", required=True, type=Path)
    parser.add_argument("--d-tsv", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-successes", type=int, default=13)
    parser.add_argument("--max-deviation", type=int, default=4)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--expected-trials", type=int, default=48)
    parser.add_argument("--expected-age", type=int, default=0)
    parser.add_argument("--expected-horizon", type=int, default=2)
    return parser.parse_args()


def load_rows(paths: list[Path], arm: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["arm"].removesuffix("-SFT") != arm:
                    continue
                row["source_tsv"] = str(path.resolve())
                rows.append(row)
    if not rows:
        raise SystemExit(f"No {arm} candidates found in: {paths}")
    return rows


def validate_trial_file(
    row: dict[str, str],
    *,
    expected_trials: int = 48,
    expected_age: int = 0,
    expected_horizon: int = 2,
) -> frozenset:
    path = Path(row["jsonl"])
    trials = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len(trials) != expected_trials:
        raise SystemExit(
            f"Expected {expected_trials} trials, got {len(trials)}: {path}"
        )
    identities = frozenset(
        (trial["task_id"], trial["trial_id"], trial["policy_noise_seed"])
        for trial in trials
    )
    if len(identities) != expected_trials:
        raise SystemExit(
            f"Expected {expected_trials} unique fixed-trial identities: {path}"
        )
    successes = sum(bool(trial["success"]) for trial in trials)
    if successes != int(row["successes"]):
        raise SystemExit(
            f"Success count mismatch for {path}: {successes} != {row['successes']}"
        )
    horizons = {trial["action_execution_horizon"] for trial in trials}
    if horizons != {expected_horizon}:
        raise SystemExit(f"Expected K={expected_horizon}, got {horizons}: {path}")
    requested_ages = {
        int(trial.get("requested_semantic_age", trial["semantic_age"]))
        for trial in trials
    }
    if requested_ages != {expected_age}:
        raise SystemExit(
            f"Expected exact semantic age {expected_age}, got {requested_ages}: {path}"
        )
    nonbootstrap_age_violations = sum(
        max(
            int(trial.get("semantic_requested_actual_age_mismatch_boundary_count", 0))
            - int(trial.get("semantic_bootstrap_clipped_boundary_count", 0)),
            0,
        )
        for trial in trials
    )
    if nonbootstrap_age_violations:
        raise SystemExit(
            f"Exact-age candidate has {nonbootstrap_age_violations} "
            f"non-bootstrap mismatches: {path}"
        )
    return identities


def choose_pair(
    c_rows: list[dict[str, str]],
    d_rows: list[dict[str, str]],
    *,
    target: int,
    max_deviation: int,
    max_gap: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Choose the best eligible pair rather than two independent optima."""

    eligible = []
    for c_row in c_rows:
        c_successes = int(c_row["successes"])
        c_deviation = abs(c_successes - target)
        for d_row in d_rows:
            d_successes = int(d_row["successes"])
            d_deviation = abs(d_successes - target)
            gap = abs(c_successes - d_successes)
            if (
                c_deviation > max_deviation
                or d_deviation > max_deviation
                or gap > max_gap
            ):
                continue
            key = (
                max(c_deviation, d_deviation),
                c_deviation + d_deviation,
                gap,
                int(c_row["step"]) + int(d_row["step"]),
                int(c_row["step"]),
                int(d_row["step"]),
                c_row["source_tsv"],
                d_row["source_tsv"],
            )
            eligible.append((key, c_row, d_row))
    if not eligible:
        c_counts = sorted(int(row["successes"]) for row in c_rows)
        d_counts = sorted(int(row["successes"]) for row in d_rows)
        raise SystemExit(
            "No eligible matched-start pair: "
            f"target={target}, max_deviation={max_deviation}, max_gap={max_gap}, "
            f"C={c_counts}, D={d_counts}"
        )
    _, c_row, d_row = min(eligible, key=lambda item: item[0])
    return c_row, d_row


def main() -> None:
    args = parse_args()
    candidates = {
        "C": load_rows(args.c_tsv, "C"),
        "D": load_rows(args.d_tsv, "D"),
    }
    reference_identities: frozenset | None = None
    for arm, rows in candidates.items():
        for row in rows:
            identities = validate_trial_file(
                row,
                expected_trials=args.expected_trials,
                expected_age=args.expected_age,
                expected_horizon=args.expected_horizon,
            )
            if reference_identities is None:
                reference_identities = identities
            elif identities != reference_identities:
                raise SystemExit(
                    f"Fixed-trial identities are not paired: {row['jsonl']}"
                )

    c_row, d_row = choose_pair(
        candidates["C"],
        candidates["D"],
        target=args.target_successes,
        max_deviation=args.max_deviation,
        max_gap=args.max_gap,
    )
    selected = {"C": c_row, "D": d_row}
    gap = abs(int(selected["C"]["successes"]) - int(selected["D"]["successes"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write(f"selection_target_successes={args.target_successes}\n")
        stream.write("selection_policy=joint_eligible_pair_v1\n")
        stream.write(f"selection_max_deviation={args.max_deviation}\n")
        stream.write(f"selection_max_gap={args.max_gap}\n")
        for arm in ("C", "D"):
            row = selected[arm]
            stream.write(f"{arm}_selected_step={row['step']}\n")
            stream.write(f"{arm}_selected_successes={row['successes']}\n")
            stream.write(f"{arm}_selected_trials={row['trials']}\n")
            stream.write(f"{arm}_selected_checkpoint={row['checkpoint']}\n")
            stream.write(f"{arm}_selection_jsonl={row['jsonl']}\n")
            stream.write(f"{arm}_selection_source_tsv={row['source_tsv']}\n")
            if row.get("checkpoint_sha256"):
                stream.write(
                    f"{arm}_selected_checkpoint_sha256={row['checkpoint_sha256']}\n"
                )
        stream.write(f"selected_start_gap={gap}\n")


if __name__ == "__main__":
    main()
