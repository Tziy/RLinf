#!/usr/bin/env python3
"""Summarize pre-registered FDVLA binary-PPO endpoints without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.analysis.fdvla_statistics import (
    index_unique_trials,
    paired_binary_summary,
    wilson_interval,
)

FORMAL_MIN_FINAL_TRIALS = 400
FORMAL_SEED_COUNT = 3
DEFAULT_UNSEEDED_METHODS = ("D-SFT",)
DEFAULT_SEEDED_METHODS = ("D-PPO", "C-PPO")
DEFAULT_CURVE_METHODS = ("D-PPO", "C-PPO")


@dataclass(frozen=True)
class EvalSpec:
    method: str
    seed: int | None
    path: Path


def _parse_eval(value: str) -> EvalSpec:
    try:
        label, path = value.split("=", 1)
        if ":" in label:
            method, seed_text = label.rsplit(":", 1)
            seed = int(seed_text)
        else:
            method = label
            seed = None
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid --eval {value!r}; expected METHOD[:SEED]=JSONL"
        ) from error
    if not method or not path:
        raise argparse.ArgumentTypeError(
            f"Invalid --eval {value!r}; METHOD and JSONL must not be empty"
        )
    return EvalSpec(method, seed, Path(path))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _curve_arrays(rows: list[dict], x_key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        ordered = sorted(rows, key=lambda row: float(row[x_key]))
        x = np.asarray([float(row[x_key]) for row in ordered], dtype=np.float64)
        y = np.asarray(
            [float(row["success_rate"]) for row in ordered], dtype=np.float64
        )
    except KeyError as error:
        raise ValueError(
            f"Learning curve is missing required column {error.args[0]!r}"
        ) from error
    if len(x) < 2:
        raise ValueError(f"Learning curve requires at least two points for {x_key}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Learning curve contains non-finite values for {x_key}")
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"Learning curve {x_key} must be strictly increasing")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("Learning-curve success rates must lie in [0, 1]")
    return x, y


def _curve_auc(rows: list[dict], x_key: str) -> float:
    x, y = _curve_arrays(rows, x_key)
    return float(np.trapz(y, x))


def _paired_curve_auc(
    reference_rows: list[dict], candidate_rows: list[dict], x_key: str
) -> dict:
    """Integrate two curves only over their shared measured x-domain."""
    reference_x, reference_y = _curve_arrays(reference_rows, x_key)
    candidate_x, candidate_y = _curve_arrays(candidate_rows, x_key)
    lower = max(float(reference_x[0]), float(candidate_x[0]))
    upper = min(float(reference_x[-1]), float(candidate_x[-1]))
    if upper <= lower:
        raise ValueError(f"Paired curves have no positive shared {x_key} interval")
    grid = np.unique(
        np.concatenate(
            (
                np.asarray([lower, upper]),
                reference_x[(reference_x > lower) & (reference_x < upper)],
                candidate_x[(candidate_x > lower) & (candidate_x < upper)],
            )
        )
    )
    reference_auc = float(np.trapz(np.interp(grid, reference_x, reference_y), grid))
    candidate_auc = float(np.trapz(np.interp(grid, candidate_x, candidate_y), grid))
    width = upper - lower
    return {
        "x_key": x_key,
        "shared_domain_min": lower,
        "shared_domain_max": upper,
        "shared_domain_width": width,
        "reference_auc": reference_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": candidate_auc - reference_auc,
        "normalized_reference_auc": reference_auc / width,
        "normalized_candidate_auc": candidate_auc / width,
        "normalized_auc_delta": (candidate_auc - reference_auc) / width,
    }


def _bootstrap_seed_mean(values, seed=20260820, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.size != FORMAL_SEED_COUNT:
        raise ValueError(
            f"Formal seed summary requires exactly three seeds, got {values.size}"
        )
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(samples, len(values)), replace=True).mean(1)
    return {
        "mean": float(values.mean()),
        "bootstrap_95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_95_high": float(np.quantile(draws, 0.975)),
    }


def _eval_condition(rows: list[dict]) -> tuple[int, int] | None:
    conditions = set()
    for row in rows:
        if "action_execution_horizon" not in row:
            return None
        if "requested_semantic_age" not in row and "semantic_age" not in row:
            return None
        conditions.add(
            (
                int(row.get("requested_semantic_age", row["semantic_age"])),
                int(row["action_execution_horizon"]),
            )
        )
    return next(iter(conditions)) if len(conditions) == 1 else None


def _final_eval_audit(
    method_seed_rows: dict[tuple[str, int | None], list[dict]],
    *,
    required_unseeded_methods: tuple[str, ...] | list[str],
    required_seeded_methods: tuple[str, ...] | list[str],
) -> dict:
    unseeded = set(required_unseeded_methods)
    seeded = set(required_seeded_methods)
    if unseeded & seeded:
        raise ValueError("A formal method cannot be both seeded and unseeded")
    expected_methods = unseeded | seeded
    observed_methods = {method for method, _seed in method_seed_rows}
    missing_methods = sorted(expected_methods - observed_methods)
    unexpected_methods = sorted(observed_methods - expected_methods)
    invalid_method_seed_shapes = []
    seed_sets = {}
    for method in sorted(expected_methods & observed_methods):
        method_seeds = {
            seed
            for candidate_method, seed in method_seed_rows
            if candidate_method == method
        }
        if method in unseeded:
            if method_seeds != {None}:
                invalid_method_seed_shapes.append(
                    {
                        "method": method,
                        "expected": "one unseeded evaluation",
                        "seeds": sorted(
                            seed for seed in method_seeds if seed is not None
                        ),
                    }
                )
        else:
            if None in method_seeds or len(method_seeds) != FORMAL_SEED_COUNT:
                invalid_method_seed_shapes.append(
                    {
                        "method": method,
                        "expected": f"exactly {FORMAL_SEED_COUNT} seeded evaluations",
                        "seeds": sorted(
                            seed for seed in method_seeds if seed is not None
                        ),
                        "contains_unseeded": None in method_seeds,
                    }
                )
            seed_sets[method] = {seed for seed in method_seeds if seed is not None}
    incompatible_seed_sets = []
    if seed_sets:
        reference_method = sorted(seed_sets)[0]
        reference = seed_sets[reference_method]
        for method in sorted(seed_sets):
            if seed_sets[method] != reference:
                incompatible_seed_sets.append(
                    {
                        "reference_method": reference_method,
                        "reference_seeds": sorted(reference),
                        "method": method,
                        "seeds": sorted(seed_sets[method]),
                    }
                )

    underfilled = []
    invalid_conditions = []
    conditions = {}
    for (method, seed), rows in sorted(
        method_seed_rows.items(),
        key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or -1),
    ):
        if len(rows) < FORMAL_MIN_FINAL_TRIALS:
            underfilled.append({"method": method, "seed": seed, "trials": len(rows)})
        condition = _eval_condition(rows)
        if condition is None:
            invalid_conditions.append({"method": method, "seed": seed})
        else:
            conditions[(method, seed)] = condition
    condition_values = set(conditions.values())
    cross_method_condition_mismatch = len(condition_values) > 1
    return {
        "complete": not (
            missing_methods
            or unexpected_methods
            or invalid_method_seed_shapes
            or incompatible_seed_sets
            or underfilled
            or invalid_conditions
            or cross_method_condition_mismatch
        ),
        "required_trials_per_evaluation": FORMAL_MIN_FINAL_TRIALS,
        "required_seed_count": FORMAL_SEED_COUNT,
        "required_unseeded_methods": sorted(unseeded),
        "required_seeded_methods": sorted(seeded),
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "invalid_method_seed_shapes": invalid_method_seed_shapes,
        "incompatible_seed_sets": incompatible_seed_sets,
        "underfilled_evaluations": underfilled,
        "invalid_or_mixed_conditions": invalid_conditions,
        "cross_method_condition_mismatch": cross_method_condition_mismatch,
        "conditions": {
            f"{method}/{'shared' if seed is None else f'seed_{seed}'}": list(condition)
            for (method, seed), condition in sorted(
                conditions.items(),
                key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or -1),
            )
        },
    }


def _curve_pairing_metadata_audit(path: Path | None, required_keys: set[str]) -> dict:
    if path is None or not path.is_file():
        return {
            "complete": False,
            "path": str(path) if path is not None else None,
            "missing_runs": sorted(required_keys),
            "nonformal_runs": [],
            "missing_condition_fingerprints": sorted(required_keys),
            "condition_fingerprint_mismatches": [],
            "reference_condition_identity": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    audits = payload.get("pairing_audit", {})
    missing = sorted(required_keys - set(audits))
    nonformal = sorted(
        key
        for key in required_keys & set(audits)
        if not audits[key].get("formal_pairing")
    )
    present = required_keys & set(audits)
    missing_condition_fingerprints = sorted(
        key
        for key in present
        if not audits[key].get("paired_condition_identity_complete")
        or not audits[key].get("paired_condition_identity_sha256")
        or int(audits[key].get("paired_condition_identity_count", 0)) <= 0
    )
    fingerprinted = sorted(present - set(missing_condition_fingerprints))
    reference_condition_identity = None
    condition_fingerprint_mismatches = []
    if fingerprinted:
        reference_key = fingerprinted[0]
        reference_condition_identity = {
            "run": reference_key,
            "count": int(audits[reference_key]["paired_condition_identity_count"]),
            "sha256": audits[reference_key]["paired_condition_identity_sha256"],
        }
        reference_value = (
            reference_condition_identity["count"],
            reference_condition_identity["sha256"],
        )
        for key in fingerprinted[1:]:
            candidate_value = (
                int(audits[key]["paired_condition_identity_count"]),
                audits[key]["paired_condition_identity_sha256"],
            )
            if candidate_value != reference_value:
                condition_fingerprint_mismatches.append(
                    {
                        "reference_run": reference_key,
                        "reference_count": reference_value[0],
                        "reference_sha256": reference_value[1],
                        "run": key,
                        "count": candidate_value[0],
                        "sha256": candidate_value[1],
                    }
                )
    return {
        "complete": not (
            missing
            or nonformal
            or missing_condition_fingerprints
            or condition_fingerprint_mismatches
        ),
        "path": str(path.resolve()),
        "missing_runs": missing,
        "nonformal_runs": nonformal,
        "missing_condition_fingerprints": missing_condition_fingerprints,
        "condition_fingerprint_mismatches": condition_fingerprint_mismatches,
        "reference_condition_identity": reference_condition_identity,
    }


def _learning_curve_audit(
    grouped: dict[tuple[str, int], list[dict]],
    *,
    required_methods: tuple[str, ...] | list[str],
    expected_seeds: set[int],
    metadata_path: Path | None,
) -> dict:
    required = set(required_methods)
    observed = {method for method, _seed in grouped}
    missing_methods = sorted(required - observed)
    unexpected_methods = sorted(observed - required)
    invalid_seed_sets = []
    invalid_curves = []
    for method in sorted(required & observed):
        seeds = {
            seed for candidate_method, seed in grouped if candidate_method == method
        }
        if seeds != expected_seeds:
            invalid_seed_sets.append(
                {
                    "method": method,
                    "expected_seeds": sorted(expected_seeds),
                    "seeds": sorted(seeds),
                }
            )
        for seed in sorted(seeds):
            rows = grouped[(method, seed)]
            try:
                frame_x, _ = _curve_arrays(rows, "environment_frames")
                wallclock_x, _ = _curve_arrays(rows, "wallclock_s")
                training_wallclock_x, _ = _curve_arrays(
                    rows, "training_wallclock_s"
                )
                if (
                    frame_x[0] != 0.0
                    or wallclock_x[0] != 0.0
                    or training_wallclock_x[0] != 0.0
                ):
                    raise ValueError("learning curve must start at frame/time zero")
            except ValueError as error:
                invalid_curves.append(
                    {"method": method, "seed": seed, "reason": str(error)}
                )
    required_keys = {
        f"{method}/seed_{seed}" for method in required for seed in expected_seeds
    }
    metadata_audit = _curve_pairing_metadata_audit(metadata_path, required_keys)
    return {
        "complete": not (
            missing_methods
            or unexpected_methods
            or invalid_seed_sets
            or invalid_curves
            or not metadata_audit["complete"]
        ),
        "required_methods": sorted(required),
        "expected_seeds": sorted(expected_seeds),
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "invalid_seed_sets": invalid_seed_sets,
        "invalid_curves": invalid_curves,
        "pairing_metadata_audit": metadata_audit,
    }


def _summary_for_rows(rows: list[dict]) -> dict:
    successes = sum(bool(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def _paired_final_comparisons(
    method_seed_rows: dict[tuple[str, int | None], list[dict]],
    *,
    require_condition_match: bool,
) -> dict:
    comparisons = {}
    candidate_keys = sorted(
        (key for key in method_seed_rows if key[0] == "D-PPO"),
        key=lambda key: (key[1] is None, key[1] or -1),
    )
    for reference_method in (
        "D-SFT",
        "C-PPO",
        "D-PPO-Fresh",
        "D-PPO-NoAge",
        "D-PPO-NoHistory",
    ):
        per_seed = {}
        deltas = []
        for _, seed in candidate_keys:
            reference_key = (
                (reference_method, seed)
                if (reference_method, seed) in method_seed_rows
                else (reference_method, None)
            )
            if reference_key not in method_seed_rows:
                continue
            summary = paired_binary_summary(
                method_seed_rows[reference_key],
                method_seed_rows[("D-PPO", seed)],
                require_condition_match=require_condition_match,
            )
            reference_rate = _summary_for_rows(method_seed_rows[reference_key])[
                "success_rate"
            ]
            delta = summary["success_rate"] - reference_rate
            summary["reference_success_rate"] = reference_rate
            summary["success_rate_delta"] = delta
            label = "shared" if seed is None else f"seed_{seed}"
            per_seed[label] = summary
            if seed is not None:
                deltas.append(delta)
        if per_seed:
            record = {"per_seed": per_seed}
            if len(deltas) == FORMAL_SEED_COUNT:
                record["three_seed_success_rate_delta"] = _bootstrap_seed_mean(deltas)
            comparisons[f"D-PPO_vs_{reference_method}"] = record
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        type=_parse_eval,
        metavar="METHOD[:SEED]=JSONL",
        help=(
            "Final fixed-trial JSONL. SFT methods are unseeded; trained methods "
            "use one entry per training seed."
        ),
    )
    parser.add_argument("--learning-curves", type=Path)
    parser.add_argument("--learning-curve-metadata", type=Path)
    parser.add_argument(
        "--required-unseeded-methods",
        nargs="+",
        default=list(DEFAULT_UNSEEDED_METHODS),
    )
    parser.add_argument(
        "--required-seeded-methods",
        nargs="+",
        default=list(DEFAULT_SEEDED_METHODS),
    )
    parser.add_argument(
        "--required-curve-methods",
        nargs="+",
        default=list(DEFAULT_CURVE_METHODS),
    )
    parser.add_argument(
        "--allow-condition-mismatch",
        action="store_true",
        help="Development-only pairing that reports, but does not accept, mismatches.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit development subsets; output remains marked non-formal.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.allow_condition_mismatch and not args.allow_incomplete:
        raise ValueError("--allow-condition-mismatch requires --allow-incomplete")

    method_seed_rows = {}
    for spec in args.eval:
        key = (spec.method, spec.seed)
        if key in method_seed_rows:
            raise ValueError(f"Duplicate final evaluation input: {key}")
        rows = _read_jsonl(spec.path)
        if not rows:
            raise ValueError(f"Final evaluation is empty: {spec.path}")
        index_unique_trials(rows)
        method_seed_rows[key] = rows

    final_eval_audit = _final_eval_audit(
        method_seed_rows,
        required_unseeded_methods=args.required_unseeded_methods,
        required_seeded_methods=args.required_seeded_methods,
    )
    if not final_eval_audit["complete"] and not args.allow_incomplete:
        raise ValueError(f"Formal final-evaluation audit failed: {final_eval_audit}")

    summaries = {}
    for (method, seed), rows in sorted(
        method_seed_rows.items(),
        key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or -1),
    ):
        label = "shared" if seed is None else f"seed_{seed}"
        summaries.setdefault(method, {"evaluations": {}})["evaluations"][label] = (
            _summary_for_rows(rows)
        )
    for method, record in summaries.items():
        seeded_rates = [
            evaluation["success_rate"]
            for label, evaluation in record["evaluations"].items()
            if label.startswith("seed_")
        ]
        if len(seeded_rates) == FORMAL_SEED_COUNT:
            record["three_seed_final_success"] = _bootstrap_seed_mean(seeded_rates)

    comparisons = _paired_final_comparisons(
        method_seed_rows,
        require_condition_match=not args.allow_condition_mismatch,
    )

    curves = {}
    grouped = {}
    if args.learning_curves:
        with args.learning_curves.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            grouped.setdefault((row["method"], int(row["seed"])), []).append(row)
        for (method, seed), method_seed_rows_for_curve in sorted(grouped.items()):
            curves[f"{method}/seed_{seed}"] = {
                "frames_auc": _curve_auc(
                    method_seed_rows_for_curve, "environment_frames"
                ),
                "wallclock_auc": _curve_auc(method_seed_rows_for_curve, "wallclock_s"),
                "training_wallclock_auc": _curve_auc(
                    method_seed_rows_for_curve, "training_wallclock_s"
                ),
            }

    expected_seed_sets = [
        {
            seed
            for candidate_method, seed in method_seed_rows
            if candidate_method == method and seed is not None
        }
        for method in args.required_seeded_methods
    ]
    expected_seeds = expected_seed_sets[0] if expected_seed_sets else set()
    metadata_path = args.learning_curve_metadata
    if metadata_path is None and args.learning_curves is not None:
        metadata_path = args.learning_curves.with_suffix(
            args.learning_curves.suffix + ".metadata.json"
        )
    learning_curve_audit = _learning_curve_audit(
        grouped,
        required_methods=args.required_curve_methods,
        expected_seeds=expected_seeds,
        metadata_path=metadata_path,
    )
    if not learning_curve_audit["complete"] and not args.allow_incomplete:
        raise ValueError(f"Formal learning-curve audit failed: {learning_curve_audit}")

    auc_comparison = {"reference": "C-PPO", "candidate": "D-PPO", "per_seed": {}}
    common_auc_seeds = sorted(
        {seed for method, seed in grouped if method == "D-PPO"}
        & {seed for method, seed in grouped if method == "C-PPO"}
    )
    wallclock_deltas = []
    normalized_wallclock_deltas = []
    training_wallclock_deltas = []
    normalized_training_wallclock_deltas = []
    frame_deltas = []
    normalized_frame_deltas = []
    for seed in common_auc_seeds:
        wallclock = _paired_curve_auc(
            grouped[("C-PPO", seed)], grouped[("D-PPO", seed)], "wallclock_s"
        )
        training_wallclock = _paired_curve_auc(
            grouped[("C-PPO", seed)],
            grouped[("D-PPO", seed)],
            "training_wallclock_s",
        )
        frames = _paired_curve_auc(
            grouped[("C-PPO", seed)],
            grouped[("D-PPO", seed)],
            "environment_frames",
        )
        auc_comparison["per_seed"][f"seed_{seed}"] = {
            "wallclock": wallclock,
            "training_wallclock": training_wallclock,
            "environment_frames": frames,
        }
        wallclock_deltas.append(wallclock["auc_delta"])
        training_wallclock_deltas.append(training_wallclock["auc_delta"])
        normalized_training_wallclock_deltas.append(
            training_wallclock["normalized_auc_delta"]
        )
        normalized_wallclock_deltas.append(wallclock["normalized_auc_delta"])
        frame_deltas.append(frames["auc_delta"])
        normalized_frame_deltas.append(frames["normalized_auc_delta"])
    if len(common_auc_seeds) == FORMAL_SEED_COUNT:
        auc_comparison["three_seed"] = {
            "training_wallclock_auc_delta": _bootstrap_seed_mean(
                training_wallclock_deltas
            ),
            "normalized_training_wallclock_auc_delta": _bootstrap_seed_mean(
                normalized_training_wallclock_deltas
            ),
            "wallclock_auc_delta": _bootstrap_seed_mean(wallclock_deltas),
            "normalized_wallclock_auc_delta": _bootstrap_seed_mean(
                normalized_wallclock_deltas
            ),
            "environment_frame_auc_delta": _bootstrap_seed_mean(frame_deltas),
            "normalized_environment_frame_auc_delta": _bootstrap_seed_mean(
                normalized_frame_deltas
            ),
        }

    development_mode = bool(args.allow_incomplete or args.allow_condition_mismatch)
    output = {
        "pre_registered_primary_endpoints": {
            "final_success": "D-PPO minus D-SFT",
            "wallclock_learning_curve_auc": (
                "D-PPO minus C-PPO on shared pure-training time domain; "
                "raw end-to-end wallclock is secondary"
            ),
            "delay_degradation": "fixed actual semantic age 0,2,4,6,8,12",
        },
        "development_mode": development_mode,
        "formal_statistical_summary_complete": (
            not development_mode
            and final_eval_audit["complete"]
            and learning_curve_audit["complete"]
        ),
        "final_evaluation_audit": final_eval_audit,
        "learning_curve_audit": learning_curve_audit,
        "method_summaries": summaries,
        "paired_comparisons": comparisons,
        "learning_curve_auc": curves,
        "primary_auc_comparison": auc_comparison,
        "reporting_rule": "All configured tasks, trials, seeds, and checkpoints retained.",
        "formal_pairing": not args.allow_condition_mismatch,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fdvla_results_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
