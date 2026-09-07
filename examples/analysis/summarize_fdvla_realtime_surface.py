#!/usr/bin/env python3
"""Summarize the paired semantic-age x action-replanning FDVLA surface."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.analysis.fdvla_statistics import exact_mcnemar_p, wilson_interval

DEFAULT_SEMANTIC_AGES = (0, 1, 2, 4, 6, 8, 12, 16)
DEFAULT_ACTION_HORIZONS = (1, 2, 4, 8, 16)
DEFAULT_METHODS = ("D-SFT", "D-PPO")
FORMAL_MIN_TRIALS_PER_CELL = 400
MANIFEST_PROVENANCE_FIELDS = (
    "schema_version",
    "artifact_hash_algorithm",
    "method",
    "task_id_filter",
    "policy_noise_seeds",
    "policy_checkpoint_path",
    "policy_checkpoint_sha256",
    "initial_checkpoint_path",
    "initial_checkpoint_sha256",
    "backbone_path",
    "backbone_sha256",
    "git_sha",
    "worktree_sha256",
    "action_prediction_horizon",
    "control_hz",
)
SHARED_PROVENANCE_FIELDS = (
    "artifact_hash_algorithm",
    "task_id_filter",
    "policy_noise_seeds",
    "initial_checkpoint_path",
    "initial_checkpoint_sha256",
    "backbone_path",
    "backbone_sha256",
    "git_sha",
    "worktree_sha256",
    "action_prediction_horizon",
    "control_hz",
)
SURFACE_TIMING_TAGS = {
    "eval_wall_clock_s": "time/rollout/evaluate",
    "rollout_predict_s": "time/rollout/predict",
    "semantic_fetch_s": "time/rollout/predict/semantic_fetch",
    "control_frames": "time/rollout/profile/control_frame_count",
    "local_vlm_forward_count": "time/rollout/profile/local_vlm_forward_count",
    "cross_episode_packet_mismatch_count": (
        "time/rollout/profile/cross_episode_packet_mismatch_count"
    ),
}


@dataclass(frozen=True)
class CellSpec:
    method: str
    semantic_age: int
    action_horizon: int
    path: Path


def _parse_cell(value: str) -> CellSpec:
    condition, path = value.split("=", 1)
    method, semantic_age, action_horizon = condition.split(":", 2)
    return CellSpec(method, int(semantic_age), int(action_horizon), Path(path))


def _read_manifest_with_provenance(path: Path) -> tuple[list[CellSpec], dict]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"method", "semantic_age", "action_horizon", "jsonl"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Surface manifest {path} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)
        specs = [
            CellSpec(
                row["method"],
                int(row["semantic_age"]),
                int(row["action_horizon"]),
                Path(row["jsonl"]),
            )
            for row in rows
        ]
    missing_provenance = [
        field
        for field in MANIFEST_PROVENANCE_FIELDS
        if field not in (reader.fieldnames or ())
    ]
    if missing_provenance:
        return specs, {
            "path": str(path.resolve()),
            "complete": False,
            "reason": f"legacy manifest missing provenance columns: {missing_provenance}",
            "identity": {},
        }
    if not rows:
        return specs, {
            "path": str(path.resolve()),
            "complete": False,
            "reason": "manifest contains no surface cells",
            "identity": {},
        }

    identity = {}
    for field in MANIFEST_PROVENANCE_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"Surface manifest {path} mixes provenance field {field}: {sorted(values)}"
            )
        identity[field] = values.pop()
    identity["schema_version"] = int(identity["schema_version"])
    identity["action_prediction_horizon"] = int(identity["action_prediction_horizon"])
    identity["control_hz"] = float(identity["control_hz"])
    if identity["schema_version"] != 3:
        raise ValueError(
            f"Unsupported surface manifest schema {identity['schema_version']} in {path}"
        )
    if identity["artifact_hash_algorithm"] != "file-raw+dir-logical-deref-sha256-v1":
        raise ValueError(
            f"Unsupported artifact hash algorithm in {path}: "
            f"{identity['artifact_hash_algorithm']}"
        )
    if identity["action_prediction_horizon"] != 16:
        raise ValueError(
            f"Surface manifest {path} does not use the registered 16-step prediction horizon"
        )
    if identity["method"] == "D-SFT" and (
        identity["policy_checkpoint_path"] != identity["initial_checkpoint_path"]
        or identity["policy_checkpoint_sha256"] != identity["initial_checkpoint_sha256"]
    ):
        raise ValueError(
            f"D-SFT surface manifest {path} does not evaluate the registered initial checkpoint"
        )
    return specs, {
        "path": str(path.resolve()),
        "complete": True,
        "reason": None,
        "identity": identity,
    }


def _read_manifest(path: Path) -> list[CellSpec]:
    return _read_manifest_with_provenance(path)[0]


def _manifest_provenance_audit(
    manifests: list[dict], *, explicit_cell_count: int, control_hz: float
) -> dict:
    incomplete = [record for record in manifests if not record["complete"]]
    mismatches = []
    complete_records = [record for record in manifests if record["complete"]]
    if complete_records:
        reference = complete_records[0]
        reference_identity = reference["identity"]
        for record in complete_records:
            identity = record["identity"]
            for field in SHARED_PROVENANCE_FIELDS:
                if identity[field] != reference_identity[field]:
                    mismatches.append(
                        {
                            "field": field,
                            "reference_manifest": reference["path"],
                            "reference_value": reference_identity[field],
                            "manifest": record["path"],
                            "value": identity[field],
                        }
                    )
            if identity["control_hz"] != float(control_hz):
                mismatches.append(
                    {
                        "field": "cli_control_hz",
                        "manifest": record["path"],
                        "value": identity["control_hz"],
                        "expected": float(control_hz),
                    }
                )

        by_method = {}
        for record in complete_records:
            identity = record["identity"]
            method = identity["method"]
            policy_identity = (
                identity["policy_checkpoint_path"],
                identity["policy_checkpoint_sha256"],
            )
            previous = by_method.setdefault(method, policy_identity)
            if previous != policy_identity:
                mismatches.append(
                    {
                        "field": "policy_checkpoint_identity",
                        "method": method,
                        "reference_value": list(previous),
                        "manifest": record["path"],
                        "value": list(policy_identity),
                    }
                )

    return {
        "complete": bool(manifests)
        and not incomplete
        and explicit_cell_count == 0
        and not mismatches,
        "explicit_cells_without_provenance": explicit_cell_count,
        "manifests": manifests,
        "mismatches": mismatches,
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _surface_timing(spec: CellSpec, *, successes: int, trials: int) -> dict:
    """Aggregate pure evaluation timing across the cell's noise-seed runs."""
    cell_dir = spec.path.parent
    tensorboard_dirs = sorted(cell_dir.glob("noise_*/tensorboard"))
    if cell_dir.name.startswith("noise_"):
        tensorboard_dirs.append(cell_dir / "tensorboard")
    tensorboard_dirs = sorted(
        {path for path in tensorboard_dirs if any(path.glob("events.out.tfevents.*"))}
    )
    empty = {
        "timing_available": False,
        "timing_run_count": 0,
        **dict.fromkeys(SURFACE_TIMING_TAGS),
        "eval_wall_clock_per_trial_s": None,
        "successful_trials_per_hour": None,
    }
    if not tensorboard_dirs:
        return empty

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    values = {key: [] for key in SURFACE_TIMING_TAGS}
    for tensorboard_dir in tensorboard_dirs:
        latest = dict.fromkeys(SURFACE_TIMING_TAGS)
        for event_path in sorted(tensorboard_dir.glob("events.out.tfevents.*")):
            accumulator = EventAccumulator(
                str(event_path), size_guidance={"scalars": 0}
            ).Reload()
            scalar_tags = set(accumulator.Tags().get("scalars", []))
            for key, tag in SURFACE_TIMING_TAGS.items():
                if tag not in scalar_tags:
                    continue
                for event in accumulator.Scalars(tag):
                    previous = latest[key]
                    if previous is None or event.wall_time > previous.wall_time:
                        latest[key] = event
        for key, event in latest.items():
            if event is not None:
                values[key].append(float(event.value))

    timing = {
        "timing_available": len(values["eval_wall_clock_s"]) == len(tensorboard_dirs),
        "timing_run_count": len(tensorboard_dirs),
    }
    for key, items in values.items():
        timing[key] = float(sum(items)) if len(items) == len(tensorboard_dirs) else None
    wall_clock = timing["eval_wall_clock_s"]
    timing["eval_wall_clock_per_trial_s"] = (
        wall_clock / trials if wall_clock is not None and trials > 0 else None
    )
    timing["successful_trials_per_hour"] = (
        3600.0 * successes / wall_clock
        if wall_clock is not None and wall_clock > 0
        else None
    )
    return timing


def _paired_key(row: dict) -> tuple[int, int, int, int, int]:
    fields = ("task_id", "trial_id", "action_execution_horizon", "policy_noise_seed")
    missing = [field for field in fields if field not in row]
    if "requested_semantic_age" not in row and "semantic_age" not in row:
        missing.append("requested_semantic_age")
    if missing:
        raise ValueError(f"Surface row is missing paired fields: {missing}")
    requested_age = row.get("requested_semantic_age", row["semantic_age"])
    return (
        int(row["task_id"]),
        int(row["trial_id"]),
        int(requested_age),
        int(row["action_execution_horizon"]),
        int(row["policy_noise_seed"]),
    )


def _trial_identity(row: dict) -> tuple[int, int, int]:
    fields = ("task_id", "trial_id", "policy_noise_seed")
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Surface row is missing trial identity fields: {missing}")
    return tuple(int(row[field]) for field in fields)


def _validate_cell(spec: CellSpec, rows: list[dict], *, min_trials: int = 1) -> dict:
    if not rows:
        raise ValueError(f"Surface cell has no trials: {spec}")
    if len(rows) < min_trials:
        raise ValueError(
            f"Surface cell {spec} has {len(rows)} trials; required {min_trials}"
        )
    seen = set()
    terminal_boundary_clipped_trials = 0
    clipped_boundaries = 0
    mismatch_boundaries = 0
    semantic_boundaries = 0
    episodes_with_bootstrap_clipping = 0
    weighted_actual_age_sum = 0.0
    actual_age_mins = []
    actual_age_maxes = []
    for row in rows:
        key = _paired_key(row)
        if key in seen:
            raise ValueError(f"Duplicate paired trial key in {spec}: {key}")
        seen.add(key)
        if int(row["action_execution_horizon"]) != spec.action_horizon:
            raise ValueError(
                f"Cell {spec} contains action horizon {row['action_execution_horizon']}"
            )
        requested_age = int(row.get("requested_semantic_age", spec.semantic_age))
        if requested_age != spec.semantic_age:
            raise ValueError(
                f"Cell {spec} contains requested semantic age {requested_age}"
            )
        terminal_boundary_clipped_trials += int(
            bool(row.get("semantic_age_bootstrap_clipped", False))
        )
        boundary_count = int(row.get("semantic_boundary_count", 1))
        clipped_boundary_count = int(
            row.get(
                "semantic_bootstrap_clipped_boundary_count",
                bool(row.get("semantic_age_bootstrap_clipped", False)),
            )
        )
        mismatch_boundary_count = int(
            row.get(
                "semantic_requested_actual_age_mismatch_boundary_count",
                int(row["semantic_age"]) != requested_age,
            )
        )
        if boundary_count <= 0:
            raise ValueError(f"Cell {spec} contains non-positive boundary count")
        if not 0 <= clipped_boundary_count <= boundary_count:
            raise ValueError(f"Cell {spec} contains invalid bootstrap count")
        if not 0 <= mismatch_boundary_count <= boundary_count:
            raise ValueError(f"Cell {spec} contains invalid age mismatch count")
        semantic_boundaries += boundary_count
        clipped_boundaries += clipped_boundary_count
        mismatch_boundaries += mismatch_boundary_count
        episodes_with_bootstrap_clipping += int(clipped_boundary_count > 0)
        weighted_actual_age_sum += (
            float(row.get("semantic_actual_age_mean", row["semantic_age"]))
            * boundary_count
        )
        actual_age_mins.append(
            int(row.get("semantic_actual_age_min", row["semantic_age"]))
        )
        actual_age_maxes.append(
            int(row.get("semantic_actual_age_max", row["semantic_age"]))
        )

    successes = sum(bool(row["success"]) for row in rows)
    low, high = wilson_interval(successes, len(rows))
    return {
        "method": spec.method,
        "semantic_age_frames": spec.semantic_age,
        "action_execution_horizon": spec.action_horizon,
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95_low": low,
        "wilson_95_high": high,
        "bootstrap_clipped_trials": episodes_with_bootstrap_clipping,
        "terminal_boundary_bootstrap_clipped_trials": (
            terminal_boundary_clipped_trials
        ),
        "semantic_boundaries": semantic_boundaries,
        "bootstrap_clipped_boundaries": clipped_boundaries,
        "requested_actual_age_mismatch_boundaries": mismatch_boundaries,
        "episodes_with_bootstrap_clipping": episodes_with_bootstrap_clipping,
        "semantic_actual_age_boundary_mean": (
            weighted_actual_age_sum / semantic_boundaries
        ),
        "actual_age_min": min(actual_age_mins),
        "actual_age_max": max(actual_age_maxes),
        "actual_age_mean": weighted_actual_age_sum / semantic_boundaries,
    }


def _deadline(
    cells: list[dict],
    *,
    baseline_rate: float,
    level: float,
    axis: str,
    control_hz: float,
    conservative: bool,
    require_contiguous: bool = False,
) -> float | None:
    if axis == "semantic":
        eligible = [cell for cell in cells if cell["action_execution_horizon"] == 1]
        value_key = "semantic_age_frames"
    elif axis == "action":
        eligible = [cell for cell in cells if cell["semantic_age_frames"] == 0]
        value_key = "action_execution_horizon"
    else:
        raise ValueError(f"Unknown deadline axis: {axis}")
    threshold = level * baseline_rate
    metric = "wilson_95_low" if conservative else "success_rate"
    last_passing = None
    for cell in sorted(eligible, key=lambda item: item[value_key]):
        if cell[metric] < threshold:
            if require_contiguous:
                break
            continue
        last_passing = cell[value_key]
    return last_passing / control_hz if last_passing is not None else None


def _matrix_audit(
    summaries: list[dict],
    expected_ages: tuple[int, ...] | list[int],
    expected_horizons: tuple[int, ...] | list[int],
    expected_methods: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, list[list[int]] | bool]]:
    expected = {
        (int(age), int(horizon))
        for age in expected_ages
        for horizon in expected_horizons
    }
    observed_methods = {cell["method"] for cell in summaries}
    required_methods = set(expected_methods or observed_methods)
    audit = {}
    for method in sorted(observed_methods | required_methods):
        observed = {
            (cell["semantic_age_frames"], cell["action_execution_horizon"])
            for cell in summaries
            if cell["method"] == method
        }
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        audit[method] = {
            "complete": (
                method in required_methods
                and method in observed_methods
                and not missing
                and not unexpected
            ),
            "missing_method": method in required_methods
            and method not in observed_methods,
            "unexpected_method": method in observed_methods
            and method not in required_methods,
            "missing_cells": [list(cell) for cell in missing],
            "unexpected_cells": [list(cell) for cell in unexpected],
        }
    return audit


def _formal_trial_audit(summaries: list[dict]) -> dict:
    """Enforce the pre-registered 400-trial floor independently of CLI knobs."""
    underfilled = [
        {
            "method": cell["method"],
            "semantic_age_frames": cell["semantic_age_frames"],
            "action_execution_horizon": cell["action_execution_horizon"],
            "trials": cell["trials"],
        }
        for cell in summaries
        if int(cell["trials"]) < FORMAL_MIN_TRIALS_PER_CELL
    ]
    return {
        "complete": not underfilled,
        "required_trials_per_cell": FORMAL_MIN_TRIALS_PER_CELL,
        "minimum_observed_trials": min(
            (int(cell["trials"]) for cell in summaries), default=0
        ),
        "underfilled_cells": underfilled,
    }


def _pareto_summary(summaries: list[dict]) -> dict:
    """Report success-vs-evaluation-wall-clock non-dominated action horizons."""
    output = {}
    groups = sorted(
        {(cell["method"], cell["semantic_age_frames"]) for cell in summaries}
    )
    for method, semantic_age in groups:
        cells = sorted(
            (
                cell
                for cell in summaries
                if cell["method"] == method
                and cell["semantic_age_frames"] == semantic_age
            ),
            key=lambda cell: cell["action_execution_horizon"],
        )
        timed = [
            cell
            for cell in cells
            if cell.get("eval_wall_clock_per_trial_s") is not None
        ]
        key = f"{method}:age_{semantic_age}"
        if len(timed) != len(cells):
            output[key] = {
                "available": False,
                "reason": "one or more cells have no TensorBoard evaluation timing",
                "points": [],
                "frontier_action_horizons": [],
            }
            continue

        frontier = []
        for cell in timed:
            dominated = any(
                other["eval_wall_clock_per_trial_s"]
                <= cell["eval_wall_clock_per_trial_s"]
                and other["success_rate"] >= cell["success_rate"]
                and (
                    other["eval_wall_clock_per_trial_s"]
                    < cell["eval_wall_clock_per_trial_s"]
                    or other["success_rate"] > cell["success_rate"]
                )
                for other in timed
                if other is not cell
            )
            if not dominated:
                frontier.append(cell["action_execution_horizon"])
        output[key] = {
            "available": True,
            "reason": None,
            "points": [
                {
                    "action_execution_horizon": cell["action_execution_horizon"],
                    "success_rate": cell["success_rate"],
                    "eval_wall_clock_s": cell["eval_wall_clock_s"],
                    "eval_wall_clock_per_trial_s": cell["eval_wall_clock_per_trial_s"],
                    "successful_trials_per_hour": cell["successful_trials_per_hour"],
                }
                for cell in timed
            ],
            "frontier_action_horizons": frontier,
        }
    return output


def _trial_identity_audit(
    raw_cells: dict[tuple[str, int, int], list[dict]],
) -> dict:
    """Require every cell and method to use the same paired trial identities."""
    identities = {
        key: {_trial_identity(row) for row in rows} for key, rows in raw_cells.items()
    }
    method_audits = {}
    method_references = {}
    for method in sorted({key[0] for key in raw_cells}):
        method_keys = sorted(key for key in raw_cells if key[0] == method)
        reference_key = method_keys[0]
        reference = identities[reference_key]
        method_references[method] = reference
        cells = {}
        for key in method_keys:
            current = identities[key]
            cells[f"age_{key[1]}_ka_{key[2]}"] = {
                "matches_reference": current == reference,
                "trial_identities": len(current),
                "missing_identity_count": len(reference - current),
                "unexpected_identity_count": len(current - reference),
            }
        method_audits[method] = {
            "reference_cell": f"age_{reference_key[1]}_ka_{reference_key[2]}",
            "trial_identities": len(reference),
            "complete": all(cell["matches_reference"] for cell in cells.values()),
            "cells": cells,
        }

    methods = sorted(method_references)
    cross_method = {
        "checked": len(methods) > 1,
        "reference_method": methods[0],
        "complete": True,
        "methods": {},
    }
    if methods:
        reference = method_references[methods[0]]
        for method in methods:
            current = method_references[method]
            cross_method["methods"][method] = {
                "matches_reference": current == reference,
                "trial_identities": len(current),
                "missing_identity_count": len(reference - current),
                "unexpected_identity_count": len(current - reference),
            }
        cross_method["complete"] = all(
            result["matches_reference"] for result in cross_method["methods"].values()
        )

    return {
        "complete": all(result["complete"] for result in method_audits.values())
        and bool(cross_method["complete"]),
        "within_method": method_audits,
        "cross_method": cross_method,
    }


def _deadline_summary(
    cells: list[dict],
    control_hz: float,
    expected_semantic_ages: tuple[int, ...] = DEFAULT_SEMANTIC_AGES,
    expected_action_horizons: tuple[int, ...] = DEFAULT_ACTION_HORIZONS,
) -> dict:
    expected_semantic_ages = tuple(expected_semantic_ages)
    expected_action_horizons = tuple(expected_action_horizons)
    observed_semantic_ages = sorted(
        {
            cell["semantic_age_frames"]
            for cell in cells
            if cell["action_execution_horizon"] == 1
        }
    )
    observed_action_horizons = sorted(
        {
            cell["action_execution_horizon"]
            for cell in cells
            if cell["semantic_age_frames"] == 0
        }
    )
    missing_semantic_ages = sorted(
        set(expected_semantic_ages) - set(observed_semantic_ages)
    )
    missing_action_horizons = sorted(
        set(expected_action_horizons) - set(observed_action_horizons)
    )
    axis_audit = {
        "complete": not missing_semantic_ages and not missing_action_horizons,
        "semantic_axis": {
            "complete": not missing_semantic_ages,
            "expected_ages": list(expected_semantic_ages),
            "observed_ages": observed_semantic_ages,
            "missing_ages": missing_semantic_ages,
        },
        "action_axis": {
            "complete": not missing_action_horizons,
            "expected_horizons": list(expected_action_horizons),
            "observed_horizons": observed_action_horizons,
            "missing_horizons": missing_action_horizons,
        },
    }
    baseline = [
        cell
        for cell in cells
        if cell["semantic_age_frames"] == 0 and cell["action_execution_horizon"] == 1
    ]
    if len(baseline) > 1:
        raise ValueError("Each method requires exactly one baseline cell P(0, 1)")
    if not baseline:
        return {
            "available": False,
            "reason": "deadline requires the P(0,1) baseline cell",
            "baseline_success_rate": None,
            "levels": {},
            "axis_audit": axis_audit,
        }
    if not axis_audit["complete"]:
        return {
            "available": False,
            "reason": "deadline requires complete registered semantic and action axes",
            "baseline_success_rate": baseline[0]["success_rate"],
            "levels": {},
            "axis_audit": axis_audit,
        }
    baseline_rate = baseline[0]["success_rate"]
    registered_cells = [
        cell
        for cell in cells
        if (
            cell["action_execution_horizon"] == 1
            and cell["semantic_age_frames"] in expected_semantic_ages
        )
        or (
            cell["semantic_age_frames"] == 0
            and cell["action_execution_horizon"] in expected_action_horizons
        )
    ]
    levels = {}
    for level in (0.95, 0.90, 0.50):
        label = str(int(level * 100))
        point_s = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="semantic",
            control_hz=control_hz,
            conservative=False,
        )
        point_a = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="action",
            control_hz=control_hz,
            conservative=False,
        )
        conservative_s = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="semantic",
            control_hz=control_hz,
            conservative=True,
        )
        conservative_a = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="action",
            control_hz=control_hz,
            conservative=True,
        )
        contiguous_point_s = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="semantic",
            control_hz=control_hz,
            conservative=False,
            require_contiguous=True,
        )
        contiguous_point_a = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="action",
            control_hz=control_hz,
            conservative=False,
            require_contiguous=True,
        )
        contiguous_conservative_s = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="semantic",
            control_hz=control_hz,
            conservative=True,
            require_contiguous=True,
        )
        contiguous_conservative_a = _deadline(
            registered_cells,
            baseline_rate=baseline_rate,
            level=level,
            axis="action",
            control_hz=control_hz,
            conservative=True,
            require_contiguous=True,
        )
        levels[label] = {
            "point_estimate": _deadline_metrics(point_s, point_a),
            "wilson_lower_bound": _deadline_metrics(conservative_s, conservative_a),
            "contiguous_sensitivity": {
                "point_estimate": _deadline_metrics(
                    contiguous_point_s, contiguous_point_a
                ),
                "wilson_lower_bound": _deadline_metrics(
                    contiguous_conservative_s, contiguous_conservative_a
                ),
            },
        }
    return {
        "available": True,
        "reason": None,
        "baseline_success_rate": baseline_rate,
        "levels": levels,
        "axis_audit": axis_audit,
    }


def _deadline_metrics(
    semantic_deadline_s: float | None, action_deadline_s: float | None
) -> dict[str, float | None]:
    return {
        "semantic_deadline_s": semantic_deadline_s,
        "action_deadline_s": action_deadline_s,
        "semantic_min_update_hz": (
            1.0 / semantic_deadline_s
            if semantic_deadline_s not in (None, 0.0)
            else None
        ),
        "action_min_replan_hz": (
            1.0 / action_deadline_s if action_deadline_s not in (None, 0.0) else None
        ),
        "asymmetry_ratio": (
            semantic_deadline_s / action_deadline_s
            if semantic_deadline_s is not None and action_deadline_s not in (None, 0.0)
            else None
        ),
    }


def _paired_transition(reference: list[dict], candidate: list[dict]) -> dict:
    reference_index = {_paired_key(row): row for row in reference}
    candidate_index = {_paired_key(row): row for row in candidate}
    if set(reference_index) != set(candidate_index):
        raise ValueError("Paired SFT/RL trial sets differ for a surface cell")
    failure_to_success = 0
    success_to_failure = 0
    actual_age_contract_violation_count = 0
    trajectory_dependent_actual_age_mean_difference_count = 0
    bootstrap_clip_mismatch_count = 0
    for key in reference_index:
        before = bool(reference_index[key]["success"])
        after = bool(candidate_index[key]["success"])
        failure_to_success += int(not before and after)
        success_to_failure += int(before and not after)
        reference_row = reference_index[key]
        candidate_row = candidate_index[key]
        reference_mean = float(
            reference_row.get("semantic_actual_age_mean", reference_row["semantic_age"])
        )
        candidate_mean = float(
            candidate_row.get("semantic_actual_age_mean", candidate_row["semantic_age"])
        )
        trajectory_dependent_actual_age_mean_difference_count += int(
            reference_mean != candidate_mean
        )

        def nonbootstrap_age_violations(row: dict) -> int:
            bootstrap_count = int(
                row.get(
                    "semantic_bootstrap_clipped_boundary_count",
                    bool(row.get("semantic_age_bootstrap_clipped", False)),
                )
            )
            if "semantic_requested_actual_age_mismatch_boundary_count" in row:
                mismatch_count = int(
                    row["semantic_requested_actual_age_mismatch_boundary_count"]
                )
            else:
                requested = float(
                    row.get("requested_semantic_age", row["semantic_age"])
                )
                mismatch_count = (
                    int(reference_mean != requested)
                    if row is reference_row
                    else int(candidate_mean != requested)
                )
            return max(mismatch_count - bootstrap_count, 0)

        actual_age_contract_violation_count += int(
            nonbootstrap_age_violations(reference_row) > 0
            or nonbootstrap_age_violations(candidate_row) > 0
        )
        reference_bootstrap = bool(
            reference_index[key].get(
                "semantic_bootstrap_clipped_boundary_count",
                reference_index[key].get("semantic_age_bootstrap_clipped", False),
            )
        )
        candidate_bootstrap = bool(
            candidate_index[key].get(
                "semantic_bootstrap_clipped_boundary_count",
                candidate_index[key].get("semantic_age_bootstrap_clipped", False),
            )
        )
        bootstrap_clip_mismatch_count += int(reference_bootstrap != candidate_bootstrap)
    return {
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
        "actual_age_contract_violation_count": actual_age_contract_violation_count,
        # Backward-compatible alias: causal bootstrap clipping is excluded.
        "actual_age_mismatch_count": actual_age_contract_violation_count,
        "trajectory_dependent_actual_age_mean_difference_count": (
            trajectory_dependent_actual_age_mean_difference_count
        ),
        "bootstrap_clip_mismatch_count": bootstrap_clip_mismatch_count,
        "exact_mcnemar_p": exact_mcnemar_p(failure_to_success, success_to_failure),
    }


def _write_heatmap(method: str, cells: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ages = sorted({cell["semantic_age_frames"] for cell in cells})
    horizons = sorted({cell["action_execution_horizon"] for cell in cells})
    indexed = {
        (cell["semantic_age_frames"], cell["action_execution_horizon"]): cell
        for cell in cells
    }
    matrix = np.full((len(ages), len(horizons)), np.nan, dtype=np.float64)
    for row, age in enumerate(ages):
        for column, horizon in enumerate(horizons):
            cell = indexed.get((age, horizon))
            if cell is not None:
                matrix[row, column] = 100.0 * cell["success_rate"]

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, vmin=0.0, vmax=100.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(horizons)), [str(value) for value in horizons])
    axis.set_yticks(range(len(ages)), [str(value) for value in ages])
    axis.set_xlabel("Action execution horizon K_a (frames)")
    axis.set_ylabel("Requested semantic age d_s (frames)")
    axis.set_title(f"{method}: fixed-trial success (%)")
    for row in range(len(ages)):
        for column in range(len(horizons)):
            if np.isfinite(matrix[row, column]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < 60 else "black",
                )
    figure.colorbar(image, ax=axis, label="Success (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_wall_clock_pareto(cells: list[dict], output_path: Path) -> bool:
    import matplotlib.pyplot as plt

    timed = [
        cell for cell in cells if cell.get("eval_wall_clock_per_trial_s") is not None
    ]
    if not timed:
        return False

    figure, axis = plt.subplots(figsize=(9, 6))
    groups = sorted({(cell["method"], cell["semantic_age_frames"]) for cell in timed})
    for method, semantic_age in groups:
        group = sorted(
            (
                cell
                for cell in timed
                if cell["method"] == method
                and cell["semantic_age_frames"] == semantic_age
            ),
            key=lambda cell: cell["eval_wall_clock_per_trial_s"],
        )
        axis.plot(
            [cell["eval_wall_clock_per_trial_s"] for cell in group],
            [100.0 * cell["success_rate"] for cell in group],
            marker="o",
            label=f"{method}, age={semantic_age}",
        )
        for cell in group:
            axis.annotate(
                f"K={cell['action_execution_horizon']}",
                (
                    cell["eval_wall_clock_per_trial_s"],
                    100.0 * cell["success_rate"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Evaluation wall-clock per fixed trial (s)")
    axis.set_ylabel("Fixed-trial success (%)")
    axis.set_title("FDVLA success vs evaluation wall-clock")
    axis.grid(alpha=0.3)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="METHOD:AGE:K=JSONL",
        help="Fixed-trial JSONL for one response-surface cell.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        type=Path,
        help="TSV manifest emitted by eval_fdvla_realtime_surface.sh.",
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--min-trials-per-cell", type=int, default=400)
    parser.add_argument(
        "--expected-semantic-ages",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEMANTIC_AGES),
    )
    parser.add_argument(
        "--expected-action-horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_ACTION_HORIZONS),
    )
    parser.add_argument(
        "--expected-methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Methods required for a formal before/after response surface.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit a development subset; output remains marked incomplete.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    if args.control_hz <= 0:
        raise ValueError("control-hz must be positive")
    if args.min_trials_per_cell <= 0:
        raise ValueError("min-trials-per-cell must be positive")

    specs = [_parse_cell(value) for value in args.cell]
    manifest_provenance = []
    for manifest in args.manifest:
        manifest_specs, provenance = _read_manifest_with_provenance(manifest)
        specs.extend(manifest_specs)
        manifest_provenance.append(provenance)

    provenance_audit = _manifest_provenance_audit(
        manifest_provenance,
        explicit_cell_count=len(args.cell),
        control_hz=args.control_hz,
    )
    if provenance_audit["mismatches"]:
        raise ValueError(
            "Realtime surface manifests have incompatible policy provenance: "
            f"{provenance_audit['mismatches']}"
        )
    if not provenance_audit["complete"] and not args.allow_incomplete:
        raise ValueError(
            "Formal realtime surface requires schema-v3 manifest provenance; "
            f"audit={provenance_audit}"
        )

    raw_cells = {}
    summaries = []
    for spec in specs:
        key = (spec.method, spec.semantic_age, spec.action_horizon)
        if key in raw_cells:
            raise ValueError(f"Duplicate surface cell: {key}")
        rows = _read_jsonl(spec.path)
        raw_cells[key] = rows
        summary = _validate_cell(spec, rows, min_trials=args.min_trials_per_cell)
        summary.update(
            _surface_timing(
                spec,
                successes=summary["successes"],
                trials=summary["trials"],
            )
        )
        summaries.append(summary)
    if not summaries:
        raise ValueError("At least one --cell or --manifest is required")

    matrix_audit = _matrix_audit(
        summaries,
        args.expected_semantic_ages,
        args.expected_action_horizons,
        args.expected_methods,
    )
    incomplete = {
        method: result
        for method, result in matrix_audit.items()
        if not result["complete"]
    }
    if incomplete and not args.allow_incomplete:
        raise ValueError(
            "Formal realtime surface requires the complete pre-registered matrix; "
            f"incomplete={incomplete}"
        )

    formal_trial_audit = _formal_trial_audit(summaries)
    if not formal_trial_audit["complete"] and not args.allow_incomplete:
        raise ValueError(
            "Formal realtime surface requires at least "
            f"{FORMAL_MIN_TRIALS_PER_CELL} trials per cell; "
            f"audit={formal_trial_audit}"
        )

    trial_identity_audit = _trial_identity_audit(raw_cells)
    if not trial_identity_audit["complete"]:
        raise ValueError(
            "Realtime surface cells do not share identical paired "
            f"(task, trial, noise) identities: {trial_identity_audit}"
        )

    methods = sorted({cell["method"] for cell in summaries})
    deadlines = {
        method: _deadline_summary(
            [cell for cell in summaries if cell["method"] == method],
            args.control_hz,
            tuple(args.expected_semantic_ages),
            tuple(args.expected_action_horizons),
        )
        for method in methods
    }
    pareto = _pareto_summary(summaries)
    paired = {}
    if "D-SFT" in methods and "D-PPO" in methods:
        conditions = sorted(
            {
                (cell["semantic_age_frames"], cell["action_execution_horizon"])
                for cell in summaries
                if cell["method"] == "D-SFT"
            }
        )
        for age, horizon in conditions:
            reference = raw_cells[("D-SFT", age, horizon)]
            candidate = raw_cells.get(("D-PPO", age, horizon))
            if candidate is None:
                raise ValueError(f"D-PPO is missing surface cell {(age, horizon)}")
            paired[f"age_{age}_ka_{horizon}"] = _paired_transition(reference, candidate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fdvla_realtime_surface.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(
            sorted(
                summaries,
                key=lambda row: (
                    row["method"],
                    row["semantic_age_frames"],
                    row["action_execution_horizon"],
                ),
            )
        )

    output = {
        "control_hz": args.control_hz,
        "expected_semantic_ages": args.expected_semantic_ages,
        "expected_action_horizons": args.expected_action_horizons,
        "expected_methods": args.expected_methods,
        "min_trials_per_cell": args.min_trials_per_cell,
        "formal_trial_audit": formal_trial_audit,
        "development_mode": bool(args.allow_incomplete),
        "matrix_audit": matrix_audit,
        "trial_identity_audit": trial_identity_audit,
        "manifest_provenance_audit": provenance_audit,
        "formal_matrix_complete": (
            not args.allow_incomplete
            and not incomplete
            and formal_trial_audit["complete"]
            and trial_identity_audit["complete"]
            and provenance_audit["complete"]
        ),
        "cells": summaries,
        "deadlines": deadlines,
        "success_vs_wall_clock_pareto": pareto,
        "paired_d_ppo_vs_d_sft": paired,
        "deadline_rule": (
            "tau is the maximum tested age/period whose success metric reaches "
            "the requested fraction of P(0,1), matching the pre-registered max "
            "definition; conservative values use the cell Wilson 95% lower bound. "
            "A last-contiguous-passing sensitivity analysis is also reported."
        ),
    }
    (args.output_dir / "fdvla_realtime_surface.json").write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.skip_plots:
        for method in methods:
            _write_heatmap(
                method,
                [cell for cell in summaries if cell["method"] == method],
                args.output_dir / f"fdvla_realtime_surface_{method}.png",
            )
        _write_wall_clock_pareto(
            summaries,
            args.output_dir / "fdvla_realtime_success_vs_wall_clock.png",
        )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
