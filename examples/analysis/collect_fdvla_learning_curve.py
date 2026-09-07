#!/usr/bin/env python3
"""Build auditable FDVLA learning-curve rows from run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from examples.analysis.fdvla_statistics import (
    index_unique_trial_identities,
    index_unique_trials,
)

_EVAL_FILE_PATTERN = re.compile(r"eval_trials_step_(?P<step>\d+)\.jsonl$")
PAIRED_CONDITION_FIELDS = (
    "task_id",
    "trial_id",
    "policy_noise_seed",
    "requested_semantic_age",
    "action_execution_horizon",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _events_by_step(events: Iterable[Any]) -> dict[int, Any]:
    """Keep the latest event for each step, including resumed event files."""
    selected: dict[int, Any] = {}
    for event in events:
        step = int(event.step)
        if step not in selected or float(event.wall_time) > float(
            selected[step].wall_time
        ):
            selected[step] = event
    return selected


def _load_training_environment_frames_per_update(run_dir: Path) -> int | None:
    """Return the exact simulator-frame budget encoded by the resolved config."""
    config_path = run_dir / "fdvla_metadata" / "resolved_config.yaml"
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        total_num_envs = int(config["env"]["train"]["total_num_envs"])
        frames_per_epoch = int(
            config["env"]["train"]["max_steps_per_rollout_epoch"]
        )
        rollout_epochs = int(config["algorithm"]["rollout_epoch"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot derive training simulator-frame budget from {config_path}"
        ) from error
    if min(total_num_envs, frames_per_epoch, rollout_epochs) <= 0:
        raise ValueError(f"Training simulator-frame budget must be positive: {config_path}")
    return total_num_envs * frames_per_epoch * rollout_epochs


def _fixed_budget_profile_accounting(
    counter_events: Iterable[Any], eval_events: Iterable[Any]
) -> dict[str, Any]:
    """Separate fixed-budget training counters from eval-contaminated windows.

    ``EmbodiedRunner`` currently consumes rollout execution metrics after its
    optional evaluation. Consequently, the first update (when pre-training
    evaluation is enabled) and every update enclosing a fixed evaluation contain
    both training and evaluation counts. Only correct this when at least two
    non-evaluation updates prove one constant per-update training budget.
    """
    counter_by_step = {
        step: float(event.value)
        for step, event in _events_by_step(counter_events).items()
    }
    evaluation_global_steps = {
        int(event.step) for event in _events_by_step(eval_events).values()
    }
    contaminated_event_steps = {
        step - 1 for step in evaluation_global_steps if step > 0
    }
    if 0 in evaluation_global_steps:
        contaminated_event_steps.add(0)

    clean_values = [
        value
        for step, value in counter_by_step.items()
        if step not in contaminated_event_steps
    ]
    fixed_budget = clean_values[0] if len(clean_values) >= 2 else None
    fixed_budget_verified = fixed_budget is not None and all(
        math.isclose(value, fixed_budget, rel_tol=1e-9, abs_tol=1e-6)
        for value in clean_values[1:]
    )
    contaminated_values_valid = fixed_budget_verified and all(
        value + 1e-6 >= fixed_budget
        for step, value in counter_by_step.items()
        if step in contaminated_event_steps
    )
    correction_applied = bool(fixed_budget_verified and contaminated_values_valid)
    training_by_step = dict(counter_by_step)
    if correction_applied:
        for step in contaminated_event_steps & counter_by_step.keys():
            training_by_step[step] = float(fixed_budget)

    raw_total = sum(counter_by_step.values())
    training_total = sum(training_by_step.values())
    return {
        "counter_by_step": counter_by_step,
        "training_counter_by_step": training_by_step,
        "profile_counter_including_eval": raw_total,
        "training_counter": training_total,
        "eval_counter": raw_total - training_total,
        "training_counter_per_update": (
            float(fixed_budget) if correction_applied else None
        ),
        "correction_applied": correction_applied,
        "contaminated_event_steps": sorted(
            contaminated_event_steps & counter_by_step.keys()
        ),
    }


def _condition_identity_fingerprint(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint the exact fixed-evaluation condition set, excluding outcomes."""
    identities = []
    missing_fields: set[str] = set()
    for row in rows:
        required = (
            "task_id",
            "trial_id",
            "policy_noise_seed",
            "action_execution_horizon",
        )
        row_missing_fields = {field for field in required if field not in row}
        if "requested_semantic_age" not in row and "semantic_age" not in row:
            row_missing_fields.add("requested_semantic_age")
        missing_fields.update(row_missing_fields)
        if row_missing_fields:
            continue
        requested_age = (
            row["requested_semantic_age"]
            if "requested_semantic_age" in row
            else row["semantic_age"]
        )
        identities.append(
            (
                int(row["task_id"]),
                int(row["trial_id"]),
                int(row["policy_noise_seed"]),
                int(requested_age),
                int(row["action_execution_horizon"]),
            )
        )
    if missing_fields:
        return {
            "complete": False,
            "count": 0,
            "sha256": None,
            "missing_fields": sorted(missing_fields),
        }
    ordered = sorted(identities)
    payload = json.dumps(ordered, separators=(",", ":")).encode("utf-8")
    return {
        "complete": True,
        "count": len(ordered),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "missing_fields": [],
    }


def _pairing_audit(run_dir: Path) -> dict[str, Any]:
    paths = sorted(
        run_dir.glob("eval_trials_step_*.jsonl"),
        key=lambda path: int(_EVAL_FILE_PATTERN.search(path.name).group("step")),
    )
    if not paths:
        raise ValueError(f"Run {run_dir} has no fixed-evaluation JSONL files")
    baseline_path = paths[0]
    baseline_rows = _read_jsonl(baseline_path)
    baseline = index_unique_trial_identities(baseline_rows)
    baseline_condition = _condition_identity_fingerprint(baseline_rows)
    comparisons = {}
    formal = baseline_condition["complete"]
    operational = True
    for path in paths[1:]:
        candidate_rows = _read_jsonl(path)
        candidate = index_unique_trial_identities(candidate_rows)
        candidate_condition = _condition_identity_fingerprint(candidate_rows)
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        shared = sorted(set(baseline) & set(candidate))
        requested_age_mismatches = sum(
            baseline[key].get(
                "requested_semantic_age", baseline[key].get("semantic_age")
            )
            != candidate[key].get(
                "requested_semantic_age", candidate[key].get("semantic_age")
            )
            for key in shared
        )
        actual_age_mismatches = sum(
            baseline[key].get("semantic_age") != candidate[key].get("semantic_age")
            for key in shared
        )
        horizon_mismatches = sum(
            baseline[key].get("action_execution_horizon")
            != candidate[key].get("action_execution_horizon")
            for key in shared
        )
        comparison_formal = not (
            missing_candidate
            or missing_baseline
            or requested_age_mismatches
            or horizon_mismatches
        ) and bool(candidate_condition["complete"])
        comparison_operational = not (
            missing_candidate or missing_baseline or horizon_mismatches
        )
        formal &= comparison_formal
        operational &= comparison_operational
        comparisons[path.name] = {
            "formal_pairing": comparison_formal,
            "operational_trial_pairing": comparison_operational,
            "missing_candidate_trial_identities": len(missing_candidate),
            "missing_baseline_trial_identities": len(missing_baseline),
            "requested_semantic_age_mismatch_count": requested_age_mismatches,
            "actual_semantic_age_mismatch_count": actual_age_mismatches,
            "action_execution_horizon_mismatch_count": horizon_mismatches,
            "paired_condition_identity_complete": candidate_condition["complete"],
            "paired_condition_identity_count": candidate_condition["count"],
            "paired_condition_identity_sha256": candidate_condition["sha256"],
            "paired_condition_missing_fields": candidate_condition["missing_fields"],
        }
    return {
        "baseline": baseline_path.name,
        "formal_pairing": formal,
        "operational_trial_pairing": operational,
        "operational_paired_fields": [
            "task_id",
            "trial_id",
            "policy_noise_seed",
            "action_execution_horizon",
        ],
        "paired_condition_fields": list(PAIRED_CONDITION_FIELDS),
        "paired_condition_identity_complete": baseline_condition["complete"],
        "paired_condition_identity_count": baseline_condition["count"],
        "paired_condition_identity_sha256": baseline_condition["sha256"],
        "paired_condition_missing_fields": baseline_condition["missing_fields"],
        "comparisons": comparisons,
    }


def _training_wallclock_accounting(
    step_time_events: Iterable[Any],
    evaluation_time_events: Iterable[Any],
    eval_events: Iterable[Any],
) -> dict[str, Any]:
    """Separate PPO-update time from periodic fixed-evaluation time.

    RLinf records ``time/step`` at zero-based update event step ``u - 1``.
    A periodic evaluation completed at global step ``u`` is included in that
    step timer and its duration is recorded at event step ``u - 1``. The
    pre-training evaluation at global step zero is not part of an update timer,
    despite sharing event step zero with the first update, so it must not be
    subtracted.
    """
    step_time_by_step = {
        step: float(event.value)
        for step, event in _events_by_step(step_time_events).items()
    }
    evaluation_time_by_step = {
        step: float(event.value)
        for step, event in _events_by_step(evaluation_time_events).items()
    }
    periodic_eval_update_steps = {
        int(event.step) - 1
        for event in _events_by_step(eval_events).values()
        if int(event.step) > 0
    }
    training_time_by_step = dict(step_time_by_step)
    subtracted_evaluation_time_by_step: dict[int, float] = {}
    for step in periodic_eval_update_steps & step_time_by_step.keys():
        if step not in evaluation_time_by_step:
            continue
        evaluation_time = evaluation_time_by_step[step]
        training_time_by_step[step] = max(
            0.0, step_time_by_step[step] - evaluation_time
        )
        subtracted_evaluation_time_by_step[step] = evaluation_time
    return {
        "step_time_by_step": step_time_by_step,
        "training_time_by_step": training_time_by_step,
        "subtracted_evaluation_time_by_step": subtracted_evaluation_time_by_step,
        "periodic_eval_update_steps": sorted(periodic_eval_update_steps),
    }


def _build_curve_rows(
    *,
    run_dir: Path,
    method: str,
    seed: int,
    eval_events: Iterable[Any],
    frame_events: Iterable[Any],
    step_time_events: Iterable[Any] = (),
    evaluation_time_events: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    eval_by_step = _events_by_step(eval_events)
    environment_frames_per_update = _load_training_environment_frames_per_update(
        run_dir
    )
    frame_accounting = _fixed_budget_profile_accounting(
        frame_events, eval_by_step.values()
    )
    frame_by_step = frame_accounting["counter_by_step"]
    training_frame_by_step = frame_accounting["training_counter_by_step"]
    wallclock_accounting = _training_wallclock_accounting(
        step_time_events, evaluation_time_events, eval_by_step.values()
    )
    if 0 not in eval_by_step:
        raise ValueError(f"Run {run_dir} has no step-0 fixed evaluation event")

    baseline_wall_time = float(eval_by_step[0].wall_time)
    curve_rows = []
    eval_paths = sorted(
        run_dir.glob("eval_trials_step_*.jsonl"),
        key=lambda path: int(_EVAL_FILE_PATTERN.search(path.name).group("step")),
    )
    if not eval_paths:
        raise ValueError(f"Run {run_dir} has no eval_trials_step_*.jsonl files")

    for path in eval_paths:
        match = _EVAL_FILE_PATTERN.search(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        eval_event_step = step
        alignment = "completed_update_step"
        if (
            eval_event_step not in eval_by_step
            and step > 0
            and step - 1 in eval_by_step
        ):
            eval_event_step = step - 1
            alignment = "legacy_step_minus_one"
        if eval_event_step not in eval_by_step:
            raise ValueError(
                f"Run {run_dir} has JSONL for step {step} but no matching eval event"
            )
        rows = _read_jsonl(path)
        if not rows:
            raise ValueError(f"Fixed evaluation is empty: {path}")
        index_unique_trials(rows)
        successes = sum(bool(row["success"]) for row in rows)
        cumulative_profile_frames = sum(
            value for event_step, value in frame_by_step.items() if event_step < step
        )
        cumulative_training_frames = sum(
            value
            for event_step, value in training_frame_by_step.items()
            if event_step < step
        )
        cumulative_update_wallclock = sum(
            value
            for event_step, value in wallclock_accounting["step_time_by_step"].items()
            if event_step < step
        )
        cumulative_training_wallclock = sum(
            value
            for event_step, value in wallclock_accounting[
                "training_time_by_step"
            ].items()
            if event_step < step
        )
        curve_rows.append(
            {
                "method": method,
                "seed": seed,
                "global_step": step,
                "environment_frames": (
                    step * environment_frames_per_update
                    if environment_frames_per_update is not None
                    else int(round(cumulative_training_frames))
                ),
                "environment_frame_source": (
                    "resolved_config_fixed_budget"
                    if environment_frames_per_update is not None
                    else "rollout_profile_fallback"
                ),
                "training_environment_frames_per_update": (
                    environment_frames_per_update
                ),
                "rollout_profile_control_frame_slots_including_eval": int(
                    round(cumulative_profile_frames)
                ),
                "rollout_profile_eval_control_frame_slots": int(
                    round(cumulative_profile_frames - cumulative_training_frames)
                ),
                "rollout_profile_frame_correction_applied": frame_accounting[
                    "correction_applied"
                ],
                "rollout_profile_training_control_frame_slots_per_update": (
                    frame_accounting["training_counter_per_update"]
                ),
                "wallclock_s": max(
                    0.0,
                    float(eval_by_step[eval_event_step].wall_time) - baseline_wall_time,
                ),
                "update_wallclock_s": cumulative_update_wallclock,
                "training_wallclock_s": cumulative_training_wallclock,
                "periodic_evaluation_wallclock_s": (
                    cumulative_update_wallclock - cumulative_training_wallclock
                ),
                "successes": successes,
                "trials": len(rows),
                "success_rate": successes / len(rows),
                "run_dir": str(run_dir.resolve()),
                "eval_jsonl": str(path.resolve()),
                "eval_event_step": eval_event_step,
                "eval_step_alignment": alignment,
            }
        )
    return curve_rows


def _load_tensorboard_events(run_dir: Path, tag: str) -> list[Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    paths = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    if not paths:
        raise ValueError(f"Run {run_dir} has no TensorBoard event files")
    events = []
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag in accumulator.Tags()["scalars"]:
            events.extend(accumulator.Scalars(tag))
    if not events:
        raise ValueError(f"Run {run_dir} has no TensorBoard scalar {tag!r}")
    return events


def _parse_run(value: str) -> tuple[str, int, Path]:
    try:
        label, path = value.split("=", 1)
        method, seed_text = label.rsplit(":", 1)
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid --run {value!r}; expected METHOD:SEED=RUN_DIR"
        ) from error
    if not method:
        raise argparse.ArgumentTypeError("METHOD must not be empty")
    return method, seed, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="METHOD:SEED=RUN_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_rows = []
    pairing_audits = {}
    for method, seed, run_dir in args.run:
        output_rows.extend(
            _build_curve_rows(
                run_dir=run_dir,
                method=method,
                seed=seed,
                eval_events=_load_tensorboard_events(run_dir, "eval/success_once"),
                frame_events=_load_tensorboard_events(
                    run_dir, "time/rollout/profile/control_frame_count"
                ),
                step_time_events=_load_tensorboard_events(run_dir, "time/step"),
                evaluation_time_events=_load_tensorboard_events(
                    run_dir, "time/env/evaluate"
                ),
            )
        )
        pairing_audits[f"{method}/seed_{seed}"] = _pairing_audit(run_dir)

    output_rows.sort(key=lambda row: (row["method"], row["seed"], row["global_step"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    metadata = {
        "wallclock_origin": "completion time of the fixed step-0 evaluation",
        "training_wallclock_s": (
            "cumulative time/step with periodic time/env/evaluate removed; "
            "the pre-training step-0 evaluation is not subtracted from update 1"
        ),
        "environment_frames": (
            "cumulative simulator control frames from the fixed resolved-config "
            "training budget; rollout-profile fallback is explicit per row"
        ),
        "rows": len(output_rows),
        "output": str(args.output.resolve()),
        "pairing_audit": pairing_audits,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
