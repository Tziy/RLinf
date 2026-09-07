#!/usr/bin/env python3
"""Summarize one FDVLA training run into auditable scalar and health reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from examples.analysis.collect_fdvla_learning_curve import (
    _events_by_step,
    _fixed_budget_profile_accounting,
    _load_training_environment_frames_per_update,
)

_REWARD_MODEL_PATTERN = re.compile(r"reward-model invocation count:\s*(\d+)")
_VLM_TRAINABLE_PATTERN = re.compile(r"VLM trainable parameter count:\s*(\d+)")
_NO_LOCAL_VLM_MARKER = "DiT-only worker contains no local VLM parameters"
_REQUIRED_HEALTH_TAGS = (
    "env/success_once",
    "env/return",
    "env/reward",
    "env/num_trajectories",
    "rollout/rewards",
    "rollout/advantages_mean",
    "rollout/advantages_min",
    "rollout/advantages_max",
    "rollout/returns_mean",
    "rollout/returns_min",
    "rollout/returns_max",
    "train/actor/policy_loss",
    "train/actor/ratio",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/grad_norm",
    "train/critic/value_loss",
    "train/critic/explained_variance",
    "time/step",
    "time/generate_rollouts",
    "time/actor_training",
    "time/sync_weights",
    "time/rollout/profile/control_frame_count",
    "time/rollout/profile/local_vlm_forward_count",
    "time/rollout/profile/peak_gpu_memory_mib",
)

_STABLE_TIMING_TAGS = (
    "time/step",
    "time/generate_rollouts",
    "time/actor_training",
    "time/sync_weights",
    "time/rollout/predict",
    "time/rollout/predict/semantic_fetch",
)
_STABLE_TIMING_WARMUP_UPDATES = 2
_POSTHOC_HEALTH_TAGS = (
    "train/posthoc/valid_fraction",
    "train/posthoc/older_fraction",
    "train/posthoc/fresher_fraction",
    "train/posthoc/hindsight_fraction",
    "train/posthoc/replay_selected",
    "train/posthoc/ppo_eligible_count",
    "train/posthoc/nonfinite_logprob_count",
    "train/posthoc/weighted_aux_loss",
)


_SYSTEM_PROFILE_TAGS = (
    "time/rollout/profile/action_boundary_count",
    "time/rollout/profile/action_boundary_blocking_ms_mean",
    "time/rollout/profile/action_boundary_blocking_ms_p50",
    "time/rollout/profile/action_boundary_blocking_ms_p95",
    "time/rollout/profile/action_generation_total_ms_mean",
    "time/rollout/profile/action_generation_total_ms_p50",
    "time/rollout/profile/action_generation_total_ms_p95",
    "time/rollout/profile/dit_generation_excluding_semantic_ms_mean",
    "time/rollout/profile/dit_generation_excluding_semantic_ms_p50",
    "time/rollout/profile/dit_generation_excluding_semantic_ms_p95",
    "time/rollout/profile/semantic_fetch_ms_mean",
    "time/rollout/profile/semantic_fetch_ms_p50",
    "time/rollout/profile/semantic_fetch_ms_p95",
    "time/rollout/profile/semantic_age_frames_mean",
    "time/rollout/profile/semantic_age_frames_p50",
    "time/rollout/profile/semantic_age_frames_p95",
    "time/rollout/profile/semantic_queue_latency_ms_mean",
    "time/rollout/profile/semantic_queue_latency_ms_p50",
    "time/rollout/profile/semantic_queue_latency_ms_p95",
    "time/rollout/profile/semantic_packet_consumption_count",
    "time/rollout/profile/semantic_unique_packet_count",
    "time/rollout/profile/semantic_consecutive_reuse_fraction",
    "time/rollout/profile/dit_forwards_per_unique_semantic_packet",
)

def _summarize_events(events: Iterable[Any]) -> dict[str, Any]:
    events = list(events)
    values = np.asarray([float(event.value) for event in events], dtype=np.float64)
    finite = np.isfinite(values)
    return {
        "count": len(events),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "first": float(values[0]) if len(values) else None,
        "last": float(values[-1]) if len(values) else None,
        "min": float(values[finite].min()) if finite.any() else None,
        "max": float(values[finite].max()) if finite.any() else None,
        "mean": float(values[finite].mean()) if finite.any() else None,
    }


def _stable_timing_summary(scalars: dict[str, list[Any]]) -> dict[str, Any]:
    """Exclude warm-up and the loop step enclosing each fixed evaluation."""
    evaluation_global_steps = sorted(
        {
            int(event.step)
            for event in scalars.get("eval/success_once", [])
            if int(event.step) > 0
        }
    )
    # RLinf increments global_step before evaluation but logs the enclosing
    # time/step event with the preceding zero-based loop step.
    excluded_event_steps = {step - 1 for step in evaluation_global_steps}
    metrics = {}
    for tag in _STABLE_TIMING_TAGS:
        events = [
            event
            for event in scalars.get(tag, [])
            if int(event.step) >= _STABLE_TIMING_WARMUP_UPDATES
            and int(event.step) not in excluded_event_steps
        ]
        summary = _summarize_events(events)
        finite_values = [
            float(event.value) for event in events if math.isfinite(float(event.value))
        ]
        summary["median"] = float(np.median(finite_values)) if finite_values else None
        metrics[tag] = summary
    return {
        "warmup_update_count": _STABLE_TIMING_WARMUP_UPDATES,
        "evaluation_global_steps": evaluation_global_steps,
        "excluded_training_event_steps": sorted(excluded_event_steps),
        "metrics": metrics,
    }


def _run_summary(
    scalars: dict[str, list[Any]],
    run_log_text: str = "",
    training_environment_frames_per_update: int | None = None,
    posthoc_enabled: bool | None = None,
) -> dict[str, Any]:
    missing = [tag for tag in _REQUIRED_HEALTH_TAGS if tag not in scalars]
    health = {
        tag: _summarize_events(scalars.get(tag, [])) for tag in _REQUIRED_HEALTH_TAGS
    }
    nonfinite = [
        {"tag": tag, "step": int(event.step), "value": float(event.value)}
        for tag, events in scalars.items()
        for event in events
        if not math.isfinite(float(event.value))
    ]

    frame_events = scalars.get("time/rollout/profile/control_frame_count", [])
    eval_events = scalars.get("eval/success_once", [])
    frame_accounting = _fixed_budget_profile_accounting(frame_events, eval_events)
    trajectory_events = scalars.get("env/num_trajectories", [])
    success_events = _events_by_step(scalars.get("env/success_once", []))
    trajectories_by_step = _events_by_step(trajectory_events)
    eval_zero = _events_by_step(scalars.get("eval/success_once", [])).get(0)
    latest_training_wall_time = max(
        (float(event.wall_time) for event in success_events.values()), default=None
    )
    elapsed_s = (
        latest_training_wall_time - float(eval_zero.wall_time)
        if eval_zero is not None and latest_training_wall_time is not None
        else None
    )
    control_frames = (
        len(success_events) * training_environment_frames_per_update
        if training_environment_frames_per_update is not None
        else int(round(frame_accounting["training_counter"]))
    )
    profile_control_frame_slots = int(
        round(frame_accounting["profile_counter_including_eval"])
    )
    completed_episodes = int(
        round(sum(float(event.value) for event in trajectory_events))
    )
    successful_episodes = int(
        round(
            sum(
                float(trajectories_by_step[step].value) * float(event.value)
                for step, event in success_events.items()
                if step in trajectories_by_step
            )
        )
    )
    reward_model_counts = [
        int(value) for value in _REWARD_MODEL_PATTERN.findall(run_log_text)
    ]
    vlm_trainable_counts = [
        int(value) for value in _VLM_TRAINABLE_PATTERN.findall(run_log_text)
    ]
    local_vlm_accounting = _fixed_budget_profile_accounting(
        scalars.get("time/rollout/profile/local_vlm_forward_count", []), eval_events
    )
    local_vlm_forward_count = int(round(local_vlm_accounting["training_counter"]))
    cross_episode_mismatch_events = scalars.get(
        "time/rollout/profile/cross_episode_packet_mismatch_count", []
    )
    peak_memory_events = scalars.get("time/rollout/profile/peak_gpu_memory_mib", [])
    posthoc_health = {
        tag: _summarize_events(scalars[tag])
        for tag in _POSTHOC_HEALTH_TAGS
        if tag in scalars
    }
    system_profile = {
        tag: _summarize_events(scalars[tag])
        for tag in _SYSTEM_PROFILE_TAGS
        if tag in scalars
    }
    posthoc_ppo_eligible = posthoc_health.get("train/posthoc/ppo_eligible_count", {})
    posthoc_nonfinite = posthoc_health.get("train/posthoc/nonfinite_logprob_count", {})
    inferred_posthoc_enabled = any(
        tag in posthoc_health
        for tag in (
            "train/posthoc/replay_selected",
            "train/posthoc/ppo_eligible_count",
            "train/posthoc/nonfinite_logprob_count",
        )
    )
    return {
        "completed_updates": len(success_events),
        "actual_control_frames": control_frames,
        "environment_frame_source": (
            "resolved_config_fixed_budget"
            if training_environment_frames_per_update is not None
            else "rollout_profile_fallback"
        ),
        "training_environment_frames_per_update": (
            training_environment_frames_per_update
        ),
        "rollout_profile_control_frame_slots_including_eval": (
            profile_control_frame_slots
        ),
        "rollout_profile_eval_control_frame_slots": int(
            round(frame_accounting["eval_counter"])
        ),
        "rollout_profile_frame_correction_applied": frame_accounting[
            "correction_applied"
        ],
        "rollout_profile_training_control_frame_slots_per_update": (
            frame_accounting["training_counter_per_update"]
        ),
        "completed_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
        "failed_episodes": completed_episodes - successful_episodes,
        "wallclock_from_step0_eval_s": elapsed_s,
        "aggregate_control_frames_per_s": (
            control_frames / elapsed_s if elapsed_s and elapsed_s > 0 else None
        ),
        "completed_episodes_per_hour": (
            3600.0 * completed_episodes / elapsed_s
            if elapsed_s and elapsed_s > 0
            else None
        ),
        "local_vlm_forward_count": local_vlm_forward_count,
        "local_vlm_forward_count_including_eval": int(
            round(local_vlm_accounting["profile_counter_including_eval"])
        ),
        "eval_local_vlm_forward_count": int(
            round(local_vlm_accounting["eval_counter"])
        ),
        "local_vlm_forward_correction_applied": local_vlm_accounting[
            "correction_applied"
        ],
        "cross_episode_packet_mismatch_count": (
            int(
                round(
                    max(float(event.value) for event in cross_episode_mismatch_events)
                )
            )
            if cross_episode_mismatch_events
            else None
        ),
        "peak_rollout_gpu_memory_mib": (
            max(float(event.value) for event in peak_memory_events)
            if peak_memory_events
            else None
        ),
        "reward_model_invocation_count": (
            reward_model_counts[-1] if reward_model_counts else None
        ),
        "trainable_vlm_parameter_count": (
            max(vlm_trainable_counts)
            if vlm_trainable_counts
            else 0
            if _NO_LOCAL_VLM_MARKER in run_log_text
            else None
        ),
        "semantic_replay_fingerprint_mismatch_log_count": run_log_text.count(
            "PPO semantic replay fingerprint mismatch"
        ),
        "system_profile": {
            "aggregation": "summary_of_per-update_profile_scalars",
            "metrics": system_profile,
        },
        "posthoc_contract": {
            "enabled": (
                inferred_posthoc_enabled
                if posthoc_enabled is None
                else posthoc_enabled
            ),
            "enabled_source": (
                "metric_inference"
                if posthoc_enabled is None
                else "resolved_config"
            ),
            "replay_fraction_mean": posthoc_health.get(
                "train/posthoc/replay_selected", {}
            ).get("mean"),
            "ppo_eligible_count_max": posthoc_ppo_eligible.get("max"),
            "nonfinite_logprob_count_max": posthoc_nonfinite.get("max"),
            "metrics": posthoc_health,
        },
        "nonfinite_scalar_count": len(nonfinite),
        "nonfinite_scalars": nonfinite,
        "missing_required_tags": missing,
        "stable_timing": _stable_timing_summary(scalars),
        "health": health,
    }


def _load_scalars(run_dir: Path) -> dict[str, list[Any]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    paths = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    if not paths:
        raise ValueError(f"Run {run_dir} has no TensorBoard event files")
    combined = defaultdict(list)
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            combined[tag].extend(accumulator.Scalars(tag))
    return {
        tag: [events[step] for step in sorted(events)]
        for tag, raw_events in combined.items()
        for events in [_events_by_step(raw_events)]
    }


def _load_posthoc_enabled(run_dir: Path) -> bool | None:
    import yaml

    paths = sorted(run_dir.glob("fdvla_metadata*/resolved_config.yaml"))
    if not paths:
        return None
    values = set()
    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        posthoc = payload.get("algorithm", {}).get(
            "posthoc_semantic_delay_augmentation", {}
        )
        if "enabled" in posthoc:
            values.add(bool(posthoc["enabled"]))
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(
            f"Run {run_dir} has conflicting resolved PosthocAug settings: {values}"
        )
    return values.pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scalars = _load_scalars(args.run_dir)
    run_log_path = args.run_log or args.run_dir / "run_embodiment.log"
    run_log_text = (
        run_log_path.read_text(encoding="utf-8", errors="replace")
        if run_log_path.exists()
        else ""
    )
    summary = _run_summary(
        scalars,
        run_log_text,
        training_environment_frames_per_update=(
            _load_training_environment_frames_per_update(args.run_dir)
        ),
        posthoc_enabled=_load_posthoc_enabled(args.run_dir),
    )
    summary.update(
        {
            "run_dir": str(args.run_dir.resolve()),
            "run_log": str(run_log_path.resolve()),
            "through_global_step": max(
                (int(event.step) + 1 for event in scalars.get("env/success_once", [])),
                default=0,
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "fdvla_training_health.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_dir / "fdvla_training_scalars.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=("tag", "step", "value", "wall_time")
        )
        writer.writeheader()
        for tag in sorted(scalars):
            for event in scalars[tag]:
                writer.writerow(
                    {
                        "tag": tag,
                        "step": int(event.step),
                        "value": float(event.value),
                        "wall_time": float(event.wall_time),
                    }
                )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
