#!/usr/bin/env python3
"""Select a healthy D-PPO placement using preregistered timing only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_CANDIDATE_SETS = {
    "placement": ("actor4_rollout3", "colocated4"),
    "batching": ("batch1", "batch2", "batch4"),
    "system_tuning": (
        "actor4_rollout3_double_fetch",
        "colocated4_double_fetch",
        "actor4_rollout3_single_exact_fetch",
        "colocated4_single_exact_fetch",
    ),
    "system_tuning_with_fallback": (
        "actor4_rollout3_double_fetch",
        "colocated4_double_fetch",
        "actor4_rollout3_single_exact_fetch",
        "colocated4_single_exact_fetch",
        "actor4_rollout3_single_exact_fetch_batch2",
        "colocated4_single_exact_fetch_batch2",
    ),
}
_BATCHING_CONFIGS = {
    "batch1": {"max_requests": 1, "target_requests": 1, "wait_ms": 0},
    "batch2": {"max_requests": 2, "target_requests": 2, "wait_ms": 1},
    "batch4": {"max_requests": 4, "target_requests": 4, "wait_ms": 2},
}

_REQUIRED_ZERO_FIELDS = (
    "nonfinite_scalar_count",
    "reward_model_invocation_count",
    "trainable_vlm_parameter_count",
    "local_vlm_forward_count",
    "cross_episode_packet_mismatch_count",
    "semantic_replay_fingerprint_mismatch_log_count",
)
_REQUIRED_TIMING_TAGS = (
    "time/step",
    "time/generate_rollouts",
    "time/actor_training",
)
_SEMANTIC_QUEUE_P95_TAG = "time/rollout/profile/semantic_queue_latency_ms_p95"
_PROVENANCE_FILES = (
    "git_sha.txt",
    "worktree_sha256.txt",
    "checkpoint_sha256.txt",
    "backbone_sha256.txt",
    "hardware.csv",
)


def _parse_candidate(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected NAME=RUN_DIR") from error
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Candidate name and path must not be empty")
    return name, Path(raw_path)


def _read_provenance(run_dir: Path) -> dict[str, str]:
    metadata_dir = run_dir / "fdvla_metadata"
    result = {}
    for filename in _PROVENANCE_FILES:
        path = metadata_dir / filename
        if not path.is_file():
            raise ValueError(f"Missing placement provenance: {path}")
        result[filename] = path.read_text(encoding="utf-8").strip()
    return result


def _load_candidate(name: str, run_dir: Path, min_stable_samples: int) -> dict:
    summary_path = run_dir / "summary/fdvla_training_health.json"
    if not summary_path.is_file():
        raise ValueError(f"Candidate {name} has no summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rejection_reasons = []
    for field in _REQUIRED_ZERO_FIELDS:
        if summary.get(field) != 0:
            rejection_reasons.append(f"{field}={summary.get(field)!r}, required=0")
    if int(summary.get("completed_updates", 0)) < min_stable_samples + 2:
        rejection_reasons.append(
            f"completed_updates={summary.get('completed_updates')!r}, "
            f"required>={min_stable_samples + 2}"
        )

    stable = summary.get("stable_timing", {}).get("metrics", {})
    timing = {}
    for tag in _REQUIRED_TIMING_TAGS:
        metric = stable.get(tag, {})
        count = int(metric.get("count", 0))
        mean = metric.get("mean")
        median = metric.get("median")
        if count < min_stable_samples:
            rejection_reasons.append(
                f"{tag}.count={count}, required>={min_stable_samples}"
            )
        if mean is None or not math.isfinite(float(mean)):
            rejection_reasons.append(f"{tag}.mean is not finite")
        if median is None or not math.isfinite(float(median)):
            rejection_reasons.append(f"{tag}.median is not finite")
        timing[tag] = {"count": count, "mean": mean, "median": median}

    queue_metric = (
        summary.get("system_profile", {})
        .get("metrics", {})
        .get(_SEMANTIC_QUEUE_P95_TAG, {})
    )
    queue_count = int(queue_metric.get("count", 0))
    queue_mean = queue_metric.get("mean")
    if queue_count < min_stable_samples:
        rejection_reasons.append(
            f"{_SEMANTIC_QUEUE_P95_TAG}.count={queue_count}, "
            f"required>={min_stable_samples}"
        )
    if queue_mean is None or not math.isfinite(float(queue_mean)):
        rejection_reasons.append(f"{_SEMANTIC_QUEUE_P95_TAG}.mean is not finite")
    timing[_SEMANTIC_QUEUE_P95_TAG] = {
        "count": queue_count,
        "mean": queue_mean,
    }

    return {
        "name": name,
        "run_dir": str(run_dir.resolve()),
        "eligible": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "timing": timing,
        "provenance": _read_provenance(run_dir),
    }


def select_placement(
    candidates: dict[str, Path],
    *,
    min_stable_samples: int = 3,
    coupled_summary_path: Path | None = None,
    candidate_set: str = "placement",
) -> dict:
    if candidate_set not in _CANDIDATE_SETS:
        raise ValueError(f"Unknown candidate set: {candidate_set}")
    required_candidates = _CANDIDATE_SETS[candidate_set]
    if set(candidates) != set(required_candidates):
        raise ValueError(
            f"Candidates must be exactly {required_candidates}, got {sorted(candidates)}"
        )
    results = [
        _load_candidate(name, candidates[name], min_stable_samples)
        for name in required_candidates
    ]
    reference = results[0]["provenance"]
    for result in results[1:]:
        if result["provenance"] != reference:
            raise ValueError("Placement candidates do not share identical provenance")

    eligible = [result for result in results if result["eligible"]]
    if not eligible:
        raise ValueError("No healthy placement candidate")
    tie_rank = {name: rank for rank, name in enumerate(required_candidates)}
    ranking = sorted(
        eligible,
        key=lambda result: (
            float(result["timing"]["time/step"]["mean"]),
            float(result["timing"]["time/generate_rollouts"]["mean"]),
            float(result["timing"][_SEMANTIC_QUEUE_P95_TAG]["mean"]),
            tie_rank[result["name"]],
        ),
    )
    selected = ranking[0]
    report = {
        "schema_version": 1,
        "candidate_set": candidate_set,
        "selection_inputs_exclude_policy_success": True,
        "ranking_metrics": [
            "mean_training_wallclock_per_update",
            "mean_rollout_wallclock",
            "mean_of_per_update_semantic_queue_latency_p95",
            "preregistered_candidate_order",
        ],
        "min_stable_samples": min_stable_samples,
        "candidates": results,
        "eligible_ranking": [result["name"] for result in ranking],
        "selected_placement": selected["name"],
        "selected_stable_step_mean_s": selected["timing"]["time/step"]["mean"],
    }
    if candidate_set == "batching":
        report["selected_batching"] = dict(_BATCHING_CONFIGS[selected["name"]])
    if coupled_summary_path is not None:
        coupled = json.loads(coupled_summary_path.read_text(encoding="utf-8"))
        coupled_step = coupled["stable_timing"]["metrics"]["time/step"]
        coupled_mean = float(coupled_step["mean"])
        selected_mean = float(selected["timing"]["time/step"]["mean"])
        report["coupled_reference"] = {
            "summary_path": str(coupled_summary_path.resolve()),
            "stable_sample_count": int(coupled_step["count"]),
            "stable_step_mean_s": coupled_mean,
        }
        report["selected_vs_coupled"] = {
            "d_time_reduction_fraction": (coupled_mean - selected_mean) / coupled_mean,
            "d_throughput_speedup_fraction": coupled_mean / selected_mean - 1.0,
            "speed_endpoint_passed": selected_mean < coupled_mean,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", type=_parse_candidate, required=True
    )
    parser.add_argument(
        "--candidate-set", choices=tuple(_CANDIDATE_SETS), default="placement"
    )
    parser.add_argument("--min-stable-samples", type=int, default=3)
    parser.add_argument("--coupled-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path)
    args = parser.parse_args()
    if args.min_stable_samples < 1:
        raise ValueError("--min-stable-samples must be positive")
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("Duplicate candidate name")
    report = select_placement(
        candidates,
        min_stable_samples=args.min_stable_samples,
        coupled_summary_path=args.coupled_summary,
        candidate_set=args.candidate_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.env_output is not None:
        if args.candidate_set != "batching":
            raise ValueError("--env-output is only valid for candidate-set=batching")
        batching = report["selected_batching"]
        args.env_output.parent.mkdir(parents=True, exist_ok=True)
        args.env_output.write_text(
            "selected_batching={}\n".format(report["selected_placement"])
            + "semantic_batch_max_requests={}\n".format(batching["max_requests"])
            + "semantic_batch_target_requests={}\n".format(batching["target_requests"])
            + "semantic_batch_wait_ms={}\n".format(batching["wait_ms"])
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
