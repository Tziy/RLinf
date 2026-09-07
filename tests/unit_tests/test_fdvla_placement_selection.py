import json

import pytest

from examples.analysis.select_fdvla_d_placement import select_placement


def _write_run(
    tmp_path,
    name,
    step_mean,
    *,
    rollout_mean=None,
    queue_p95_mean=20.0,
    nonfinite=0,
    trainable_vlm=0,
    worktree="same",
):
    run = tmp_path / name
    metadata = run / "fdvla_metadata"
    summary_dir = run / "summary"
    metadata.mkdir(parents=True)
    summary_dir.mkdir()
    provenance = {
        "git_sha.txt": "git",
        "worktree_sha256.txt": worktree,
        "checkpoint_sha256.txt": "checkpoint",
        "backbone_sha256.txt": "backbone",
        "hardware.csv": "0,A100,uuid,81920,driver",
    }
    for filename, value in provenance.items():
        (metadata / filename).write_text(value)
    metrics = {
        tag: {"count": 3, "mean": value, "median": value}
        for tag, value in {
            "time/step": step_mean,
            "time/generate_rollouts": (
                step_mean / 2 if rollout_mean is None else rollout_mean
            ),
            "time/actor_training": step_mean / 3,
        }.items()
    }
    summary = {
        "completed_updates": 5,
        "nonfinite_scalar_count": nonfinite,
        "reward_model_invocation_count": 0,
        "trainable_vlm_parameter_count": trainable_vlm,
        "local_vlm_forward_count": 0,
        "cross_episode_packet_mismatch_count": 0,
        "semantic_replay_fingerprint_mismatch_log_count": 0,
        "stable_timing": {"metrics": metrics},
        "system_profile": {
            "metrics": {
                "time/rollout/profile/semantic_queue_latency_ms_p95": {
                    "count": 3,
                    "mean": queue_p95_mean,
                }
            }
        },
    }
    (summary_dir / "fdvla_training_health.json").write_text(json.dumps(summary))
    return run


def test_selects_fastest_healthy_placement_without_success_input(tmp_path):
    a = _write_run(tmp_path, "actor4_rollout3", 220.0)
    b = _write_run(tmp_path, "colocated4", 200.0)
    coupled = tmp_path / "coupled.json"
    coupled.write_text(
        json.dumps(
            {"stable_timing": {"metrics": {"time/step": {"count": 40, "mean": 210.0}}}}
        )
    )

    report = select_placement(
        {"actor4_rollout3": a, "colocated4": b},
        coupled_summary_path=coupled,
    )

    assert report["selection_inputs_exclude_policy_success"] is True
    assert report["selected_placement"] == "colocated4"
    assert report["eligible_ranking"] == ["colocated4", "actor4_rollout3"]
    assert report["selected_vs_coupled"]["speed_endpoint_passed"] is True
    assert report["selected_vs_coupled"]["d_time_reduction_fraction"] == pytest.approx(
        10 / 210
    )


def test_rejects_faster_but_nonfinite_candidate(tmp_path):
    a = _write_run(tmp_path, "actor4_rollout3", 220.0)
    b = _write_run(tmp_path, "colocated4", 180.0, nonfinite=1)

    report = select_placement({"actor4_rollout3": a, "colocated4": b})

    assert report["selected_placement"] == "actor4_rollout3"
    rejected = next(
        item for item in report["candidates"] if item["name"] == "colocated4"
    )
    assert rejected["eligible"] is False
    assert "nonfinite_scalar_count=1, required=0" in rejected["rejection_reasons"]


def test_rejects_faster_candidate_with_trainable_vlm(tmp_path):
    a = _write_run(tmp_path, "actor4_rollout3", 220.0)
    b = _write_run(tmp_path, "colocated4", 180.0, trainable_vlm=1)

    report = select_placement({"actor4_rollout3": a, "colocated4": b})

    assert report["selected_placement"] == "actor4_rollout3"
    rejected = next(
        item for item in report["candidates"] if item["name"] == "colocated4"
    )
    assert rejected["eligible"] is False
    assert (
        "trainable_vlm_parameter_count=1, required=0" in rejected["rejection_reasons"]
    )


def test_selects_preregistered_four_candidate_system_tuning_set(tmp_path):
    candidates = {
        "actor4_rollout3_double_fetch": _write_run(
            tmp_path, "actor4_rollout3_double_fetch", 230.0
        ),
        "colocated4_double_fetch": _write_run(
            tmp_path, "colocated4_double_fetch", 215.0
        ),
        "actor4_rollout3_single_exact_fetch": _write_run(
            tmp_path, "actor4_rollout3_single_exact_fetch", 218.0
        ),
        "colocated4_single_exact_fetch": _write_run(
            tmp_path, "colocated4_single_exact_fetch", 205.0
        ),
    }

    report = select_placement(candidates, candidate_set="system_tuning")

    assert report["candidate_set"] == "system_tuning"
    assert report["selected_placement"] == "colocated4_single_exact_fetch"
    assert report["eligible_ranking"] == [
        "colocated4_single_exact_fetch",
        "colocated4_double_fetch",
        "actor4_rollout3_single_exact_fetch",
        "actor4_rollout3_double_fetch",
    ]


def test_selects_six_candidate_fallback_set(tmp_path):
    names = (
        "actor4_rollout3_double_fetch",
        "colocated4_double_fetch",
        "actor4_rollout3_single_exact_fetch",
        "colocated4_single_exact_fetch",
        "actor4_rollout3_single_exact_fetch_batch2",
        "colocated4_single_exact_fetch_batch2",
    )
    candidates = {
        name: _write_run(tmp_path, name, 230.0 - index * 3.0)
        for index, name in enumerate(names)
    }

    report = select_placement(candidates, candidate_set="system_tuning_with_fallback")

    assert report["selected_placement"] == names[-1]
    assert len(report["eligible_ranking"]) == 6


def test_rejects_provenance_mismatch(tmp_path):
    a = _write_run(tmp_path, "actor4_rollout3", 220.0)
    b = _write_run(tmp_path, "colocated4", 200.0, worktree="different")

    with pytest.raises(ValueError, match="identical provenance"):
        select_placement({"actor4_rollout3": a, "colocated4": b})


def test_selects_preregistered_semantic_batching_set(tmp_path):
    candidates = {
        "batch1": _write_run(tmp_path, "batch1", 220.0),
        "batch2": _write_run(tmp_path, "batch2", 205.0),
        "batch4": _write_run(tmp_path, "batch4", 210.0),
    }

    report = select_placement(candidates, candidate_set="batching")

    assert report["selection_inputs_exclude_policy_success"] is True
    assert report["selected_placement"] == "batch2"
    assert report["eligible_ranking"] == ["batch2", "batch4", "batch1"]
    assert report["selected_batching"] == {
        "max_requests": 2,
        "target_requests": 2,
        "wait_ms": 1,
    }


def test_batching_ties_use_rollout_then_queue_then_candidate_order(tmp_path):
    candidates = {
        "batch1": _write_run(
            tmp_path, "batch1", 200.0, rollout_mean=90.0, queue_p95_mean=30.0
        ),
        "batch2": _write_run(
            tmp_path, "batch2", 200.0, rollout_mean=80.0, queue_p95_mean=40.0
        ),
        "batch4": _write_run(
            tmp_path, "batch4", 200.0, rollout_mean=80.0, queue_p95_mean=20.0
        ),
    }

    report = select_placement(candidates, candidate_set="batching")

    assert report["eligible_ranking"] == ["batch4", "batch2", "batch1"]
    assert report["ranking_metrics"] == [
        "mean_training_wallclock_per_update",
        "mean_rollout_wallclock",
        "mean_of_per_update_semantic_queue_latency_p95",
        "preregistered_candidate_order",
    ]
