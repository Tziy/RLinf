from collections import defaultdict
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from examples.analysis.summarize_fdvla_system_profile import (
    add_control_frame_normalization,
    parse_semantic_server_logs,
)
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Worker
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
    _fdvla_sample_summary,
)


def test_rollout_profile_summary_reports_distribution():
    summary = _fdvla_sample_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["mean"] == 2.5
    assert summary["p50"] == 2.5
    assert summary["p95"] > summary["p50"]


def test_action_boundary_blocking_uses_complete_semantic_path():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.hf_model = SimpleNamespace(_last_semantic_fetch_s=0.012)
    worker.model_cfg = {"rl_head_config": {"execution_mode": "decoupled"}}
    worker._fdvla_profile_samples = defaultdict(list)
    worker._fdvla_action_boundaries = 0
    worker._fdvla_control_frames = 0
    worker._fdvla_local_vlm_forward_count = 0

    worker._record_fdvla_profile(
        actions=torch.zeros(2, 16, 7),
        result={"forward_inputs": {}},
        prediction_s=0.030,
        semantic_fetch_s=0.012,
    )

    assert worker._fdvla_profile_samples["action_boundary_blocking_ms"] == [12.0]
    assert worker._fdvla_profile_samples["dit_generation_excluding_semantic_ms"] == [
        18.0
    ]


def test_rollout_profile_exports_cross_episode_mismatch_count(monkeypatch):
    monkeypatch.setattr(Worker, "pop_execution_times", lambda _self: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker = object.__new__(MultiStepRolloutWorker)
    worker.hf_model = SimpleNamespace(
        _last_semantic_fetch_s=0.0,
        _semantic_cross_episode_packet_mismatch_count=0,
    )
    worker._fdvla_profile_samples = defaultdict(list)
    worker._fdvla_action_boundaries = 0
    worker._fdvla_control_frames = 0
    worker._fdvla_local_vlm_forward_count = 0

    metrics = worker.pop_execution_times()

    assert metrics["profile/cross_episode_packet_mismatch_count"] == 0


def test_rollout_profile_counts_exact_semantic_packet_reuse(monkeypatch):
    monkeypatch.setattr(Worker, "pop_execution_times", lambda _self: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker = object.__new__(MultiStepRolloutWorker)
    worker.hf_model = SimpleNamespace(
        _last_semantic_fetch_s=0.0,
        _semantic_cross_episode_packet_mismatch_count=0,
    )
    worker.model_cfg = {"rl_head_config": {"execution_mode": "decoupled"}}
    worker._fdvla_profile_samples = defaultdict(list)
    worker._fdvla_action_boundaries = 0
    worker._fdvla_control_frames = 0
    worker._fdvla_local_vlm_forward_count = 0
    packet = {
        "rollout_semantic_env_ids": torch.tensor([10, 11]),
        "rollout_semantic_episode_generations": torch.tensor([2, 2]),
        "rollout_semantic_source_frame_ids": torch.tensor([16, 16]),
        "rollout_semantic_versions": torch.tensor([4, 4]),
        "rollout_semantic_fingerprint": torch.tensor([101, 202]),
    }

    for _ in range(2):
        worker._record_fdvla_profile(
            actions=torch.zeros(2, 8, 7),
            result={"forward_inputs": packet},
            prediction_s=0.02,
            semantic_fetch_s=0.0,
        )

    metrics = worker.pop_execution_times()

    assert metrics["profile/semantic_packet_consumption_count"] == 4
    assert metrics["profile/semantic_unique_packet_count"] == 2
    assert metrics["profile/semantic_consecutive_reuse_count"] == 2
    assert metrics["profile/dit_forwards_per_unique_semantic_packet"] == 2
    assert metrics["profile/semantic_consecutive_reuse_fraction"] == 0.5
    assert metrics["profile/semantic_identity_incomplete_count"] == 0


def test_rollout_profile_counts_packet_identity_without_optional_fingerprint(
    monkeypatch,
):
    monkeypatch.setattr(Worker, "pop_execution_times", lambda _self: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker = object.__new__(MultiStepRolloutWorker)
    worker.hf_model = SimpleNamespace(
        _last_semantic_fetch_s=0.0,
        _semantic_cross_episode_packet_mismatch_count=0,
    )
    worker.model_cfg = {"rl_head_config": {"execution_mode": "decoupled"}}
    worker._fdvla_profile_samples = defaultdict(list)
    worker._fdvla_action_boundaries = 0
    worker._fdvla_control_frames = 0
    worker._fdvla_local_vlm_forward_count = 0
    packet = {
        "rollout_semantic_env_ids": torch.tensor([10, 11]),
        "rollout_semantic_episode_generations": torch.tensor([2, 2]),
        "rollout_semantic_source_frame_ids": torch.tensor([16, 16]),
        "rollout_semantic_versions": torch.tensor([4, 4]),
    }

    for _ in range(2):
        worker._record_fdvla_profile(
            actions=torch.zeros(2, 8, 7),
            result={"forward_inputs": packet},
            prediction_s=0.02,
            semantic_fetch_s=0.0,
        )

    metrics = worker.pop_execution_times()

    assert metrics["profile/semantic_packet_consumption_count"] == 4
    assert metrics["profile/semantic_unique_packet_count"] == 2
    assert metrics["profile/dit_forwards_per_unique_semantic_packet"] == 2
    assert metrics["profile/semantic_identity_incomplete_count"] == 0


def test_semantic_server_profile_counts_forwards_and_rows(tmp_path):
    log = tmp_path / "semantic.log"
    log.write_text(
        "Semantic batch requests=1 envs=2 raw_prep_ms=3.0 "
        "raw_prep_wait_ms=4.0 merge_h2d_ms=5.0 forward_ms=10.0 "
        "queue_age_ms=20.0\n"
        "Semantic batch requests=2 envs=3 raw_prep_ms=6.0 "
        "raw_prep_wait_ms=7.0 merge_h2d_ms=8.0 forward_ms=30.0 "
        "queue_age_ms=40.0\n"
    )
    summary = parse_semantic_server_logs([log])
    assert summary["vlm_forward_count"] == 2
    assert summary["semantic_rows"] == 5
    assert summary["vlm_forward_latency_ms"]["p50"] == 20.0
    assert summary["semantic_queue_latency_ms"]["p95"] > 20.0


def test_system_profile_normalizes_physical_calls_and_logical_env_rows():
    summary = add_control_frame_normalization(
        {"vlm_forward_count": 2, "semantic_rows": 8}, 40
    )

    assert summary["physical_vlm_forwards_per_1000_env_control_frames"] == 50.0
    assert summary["logical_semantic_rows_per_1000_env_control_frames"] == 200.0
    assert summary["mean_env_control_frames_per_semantic_row"] == 5.0
    assert summary["mean_semantic_rows_per_physical_vlm_forward"] == 4.0


class _Handle:
    def __init__(self, result, timers):
        self.result = result
        self.timers = timers

    def wait(self):
        return self.result

    def consume_durations(self):
        return self.timers


class _Group:
    def __init__(self, handle):
        self.handle = handle

    def evaluate(self, **_kwargs):
        return self.handle


def test_eval_only_runner_consumes_rollout_profile_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rlinf.runners.embodied_eval_runner.write_evaluate_trials",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        "rlinf.runners.embodied_eval_runner.compute_evaluate_metrics",
        lambda *_args, **_kwargs: {"success_once": 0.0},
    )
    runner = object.__new__(EmbodiedEvalRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "enable_decoupled_mode": True,
                "logger": {"log_path": str(tmp_path)},
            }
        }
    )
    runner.env_channel = object()
    runner.rollout_channel = object()
    runner.env = _Group(_Handle([{}], {"evaluate": 2.0}))
    runner.rollout = _Group(_Handle([], {"profile/semantic_fetch_ms_p95": 7.5}))

    metrics = runner.evaluate()

    assert metrics["success_once"] == 0.0
    assert runner._last_time_metrics["time/env/evaluate"] == 2.0
    assert (
        runner._last_time_metrics["time/rollout/profile/semantic_fetch_ms_p95"] == 7.5
    )


def test_rollout_inference_microbatch_preserves_semantic_logprob_row_alignment():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.inference_micro_batch_size = 2
    worker.enable_dagger = False
    worker.rlt_feature_model = None
    calls = []

    def predict(obs, mode="train"):
        row_ids = obs["states"][:, 0].long()
        calls.append((row_ids.tolist(), list(obs["task_descriptions"]), mode))
        actions = row_ids[:, None, None].expand(-1, 2, 1).float()
        result = {
            "prev_logprobs": (row_ids + 100)[:, None].float(),
            "prev_values": (row_ids + 200)[:, None].float(),
            "forward_inputs": {
                "semantic_backbone_features": row_ids[:, None, None].float(),
                "rollout_semantic_env_ids": row_ids.clone(),
                "rollout_semantic_fingerprint": row_ids + 300,
            },
            "expert_label_flag": False,
        }
        return actions, result

    worker.predict = predict
    observations = {
        "states": torch.arange(5).reshape(5, 1),
        "task_descriptions": [f"task-{index}" for index in range(5)],
        "__rlinf_semantic_env_ids": torch.arange(5),
    }

    actions, result = worker._predict_rollout_actions(observations, mode="train")

    assert calls == [
        ([0, 1], ["task-0", "task-1"], "train"),
        ([2, 3], ["task-2", "task-3"], "train"),
        ([4], ["task-4"], "train"),
    ]
    expected = torch.arange(5)
    torch.testing.assert_close(actions[:, 0, 0], expected.float())
