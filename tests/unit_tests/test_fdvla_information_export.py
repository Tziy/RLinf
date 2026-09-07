import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.analysis.collect_fdvla_information_dataset import (
    _label_raw_export,
    _load_raw_export,
    _sha256_path,
)
from rlinf.utils.fdvla_information_export import (
    RAW_SCHEMA_VERSION,
    FdvlaInformationRawExporter,
)


def _observation(frame_ids=(0, 2)):
    return {
        "states": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        "main_images": torch.zeros(2, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(2, 8, 8, 3, dtype=torch.uint8),
        "task_descriptions": ["task zero", "task zero"],
        "elapsed_steps": torch.tensor(frame_ids),
    }


def _forward_inputs(frame_ids=(0, 2)):
    return {
        "semantic_backbone_features": torch.arange(24, dtype=torch.float32).reshape(
            2, 3, 4
        ),
        "semantic_backbone_attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "semantic_image_mask": torch.tensor(
            [[True, False, False], [True, True, False]]
        ),
        "state": torch.arange(16, dtype=torch.float32).reshape(2, 1, 8),
        "action_history": torch.zeros(2, 4, 7),
        "rollout_semantic_actual_age_frames": torch.tensor([0, 2]),
        "rollout_task_ids": torch.tensor([0, 0]),
        "rollout_trial_ids": torch.tensor([10, 11]),
        "action_frame_ids": torch.tensor(frame_ids),
        "rollout_semantic_source_frame_ids": torch.tensor([0, 0]),
        "rollout_semantic_versions": torch.tensor([3, 4]),
        "rollout_semantic_episode_generations": torch.tensor([5, 6]),
    }


def test_raw_export_records_exact_packet_and_episode_outcome(tmp_path):
    exporter = FdvlaInformationRawExporter(
        tmp_path, rank=0, stage_count=1, num_envs_per_stage=2
    )
    exporter.reset_stage(0)
    exporter.record_boundary(0, _observation(), _forward_inputs())
    paths = exporter.finish_step(
        0,
        terminations=torch.tensor([[False, True], [False, False]]),
        truncations=torch.tensor([[False, False], [False, True]]),
    )
    exporter.assert_stage_complete(0)

    assert len(paths) == 2
    rows = [_load_raw_export(path) for path in sorted(paths)]
    assert [bool(row["episode_outcome"][0]) for row in rows] == [True, False]
    assert rows[0]["actual_semantic_age"].tolist() == [0]
    assert rows[1]["actual_semantic_age"].tolist() == [2]
    np.testing.assert_array_equal(
        rows[1]["semantic"][0],
        _forward_inputs()["semantic_backbone_features"][1].numpy(),
    )
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 2


def test_raw_export_requires_real_semantic_image_mask(tmp_path):
    exporter = FdvlaInformationRawExporter(
        tmp_path, rank=0, stage_count=1, num_envs_per_stage=2
    )
    forward_inputs = _forward_inputs()
    del forward_inputs["semantic_image_mask"]
    with pytest.raises(ValueError, match="semantic_image_mask"):
        exporter.record_boundary(0, _observation(), forward_inputs)


def test_raw_export_resume_does_not_overwrite(tmp_path):
    first = FdvlaInformationRawExporter(
        tmp_path, rank=2, stage_count=1, num_envs_per_stage=2
    )
    first.record_boundary(0, _observation(), _forward_inputs())
    first_paths = first.finish_step(
        0,
        terminations=np.array([[True], [True]]),
        truncations=np.zeros((2, 1), dtype=bool),
    )
    second = FdvlaInformationRawExporter(
        tmp_path, rank=2, stage_count=1, num_envs_per_stage=2
    )
    second.record_boundary(0, _observation(), _forward_inputs())
    second_paths = second.finish_step(
        0,
        terminations=np.array([[True], [True]]),
        truncations=np.zeros((2, 1), dtype=bool),
    )
    assert set(first_paths).isdisjoint(second_paths)
    assert len(list(tmp_path.glob("*.npz"))) == 4


class _FakeTeacher:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.compute_dtype = torch.float32

    def parameters(self):
        yield self.weight

    def predict_action_batch(self, observation, **_kwargs):
        frames = observation["elapsed_steps"].float()
        actions = frames[:, None, None].expand(-1, 16, 7).clone()
        return actions, {}


def test_offline_teacher_labels_are_current_frame_only():
    raw = {
        "task_id": np.array([0, 0]),
        "init_state_id": np.array([10, 10]),
        "frame_id": np.array([2, 4]),
        "state": np.zeros((2, 1, 8), dtype=np.float32),
        "raw_state": np.zeros((2, 8), dtype=np.float32),
        "action_history": np.zeros((2, 4, 7), dtype=np.float32),
        "semantic": np.zeros((2, 3, 4), dtype=np.float16),
        "semantic_attention_mask": np.ones((2, 3), dtype=bool),
        "semantic_image_mask": np.zeros((2, 3), dtype=bool),
        "actual_semantic_age": np.array([2, 4]),
        "episode_outcome": np.array([False, False]),
        "episode_generation": np.array([0, 0]),
        "raw_main_image": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "raw_wrist_image": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "task_description": np.array(["task zero", "task zero"]),
    }
    labelled = _label_raw_export(raw, _FakeTeacher(), batch_size=1)
    assert labelled["teacher_action"].shape == (2, 16, 7)
    assert labelled["teacher_action"][:, 0, 0].tolist() == [2.0, 4.0]
    assert "teacher_action" not in raw


def test_raw_loader_rejects_causally_impossible_age(tmp_path):
    path = Path(tmp_path) / "bad.npz"
    arrays = {
        "schema_version": np.asarray(RAW_SCHEMA_VERSION),
        "task_id": np.array([0]),
        "init_state_id": np.array([0]),
        "frame_id": np.array([1]),
        "state": np.zeros((1, 1, 8)),
        "raw_state": np.zeros((1, 8)),
        "action_history": np.zeros((1, 4, 7)),
        "semantic": np.zeros((1, 3, 4)),
        "semantic_attention_mask": np.ones((1, 3), dtype=bool),
        "semantic_image_mask": np.zeros((1, 3), dtype=bool),
        "actual_semantic_age": np.array([2]),
        "semantic_source_frame_id": np.array([-1]),
        "semantic_version": np.array([0]),
        "episode_outcome": np.array([False]),
        "episode_generation": np.array([0]),
        "raw_main_image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "raw_wrist_image": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        "task_description": np.array(["task zero"]),
    }
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="causally impossible"):
        _load_raw_export(path)


def test_raw_loader_rejects_inconsistent_source_frame(tmp_path):
    exporter = FdvlaInformationRawExporter(
        tmp_path, rank=0, stage_count=1, num_envs_per_stage=2
    )
    exporter.record_boundary(0, _observation(), _forward_inputs())
    paths = exporter.finish_step(
        0,
        terminations=np.ones((2, 1), dtype=bool),
        truncations=np.zeros((2, 1), dtype=bool),
    )
    path = paths[0]
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["semantic_source_frame_id"] = arrays["semantic_source_frame_id"] + 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="source frame"):
        _load_raw_export(path)


def test_teacher_checkpoint_file_uses_standard_sha256(tmp_path):
    checkpoint = tmp_path / "full_weights.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    assert _sha256_path(checkpoint) == hashlib.sha256(b"checkpoint bytes").hexdigest()
