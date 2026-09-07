# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eval-only raw exporter for the FDVLA predictive-information dataset."""

from __future__ import annotations

import fcntl
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

RAW_SCHEMA_VERSION = "fdvla-information-raw-v1"


def _as_numpy(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _batch_size(observation: dict[str, Any]) -> int:
    for key in ("states", "main_images", "task_ids", "trial_ids"):
        value = observation.get(key)
        if value is not None:
            return int(len(value))
    raise ValueError("Cannot infer information-export observation batch size")


def _row(value: Any, index: int, batch_size: int, *, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), batch_size, axis=0)
    if len(array) != batch_size:
        raise ValueError(
            f"Information-export field {name} has {len(array)} rows; "
            f"expected {batch_size}"
        )
    return np.array(array[index], copy=True)


def _atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FdvlaInformationRawExporter:
    """Persist the exact semantic packet consumed at each eval action boundary.

    This eval-only path stores no reward, return, advantage, or teacher action.
    A frozen coupled teacher is run later in a separate offline process.
    """

    _REQUIRED_FORWARD_FIELDS = (
        "semantic_backbone_features",
        "semantic_image_mask",
        "state",
        "action_history",
        "rollout_semantic_actual_age_frames",
        "rollout_task_ids",
        "rollout_trial_ids",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        rank: int,
        stage_count: int,
        num_envs_per_stage: int,
    ) -> None:
        output = Path(output_dir).expanduser()
        if not output.is_absolute():
            raise ValueError("FDVLA information export directory must be absolute")
        repo_root = Path(__file__).resolve().parents[2]
        try:
            output.resolve().relative_to(repo_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "FDVLA information NPZ export must be outside the Git source tree"
            )
        self.output_dir = output.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.stage_count = int(stage_count)
        self.num_envs_per_stage = int(num_envs_per_stage)
        self._buffers = [
            [defaultdict(list) for _ in range(self.num_envs_per_stage)]
            for _ in range(self.stage_count)
        ]
        self._completed = [
            [False for _ in range(self.num_envs_per_stage)]
            for _ in range(self.stage_count)
        ]
        existing_counters = []
        for path in self.output_dir.glob(f"raw_rank{self.rank:03d}_*.npz"):
            try:
                existing_counters.append(int(path.stem.rsplit("_", 1)[1]))
            except ValueError:
                continue
        self._file_counter = max(existing_counters, default=-1) + 1

    def reset_stage(self, stage_id: int) -> None:
        """Start a fixed-trial eval epoch and reject incomplete old episodes."""
        for env_id, buffer in enumerate(self._buffers[stage_id]):
            if buffer:
                raise RuntimeError(
                    "FDVLA information export reset with an incomplete episode: "
                    f"stage={stage_id} env={env_id}"
                )
        self._completed[stage_id] = [False] * self.num_envs_per_stage

    @staticmethod
    def _forward_field(
        forward_inputs: dict[str, Any],
        name: str,
        *,
        fallback: Any | None = None,
    ) -> Any:
        value = forward_inputs.get(name, fallback)
        if value is None:
            raise ValueError(f"FDVLA information export is missing {name}")
        return value

    def record_boundary(
        self,
        stage_id: int,
        observation: dict[str, Any],
        forward_inputs: dict[str, Any],
    ) -> None:
        """Record actor inputs and current raw images before executing a chunk."""
        missing = sorted(set(self._REQUIRED_FORWARD_FIELDS) - set(forward_inputs))
        if missing:
            raise ValueError(
                f"FDVLA information export is missing rollout fields: {missing}"
            )
        batch_size = _batch_size(observation)
        if batch_size != self.num_envs_per_stage:
            raise ValueError(
                f"FDVLA information export got batch {batch_size}; "
                f"expected {self.num_envs_per_stage}"
            )
        semantic = self._forward_field(forward_inputs, "semantic_backbone_features")
        attention = forward_inputs.get("semantic_backbone_attention_mask")
        if attention is None:
            semantic_shape = _as_numpy(semantic).shape
            attention = np.ones(semantic_shape[:-1], dtype=bool)
        image_mask = self._forward_field(forward_inputs, "semantic_image_mask")
        actual_age = self._forward_field(
            forward_inputs, "rollout_semantic_actual_age_frames"
        )
        current_frames = forward_inputs.get(
            "action_frame_ids", observation.get("elapsed_steps")
        )
        if current_frames is None:
            raise ValueError("FDVLA information export is missing current frame ids")
        task_descriptions = observation.get("task_descriptions")
        if task_descriptions is None or len(task_descriptions) != batch_size:
            raise ValueError(
                "FDVLA information export requires one task description per row"
            )

        source_frames = forward_inputs.get("rollout_semantic_source_frame_ids")
        if source_frames is None:
            source_frames = _as_numpy(current_frames, np.int64) - _as_numpy(
                actual_age, np.int64
            )
        versions = forward_inputs.get(
            "rollout_semantic_versions", np.full(batch_size, -1, dtype=np.int64)
        )
        generations = forward_inputs.get(
            "rollout_semantic_episode_generations",
            np.full(batch_size, -1, dtype=np.int64),
        )

        for env_id in range(batch_size):
            if self._completed[stage_id][env_id]:
                continue
            buffer = self._buffers[stage_id][env_id]
            values = {
                "task_id": _row(
                    forward_inputs["rollout_task_ids"],
                    env_id,
                    batch_size,
                    name="rollout_task_ids",
                ),
                "init_state_id": _row(
                    forward_inputs["rollout_trial_ids"],
                    env_id,
                    batch_size,
                    name="rollout_trial_ids",
                ),
                "frame_id": _row(
                    current_frames, env_id, batch_size, name="current_frames"
                ),
                "state": _row(
                    forward_inputs["state"], env_id, batch_size, name="state"
                ),
                "raw_state": _row(
                    observation["states"], env_id, batch_size, name="raw_state"
                ),
                "action_history": _row(
                    forward_inputs["action_history"],
                    env_id,
                    batch_size,
                    name="action_history",
                ),
                "semantic": _row(semantic, env_id, batch_size, name="semantic").astype(
                    np.float16
                ),
                "semantic_attention_mask": _row(
                    attention, env_id, batch_size, name="semantic_attention_mask"
                ).astype(bool),
                "semantic_image_mask": _row(
                    image_mask, env_id, batch_size, name="semantic_image_mask"
                ).astype(bool),
                "actual_semantic_age": _row(
                    actual_age, env_id, batch_size, name="actual_semantic_age"
                ),
                "semantic_source_frame_id": _row(
                    source_frames,
                    env_id,
                    batch_size,
                    name="semantic_source_frame_id",
                ),
                "semantic_version": _row(
                    versions, env_id, batch_size, name="semantic_version"
                ),
                "episode_generation": _row(
                    generations, env_id, batch_size, name="episode_generation"
                ),
                "raw_main_image": _row(
                    observation["main_images"],
                    env_id,
                    batch_size,
                    name="main_images",
                ).astype(np.uint8),
                "raw_wrist_image": _row(
                    observation["wrist_images"],
                    env_id,
                    batch_size,
                    name="wrist_images",
                ).astype(np.uint8),
                "task_description": np.asarray(str(task_descriptions[env_id])),
            }
            for name, value in values.items():
                buffer[name].append(value)

    def _save_episode(
        self,
        stage_id: int,
        env_id: int,
        *,
        success: bool,
        truncated: bool,
    ) -> Path:
        buffer = self._buffers[stage_id][env_id]
        if not buffer:
            raise RuntimeError(
                "FDVLA information export completed an episode with no rows"
            )
        arrays = {name: np.stack(values, axis=0) for name, values in buffer.items()}
        row_count = len(arrays["task_id"])
        arrays["episode_outcome"] = np.full(row_count, success, dtype=bool)
        arrays["collection_truncated"] = np.full(row_count, truncated, dtype=bool)
        arrays["schema_version"] = np.asarray(RAW_SCHEMA_VERSION)
        task_id = int(arrays["task_id"][0])
        trial_id = int(arrays["init_state_id"][0])
        filename = (
            f"raw_rank{self.rank:03d}_stage{stage_id:02d}_env{env_id:04d}_"
            f"task{task_id:02d}_trial{trial_id:06d}_{self._file_counter:08d}.npz"
        )
        self._file_counter += 1
        path = self.output_dir / filename
        _atomic_savez(path, arrays)
        _append_manifest(
            self.output_dir / "manifest.jsonl",
            {
                "schema_version": RAW_SCHEMA_VERSION,
                "path": filename,
                "rank": self.rank,
                "stage_id": stage_id,
                "env_id": env_id,
                "task_id": task_id,
                "init_state_id": trial_id,
                "rows": row_count,
                "success": bool(success),
                "truncated": bool(truncated),
            },
        )
        buffer.clear()
        return path

    def finish_step(
        self,
        stage_id: int,
        *,
        terminations: torch.Tensor | np.ndarray,
        truncations: torch.Tensor | np.ndarray,
    ) -> list[Path]:
        """Finalize newly terminated/truncated episodes after a chunk."""
        terminations = _as_numpy(terminations, bool)
        truncations = _as_numpy(truncations, bool)
        if terminations.ndim == 1:
            terminations = terminations[:, None]
        if truncations.ndim == 1:
            truncations = truncations[:, None]
        if terminations.shape[0] != self.num_envs_per_stage:
            raise ValueError("FDVLA information termination batch mismatch")
        if truncations.shape[0] != self.num_envs_per_stage:
            raise ValueError("FDVLA information truncation batch mismatch")
        paths = []
        for env_id in range(self.num_envs_per_stage):
            if self._completed[stage_id][env_id]:
                continue
            success = bool(terminations[env_id].any())
            truncated = bool(truncations[env_id].any())
            if not (success or truncated):
                continue
            paths.append(
                self._save_episode(
                    stage_id,
                    env_id,
                    success=success,
                    truncated=truncated and not success,
                )
            )
            self._completed[stage_id][env_id] = True
        return paths

    def assert_stage_complete(self, stage_id: int) -> None:
        incomplete = [
            env_id
            for env_id, completed in enumerate(self._completed[stage_id])
            if not completed
        ]
        if incomplete:
            raise RuntimeError(
                "FDVLA information eval ended before episode completion for "
                f"stage={stage_id}, envs={incomplete[:16]}"
            )
