#!/usr/bin/env python3
"""Collect a leakage-safe FDVLA information-probe dataset.

Raw mode consumes eval-only exports containing the exact semantic packet used by
the policy, runs a separate frozen coupled teacher on each synchronized current
observation, and writes teacher actions only as offline probe labels. Legacy
labelled-input mode assembles already labelled shards. Neither mode runs PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUIRED_FIELDS = (
    "task_id",
    "init_state_id",
    "frame_id",
    "state",
    "action_history",
    "semantic",
    "semantic_attention_mask",
    "semantic_image_mask",
    "actual_semantic_age",
    "teacher_action",
    "episode_outcome",
)
ALLOWED_AGES = {0, 1, 2, 4, 6, 8, 12}
RAW_SCHEMA_VERSION = "fdvla-information-raw-v1"
RAW_REQUIRED_FIELDS = (
    "task_id",
    "init_state_id",
    "frame_id",
    "state",
    "raw_state",
    "action_history",
    "semantic",
    "semantic_attention_mask",
    "semantic_image_mask",
    "actual_semantic_age",
    "semantic_source_frame_id",
    "semantic_version",
    "episode_outcome",
    "episode_generation",
    "raw_main_image",
    "raw_wrist_image",
    "task_description",
)


def _sha256_path(path: Path) -> str:
    """Hash a checkpoint file or directory deterministically without loading it."""
    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
        include_names = False
    elif path.is_dir():
        files = sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
        root = path
        include_names = True
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    for file_path in files:
        if include_names:
            digest.update(str(file_path.relative_to(root)).encode())
            digest.update(b"\0")
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _group_split(task_id: int, init_state_id: int, seed: int) -> int:
    payload = f"{seed}:{task_id}:{init_state_id}".encode()
    bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100
    if bucket < 80:
        return 0
    if bucket < 90:
        return 1
    return 2


def _load_export(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(set(REQUIRED_FIELDS) - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        arrays = {key: np.asarray(data[key]) for key in REQUIRED_FIELDS}
    row_count = len(arrays["task_id"])
    inconsistent = {
        key: len(value)
        for key, value in arrays.items()
        if not value.shape or len(value) != row_count
    }
    if inconsistent:
        raise ValueError(f"{path} has inconsistent row counts: {inconsistent}")
    return arrays


def assemble(inputs: list[Path], split_seed: int) -> dict[str, np.ndarray]:
    if not inputs:
        raise ValueError("At least one labelled FDVLA export is required")
    exports = [_load_export(path) for path in inputs]
    combined = {
        key: np.concatenate([export[key] for export in exports], axis=0)
        for key in REQUIRED_FIELDS
    }
    ages = combined["actual_semantic_age"].astype(np.int64)
    invalid_ages = sorted(set(ages.tolist()) - ALLOWED_AGES)
    if invalid_ages:
        raise ValueError(f"Unexpected actual semantic ages: {invalid_ages}")
    frames = combined["frame_id"].astype(np.int64)
    if np.any(ages < 0) or np.any(ages > frames):
        raise ValueError("actual_semantic_age must satisfy 0 <= age <= current frame")

    task_ids = combined["task_id"].astype(np.int64)
    init_ids = combined["init_state_id"].astype(np.int64)
    split = np.asarray(
        [
            _group_split(int(task_id), int(init_id), split_seed)
            for task_id, init_id in zip(task_ids, init_ids, strict=True)
        ],
        dtype=np.int8,
    )
    group_splits: dict[tuple[int, int], set[int]] = {}
    for task_id, init_id, split_id in zip(task_ids, init_ids, split, strict=True):
        group_splits.setdefault((int(task_id), int(init_id)), set()).add(int(split_id))
    leaked = [group for group, values in group_splits.items() if len(values) != 1]
    if leaked:
        raise RuntimeError(f"Grouped split leakage detected: {leaked[:5]}")

    combined["split"] = split
    combined["schema_version"] = np.asarray([1], dtype=np.int64)
    return combined


def _require_external_absolute_path(path: Path, name: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    path = path.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        path.relative_to(repo_root)
    except ValueError:
        return path
    raise ValueError(f"{name} must be outside the Git source tree")


def _load_raw_export(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema_version"]).item())
        if schema != RAW_SCHEMA_VERSION:
            raise ValueError(f"{path} has unsupported raw schema {schema!r}")
        missing = sorted(set(RAW_REQUIRED_FIELDS) - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing raw fields: {missing}")
        arrays = {key: np.asarray(data[key]) for key in RAW_REQUIRED_FIELDS}
    row_count = len(arrays["task_id"])
    inconsistent = {
        key: len(value)
        for key, value in arrays.items()
        if not value.shape or len(value) != row_count
    }
    if inconsistent:
        raise ValueError(f"{path} has inconsistent raw row counts: {inconsistent}")
    expected_source = arrays["frame_id"].astype(np.int64) - arrays[
        "actual_semantic_age"
    ].astype(np.int64)
    if np.any(expected_source < 0):
        raise ValueError(f"{path} contains causally impossible semantic ages")
    actual_source = arrays["semantic_source_frame_id"].astype(np.int64)
    if not np.array_equal(expected_source, actual_source):
        raise ValueError(
            f"{path} semantic source frame does not equal current frame minus actual age"
        )
    return arrays


def _build_frozen_coupled_teacher(
    checkpoint: Path,
    backbone_model_path: Path,
    *,
    device: str,
    action_chunk: int,
    denoising_steps: int,
):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    config_path = (
        Path(__file__).resolve().parents[1]
        / "embodiment"
        / "config"
        / "model"
        / "gr00t_n1d7.yaml"
    )
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg.model_path = str(checkpoint)
    cfg.backbone_model_path = str(backbone_model_path)
    cfg.num_action_chunks = int(action_chunk)
    cfg.denoising_steps = int(denoising_steps)
    cfg.add_value_head = False
    head = cfg.rl_head_config
    head.add_value_head = False
    head.execution_mode = "coupled"
    head.semantic_server_enabled = False
    head.semantic_server_central_cache = False
    head.drop_local_backbone = False
    head.dit_only_train = True
    head.trainable_modules = [
        "action_head.model",
        "action_head.packet_age_adapter",
        "action_head.action_history_adapter",
        "action_head.value_head",
    ]
    head.require_frozen_vlm = True
    head.action_noise_scale = 0.0
    head.disable_dropout = True
    head.semantic_train_random_age_min_frames = -1
    head.semantic_train_random_age_max_frames = -1
    head.semantic_eval_random_age_min_frames = -1
    head.semantic_eval_random_age_max_frames = -1
    head.semantic_eval_fixed_age_frames = -1
    head.semantic_feature_tokens = 160
    head.semantic_control_only_transform = False
    head.require_packet_age_input = True
    head.initialize_packet_age_adapter = True
    head.initialize_action_history_length = 4
    head.zero_init_new_delay_adapters = True
    head.eval_noise_seed = 1234
    model = get_model(cfg, torch_dtype=torch.bfloat16).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Offline coupled teacher must be completely frozen")
    return model


def _teacher_batch_observation(
    raw: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    batch_size = len(indices)
    return {
        "main_images": torch.from_numpy(raw["raw_main_image"][indices]),
        "wrist_images": torch.from_numpy(raw["raw_wrist_image"][indices]),
        "states": torch.from_numpy(raw["raw_state"][indices]).float(),
        "task_descriptions": [
            str(value) for value in raw["task_description"][indices].tolist()
        ],
        "elapsed_steps": torch.from_numpy(raw["frame_id"][indices].astype(np.int64)),
        "__rlinf_semantic_env_ids": torch.arange(batch_size, dtype=torch.int64),
        "__rlinf_semantic_frame_ids": torch.from_numpy(
            raw["frame_id"][indices].astype(np.int64)
        ),
        "__rlinf_semantic_generations": torch.from_numpy(
            raw["episode_generation"][indices].astype(np.int64)
        ),
        "__rlinf_task_ids": torch.from_numpy(raw["task_id"][indices].astype(np.int64)),
        "__rlinf_trial_ids": torch.from_numpy(
            raw["init_state_id"][indices].astype(np.int64)
        ),
    }


@torch.inference_mode()
def _label_raw_export(
    raw: dict[str, np.ndarray],
    teacher,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    actions = []
    device = next(teacher.parameters()).device
    compute_dtype = getattr(teacher, "compute_dtype", torch.bfloat16)
    for start in range(0, len(raw["task_id"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(raw["task_id"])))
        history = torch.from_numpy(raw["action_history"][indices]).to(
            device=device, dtype=compute_dtype
        )
        teacher._action_history = history.clone()
        teacher._action_history_by_env = {}
        teacher._current_action_history_keys = []
        predicted, _ = teacher.predict_action_batch(
            _teacher_batch_observation(raw, indices),
            mode="eval",
            return_semantic_features=False,
        )
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().float().numpy()
        actions.append(np.asarray(predicted, dtype=np.float32))
    teacher_actions = np.concatenate(actions, axis=0)
    if len(teacher_actions) != len(raw["task_id"]):
        raise RuntimeError("Coupled teacher action count does not match raw rows")
    return {
        "task_id": raw["task_id"],
        "init_state_id": raw["init_state_id"],
        "frame_id": raw["frame_id"],
        "state": raw["state"],
        "action_history": raw["action_history"],
        "semantic": raw["semantic"],
        "semantic_attention_mask": raw["semantic_attention_mask"],
        "semantic_image_mask": raw["semantic_image_mask"],
        "actual_semantic_age": raw["actual_semantic_age"],
        "teacher_action": teacher_actions,
        "episode_outcome": raw["episode_outcome"],
    }


def _assemble_raw_with_teacher(
    raw_inputs: list[Path],
    teacher,
    *,
    batch_size: int,
    split_seed: int,
) -> dict[str, np.ndarray]:
    labelled = [
        _label_raw_export(_load_raw_export(path), teacher, batch_size=batch_size)
        for path in raw_inputs
    ]
    combined = {
        key: np.concatenate([export[key] for export in labelled], axis=0)
        for key in REQUIRED_FIELDS
    }
    ages = combined["actual_semantic_age"].astype(np.int64)
    invalid_ages = sorted(set(ages.tolist()) - ALLOWED_AGES)
    if invalid_ages:
        raise ValueError(f"Unexpected actual semantic ages: {invalid_ages}")
    task_ids = combined["task_id"].astype(np.int64)
    init_ids = combined["init_state_id"].astype(np.int64)
    combined["split"] = np.asarray(
        [
            _group_split(int(task_id), int(init_id), split_seed)
            for task_id, init_id in zip(task_ids, init_ids, strict=True)
        ],
        dtype=np.int8,
    )
    combined["schema_version"] = np.asarray([1], dtype=np.int64)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="Already labelled NPZ shards.",
    )
    source.add_argument(
        "--raw-input",
        type=Path,
        nargs="+",
        help="Eval-only raw episode NPZ files requiring coupled-teacher labels.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260820)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-backbone-model-path", type=Path)
    parser.add_argument("--teacher-device", default="cuda:0")
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--action-chunk", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()

    output = _require_external_absolute_path(args.output, "--output")
    teacher_metadata = None
    if args.raw_input:
        if args.teacher_checkpoint is None or args.teacher_backbone_model_path is None:
            parser.error(
                "--raw-input requires --teacher-checkpoint and "
                "--teacher-backbone-model-path"
            )
        if args.teacher_batch_size <= 0:
            parser.error("--teacher-batch-size must be positive")
        checkpoint = args.teacher_checkpoint.expanduser().resolve()
        backbone = args.teacher_backbone_model_path.expanduser().resolve()
        teacher = _build_frozen_coupled_teacher(
            checkpoint,
            backbone,
            device=args.teacher_device,
            action_chunk=args.action_chunk,
            denoising_steps=args.denoising_steps,
        )
        arrays = _assemble_raw_with_teacher(
            [path.expanduser().resolve() for path in args.raw_input],
            teacher,
            batch_size=args.teacher_batch_size,
            split_seed=args.split_seed,
        )
        teacher_metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256_path(checkpoint),
            "backbone_model_path": str(backbone),
            "execution_mode": "coupled_current_frame",
            "all_parameters_frozen": True,
            "teacher_action_usage": "probe_label_only",
        }
    else:
        arrays = assemble(
            [path.expanduser().resolve() for path in args.input], args.split_seed
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    if teacher_metadata is not None:
        output.with_suffix(output.suffix + ".metadata.json").write_text(
            json.dumps(teacher_metadata, indent=2, sort_keys=True)
        )
    counts = {
        name: int((arrays["split"] == split_id).sum())
        for name, split_id in (("train", 0), ("validation", 1), ("test", 2))
    }
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": int(len(arrays["task_id"])),
                "split_counts": counts,
                "teacher_action_usage": "probe_label_only",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
