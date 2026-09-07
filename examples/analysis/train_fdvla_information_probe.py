#!/usr/bin/env python3
"""Train capacity-matched probes for FDVLA predictive-information estimates."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _tensor(array, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), dtype=dtype)


def _masked_semantic_mean(semantic: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if semantic.ndim == 2:
        return semantic
    mask = mask.to(dtype=semantic.dtype)
    while mask.ndim < semantic.ndim:
        mask = mask.unsqueeze(-1)
    return (semantic * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _kmeans(data: torch.Tensor, k: int, seed: int, iterations: int = 50):
    generator = torch.Generator().manual_seed(seed)
    centers = data[torch.randperm(len(data), generator=generator)[:k]].clone()
    for _ in range(iterations):
        labels = torch.cdist(data, centers).argmin(dim=1)
        updated = []
        for cluster in range(k):
            members = data[labels == cluster]
            updated.append(centers[cluster] if len(members) == 0 else members.mean(0))
        next_centers = torch.stack(updated)
        if torch.allclose(next_centers, centers, rtol=0, atol=1e-6):
            break
        centers = next_centers
    return centers, torch.cdist(data, centers).argmin(dim=1)


class Probe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, inputs):
        return self.net(inputs)


def _fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    split: torch.Tensor,
    hidden_dim: int,
    classes: int,
    epochs: int,
    seed: int,
):
    torch.manual_seed(seed)
    model = Probe(features.shape[1], hidden_dim, classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_ids = torch.where(split == 0)[0]
    val_ids = torch.where(split == 1)[0]
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Probe dataset requires non-empty train and validation splits")
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_state = None
    for _ in range(epochs):
        order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        model.train()
        for batch in order.split(256):
            loss = nn.functional.cross_entropy(model(features[batch]), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(
                model(features[val_ids]), labels[val_ids]
            ).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def _ce_by_age(model, features, labels, split, ages):
    model.eval()
    result = {}
    with torch.no_grad():
        logits = model(features)
        losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    test = split == 2
    for age in sorted({int(value) for value in ages.tolist()}):
        mask = test & (ages == age)
        if mask.any():
            result[str(age)] = float(losses[mask].mean().item())
    return result


def _same_task_time_permutation(task_ids, trial_ids, frame_ids, split, seed):
    """Shuffle semantic rows within a task and split, never across splits."""
    permutation = torch.arange(len(task_ids))
    generator = torch.Generator().manual_seed(seed)
    for split_id in torch.unique(split):
        for task_id in torch.unique(task_ids[split == split_id]):
            ids = torch.where((split == split_id) & (task_ids == task_id))[0]
            if len(ids) < 2:
                raise ValueError(
                    "same_task_time_shuffle requires at least two rows per "
                    f"(split, task); got split={int(split_id)} task={int(task_id)}"
                )
            shuffled = ids[torch.randperm(len(ids), generator=generator)]
            permutation[shuffled] = torch.roll(shuffled, shifts=1)
    same_source = (trial_ids[permutation] == trial_ids) & (
        frame_ids[permutation] == frame_ids
    )
    if same_source.any():
        raise ValueError(
            "same_task_time_shuffle could not change every (trial, frame) source"
        )
    return permutation


def _different_trial_permutation(task_ids, trial_ids, split, seed):
    """Select a same-task, different-trial semantic row within each split."""
    permutation = torch.arange(len(task_ids))
    generator = torch.Generator().manual_seed(seed)
    for index in range(len(task_ids)):
        candidates = torch.where(
            (split == split[index])
            & (task_ids == task_ids[index])
            & (trial_ids != trial_ids[index])
        )[0]
        if not len(candidates):
            raise ValueError(
                "different_trial_semantic requires another trial in the same "
                f"(split, task); row={index} split={int(split[index])} "
                f"task={int(task_ids[index])}"
            )
        selected = torch.randint(len(candidates), (1,), generator=generator).item()
        permutation[index] = candidates[selected]
    return permutation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, choices=(32, 64), default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        raise ValueError(
            "--output-dir must be absolute and outside the Git source tree"
        )

    with np.load(args.dataset, allow_pickle=False) as data:
        task_ids = _tensor(data["task_id"], torch.long)
        trial_ids = _tensor(data["init_state_id"], torch.long)
        frame_ids = _tensor(data["frame_id"], torch.long)
        state = _tensor(data["state"]).flatten(1)
        history = _tensor(data["action_history"]).flatten(1)
        semantic = _tensor(data["semantic"])
        semantic_mask = _tensor(data["semantic_attention_mask"])
        ages = _tensor(data["actual_semantic_age"], torch.long)
        teacher = _tensor(data["teacher_action"]).flatten(1)
        split = _tensor(data["split"], torch.long)

    split_values = set(split.tolist())
    if split_values != {0, 1, 2}:
        raise ValueError(
            "Probe dataset requires non-empty train/validation/test splits; "
            f"observed {sorted(split_values)}"
        )

    train_teacher = teacher[split == 0]
    if len(train_teacher) < args.clusters:
        raise ValueError("Training rows must be at least the K-means cluster count")
    centers, _ = _kmeans(train_teacher, args.clusters, args.seed)
    labels = torch.cdist(teacher, centers).argmin(dim=1)

    task_one_hot = nn.functional.one_hot(
        task_ids, num_classes=int(task_ids.max().item()) + 1
    ).float()
    age_feature = ages.float().unsqueeze(1) / 12.0
    base = torch.cat((task_one_hot, state, history, age_feature), dim=1)
    semantic_pooled = _masked_semantic_mean(semantic, semantic_mask)
    full = torch.cat((base, semantic_pooled), dim=1)
    baseline = torch.cat((base, torch.zeros_like(semantic_pooled)), dim=1)

    within_task = semantic_pooled[
        _same_task_time_permutation(task_ids, trial_ids, frame_ids, split, args.seed)
    ]
    different_trial = semantic_pooled[
        _different_trial_permutation(task_ids, trial_ids, split, args.seed)
    ]
    task_only_base = torch.zeros_like(base)
    task_only_base[:, : task_one_hot.shape[1]] = task_one_hot

    feature_sets = {
        "state_history": baseline,
        "semantic": full,
        "same_task_time_shuffle": torch.cat((base, within_task), dim=1),
        "different_trial_semantic": torch.cat((base, different_trial), dim=1),
        "zero_semantic": baseline,
        "task_id_only": torch.cat(
            (task_only_base, torch.zeros_like(semantic_pooled)), dim=1
        ),
    }
    metrics = {}
    models = {}
    for name, features in feature_sets.items():
        model = _fit_probe(
            features,
            labels,
            split,
            args.hidden_dim,
            args.clusters,
            args.epochs,
            args.seed,
        )
        models[name] = model.state_dict()
        metrics[name] = {
            "test_ce_nats_by_age": _ce_by_age(model, features, labels, split, ages)
        }

    base_ce = metrics["state_history"]["test_ce_nats_by_age"]
    semantic_ce = metrics["semantic"]["test_ce_nats_by_age"]
    information = {
        age: (base_ce[age] - semantic_ce[age]) / math.log(2.0)
        for age in sorted(set(base_ce) & set(semantic_ce), key=int)
    }
    if "0" not in information:
        raise ValueError(
            "Test split must contain semantic age 0 for R(d) normalization"
        )
    denominator = max(information.get("0", 0.0), 1e-8)
    retention = {
        age: max(value, 0.0) / denominator for age, value in information.items()
    }
    output = {
        "name": "variational predictive-information estimate",
        "clusters": args.clusters,
        "hidden_dim": args.hidden_dim,
        "split_unit": "(task_id, init_state_id)",
        "negative_control_scope": "within identical split and task only",
        "probe_initialization_seed": args.seed,
        "information_bits_by_age": information,
        "retention_R_by_age": retention,
        "negative_controls": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"codebook": centers, "models": models},
        args.output_dir / "probe_checkpoint.pt",
    )
    (args.output_dir / "probe_metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True)
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
