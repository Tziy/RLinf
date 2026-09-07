"""Causal post-hoc semantic-delay augmentation for embodied rollouts.

The augmentation implemented here is deliberately *not* a PPO sample.  It
re-pairs the current state/action history with a semantic tensor that was
already present in the same rollout buffer and causally available at the
current action boundary.  The returned ``ppo_eligible`` mask is therefore
always false; callers may use the rows only in an explicitly separate
auxiliary or offline objective.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

_REQUIRED_METADATA = (
    "rollout_semantic_env_ids",
    "rollout_semantic_episode_generations",
    "rollout_semantic_source_frame_ids",
    "rollout_semantic_versions",
    "rollout_semantic_completed_wallclock_s",
    "action_frame_ids",
    "rollout_semantic_fetch_completed_wallclock_s",
)


def _time_batch_tensor(
    forward_inputs: dict[str, Any], key: str, time_steps: int, batch_size: int
) -> torch.Tensor:
    value = forward_inputs.get(key)
    if not torch.is_tensor(value) or value.shape[:2] != (time_steps, batch_size):
        shape = None if not torch.is_tensor(value) else tuple(value.shape)
        raise ValueError(
            f"Posthoc semantic delay requires {key} with leading shape "
            f"[{time_steps}, {batch_size}], got {shape}"
        )
    return value


def _deterministic_delta_index(
    flat_index: int, *, seed: int, global_step: int, count: int
) -> int:
    # Integer-only mixing keeps selection reproducible across Python processes.
    mixed = (
        int(flat_index) * 1_103_515_245
        + int(seed) * 12_345
        + int(global_step) * 2_654_435_761
    ) & 0x7FFF_FFFF
    return mixed % count


def build_posthoc_semantic_delay_augmentation(
    forward_inputs: dict[str, Any],
    *,
    control_hz: float,
    delta_frames: Sequence[int] = (-4, -2, 2, 4, 8),
    seed: int = 0,
    global_step: int = 0,
    causal_epsilon_s: float = 1e-6,
    allow_hindsight_completion: bool = False,
) -> dict[str, Any]:
    """Build one fixed-shape semantic re-pairing per row.

    Negative deltas request a fresher packet and positive deltas request an
    older packet relative to the semantic tensor actually consumed during the
    rollout. A row is invalid when the requested direction has no distinct,
    eligible packet. Invalid rows retain their original tensor so the result
    remains stackable, but their ``valid_mask`` is false.

    Candidate packets must have the same ``(env_id, episode_generation)``, a
    source frame no later than the current action frame, and by default a
    completion time no later than the semantic-fetch cutoff. Explicit hindsight
    mode may relax only the completion cutoff. Such rows are labeled as not
    online-available and remain ineligible for PPO. Ages are recomputed from
    frame IDs; requested ages are never written as if they were actual ages.
    """

    if not delta_frames:
        raise ValueError("posthoc semantic delay delta_frames must not be empty")
    deltas = tuple(int(value) for value in delta_frames)
    if any(value == 0 for value in deltas):
        raise ValueError("posthoc semantic delay deltas must be non-zero")
    if not any(value < 0 for value in deltas) or not any(value > 0 for value in deltas):
        raise ValueError(
            "posthoc semantic delay requires both negative and positive deltas"
        )
    if not torch.isfinite(torch.tensor(float(control_hz))) or control_hz <= 0:
        raise ValueError(f"control_hz must be positive and finite, got {control_hz}")

    semantic_features = forward_inputs.get("semantic_backbone_features")
    if not torch.is_tensor(semantic_features) or semantic_features.ndim < 3:
        raise ValueError(
            "Posthoc semantic delay requires semantic_backbone_features with "
            "leading [time, batch] dimensions"
        )
    time_steps, batch_size = semantic_features.shape[:2]
    for key in _REQUIRED_METADATA:
        _time_batch_tensor(forward_inputs, key, time_steps, batch_size)

    flat_count = time_steps * batch_size
    flattened = {
        key: value.reshape(flat_count, *value.shape[2:])
        for key, value in forward_inputs.items()
        if torch.is_tensor(value) and value.shape[:2] == (time_steps, batch_size)
    }
    actual_env_ids = flattened["rollout_semantic_env_ids"].reshape(flat_count).cpu()
    actual_generations = (
        flattened["rollout_semantic_episode_generations"].reshape(flat_count).cpu()
    )
    actual_source_frames = (
        flattened["rollout_semantic_source_frame_ids"].reshape(flat_count).cpu()
    )
    actual_versions = flattened["rollout_semantic_versions"].reshape(flat_count).cpu()
    actual_completed = (
        flattened["rollout_semantic_completed_wallclock_s"]
        .reshape(flat_count)
        .double()
        .cpu()
    )
    action_frames = flattened["action_frame_ids"].reshape(flat_count).cpu()
    fetch_cutoffs = (
        flattened["rollout_semantic_fetch_completed_wallclock_s"]
        .reshape(flat_count)
        .double()
        .cpu()
    )

    has_fresh_bank = "posthoc_fresh_valid_mask" in flattened
    candidate_env_ids = actual_env_ids
    candidate_generations = actual_generations
    candidate_source_frames = actual_source_frames
    candidate_versions = actual_versions
    candidate_completed = actual_completed
    candidate_available = torch.ones(flat_count, dtype=torch.bool)
    if has_fresh_bank:
        required_fresh_metadata = {
            "posthoc_fresh_semantic_env_ids",
            "posthoc_fresh_semantic_episode_generations",
            "posthoc_fresh_semantic_source_frame_ids",
            "posthoc_fresh_semantic_versions",
            "posthoc_fresh_semantic_completed_wallclock_s",
        }
        missing_fresh = required_fresh_metadata - set(flattened)
        if missing_fresh:
            raise ValueError(
                "Posthoc fresh semantic bank is incomplete: "
                f"missing={sorted(missing_fresh)}"
            )
        candidate_env_ids = torch.cat(
            (
                candidate_env_ids,
                flattened["posthoc_fresh_semantic_env_ids"].reshape(flat_count).cpu(),
            )
        )
        candidate_generations = torch.cat(
            (
                candidate_generations,
                flattened["posthoc_fresh_semantic_episode_generations"]
                .reshape(flat_count)
                .cpu(),
            )
        )
        candidate_source_frames = torch.cat(
            (
                candidate_source_frames,
                flattened["posthoc_fresh_semantic_source_frame_ids"]
                .reshape(flat_count)
                .cpu(),
            )
        )
        candidate_versions = torch.cat(
            (
                candidate_versions,
                flattened["posthoc_fresh_semantic_versions"].reshape(flat_count).cpu(),
            )
        )
        candidate_completed = torch.cat(
            (
                candidate_completed,
                flattened["posthoc_fresh_semantic_completed_wallclock_s"]
                .reshape(flat_count)
                .double()
                .cpu(),
            )
        )
        candidate_available = torch.cat(
            (
                candidate_available,
                flattened["posthoc_fresh_valid_mask"].reshape(flat_count).bool().cpu(),
            )
        )

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen_packets: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for index in range(candidate_env_ids.numel()):
        if not bool(candidate_available[index]):
            continue
        group = (int(candidate_env_ids[index]), int(candidate_generations[index]))
        packet = (
            int(candidate_source_frames[index]),
            int(candidate_versions[index]),
        )
        if packet not in seen_packets[group]:
            seen_packets[group].add(packet)
            groups[group].append(index)

    selected = torch.arange(flat_count, dtype=torch.long)
    valid = torch.zeros(flat_count, dtype=torch.bool)
    requested_delta = torch.empty(flat_count, dtype=torch.int64)
    target_age = torch.empty(flat_count, dtype=torch.int64)
    original_age = (action_frames - actual_source_frames).to(torch.int64)
    augmented_age = original_age.clone()
    candidate_was_online_available = torch.ones(flat_count, dtype=torch.bool)

    for index in range(flat_count):
        delta = deltas[
            _deterministic_delta_index(
                index, seed=seed, global_step=global_step, count=len(deltas)
            )
        ]
        requested_delta[index] = delta
        target = max(0, int(original_age[index]) + delta)
        target_age[index] = target
        group = (int(actual_env_ids[index]), int(actual_generations[index]))
        candidates: list[tuple[int, int, int]] = []
        for candidate in groups[group]:
            if int(candidate_source_frames[candidate]) == int(
                actual_source_frames[index]
            ) and int(candidate_versions[candidate]) == int(actual_versions[index]):
                continue
            if int(candidate_source_frames[candidate]) > int(action_frames[index]):
                continue
            online_available = float(candidate_completed[candidate]) <= (
                float(fetch_cutoffs[index]) + causal_epsilon_s
            )
            if not online_available and not allow_hindsight_completion:
                continue
            age = int(action_frames[index]) - int(candidate_source_frames[candidate])
            if (delta < 0 and age >= int(original_age[index])) or (
                delta > 0 and age <= int(original_age[index])
            ):
                continue
            candidates.append((abs(age - target), age, candidate))
        if not candidates:
            continue
        _, age, candidate = min(
            candidates,
            key=lambda item: (
                item[0],
                abs(item[1] - int(original_age[index])),
                item[2],
            ),
        )
        selected[index] = candidate
        augmented_age[index] = age
        candidate_was_online_available[index] = online_available
        valid[index] = True

    replacements: dict[str, torch.Tensor] = {}
    replacement_keys = {
        key
        for key in flattened
        if key.startswith("semantic_")
        or key
        in {
            "rollout_semantic_env_ids",
            "rollout_semantic_episode_generations",
            "rollout_semantic_source_frame_ids",
            "rollout_semantic_versions",
            "rollout_semantic_source_wallclock_s",
            "rollout_semantic_completed_wallclock_s",
            "rollout_semantic_fingerprint",
        }
    }
    fresh_metadata_keys = {
        "rollout_semantic_env_ids": "posthoc_fresh_semantic_env_ids",
        "rollout_semantic_episode_generations": (
            "posthoc_fresh_semantic_episode_generations"
        ),
        "rollout_semantic_source_frame_ids": (
            "posthoc_fresh_semantic_source_frame_ids"
        ),
        "rollout_semantic_versions": "posthoc_fresh_semantic_versions",
        "rollout_semantic_source_wallclock_s": (
            "posthoc_fresh_semantic_source_wallclock_s"
        ),
        "rollout_semantic_completed_wallclock_s": (
            "posthoc_fresh_semantic_completed_wallclock_s"
        ),
        "rollout_semantic_fingerprint": "posthoc_fresh_semantic_fingerprint",
    }
    for key in replacement_keys:
        value = flattened[key]
        value_pool = value
        if has_fresh_bank:
            if key.startswith("semantic_"):
                fresh_key = f"posthoc_fresh_{key}"
            else:
                fresh_key = fresh_metadata_keys[key]
            if fresh_key not in flattened:
                raise ValueError(
                    f"Posthoc fresh semantic bank is missing payload {fresh_key}"
                )
            fresh_value = flattened[fresh_key].to(value.device)
            value_pool = torch.cat((value, fresh_value), dim=0)
        replacements[key] = value_pool.index_select(
            0, selected.to(value.device)
        ).reshape(time_steps, batch_size, *value.shape[1:])

    age_shape = flattened.get("packet_age_s")
    age_dtype = (
        age_shape.dtype if torch.is_tensor(age_shape) else semantic_features.dtype
    )
    age_device = semantic_features.device
    replacements["packet_age_s"] = (
        augmented_age.to(device=age_device, dtype=age_dtype) / float(control_hz)
    ).reshape(time_steps, batch_size)
    replacements["rollout_semantic_actual_age_frames"] = augmented_age.to(
        age_device
    ).reshape(time_steps, batch_size)
    replacements["rollout_semantic_requested_age_frames"] = target_age.to(
        age_device
    ).reshape(time_steps, batch_size)
    replacements["rollout_semantic_bootstrap_clipped"] = (
        (augmented_age != target_age).to(age_device).reshape(time_steps, batch_size)
    )
    replacements["rollout_posthoc_ppo_eligible"] = torch.zeros(
        (time_steps, batch_size), dtype=torch.bool, device=age_device
    )

    return {
        "valid_mask": valid.to(age_device).reshape(time_steps, batch_size),
        "ppo_eligible": torch.zeros(
            (time_steps, batch_size), dtype=torch.bool, device=age_device
        ),
        "candidate_flat_indices": selected.to(age_device).reshape(
            time_steps, batch_size
        ),
        "candidate_is_fresh_bank": selected.ge(flat_count)
        .to(age_device)
        .reshape(time_steps, batch_size),
        "candidate_was_online_available": candidate_was_online_available.to(
            age_device
        ).reshape(time_steps, batch_size),
        "requested_delta_frames": requested_delta.to(age_device).reshape(
            time_steps, batch_size
        ),
        "target_age_frames": target_age.to(age_device).reshape(time_steps, batch_size),
        "original_age_frames": original_age.to(age_device).reshape(
            time_steps, batch_size
        ),
        "augmented_age_frames": augmented_age.to(age_device).reshape(
            time_steps, batch_size
        ),
        "replacements": replacements,
    }


def posthoc_semantic_consistency_loss(
    current_logprobs: torch.Tensor,
    augmented_logprobs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    action_execution_horizon: int = -1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a masked auxiliary log-prob consistency loss.

    The current log-probabilities act as a stop-gradient self-distillation
    target. No behavior/old log-probability enters this objective, so it cannot
    be mistaken for an importance-corrected PPO sample.
    """

    if current_logprobs.shape != augmented_logprobs.shape:
        raise ValueError(
            "Posthoc consistency logprob shape mismatch: "
            f"current={tuple(current_logprobs.shape)} "
            f"augmented={tuple(augmented_logprobs.shape)}"
        )
    if valid_mask.shape[0] != current_logprobs.shape[0]:
        raise ValueError(
            "Posthoc consistency valid_mask must have the same batch dimension"
        )
    mask = valid_mask.bool()
    while mask.ndim < current_logprobs.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(current_logprobs)

    if loss_mask is not None:
        action_mask = loss_mask.bool()
        while action_mask.ndim < current_logprobs.ndim:
            action_mask = action_mask.unsqueeze(-1)
        try:
            action_mask = action_mask.expand_as(current_logprobs)
        except RuntimeError as exc:
            raise ValueError(
                "Posthoc consistency loss_mask is not broadcastable to logprobs: "
                f"mask={tuple(loss_mask.shape)} logprobs={tuple(current_logprobs.shape)}"
            ) from exc
        mask = mask & action_mask

    if action_execution_horizon > 0 and current_logprobs.ndim >= 3:
        action_axis = current_logprobs.ndim - 2
        horizon_mask = torch.arange(
            current_logprobs.shape[action_axis], device=current_logprobs.device
        ) < min(action_execution_horizon, current_logprobs.shape[action_axis])
        horizon_shape = [1] * current_logprobs.ndim
        horizon_shape[action_axis] = current_logprobs.shape[action_axis]
        mask = mask & horizon_mask.reshape(horizon_shape)

    target = current_logprobs.detach().float()
    prediction = augmented_logprobs.float()
    requested_count = mask.sum()
    # Diffusion log-prob tensors can legitimately contain -Inf outside the
    # executed action prefix. Forming (-Inf) - (-Inf) before multiplying by a
    # mask produces NaN (and even a zero-weight control would then poison the
    # PPO gradient). Select finite, executed entries before subtraction.
    finite_mask = mask & torch.isfinite(target) & torch.isfinite(prediction)
    selected_target = target[finite_mask]
    selected_prediction = prediction[finite_mask]
    difference = selected_prediction - selected_target
    count = finite_mask.sum().clamp_min(1).to(prediction.dtype)
    if difference.numel() > 0:
        loss = 0.5 * difference.square().sum() / count
        abs_delta = difference.abs().sum().detach() / count
    else:
        # Keep the replay forward connected to autograd while contributing a
        # finite zero when a micro-batch has no eligible finite log-probability.
        finite_prediction = torch.where(
            torch.isfinite(prediction), prediction, torch.zeros_like(prediction)
        )
        loss = finite_prediction.sum() * 0.0
        abs_delta = torch.zeros((), device=prediction.device)
    metrics = {
        "posthoc/consistency_loss": loss.detach(),
        "posthoc/logprob_abs_delta": abs_delta,
        "posthoc/valid_logprob_count": finite_mask.sum().detach().float(),
        "posthoc/nonfinite_logprob_count": (requested_count - finite_mask.sum())
        .detach()
        .float(),
    }
    return loss, metrics


def posthoc_replay_selected(
    microbatch_ordinal: int,
    replay_microbatch_interval: int,
    replay_microbatch_offset: int = 0,
) -> bool:
    """Return a rank-synchronous deterministic replay decision."""
    if microbatch_ordinal < 0:
        raise ValueError("microbatch_ordinal must be non-negative")
    if replay_microbatch_interval < 1:
        raise ValueError("replay_microbatch_interval must be at least 1")
    if not 0 <= replay_microbatch_offset < replay_microbatch_interval:
        raise ValueError(
            "replay_microbatch_offset must be in [0, replay_microbatch_interval)"
        )
    return (
        microbatch_ordinal - replay_microbatch_offset
    ) % replay_microbatch_interval == 0
