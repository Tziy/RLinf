import pytest
import torch
from torch import nn

from rlinf.models.embodiment.gr00t.gr00t_n1d7 import (
    _initialize_value_head_with_seed,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    GR00T_N1_7_ForRLActionPrediction,
    _parameter_count,
    _sampled_parameter_fingerprint,
    _semantic_generation_mismatch_count,
    _semantic_replay_fingerprint,
)


class _AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.action_head = nn.Module()
        self.action_head.model = nn.Linear(4, 5)
        self.action_head.value_head = nn.Linear(5, 1)


class _SeededValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def _init_weights(self):
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)


def test_value_head_seed_is_replicated_and_preserves_caller_rng():
    first = _SeededValueHead()
    second = _SeededValueHead()

    torch.manual_seed(101)
    caller_rng = torch.random.get_rng_state().clone()
    _initialize_value_head_with_seed(first, 7)
    assert torch.equal(torch.random.get_rng_state(), caller_rng)

    torch.manual_seed(202)
    _initialize_value_head_with_seed(second, 7)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_semantic_replay_fingerprint_is_exact_for_cached_tensor():
    semantic = torch.arange(2 * 7 * 5, dtype=torch.bfloat16).reshape(2, 7, 5)
    rollout_fingerprint = _semantic_replay_fingerprint(semantic)

    train_fingerprint = _semantic_replay_fingerprint(semantic.detach().cpu().clone())

    torch.testing.assert_close(
        rollout_fingerprint.cpu(), train_fingerprint, rtol=0, atol=0
    )


def test_semantic_replay_fingerprint_detects_changed_packet():
    semantic = torch.arange(2 * 7 * 5, dtype=torch.float32).reshape(2, 7, 5)
    rollout_fingerprint = _semantic_replay_fingerprint(semantic)
    changed = semantic.clone()
    changed[0, 0, 0] += 1

    train_fingerprint = _semantic_replay_fingerprint(changed)

    assert not torch.equal(rollout_fingerprint[0], train_fingerprint[0])
    assert torch.equal(rollout_fingerprint[1], train_fingerprint[1])


def test_semantic_episode_generation_mismatch_is_counted_and_fatal():
    assert _semantic_generation_mismatch_count([2, 5], [2, 5]) == 0
    assert _semantic_generation_mismatch_count([2, 5], [2, 4]) == 1
    assert _semantic_generation_mismatch_count([2, 5], [2]) == 1

    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    model._semantic_cross_episode_packet_mismatch_count = 0
    model._validate_semantic_episode_generations(
        [2, 5],
        {"episode_generations": [2, 5]},
        context="test",
    )
    assert model._semantic_cross_episode_packet_mismatch_count == 0

    with pytest.raises(RuntimeError, match="cross-episode semantic packets"):
        model._validate_semantic_episode_generations(
            [2, 5],
            {"episode_generations": [2, 4]},
            context="test",
        )
    assert model._semantic_cross_episode_packet_mismatch_count == 1


def test_frozen_vlm_boundary_and_common_initialization_fingerprint():
    first = _AuditModel()
    second = _AuditModel()
    second.load_state_dict(first.state_dict())
    first.backbone.requires_grad_(False)
    second.backbone.requires_grad_(False)

    assert _parameter_count(first, ("backbone.",)) > 0
    assert _parameter_count(first, ("backbone.",), trainable_only=True) == 0
    first_hash = _sampled_parameter_fingerprint(first, ("action_head.model.",))
    second_hash = _sampled_parameter_fingerprint(second, ("action_head.model.",))
    assert first_hash == second_hash
    first_value_hash = _sampled_parameter_fingerprint(
        first, ("action_head.value_head.",), trainable_only=True
    )
    second_value_hash = _sampled_parameter_fingerprint(
        second, ("action_head.value_head.",), trainable_only=True
    )
    first_trainable_hash = _sampled_parameter_fingerprint(
        first, ("",), trainable_only=True
    )
    second_trainable_hash = _sampled_parameter_fingerprint(
        second, ("",), trainable_only=True
    )
    assert first_value_hash == second_value_hash
    assert first_trainable_hash == second_trainable_hash

    with torch.no_grad():
        second.action_head.value_head.weight[0, 0] += 1
    assert first_value_hash != _sampled_parameter_fingerprint(
        second, ("action_head.value_head.",), trainable_only=True
    )
    assert first_trainable_hash != _sampled_parameter_fingerprint(
        second, ("",), trainable_only=True
    )

    with torch.no_grad():
        second.action_head.model.weight[0, 0] += 1
    assert first_hash != _sampled_parameter_fingerprint(second, ("action_head.model.",))
