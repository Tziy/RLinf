import pytest
import torch

from rlinf.algorithms.posthoc_semantic_delay import (
    build_posthoc_semantic_delay_augmentation,
    posthoc_replay_selected,
    posthoc_semantic_consistency_loss,
)


def _rollout_inputs(*, future_packet_completed_at: float = 2.5):
    # Packet source=2 is first consumed at t=2, but it was already completed at
    # wall-clock 2.5. It is therefore available by the t=1 semantic-fetch cutoff
    # at wall-clock 3.0 and can safely make that deliberately delayed row fresher.
    semantic = torch.tensor([10.0, 10.0, 20.0, 20.0]).reshape(4, 1, 1, 1)
    return {
        "semantic_backbone_features": semantic,
        "semantic_backbone_attention_mask": torch.tensor(
            [1, 1, 2, 2], dtype=torch.int64
        ).reshape(4, 1, 1),
        "rollout_semantic_fingerprint": torch.tensor(
            [[101], [101], [202], [202]], dtype=torch.int64
        ).reshape(4, 1, 1),
        "rollout_semantic_env_ids": torch.zeros(4, 1, dtype=torch.int64),
        "rollout_semantic_episode_generations": torch.zeros(4, 1, dtype=torch.int64),
        "rollout_semantic_source_frame_ids": torch.tensor(
            [0, 0, 2, 2], dtype=torch.int64
        ).reshape(4, 1),
        "rollout_semantic_versions": torch.tensor(
            [1, 1, 2, 2], dtype=torch.int64
        ).reshape(4, 1),
        "rollout_semantic_source_wallclock_s": torch.tensor(
            [0.0, 0.0, 2.0, 2.0], dtype=torch.float64
        ).reshape(4, 1),
        "rollout_semantic_completed_wallclock_s": torch.tensor(
            [0.5, 0.5, future_packet_completed_at, future_packet_completed_at],
            dtype=torch.float64,
        ).reshape(4, 1),
        "action_frame_ids": torch.tensor([0, 2, 4, 6], dtype=torch.int64).reshape(4, 1),
        "action_wallclock_s": torch.tensor(
            [1.0, 3.0, 5.0, 7.0], dtype=torch.float64
        ).reshape(4, 1),
        "rollout_semantic_fetch_completed_wallclock_s": torch.tensor(
            [1.0, 3.0, 5.0, 7.0], dtype=torch.float64
        ).reshape(4, 1),
        "packet_age_s": torch.tensor([0.0, 0.1, 0.1, 0.2]).reshape(4, 1),
    }


def _add_causal_fresh_bank(forward_inputs):
    forward_inputs.update(
        {
            "posthoc_fresh_semantic_backbone_features": torch.tensor(
                [10.0, 30.0, 40.0, 50.0]
            ).reshape(4, 1, 1, 1),
            "posthoc_fresh_semantic_backbone_attention_mask": torch.tensor(
                [1, 3, 4, 5], dtype=torch.int64
            ).reshape(4, 1, 1),
            "posthoc_fresh_semantic_fingerprint": torch.tensor(
                [101, 303, 404, 505], dtype=torch.int64
            ).reshape(4, 1, 1),
            "posthoc_fresh_semantic_env_ids": torch.zeros(4, 1, dtype=torch.int64),
            "posthoc_fresh_semantic_episode_generations": torch.zeros(
                4, 1, dtype=torch.int64
            ),
            "posthoc_fresh_semantic_source_frame_ids": torch.tensor(
                [0, 2, 4, 6], dtype=torch.int64
            ).reshape(4, 1),
            "posthoc_fresh_semantic_versions": torch.tensor(
                [1, 3, 4, 5], dtype=torch.int64
            ).reshape(4, 1),
            "posthoc_fresh_semantic_source_wallclock_s": torch.tensor(
                [0.0, 2.0, 4.0, 6.0], dtype=torch.float64
            ).reshape(4, 1),
            "posthoc_fresh_semantic_completed_wallclock_s": torch.tensor(
                [0.5, 2.5, 4.5, 6.5], dtype=torch.float64
            ).reshape(4, 1),
            "posthoc_fresh_valid_mask": torch.ones(4, 1, dtype=torch.bool),
        }
    )
    return forward_inputs


def test_posthoc_delay_can_truthfully_shrink_and_enlarge_age():
    augmented = build_posthoc_semantic_delay_augmentation(
        _rollout_inputs(),
        control_hz=20.0,
        delta_frames=(2, -2),
        seed=0,
        global_step=0,
    )

    # t=1 shrinks age 2 -> 0 using source frame 2.  t=2 enlarges age 2 -> 4
    # using source frame 0.  Both packets completed before the action boundary.
    assert augmented["valid_mask"][:, 0].tolist() == [False, True, True, False]
    assert augmented["original_age_frames"][:, 0].tolist() == [0, 2, 2, 4]
    assert augmented["augmented_age_frames"][:, 0].tolist() == [0, 0, 4, 4]
    assert augmented["target_age_frames"][:, 0].tolist() == [2, 0, 4, 2]

    replacements = augmented["replacements"]
    assert replacements["semantic_backbone_features"][:, 0, 0, 0].tolist() == [
        10.0,
        20.0,
        10.0,
        20.0,
    ]
    assert replacements["semantic_backbone_attention_mask"][:, 0, 0].tolist() == [
        1,
        2,
        1,
        2,
    ]
    assert replacements["rollout_semantic_fingerprint"][:, 0, 0].tolist() == [
        101,
        202,
        101,
        202,
    ]
    assert replacements["packet_age_s"][:, 0].tolist() == pytest.approx(
        [0.0, 0.0, 0.2, 0.2]
    )
    assert not augmented["ppo_eligible"].any()
    assert not replacements["rollout_posthoc_ppo_eligible"].any()


def test_posthoc_delay_rejects_packet_completed_after_action_boundary():
    augmented = build_posthoc_semantic_delay_augmentation(
        _rollout_inputs(future_packet_completed_at=3.5),
        control_hz=20.0,
        delta_frames=(2, -2),
    )

    # The otherwise fresher source=2 packet did not exist at t=1 action time.
    assert not augmented["valid_mask"][1, 0]
    assert augmented["candidate_flat_indices"][1, 0].item() == 1


def test_posthoc_delay_labels_hindsight_completion_and_keeps_it_out_of_ppo():
    augmented = build_posthoc_semantic_delay_augmentation(
        _rollout_inputs(future_packet_completed_at=3.5),
        control_hz=20.0,
        delta_frames=(2, -2),
        allow_hindsight_completion=True,
    )

    # Source frame 2 belongs to the current state's past, but its frozen-VLM
    # computation completed after the action. It is legal only as hindsight aux.
    assert augmented["valid_mask"][1, 0]
    assert augmented["augmented_age_frames"][1, 0].item() == 0
    assert not augmented["candidate_was_online_available"][1, 0]
    assert not augmented["ppo_eligible"][1, 0]


def test_posthoc_delay_uses_causally_available_fresh_bank_to_shrink_age():
    forward_inputs = _add_causal_fresh_bank(
        _rollout_inputs(future_packet_completed_at=3.5)
    )
    augmented = build_posthoc_semantic_delay_augmentation(
        forward_inputs,
        control_hz=20.0,
        delta_frames=(2, -2),
        seed=0,
        global_step=0,
    )

    # At t=1 the rollout used source=0 (age 2).  Source=2 had completed before
    # the fetch boundary, so it is a truthful age-0 auxiliary sample only.
    assert augmented["valid_mask"][1, 0]
    assert augmented["candidate_is_fresh_bank"][1, 0]
    assert augmented["original_age_frames"][1, 0].item() == 2
    assert augmented["augmented_age_frames"][1, 0].item() == 0
    replacements = augmented["replacements"]
    assert replacements["semantic_backbone_features"][1, 0, 0, 0].item() == 30.0
    assert replacements["rollout_semantic_source_frame_ids"][1, 0].item() == 2
    assert replacements["rollout_semantic_fingerprint"][1, 0, 0].item() == 303
    assert not augmented["ppo_eligible"][1, 0]


def test_posthoc_delay_never_crosses_episode_generation():
    forward_inputs = _rollout_inputs()
    forward_inputs["rollout_semantic_episode_generations"][2:] = 1
    augmented = build_posthoc_semantic_delay_augmentation(
        forward_inputs,
        control_hz=20.0,
        delta_frames=(2, -2),
    )

    assert not augmented["valid_mask"].any()


@pytest.mark.parametrize("delta_frames", [(), (2, 4), (-4, -2), (-2, 0, 2)])
def test_posthoc_delay_requires_two_sided_nonzero_delta_grid(delta_frames):
    with pytest.raises(ValueError):
        build_posthoc_semantic_delay_augmentation(
            _rollout_inputs(), control_hz=20.0, delta_frames=delta_frames
        )


def test_posthoc_consistency_uses_only_valid_executed_actions():
    current = torch.zeros(2, 4, 2)
    augmented = torch.ones(2, 4, 2, requires_grad=True)
    valid = torch.tensor([True, False])
    loss, metrics = posthoc_semantic_consistency_loss(
        current,
        augmented,
        valid,
        loss_mask=torch.ones(2, 4, dtype=torch.bool),
        action_execution_horizon=2,
    )

    assert loss.item() == pytest.approx(0.5)
    assert metrics["posthoc/valid_logprob_count"].item() == 4
    loss.backward()
    assert augmented.grad[0, :2].ne(0).all()
    assert augmented.grad[0, 2:].eq(0).all()
    assert augmented.grad[1].eq(0).all()


def test_posthoc_consistency_filters_nonfinite_logprobs_before_subtraction():
    current = torch.tensor([[[0.0], [float("-inf")], [float("-inf")]]])
    augmented = torch.tensor(
        [[[1.0], [float("-inf")], [float("-inf")]]], requires_grad=True
    )
    loss, metrics = posthoc_semantic_consistency_loss(
        current,
        augmented,
        torch.tensor([True]),
        loss_mask=torch.ones(1, 3, dtype=torch.bool),
        action_execution_horizon=2,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5)
    assert metrics["posthoc/valid_logprob_count"].item() == 1
    assert metrics["posthoc/nonfinite_logprob_count"].item() == 1
    loss.backward()
    assert torch.isfinite(augmented.grad).all()
    assert augmented.grad[0, 0].item() == pytest.approx(1.0)
    assert augmented.grad[0, 1:].eq(0).all()


def test_posthoc_consistency_all_nonfinite_is_connected_finite_zero():
    current = torch.full((1, 2, 1), float("-inf"))
    augmented = torch.full((1, 2, 1), float("-inf"), requires_grad=True)
    loss, metrics = posthoc_semantic_consistency_loss(
        current,
        augmented,
        torch.tensor([True]),
        loss_mask=torch.ones(1, 2, dtype=torch.bool),
        action_execution_horizon=2,
    )

    assert loss.item() == 0.0
    assert metrics["posthoc/valid_logprob_count"].item() == 0
    assert metrics["posthoc/nonfinite_logprob_count"].item() == 2
    loss.backward()
    assert torch.isfinite(augmented.grad).all()
    assert augmented.grad.eq(0).all()


def test_posthoc_replay_schedule_has_exact_rank_synchronous_one_in_eight_coverage():
    selected = [
        ordinal
        for ordinal in range(64)
        if posthoc_replay_selected(
            ordinal,
            replay_microbatch_interval=8,
            replay_microbatch_offset=3,
        )
    ]

    assert selected == list(range(3, 64, 8))


@pytest.mark.parametrize(
    ("ordinal", "interval", "offset"),
    [(-1, 8, 0), (0, 0, 0), (0, 8, -1), (0, 8, 8)],
)
def test_posthoc_replay_schedule_rejects_invalid_coordinates(ordinal, interval, offset):
    with pytest.raises(ValueError):
        posthoc_replay_selected(
            ordinal,
            replay_microbatch_interval=interval,
            replay_microbatch_offset=offset,
        )
