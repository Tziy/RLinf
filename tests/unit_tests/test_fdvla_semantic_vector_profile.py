from types import SimpleNamespace

import pytest
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def packet(fingerprint):
    return {
        "rollout_semantic_env_ids": torch.tensor([10, 11]),
        "rollout_semantic_episode_generations": torch.tensor([2, 2]),
        "rollout_semantic_source_frame_ids": torch.tensor([16, 16]),
        "rollout_semantic_versions": torch.tensor([4, 4]),
        "rollout_semantic_fingerprint": fingerprint,
    }


def test_real_vector_fingerprint_counts_reuse_and_fractional_changes():
    worker = SimpleNamespace()
    fingerprint = torch.arange(128, dtype=torch.float32).reshape(2, 64) / 1000
    original = fingerprint.clone()
    inputs = packet(fingerprint)
    record = MultiStepRolloutWorker._record_fdvla_semantic_reuse
    record(worker, inputs)
    record(worker, inputs)
    assert getattr(worker, "_fdvla_semantic_packet_consumptions", 0) == 4
    assert len(worker._fdvla_semantic_unique_packets) == 2
    assert worker._fdvla_semantic_consecutive_reuses == 2
    assert torch.equal(fingerprint, original)
    changed = fingerprint.clone()
    changed[1, 63] += 0.0005
    record(worker, packet(changed))
    assert worker._fdvla_semantic_packet_consumptions == 6
    assert len(worker._fdvla_semantic_unique_packets) == 3
    assert worker._fdvla_semantic_consecutive_reuses == 3
    assert getattr(worker, "_fdvla_semantic_identity_incomplete_count", 0) == 0


@pytest.mark.parametrize(
    "fingerprint",
    [
        torch.zeros(4, 32),
        torch.empty(2, 0),
        torch.tensor(1.0),
        torch.full((2, 64), float("nan")),
    ],
)
def test_invalid_vector_fingerprint_is_marked_incomplete(fingerprint):
    worker = SimpleNamespace()
    MultiStepRolloutWorker._record_fdvla_semantic_reuse(worker, packet(fingerprint))
    assert worker._fdvla_semantic_identity_incomplete_count == 1
    assert getattr(worker, "_fdvla_semantic_packet_consumptions", 0) == 0
