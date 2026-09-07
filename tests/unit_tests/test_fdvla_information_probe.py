import torch

from examples.analysis.train_fdvla_information_probe import (
    _different_trial_permutation,
    _same_task_time_permutation,
)


def _fixture():
    # Each split has two trials and two frames for the same task.
    task = torch.zeros(12, dtype=torch.long)
    split = torch.arange(3, dtype=torch.long).repeat_interleave(4)
    trial = torch.tensor([0, 0, 1, 1] * 3)
    frame = torch.tensor([0, 16, 0, 16] * 3)
    return task, trial, frame, split


def test_same_task_time_shuffle_never_crosses_split_or_reuses_source():
    task, trial, frame, split = _fixture()
    permutation = _same_task_time_permutation(task, trial, frame, split, seed=7)

    assert torch.equal(split[permutation], split)
    assert torch.equal(task[permutation], task)
    assert not torch.any((trial[permutation] == trial) & (frame[permutation] == frame))


def test_different_trial_shuffle_never_crosses_split():
    task, trial, _frame, split = _fixture()
    permutation = _different_trial_permutation(task, trial, split, seed=7)

    assert torch.equal(split[permutation], split)
    assert torch.equal(task[permutation], task)
    assert not torch.any(trial[permutation] == trial)
