from types import SimpleNamespace

import numpy as np
import torch

from rlinf.envs.libero.libero_env import LiberoEnv


def _binary_env(num_envs: int = 1) -> LiberoEnv:
    env = object.__new__(LiberoEnv)
    env.cfg = SimpleNamespace(reward_coef=1.0)
    env.num_envs = num_envs
    env.use_rel_reward = False
    env.use_step_penalty = False
    env.prev_step_reward = np.zeros(num_envs, dtype=np.float32)
    env.success_once = np.zeros(num_envs, dtype=bool)
    return env


def _reward_then_record(env: LiberoEnv, terminated) -> np.ndarray:
    terminated = np.asarray(terminated, dtype=bool)
    reward = np.asarray(env._calc_step_reward(terminated), dtype=np.float32)
    env.success_once |= terminated
    return reward


def test_success_episode_has_exactly_one_binary_reward():
    env = _binary_env()
    rewards = [
        _reward_then_record(env, [False]),
        _reward_then_record(env, [True]),
        _reward_then_record(env, [True]),
    ]

    assert np.concatenate(rewards).tolist() == [0.0, 1.0, 0.0]


def test_failed_episode_rewards_are_all_zero():
    env = _binary_env()

    rewards = [_reward_then_record(env, [False]) for _ in range(5)]

    assert np.concatenate(rewards).tolist() == [0.0] * 5


def test_truncation_is_not_success():
    env = _binary_env()
    truncation = torch.tensor([True])

    reward = _reward_then_record(env, [False])

    assert truncation.item() is True
    assert reward.tolist() == [0.0]
    assert env.success_once.tolist() == [False]


def test_binary_reward_is_not_scaled_differenced_or_penalized():
    env = _binary_env(num_envs=3)

    first = _reward_then_record(env, [True, False, False])
    second = _reward_then_record(env, [True, False, True])

    assert first.tolist() == [1.0, 0.0, 0.0]
    assert second.tolist() == [0.0, 0.0, 1.0]
    assert env.cfg.reward_coef == 1.0
    assert env.use_rel_reward is False
    assert env.use_step_penalty is False
