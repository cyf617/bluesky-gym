"""Flattened single-agent env factory for SB3.

BlueSky is a process-global singleton, so training parallelism must be
one env per process (SubprocVecEnv), never DummyVecEnv / concat_vec_envs.
"""

from __future__ import annotations

import os

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from competitor.envs import CompetitionEnvV1
from competitor.mdp import REWARD_TERMS


class RewardTermsInfo(gym.Wrapper):
    """Sum ``last_reward_terms`` over an episode and put it in the final info.

    ``info["reward_terms"]`` appears only on the step that ends the episode,
    so ``competitor.telemetry`` can log the per-term episode return from a
    SubprocVecEnv without touching the env's scoring ``info`` keys.
    """

    def reset(self, **kwargs):
        self._sums = dict.fromkeys(REWARD_TERMS, 0.0)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        terms = base.last_reward_terms.get(base.agent, {})
        for k in REWARD_TERMS:
            self._sums[k] += terms.get(k, 0.0)
        if terminated or truncated:
            info = dict(info)
            info["reward_terms"] = dict(self._sums)
        return obs, reward, terminated, truncated, info


def make_flattened_sa_env(**kwargs):
    """Create one CompetitionEnvV1 with Dict obs flattened to a 1-D Box.

    Extra kwargs are forwarded to ``CompetitionEnvV1`` (for example
    ``mdp_overrides`` or a lower ``n_intruders`` during a later curriculum).
    Official evaluation must keep the defaults (10 intruders, 5 obstacles).
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = CompetitionEnvV1(render_mode=None, **kwargs)
    return FlattenObservation(RewardTermsInfo(env))
