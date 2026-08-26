"""Flattened single-agent env factory for SB3.

BlueSky is a process-global singleton, so training parallelism must be
one env per process (SubprocVecEnv), never DummyVecEnv / concat_vec_envs.
"""

from __future__ import annotations

import os

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from competitor.envs import CompetitionEnvV1


def make_flattened_sa_env(**kwargs):
    """Create one CompetitionEnvV1 with Dict obs flattened to a 1-D Box.

    Extra kwargs are forwarded to ``CompetitionEnvV1`` (for example a lower
    ``n_intruders`` during a later curriculum). Official evaluation must keep
    the defaults (10 intruders, 5 obstacles).
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = CompetitionEnvV1(render_mode=None, **kwargs)
    return FlattenObservation(env)
