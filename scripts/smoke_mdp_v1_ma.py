"""Smoke-test multi-agent MDP v1 in its own process (BlueSky is a singleton)."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from competitor.config import ACTION_FREQUENCY, D_HEADING, D_SPEED
from competitor.envs import CompetitionZooEnvV1


def main():
    env = CompetitionZooEnvV1(n_agents=2, render_mode=None)
    assert env.d_heading == D_HEADING
    assert env.d_speed == D_SPEED
    assert env.action_frequency == ACTION_FREQUENCY
    observations, infos = env.reset(seed=1)
    assert len(env.agents) == 2
    agent = env.agents[0]
    obs = observations[agent]
    assert obs["valid"].shape == (1,)
    assert "sector_signed_distance" in obs
    actions = {a: np.zeros(2, dtype=np.float64) for a in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for a in observations:
        assert isinstance(rewards[a], float)
        assert "waypoint_reached" in infos[a]
    print(f"multi-agent MDP v1 smoke: ok (n_agents={env.n_agents})")


if __name__ == "__main__":
    main()
