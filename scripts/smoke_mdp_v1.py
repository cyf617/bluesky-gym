"""Smoke-test MDP v1: reset, a few steps, obs keys, and action scales."""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from competitor.config import ACTION_FREQUENCY, D_HEADING, D_SPEED
from competitor.envs import CompetitionEnvV1
from competitor.vec_env import RewardTermsInfo


REQUIRED_KEYS = {
    "cos_drift", "sin_drift", "waypoint_distance", "airspeed",
    "in_conflict", "in_restricted", "outside_sector",
    "x_r", "y_r", "vx_r", "vy_r", "neighbor_distance",
    "tcpa", "dcpa", "cos_rel_hdg", "sin_rel_hdg", "intruding", "valid",
    "obstacle_distance", "obstacle_cos_bearing", "obstacle_sin_bearing", "obstacle_radius",
    "obstacle_signed_distance", "obstacle_boundary_cos", "obstacle_boundary_sin",
    "sector_point_distance", "sector_point_cos_bearing", "sector_point_sin_bearing",
    "inside_sector", "sector_signed_distance",
}


def _check_obs(obs, n_neighbors):
    missing = REQUIRED_KEYS - set(obs)
    assert not missing, f"missing obs keys: {missing}"
    assert obs["valid"].shape == (n_neighbors,)
    assert obs["x_r"].shape == (n_neighbors,)
    live = int(obs["valid"].sum())
    assert 0 <= live <= n_neighbors
    # Empty slots must be zeroed.
    if live < n_neighbors:
        assert np.allclose(obs["x_r"][live:], 0.0)


def smoke_sa():
    env = CompetitionEnvV1(render_mode=None)
    assert env.d_heading == D_HEADING
    assert env.d_speed == D_SPEED
    assert env.action_frequency == ACTION_FREQUENCY
    obs, info = env.reset(seed=0)
    _check_obs(obs, n_neighbors=env.n_intruders)
    assert "waypoint_reached" in info
    assert "total_reward" in info
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float64))
        _check_obs(obs, n_neighbors=env.n_intruders)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    print("single-agent MDP v1 smoke: ok "
          f"(neighbors={env.n_intruders}, obs_keys={len(obs)})")
    # Do not env.close() — BlueSky quit() poisons the process-global sim.


def smoke_goal_relative():
    """A constant heading offset must still arrive (it circles in relative mode)."""
    env = RewardTermsInfo(CompetitionEnvV1(
        render_mode=None, mdp_overrides={"ACTION_MODE": "goal_relative"}))
    env.reset(seed=0)
    terminated = truncated = False
    total = 0.0
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(np.array([0.3, 0.0]))
        total += reward
    assert info["waypoint_reached"] == 1, "goal_relative constant offset did not arrive"
    assert np.isclose(sum(info["reward_terms"].values()), total)
    print(f"goal_relative smoke: ok (arrived in {info['flight_time']:.0f} s)")


if __name__ == "__main__":
    smoke_sa()
    smoke_goal_relative()
    print("all smoke checks passed")
