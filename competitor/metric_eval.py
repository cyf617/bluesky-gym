"""Official-metric evaluation used to pick checkpoints.

This is a *validation* loop (default seed 123), not the seed-42 submission
protocol. It still reads the fixed scoring ``info`` keys — never training
return — so checkpoint decisions match how the competition is judged.
"""

from __future__ import annotations

import numpy as np

METRIC_KEYS = [
    "waypoint_reached", "flight_time",
    "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector",
    "total_reward",
]

# Neutral-policy 20-episode means on the default validation seed 123, from
# competitor/results/baseline_sa_neutral_seed123_20.json (seed 42: baseline_sa_neutral_20.json).
NEUTRAL_BASELINE = {
    "waypoint_reached": 1.0,
    "flight_time": 885.5,
    "intrusion_events": 0.9,
    "intrusion_time": 64.75,
    "restricted_area_events": 1.15,
    "time_in_restricted_area": 159.15,
    "sector_exit_events": 0.0,
    "time_outside_sector": 0.0,
}


def run_metric_episodes(env, act, n_episodes, seed):
    """Run ``n_episodes`` and return per-episode metric dicts.

    ``seed`` is set only on the first reset so the remaining episodes follow
    the same RNG stream (same convention as the official harness).
    """
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed if ep == 0 else None)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(act(obs))
        records.append({k: info[k] for k in METRIC_KEYS})
    return records


def summarize_records(records):
    cols = {k: np.array([r[k] for r in records], dtype=float) for k in METRIC_KEYS}
    means = {k: float(cols[k].mean()) for k in METRIC_KEYS}
    stds = {k: float(cols[k].std()) for k in METRIC_KEYS}
    return means, stds


def arrival_and_safety(means):
    """The two numbers used to pick a checkpoint. Training return is ignored."""
    return float(means["waypoint_reached"]), float(means["intrusion_time"])
