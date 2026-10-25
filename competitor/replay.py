"""Record evaluation episodes as JSON for the dashboard replay viewer.

Each episode file holds the static scenario (sector, obstacles, goal, intruder
routes) and one frame per env step: ownship and intruder positions in a local
east/north km frame, the action, the reward terms, and the cumulative official
metrics. The neutral [0, 0] policy is recorded on the same scenarios so the
viewer can draw it as a ghost track.

The scenario sequence depends only on the reset seed (intruders fly fixed
routes and the env RNG is consumed by scenario generation), so episode i of
the model run and episode i of the neutral run are the same scenario; this is
asserted per episode.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import bluesky as bs

from competitor.config import NM2KM
from competitor.geometry import latlon_to_nm
from competitor.metric_eval import METRIC_KEYS
from competitor.vec_env import make_flattened_sa_env

INTRUSION_KM = 5.0 * NM2KM
GOAL_RADIUS_KM = 5.0
MpS2Kt = 1.94384


def _xy(center, lat, lon):
    north, east = latlon_to_nm(center, float(lat), float(lon))
    return [round(east * NM2KM, 3), round(north * NM2KM, 3)]


def _static(base):
    sc = base.scenario
    c = sc.center
    return {
        "sector": [_xy(c, lat, lon) for lat, lon in sc.sector],
        "obstacles": [[_xy(c, lat, lon) for lat, lon in obs.vertices] for obs in sc.obstacles],
        "goal": _xy(c, base.goal_lat, base.goal_lon),
        "routes": {r.ac_id: [_xy(c, lat, lon) for lat, lon in r.waypoints] for r in sc.intruder_routes},
        "intrusion_radius_km": INTRUSION_KM,
        "goal_radius_km": GOAL_RADIUS_KM,
        "agent": base.agent,
    }


def _frame(base, action, reward, terms):
    c = base.scenario.center
    i = bs.traf.id2idx(base.agent)
    m = base.metrics[base.agent]
    intruders = []
    for ac_id in base.intruder_ids:
        j = bs.traf.id2idx(ac_id)
        if j < 0:
            continue
        intruders.append([ac_id, *_xy(c, bs.traf.lat[j], bs.traf.lon[j]), round(float(bs.traf.hdg[j]), 1)])
    return {
        "t": m["flight_time"],
        "agent": [*_xy(c, bs.traf.lat[i], bs.traf.lon[i]), round(float(bs.traf.hdg[i]), 1),
                  round(float(bs.traf.tas[i]) * MpS2Kt, 1)],
        "intruders": intruders,
        "conflict_with": sorted(base._intruders_in_intrusion),
        "in_restricted": bool(base._obstacles_occupied),
        "outside": bool(base._outside_sector),
        "action": [round(float(a), 3) for a in action] if action is not None else None,
        "reward": round(float(reward), 4),
        "terms": {k: round(float(v), 4) for k, v in terms.items() if v},
        "metrics": {k: round(float(m[k]), 3) for k in METRIC_KEYS},
        "shield": bool(getattr(base, "last_shield", {}).get(base.agent, False)) if action is not None else False,
        "shield_count": int(getattr(base, "shield_interventions", {}).get(base.agent, 0)),
    }


def run_episode(env, act, seed=None):
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped
    static = _static(base)
    frames = [_frame(base, None, 0.0, {})]
    done = False
    while not done:
        action = np.asarray(act(obs), dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(_frame(base, action, reward, base.last_reward_terms.get(base.agent, {})))
        done = terminated or truncated
    return static, frames, {**{k: float(info[k]) for k in METRIC_KEYS},
                            "shield_interventions": float(info.get("shield_interventions", 0))}


def load_policy(model_path, algo=None):
    import torch
    from stable_baselines3 import PPO, SAC
    torch.set_num_threads(1)
    classes = {"sac": SAC, "ppo": PPO}
    order = [classes[algo]] if algo in classes else [SAC, PPO]
    for cls in order:
        try:
            model = cls.load(model_path, device="cpu")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"could not load {model_path} as SAC or PPO")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def record_replays(out_dir, act, env_kwargs=None, seed=42, n_episodes=5,
                   with_neutral=True, label="model", meta=None):
    """Write ``ep{i}.json`` + ``index.json`` into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    env = make_flattened_sa_env(**(env_kwargs or {}))
    neutral = lambda obs: np.zeros(2)  # noqa: E731

    model_eps = [run_episode(env, act, seed if i == 0 else None) for i in range(n_episodes)]
    neutral_eps = [None] * n_episodes
    if with_neutral:
        # The ghost is the plain neutral policy: same action mapping, never shielded.
        ghost_kwargs = dict(env_kwargs or {})
        ghost_kwargs["mdp_overrides"] = {**ghost_kwargs.get("mdp_overrides", {}), "SHIELD": 0}
        ghost_env = make_flattened_sa_env(**ghost_kwargs)
        neutral_eps = [run_episode(ghost_env, neutral, seed if i == 0 else None)
                       for i in range(n_episodes)]

    index = []
    for i, ((static, frames, metrics), ghost) in enumerate(zip(model_eps, neutral_eps)):
        payload = {"episode_index": i, "seed": seed, "policy": label, "static": static,
                   "frames": frames, "metrics": metrics}
        if ghost is not None:
            g_static, g_frames, g_metrics = ghost
            assert g_static["goal"] == static["goal"], "neutral replay is a different scenario"
            payload["neutral"] = {
                "frames": [{"t": f["t"], "agent": f["agent"], "conflict_with": f["conflict_with"],
                            "in_restricted": f["in_restricted"], "outside": f["outside"],
                            "metrics": f["metrics"]} for f in g_frames],
                "metrics": g_metrics,
            }
        name = f"ep{i}.json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        index.append({"file": name, "episode_index": i, "metrics": metrics,
                      "neutral_metrics": payload.get("neutral", {}).get("metrics"),
                      "steps": len(frames) - 1})
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"seed": seed, "policy": label, "created_at": time.time(),
                   "episodes": index, **(meta or {})}, f, indent=2)
    return index
