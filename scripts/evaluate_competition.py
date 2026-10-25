"""
Official evaluation harness for the competition.

Runs the canonical scoring protocol — the **first 1000 episodes of seed 42** —
and prints the standard metrics table (plus a per-episode CSV you can share so
judges can reproduce the numbers). Seed 42 is set on the first reset only;
subsequent resets continue the same RNG stream, giving a deterministic,
reproducible sequence of scenarios.

Run it as a module from the repo root:

    python -m scripts.evaluate_competition --env sa                 # required track
    python -m scripts.evaluate_competition --env ma                 # stretch track
    python -m scripts.evaluate_competition --env sa --episodes 20   # quick check
    python -m scripts.evaluate_competition --env sa --model my_ppo   # your policy
    python -m scripts.evaluate_competition --env sa --mdp-v1         # competitor MDP v1
    python -m scripts.evaluate_competition --env sa --run-dir competitor/results/<run>
                                    # MDP v1 + that run's overrides + its selected.zip

The two functions in the EDIT block below are the only things a competitor
changes: point `make_env` at your env subclass / wrapper stack, and load your
trained policy in `load_policy`. Everything else — the fixed scoring params,
the scenario distribution, and the metrics — stays as-is.
"""

import argparse
import csv
import os

import numpy as np


SEED = 42            # official seed — do not change for a reported run
N_EPISODES = 1000    # official episode count
N_AGENTS_MA = 10     # fixed multi-agent competition config

# The nine scored metrics (produced by the fixed _update_metrics / _get_info).
METRIC_KEYS = [
    "waypoint_reached", "flight_time",
    "intrusion_events", "intrusion_time",
    "restricted_area_events", "time_in_restricted_area",
    "sector_exit_events", "time_outside_sector",
    "total_reward",
]


# ==================== EDIT THESE TWO FOR YOUR SUBMISSION ======================
def make_env(kind, n_agents=N_AGENTS_MA, mdp_v1=False, mdp_overrides=None):
    """Return the environment to evaluate.

    Default = the base competition env. Swap in your own subclass or wrapper
    stack here, e.g. ``return MyObsWrapper(gym.make("CompetitionEnv-v0"))``.
    Do NOT change the fixed scoring params (intrusion_distance, episode_time_limit,
    distance_margin, n_intruders/n_agents, n_obstacles, ...).

    ``mdp_v1=True`` (or ``--mdp-v1``) uses the competitor MDP v1 subclass.
    The scored metrics are unchanged; only observation / reward / action
    mapping differ. Neutral ``[0, 0]`` trajectories match the base env.
    """
    if mdp_v1:
        from gymnasium.wrappers import FlattenObservation
        from competitor.envs import CompetitionEnvV1, CompetitionZooEnvV1
        if kind == "sa":
            # Training uses FlattenObservation + MlpPolicy; eval must match.
            return FlattenObservation(
                CompetitionEnvV1(render_mode=None, mdp_overrides=mdp_overrides))
        return CompetitionZooEnvV1(n_agents=n_agents, mdp_overrides=mdp_overrides)
    if kind == "sa":
        import gymnasium as gym
        import bluesky_gym
        bluesky_gym.register_envs()
        return gym.make("CompetitionEnv-v0")
    from bluesky_zoo.competition_v0 import CompetitionZooEnv
    return CompetitionZooEnv(n_agents=n_agents)


def load_policy(kind, model_path=None):
    """Return an ``act(obs) -> action`` callable.

    Default is the neutral 'fly straight to the goal' policy. Load a trained
    SB3 SAC/PPO zip with ``--model path/to/selected`` (the ``.zip`` suffix is
    optional). The env must match training: for MDP v1 single-agent that means
    ``--mdp-v1`` so observations are flattened.
    """
    if model_path is None:
        return lambda obs: np.zeros(2, dtype=np.float32)
    from stable_baselines3 import PPO, SAC
    try:
        model = SAC.load(model_path)
    except Exception:
        model = PPO.load(model_path)

    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    return act
# =============================================================================


def run_single_agent(n_episodes, act, mdp_v1=False, mdp_overrides=None):
    """One record (final info dict) per episode."""
    env = make_env("sa", mdp_v1=mdp_v1, mdp_overrides=mdp_overrides)
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED if ep == 0 else None)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, info = env.step(act(obs))
        records.append(dict(info))
    env.close()
    return records


def run_multi_agent(n_episodes, act, n_agents, mdp_v1=False, mdp_overrides=None):
    """One record per agent-episode (each agent's final info as it leaves)."""
    env = make_env("ma", n_agents=n_agents, mdp_v1=mdp_v1, mdp_overrides=mdp_overrides)
    records = []
    for ep in range(n_episodes):
        observations, infos = env.reset(seed=SEED if ep == 0 else None)
        while env.agents:
            actions = {a: act(observations[a]) for a in env.agents}
            observations, _, terminations, truncations, infos = env.step(actions)
            for a in infos:
                if terminations[a] or truncations[a]:
                    records.append(dict(infos[a]))
    env.close()
    return records


# Rows shown in the summary table: (label, metric key, transform).
SUMMARY_ROWS = [
    ("Goal completion rate [%]", "waypoint_reached", lambda x: 100.0 * x),
    ("Flight time [s]",          "flight_time",             None),
    ("Intrusion events",         "intrusion_events",        None),
    ("Intrusion time [s]",       "intrusion_time",          None),
    ("Restricted-area events",   "restricted_area_events",  None),
    ("Restricted-area time [s]", "time_in_restricted_area", None),
    ("Sector-exit events",       "sector_exit_events",      None),
    ("Time outside sector [s]",  "time_outside_sector",     None),
]


def summarize(records, label):
    """Print a Markdown metrics table. total_reward is reported separately
    because it comes from the competitor-defined reward, so it is not
    comparable across teams."""
    n = len(records)
    cols = {k: np.array([r[k] for r in records], dtype=float) for k in METRIC_KEYS}

    print(f"\n### {label} — mean over {n} {'episodes' if label.startswith('Single') else 'agent-episodes'} (seed {SEED})\n")
    print("| Metric | Mean | Std |")
    print("| --- | ---: | ---: |")
    for name, key, tf in SUMMARY_ROWS:
        vals = tf(cols[key]) if tf else cols[key]
        print(f"| {name} | {vals.mean():.3f} | {vals.std():.3f} |")
    print(f"\n_total_reward (own reward, not comparable across teams): "
          f"{cols['total_reward'].mean():.3f} ± {cols['total_reward'].std():.3f}_")


def write_csv(records, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index"] + METRIC_KEYS)
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow({"episode_index": i, **{k: r[k] for k in METRIC_KEYS}})
    print(f"\nwrote per-episode metrics to {out_path}")


def evaluate(kind, n_episodes, n_agents, model_path, out_path, mdp_v1=False,
             mdp_overrides=None):
    act = load_policy(kind, model_path)
    if kind == "sa":
        records = run_single_agent(n_episodes, act, mdp_v1=mdp_v1, mdp_overrides=mdp_overrides)
        label = "Single-agent"
    else:
        records = run_multi_agent(n_episodes, act, n_agents, mdp_v1=mdp_v1,
                                  mdp_overrides=mdp_overrides)
        label = "Multi-agent"
    summarize(records, label)
    write_csv(records, out_path or f"competition_eval_{kind}.csv")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["sa", "ma", "both"], default="sa",
                        help="single-agent (required), multi-agent (stretch), or both")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help=f"number of episodes (official: {N_EPISODES})")
    parser.add_argument("--n-agents", type=int, default=N_AGENTS_MA,
                        help=f"agents for the multi-agent env (fixed: {N_AGENTS_MA})")
    parser.add_argument("--model", default=None, help="trained model path (see load_policy)")
    parser.add_argument("--out", default=None, help="per-episode CSV output path")
    parser.add_argument("--mdp-v1", action="store_true",
                        help="evaluate competitor MDP v1 (CompetitionEnvV1 / CompetitionZooEnvV1)")
    parser.add_argument("--run-dir", default=None,
                        help="training run directory: implies --mdp-v1, applies its "
                             "config.json mdp_overrides, and defaults --model to selected.zip")
    args = parser.parse_args()

    mdp_overrides = None
    if args.run_dir:
        import json
        args.mdp_v1 = True
        config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                mdp_overrides = json.load(f).get("mdp_overrides") or None
        if args.model is None:
            args.model = os.path.join(args.run_dir, "selected")
        print(f"run dir {args.run_dir}: model={args.model}  mdp_overrides={mdp_overrides}")

    if args.episodes != N_EPISODES:
        print(f"NOTE: running {args.episodes} episodes — the official protocol is "
              f"{N_EPISODES} episodes of seed {SEED}.")

    kinds = ["sa", "ma"] if args.env == "both" else [args.env]
    for kind in kinds:
        evaluate(kind, args.episodes, args.n_agents, args.model,
                 args.out if args.env != "both" else None, mdp_v1=args.mdp_v1,
                 mdp_overrides=mdp_overrides)


if __name__ == "__main__":
    main()
