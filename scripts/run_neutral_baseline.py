"""Step 0: official-env + neutral [0, 0] baseline.

Runs the official evaluation harness for 20 episodes of seed 42 (the quick
check). Full 1000-episode numbers belong in the submission, not here.

``--seed 123`` produces the same table on the training validation seed, so
checkpoint evals (and the dashboard) compare against the same scenarios.

    SDL_VIDEODRIVER=dummy python -m scripts.run_neutral_baseline
    SDL_VIDEODRIVER=dummy python -m scripts.run_neutral_baseline --seed 123
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from scripts.evaluate_competition import (  # noqa: E402
    METRIC_KEYS,
    SEED,
    SUMMARY_ROWS,
    evaluate,
)

N_QUICK = 20
RESULTS = os.path.join("competitor", "results")


def out_paths(seed, n):
    stem = f"baseline_sa_neutral_{n}" if seed == SEED else f"baseline_sa_neutral_seed{seed}_{n}"
    return os.path.join(RESULTS, stem + ".csv"), os.path.join(RESULTS, stem + ".json")


def run_other_seed(seed, n):
    """Neutral [0, 0] on the official env for a non-official seed."""
    import gymnasium as gym
    import bluesky_gym
    from competitor.metric_eval import run_metric_episodes
    from scripts.evaluate_competition import write_csv

    bluesky_gym.register_envs()
    env = gym.make("CompetitionEnv-v0")
    records = run_metric_episodes(env, lambda obs: np.zeros(2, dtype=np.float32), n, seed)
    write_csv(records, out_paths(seed, n)[0])
    return records


def table_from_records(records, seed=SEED):
    cols = {k: np.array([r[k] for r in records], dtype=float) for k in METRIC_KEYS}
    rows = []
    for name, key, tf in SUMMARY_ROWS:
        vals = tf(cols[key]) if tf else cols[key]
        rows.append({
            "metric": name,
            "key": key,
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        })
    return {
        "policy": "neutral_[0,0]",
        "env": "CompetitionEnv-v0",
        "seed": seed,
        "episodes": len(records),
        "note": "Do not use total_reward. Compare later runs on these 8 numbers: "
                "goal rate must not drop; safety items must go down.",
        "metrics": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episodes", type=int, default=N_QUICK)
    args = parser.parse_args()
    out_csv, out_json = out_paths(args.seed, args.episodes)

    print(f"Step 0 baseline: {args.episodes} episodes, seed {args.seed}, official env, "
          f"neutral [0, 0]. Official reported run is 1000 episodes of seed {SEED}.")
    if args.seed == SEED:
        records = evaluate("sa", args.episodes, n_agents=10, model_path=None,
                           out_path=out_csv, mdp_v1=False)
    else:
        records = run_other_seed(args.seed, args.episodes)
    summary = table_from_records(records, seed=args.seed)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwrote 8-metric summary to {out_json}")
    print("\nBaseline table (ignore total_reward):")
    print("| Metric | Mean | Std |")
    print("| --- | ---: | ---: |")
    for row in summary["metrics"]:
        print(f"| {row['metric']} | {row['mean']:.3f} | {row['std']:.3f} |")


if __name__ == "__main__":
    main()
