"""Step 0: official-env + neutral [0, 0] baseline.

Runs the official evaluation harness for 20 episodes of seed 42 (the quick
check). Full 1000-episode numbers belong in the submission, not here.

    SDL_VIDEODRIVER=dummy python -m scripts.run_neutral_baseline
"""

from __future__ import annotations

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
OUT_CSV = os.path.join("competitor", "results", "baseline_sa_neutral_20.csv")
OUT_JSON = os.path.join("competitor", "results", "baseline_sa_neutral_20.json")


def table_from_records(records):
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
        "seed": SEED,
        "episodes": len(records),
        "note": "Do not use total_reward. Compare later runs on these 8 numbers: "
                "goal rate must not drop; safety items must go down.",
        "metrics": rows,
    }


def main():
    print(f"Step 0 baseline: {N_QUICK} episodes, seed {SEED}, official env, "
          f"neutral [0, 0]. Official reported run is 1000 episodes.")
    records = evaluate("sa", N_QUICK, n_agents=10, model_path=None,
                       out_path=OUT_CSV, mdp_v1=False)
    summary = table_from_records(records)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwrote 8-metric summary to {OUT_JSON}")
    print("\nBaseline table (ignore total_reward):")
    print("| Metric | Mean | Std |")
    print("| --- | ---: | ---: |")
    for row in summary["metrics"]:
        print(f"| {row['metric']} | {row['mean']:.3f} | {row['std']:.3f} |")


if __name__ == "__main__":
    main()
