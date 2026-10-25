"""Paired comparison of official-protocol CSVs against a neutral reference.

All CSVs must come from ``scripts.evaluate_competition`` with the same seed and
episode count, so row i is the same scenario everywhere.

    python -m scripts.compare_official \\
        --neutral competitor/results/official_seed42_1000_neutral.csv \\
        final=competitor/results/exp2_ppo_goal_reach25/official_seed42_1000_final.csv \\
        final+shield=competitor/results/exp2_ppo_goal_reach25/official_seed42_1000_final_shield.csv
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

METRICS = [
    ("waypoint_reached", "Goal completion [%]", 100.0),
    ("flight_time", "Flight time [s]", 1.0),
    ("intrusion_events", "Intrusion events", 1.0),
    ("intrusion_time", "Intrusion time [s]", 1.0),
    ("restricted_area_events", "Restricted-area events", 1.0),
    ("time_in_restricted_area", "Restricted-area time [s]", 1.0),
    ("sector_exit_events", "Sector-exit events", 1.0),
    ("time_outside_sector", "Time outside sector [s]", 1.0),
]
SAFETY_TIMES = ("intrusion_time", "time_in_restricted_area", "time_outside_sector")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "episode_index"}


def paired(a, b):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return float(d.mean()), float(1.96 * se)


def compare(neutral, runs):
    out = {"neutral": {"means": {k: float(neutral[k].mean()) for k, _, _ in METRICS},
                       "clean_frac": float(np.mean(np.all([neutral[k] == 0 for k in SAFETY_TIMES], axis=0)))},
           "runs": {}}
    for label, cols in runs.items():
        if len(cols["flight_time"]) != len(neutral["flight_time"]):
            raise ValueError(f"{label}: {len(cols['flight_time'])} episodes vs neutral {len(neutral['flight_time'])}")
        entry = {"means": {}, "diff": {}, "ci95": {}}
        for k, _, scale in METRICS:
            entry["means"][k] = float(cols[k].mean())
            entry["diff"][k], entry["ci95"][k] = paired(cols[k], neutral[k])
        entry["clean_frac"] = float(np.mean(np.all([cols[k] == 0 for k in SAFETY_TIMES], axis=0)))
        if "shield_interventions" in cols:
            entry["shield_interventions"] = float(cols["shield_interventions"].mean())
        out["runs"][label] = entry
    return out


def markdown(result, n):
    lines = [f"Paired vs neutral on the same {n} scenarios; Δ = mean(model − neutral) ± 95% CI.", ""]
    labels = list(result["runs"])
    lines.append("| Metric | Neutral | " + " | ".join(labels) + " |")
    lines.append("| --- | ---: | " + " | ".join("---:" for _ in labels) + " |")
    for k, name, scale in METRICS:
        cells = [f"{result['neutral']['means'][k] * scale:.3g}"]
        for lab in labels:
            r = result["runs"][lab]
            cells.append(f"{r['means'][k] * scale:.4g} (Δ {r['diff'][k] * scale:+.3g} ± {r['ci95'][k] * scale:.2g})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = [f"{100 * result['neutral']['clean_frac']:.1f}%"] + [
        f"{100 * result['runs'][lab]['clean_frac']:.1f}%" for lab in labels]
    lines.append("| Episodes with all 3 safety times = 0 | " + " | ".join(cells) + " |")
    cells = ["—"] + [f"{result['runs'][lab].get('shield_interventions', 0):.1f}" for lab in labels]
    lines.append("| Shield interventions / episode | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neutral", required=True)
    p.add_argument("runs", nargs="+", metavar="LABEL=CSV")
    p.add_argument("--json", default=None, help="also write the numbers as JSON")
    args = p.parse_args()
    neutral = load(args.neutral)
    runs = {}
    for item in args.runs:
        label, path = item.split("=", 1)
        runs[label] = load(path)
    result = compare(neutral, runs)
    print(markdown(result, len(neutral["flight_time"])))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
