"""Share of each reward term early vs late in training, from a run's episodes.csv.

Share = mean |episode sum of the term| / sum over terms of the same, so the
columns add up to 100 % and sign is shown separately (mean episode sum).

    python -m scripts.reward_term_shares competitor/results/<run> [--n 200]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np


def term_shares(rows, terms):
    sums = {t: np.array([float(r[f"r_{t}"] or 0.0) for r in rows]) for t in terms}
    mag = {t: float(np.mean(np.abs(v))) for t, v in sums.items()}
    total = sum(mag.values()) or 1.0
    return {t: {"share": mag[t] / total, "mean": float(np.mean(sums[t]))} for t in terms}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--n", type=int, default=200, help="episodes in each window")
    args = p.parse_args()
    for run_dir in args.run_dirs:
        with open(os.path.join(run_dir, "episodes.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        terms = [k[2:] for k in rows[0] if k.startswith("r_")]
        early, late = term_shares(rows[:args.n], terms), term_shares(rows[-args.n:], terms)
        safety = [t for t in terms if t not in ("progress", "reach", "timeout", "drift")]
        print(f"\n### {os.path.basename(os.path.normpath(run_dir))} "
              f"({len(rows)} training episodes; first / last {args.n})\n")
        print("| Term | Early share | Early mean | Late share | Late mean |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for t in terms:
            if early[t]["share"] < 1e-4 and late[t]["share"] < 1e-4:
                continue
            print(f"| {t} | {100 * early[t]['share']:.1f}% | {early[t]['mean']:+.2f} | "
                  f"{100 * late[t]['share']:.1f}% | {late[t]['mean']:+.2f} |")
        es = sum(early[t]["share"] for t in safety)
        ls = sum(late[t]["share"] for t in safety)
        print(f"| **safety terms total** | {100 * es:.1f}% | | {100 * ls:.1f}% | |")


if __name__ == "__main__":
    main()
