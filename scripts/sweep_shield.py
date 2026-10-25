"""Quick shield parameter sweep on a validation seed (not the official seed 42).

For one policy (neutral [0, 0] or a trained model), runs N episodes per shield
configuration and prints/writes means plus the paired difference to the same
policy without the shield on the same scenarios.

    SDL_VIDEODRIVER=dummy python -m scripts.sweep_shield --policy neutral --bank 0 25
    SDL_VIDEODRIVER=dummy python -m scripts.sweep_shield --run-dir competitor/results/exp2_ppo_goal_reach25 --model final
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402

from competitor.metric_eval import EXTRA_KEYS, METRIC_KEYS, run_metric_episodes  # noqa: E402
from competitor.replay import load_policy  # noqa: E402
from competitor.vec_env import make_flattened_sa_env  # noqa: E402

SAFETY = ("intrusion_time", "time_in_restricted_area", "time_outside_sector")


def summarize(records, ref=None):
    out = {k: float(np.mean([r[k] for r in records])) for k in METRIC_KEYS + EXTRA_KEYS}
    out["all_clean"] = float(np.mean([all(r[k] == 0 for k in SAFETY) for r in records]))
    if ref is not None:
        for k in ("waypoint_reached", "flight_time") + SAFETY:
            d = np.array([a[k] - b[k] for a, b in zip(records, ref)], dtype=float)
            out[f"d_{k}"] = float(d.mean())
            out[f"se_{k}"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["neutral", "model"], default="model")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--model", default="final")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--bank", type=float, nargs="+", default=[0.0, 25.0])
    p.add_argument("--lookahead", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    p.add_argument("--buffer", type=float, nargs="+", default=[1.0, 2.0],
                   help="obstacle and sector buffer (km), set together")
    p.add_argument("--intruder-nm", type=float, nargs="+", default=[6.0])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    overrides = {"ACTION_MODE": "goal_relative"}
    algo = None
    if args.policy == "model":
        with open(os.path.join(args.run_dir, "config.json")) as f:
            cfg = json.load(f)
        overrides.update(cfg.get("mdp_overrides") or {})
        algo = cfg.get("algo")
        act = load_policy(os.path.join(args.run_dir, args.model), algo)
        label = f"{os.path.basename(os.path.normpath(args.run_dir))}_{args.model}"
    else:
        act = lambda obs: np.zeros(2)  # noqa: E731
        label = "neutral"

    def run(extra):
        env = make_flattened_sa_env(mdp_overrides={**overrides, **extra})
        return run_metric_episodes(env, act, args.episodes, args.seed)

    ref = run({})
    rows = [{"config": "no shield", **summarize(ref)}]
    for bank, look, buf, inm in itertools.product(args.bank, args.lookahead, args.buffer, args.intruder_nm):
        extra = {"SHIELD": 1, "SHIELD_BANK_DEG": bank, "SHIELD_LOOKAHEAD_S": look,
                 "SHIELD_OBSTACLE_BUFFER_KM": buf, "SHIELD_SECTOR_BUFFER_KM": buf,
                 "SHIELD_INTRUDER_NM": inm}
        name = f"bank={bank:g} look={look:g} buf={buf:g} intr={inm:g}"
        rows.append({"config": name, **summarize(run(extra), ref)})
        r = rows[-1]
        print(f"[{label}] {name}: arrival={r['waypoint_reached']:.2f} "
              f"intr={r['intrusion_time']:.1f} ({r['d_intrusion_time']:+.1f}±{r['se_intrusion_time']:.1f}) "
              f"restr={r['time_in_restricted_area']:.1f} ({r['d_time_in_restricted_area']:+.1f}) "
              f"out={r['time_outside_sector']:.1f} flight={r['flight_time']:.0f} "
              f"clean={r['all_clean']:.2f} shield={r['shield_interventions']:.0f}", flush=True)

    out = args.out or os.path.join("competitor", "results", "shield_sweep",
                                   f"{label}_seed{args.seed}_{args.episodes}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
