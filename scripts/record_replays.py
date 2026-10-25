"""Record a trained policy (plus the neutral ghost) for the dashboard replay tab.

    SDL_VIDEODRIVER=dummy python -m scripts.record_replays --run-dir competitor/results/<run>
    SDL_VIDEODRIVER=dummy python -m scripts.record_replays --run-dir <run> --model best_arrival --episodes 10

Uses the run's ``config.json`` MDP overrides so the policy sees the same
observation/action mapping it was trained with. Writes ``<run>/replays/``.
The default seed 42 replays the first episodes of the official protocol.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from competitor.replay import load_policy, record_replays  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="selected",
                        help="checkpoint name inside the run dir, or a path (default selected)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-neutral", action="store_true", help="skip the neutral ghost track")
    args = parser.parse_args()

    config = {}
    config_path = os.path.join(args.run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    overrides = config.get("mdp_overrides") or {}
    model_path = args.model if os.path.sep in args.model else os.path.join(args.run_dir, args.model)

    print(f"recording {args.episodes} episodes of seed {args.seed}: model={model_path} "
          f"overrides={overrides or 'none'}")
    index = record_replays(
        os.path.join(args.run_dir, "replays"),
        load_policy(model_path, config.get("algo")),
        env_kwargs={"mdp_overrides": overrides} if overrides else {},
        seed=args.seed, n_episodes=args.episodes, with_neutral=not args.no_neutral,
        label=os.path.basename(model_path), meta={"model": os.path.basename(model_path)},
    )
    for e in index:
        m, n = e["metrics"], e["neutral_metrics"] or {}
        print(f"  ep{e['episode_index']}: reached={m['waypoint_reached']:.0f} "
              f"flight={m['flight_time']:.0f}s intrusion={m['intrusion_time']:.0f}s "
              f"restricted={m['time_in_restricted_area']:.0f}s"
              + (f"  | neutral intrusion={n['intrusion_time']:.0f}s "
                 f"restricted={n['time_in_restricted_area']:.0f}s" if n else ""))


if __name__ == "__main__":
    main()
