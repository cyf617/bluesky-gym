"""Checkpoint selection from official metrics, not SB3 episode return.

Why not training return
-----------------------
SB3's ``ep_rew_mean`` is the sum of *our* shaped reward (progress per km,
conflict penalties, +80 on arrival, …). That scalar is:

- not what the competition scores
- dominated by the +80 arrival bonus, so a policy can keep a high return
  while still flying through conflicts (the neutral [0,0] policy already
  arrives 100% of the time)
- not comparable after we change reward weights

What we store instead
---------------------
Every eval, run N episodes and read ``info['waypoint_reached']`` and
``info['intrusion_time']`` (plus the other official safety columns for the
log). Save:

- ``best_arrival`` — highest arrival rate; tie-break is lower intrusion time
- ``best_safety``  — lowest intrusion time among evals whose arrival rate
  stays at least ``min_arrival`` (default 0.90)
- ``selected``     — the safety checkpoint if one exists, otherwise the
  arrival checkpoint (the model we would actually submit next)

``total_reward`` / SB3 return are logged only so we can *see* they disagree
with the official columns.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from competitor.checkpoint_rule import is_better_arrival, is_better_safety
from competitor.metric_eval import (
    METRIC_KEYS,
    NEUTRAL_BASELINE,
    arrival_and_safety,
    run_metric_episodes,
    summarize_records,
)
from competitor.vec_env import make_flattened_sa_env


class OfficialMetricCheckpointCallback(BaseCallback):
    """Evaluate on official metrics every ``eval_freq`` timesteps and save."""

    def __init__(self, save_dir, eval_freq, eval_episodes, eval_seed,
                 min_arrival=0.90, env_kwargs=None, verbose=1):
        super().__init__(verbose)
        self.env_kwargs = dict(env_kwargs or {})
        self.save_dir = save_dir
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed = int(eval_seed)
        self.min_arrival = float(min_arrival)
        self._next_eval = 0
        self._eval_env = None
        self.best_arrival = None
        self.best_safety = None
        self.history = []
        self.log_path = os.path.join(save_dir, "eval_log.csv")
        self.log_fields = (["timesteps", "sb3_ep_rew_mean"] + METRIC_KEYS
                           + [f"{k}_std" for k in METRIC_KEYS])
        self.selection_path = os.path.join(save_dir, "selection.json")

    def _init_callback(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.log_fields)
                writer.writeheader()

    def _get_eval_env(self):
        if self._eval_env is None:
            self._eval_env = make_flattened_sa_env(**self.env_kwargs)
        return self._eval_env

    def _act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    def _on_step(self):
        if self.num_timesteps < self._next_eval:
            return True
        self._run_eval()
        self._next_eval = self.num_timesteps + self.eval_freq
        return True

    def _run_eval(self):
        env = self._get_eval_env()
        records = run_metric_episodes(
            env, self._act, self.eval_episodes, self.eval_seed)
        means, stds = summarize_records(records)
        arrival, intrusion = arrival_and_safety(means)
        ep_buffer = getattr(self.model, "ep_info_buffer", None) or []
        sb3_return = (float(np.mean([ep["r"] for ep in ep_buffer]))
                      if len(ep_buffer) else float("nan"))

        row = {"timesteps": int(self.num_timesteps), "sb3_ep_rew_mean": sb3_return}
        row.update(means)
        row.update({f"{k}_std": v for k, v in stds.items()})
        self.history.append({"means": means, "stds": stds, "timesteps": self.num_timesteps})
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fields)
            writer.writerow(row)

        self.model.save(os.path.join(self.save_dir, "latest"))

        saved = []
        if is_better_arrival(means, self.best_arrival):
            self.best_arrival = dict(means)
            self.best_arrival_step = int(self.num_timesteps)
            self.model.save(os.path.join(self.save_dir, "best_arrival"))
            saved.append("best_arrival")
        if is_better_safety(means, self.best_safety, self.min_arrival):
            self.best_safety = dict(means)
            self.best_safety_step = int(self.num_timesteps)
            self.model.save(os.path.join(self.save_dir, "best_safety"))
            saved.append("best_safety")

        # selected = safety winner if we have one, else arrival winner
        selected_name = "best_safety" if self.best_safety is not None else "best_arrival"
        selected_src = os.path.join(self.save_dir, selected_name)
        if os.path.exists(selected_src + ".zip"):
            self.model.save(os.path.join(self.save_dir, "selected"))

        self._write_selection()

        if self.verbose:
            vs_neutral = intrusion - NEUTRAL_BASELINE["intrusion_time"]
            print(
                f"\n[eval t={self.num_timesteps}] "
                f"arrival={100 * arrival:.1f}%  "
                f"intrusion_time={intrusion:.1f}s "
                f"(neutral {NEUTRAL_BASELINE['intrusion_time']:.1f}s, "
                f"Δ={vs_neutral:+.1f})  "
                f"restricted_time={means['time_in_restricted_area']:.1f}s  "
                f"sb3_return={sb3_return}  "
                f"saved={saved or 'none'}"
            )
            print("  (checkpoint rule: arrival + intrusion_time; sb3_return is ignored)\n")

    def _write_selection(self):
        payload = {
            "min_arrival": self.min_arrival,
            "eval_seed": self.eval_seed,
            "eval_episodes": self.eval_episodes,
            "note": "selected uses official metrics only; sb3_ep_rew_mean is not used",
            "neutral_baseline": NEUTRAL_BASELINE,
            "best_arrival_step": getattr(self, "best_arrival_step", None),
            "best_arrival": self.best_arrival,
            "best_safety_step": getattr(self, "best_safety_step", None),
            "best_safety": self.best_safety,
            "selected": "best_safety" if self.best_safety is not None else "best_arrival",
        }
        with open(self.selection_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def _on_training_end(self):
        # Final eval at the last timestep if we didn't just eval.
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self._run_eval()
        if self._eval_env is not None:
            # Do not call env.close() — BlueSky quit() poisons the process.
            self._eval_env = None
