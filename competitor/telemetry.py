"""Training telemetry for the dashboard (``python -m dashboard``).

Writes into the run directory:

- ``episodes.csv``  one row per finished *training* episode: SB3 return and
  length, the official metrics from ``info``, and the per-term reward sums
  from ``competitor.vec_env.RewardTermsInfo``.
- ``status.json``   heartbeat (state, timesteps, fps, wall time). A run whose
  heartbeat is stale while ``state == "running"`` was killed.

It also records ``terms/<name>`` and ``train_metrics/<name>`` means into the
SB3 logger so they land in ``progress.csv`` next to the losses.
"""

from __future__ import annotations

import csv
import json
import os
import time

from stable_baselines3.common.callbacks import BaseCallback

from competitor.mdp import REWARD_TERMS
from competitor.metric_eval import METRIC_KEYS

EPISODE_FIELDS = (
    ["timesteps", "wall_time", "env_index", "ep_return", "ep_length"]
    + METRIC_KEYS
    + [f"r_{k}" for k in REWARD_TERMS]
)


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


class TrainingTelemetryCallback(BaseCallback):
    def __init__(self, save_dir, total_timesteps, heartbeat_s=10.0, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.total_timesteps = int(total_timesteps)
        self.heartbeat_s = float(heartbeat_s)
        self.episodes_path = os.path.join(save_dir, "episodes.csv")
        self.status_path = os.path.join(save_dir, "status.json")
        self._file = None
        self._writer = None
        self._started = None
        self._last_beat = 0.0
        self._n_episodes = 0

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        new_file = not os.path.exists(self.episodes_path)
        self._file = open(self.episodes_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_FIELDS)
        if new_file:
            self._writer.writeheader()
        self._started = time.time()
        self._write_status("running")

    def _on_step(self):
        for env_index, info in enumerate(self.locals.get("infos", ())):
            terms = info.get("reward_terms")
            if terms is None:
                continue
            episode = info.get("episode", {})
            row = {
                "timesteps": int(self.num_timesteps),
                "wall_time": round(time.time() - self._started, 2),
                "env_index": env_index,
                "ep_return": episode.get("r", ""),
                "ep_length": episode.get("l", ""),
            }
            row.update({k: info.get(k, "") for k in METRIC_KEYS})
            row.update({f"r_{k}": terms.get(k, 0.0) for k in REWARD_TERMS})
            self._writer.writerow(row)
            self._n_episodes += 1
            for k in REWARD_TERMS:
                self.logger.record_mean(f"terms/{k}", float(terms.get(k, 0.0)))
            for k in METRIC_KEYS:
                if k != "total_reward" and k in info:
                    self.logger.record_mean(f"train_metrics/{k}", float(info[k]))
            self._file.flush()

        now = time.time()
        if now - self._last_beat >= self.heartbeat_s:
            self._write_status("running")
        return True

    def _on_training_end(self):
        self._write_status("finished")
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write_status(self, state):
        now = time.time()
        self._last_beat = now
        elapsed = max(now - self._started, 1e-9)
        write_json_atomic(self.status_path, {
            "state": state,
            "timesteps": int(self.num_timesteps),
            "total_timesteps": self.total_timesteps,
            "episodes": self._n_episodes,
            "fps": round(self.num_timesteps / elapsed, 1),
            "started_at": self._started,
            "updated_at": now,
        })
