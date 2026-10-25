"""Training telemetry for the dashboard (``python -m dashboard``).

Writes into the run directory:

- ``episodes.csv``  one row per finished *training* episode: SB3 return and
  length, the official metrics from ``info``, and the per-term reward sums
  from ``competitor.vec_env.RewardTermsInfo``.
- ``status.json``   heartbeat (state, timesteps, fps, wall time), refreshed by
  a daemon thread so it stays fresh during long evals. A run whose heartbeat
  is stale while ``state == "running"`` was killed.

It also records ``terms/<name>`` and ``train_metrics/<name>`` means into the
SB3 logger so they land in ``progress.csv`` next to the losses. With a
VecNormalize reward it counts how often the normalised reward sits on the
``clip_reward`` bound (``train/reward_clip_frac``, totals in status.json).
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from competitor.mdp import REWARD_TERMS
from competitor.metric_eval import EXTRA_KEYS, METRIC_KEYS

EPISODE_FIELDS = (
    ["timesteps", "wall_time", "env_index", "ep_return", "ep_length"]
    + METRIC_KEYS + EXTRA_KEYS
    + [f"r_{k}" for k in REWARD_TERMS]
)


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


class TrainingTelemetryCallback(BaseCallback):
    def __init__(self, save_dir, total_timesteps, heartbeat_s=10.0, stop_rule=None, verbose=0):
        super().__init__(verbose)
        self.stop_rule = stop_rule
        self.save_dir = save_dir
        self.total_timesteps = int(total_timesteps)
        self.heartbeat_s = float(heartbeat_s)
        self.episodes_path = os.path.join(save_dir, "episodes.csv")
        self.status_path = os.path.join(save_dir, "status.json")
        self._file = None
        self._writer = None
        self._started = None
        self._n_episodes = 0
        self._clip = None
        self._clip_counts = {"clipped": 0, "total": 0}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        new_file = not os.path.exists(self.episodes_path)
        self._file = open(self.episodes_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_FIELDS)
        if new_file:
            self._writer.writeheader()
        self._started = time.time()
        env = self.training_env
        if isinstance(env, VecNormalize) and env.norm_reward:
            self._clip = float(env.clip_reward)
        self._write_status("running")
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self):
        while not self._stop.wait(self.heartbeat_s):
            self._write_status("running")

    def _on_step(self):
        if self._clip is not None and "rewards" in self.locals:
            r = np.abs(np.asarray(self.locals["rewards"], dtype=float))
            hit = int(np.sum(r >= self._clip * (1 - 1e-6)))
            self._clip_counts["clipped"] += hit
            self._clip_counts["total"] += r.size
            self.logger.record_mean("train/reward_clip_frac", hit / max(r.size, 1))
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
            row.update({k: info.get(k, "") for k in METRIC_KEYS + EXTRA_KEYS})
            row.update({f"r_{k}": terms.get(k, 0.0) for k in REWARD_TERMS})
            self._writer.writerow(row)
            self._n_episodes += 1
            for k in REWARD_TERMS:
                self.logger.record_mean(f"terms/{k}", float(terms.get(k, 0.0)))
            for k in METRIC_KEYS + EXTRA_KEYS:
                if k != "total_reward" and k in info:
                    self.logger.record_mean(f"train_metrics/{k}", float(info[k]))
            self._file.flush()
        return True

    def _on_training_end(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        stopped = self.stop_rule is not None and self.stop_rule.reason
        self._write_status("early_stopped" if stopped else "finished")
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write_status(self, state):
        with self._lock:
            now = time.time()
            elapsed = max(now - self._started, 1e-9)
            write_json_atomic(self.status_path, {
                "state": state,
                "timesteps": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "episodes": self._n_episodes,
                "fps": round(self.num_timesteps / elapsed, 1),
                "started_at": self._started,
                "updated_at": now,
                "heartbeat_s": self.heartbeat_s,
                "reward_clip": dict(self._clip_counts, bound=self._clip) if self._clip else None,
                "stop_reason": self.stop_rule.reason if self.stop_rule is not None else None,
            })
