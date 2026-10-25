"""Read training run directories for the dashboard.

A run directory is any child of the results directory that contains at least
one of the files written by ``scripts/train_sa_sac.py``. Older runs that only
have SB3 stdout (``train.log``) are parsed from the printed tables.

Nothing here imports BlueSky, torch or SB3, so the dashboard starts instantly
and can run on a machine that only has the copied result folders.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import time

RUN_MARKERS = ("config.json", "eval_log.csv", "progress.csv", "episodes.csv", "train.log")
MAX_POINTS = 2000
# Older telemetry wrote the heartbeat from the training loop only, so it went
# quiet during evals; the threaded heartbeat records ``heartbeat_s``.
LEGACY_STALE_S = 900.0


def _num(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return value
    return out if math.isfinite(out) else None


def read_csv_columns(path):
    """CSV -> {column: [values]} with numbers parsed and blanks as None."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {}
        cols = {h: [] for h in header}
        for row in reader:
            if len(row) < len(header):
                # Partially flushed last line while training is writing.
                continue
            for h, v in zip(header, row):
                cols[h].append(_num(v))
    return cols


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def downsample_columns(cols, max_points=MAX_POINTS):
    n = len(next(iter(cols.values()), []))
    if n <= max_points:
        return cols
    step = math.ceil(n / max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return {k: [v[i] for i in idx] for k, v in cols.items()}


def parse_sb3_stdout(path):
    """Parse SB3 ``verbose=1`` tables from a log into progress.csv-style columns."""
    if not os.path.exists(path):
        return {}
    rows, current, section = [], {}, ""
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("---"):
                if current:
                    rows.append(current)
                current, section = {}, ""
                continue
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) != 2:
                continue
            key, value = parts
            if key.endswith("/") and value == "":
                section = key
                continue
            parsed = _num(value)
            if isinstance(parsed, float):
                current[section + key] = parsed
    if current:
        rows.append(current)
    rows = [r for r in rows if "time/total_timesteps" in r]
    keys = sorted({k for r in rows for k in r})
    return {k: [r.get(k) for r in rows] for k in keys}


def discover_runs(results_dir):
    if not os.path.isdir(results_dir):
        return []
    names = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path) and any(os.path.exists(os.path.join(path, m)) for m in RUN_MARKERS):
            names.append(name)
    return names


def _last_modified(run_dir):
    latest = 0.0
    for name in os.listdir(run_dir):
        try:
            latest = max(latest, os.path.getmtime(os.path.join(run_dir, name)))
        except OSError:
            pass
    return latest


def run_status(run_dir, now=None):
    """(state, status dict). state ∈ running / finished / stopped."""
    now = time.time() if now is None else now
    status = read_json(os.path.join(run_dir, "status.json"))
    if status:
        if status.get("state") in ("finished", "early_stopped", "stopped"):
            return status["state"], status
        stale_after = 6 * status["heartbeat_s"] if "heartbeat_s" in status else LEGACY_STALE_S
        if now - status.get("updated_at", 0) > stale_after:
            return "stopped", status
        return "running", status
    # No status.json: written by the pre-telemetry script, which only left
    # train.log behind (file mtimes are unreliable after a git checkout).
    log = os.path.join(run_dir, "train.log")
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            if "training finished" in f.read():
                return "finished", None
    return "stopped", None


def _progress(run_dir):
    cols = read_csv_columns(os.path.join(run_dir, "progress.csv"))
    if cols:
        return cols, "progress.csv"
    cols = parse_sb3_stdout(os.path.join(run_dir, "train.log"))
    return cols, ("train.log" if cols else None)


def _last_row(cols):
    n = len(next(iter(cols.values()), []))
    if not n:
        return None
    return {k: v[n - 1] for k, v in cols.items()}


def _infer_algo(name, progress):
    if any(k.startswith("train/actor_loss") for k in progress):
        return "sac"
    if any(k.startswith("train/approx_kl") for k in progress):
        return "ppo"
    for algo in ("sac", "ppo"):
        if algo in name.lower():
            return algo
    return None


def run_summary(results_dir, name, now=None):
    run_dir = os.path.join(results_dir, name)
    config = read_json(os.path.join(run_dir, "config.json")) or {}
    progress, _ = _progress(run_dir)
    evals = read_csv_columns(os.path.join(run_dir, "eval_log.csv"))
    selection = read_json(os.path.join(run_dir, "selection.json"))
    state, status = run_status(run_dir, now)

    timesteps = (status or {}).get("timesteps")
    if timesteps is None and progress.get("time/total_timesteps"):
        timesteps = progress["time/total_timesteps"][-1]
    total = (status or {}).get("total_timesteps") or config.get("timesteps")
    return {
        "name": name,
        "algo": config.get("algo") or _infer_algo(name, progress),
        "action_mode": (config.get("mdp") or {}).get("ACTION_MODE"),
        "mdp_overrides": config.get("mdp_overrides"),
        "state": state,
        "timesteps": timesteps,
        "total_timesteps": total,
        "fps": (status or {}).get("fps"),
        "updated_at": (status or {}).get("updated_at") or _last_modified(run_dir),
        "has_config": bool(config),
        "last_eval": _last_row(evals),
        "n_evals": len(evals.get("timesteps", [])),
        "eval_seed": (config.get("eval") or {}).get("seed", (selection or {}).get("eval_seed")),
        "eval_episodes": (config.get("eval") or {}).get("episodes", (selection or {}).get("eval_episodes")),
        "selected": (selection or {}).get("selected"),
        "n_replays": len((read_json(os.path.join(run_dir, "replays", "index.json")) or {}).get("episodes", [])),
        "best_safety_step": (selection or {}).get("best_safety_step"),
        "best_arrival_step": (selection or {}).get("best_arrival_step"),
    }


def run_detail(results_dir, name, now=None):
    run_dir = os.path.join(results_dir, name)
    progress, progress_source = _progress(run_dir)
    return {
        "summary": run_summary(results_dir, name, now),
        "config": read_json(os.path.join(run_dir, "config.json")),
        "status": read_json(os.path.join(run_dir, "status.json")),
        "selection": read_json(os.path.join(run_dir, "selection.json")),
        "progress": downsample_columns(progress),
        "progress_source": progress_source,
        "evals": read_csv_columns(os.path.join(run_dir, "eval_log.csv")),
        "episodes": downsample_columns(read_csv_columns(os.path.join(run_dir, "episodes.csv"))),
        "official": official_results(results_dir, name),
    }


OFFICIAL_KEYS = ("waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
                 "restricted_area_events", "time_in_restricted_area",
                 "sector_exit_events", "time_outside_sector")
PAIRED_KEYS = OFFICIAL_KEYS
SAFETY_TIME_KEYS = ("intrusion_time", "time_in_restricted_area", "time_outside_sector")


def _official_table(path, neutral=None):
    cols = read_csv_columns(path)
    n = len(cols.get("episode_index", []))
    if not n:
        return None
    out = {"file": os.path.basename(path), "episodes": n, "means": {}, "stds": {}}
    for k in OFFICIAL_KEYS:
        vals = [v for v in cols.get(k, []) if isinstance(v, float)]
        if vals:
            mean = sum(vals) / len(vals)
            out["means"][k] = mean
            out["stds"][k] = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    clean = [all(cols[k][i] == 0 for k in SAFETY_TIME_KEYS) for i in range(n)]
    out["clean_frac"] = sum(clean) / n
    si = [v for v in cols.get("shield_interventions", []) if isinstance(v, float)]
    if si:
        out["shield_interventions"] = sum(si) / len(si)
    if neutral and len(neutral.get("episode_index", [])) == n:
        out["paired"] = {}
        for k in PAIRED_KEYS:
            a, b = cols.get(k, []), neutral.get(k, [])
            diffs = [x - y for x, y in zip(a, b) if isinstance(x, float) and isinstance(y, float)]
            if not diffs:
                continue
            mean = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
            out["paired"][k] = {
                "better": sum(d < 0 for d in diffs), "worse": sum(d > 0 for d in diffs),
                "mean_diff": mean, "se": sd / math.sqrt(len(diffs)),
            }
    return out


def official_results(results_dir, name):
    """Official-protocol (seed 42) CSVs from evaluate_competition.py, paired with neutral.

    The neutral reference is ``<results>/official_seed42_<N>_neutral.csv``; the
    same seed gives the same scenario sequence, so episode i is compared with
    episode i (paired difference, standard error over episodes).
    """
    run_dir = os.path.join(results_dir, name)
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "official_seed42_*.csv"))):
        n_eps = os.path.basename(path).split("_")[2]
        neutral = read_csv_columns(os.path.join(results_dir, f"official_seed42_{n_eps}_neutral.csv"))
        table = _official_table(path, neutral)
        if table:
            rows.append(table)
    neutral_tables = [_official_table(p) for p in
                      sorted(glob.glob(os.path.join(results_dir, "official_seed42_*_neutral.csv")))]
    return {"runs": rows, "neutral": [t for t in neutral_tables if t]}


def replay_index(results_dir, name):
    """``replays/index.json`` written by ``competitor.replay``, or an empty index."""
    index = read_json(os.path.join(results_dir, name, "replays", "index.json"))
    return index or {"episodes": []}


def load_baselines(results_dir):
    """{"<seed>:<episodes>": baseline} for every ``baseline_sa_neutral*.json``.

    Checkpoint evals run on a validation seed and episode count, and the mean
    over the first 20 vs 50 scenarios differs a lot, so each run is compared
    with the neutral policy on exactly the same scenarios
    (``scripts.run_neutral_baseline --seed S --episodes N``).
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "baseline_sa_neutral*.json"))):
        baseline = load_baseline(path)
        if baseline and baseline["seed"] is not None:
            out.setdefault(f"{baseline['seed']}:{baseline['episodes']}", baseline)
    return out


def load_baseline(path):
    """Neutral [0, 0] policy means keyed by metric, or None."""
    data = read_json(path)
    if not data:
        return None
    means = {m["key"]: m["mean"] for m in data.get("metrics", [])}
    stds = {m["key"]: m["std"] for m in data.get("metrics", [])}
    # The baseline table stores goal completion in percent; eval_log uses 0..1.
    for table in (means, stds):
        if "waypoint_reached" in table:
            table["waypoint_reached"] /= 100.0
    return {
        "seed": data.get("seed"),
        "episodes": data.get("episodes"),
        "label": f"{data.get('policy', 'neutral')} · seed {data.get('seed')} · "
                 f"{data.get('episodes')} eps",
        "means": means,
        "stds": stds,
    }
