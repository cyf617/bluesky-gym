import json
import os

from dashboard import data

SB3_LOG = """\
Step 2 train: algo=sac
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 574      |
|    ep_rew_mean     | -70.5    |
| time/              |          |
|    total_timesteps | 66076    |
| train/             |          |
|    actor_loss      | -0.412   |
---------------------------------
[eval t=50004] arrival=0.0%
---------------------------------
| rollout/           |          |
|    ep_rew_mean     | -69.8    |
| time/              |          |
|    total_timesteps | 68204    |
---------------------------------
"""


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_parse_sb3_stdout(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(SB3_LOG)
    cols = data.parse_sb3_stdout(str(log))
    assert cols["time/total_timesteps"] == [66076.0, 68204.0]
    assert cols["rollout/ep_rew_mean"] == [-70.5, -69.8]
    assert cols["train/actor_loss"] == [-0.412, None]
    assert cols["rollout/ep_len_mean"] == [574.0, None]


def test_read_csv_skips_partial_last_line_and_nan(tmp_path):
    p = tmp_path / "eval_log.csv"
    p.write_text("timesteps,sb3_ep_rew_mean,waypoint_reached\n4,nan,0.0\n25004,1.5,0.5\n50004,")
    cols = data.read_csv_columns(str(p))
    assert cols["timesteps"] == [4.0, 25004.0]
    assert cols["sb3_ep_rew_mean"] == [None, 1.5]


def test_status_inference(tmp_path):
    now = 10_000.0
    fresh = tmp_path / "fresh"
    write(str(fresh / "status.json"), json.dumps(
        {"state": "running", "updated_at": now - 20, "heartbeat_s": 10}))
    assert data.run_status(str(fresh), now)[0] == "running"

    dead = tmp_path / "dead"
    write(str(dead / "status.json"), json.dumps(
        {"state": "running", "updated_at": now - 120, "heartbeat_s": 10}))
    assert data.run_status(str(dead), now)[0] == "stopped"

    done = tmp_path / "done"
    write(str(done / "status.json"), json.dumps({"state": "finished", "updated_at": 0}))
    assert data.run_status(str(done), now)[0] == "finished"

    legacy = tmp_path / "legacy"
    write(str(legacy / "train.log"), SB3_LOG)
    assert data.run_status(str(legacy), now)[0] == "stopped"
    write(str(legacy / "train.log"), SB3_LOG + "training finished. selected checkpoint\n")
    assert data.run_status(str(legacy), now)[0] == "finished"


def test_discover_and_summary(tmp_path):
    results = tmp_path / "results"
    write(str(results / "old_run" / "train.log"), SB3_LOG)
    write(str(results / "old_run" / "eval_log.csv"),
          "timesteps,waypoint_reached,intrusion_time\n4,0.0,157.2\n50004,0.0,61.8\n")
    write(str(results / "not_a_run" / "notes.txt"), "x")
    write(str(results / "baseline_sa_neutral_20.json"), json.dumps({
        "policy": "neutral", "seed": 42, "episodes": 20,
        "metrics": [{"key": "waypoint_reached", "mean": 100.0, "std": 0.0},
                    {"key": "intrusion_time", "mean": 77.1, "std": 94.7}]}))

    assert data.discover_runs(str(results)) == ["old_run"]
    summary = data.run_summary(str(results), "old_run")
    assert summary["algo"] == "sac"
    assert summary["timesteps"] == 68204.0
    assert summary["last_eval"]["intrusion_time"] == 61.8
    write(str(results / "baseline_sa_neutral_seed123_20.json"), json.dumps({
        "policy": "neutral", "seed": 123, "episodes": 20,
        "metrics": [{"key": "intrusion_time", "mean": 64.75, "std": 88.5}]}))
    baselines = data.load_baselines(str(results))
    assert set(baselines) == {"42", "123"}
    assert baselines["42"]["means"]["waypoint_reached"] == 1.0
    assert baselines["123"]["means"]["intrusion_time"] == 64.75


def test_downsample_keeps_last_point():
    cols = {"x": list(range(5000)), "y": list(range(5000))}
    out = data.downsample_columns(cols, max_points=1000)
    assert len(out["x"]) <= 1001
    assert out["x"][-1] == 4999
