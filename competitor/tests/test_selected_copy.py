import os

from competitor.callbacks import OfficialMetricCheckpointCallback


class _FakeModel:
    def __init__(self):
        self.version = 0

    def save(self, path):
        with open(path + ".zip", "w") as f:
            f.write(f"v{self.version}")


def test_selected_is_a_copy_of_the_best_checkpoint(tmp_path, monkeypatch):
    cb = OfficialMetricCheckpointCallback(str(tmp_path), eval_freq=1, eval_episodes=1,
                                          eval_seed=0, min_arrival=0.5, verbose=0)
    cb.model = _FakeModel()
    cb.num_timesteps = 0
    evals = iter([
        {"waypoint_reached": 1.0, "intrusion_time": 10.0},   # v0: best safety
        {"waypoint_reached": 1.0, "intrusion_time": 50.0},   # v1: worse, must not become selected
    ])

    def fake_eval(env, act, n, seed):
        return [dict(next(evals), flight_time=0, intrusion_events=0, restricted_area_events=0,
                     time_in_restricted_area=0, sector_exit_events=0, time_outside_sector=0,
                     total_reward=0)]

    monkeypatch.setattr("competitor.callbacks.run_metric_episodes", fake_eval)
    monkeypatch.setattr(cb, "_get_eval_env", lambda: None)
    cb._init_callback()
    for version in (0, 1):
        cb.model.version = version
        cb.num_timesteps = version
        cb._run_eval()
    with open(os.path.join(tmp_path, "selected.zip")) as f:
        assert f.read() == "v0"
    with open(os.path.join(tmp_path, "latest.zip")) as f:
        assert f.read() == "v1"
