import numpy as np

from scripts.compare_official import compare

KEYS = ["waypoint_reached", "flight_time", "intrusion_events", "intrusion_time",
        "restricted_area_events", "time_in_restricted_area", "sector_exit_events",
        "time_outside_sector"]


def cols(**over):
    base = {k: np.zeros(4) for k in KEYS}
    base["waypoint_reached"] = np.ones(4)
    base.update({k: np.asarray(v, dtype=float) for k, v in over.items()})
    return base


def test_paired_difference_ci_and_clean_fraction():
    neutral = cols(intrusion_time=[10, 0, 20, 30], flight_time=[800, 800, 800, 800])
    model = cols(intrusion_time=[0, 0, 10, 10], flight_time=[810, 820, 830, 840],
                 shield_interventions=[1, 2, 3, 4])
    out = compare(neutral, {"m": model})
    r = out["runs"]["m"]
    assert np.isclose(r["diff"]["intrusion_time"], -10.0)
    d = np.array([-10, 0, -10, -20])
    assert np.isclose(r["ci95"]["intrusion_time"], 1.96 * d.std(ddof=1) / 2)
    assert np.isclose(r["diff"]["flight_time"], 25.0)
    assert r["clean_frac"] == 0.5 and out["neutral"]["clean_frac"] == 0.25
    assert r["shield_interventions"] == 2.5
