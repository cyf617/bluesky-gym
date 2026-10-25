import numpy as np

from competitor.shield import ShieldGeometry, candidate_offsets, choose_heading

# Large square sector (north, east) in NM; ownship at the origin flying north at 150 m/s
# (~0.081 NM/s, ~9.7 NM in 120 s).
SECTOR = [(-100, -100), (-100, 100), (100, 100), (100, -100)]
SPEED = 150.0


def geom(obstacles=()):
    return ShieldGeometry(list(obstacles), SECTOR)


def test_offsets_are_ordered_by_distance_from_policy():
    offs = candidate_offsets()
    assert offs[0] == 0 and len(offs) == 37
    assert list(offs[:5]) == [0, 5, -5, 10, -10]
    assert np.all(np.diff(np.abs(offs[1::2])) >= 0)


def test_clear_airspace_keeps_policy_heading():
    hdg, intervened, margins = choose_heading(0.0, (0.0, 0.0), SPEED, [], [], geom())
    assert hdg == 0.0 and not intervened
    assert margins[0] > 0


def test_obstacle_ahead_deflects_to_nearest_safe_heading():
    # Thin wall 4-6 NM north spanning east -1.5..1.5 NM.
    wall = [(4, -1.5), (4, 1.5), (6, 1.5), (6, -1.5)]
    hdg, intervened, margins = choose_heading(0.0, (0.0, 0.0), SPEED, [], [], geom([wall]))
    assert intervened
    assert margins[0] < 0
    # Nearest safe candidate: everything closer to 0 deg is unsafe.
    offs = candidate_offsets()
    first_safe = int(np.flatnonzero(margins >= 0)[0])
    assert np.all(margins[:first_safe] < 0)
    assert np.isclose(hdg, offs[first_safe])
    assert 5 <= abs(hdg) <= 90


def test_head_on_intruder_triggers_avoidance():
    # Intruder 15 NM north flying south at the same speed: closes within 6 NM in < 120 s.
    v = SPEED / 1852.0
    hdg, intervened, _ = choose_heading(0.0, (0.0, 0.0), SPEED, [(15.0, 0.0)], [(-v, 0.0)], geom())
    assert intervened and abs(hdg) >= 5


def test_diverging_intruder_is_ignored():
    v = SPEED / 1852.0
    _, intervened, _ = choose_heading(0.0, (0.0, 0.0), SPEED, [(-15.0, 0.0)], [(-v, 0.0)], geom())
    assert not intervened


def test_no_safe_candidate_picks_max_margin():
    # Ownship already outside the sector: every candidate violates the sector buffer,
    # so the heading that gets back inside fastest (west, -90 deg) has the largest margin.
    sector = [(-50, -50), (-50, 50), (50, 50), (50, -50)]
    g = ShieldGeometry([], sector)
    hdg, intervened, margins = choose_heading(0.0, (0.0, 55.0), SPEED, [], [], g)
    assert np.all(margins < 0)
    assert intervened
    assert np.isclose(hdg, -90.0)


def test_sector_edge_ahead_is_avoided():
    sector = [(-50, -50), (-50, 50), (5, 50), (5, -50)]  # edge 5 NM north
    hdg, intervened, _ = choose_heading(0.0, (0.0, 0.0), SPEED, [], [], ShieldGeometry([], sector))
    assert intervened


def test_turn_model_matches_coordinated_turn_rate():
    from competitor.shield import trajectories, turn_rate_dps
    omega = turn_rate_dps(150.0, 25.0)
    assert 1.6 < omega < 1.9  # deg/s, BlueSky A320 standard bank
    times = np.array([10.0, 120.0])
    n, e = trajectories((0.0, 0.0), 0.0, [0.0, 90.0], 150.0, times, bank_deg=25.0)
    # Straight ahead is unchanged; a 90 deg right turn is still mostly north after 10 s.
    assert np.isclose(n[0, 0], 150 / 1852 * 10) and np.isclose(e[0, 0], 0.0)
    assert n[1, 0] > 3 * e[1, 0] > 0
    # Straight-line model would have it due east at once.
    n0, e0 = trajectories((0.0, 0.0), 0.0, [90.0], 150.0, times, bank_deg=0.0)
    assert np.isclose(n0[0, 0], 0.0, atol=1e-9) and e0[0, 0] > 0


def test_turn_model_sees_obstacle_in_the_turn_arc():
    # Obstacle just ahead (north); the policy wants due east. Straight-line extrapolation
    # from the current position says east is clear, but the turn arc crosses the obstacle.
    block = [(1.0, -1.0), (1.0, 3.0), (3.0, 3.0), (3.0, -1.0)]
    g = geom([block])
    _, straight = choose_heading(90.0, (0.0, 0.0), SPEED, [], [], g, own_hdg=0.0,
                                 obstacle_buffer_km=0.5)[1:]
    _, turned = choose_heading(90.0, (0.0, 0.0), SPEED, [], [], g, own_hdg=0.0,
                               obstacle_buffer_km=0.5, bank_deg=25.0)[1:]
    assert straight[0] >= 0 > turned[0]
