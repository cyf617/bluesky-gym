"""Heading shield: replace an unsafe policy heading with the nearest safe one.

Pure geometry (no BlueSky) so it can be unit-tested. Frame: (north, east) in
nautical miles around the sector centre, the same frame as
``competitor.geometry.latlon_to_nm``; headings are true degrees.

A candidate heading is extrapolated at the current speed and sampled every
``step_s`` seconds up to ``lookahead_s`` (t = 0 excluded, the current position
cannot be changed by the action). With ``bank_deg = 0`` the candidate is flown
in a straight line from the current position at once. With ``bank_deg > 0``
the aircraft first turns from its current heading at the coordinated-turn rate
g*tan(bank)/v (the shortest way, as BlueSky's HDG command does) and then flies
straight; without this, large deflections are predicted as instantaneous and
the aircraft can circle into the obstacle it was meant to avoid. It is unsafe if any
sample is closer than the buffer to a restricted area (or inside it), closer
than the buffer to the sector edge (or outside it), or closer than
``intruder_nm`` to any other aircraft extrapolated at constant velocity.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon

NM2KM = 1.852
G = 9.80665


def candidate_offsets(step_deg=5.0, max_deg=90.0):
    """0, +5, -5, +10, -10, ... so the first safe entry is the nearest one."""
    out = [0.0]
    k = 1
    while k * step_deg <= max_deg + 1e-9:
        out += [k * step_deg, -k * step_deg]
        k += 1
    return np.array(out)


def bound_180(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


class ShieldGeometry:
    """Restricted-area and sector polygons in the (north, east) NM frame."""

    def __init__(self, obstacles_nm, sector_nm):
        self.obstacles = [Polygon(v) for v in obstacles_nm]
        self.sector = Polygon(sector_nm)
        for poly in [*self.obstacles, self.sector]:
            shapely.prepare(poly)


def _signed_boundary_km(poly, north, east, inside_sign):
    """Distance to the polygon boundary in km, multiplied by ``inside_sign`` inside."""
    pts = shapely.points(north, east)
    d = shapely.distance(poly.exterior, pts) * NM2KM
    inside = shapely.contains_xy(poly, north, east)
    return np.where(inside, inside_sign * d, -inside_sign * d)


def turn_rate_dps(speed_mps, bank_deg):
    if bank_deg <= 0 or speed_mps <= 0:
        return 0.0
    return float(np.degrees(G * np.tan(np.radians(bank_deg)) / speed_mps))


def trajectories(own_ne, own_hdg, target_hdgs, speed_mps, times, bank_deg=0.0, dt=1.0):
    """(north, east) arrays of shape (C, T) for each target heading at ``times``."""
    target_hdgs = np.asarray(target_hdgs, dtype=float)
    v_nm_s = speed_mps / 1852.0
    omega = turn_rate_dps(speed_mps, bank_deg)
    if omega == 0.0:
        rad = np.deg2rad(target_hdgs)
        dist = v_nm_s * times
        return (own_ne[0] + np.cos(rad)[:, None] * dist[None, :],
                own_ne[1] + np.sin(rad)[:, None] * dist[None, :])
    heading = np.full(target_hdgs.shape, float(own_hdg))
    remaining = bound_180(target_hdgs - own_hdg)
    n = np.full(target_hdgs.shape, float(own_ne[0]))
    e = np.full(target_hdgs.shape, float(own_ne[1]))
    sample_steps = {int(round(t / dt)): j for j, t in enumerate(times)}
    out_n = np.empty((target_hdgs.size, times.size))
    out_e = np.empty_like(out_n)
    for k in range(1, max(sample_steps) + 1):
        turn = np.clip(remaining, -omega * dt, omega * dt)
        remaining = remaining - turn
        # Midpoint heading over the substep.
        mid = np.deg2rad(heading + 0.5 * turn)
        heading = heading + turn
        n = n + np.cos(mid) * v_nm_s * dt
        e = e + np.sin(mid) * v_nm_s * dt
        if k in sample_steps:
            out_n[:, sample_steps[k]] = n
            out_e[:, sample_steps[k]] = e
    return out_n, out_e


def candidate_margins(policy_hdg, own_ne, own_speed_mps, others_ne, others_vel_ne, geom,
                      lookahead_s=120.0, obstacle_buffer_km=2.0, sector_buffer_km=2.0,
                      intruder_nm=6.0, step_s=10.0, offsets=None, own_hdg=None, bank_deg=0.0):
    """(headings, margins_km): per-candidate worst margin; >= 0 means safe."""
    offsets = candidate_offsets() if offsets is None else np.asarray(offsets)
    hdgs = bound_180(policy_hdg + offsets)
    times = np.arange(step_s, lookahead_s + 1e-9, step_s)
    if times.size == 0:
        times = np.array([step_s])
    north, east = trajectories(own_ne, policy_hdg if own_hdg is None else own_hdg, hdgs,
                               own_speed_mps, times, bank_deg=bank_deg)
    n_flat, e_flat = north.ravel(), east.ravel()

    margin = np.full(north.shape, np.inf)
    for poly in geom.obstacles:
        # Inside an obstacle counts as negative clearance.
        clearance = _signed_boundary_km(poly, n_flat, e_flat, inside_sign=-1.0)
        margin = np.minimum(margin, clearance.reshape(north.shape) - obstacle_buffer_km)
    # Inside the sector counts as positive clearance.
    clearance = _signed_boundary_km(geom.sector, n_flat, e_flat, inside_sign=1.0)
    margin = np.minimum(margin, clearance.reshape(north.shape) - sector_buffer_km)

    others_ne = np.asarray(others_ne, dtype=float).reshape(-1, 2)
    if len(others_ne):
        vel = np.asarray(others_vel_ne, dtype=float).reshape(-1, 2)
        o_n = others_ne[:, 0, None] + vel[:, 0, None] * times[None, :]       # (N, T)
        o_e = others_ne[:, 1, None] + vel[:, 1, None] * times[None, :]
        d = np.hypot(north[:, None, :] - o_n[None], east[:, None, :] - o_e[None])  # (C, N, T)
        margin = np.minimum(margin, (d.min(axis=1) - intruder_nm) * NM2KM)
    return hdgs, margin.min(axis=1)


def choose_heading(policy_hdg, own_ne, own_speed_mps, others_ne, others_vel_ne, geom, **kw):
    """(heading, intervened, margins). Nearest safe candidate, else the max-margin one."""
    hdgs, margins = candidate_margins(policy_hdg, own_ne, own_speed_mps, others_ne,
                                      others_vel_ne, geom, **kw)
    safe = np.flatnonzero(margins >= 0.0)
    idx = int(safe[0]) if safe.size else int(np.argmax(margins))
    return float(hdgs[idx]), idx != 0, margins
