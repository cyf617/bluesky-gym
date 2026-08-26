"""Observation building blocks for competition MDP v1."""

from __future__ import annotations

import numpy as np
import bluesky as bs
from gymnasium import spaces

from core.observations import (
    NM2KM,
    _bound_angle,
    _cpa,
    _sort_by_tcpa,
)
from competitor.config import (
    DCPA_NORM_NM,
    INTRUSION_NM,
    NEIGHBOR_DIST_NORM_KM,
    NEIGHBOR_POS_NORM_M,
    NEIGHBOR_SPD_NORM_MPS,
    POLYGON_DISTANCE_NORM_KM,
    TCPA_NORM_S,
)
from competitor.geometry import (
    bound_angle_deg,
    latlon_to_nm,
    polygon_clearance,
    vertices_to_nm,
)


class OwnStatusObservation:
    """Three binary flags: currently in conflict / restricted area / outside sector."""

    def space(self):
        return {
            "in_conflict": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
            "in_restricted": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
            "outside_sector": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float64),
        }

    def observe(self, in_conflict, in_restricted, outside_sector):
        return {
            "in_conflict": np.array([1.0 if in_conflict else 0.0], dtype=np.float64),
            "in_restricted": np.array([1.0 if in_restricted else 0.0], dtype=np.float64),
            "outside_sector": np.array([1.0 if outside_sector else 0.0], dtype=np.float64),
        }


class NeighborObservationV1:
    """Body-frame neighbours sorted by TCPA, with CPA features and a valid mask.

    Empty slots (fewer live aircraft than ``n``, including multi-agent
    aircraft that have already reached the goal and been deleted) are
    zero-padded with ``valid=0``.
    """

    def __init__(self, n, pos_norm=NEIGHBOR_POS_NORM_M, spd_norm=NEIGHBOR_SPD_NORM_MPS,
                 dist_norm=NEIGHBOR_DIST_NORM_KM, tcpa_norm=TCPA_NORM_S,
                 dcpa_norm_nm=DCPA_NORM_NM, intrusion_nm=INTRUSION_NM):
        self.n = n
        self.pos_norm = pos_norm
        self.spd_norm = spd_norm
        self.dist_norm = dist_norm
        self.tcpa_norm = tcpa_norm
        self.dcpa_norm_nm = dcpa_norm_nm
        self.intrusion_nm = intrusion_nm

    def space(self):
        shape = (self.n,)
        return {
            "x_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "y_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "vx_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "vy_r": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "neighbor_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "tcpa": spaces.Box(0.0, 1.0, shape=shape, dtype=np.float64),
            "dcpa": spaces.Box(0.0, np.inf, shape=shape, dtype=np.float64),
            "cos_rel_hdg": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "sin_rel_hdg": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "intruding": spaces.Box(0.0, 1.0, shape=shape, dtype=np.float64),
            "valid": spaces.Box(0.0, 1.0, shape=shape, dtype=np.float64),
        }

    def observe(self, ac_id):
        ac_idx = bs.traf.id2idx(ac_id)
        x_r = np.zeros(self.n)
        y_r = np.zeros(self.n)
        vx_r = np.zeros(self.n)
        vy_r = np.zeros(self.n)
        distances = np.zeros(self.n)
        tcpa_n = np.zeros(self.n)
        dcpa_n = np.zeros(self.n)
        cos_rel = np.zeros(self.n)
        sin_rel = np.zeros(self.n)
        intruding = np.zeros(self.n)
        valid = np.zeros(self.n)

        if ac_idx < 0:
            return self._pack(x_r, y_r, vx_r, vy_r, distances, tcpa_n, dcpa_n,
                              cos_rel, sin_rel, intruding, valid)

        own_lat = bs.traf.lat[ac_idx]
        own_lon = bs.traf.lon[ac_idx]
        own_hdg = bs.traf.hdg[ac_idx]
        own_tas = bs.traf.tas[ac_idx]
        own_hdg_rad = np.deg2rad(own_hdg)
        vn_own = np.cos(own_hdg_rad) * own_tas
        ve_own = np.sin(own_hdg_rad) * own_tas

        other_idx = [i for i in range(bs.traf.ntraf) if i != ac_idx]
        order = _sort_by_tcpa(ac_idx, other_idx)[:self.n]

        for slot, i in enumerate(order):
            brg, dist_nm = bs.tools.geo.kwikqdrdist(
                own_lat, own_lon, bs.traf.lat[i], bs.traf.lon[i])
            dist_m = dist_nm * NM2KM * 1000.0
            rel_rad = np.deg2rad(brg - own_hdg)
            x_r[slot] = dist_m * np.cos(rel_rad)
            y_r[slot] = dist_m * np.sin(rel_rad)

            int_hdg_rad = np.deg2rad(bs.traf.hdg[i])
            vn_int = np.cos(int_hdg_rad) * bs.traf.tas[i]
            ve_int = np.sin(int_hdg_rad) * bs.traf.tas[i]
            vn_r = vn_int - vn_own
            ve_r = ve_int - ve_own
            vx_r[slot] = vn_r * np.cos(own_hdg_rad) + ve_r * np.sin(own_hdg_rad)
            vy_r[slot] = -vn_r * np.sin(own_hdg_rad) + ve_r * np.cos(own_hdg_rad)

            tcpa_s, dcpa_m = _cpa(ac_idx, i)
            if not np.isfinite(tcpa_s) or tcpa_s < 0:
                tcpa_s = self.tcpa_norm
            tcpa_n[slot] = np.clip(tcpa_s / self.tcpa_norm, 0.0, 1.0)
            dcpa_n[slot] = (dcpa_m / 1852.0) / self.dcpa_norm_nm

            rel_hdg = np.deg2rad(_bound_angle(bs.traf.hdg[i] - own_hdg))
            cos_rel[slot] = np.cos(rel_hdg)
            sin_rel[slot] = np.sin(rel_hdg)
            distances[slot] = dist_nm * NM2KM
            intruding[slot] = 1.0 if dist_nm < self.intrusion_nm else 0.0
            valid[slot] = 1.0

        return self._pack(x_r, y_r, vx_r, vy_r, distances, tcpa_n, dcpa_n,
                          cos_rel, sin_rel, intruding, valid)

    def _pack(self, x_r, y_r, vx_r, vy_r, distances, tcpa_n, dcpa_n,
              cos_rel, sin_rel, intruding, valid):
        return {
            "x_r": x_r / self.pos_norm,
            "y_r": y_r / self.pos_norm,
            "vx_r": vx_r / self.spd_norm,
            "vy_r": vy_r / self.spd_norm,
            "neighbor_distance": distances / self.dist_norm,
            "tcpa": tcpa_n,
            "dcpa": dcpa_n,
            "cos_rel_hdg": cos_rel,
            "sin_rel_hdg": sin_rel,
            "intruding": intruding,
            "valid": valid,
        }


class ObstaclePolygonObservation:
    """Signed distance to each obstacle polygon plus nearest-boundary bearing.

    Slots follow ``order`` (typically ``ObstacleObservation.last_order``) so
    the circle features and polygon features stay aligned. Missing slots are
    padded with a large positive clearance.
    """

    def __init__(self, n, distance_norm=POLYGON_DISTANCE_NORM_KM):
        self.n = n
        self.distance_norm = distance_norm

    def space(self):
        shape = (self.n,)
        return {
            "obstacle_signed_distance": spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64),
            "obstacle_boundary_cos": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
            "obstacle_boundary_sin": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float64),
        }

    def observe(self, ac_id, center, obstacles, order, own_hdg):
        signed = np.full(self.n, 1.0, dtype=np.float64)  # 1.0 * distance_norm km of clearance
        cos_b = np.zeros(self.n)
        sin_b = np.zeros(self.n)

        ac_idx = bs.traf.id2idx(ac_id)
        if ac_idx < 0:
            return self._pack(signed, cos_b, sin_b)

        own_nm = latlon_to_nm(center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        for slot, k in enumerate(order):
            if k >= len(obstacles):
                continue
            verts_nm = vertices_to_nm(center, obstacles[k].vertices)
            sd_km, qdr = polygon_clearance(own_nm, verts_nm, inside_negative=True)
            bearing = np.deg2rad(bound_angle_deg(own_hdg - qdr))
            signed[slot] = sd_km / self.distance_norm
            cos_b[slot] = np.cos(bearing)
            sin_b[slot] = np.sin(bearing)
        return self._pack(signed, cos_b, sin_b)

    def _pack(self, signed, cos_b, sin_b):
        return {
            "obstacle_signed_distance": signed,
            "obstacle_boundary_cos": cos_b,
            "obstacle_boundary_sin": sin_b,
        }


class SectorClearanceObservation:
    """Signed clearance to the sector boundary: positive inside, negative outside."""

    def __init__(self, distance_norm=POLYGON_DISTANCE_NORM_KM):
        self.distance_norm = distance_norm

    def space(self):
        return {
            "sector_signed_distance": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
        }

    def observe(self, ac_id, center, sector_vertices, own_hdg):
        del own_hdg  # bearing is available from SectorBoundaryObservation points
        ac_idx = bs.traf.id2idx(ac_id)
        if ac_idx < 0:
            return {"sector_signed_distance": np.array([0.0], dtype=np.float64)}
        own_nm = latlon_to_nm(center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        verts_nm = vertices_to_nm(center, sector_vertices)
        sd_km, _ = polygon_clearance(own_nm, verts_nm, inside_negative=False)
        return {
            "sector_signed_distance": np.array([sd_km / self.distance_norm], dtype=np.float64),
        }
