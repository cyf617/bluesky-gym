"""Shared MDP v1 hooks for the single-agent and multi-agent competition envs.

Both official env classes already take ``ac_id`` on ``_get_obs`` /
``_get_reward`` / ``_get_action``. This mixin overrides those three hooks and
rebuilds the observation space; it does **not** touch ``_update_metrics`` or
``_get_info``.
"""

from __future__ import annotations

import numpy as np
import bluesky as bs
from gymnasium import spaces

from core.observations import (
    NM2KM,
    ObstacleObservation,
    OwnAirspeedObservation,
    SectorBoundaryObservation,
    WaypointObservation,
    _cpa,
)
from core.tools import kwikqdrdist
from core.actions import HeadingAction, SpeedAction, combine_action_spaces
import bluesky_gym.envs.common.functions as fn

from competitor.config import (
    AIRSPEED_MEAN,
    AIRSPEED_STD,
    CONFLICT_WEIGHT,
    D_HEADING,
    D_SPEED,
    ACTION_FREQUENCY,
    DCPA_LIMIT_NM,
    DRIFT_PENALTY,
    INTRUSION_NM,
    OBSTACLE_DISTANCE_NORM_KM,
    PROGRESS_PER_KM,
    PROXIMITY_HORIZON_KM,
    PROXIMITY_WEIGHT,
    REACH_REWARD,
    RESTRICTED_EVENT_PENALTY,
    SECTOR_DISTANCE_NORM_KM,
    SECTOR_EXIT_EVENT_PENALTY,
    TCPA_HORIZON_S,
    TIMEOUT_PENALTY,
    WAYPOINT_DISTANCE_NORM_KM,
)
from competitor.geometry import latlon_to_nm, polygon_clearance, vertices_to_nm
from competitor.observations import (
    NeighborObservationV1,
    ObstaclePolygonObservation,
    OwnStatusObservation,
    SectorClearanceObservation,
)


class CompetitionMdpV1:
    """Mixin: drop-in MDP v1 on top of CompetitionEnv / CompetitionZooEnv."""

    def _setup_mdp_v1(self, n_neighbors):
        self.waypoint_obs = WaypointObservation(n=1, distance_norm=WAYPOINT_DISTANCE_NORM_KM)
        self.airspeed_obs = OwnAirspeedObservation(spd_mean=AIRSPEED_MEAN, spd_std=AIRSPEED_STD)
        self.own_status_obs = OwnStatusObservation()
        self.neighbor_obs = NeighborObservationV1(n=n_neighbors)
        self.obstacle_obs = ObstacleObservation(
            n=self.n_obstacles,
            distance_norm=OBSTACLE_DISTANCE_NORM_KM,
            radius_norm=float(np.sqrt(1000.0 / np.pi)),
        )
        self.obstacle_polygon_obs = ObstaclePolygonObservation(n=self.n_obstacles)
        self.sector_obs = SectorBoundaryObservation(n=12, distance_norm=SECTOR_DISTANCE_NORM_KM)
        self.sector_clearance_obs = SectorClearanceObservation()

        observation_space = spaces.Dict({
            **self.waypoint_obs.space(),
            **self.airspeed_obs.space(),
            **self.own_status_obs.space(),
            **self.neighbor_obs.space(),
            **self.obstacle_obs.space(),
            **self.obstacle_polygon_obs.space(),
            **self.sector_obs.space(),
            **self.sector_clearance_obs.space(),
        })
        if hasattr(self, "observation_spaces"):
            self.observation_spaces = {a: observation_space for a in self.possible_agents}
        else:
            self.observation_space = observation_space

        self.heading_action = HeadingAction(d_heading=self.d_heading)
        self.speed_action = SpeedAction(d_speed=self.d_speed)
        action_space = combine_action_spaces([self.heading_action, self.speed_action])
        if hasattr(self, "action_spaces"):
            self.action_spaces = {a: action_space for a in self.possible_agents}
        else:
            self.action_space = action_space

        self._prev_goal_dist_km = {}

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self._prev_goal_dist_km = {}
        return result

    # ------------------------------------------------------------------ hooks
    def _get_obs(self, ac_id):
        goal_lat, goal_lon = self._mdp_goal(ac_id)
        in_conflict, in_restricted, outside = self._mdp_status(ac_id)
        ac_idx = bs.traf.id2idx(ac_id)
        own_hdg = float(bs.traf.hdg[ac_idx]) if ac_idx >= 0 else 0.0
        center = self.scenario.center

        circle = self.obstacle_obs.observe(
            ac_id, self.obstacle_centre_lat, self.obstacle_centre_lon, self.obstacle_radius)
        return {
            **self.waypoint_obs.observe(ac_id, [goal_lat], [goal_lon]),
            **self.airspeed_obs.observe(ac_id),
            **self.own_status_obs.observe(in_conflict, in_restricted, outside),
            **self.neighbor_obs.observe(ac_id),
            **circle,
            **self.obstacle_polygon_obs.observe(
                ac_id, center, self.scenario.obstacles, self.obstacle_obs.last_order, own_hdg),
            **self.sector_obs.observe(ac_id, self.scenario.sector, not outside),
            **self.sector_clearance_obs.observe(ac_id, center, self.scenario.sector, own_hdg),
        }

    def _get_action(self, ac_id, action):
        self.heading_action.execute(ac_id, action[0])
        self.speed_action.execute(ac_id, action[1])

    def _get_reward(self, ac_id):
        """Training signal only — does not affect scored metrics or termination."""
        ac_idx = bs.traf.id2idx(ac_id)
        if ac_idx < 0:
            return 0.0

        m = self.metrics[ac_id]
        base = self._mdp_baseline(ac_id)
        goal_lat, goal_lon = self._mdp_goal(ac_id)
        lat, lon = float(bs.traf.lat[ac_idx]), float(bs.traf.lon[ac_idx])
        hdg = float(bs.traf.hdg[ac_idx])

        _, dist_nm = kwikqdrdist(lat, lon, goal_lat, goal_lon)
        dist_km = dist_nm * NM2KM
        prev = self._prev_goal_dist_km.get(ac_id, dist_km)
        reward = PROGRESS_PER_KM * (prev - dist_km)
        self._prev_goal_dist_km[ac_id] = dist_km

        threat = False

        # Predicted and current losses of separation.
        for other_idx in range(bs.traf.ntraf):
            if other_idx == ac_idx:
                continue
            tcpa_s, dcpa_m = _cpa(ac_idx, other_idx)
            dcpa_nm = dcpa_m / 1852.0
            _, pair_nm = kwikqdrdist(lat, lon, bs.traf.lat[other_idx], bs.traf.lon[other_idx])
            if pair_nm < INTRUSION_NM:
                reward -= CONFLICT_WEIGHT * (1.0 - pair_nm / INTRUSION_NM)
                threat = True
            if np.isfinite(tcpa_s) and 0.0 <= tcpa_s <= TCPA_HORIZON_S and dcpa_nm < DCPA_LIMIT_NM:
                severity = (1.0 - dcpa_nm / DCPA_LIMIT_NM) * (1.0 - tcpa_s / TCPA_HORIZON_S)
                reward -= CONFLICT_WEIGHT * max(0.0, severity)
                threat = True

        center = self.scenario.center
        own_nm = latlon_to_nm(center, lat, lon)
        min_obs = PROXIMITY_HORIZON_KM
        for obs in self.scenario.obstacles:
            sd_km, _ = polygon_clearance(
                own_nm, vertices_to_nm(center, obs.vertices), inside_negative=True)
            min_obs = min(min_obs, sd_km)
        if min_obs < PROXIMITY_HORIZON_KM:
            clipped = max(min_obs, -PROXIMITY_HORIZON_KM)
            reward -= PROXIMITY_WEIGHT * (PROXIMITY_HORIZON_KM - clipped) / PROXIMITY_HORIZON_KM
            threat = True

        sector_clearance, _ = polygon_clearance(
            own_nm, vertices_to_nm(center, self.scenario.sector), inside_negative=False)
        if sector_clearance < PROXIMITY_HORIZON_KM:
            clipped = max(sector_clearance, -PROXIMITY_HORIZON_KM)
            reward -= PROXIMITY_WEIGHT * (PROXIMITY_HORIZON_KM - clipped) / PROXIMITY_HORIZON_KM
            threat = True

        d_restricted = m["restricted_area_events"] - base["restricted_area_events"]
        d_sector = m["sector_exit_events"] - base["sector_exit_events"]
        reward -= RESTRICTED_EVENT_PENALTY * d_restricted
        reward -= SECTOR_EXIT_EVENT_PENALTY * d_sector
        if d_restricted > 0 or d_sector > 0:
            threat = True

        if not threat:
            qdr, _ = kwikqdrdist(lat, lon, goal_lat, goal_lon)
            drift = abs(np.deg2rad(fn.bound_angle_positive_negative_180(hdg - qdr)))
            reward += DRIFT_PENALTY * drift

        reached = m["waypoint_reached"] - base["waypoint_reached"]
        reward += REACH_REWARD * reached

        if (m["flight_time"] >= self.episode_time_limit
                and not m["waypoint_reached"]):
            reward += TIMEOUT_PENALTY

        return float(reward)

    # ---------------------------------------------------------- env adapters
    def _mdp_goal(self, ac_id):
        if hasattr(self, "_goal"):
            return self._goal[ac_id]
        return self.goal_lat, self.goal_lon

    def _mdp_status(self, ac_id):
        if hasattr(self, "_in_intrusion_with"):
            in_conflict = bool(self._in_intrusion_with.get(ac_id))
            in_restricted = bool(self._obstacles_occupied.get(ac_id))
            outside = bool(self._outside_sector.get(ac_id, False))
        else:
            in_conflict = bool(getattr(self, "_intruders_in_intrusion", ()))
            in_restricted = bool(getattr(self, "_obstacles_occupied", ()))
            outside = bool(getattr(self, "_outside_sector", False))
        return in_conflict, in_restricted, outside

    def _mdp_baseline(self, ac_id):
        base = self._reward_baseline
        if isinstance(base, dict) and "flight_time" in base:
            return base
        return base[ac_id]


def mdp_v1_init_kwargs(kwargs):
    """Apply v1 action defaults without overriding an explicit caller value."""
    kwargs.setdefault("d_heading", D_HEADING)
    kwargs.setdefault("d_speed", D_SPEED)
    kwargs.setdefault("action_frequency", ACTION_FREQUENCY)
    return kwargs
