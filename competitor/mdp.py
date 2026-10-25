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
from core.actions import MpS2Kt, HeadingAction, SpeedAction, combine_action_spaces
import bluesky_gym.envs.common.functions as fn

from competitor.config import (
    AIRSPEED_MEAN,
    AIRSPEED_STD,
    INTRUSION_NM,
    OBSTACLE_DISTANCE_NORM_KM,
    SECTOR_DISTANCE_NORM_KM,
    WAYPOINT_DISTANCE_NORM_KM,
    resolve_mdp_config,
)
from competitor.geometry import latlon_to_nm, polygon_clearance, vertices_to_nm
from competitor.shield import ShieldGeometry, choose_heading
from competitor.observations import (
    NeighborObservationV1,
    ObstaclePolygonObservation,
    OwnStatusObservation,
    SectorClearanceObservation,
)


REWARD_TERMS = (
    "progress", "conflict_now", "conflict_predicted", "obstacle_proximity",
    "sector_proximity", "restricted_event", "sector_exit_event",
    "intrusion_time", "restricted_time", "sector_time", "intrusion_event",
    "drift", "reach", "timeout",
)


class CompetitionMdpV1:
    """Mixin: drop-in MDP v1 on top of CompetitionEnv / CompetitionZooEnv."""

    def _setup_mdp_v1(self, n_neighbors, mdp_cfg=None):
        self.mdp_cfg = dict(mdp_cfg) if mdp_cfg is not None else resolve_mdp_config()
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
        self.last_reward_terms = {}
        self.shield_interventions = {}
        self.last_shield = {}
        self._shield_geom = None

    def reset(self, **kwargs):
        # Cleared before super().reset() because the env builds the first
        # observation inside reset.
        self._prev_goal_dist_km = {}
        self.last_reward_terms = {}
        self.shield_interventions = {}
        self.last_shield = {}
        self._shield_geom = None
        return super().reset(**kwargs)

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
        if self.mdp_cfg["ACTION_MODE"] == "goal_relative":
            self._goal_relative_action(ac_id, action)
            return
        self.heading_action.execute(ac_id, action[0])
        self.speed_action.execute(ac_id, action[1])

    def _goal_relative_action(self, ac_id, action):
        ac_idx = bs.traf.id2idx(ac_id)
        if ac_idx < 0:
            return
        goal_lat, goal_lon = self._mdp_goal(ac_id)
        qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], goal_lat, goal_lon)
        a_hdg = float(np.clip(action[0], -1.0, 1.0))
        a_spd = float(np.clip(action[1], -1.0, 1.0))
        new_hdg = fn.bound_angle_positive_negative_180(
            qdr + a_hdg * self.mdp_cfg["GOAL_MAX_OFFSET_DEG"])
        if self.mdp_cfg["SHIELD"]:
            new_hdg = self._shielded_heading(ac_id, ac_idx, new_hdg)
        nominal_kt = getattr(self, "ac_spd", AIRSPEED_MEAN) * MpS2Kt
        new_spd = nominal_kt + a_spd * self.mdp_cfg["SPEED_RANGE_KT"]
        bs.stack.stack(f"HDG {ac_id} {new_hdg}")
        bs.stack.stack(f"SPD {ac_id} {new_spd}")

    def _shielded_heading(self, ac_id, ac_idx, policy_hdg):
        cfg = self.mdp_cfg
        center = self.scenario.center
        if self._shield_geom is None:
            self._shield_geom = ShieldGeometry(
                [vertices_to_nm(center, obs.vertices) for obs in self.scenario.obstacles],
                vertices_to_nm(center, self.scenario.sector))
        others = [j for j in range(bs.traf.ntraf) if j != ac_idx]
        others_ne = [latlon_to_nm(center, bs.traf.lat[j], bs.traf.lon[j]) for j in others]
        hdg_rad = np.deg2rad(bs.traf.hdg[others]) if others else np.zeros(0)
        spd_nm_s = (bs.traf.tas[others] / 1852.0) if others else np.zeros(0)
        others_vel = np.stack([np.cos(hdg_rad) * spd_nm_s, np.sin(hdg_rad) * spd_nm_s], axis=-1)
        heading, intervened, _ = choose_heading(
            policy_hdg,
            latlon_to_nm(center, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]),
            float(bs.traf.tas[ac_idx]), others_ne, others_vel, self._shield_geom,
            lookahead_s=cfg["SHIELD_LOOKAHEAD_S"],
            obstacle_buffer_km=cfg["SHIELD_OBSTACLE_BUFFER_KM"],
            sector_buffer_km=cfg["SHIELD_SECTOR_BUFFER_KM"],
            intruder_nm=cfg["SHIELD_INTRUDER_NM"],
            own_hdg=float(bs.traf.hdg[ac_idx]), bank_deg=cfg["SHIELD_BANK_DEG"])
        self.shield_interventions[ac_id] = self.shield_interventions.get(ac_id, 0) + int(intervened)
        self.last_shield[ac_id] = intervened
        return heading

    def _get_reward(self, ac_id):
        """Training signal only — does not affect scored metrics or termination.

        Each additive term is also stored in ``last_reward_terms[ac_id]`` so
        training telemetry can show which term dominates the return.
        """
        ac_idx = bs.traf.id2idx(ac_id)
        if ac_idx < 0:
            self.last_reward_terms[ac_id] = dict.fromkeys(REWARD_TERMS, 0.0)
            return 0.0

        cfg = self.mdp_cfg
        terms = dict.fromkeys(REWARD_TERMS, 0.0)
        m = self.metrics[ac_id]
        base = self._mdp_baseline(ac_id)
        goal_lat, goal_lon = self._mdp_goal(ac_id)
        lat, lon = float(bs.traf.lat[ac_idx]), float(bs.traf.lon[ac_idx])
        hdg = float(bs.traf.hdg[ac_idx])

        _, dist_nm = kwikqdrdist(lat, lon, goal_lat, goal_lon)
        dist_km = dist_nm * NM2KM
        prev = self._prev_goal_dist_km.get(ac_id, dist_km)
        terms["progress"] = cfg["PROGRESS_PER_KM"] * (prev - dist_km)
        self._prev_goal_dist_km[ac_id] = dist_km

        threat = False

        # Predicted and current losses of separation.
        tcpa_horizon = cfg["TCPA_HORIZON_S"]
        dcpa_limit = cfg["DCPA_LIMIT_NM"]
        for other_idx in range(bs.traf.ntraf):
            if other_idx == ac_idx:
                continue
            tcpa_s, dcpa_m = _cpa(ac_idx, other_idx)
            dcpa_nm = dcpa_m / 1852.0
            _, pair_nm = kwikqdrdist(lat, lon, bs.traf.lat[other_idx], bs.traf.lon[other_idx])
            if pair_nm < INTRUSION_NM:
                terms["conflict_now"] -= cfg["CONFLICT_WEIGHT"] * (1.0 - pair_nm / INTRUSION_NM)
                threat = True
            if np.isfinite(tcpa_s) and 0.0 <= tcpa_s <= tcpa_horizon and dcpa_nm < dcpa_limit:
                severity = (1.0 - dcpa_nm / dcpa_limit) * (1.0 - tcpa_s / tcpa_horizon)
                terms["conflict_predicted"] -= cfg["CONFLICT_WEIGHT"] * max(0.0, severity)
                threat = True

        horizon_km = cfg["PROXIMITY_HORIZON_KM"]
        center = self.scenario.center
        own_nm = latlon_to_nm(center, lat, lon)
        min_obs = horizon_km
        for obs in self.scenario.obstacles:
            sd_km, _ = polygon_clearance(
                own_nm, vertices_to_nm(center, obs.vertices), inside_negative=True)
            min_obs = min(min_obs, sd_km)
        if min_obs < horizon_km:
            clipped = max(min_obs, -horizon_km)
            terms["obstacle_proximity"] = -cfg["PROXIMITY_WEIGHT"] * (horizon_km - clipped) / horizon_km
            threat = True

        sector_clearance, _ = polygon_clearance(
            own_nm, vertices_to_nm(center, self.scenario.sector), inside_negative=False)
        if sector_clearance < horizon_km:
            clipped = max(sector_clearance, -horizon_km)
            terms["sector_proximity"] = -cfg["PROXIMITY_WEIGHT"] * (horizon_km - clipped) / horizon_km
            threat = True

        d_restricted = m["restricted_area_events"] - base["restricted_area_events"]
        d_sector = m["sector_exit_events"] - base["sector_exit_events"]
        terms["restricted_event"] = -cfg["RESTRICTED_EVENT_PENALTY"] * d_restricted
        terms["sector_exit_event"] = -cfg["SECTOR_EXIT_EVENT_PENALTY"] * d_sector
        if d_restricted > 0 or d_sector > 0:
            threat = True

        d_intrusion_s = m["intrusion_time"] - base["intrusion_time"]
        d_restricted_s = m["time_in_restricted_area"] - base["time_in_restricted_area"]
        terms["intrusion_time"] = -cfg["INTRUSION_TIME_WEIGHT"] * d_intrusion_s
        terms["restricted_time"] = -cfg["RESTRICTED_TIME_WEIGHT"] * d_restricted_s

        d_outside_s = m["time_outside_sector"] - base["time_outside_sector"]
        d_intrusion_events = m["intrusion_events"] - base["intrusion_events"]
        terms["sector_time"] = -cfg["SECTOR_TIME_WEIGHT"] * d_outside_s
        terms["intrusion_event"] = -cfg["INTRUSION_EVENT_PENALTY"] * d_intrusion_events
        if terms["sector_time"] or terms["intrusion_event"]:
            threat = True

        if not threat:
            qdr, _ = kwikqdrdist(lat, lon, goal_lat, goal_lon)
            drift = abs(np.deg2rad(fn.bound_angle_positive_negative_180(hdg - qdr)))
            terms["drift"] = cfg["DRIFT_PENALTY"] * drift

        reached = m["waypoint_reached"] - base["waypoint_reached"]
        terms["reach"] = cfg["REACH_REWARD"] * reached

        if (m["flight_time"] >= self.episode_time_limit
                and not m["waypoint_reached"]):
            terms["timeout"] = cfg["TIMEOUT_PENALTY"]

        self.last_reward_terms[ac_id] = terms
        return float(sum(terms.values()))

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


def mdp_v1_init_kwargs(kwargs, mdp_cfg):
    """Apply v1 action defaults without overriding an explicit caller value."""
    kwargs.setdefault("d_heading", mdp_cfg["D_HEADING"])
    kwargs.setdefault("d_speed", mdp_cfg["D_SPEED"])
    kwargs.setdefault("action_frequency", mdp_cfg["ACTION_FREQUENCY"])
    return kwargs
