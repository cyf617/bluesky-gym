"""Tunable constants for competition MDP v1.

Keep this file the single place to tweak scales and reward weights. The
observation / action / reward implementations import from here so a later
coefficient sweep does not require hunting through env code.

Names listed in ``TUNABLE_KEYS`` can be overridden per run without editing
this file (``CompetitionEnvV1(mdp_overrides={...})`` or
``train_sa_sac.py --set KEY=VALUE``); ``resolve_mdp_config`` merges them.
"""

# --- action ---
# Official defaults are d_heading=45, d_speed=20/3 kt, action_frequency=10.
# v1 uses a smaller heading stick, a usable speed stick, and a 5 s decision
# interval (metrics and the 3000 s budget are still sampled every 1 s).
D_HEADING = 15.0          # deg at |action| = 1
D_SPEED = 25.0            # kt at |action| = 1
ACTION_FREQUENCY = 5      # simulated seconds per env step
# "relative": heading/speed change on top of the current state (action 0 = hold).
#   Exploration noise integrates into a random walk, so an untrained policy
#   wanders off and 0 % arrival is the starting point.
# "goal_relative": heading = bearing to goal + a0 * GOAL_MAX_OFFSET_DEG and
#   speed = AIRSPEED_MEAN + a1 * SPEED_RANGE_KT. Action 0 = the neutral
#   fly-to-goal policy, so exploration starts from 100 % arrival.
ACTION_MODE = "relative"
GOAL_MAX_OFFSET_DEG = 90.0
SPEED_RANGE_KT = 40.0

# --- observation scales ---
WAYPOINT_DISTANCE_NORM_KM = 170.0
# Neighbours are a local problem: the official IntruderObservation default
# pos_norm=1e6 m (~1000 km) crushes the signal. 25 km / 50 m/s matches
# "a few tens of km / a few tens of m/s".
NEIGHBOR_POS_NORM_M = 25_000.0
NEIGHBOR_SPD_NORM_MPS = 50.0
NEIGHBOR_DIST_NORM_KM = 25.0
TCPA_NORM_S = 120.0
DCPA_NORM_NM = 5.0
AIRSPEED_MEAN = 150.0     # m/s, matches env ac_spd
AIRSPEED_STD = 20.0
OBSTACLE_DISTANCE_NORM_KM = 170.0
SECTOR_DISTANCE_NORM_KM = 170.0
POLYGON_DISTANCE_NORM_KM = 20.0

# --- reward ---
PROGRESS_PER_KM = 0.15    # 100 km closer ≈ +15; 0.01 was drowned by conflict terms
REACH_REWARD = 80.0
TIMEOUT_PENALTY = -80.0   # same order as reach, so wandering for 3000 s is not cheap
DRIFT_PENALTY = -0.01     # only applied when no threat
TCPA_HORIZON_S = 120.0
DCPA_LIMIT_NM = 5.0
CONFLICT_WEIGHT = 0.05    # per-step; keep below typical progress so the agent still goes to the goal
PROXIMITY_HORIZON_KM = 5.0
PROXIMITY_WEIGHT = 0.05
RESTRICTED_EVENT_PENALTY = 1.0
SECTOR_EXIT_EVENT_PENALTY = 2.0

# --- world (must match official scoring; do not change for evaluation) ---
INTRUSION_NM = 5.0
NM2KM = 1.852

TUNABLE_KEYS = (
    "D_HEADING", "D_SPEED", "ACTION_FREQUENCY",
    "ACTION_MODE", "GOAL_MAX_OFFSET_DEG", "SPEED_RANGE_KT",
    "PROGRESS_PER_KM", "REACH_REWARD", "TIMEOUT_PENALTY", "DRIFT_PENALTY",
    "TCPA_HORIZON_S", "DCPA_LIMIT_NM", "CONFLICT_WEIGHT",
    "PROXIMITY_HORIZON_KM", "PROXIMITY_WEIGHT",
    "RESTRICTED_EVENT_PENALTY", "SECTOR_EXIT_EVENT_PENALTY",
)
ACTION_MODES = ("relative", "goal_relative")


def resolve_mdp_config(overrides=None):
    """Module defaults for ``TUNABLE_KEYS`` updated with ``overrides``.

    Values are coerced to the type of the default so ``--set KEY=1`` works for
    float keys. Unknown keys raise instead of being silently ignored.
    """
    cfg = {k: globals()[k] for k in TUNABLE_KEYS}
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"unknown MDP config key {key!r}; tunable: {', '.join(TUNABLE_KEYS)}")
        cfg[key] = type(cfg[key])(value)
    if cfg["ACTION_MODE"] not in ACTION_MODES:
        raise ValueError(f"ACTION_MODE must be one of {ACTION_MODES}, got {cfg['ACTION_MODE']!r}")
    return cfg


def parse_overrides(pairs):
    """``["KEY=VALUE", ...]`` -> dict (values stay strings until resolve)."""
    out = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out
