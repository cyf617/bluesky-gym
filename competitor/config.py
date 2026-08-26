"""Tunable constants for competition MDP v1.

Keep this file the single place to tweak scales and reward weights. The
observation / action / reward implementations import from here so a later
coefficient sweep does not require hunting through env code.
"""

# --- action ---
# Official defaults are d_heading=45, d_speed=20/3 kt, action_frequency=10.
# v1 uses a smaller heading stick, a usable speed stick, and a 5 s decision
# interval (metrics and the 3000 s budget are still sampled every 1 s).
D_HEADING = 15.0          # deg at |action| = 1
D_SPEED = 25.0            # kt at |action| = 1
ACTION_FREQUENCY = 5      # simulated seconds per env step

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
