"""Rules for keeping a checkpoint. No BlueSky / SB3 imports.

Decisions use official metric means only. ``total_reward`` is accepted in
the dict so callers can pass a full ``info`` summary, but it is never read.
"""


def is_better_arrival(candidate, current):
    """True if ``candidate`` should replace the best-arrival checkpoint."""
    if current is None:
        return True
    if candidate["waypoint_reached"] > current["waypoint_reached"] + 1e-9:
        return True
    if abs(candidate["waypoint_reached"] - current["waypoint_reached"]) <= 1e-9:
        # Do not shuffle 0%-arrival policies by intrusion time; that rewards
        # "flee the sector" local minima.
        if candidate["waypoint_reached"] <= 0.0:
            return False
        return candidate["intrusion_time"] < current["intrusion_time"]
    return False


def is_better_safety(candidate, current, min_arrival):
    """True if ``candidate`` should replace the best-safety checkpoint."""
    if candidate["waypoint_reached"] + 1e-9 < min_arrival:
        return False
    if current is None:
        return True
    if candidate["intrusion_time"] < current["intrusion_time"] - 1e-9:
        return True
    if abs(candidate["intrusion_time"] - current["intrusion_time"]) <= 1e-9:
        if candidate["waypoint_reached"] > current["waypoint_reached"] + 1e-9:
            return True
        return candidate["time_in_restricted_area"] < current["time_in_restricted_area"]
    return False
