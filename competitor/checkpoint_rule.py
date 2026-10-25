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


def safety_total(m):
    """Seconds in loss of separation + inside restricted areas + outside the sector."""
    return m["intrusion_time"] + m["time_in_restricted_area"] + m["time_outside_sector"]


def is_better_safety_total(candidate, current, min_arrival):
    """True if ``candidate`` should replace the best-safety-total checkpoint."""
    if candidate["waypoint_reached"] + 1e-9 < min_arrival:
        return False
    if current is None:
        return True
    if safety_total(candidate) < safety_total(current) - 1e-9:
        return True
    if abs(safety_total(candidate) - safety_total(current)) <= 1e-9:
        return candidate["waypoint_reached"] > current["waypoint_reached"] + 1e-9
    return False


class ArrivalStopRule:
    """Stop once arrival < ``min_arrival`` for ``patience`` consecutive evals after ``after`` steps."""

    def __init__(self, after, min_arrival=0.95, patience=2):
        self.after = after
        self.min_arrival = min_arrival
        self.patience = patience
        self.streak = 0
        self.reason = None

    def update(self, timesteps, arrival):
        if self.after is None or timesteps <= self.after:
            return None
        self.streak = self.streak + 1 if arrival + 1e-9 < self.min_arrival else 0
        if self.streak >= self.patience and self.reason is None:
            self.reason = (f"arrival < {self.min_arrival:.2f} for {self.patience} consecutive evals "
                           f"after {self.after} steps (last {arrival:.3f} at t={timesteps})")
        return self.reason
