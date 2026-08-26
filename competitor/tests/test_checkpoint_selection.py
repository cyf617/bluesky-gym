"""Unit tests for checkpoint selection (no BlueSky)."""

import unittest

from competitor.checkpoint_rule import is_better_arrival, is_better_safety


def _m(arrival, intrusion, restricted=0.0):
    return {
        "waypoint_reached": arrival,
        "intrusion_time": intrusion,
        "time_in_restricted_area": restricted,
        "total_reward": 999.0,  # must never influence the decision
    }


class SelectionTests(unittest.TestCase):
    def test_arrival_prefers_higher_rate(self):
        self.assertTrue(is_better_arrival(_m(1.0, 80.0), _m(0.9, 10.0)))
        self.assertFalse(is_better_arrival(_m(0.9, 10.0), _m(1.0, 80.0)))

    def test_arrival_tie_breaks_on_intrusion_not_return(self):
        self.assertTrue(is_better_arrival(_m(1.0, 50.0), _m(1.0, 80.0)))
        self.assertFalse(is_better_arrival(_m(1.0, 80.0), _m(1.0, 50.0)))

    def test_safety_rejects_collapsed_arrival(self):
        self.assertFalse(is_better_safety(_m(0.5, 1.0), None, min_arrival=0.90))
        self.assertTrue(is_better_safety(_m(0.95, 40.0), None, min_arrival=0.90))

    def test_safety_prefers_lower_intrusion(self):
        current = _m(0.95, 50.0)
        self.assertTrue(is_better_safety(_m(0.95, 30.0), current, min_arrival=0.90))
        self.assertFalse(is_better_safety(_m(1.0, 60.0), current, min_arrival=0.90))

    def test_training_return_is_irrelevant(self):
        low_return = _m(1.0, 20.0)
        low_return["total_reward"] = -1e6
        high_return = _m(1.0, 80.0)
        high_return["total_reward"] = 1e6
        self.assertTrue(is_better_arrival(low_return, high_return))
        self.assertTrue(is_better_safety(low_return, high_return, min_arrival=0.90))


if __name__ == "__main__":
    unittest.main()
