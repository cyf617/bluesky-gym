"""Geometry unit tests that do not need BlueSky."""

import unittest

from competitor.geometry import latlon_to_nm, polygon_clearance


class GeometryTests(unittest.TestCase):
    def test_latlon_to_nm_roundtrip_offsets(self):
        center = (52.0, 4.0)
        # 1 degree of latitude is 60 NM north.
        north, east = latlon_to_nm(center, 53.0, 4.0)
        self.assertAlmostEqual(north, 60.0, places=6)
        self.assertAlmostEqual(east, 0.0, places=6)

    def test_obstacle_signed_distance_inside_negative(self):
        # 10 NM square centred on origin. 5 NM from centre to each side.
        half = 5.0
        square = [(-half, -half), (half, -half), (half, half), (-half, half)]
        inside, _ = polygon_clearance((0.0, 0.0), square, inside_negative=True)
        # 5 NM * 1.852 km/NM ≈ 9.26 km, negative because we are inside.
        self.assertAlmostEqual(inside, -5.0 * 1.852, places=5)

        outside, qdr = polygon_clearance((10.0, 0.0), square, inside_negative=True)
        self.assertAlmostEqual(outside, 5.0 * 1.852, places=5)
        # Nearest boundary is due south of a point 10 NM north of origin.
        self.assertAlmostEqual(qdr % 360.0, 180.0, places=4)

    def test_sector_signed_distance_inside_positive(self):
        half = 5.0
        square = [(-half, -half), (half, -half), (half, half), (-half, half)]
        inside, _ = polygon_clearance((0.0, 0.0), square, inside_negative=False)
        self.assertGreater(inside, 0.0)
        outside, _ = polygon_clearance((10.0, 0.0), square, inside_negative=False)
        self.assertLess(outside, 0.0)
        self.assertAlmostEqual(abs(inside), abs(outside), places=5)

    def test_boundary_point_is_zero(self):
        half = 5.0
        square = [(-half, -half), (half, -half), (half, half), (-half, half)]
        sd, _ = polygon_clearance((half, 0.0), square, inside_negative=True)
        self.assertAlmostEqual(sd, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
