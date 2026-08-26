"""Local-flat geometry helpers for polygon clearance.

Scenario generation already works in a north/east nautical-mile frame around
the sector centre. Observations reuse that frame so signed distances are in
real kilometres rather than degrees.
"""

from __future__ import annotations

import math

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from competitor.config import NM2KM


def latlon_to_nm(center, lat, lon):
    """Convert (lat, lon) degrees to (north, east) NM around ``center``."""
    north = (lat - center[0]) * 60.0
    east = (lon - center[1]) * 60.0 * math.cos(math.radians(center[0]))
    return north, east


def vertices_to_nm(center, vertices):
    """``vertices`` is a sequence of (lat, lon) pairs."""
    return [latlon_to_nm(center, lat, lon) for lat, lon in vertices]


def polygon_clearance(own_nm, vertices_nm, inside_negative=True):
    """Signed distance (km) and true heading (deg) to the nearest boundary.

    Parameters
    ----------
    own_nm : (north, east) in nautical miles
    vertices_nm : ring of (north, east) NM points, unclosed is fine
    inside_negative : if True (obstacles), inside the polygon is negative.
        If False (sector), inside is positive — "clearance remaining".

    Returns
    -------
    signed_km : float
    qdr_deg : float
        True heading from the ownship to the nearest boundary point.
    """
    if len(vertices_nm) < 3:
        return 1e6, 0.0

    poly = Polygon(vertices_nm)
    if poly.is_empty or poly.area <= 0.0:
        return 1e6, 0.0
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty:
            return 1e6, 0.0

    pt = Point(own_nm[0], own_nm[1])
    dist_km = pt.distance(poly.exterior) * NM2KM
    inside = bool(poly.covers(pt))
    if inside_negative:
        signed_km = -dist_km if inside else dist_km
    else:
        signed_km = dist_km if inside else -dist_km

    nearest = nearest_points(pt, poly.exterior)[1]
    dn = nearest.x - own_nm[0]
    de = nearest.y - own_nm[1]
    qdr_deg = (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0
    return float(signed_km), float(qdr_deg)


def bound_angle_deg(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0
