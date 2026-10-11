import json
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt

from ..config import DATA_DIR


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


@lru_cache
def _germany() -> list[tuple[tuple[float, float, float, float], list[list[list[float]]]]]:
    """Polygons of Germany as (bbox, rings); Natural Earth 10m, about 100 m accuracy."""
    feature = json.loads((DATA_DIR / "germany.geojson").read_text(encoding="utf-8"))
    polygons = []
    for rings in feature["geometry"]["coordinates"]:
        xs = [p[0] for p in rings[0]]
        ys = [p[1] for p in rings[0]]
        polygons.append(((min(xs), min(ys), max(xs), max(ys)), rings))
    return polygons


def _in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_germany(lat: float, lon: float) -> bool:
    for (minx, miny, maxx, maxy), rings in _germany():
        if not (minx <= lon <= maxx and miny <= lat <= maxy):
            continue
        if _in_ring(lon, lat, rings[0]) and not any(_in_ring(lon, lat, h) for h in rings[1:]):
            return True
    return False


def km_outside_germany(lat: float, lon: float) -> float:
    """0 inside Germany, otherwise the approximate distance to the border."""
    if in_germany(lat, lon):
        return 0.0
    kx, ky = 111.32 * cos(radians(lat)), 110.57
    best = float("inf")
    for (minx, miny, maxx, maxy), rings in _germany():
        if lon < minx - 1 or lon > maxx + 1 or lat < miny - 1 or lat > maxy + 1:
            continue
        ring = rings[0]
        for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
            ax, ay = (x1 - lon) * kx, (y1 - lat) * ky
            bx, by = (x2 - lon) * kx, (y2 - lat) * ky
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length2))
            best = min(best, sqrt((ax + t * dx) ** 2 + (ay + t * dy) ** 2))
    return best
