"""The curated destination list and what can be reached from a starting point."""

import json
from datetime import datetime
from functools import lru_cache

from pydantic import BaseModel

from .config import DATA_DIR
from .transit import dticket
from .transit.geo import haversine_km
from .transit.models import Stop
from .transit.transitous import TransitousProvider
from .transit.when import default_departure

MIN_DISTANCE_KM = 10


class Destination(BaseModel):
    name: str
    station: str
    lat: float
    lon: float
    region: str
    tags: list[str]
    blurb: str


class ReachableDestination(BaseModel):
    destination: Destination
    distance_km: int
    # from the departure time searched, so it includes the wait for the first train
    approx_min: int
    transfers: int


@lru_cache
def load_destinations() -> list[Destination]:
    raw = json.loads((DATA_DIR / "destinations.json").read_text(encoding="utf-8"))
    return [Destination(**d) for d in raw]


async def reachable_destinations(
    transit: TransitousProvider,
    lat: float,
    lon: float,
    name: str = "",
    station_id: str | None = None,
    when: datetime | None = None,
) -> list[ReachableDestination]:
    """Curated destinations reachable with the ticket from a point, fastest first.

    Travel times for all destinations come from a single upstream request,
    because per-destination journey searches are rate-limited to about one
    every three seconds. An empty list means nothing is reachable.
    """
    # Starting abroad anywhere but a listed border station is never covered,
    # even though regional trains (and so travel times) exist there.
    problem, _ = dticket.check_endpoint(Stop(name=name, lat=lat, lon=lon, stop_id=station_id))
    if problem:
        return []
    candidates = [
        (km, d)
        for d in load_destinations()
        if (km := haversine_km(lat, lon, d.lat, d.lon)) >= MIN_DISTANCE_KM
    ]
    times = await transit.travel_times(
        (lat, lon), [(d.lat, d.lon) for _, d in candidates], when or default_departure()
    )
    routes = [
        ReachableDestination(destination=d, distance_km=round(km), approx_min=t[0], transfers=t[1])
        for (km, d), t in zip(candidates, times)
        if t is not None
    ]
    routes.sort(key=lambda r: r.approx_min)
    return routes
