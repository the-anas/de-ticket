import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import DATA_DIR
from ..deps import get_transit
from ..transit import dticket
from ..transit.geo import haversine_km
from ..transit.models import Stop
from ..transit.when import default_departure

router = APIRouter()

MIN_DISTANCE_KM = 10


class Destination(BaseModel):
    name: str
    station: str
    lat: float
    lon: float
    region: str
    tags: list[str]
    blurb: str


class PopularRoute(BaseModel):
    destination: Destination
    distance_km: int
    # from the default departure time, so it includes the wait for the first train
    approx_min: int
    transfers: int


@lru_cache
def load_destinations() -> list[Destination]:
    raw = json.loads((DATA_DIR / "destinations.json").read_text(encoding="utf-8"))
    return [Destination(**d) for d in raw]


@router.get("/api/destinations")
async def destinations() -> list[Destination]:
    return load_destinations()


@router.get("/api/popular")
async def popular(
    lat: float, lon: float, name: str = "", station_id: str | None = None, limit: int = 8
) -> list[PopularRoute]:
    """Curated destinations reachable with the ticket, fastest first.

    Travel times for all destinations come from a single upstream request,
    because per-destination journey searches are rate-limited to about one
    every three seconds. The full journey is fetched from /api/journeys when
    the user opens a destination. An empty list means nothing is reachable.
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
    try:
        times = await get_transit().travel_times(
            (lat, lon), [(d.lat, d.lon) for _, d in candidates], default_departure()
        )
    except Exception as e:
        raise HTTPException(502, f"Transit lookup failed: {type(e).__name__}") from e

    routes = [
        PopularRoute(destination=d, distance_km=round(km), approx_min=t[0], transfers=t[1])
        for (km, d), t in zip(candidates, times)
        if t is not None
    ]
    routes.sort(key=lambda r: r.approx_min)
    return routes[:limit]
