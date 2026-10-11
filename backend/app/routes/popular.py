import json
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import DATA_DIR
from ..transit.geo import haversine_km

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


@lru_cache
def load_destinations() -> list[Destination]:
    raw = json.loads((DATA_DIR / "destinations.json").read_text(encoding="utf-8"))
    return [Destination(**d) for d in raw]


@router.get("/api/destinations")
async def destinations() -> list[Destination]:
    return load_destinations()


@router.get("/api/popular")
async def popular(lat: float, lon: float, limit: int = 8) -> list[PopularRoute]:
    """Curated destinations nearest to the start point, closest first.

    Journey times are not included: the transit API is rate-limited, so the
    client fetches them one by one from /api/journeys and fills them in.
    """
    ranked = sorted(
        ((haversine_km(lat, lon, d.lat, d.lon), d) for d in load_destinations()),
        key=lambda pair: pair[0],
    )
    return [
        PopularRoute(destination=d, distance_km=round(km))
        for km, d in ranked
        if km >= MIN_DISTANCE_KM
    ][:limit]
