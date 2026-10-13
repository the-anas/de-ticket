from fastapi import APIRouter, HTTPException

from ..deps import get_transit
from ..destinations import Destination, ReachableDestination, load_destinations, reachable_destinations

router = APIRouter()


@router.get("/api/destinations")
async def destinations() -> list[Destination]:
    return load_destinations()


@router.get("/api/popular")
async def popular(
    lat: float, lon: float, name: str = "", station_id: str | None = None, limit: int = 8
) -> list[ReachableDestination]:
    """Curated destinations reachable with the ticket, fastest first. The full
    journey is fetched from /api/journeys when the user opens a destination."""
    try:
        routes = await reachable_destinations(get_transit(), lat, lon, name, station_id)
    except Exception as e:
        raise HTTPException(502, f"Transit lookup failed: {type(e).__name__}") from e
    return routes[:limit]
