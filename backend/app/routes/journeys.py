from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from ..deps import get_transit
from ..transit.models import Journey
from ..transit.when import default_departure

router = APIRouter()

PLACE = r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$"


@router.get("/api/journeys")
async def journeys(
    origin: str = Query(alias="from", pattern=PLACE, description="lat,lon"),
    destination: str = Query(alias="to", pattern=PLACE, description="lat,lon"),
    time: datetime | None = None,
    arrive_by: bool = False,
) -> list[Journey]:
    return await get_transit().plan_journeys(origin, destination, time or default_departure(), arrive_by)


@router.get("/api/journeys/{journey_id}")
async def journey(journey_id: str) -> Journey:
    found = get_transit().journeys.get(journey_id)
    if found is None:
        raise HTTPException(404, "Journey not found or expired")
    return found
