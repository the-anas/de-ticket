from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from ..deps import get_transit
from ..transit.models import Journey
from ..transit.transitous import BERLIN

router = APIRouter()

PLACE = r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?$"


def default_departure(now: datetime | None = None) -> datetime:
    """Now during the day, otherwise the next morning, so that browsing in the
    evening does not show journeys stretched by the overnight timetable gap."""
    now = now or datetime.now(BERLIN)
    if 6 <= now.hour < 18:
        return now
    day = now if now.hour < 6 else now + timedelta(days=1)
    return day.replace(hour=8, minute=0, second=0, microsecond=0)


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
