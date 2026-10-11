from fastapi import APIRouter, Query

from ..deps import get_transit
from ..transit.models import Station

router = APIRouter()


@router.get("/api/stations")
async def stations(q: str = Query(min_length=2)) -> list[Station]:
    return await get_transit().search_stations(q)
