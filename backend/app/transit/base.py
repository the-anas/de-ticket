from datetime import datetime
from typing import Protocol

from .models import Journey, Station


class TransitProvider(Protocol):
    async def search_stations(self, query: str, limit: int = 8) -> list[Station]: ...

    async def plan_journeys(
        self,
        origin: str,
        destination: str,
        when: datetime | None = None,
        arrive_by: bool = False,
        limit: int = 4,
    ) -> list[Journey]:
        """origin/destination are either a provider stop id or "lat,lon"."""
        ...
