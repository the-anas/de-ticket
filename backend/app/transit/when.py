from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")


def default_departure(now: datetime | None = None) -> datetime:
    """Departure time used when the client does not give one.

    Now during the day, otherwise the next morning, so that browsing in the
    evening does not show journeys stretched by the overnight timetable gap.
    Rounded up to five minutes so that requests made close together share a
    cache entry.
    """
    now = now or datetime.now(BERLIN)
    if 6 <= now.hour < 18:
        floored = now.replace(minute=now.minute - now.minute % 5, second=0, microsecond=0)
        return floored if floored == now else floored + timedelta(minutes=5)
    day = now if now.hour < 6 else now + timedelta(days=1)
    return day.replace(hour=8, minute=0, second=0, microsecond=0)
