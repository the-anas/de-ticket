import asyncio
import hashlib
import time
from datetime import datetime, timezone

import httpx

from ..config import get_settings
from . import dticket
from .models import Journey, Leg, Station, Stop
from .when import BERLIN
CACHE_TTL_S = 1800
# api.transitous.org rate-limits per client: a small burst is queued, anything
# beyond it gets 429, and the sustained rate is roughly one request every 3 s.
MAX_CONCURRENCY = 3
RETRY_DELAYS_S = (2, 4, 8)


def decode_polyline(encoded: str, precision: int) -> list[tuple[float, float]]:
    """Google encoded polyline -> [(lon, lat), ...]."""
    factor = 10**precision
    coords: list[tuple[float, float]] = []
    index = lat = lon = 0
    while index < len(encoded):
        for is_lon in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lon:
                lon += delta
            else:
                lat += delta
        coords.append((lon / factor, lat / factor))
    return coords


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(BERLIN)


def _stop(raw: dict, time_key: str | None = None) -> Stop:
    return Stop(
        name=raw.get("name") or "",
        lat=raw["lat"],
        lon=raw["lon"],
        stop_id=raw.get("stopId"),
        time=_time(raw.get(time_key)) if time_key else None,
        track=raw.get("track") or raw.get("scheduledTrack"),
    )


def parse_leg(raw: dict) -> Leg:
    geometry = raw.get("legGeometry") or {}
    start, end = _time(raw["startTime"]), _time(raw["endTime"])
    color = raw.get("routeColor")
    return Leg(
        mode=raw["mode"],
        line=raw.get("routeShortName") or raw.get("displayName") or None,
        operator=raw.get("agencyName") or None,
        headsign=raw.get("headsign") or None,
        origin=_stop(raw["from"], "departure"),
        destination=_stop(raw["to"], "arrival"),
        departure=start,
        arrival=end,
        duration_min=round(raw.get("duration", (end - start).total_seconds()) / 60),
        color=f"#{color}" if color else None,
        geometry=decode_polyline(geometry["points"], geometry.get("precision", 6))
        if geometry.get("points")
        else [],
        via=[_stop(s) for s in raw.get("intermediateStops", [])],
    )


def parse_itinerary(raw: dict) -> Journey:
    legs = [parse_leg(l) for l in raw["legs"]]
    # MOTIS ids are long opaque blobs; a short stable digest is easier to pass around.
    digest = hashlib.sha1(
        "|".join(f"{l.mode}{l.line}{l.departure.isoformat()}{l.origin.name}" for l in legs).encode()
    ).hexdigest()[:10]
    journey = Journey(
        id=digest,
        departure=_time(raw["startTime"]),
        arrival=_time(raw["endTime"]),
        duration_min=round(raw["duration"] / 60),
        transfers=raw.get("transfers", 0),
        legs=legs,
    )
    return dticket.annotate(journey)


class TransitousProvider:
    def __init__(self, client: httpx.AsyncClient | None = None):
        settings = get_settings()
        self._client = client or httpx.AsyncClient(
            base_url=settings.transit_base_url,
            headers={"User-Agent": settings.transit_user_agent},
            timeout=30.0,
        )
        self._cache: dict[tuple, tuple[float, object]] = {}
        self._sem = asyncio.Semaphore(MAX_CONCURRENCY)
        # journeys seen recently, so they can be referred to by id (chat -> map)
        self.journeys: dict[str, Journey] = {}

    async def _get(self, path: str, params: dict, throttled: bool = True) -> object:
        key = (path, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
            return hit[1]
        for delay in (*RETRY_DELAYS_S, None):
            if throttled:
                async with self._sem:
                    resp = await self._client.get(path, params=params)
            else:
                resp = await self._client.get(path, params=params)
            if resp.status_code not in (429, 503) or delay is None:
                break
            # wait outside the semaphore so one throttled request does not
            # hold up the others
            await asyncio.sleep(delay)
        resp.raise_for_status()
        data = resp.json()
        if len(self._cache) > 500:
            self._cache.clear()
        self._cache[key] = (time.monotonic(), data)
        return data

    async def search_stations(self, query: str, limit: int = 8) -> list[Station]:
        # Geocoding is not rate-limited upstream, and it backs the autocomplete,
        # so it must not queue behind journey requests.
        data = await self._get("/api/v1/geocode", {"text": query, "type": "STOP"}, throttled=False)
        rail = {"REGIONAL_RAIL", "REGIONAL_FAST_RAIL", "SUBURBAN", "HIGHSPEED_RAIL", "LONG_DISTANCE"}
        # The geocoder ranks by text match alone. Re-order (stably) so that
        # stops in Germany come first, and within each group railway stations
        # come before bus stops like "Marburg Markt".
        data = sorted(
            data,
            key=lambda r: ((r.get("country") or "").lower() != "de", not rail & set(r.get("modes", []))),
        )
        stations = [
            Station(
                id=r["id"],
                name=r["name"],
                lat=r["lat"],
                lon=r["lon"],
                country=(r.get("country") or "").lower() or None,
                # OSM admin level 4 is the Bundesland (and states/regions abroad)
                region=next((a["name"] for a in r.get("areas", []) if a.get("adminLevel") == 4), None),
            )
            for r in data
            if r.get("type") == "STOP"
        ]
        return stations[:limit]

    async def plan_journeys(
        self,
        origin: str,
        destination: str,
        when: datetime | None = None,
        arrive_by: bool = False,
        limit: int = 4,
    ) -> list[Journey]:
        params = {
            "fromPlace": origin,
            "toPlace": destination,
            "transitModes": ",".join(dticket.ALLOWED_TRANSIT_MODES),
            "numItineraries": limit,
            "arriveBy": "true" if arrive_by else "false",
        }
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=BERLIN)
            # truncate to the minute so near-identical requests share a cache entry
            params["time"] = (
                when.astimezone(timezone.utc).replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        data = await self._get("/api/v5/plan", params)
        journeys = [parse_itinerary(it) for it in data.get("itineraries", [])]
        journeys.sort(
            key=lambda j: (
                not j.dticket_valid,
                -j.departure.timestamp() if arrive_by else j.arrival.timestamp(),
            )
        )
        if len(self.journeys) > 2000:
            self.journeys.clear()
        for j in journeys:
            self.journeys[j.id] = j
        return journeys[:limit]

    async def travel_times(
        self, origin: tuple[float, float], places: list[tuple[float, float]], when: datetime
    ) -> list[tuple[int, int] | None]:
        """Fastest covered connection from origin to each place, in one request.

        Returns (minutes, transfers) per place, or None where nothing was found.
        Minutes count from `when`, so they include the wait for the first
        departure. Points are (lat, lon).
        """
        if when.tzinfo is None:
            when = when.replace(tzinfo=BERLIN)
        data = await self._get(
            "/api/experimental/one-to-many-intermodal",
            {
                "one": f"{origin[0]};{origin[1]}",
                "many": ",".join(f"{lat};{lon}" for lat, lon in places),
                "time": when.astimezone(timezone.utc).replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "transitModes": ",".join(dticket.ALLOWED_TRANSIT_MODES),
                "arriveBy": "false",
            },
        )
        out: list[tuple[int, int] | None] = []
        for options in data["transit_durations"]:
            if not options:
                out.append(None)
                continue
            best = min(options, key=lambda o: o["duration"])
            out.append((round(best["duration"] / 60), int(best.get("transfers", 0))))
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
