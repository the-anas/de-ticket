"""Tools the assistant can call, and their handlers."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime

from ..destinations import reachable_destinations
from ..rag.store import get_store
from ..transit.models import Journey, Station
from ..transit.transitous import BERLIN, TransitousProvider
from .base import ToolSpec

TOOLS = [
    ToolSpec(
        name="search_stations",
        description=(
            "Look up public transport stations by name. Use it to check that a place has a station "
            "and to get its exact name before planning. Returns up to 6 matches, railway stations first."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Station or town name, e.g. 'Bamberg'"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="suggest_destinations",
        description=(
            "List places worth visiting that can be reached from a starting point with the Deutschlandticket, "
            "nearest by travel time first. Use this first for any 'where can I go', day-trip or weekend-trip "
            "question; it knows which destinations are actually within reach, which the documents do not. "
            "Each result has the destination, its region, what it is known for, and the approximate travel "
            "time in minutes counted from the search time (so it includes waiting for the first train). "
            "Then call find_journeys for the ones you recommend to get exact trains."
        ),
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Station or town to start from, e.g. 'Köln Hbf'"},
                "max_minutes": {
                    "type": "integer",
                    "description": "Only places reachable within this many minutes. About 150 suits a day trip.",
                },
                "datetime": {
                    "type": "string",
                    "description": "Local German departure time, ISO format like 2026-10-17T09:00. Defaults to now.",
                },
            },
            "required": ["origin"],
        },
    ),
    ToolSpec(
        name="find_journeys",
        description=(
            "Plan connections between two places using only transport covered by the Deutschlandticket "
            "(regional trains, S-Bahn, U-Bahn, trams, buses). Long-distance trains are never offered. "
            "Returns up to 4 journeys from the live timetable with times, lines, transfers, an id, and a "
            "validity check; journeys with dticket_valid=false must not be recommended as covered. "
            "Call it once per origin/destination pair; for a round trip, call it again for the return."
        ),
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Station or town name, e.g. 'Köln Hbf'"},
                "destination": {"type": "string", "description": "Station or town name"},
                "datetime": {
                    "type": "string",
                    "description": "Local German time, ISO format like 2026-10-17T09:00. Defaults to now.",
                },
                "arrive_by": {
                    "type": "boolean",
                    "description": "If true, datetime is the latest arrival instead of the earliest departure.",
                },
            },
            "required": ["origin", "destination"],
        },
    ),
    ToolSpec(
        name="search_docs",
        description=(
            "Search the app's reference documents: Deutschlandticket rules and exclusions, border stations "
            "abroad, destination ideas by region, known problem routes and practical tips. Use it for any "
            "question about what the ticket covers, where to go, or what can go wrong, and cite the "
            "returned source and section in the answer."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for, in German or English"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="show_on_map",
        description=(
            "Draw journeys and/or places on the user's map. Pass journey ids returned by find_journeys to "
            "draw routes, and place names to drop markers. Call it whenever the answer recommends specific "
            "routes or destinations, so the user sees them. It replaces what the assistant drew before."
        ),
        parameters={
            "type": "object",
            "properties": {
                "journey_ids": {"type": "array", "items": {"type": "string"}},
                "places": {"type": "array", "items": {"type": "string"}, "description": "Station or town names"},
            },
        },
    ),
]


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    # extra events to forward to the browser (map updates, cited sources)
    events: list[dict] = field(default_factory=list)


def _require(args: dict, *names: str) -> None:
    for name in names:
        if not isinstance(args.get(name), str) or not args[name].strip():
            raise ValueError(f"'{name}' is required and must be a non-empty string")


def summarize_journey(j: Journey) -> dict:
    """Compact form for the model: no geometry, local clock times."""
    return {
        "id": j.id,
        "departure": j.departure.strftime("%Y-%m-%d %H:%M"),
        "arrival": j.arrival.strftime("%Y-%m-%d %H:%M"),
        "duration_min": j.duration_min,
        "transfers": j.transfers,
        "dticket_valid": j.dticket_valid,
        "notes": j.dticket_notes,
        "legs": [
            {
                "line": l.line or l.mode.replace("_", " ").title(),
                "towards": l.headsign,
                "from": l.origin.name,
                "dep": l.departure.strftime("%H:%M"),
                "to": l.destination.name,
                "arr": l.arrival.strftime("%H:%M"),
                **({"platform": l.origin.track} if l.origin.track else {}),
            }
            for l in j.legs
            if l.mode != "WALK"
        ],
    }


class ToolRunner:
    def __init__(self, transit: TransitousProvider):
        self._transit = transit

    async def _resolve(self, name: str) -> Station:
        hits = await self._transit.search_stations(name, limit=1)
        if not hits:
            raise ValueError(f"No station found for '{name}'. Try search_stations with a different spelling.")
        return hits[0]

    async def run(self, name: str, args: dict) -> ToolOutcome:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return ToolOutcome(f"Unknown tool '{name}'.", is_error=True)
        if "_invalid_json" in args:
            return ToolOutcome("The tool arguments were not valid JSON; send them again.", is_error=True)
        try:
            return await handler(args)
        except ValueError as e:
            return ToolOutcome(str(e), is_error=True)
        except Exception as e:  # network or upstream failure: let the model tell the user
            return ToolOutcome(f"{name} failed: {type(e).__name__}: {e}", is_error=True)

    async def _search_stations(self, args: dict) -> ToolOutcome:
        _require(args, "query")
        hits = await self._transit.search_stations(args["query"], limit=6)
        return ToolOutcome(
            json.dumps([{"name": s.name, "country": s.country} for s in hits], ensure_ascii=False)
        )

    @staticmethod
    def _when(args: dict) -> datetime | None:
        if not args.get("datetime"):
            return None
        try:
            when = datetime.fromisoformat(args["datetime"])
        except (TypeError, ValueError):
            raise ValueError("datetime must be ISO format, e.g. 2026-10-17T09:00") from None
        return when if when.tzinfo else when.replace(tzinfo=BERLIN)

    async def _suggest_destinations(self, args: dict) -> ToolOutcome:
        _require(args, "origin")
        limit = args.get("max_minutes")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError("max_minutes must be a positive whole number")
        origin = await self._resolve(args["origin"])
        routes = await reachable_destinations(
            self._transit, origin.lat, origin.lon, origin.name, origin.id, self._when(args)
        )
        within = [r for r in routes if limit is None or r.approx_min <= limit]
        result: dict = {
            "origin": origin.name,
            "destinations": [
                {
                    "name": r.destination.name,
                    "station": r.destination.station,
                    "region": r.destination.region,
                    "known_for": r.destination.blurb,
                    "tags": r.destination.tags,
                    "approx_minutes": r.approx_min,
                    "transfers": r.transfers,
                }
                for r in within[:12]
            ],
        }
        if not routes:
            result["note"] = "Nothing on the curated list is reachable with the Deutschlandticket from here."
        elif not within:
            nearest = routes[0]
            result["note"] = (
                f"Nothing within {limit} minutes. The nearest is {nearest.destination.name} "
                f"at about {nearest.approx_min} minutes."
            )
        return ToolOutcome(json.dumps(result, ensure_ascii=False))

    async def _find_journeys(self, args: dict) -> ToolOutcome:
        _require(args, "origin", "destination")
        when = self._when(args)
        origin = await self._resolve(args["origin"])
        destination = await self._resolve(args["destination"])
        journeys = await self._transit.plan_journeys(
            f"{origin.lat},{origin.lon}",
            f"{destination.lat},{destination.lon}",
            when,
            arrive_by=bool(args.get("arrive_by")),
        )
        result = {
            "origin": origin.name,
            "destination": destination.name,
            "journeys": [summarize_journey(j) for j in journeys],
        }
        if not journeys:
            result["note"] = "No regional connection found for this time. Try another time or a nearby station."
        return ToolOutcome(json.dumps(result, ensure_ascii=False))

    async def _search_docs(self, args: dict) -> ToolOutcome:
        _require(args, "query")
        # off the event loop: the reranker takes long enough to stall other requests
        hits = await asyncio.to_thread(lambda: get_store().search(args["query"], k=5))
        results = [
            {"source": h.chunk.source, "document": h.chunk.title, "section": h.chunk.section, "text": h.chunk.text}
            for h in hits
        ]
        sources = [{"source": r["source"], "document": r["document"], "section": r["section"]} for r in results]
        return ToolOutcome(
            json.dumps(results, ensure_ascii=False), events=[{"type": "sources", "sources": sources}]
        )

    async def _show_on_map(self, args: dict) -> ToolOutcome:
        ids = args.get("journey_ids") or []
        names = args.get("places") or []
        if not isinstance(ids, list) or not isinstance(names, list):
            raise ValueError("journey_ids and places must be arrays")
        if not ids and not names:
            raise ValueError("Pass at least one journey id or place name")
        journeys = [self._transit.journeys[i] for i in ids if i in self._transit.journeys]
        missing = [i for i in ids if i not in self._transit.journeys]
        places = []
        for name in names:
            try:
                places.append(await self._resolve(str(name)))
            except ValueError:
                missing.append(str(name))
        event = {
            "type": "map",
            "journeys": [j.model_dump(mode="json") for j in journeys],
            "places": [p.model_dump(mode="json") for p in places],
        }
        shown = f"Showing {len(journeys)} journey(s) and {len(places)} place(s) on the map."
        if missing:
            shown += f" Not found: {', '.join(missing)}."
        return ToolOutcome(shown, events=[event] if journeys or places else [])
