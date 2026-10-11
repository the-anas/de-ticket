"""Best-effort Deutschlandticket validity check for a planned journey.

The ticket covers local and regional transport in Germany (2nd class) plus a
short list of foreign border stations. It does not cover long-distance trains
or coaches. The rules here are heuristics over timetable data, not a fare
engine, so results are flagged rather than guaranteed.
"""

import json
import re
import unicodedata
from datetime import date
from functools import lru_cache

from ..config import DATA_DIR
from .geo import in_germany, km_outside_germany
from .models import Journey, Leg, Stop

# Modes requested from the router. Long-distance modes are left out so they are
# never offered in the first place.
ALLOWED_TRANSIT_MODES = [
    "REGIONAL_RAIL",
    "REGIONAL_FAST_RAIL",
    "SUBURBAN",
    "SUBWAY",
    "TRAM",
    "BUS",
    "FERRY",
]
FOOT_MODES = {"WALK"}
BLOCKED_MODES = {"HIGHSPEED_RAIL", "LONG_DISTANCE", "NIGHT_RAIL", "COACH", "AIRPLANE"}
CAUTION_MODES = {
    "FERRY": "Many ferries are excluded from the Deutschlandticket; check this one with the operator.",
    "FUNICULAR": "Funiculars and cable cars often need a separate ticket.",
    "CABLE_CAR": "Funiculars and cable cars often need a separate ticket.",
    "AERIAL_LIFT": "Funiculars and cable cars often need a separate ticket.",
}

LONG_DISTANCE_LINE = re.compile(
    r"^(ICE|IC|ECE|EC|RJX|RJ|NJ|EN|TGV|THA|EST|FLX|WB|D)\s?\d", re.IGNORECASE
)
EXCLUDED_OPERATORS = ("flix", "db fernverkehr", "westbahn", "european sleeper", "snälltåget")

_TRANSLIT = str.maketrans({"ø": "o", "ł": "l", "ß": "ss", "æ": "ae", "đ": "d"})


def normalize(name: str) -> str:
    name = name.lower().translate(_TRANSLIT)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name).strip()


def dhid_country(stop_id: str | None) -> str | None:
    """Country code from a DHID-style stop id such as "de-DELFI_de:14713:8010205".

    Only the part after the dataset tag is used: the same station can show up
    under a foreign or pan-European dataset (be-sncb, eu-flixbus), so the
    dataset prefix says nothing about where the stop is.
    """
    if not stop_id:
        return None
    m = re.match(r"^([a-z]{2}):", stop_id.partition("_")[2])
    return m.group(1) if m else None


# The border polygon is generalised (it is about 1 km off at Konstanz), so a
# stop without a DHID only counts as foreign when it is clearly outside.
BORDER_TOLERANCE_KM = 1.5


def is_abroad(stop: Stop) -> bool:
    country = dhid_country(stop.stop_id)
    if country is not None:
        return country != "de"
    return km_outside_germany(stop.lat, stop.lon) > BORDER_TOLERANCE_KM


@lru_cache
def _border_stations() -> list[dict]:
    data = json.loads((DATA_DIR / "border_stations.json").read_text(encoding="utf-8"))
    return data["stations"]


def border_station(name: str, on: date | None = None) -> dict | None:
    n = normalize(name)
    for entry in _border_stations():
        until = entry.get("valid_until")
        if until and on and on > date.fromisoformat(until):
            continue
        if n in entry.get("names", []):
            return entry
        if any(n.startswith(p) for p in entry.get("prefixes", [])):
            return entry
    return None


def _check_endpoint(stop: Stop, on: date) -> tuple[str | None, str | None]:
    """Returns (problem, note) for a boarding or alighting stop."""
    entry = border_station(stop.name, on)
    country = dhid_country(stop.stop_id)
    if entry and country != "de" and (country or not in_germany(stop.lat, stop.lon)):
        return None, f"{entry['label']} is abroad but covered: {entry['via']}."
    if is_abroad(stop):
        return f"{stop.name} is outside Germany and not a recognised Deutschlandticket border station.", None
    return None, None


def check_leg(leg: Leg) -> tuple[list[str], list[str]]:
    """Returns (problems, notes). Any problem makes the journey invalid."""
    problems: list[str] = []
    notes: list[str] = []
    if leg.mode in FOOT_MODES:
        return problems, notes

    label = leg.line or leg.mode
    if leg.mode in BLOCKED_MODES:
        problems.append(f"{label} is long-distance transport, which the Deutschlandticket does not cover.")
    elif leg.line and LONG_DISTANCE_LINE.match(leg.line.strip()):
        problems.append(f"{label} looks like a long-distance train, which the Deutschlandticket does not cover.")
    elif leg.operator and any(op in leg.operator.lower() for op in EXCLUDED_OPERATORS):
        problems.append(f"{label} is run by {leg.operator}, which does not accept the Deutschlandticket.")

    if leg.mode in CAUTION_MODES:
        notes.append(CAUTION_MODES[leg.mode])

    on = leg.departure.date()
    for stop in (leg.origin, leg.destination):
        problem, note = _check_endpoint(stop, on)
        if problem:
            problems.append(problem)
        if note:
            notes.append(note)

    # Riding through foreign territory between two valid points is a normal
    # corridor case (Schaffhausen, Außerfernbahn, Zittau–Varnsdorf); mention it.
    abroad = [s.name for s in leg.via if is_abroad(s) and border_station(s.name, on) is None]
    if abroad and not problems:
        notes.append(
            f"{label} runs through foreign territory ({abroad[0]}"
            f"{' and others' if len(abroad) > 1 else ''}); fine as through travel, "
            "but the ticket is not valid for getting on or off there."
        )
    return problems, notes


def annotate(journey: Journey) -> Journey:
    problems: list[str] = []
    notes: list[str] = []
    for leg in journey.legs:
        p, n = check_leg(leg)
        problems += p
        notes += n
    journey.dticket_valid = not problems
    # de-duplicate, keep order, problems first
    journey.dticket_notes = list(dict.fromkeys(problems + notes))
    return journey
