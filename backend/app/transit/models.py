from datetime import datetime

from pydantic import BaseModel


class Station(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    country: str | None = None


class Stop(BaseModel):
    name: str
    lat: float
    lon: float
    stop_id: str | None = None
    time: datetime | None = None
    track: str | None = None


class Leg(BaseModel):
    mode: str
    line: str | None = None
    operator: str | None = None
    headsign: str | None = None
    origin: Stop
    destination: Stop
    departure: datetime
    arrival: datetime
    duration_min: int
    color: str | None = None
    # [lon, lat] pairs, GeoJSON order
    geometry: list[tuple[float, float]] = []
    # stop ids/names passed through, used for the validity check only
    via: list[Stop] = []


class Journey(BaseModel):
    id: str
    departure: datetime
    arrival: datetime
    duration_min: int
    transfers: int
    legs: list[Leg]
    dticket_valid: bool = True
    dticket_notes: list[str] = []
