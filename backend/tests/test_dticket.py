import json
from datetime import datetime
from pathlib import Path

from app.transit import dticket
from app.transit.geo import in_germany
from app.transit.models import Leg, Stop
from app.transit.transitous import BERLIN, decode_polyline, parse_itinerary

FIXTURES = Path(__file__).parent / "fixtures"

LEIPZIG = Stop(name="Leipzig Hbf", lat=51.3455, lon=12.3821, stop_id="de-DELFI_de:14713:8010205")
DRESDEN = Stop(name="Dresden Hbf", lat=51.0404, lon=13.7320, stop_id="de-DELFI_de:14612:28")
FREILASSING = Stop(name="Freilassing", lat=47.8370, lon=12.9770, stop_id="de-DELFI_de:09172:42293")
SALZBURG = Stop(name="Salzburg Hbf", lat=47.8131, lon=13.0451, stop_id="at-Railway_Pat:45:50002")
WIEN = Stop(name="Wien Hbf", lat=48.1853, lon=16.3762, stop_id="at-Railway_Pat:49:1349")
# same German station, listed under a Belgian dataset
KOBLENZ = Stop(name="Koblenz Hauptbahnhof", lat=50.3508, lon=7.5886, stop_id="be-sncb_8019023")


def leg(origin, destination, mode="REGIONAL_RAIL", line="RE50", operator="DB Regio", via=()):
    t = datetime(2026, 10, 12, 9, 0, tzinfo=BERLIN)
    return Leg(
        mode=mode, line=line, operator=operator, origin=origin, destination=destination,
        departure=t, arrival=t, duration_min=60, via=list(via),
    )


def test_regional_train_is_valid():
    assert dticket.check_leg(leg(LEIPZIG, DRESDEN)) == ([], [])


def test_long_distance_mode_rejected():
    problems, _ = dticket.check_leg(leg(LEIPZIG, DRESDEN, mode="HIGHSPEED_RAIL", line="ICE 1557"))
    assert problems


def test_long_distance_line_name_rejected_even_if_mode_is_regional():
    for line in ["IC 2223", "ICE1557", "EC 8", "FLX 35", "RJX 62", "NJ 40420"]:
        problems, _ = dticket.check_leg(leg(LEIPZIG, DRESDEN, line=line))
        assert problems, line


def test_regional_line_names_not_mistaken_for_long_distance():
    for line in ["RE50", "IRE 200", "S 1", "RB 26", "ICB", "MEX 16", "U2", "E 12", "Bus 100", "D"]:
        problems, _ = dticket.check_leg(leg(LEIPZIG, DRESDEN, line=line))
        assert not problems, line


def test_flixtrain_operator_rejected():
    problems, _ = dticket.check_leg(leg(LEIPZIG, DRESDEN, line="20", operator="FlixTrain GmbH"))
    assert problems


def test_border_station_is_valid_with_note():
    problems, notes = dticket.check_leg(leg(FREILASSING, SALZBURG, line="S3"))
    assert problems == []
    assert any("Salzburg" in n for n in notes)


def test_foreign_station_not_on_list_rejected():
    problems, _ = dticket.check_leg(leg(SALZBURG, WIEN, line="REX 1"))
    assert any("Wien" in p for p in problems)


def test_german_station_under_foreign_dataset_is_not_abroad():
    assert not dticket.is_abroad(KOBLENZ)
    assert dticket.check_leg(leg(KOBLENZ, LEIPZIG)) == ([], [])


def test_through_travel_abroad_is_noted_not_rejected():
    reutte = Stop(name="Reutte in Tirol", lat=47.4900, lon=10.7190, stop_id="at-Railway_Pat:47:999")
    kempten = Stop(name="Kempten (Allgäu) Hbf", lat=47.7130, lon=10.3150, stop_id="de-DELFI_de:09763:1")
    garmisch = Stop(name="Garmisch-Partenkirchen", lat=47.4910, lon=11.0970, stop_id="de-DELFI_de:09180:42837")
    problems, notes = dticket.check_leg(leg(kempten, garmisch, line="RB 6", via=[reutte]))
    assert problems == []
    assert any("Reutte" in n for n in notes)


def test_border_entry_expires():
    lux = Stop(name="Luxembourg, Gare Centrale", lat=49.6000, lon=6.1340, stop_id="lu-x_1")
    assert dticket.border_station(lux.name, datetime(2026, 11, 1).date())
    assert dticket.border_station(lux.name, datetime(2027, 1, 2).date()) is None


def test_normalize_handles_non_decomposable_letters():
    assert dticket.normalize("Tønder") == "tonder"
    assert dticket.normalize("Świnoujście  Centrum") == "swinoujscie centrum"


def test_in_germany():
    assert in_germany(52.52, 13.40)  # Berlin
    assert in_germany(54.9070, 8.3100)  # Westerland, on an island polygon
    assert not in_germany(47.8131, 13.0451)  # Salzburg Hbf
    assert not in_germany(47.5673, 7.6077)  # Basel Bad Bf
    assert not in_germany(51.3640, 6.1720)  # Venlo


def test_station_near_border_without_dhid_is_not_flagged():
    # the border polygon puts Konstanz station about 1 km outside Germany
    konstanz = Stop(name="Konstanz", lat=47.6596, lon=9.1770, stop_id="eu-flixbus_abc")
    kreuzlingen = Stop(name="Kreuzlingen", lat=47.6500, lon=9.1740, stop_id="ch-sbb_8506131")
    assert not dticket.is_abroad(konstanz)
    assert dticket.is_abroad(kreuzlingen)


def test_decode_polyline():
    # reference example from the polyline format documentation (precision 5)
    assert decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@", 5) == [
        (-120.2, 38.5), (-120.95, 40.7), (-126.453, 43.252),
    ]


def test_parse_recorded_plan():
    raw = json.loads((FIXTURES / "plan_leipzig_dresden.json").read_text())
    journey = parse_itinerary(raw["itineraries"][0])
    assert journey.dticket_valid
    assert journey.duration_min == 97
    train = next(l for l in journey.legs if l.mode != "WALK")
    assert train.line == "RE50"
    assert train.origin.name == "Leipzig Hbf"
    assert train.departure.utcoffset() is not None
    assert len(train.geometry) > 10
    lon, lat = train.geometry[0]
    assert 12 < lon < 13 and 51 < lat < 52
    assert len(journey.id) == 10


def test_default_departure_avoids_overnight_searches():
    from app.transit.when import default_departure

    assert default_departure(datetime(2026, 10, 12, 13, 5, tzinfo=BERLIN)) == datetime(2026, 10, 12, 13, 5, tzinfo=BERLIN)
    # rounded up to five minutes so nearby requests share a cache entry
    assert default_departure(datetime(2026, 10, 12, 13, 6, 30, tzinfo=BERLIN)) == datetime(2026, 10, 12, 13, 10, tzinfo=BERLIN)
    assert default_departure(datetime(2026, 10, 12, 13, 58, tzinfo=BERLIN)) == datetime(2026, 10, 12, 14, 0, tzinfo=BERLIN)
    assert default_departure(datetime(2026, 10, 12, 22, 0, tzinfo=BERLIN)) == datetime(2026, 10, 13, 8, 0, tzinfo=BERLIN)
    assert default_departure(datetime(2026, 10, 12, 3, 0, tzinfo=BERLIN)) == datetime(2026, 10, 12, 8, 0, tzinfo=BERLIN)


def test_check_endpoint_for_start_locations():
    paris = Stop(name="Paris Gare de Lyon", lat=48.8443, lon=2.3743, stop_id="fr-sncf_123")
    assert dticket.check_endpoint(paris)[0]
    assert dticket.check_endpoint(SALZBURG) == (None, dticket.check_endpoint(SALZBURG)[1])
    assert dticket.check_endpoint(SALZBURG)[0] is None
    # a start picked on the map has no station id, only a name and coordinates
    assert dticket.check_endpoint(Stop(name="Leipzig Hbf", lat=51.3455, lon=12.3821)) == (None, None)
