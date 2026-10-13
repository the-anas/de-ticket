from datetime import date

from evals.checks import (
    ToolUse, Trace, check_date, check_grounded_lines, check_grounded_times, check_real_queries,
    detect_language, grade, lines_in, resolve_day, times_in,
)

SUNDAY = date(2026, 10, 11)


def test_times_and_lines_are_extracted():
    assert times_in("Abfahrt 9:05, an 10:41. Preis 63.00 Euro, am 17.10.") == {"09:05", "10:41"}
    assert lines_in("Take the RE 5 or RB26, then S6 and the U2; IRE 200 also runs.") == {"RE5", "RB26", "S6", "U2", "IRE200"}
    assert lines_in("ICE 1557 and Saturday") == set()


def test_resolve_day():
    assert resolve_day("saturday", SUNDAY) == date(2026, 10, 17)
    assert resolve_day("sunday", SUNDAY) == SUNDAY
    assert resolve_day("tomorrow", SUNDAY) == date(2026, 10, 12)


def test_detect_language():
    assert detect_language("Der Zug fährt um 9 Uhr ab und ist nicht voll.") == "de"
    assert detect_language("The train leaves at 9 and is not crowded.") == "en"


def journey(args, result='{"journeys": [{"departure": "2026-10-17 09:05", "legs": [{"line": "RE5", "dep": "09:06", "arr": "10:20"}]}]}'):
    return ToolUse("find_journeys", args, result)


def test_grounding_flags_invented_times_and_lines():
    t = Trace(prompt="Be there by 12:00", answer="Take the RE5 at 09:06, arriving 10:20, well before 12:00. The RE1 at 09:30 also works.",
              tools=[journey({})])
    assert check_grounded_times(t) == (False, "not in any lookup: 09:30")
    assert check_grounded_lines(t) == (False, "not in any lookup: RE1")
    t.answer = "Take the RE5 at 09:06, arriving 10:20."
    assert check_grounded_times(t)[0] and check_grounded_lines(t)[0]


def test_date_check_requires_the_asked_day_on_every_search():
    ok = Trace(prompt="", tools=[journey({"datetime": "2026-10-17T09:00"})])
    assert check_date(ok, "saturday", SUNDAY) == (True, "")
    no_date = Trace(prompt="", tools=[journey({"datetime": "2026-10-17T09:00"}), journey({})])
    assert check_date(no_date, "saturday", SUNDAY)[0] is False
    assert "searched from now" in check_date(no_date, "saturday", SUNDAY)[1]
    assert check_date(Trace(prompt=""), "saturday", SUNDAY) == (False, "no timetable search")


def test_filename_queries_are_flagged():
    t = Trace(prompt="", tools=[ToolUse("search_docs", {"query": "destinations.md"}, "[]")])
    assert check_real_queries(t)[0] is False
    t.tools = [ToolUse("search_docs", {"query": "day trips along the Rhine"}, "[]")]
    assert check_real_queries(t)[0] is True


def test_grade_runs_all_requested_checks():
    t = Trace(prompt="What does it cost?", answer="It costs 63 euros a month.", sections=["What the ticket is and what it costs"],
              tools=[ToolUse("search_docs", {"query": "ticket price"}, "63 euros")])
    results = grade(t, {"calls": ["search_docs"], "docs_section": ["What the ticket is and what it costs"],
                        "must_contain_any": ["63"], "must_not_contain": ["49 euro"], "language": "en", "max_chars": 200}, SUNDAY)
    assert all(ok for ok, _ in results.values()) and "no_error" in results
    t.errors.append("rate limited")
    assert grade(t, {}, SUNDAY)["no_error"] == (False, "rate limited")


def test_cites_check():
    from evals.checks import check_cites

    t = Trace(prompt="", sections=["A", "B"], cited=["A"])
    assert check_cites(t, ["A"]) == (True, "")
    assert check_cites(t, ["B"]) == (False, "sources shown: A")
    t.cited = ["C"]
    assert "never retrieved" in check_cites(t, ["A"])[1]
