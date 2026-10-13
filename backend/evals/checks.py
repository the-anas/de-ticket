"""Deterministic checks on one assistant run. No model is used to grade.

Each check returns (passed, detail). `Trace` is what the eval runner recorded:
the answer text, every tool call with its arguments and result, and the
document sections that were retrieved.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

TIME = re.compile(r"(?<![\d:.])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
# regional line names as they appear in timetables: RE5, RB 26, S6, IRE 200, U2
LINE = re.compile(r"\b(IRE|RE|RB|MEX|S|U)\s?(\d{1,3}[a-zA-Z]?)\b")
FILENAME = re.compile(r"\.(md|json|txt|pdf)\b", re.IGNORECASE)

GERMAN_WORDS = {"der", "die", "das", "und", "ist", "nicht", "mit", "du", "sie", "nach", "von", "für", "ein", "eine", "auf", "auch", "kannst", "fährt", "oder", "wird", "zum", "zur", "dem", "den", "im", "ab", "um", "uhr"}
ENGLISH_WORDS = {"the", "and", "is", "not", "with", "you", "to", "from", "for", "a", "an", "on", "also", "can", "or", "of", "it", "this", "that", "at", "are", "your", "by", "will"}


@dataclass
class ToolUse:
    name: str
    args: dict
    result: str
    is_error: bool = False


@dataclass
class Trace:
    prompt: str
    answer: str = ""
    tools: list[ToolUse] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)  # returned by document search
    cited: list[str] = field(default_factory=list)  # shown to the user as sources
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    model_turns: int = 0


def times_in(text: str) -> set[str]:
    return {f"{int(h):02d}:{m}" for h, m in TIME.findall(text)}


def lines_in(text: str) -> set[str]:
    return {f"{kind}{number}".upper() for kind, number in LINE.findall(text)}


def resolve_day(name: str, today: date) -> date:
    if name == "tomorrow":
        return today + timedelta(days=1)
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    # the next such weekday; "Saturday" said on a Saturday means today
    return today + timedelta(days=(weekdays.index(name) - today.weekday()) % 7)


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    german = sum(w in GERMAN_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    return "de" if german > english else "en"


def _calls(trace: Trace, name: str) -> list[ToolUse]:
    return [t for t in trace.tools if t.name == name]


def check_no_error(trace: Trace, _=None):
    return (not trace.errors, "; ".join(trace.errors))


def check_calls(trace: Trace, names: list[str]):
    missing = [n for n in names if not _calls(trace, n)]
    return (not missing, f"never called: {', '.join(missing)}" if missing else "")


def check_must_not_call(trace: Trace, names: list[str]):
    called = [n for n in names if _calls(trace, n)]
    return (not called, f"called: {', '.join(called)}" if called else "")


def check_date(trace: Trace, day: str, today: date | None = None):
    """Every timetable search must be for the day the user asked about."""
    want = resolve_day(day, today or date.today()).isoformat()
    searches = _calls(trace, "find_journeys")
    if not searches:
        return (False, "no timetable search")
    wrong = [str(t.args.get("datetime") or "no date (searched from now)") for t in searches
             if not str(t.args.get("datetime") or "").startswith(want)]
    return (not wrong, f"wanted {want}, searched: {', '.join(wrong)}" if wrong else "")


def check_arrive_by(trace: Trace, _=None):
    ok = any(t.args.get("arrive_by") is True for t in _calls(trace, "find_journeys"))
    return (ok, "" if ok else "searched by departure although the user gave an arrival deadline")


def check_quotes_times(trace: Trace, _=None):
    ok = bool(times_in(trace.answer))
    return (ok, "" if ok else "answer gives no departure or arrival time")


def check_grounded_times(trace: Trace, _=None):
    """Clock times in the answer must come from a tool result or the question."""
    known = times_in(trace.prompt)
    for t in trace.tools:
        known |= times_in(t.result)
    invented = sorted(times_in(trace.answer) - known)
    return (not invented, f"not in any lookup: {', '.join(invented)}" if invented else "")


def check_grounded_lines(trace: Trace, _=None):
    """Train lines named in the answer must come from a tool result or the question."""
    known = lines_in(trace.prompt)
    for t in trace.tools:
        known |= lines_in(t.result)
    invented = sorted(lines_in(trace.answer) - known)
    return (not invented, f"not in any lookup: {', '.join(invented)}" if invented else "")


def check_docs_section(trace: Trace, expected: list[str]):
    ok = any(s in trace.sections for s in expected)
    if ok:
        return (True, "")
    if not _calls(trace, "search_docs"):
        return (False, "documents were not searched")
    return (False, f"search did not return: {' / '.join(expected)}")


def check_cites(trace: Trace, expected: list[str]):
    """The sources shown to the user must include a section that holds the
    answer, and nothing that was not actually retrieved."""
    stray = [s for s in trace.cited if s not in trace.sections]
    if stray:
        return (False, f"shown as a source but never retrieved: {', '.join(stray)}")
    ok = any(s in trace.cited for s in expected)
    shown = ", ".join(trace.cited) or "none"
    return (ok, "" if ok else f"sources shown: {shown}")


def check_real_queries(trace: Trace, _=None):
    """Document searches should be questions or phrases, not file names."""
    bad = [str(t.args.get("query")) for t in _calls(trace, "search_docs") if FILENAME.search(str(t.args.get("query", "")))]
    return (not bad, f"searched for: {', '.join(bad)}" if bad else "")


def check_must_contain_any(trace: Trace, needles: list[str]):
    ok = any(n.lower() in trace.answer.lower() for n in needles)
    return (ok, "" if ok else f"answer mentions none of: {', '.join(needles)}")


def check_must_not_contain(trace: Trace, needles: list[str]):
    found = [n for n in needles if n.lower() in trace.answer.lower()]
    return (not found, f"answer contains: {', '.join(found)}" if found else "")


def check_language(trace: Trace, lang: str):
    got = detect_language(trace.answer)
    return (got == lang, "" if got == lang else f"answered in {got}, asked in {lang}")


def check_max_chars(trace: Trace, limit: int):
    n = len(trace.answer)
    return (0 < n <= limit, "" if 0 < n <= limit else ("empty answer" if n == 0 else f"{n} characters, limit {limit}"))


CHECKS = {
    "calls": check_calls,
    "must_not_call": check_must_not_call,
    "date": check_date,
    "arrive_by": check_arrive_by,
    "quotes_times": check_quotes_times,
    "grounded_times": check_grounded_times,
    "grounded_lines": check_grounded_lines,
    "docs_section": check_docs_section,
    "cites": check_cites,
    "real_queries": check_real_queries,
    "must_contain_any": check_must_contain_any,
    "must_not_contain": check_must_not_contain,
    "language": check_language,
    "max_chars": check_max_chars,
}

# what each check means, for the report
DESCRIPTIONS = {
    "no_error": "Run finished without an error",
    "calls": "Used the tools the question needs",
    "must_not_call": "Did not use tools when it should have asked first or declined",
    "date": "Searched the timetable for the day asked about",
    "arrive_by": "Searched by arrival time when given a deadline",
    "quotes_times": "Gave actual departure times",
    "grounded_times": "Every time quoted came from a lookup",
    "grounded_lines": "Every train line named came from a lookup",
    "docs_section": "Document search returned the right section",
    "cites": "Sources shown to the user include the section that holds the answer",
    "real_queries": "Searched documents with real questions, not file names",
    "must_contain_any": "Answer contains the key fact",
    "must_not_contain": "Answer avoids a known wrong statement",
    "language": "Answered in the user's language",
    "max_chars": "Answer short enough for the chat panel",
}


def grade(trace: Trace, checks: dict, today: date | None = None) -> dict[str, tuple[bool, str]]:
    results = {"no_error": check_no_error(trace)}
    for name, arg in checks.items():
        fn = CHECKS[name]
        results[name] = fn(trace, arg, today) if name == "date" else fn(trace, arg)
    return results


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)
