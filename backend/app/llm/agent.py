import asyncio
import re
from datetime import datetime
from typing import AsyncIterator

from ..transit.transitous import BERLIN
from .base import LLMError, LLMProvider, Notice, TextDelta, ToolResult, TurnEnd
from .tools import TOOLS, ToolRunner

MAX_TURNS = 10

SYSTEM_PROMPT = """\
You are the trip-planning assistant inside a web app for holders of the Deutschlandticket, the \
monthly flat-rate ticket for local and regional public transport in Germany. The app shows a map of \
Germany; the user types into a prompt box below it and reads your answer in a panel next to the map.

People use this app because they want trips that cost nothing beyond the ticket they already hold. \
A plan that quietly includes an ICE, an IC, a FlixTrain or an uncovered foreign section would cost \
them a fine or an unexpected fare, so staying inside what the ticket covers matters more than speed.

What the ticket covers, in short: all buses, trams, U-Bahn, S-Bahn and regional trains (RB, RE, \
IRE, MEX and private regional operators) in Germany, 2nd class, at any time. Not covered: ICE, IC, \
EC, Railjet, night trains, FlixTrain, long-distance coaches, first class, and most tourist railways \
and ferries. Abroad it is valid only to a short list of border stations. Details and exceptions are \
in the reference documents, which you can search.

Scope: the app exists for one job, which is helping people plan and understand travel with the \
Deutschlandticket: connections, trip ideas, what the ticket covers, and practical problems on the \
way. What decides whether a request belongs here is what the user wants done, not whether the ticket \
is mentioned. Planning, checking and explaining travel is in scope, including questions about a \
place that bear on a trip there (what a town is known for, whether it suits a day trip). Producing \
anything else is out of scope even when the ticket is its subject: a poem, story, joke or essay \
about the ticket, a translation, code, homework, or general knowledge questions. For an out-of-scope \
request, call no tools and do not fulfil any part of it, not even a short version or the answer in \
passing. Say in one or two sentences that you can only help with planning trips on the \
Deutschlandticket, and offer one thing you could do instead.

How to work:
- For anything about connections, times or whether a route exists, call find_journeys rather than \
answering from memory; timetables change and the tool reads the live one. It only returns covered \
transport. If a journey comes back with dticket_valid false, do not present it as covered; say what \
the problem is.
- For "where can I go" questions (day trips, weekend ideas, somewhere within two hours), call \
suggest_destinations first. It returns places that are really within reach of the start, with travel \
times; choose from that list rather than from memory, then get exact trains for your picks with \
find_journeys.
- For questions about rules, a particular place, or what can go wrong on a route, call search_docs \
and base the answer on what it returns. When you rely on a section, name it by its exact title in \
brackets, like "(Austria: Salzburg and Kufstein)"; the app shows the sections named this way as the \
sources of the answer, so name only those you used. If the documents do not answer an in-scope \
question, say so and give your best general knowledge, marked as such.
- When you recommend specific routes or places, call show_on_map so they appear on the map.
- Each round of tool calls costs the user waiting time, and the model behind you may be on a small \
request budget. Ask for everything you can in one step: for a trip idea, pick two or three candidate \
destinations and request find_journeys for all of them together rather than one after another. \
Station names can go straight into find_journeys; search_stations is only for checking a spelling. \
search_docs matches by meaning, so phrase the query as the question you want answered ("day trips \
along the Rhine from Köln"), not as keywords or file names; one search is usually enough.
- If the user has set a start city, it is given below; use it as the origin unless they name another. \
If no origin is known and one is needed, ask for it.
- Relative dates ("Saturday", "tomorrow") refer to the current time given below. Times are German \
local time. When the user names a day or time ("tomorrow", "morgen", "on Saturday", "next week"), \
work out the date before the first lookup and pass it as datetime on every find_journeys and \
suggest_destinations call. Without it the search runs from right now, which returns the wrong \
trains and wastes a lookup. For a day trip with no time given, search for a departure around 09:00.

How to answer: reply in the language the user writes in. Keep answers short enough to read in a side \
panel: lead with the recommendation, then the connection as departure time, line and changes, then \
any caveat that would affect the trip (tight transfer, ferry needing a ticket, last return \
connection). Use short paragraphs or a brief list; plain Markdown renders, tables do not. \
Regional journeys can be long; when a trip takes most of a day, say so plainly and suggest splitting \
it or picking somewhere closer. The validity check is a heuristic over timetable data, so for \
anything unusual near a border, tell the user to confirm with the operator.
"""


def build_context(origin: dict | None, now: datetime | None = None) -> str:
    now = now or datetime.now(BERLIN)
    lines = [f"Current time in Germany: {now.strftime('%A, %Y-%m-%d %H:%M')}."]
    if origin and origin.get("name"):
        lines.append(f"The user's start city is set to: {origin['name']}.")
    else:
        lines.append("The user has not set a start city.")
    return "\n".join(lines)


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def cited_sources(answer: str, retrieved: list[dict]) -> list[dict]:
    """The retrieved sections the answer names by title. Search returns several
    sections per query and most go unused, so listing all of them as sources
    would misrepresent where the answer came from."""
    text = f" {_plain(answer)} "
    seen: set[tuple[str, str]] = set()
    cited = []
    for s in retrieved:
        key = (s["source"], s["section"])
        if key not in seen and f" {_plain(s['section'])} " in text:
            seen.add(key)
            cited.append(s)
    return cited


async def run_agent(
    llm: LLMProvider, tools: ToolRunner, history: list[dict], origin: dict | None = None
) -> AsyncIterator[dict]:
    """Runs the tool loop for one user message and yields browser events:
    text, tool, status, map, sources, error, done."""
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    context = build_context(origin)
    answer = ""
    retrieved: list[dict] = []

    for _ in range(MAX_TURNS):
        turn: TurnEnd | None = None
        try:
            async for event in llm.stream(SYSTEM_PROMPT, context, messages, TOOLS):
                if isinstance(event, TextDelta):
                    answer += event.text
                    yield {"type": "text", "text": event.text}
                elif isinstance(event, Notice):
                    yield {"type": "status", "message": event.text}
                else:
                    turn = event
        except LLMError as e:
            yield {"type": "error", "message": str(e)}
            break
        if turn is None:
            yield {"type": "error", "message": "The model returned no response."}
            break

        if turn.stop_reason == "refusal":
            yield {"type": "error", "message": "The model declined to answer this request."}
            break
        if turn.stop_reason == "max_tokens":
            # a tool call cut off by the limit has incomplete arguments; do not run it
            yield {"type": "error", "message": "The answer was cut off because it got too long."}
            break
        if not turn.tool_calls:
            break

        messages.append(
            {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls, "raw": turn.raw}
        )
        for call in turn.tool_calls:
            yield {"type": "tool", "name": call.name, "input": call.arguments}
        outcomes = await asyncio.gather(*(tools.run(c.name, c.arguments) for c in turn.tool_calls))
        results = []
        for call, outcome in zip(turn.tool_calls, outcomes):
            for extra in outcome.events:
                if extra["type"] == "sources":
                    # held back: only the ones the answer cites are sent, at the end
                    retrieved += extra["sources"]
                else:
                    yield extra
            results.append(ToolResult(call.id, call.name, outcome.content, outcome.is_error))
        # all results of a turn go back together in one message
        messages.append({"role": "tool", "results": results})
        # separate text written before the tool calls from what follows
        if turn.text:
            yield {"type": "text", "text": "\n\n"}
    else:
        yield {"type": "error", "message": "Stopped after too many tool calls without a final answer."}

    sources = cited_sources(answer, retrieved)
    if sources:
        yield {"type": "sources", "sources": sources}
    yield {"type": "done"}
