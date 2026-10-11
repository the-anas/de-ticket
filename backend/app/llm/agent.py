import asyncio
from datetime import datetime
from typing import AsyncIterator

from ..transit.transitous import BERLIN
from .base import LLMError, LLMProvider, TextDelta, ToolResult, TurnEnd
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

How to work:
- For anything about connections, times or whether a route exists, call find_journeys rather than \
answering from memory; timetables change and the tool reads the live one. It only returns covered \
transport. If a journey comes back with dticket_valid false, do not present it as covered; say what \
the problem is.
- For questions about rules, where to go, or what can go wrong on a route, call search_docs and \
base the answer on what it returns. Mention the document section you relied on, in words, like \
"(Border stations: Austria)". If the documents do not answer the question, say so and give your \
best general knowledge, marked as such.
- When you recommend specific routes or places, call show_on_map so they appear on the map.
- If the user has set a start city, it is given below; use it as the origin unless they name another. \
If no origin is known and one is needed, ask for it.
- Relative dates ("Saturday", "tomorrow") refer to the current time given below. Times are German \
local time.

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


async def run_agent(
    llm: LLMProvider, tools: ToolRunner, history: list[dict], origin: dict | None = None
) -> AsyncIterator[dict]:
    """Runs the tool loop for one user message and yields browser events:
    text, tool, map, sources, error, done."""
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    context = build_context(origin)

    for _ in range(MAX_TURNS):
        turn: TurnEnd | None = None
        try:
            async for event in llm.stream(SYSTEM_PROMPT, context, messages, TOOLS):
                if isinstance(event, TextDelta):
                    yield {"type": "text", "text": event.text}
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
                yield extra
            results.append(ToolResult(call.id, call.name, outcome.content, outcome.is_error))
        # all results of a turn go back together in one message
        messages.append({"role": "tool", "results": results})
        # separate text written before the tool calls from what follows
        if turn.text:
            yield {"type": "text", "text": "\n\n"}
    else:
        yield {"type": "error", "message": "Stopped after too many tool calls without a final answer."}

    yield {"type": "done"}
