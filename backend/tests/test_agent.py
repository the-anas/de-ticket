import json
from datetime import datetime

from app.llm.agent import build_context, run_agent
from app.llm.base import TextDelta, ToolCall, TurnEnd
from app.llm.tools import ToolOutcome
from app.rag.ingest import chunk_markdown
from app.transit.transitous import BERLIN


class ScriptedLLM:
    """Replays fixed turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    async def stream(self, system, context, messages, tools):
        self.seen.append([dict(m) for m in messages])
        for event in self.turns.pop(0):
            yield event


class FakeTools:
    def __init__(self):
        self.calls = []

    async def run(self, name, args):
        self.calls.append((name, args))
        if name == "show_on_map":
            return ToolOutcome("shown", events=[{"type": "map", "journeys": [], "places": []}])
        if name == "broken":
            return ToolOutcome("boom", is_error=True)
        return ToolOutcome(json.dumps({"ok": True}))


async def collect(llm, tools, history):
    return [e async for e in run_agent(llm, tools, history)]


async def test_plain_answer():
    llm = ScriptedLLM([[TextDelta("Hallo"), TurnEnd("Hallo")]])
    events = await collect(llm, FakeTools(), [{"role": "user", "content": "Hi"}])
    assert events == [{"type": "text", "text": "Hallo"}, {"type": "done"}]


async def test_tool_loop_feeds_results_back_in_one_message():
    calls = [ToolCall("a", "find_journeys", {"origin": "Köln", "destination": "Bonn"}),
             ToolCall("b", "show_on_map", {"places": ["Bonn"]})]
    llm = ScriptedLLM([
        [TextDelta("Moment."), TurnEnd("Moment.", calls, "tool_use", raw=["opaque"])],
        [TextDelta("Fertig."), TurnEnd("Fertig.")],
    ])
    tools = FakeTools()
    events = await collect(llm, tools, [{"role": "user", "content": "Köln nach Bonn"}])

    assert [c[0] for c in tools.calls] == ["find_journeys", "show_on_map"]
    assert [e["type"] for e in events] == ["text", "tool", "tool", "map", "text", "text", "done"]
    second = llm.seen[1]
    assert second[1]["role"] == "assistant" and second[1]["raw"] == ["opaque"]
    assert second[2]["role"] == "tool"
    assert [r.id for r in second[2]["results"]] == ["a", "b"]


async def test_tool_error_is_reported_to_model_not_raised():
    llm = ScriptedLLM([
        [TurnEnd("", [ToolCall("x", "broken", {})], "tool_use")],
        [TextDelta("Sorry"), TurnEnd("Sorry")],
    ])
    events = await collect(llm, FakeTools(), [{"role": "user", "content": "?"}])
    assert llm.seen[1][-1]["results"][0].is_error
    assert events[-1] == {"type": "done"}


async def test_truncated_turn_does_not_run_tools():
    llm = ScriptedLLM([[TurnEnd("", [ToolCall("x", "find_journeys", {})], "max_tokens")]])
    tools = FakeTools()
    events = await collect(llm, tools, [{"role": "user", "content": "?"}])
    assert tools.calls == []
    assert events[0]["type"] == "error"


def test_context_states_time_and_origin():
    now = datetime(2026, 10, 12, 9, 30, tzinfo=BERLIN)
    ctx = build_context({"name": "Leipzig Hbf"}, now)
    assert "Monday, 2026-10-12 09:30" in ctx and "Leipzig Hbf" in ctx
    assert "not set" in build_context(None, now)


def test_chunk_markdown_splits_on_sections():
    chunks = chunk_markdown("x.md", "# Title\n\nintro\n\n## One\n\nbody one\n\n## Two\n\nbody two\n")
    assert [(c.title, c.section, c.text) for c in chunks] == [
        ("Title", "One", "body one"), ("Title", "Two", "body two"),
    ]


def test_provider_message_conversion():
    from app.llm.anthropic_provider import AnthropicProvider
    from app.llm.base import ToolResult
    from app.llm.openai_compat_provider import OpenAICompatProvider

    call = ToolCall("t1", "search_docs", {"query": "Sylt"})
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "and Sylt?"},
        {"role": "assistant", "content": "", "tool_calls": [call], "raw": ["thinking", "tool_use"]},
        {"role": "tool", "results": [ToolResult("t1", "search_docs", "[]"), ToolResult("t2", "x", "bad", True)]},
    ]

    a = AnthropicProvider._messages(history)
    assert a[1] == {"role": "assistant", "content": "earlier answer"}
    assert a[3]["content"] == ["thinking", "tool_use"]  # original blocks echoed back unchanged
    assert a[4]["role"] == "user" and [b["tool_use_id"] for b in a[4]["content"]] == ["t1", "t2"]
    assert a[4]["content"][1]["is_error"] is True

    o = OpenAICompatProvider._messages("SYS", "CTX", history)
    assert o[0] == {"role": "system", "content": "SYS\n\nCTX"}
    assert o[4]["content"] is None
    assert json.loads(o[4]["tool_calls"][0]["function"]["arguments"]) == {"query": "Sylt"}
    assert [m["tool_call_id"] for m in o[5:]] == ["t1", "t2"]
