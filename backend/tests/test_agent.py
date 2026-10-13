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


# ---------- Gemini adapter ----------

from google.genai import types as gtypes  # noqa: E402

from app.llm.base import LLMError, ToolResult  # noqa: E402
from app.llm.gemini_provider import SYNTHETIC_ID, describe_error, read_stream, retry_delay_s, to_contents  # noqa: E402


def g_chunk(*parts, finish=None, block=None):
    return gtypes.GenerateContentResponse(
        candidates=[gtypes.Candidate(content=gtypes.Content(role="model", parts=list(parts)), finish_reason=finish)],
        prompt_feedback=gtypes.GenerateContentResponsePromptFeedback(block_reason=block) if block else None,
    )


async def g_read(*chunks):
    async def gen():
        for c in chunks:
            yield c

    return [e async for e in read_stream(gen())]


def g_call(name, args, call_id=None, signature=None):
    return gtypes.Part(function_call=gtypes.FunctionCall(id=call_id, name=name, args=args), thought_signature=signature)


async def test_gemini_streams_text():
    events = await g_read(g_chunk(gtypes.Part(text="Hal")), g_chunk(gtypes.Part(text="lo"), finish="STOP"))
    assert events[:2] == [TextDelta("Hal"), TextDelta("lo")]
    end = events[-1]
    assert (end.text, end.tool_calls, end.stop_reason) == ("Hallo", [], "end")


async def test_gemini_thought_parts_are_kept_but_not_shown():
    thought = gtypes.Part(text="planning...", thought=True)
    events = await g_read(g_chunk(thought, gtypes.Part(text="Answer"), finish="STOP"))
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("Answer")]
    assert events[-1].raw.parts[0] is thought


async def test_gemini_parallel_calls_without_ids_get_distinct_ids():
    events = await g_read(
        g_chunk(g_call("find_journeys", {"origin": "Köln", "destination": "Bonn"}, signature=b"sig")),
        g_chunk(g_call("search_docs", {"query": "Bonn"}), finish="STOP"),
    )
    end = events[-1]
    assert end.stop_reason == "tool_use"
    assert [c.id for c in end.tool_calls] == [f"{SYNTHETIC_ID}0", f"{SYNTHETIC_ID}1"]
    assert end.tool_calls[0].arguments == {"origin": "Köln", "destination": "Bonn"}
    # the original parts, signature included, are what gets sent back
    assert end.raw.role == "model" and end.raw.parts[0].thought_signature == b"sig"


async def test_gemini_real_call_id_is_preserved():
    events = await g_read(g_chunk(g_call("search_docs", {"query": "x"}, call_id="abc"), finish="STOP"))
    assert events[-1].tool_calls[0].id == "abc"


async def test_gemini_stop_reasons():
    assert (await g_read(g_chunk(gtypes.Part(text="x"), finish="MAX_TOKENS")))[-1].stop_reason == "max_tokens"
    assert (await g_read(g_chunk(finish="SAFETY")))[-1].stop_reason == "refusal"
    assert (await g_read(g_chunk(block="SAFETY")))[-1].stop_reason == "refusal"
    try:
        await g_read(g_chunk(finish="MALFORMED_FUNCTION_CALL"))
    except LLMError:
        pass
    else:
        raise AssertionError("expected LLMError")


async def test_gemini_empty_stream_ends_cleanly():
    events = await g_read(gtypes.GenerateContentResponse(candidates=None))
    assert events[-1].stop_reason == "end" and events[-1].text == ""


def test_gemini_history_conversion():
    raw = gtypes.Content(role="model", parts=[g_call("search_docs", {"query": "Sylt"}, signature=b"sig")])
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "and Sylt?"},
        {"role": "assistant", "content": "", "tool_calls": [], "raw": raw},
        {"role": "tool", "results": [
            ToolResult(f"{SYNTHETIC_ID}0", "search_docs", "[]"),
            ToolResult("real-id", "find_journeys", "boom", True),
        ]},
    ]
    c = to_contents(history)
    assert [x.role for x in c] == ["user", "model", "user", "model", "user"]
    assert c[1].parts[0].text == "earlier answer"
    assert c[3] is raw  # echoed unchanged
    first, second = (p.function_response for p in c[4].parts)
    assert (first.id, first.name, first.response) == (None, "search_docs", {"result": "[]"})
    assert (second.id, second.name, second.response) == ("real-id", "find_journeys", {"error": "boom"})


def test_gemini_error_messages():
    from google.genai import errors as gerrors

    def err(code, message):
        return gerrors.APIError(code, {"error": {"code": code, "message": message, "status": "X"}})

    assert "rate limit" in describe_error(err(429, "quota"))
    assert "API key" in describe_error(err(400, "API key not valid. Please pass a valid API key."))
    assert "API key" in describe_error(err(403, "denied"))
    assert "LLM_MODEL" in describe_error(err(404, "models/nope is not found"))
    assert "(500)" in describe_error(err(500, "internal"))
    assert "overloaded" in describe_error(err(503, "high demand"))

    # shape of a real free-tier 429 (recorded 2026-10-11)
    quota = gerrors.APIError(429, {"error": {"code": 429, "message": "You exceeded your current quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaValue": "5"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45s"},
    ]}})
    assert retry_delay_s(quota) == 45.0
    assert retry_delay_s(err(429, "no hint")) is None


def test_cited_sources_keeps_only_sections_the_answer_names():
    from app.llm.agent import cited_sources

    retrieved = [
        {"source": "b.md", "document": "Border", "section": "Austria: Salzburg and Kufstein"},
        {"source": "d.md", "document": "Dest", "section": "Black Forest"},
        {"source": "b.md", "document": "Border", "section": "Austria: Salzburg and Kufstein"},
        {"source": "t.md", "document": "Tips", "section": "Planning a day trip"},
    ]
    answer = "Yes, via Freilassing (*Austria – Salzburg and Kufstein*). The Black Forestry office is unrelated."
    assert [s["section"] for s in cited_sources(answer, retrieved)] == ["Austria: Salzburg and Kufstein"]
    assert cited_sources("No sections named here.", retrieved) == []


async def test_agent_sends_only_cited_sources_once_at_the_end():
    class DocTools(FakeTools):
        async def run(self, name, args):
            return ToolOutcome("[]", events=[{"type": "sources", "sources": [
                {"source": "a.md", "document": "A", "section": "Used section"},
                {"source": "a.md", "document": "A", "section": "Unused section"},
            ]}])

    llm = ScriptedLLM([
        [TurnEnd("", [ToolCall("1", "search_docs", {"query": "q"})], "tool_use")],
        [TextDelta("Answer (Used section)."), TurnEnd("Answer (Used section).")],
    ])
    events = await collect(llm, DocTools(), [{"role": "user", "content": "?"}])
    sources = [e for e in events if e["type"] == "sources"]
    assert len(sources) == 1 and [s["section"] for s in sources[0]["sources"]] == ["Used section"]
    assert events[-2]["type"] == "sources" and events[-1] == {"type": "done"}
