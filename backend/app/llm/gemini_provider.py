import asyncio
import re
from typing import AsyncIterator

import httpx
from google import genai
from google.genai import errors, types

from .base import LLMError, Notice, TextDelta, ToolCall, ToolSpec, TurnEnd

# Gemini does not always give function calls an id; these stand in for one.
SYNTHETIC_ID = "gemini_call_"

REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}
BAD_TOOL_CALL_REASONS = {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}


def to_contents(messages: list[dict]) -> list[types.Content]:
    """Neutral history -> Gemini contents (roles "user" and "model")."""
    out: list[types.Content] = []
    for m in messages:
        if m["role"] == "user":
            out.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
        elif m["role"] == "assistant":
            # Turns produced in this request carry the model's original content,
            # which goes back unchanged: its function-call parts hold thought
            # signatures the API checks on the next request. Earlier turns
            # arrive from the browser as plain text.
            if m.get("raw") is not None:
                out.append(m["raw"])
            else:
                out.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))
        else:
            # all function responses of a turn travel together, in call order
            parts = []
            for r in m["results"]:
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=None if r.id.startswith(SYNTHETIC_ID) else r.id,
                            name=r.name,
                            response={"error": r.content} if r.is_error else {"result": r.content},
                        )
                    )
                )
            out.append(types.Content(role="user", parts=parts))
    return out


async def read_stream(
    chunks: AsyncIterator[types.GenerateContentResponse],
) -> AsyncIterator[TextDelta | TurnEnd]:
    """Turns Gemini's streamed chunks into text deltas and one final TurnEnd."""
    parts: list[types.Part] = []
    text = ""
    calls: list[ToolCall] = []
    finish: str | None = None
    blocked = False

    async for chunk in chunks:
        if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
            blocked = True
        for candidate in chunk.candidates or []:
            if candidate.finish_reason:
                finish = candidate.finish_reason.name
            for part in (candidate.content.parts if candidate.content else None) or []:
                # kept as received, including thought parts and signatures
                parts.append(part)
                if part.function_call:
                    fc = part.function_call
                    calls.append(
                        ToolCall(
                            id=fc.id or f"{SYNTHETIC_ID}{len(calls)}",
                            name=fc.name or "",
                            arguments=dict(fc.args or {}),
                        )
                    )
                elif part.text and not part.thought:
                    text += part.text
                    yield TextDelta(part.text)

    if finish in BAD_TOOL_CALL_REASONS:
        raise LLMError("The model produced an invalid tool call. Please try again.")
    if blocked or finish in REFUSAL_REASONS:
        stop = "refusal"
    elif finish == "MAX_TOKENS":
        stop = "max_tokens"
    elif calls:
        stop = "tool_use"
    else:
        stop = "end"
    yield TurnEnd(
        text=text,
        tool_calls=calls,
        stop_reason=stop,
        raw=types.Content(role="model", parts=parts),
    )


# The free tier allows only a few requests per minute and one answer takes
# several, so a 429 is waited out when Google says the wait is short. 503
# ("model overloaded") is common on the free tier and usually clears quickly.
MAX_RATE_LIMIT_WAIT_S = 65
OVERLOAD_DELAYS_S = (2, 5, 10)


def retry_delay_s(e: errors.APIError) -> float | None:
    """Seconds Google asks us to wait after a 429, if it says."""
    details = e.details if isinstance(e.details, dict) else {}
    for item in details.get("error", {}).get("details", []):
        match = re.fullmatch(r"([\d.]+)s", str(item.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None


def describe_error(e: errors.APIError) -> str:
    if e.code == 429:
        return "The Gemini free-tier rate limit was reached; wait a minute and try again."
    if e.code == 503:
        return "Gemini is overloaded right now; try again in a moment."
    if e.code in (401, 403) or (e.code == 400 and "API key" in (e.message or "")):
        return "The Gemini API key is missing or invalid. Set LLM_API_KEY in .env."
    if e.code == 404:
        return f"Gemini does not know this model. Check LLM_MODEL in .env. ({e.message})"
    return f"The Gemini API returned an error ({e.code}): {e.message}"


class GeminiProvider:
    """Google Gemini through the native API (google-genai SDK)."""

    def __init__(self, model: str, api_key: str = ""):
        try:
            # With no key given, the SDK reads GEMINI_API_KEY / GOOGLE_API_KEY.
            self._client = genai.Client(api_key=api_key or None)
        except ValueError as e:
            raise LLMError(
                "No Gemini API key found. Create one at aistudio.google.com and set LLM_API_KEY in .env."
            ) from e
        self._model = model

    async def stream(
        self, system: str, context: str, messages: list[dict], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | Notice | TurnEnd]:
        config = types.GenerateContentConfig(
            system_instruction=f"{system}\n\n{context}",
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ],
            # The app's own loop runs the tools; the SDK must not.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = to_contents(messages)
        overload_delays = list(OVERLOAD_DELAYS_S)
        waited_for_quota = False
        while True:
            started = False
            try:
                chunks = await self._client.aio.models.generate_content_stream(
                    model=self._model, contents=contents, config=config
                )
                async for event in read_stream(chunks):
                    started = True
                    yield event
                return
            except errors.APIError as e:
                # only retry a turn that has produced nothing yet
                delay = None
                if not started and e.code == 503 and overload_delays:
                    delay = overload_delays.pop(0)
                elif not started and e.code == 429 and not waited_for_quota:
                    wait = retry_delay_s(e)
                    if wait is not None and wait <= MAX_RATE_LIMIT_WAIT_S:
                        waited_for_quota = True
                        delay = wait + 1
                        yield Notice(f"Free-tier rate limit reached, continuing in {round(delay)} s…")
                if delay is None:
                    raise LLMError(describe_error(e)) from e
                await asyncio.sleep(delay)
            except httpx.HTTPError as e:
                raise LLMError("Could not reach the Gemini API.") from e
