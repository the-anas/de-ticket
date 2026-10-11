from typing import AsyncIterator

import anthropic

from .base import LLMError, TextDelta, ToolCall, ToolSpec, TurnEnd

MAX_TOKENS = 16000


class AnthropicProvider:
    def __init__(self, model: str, api_key: str = ""):
        # With no key given, the SDK resolves ANTHROPIC_API_KEY or an
        # `ant auth login` profile from the environment.
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self._model = model

    @staticmethod
    def _messages(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # Turns produced in this request carry the original blocks,
                # which must go back unchanged (thinking blocks are bound to
                # the conversation). Earlier turns arrive from the browser as
                # plain text.
                out.append({"role": "assistant", "content": m.get("raw") or m["content"]})
            else:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in m["results"]
                        ],
                    }
                )
        return out

    async def stream(
        self, system: str, context: str, messages: list[dict], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnEnd]:
        try:
            async with self._client.beta.messages.stream(
                model=self._model,
                max_tokens=MAX_TOKENS,
                # Stable prompt first with a cache breakpoint; the per-request
                # context comes after it so it does not invalidate the cache.
                system=[
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": context},
                ],
                # Thinking is always on for current Claude models; effort is the
                # control. Medium is enough for itinerary planning.
                output_config={"effort": "medium"},
                # If a safety classifier declines the request, let the API
                # retry it on a fallback model instead of failing the turn.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                # Tool inputs here are a few short strings, so they are left
                # buffered (server-validated) rather than eagerly streamed.
                tools=[
                    {"name": t.name, "description": t.description, "input_schema": t.parameters}
                    for t in tools
                ],
                messages=self._messages(messages),
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                message = await stream.get_final_message()
        except TypeError as e:
            # the SDK raises this before sending when it finds no credentials
            raise LLMError(
                "No Anthropic credentials found. Set LLM_API_KEY in .env (or ANTHROPIC_API_KEY)."
            ) from e
        except anthropic.AuthenticationError as e:
            raise LLMError("The Anthropic API key is missing or invalid. Set LLM_API_KEY in .env.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("The model is rate limited right now; try again in a moment.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"The model API returned an error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API.") from e

        text = "".join(b.text for b in message.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in message.content
            if b.type == "tool_use"
        ]
        if message.stop_reason == "refusal":
            stop = "refusal"
        elif message.stop_reason == "max_tokens":
            stop = "max_tokens"
        elif calls:
            stop = "tool_use"
        else:
            stop = "end"
        yield TurnEnd(text=text, tool_calls=calls, stop_reason=stop, raw=message.content)
