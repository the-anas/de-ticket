import json
from typing import AsyncIterator

import openai

from .base import LLMError, TextDelta, ToolCall, ToolSpec, TurnEnd


class OpenAICompatProvider:
    """Chat Completions API: OpenAI itself, or any compatible server (Ollama,
    vLLM, LM Studio, ...) via LLM_BASE_URL."""

    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        # local servers ignore the key, but the SDK insists on one
        self._client = openai.AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url or None)
        self._model = model

    @staticmethod
    def _messages(system: str, context: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": f"{system}\n\n{context}"}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m["content"] or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            else:
                for r in m["results"]:
                    out.append({"role": "tool", "tool_call_id": r.id, "content": r.content})
        return out

    async def stream(
        self, system: str, context: str, messages: list[dict], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | TurnEnd]:
        text = ""
        partial: dict[int, dict] = {}
        finish = None
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(system, context, messages),
                tools=[
                    {
                        "type": "function",
                        "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                    }
                    for t in tools
                ],
                stream=True,
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                delta = choice.delta
                if delta.content:
                    text += delta.content
                    yield TextDelta(delta.content)
                for tc in delta.tool_calls or []:
                    slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except openai.AuthenticationError as e:
            raise LLMError("The LLM API key is missing or invalid. Set LLM_API_KEY in .env.") from e
        except openai.RateLimitError as e:
            raise LLMError("The model is rate limited right now; try again in a moment.") from e
        except openai.APIStatusError as e:
            raise LLMError(f"The model API returned an error ({e.status_code}): {e.message}") from e
        except openai.APIConnectionError as e:
            raise LLMError("Could not reach the LLM endpoint; check LLM_BASE_URL.") from e

        calls = []
        for index in sorted(partial):
            slot = partial[index]
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                # passed through as-is; the tool handler rejects it and the
                # model sees the error and can retry
                arguments = {"_invalid_json": slot["arguments"]}
            calls.append(ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"], arguments=arguments))

        if finish == "length":
            stop = "max_tokens"
        elif finish == "content_filter":
            stop = "refusal"
        elif calls:
            stop = "tool_use"
        else:
            stop = "end"
        yield TurnEnd(text=text, tool_calls=calls, stop_reason=stop)
