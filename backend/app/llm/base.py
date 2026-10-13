"""Provider-neutral chat interface.

The agent keeps history in a neutral shape and each provider converts it:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall], "raw": Any}
    {"role": "tool", "results": [ToolResult]}      # all results of one turn

"raw" is whatever the provider returned for that assistant turn. Providers that
need their own blocks echoed back unchanged (Anthropic thinking blocks) read it;
others ignore it.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class TextDelta:
    text: str


@dataclass
class Notice:
    """A short status line for the user, e.g. while waiting out a rate limit."""

    text: str


@dataclass
class TurnEnd:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # "end" | "tool_use" | "max_tokens" | "refusal"
    stop_reason: str = "end"
    raw: Any = None


class LLMError(Exception):
    """A provider failure with a message that is safe to show to the user."""


class LLMProvider(Protocol):
    def stream(
        self, system: str, context: str, messages: list[dict], tools: list[ToolSpec]
    ) -> AsyncIterator[TextDelta | Notice | TurnEnd]:
        """Streams one model turn. `system` is stable across requests (and may
        be cached); `context` holds per-request facts such as the current time."""
        ...
