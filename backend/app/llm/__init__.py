from functools import lru_cache

from ..config import get_settings
from .base import LLMProvider


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(s.llm_model, s.llm_api_key)
    if provider in ("openai", "openai-compatible", "ollama"):
        from .openai_compat_provider import OpenAICompatProvider

        return OpenAICompatProvider(s.llm_model, s.llm_api_key, s.llm_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER {s.llm_provider!r}; use 'anthropic' or 'openai'.")
