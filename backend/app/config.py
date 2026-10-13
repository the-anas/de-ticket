from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"), extra="ignore"
    )

    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_api_key: str = ""
    llm_base_url: str = ""

    transit_base_url: str = "https://api.transitous.org"
    transit_user_agent: str = "dticket-planner/0.1"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Cross-encoder that re-orders the best search results; empty turns it off.
    # Off by default: BAAI/bge-reranker-v2-m3-int8 raises the eval score (right
    # section first 84% -> 95%) but adds about 1.6 s per search. See CHANGES.txt.
    reranker_model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
