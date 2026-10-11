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


@lru_cache
def get_settings() -> Settings:
    return Settings()
