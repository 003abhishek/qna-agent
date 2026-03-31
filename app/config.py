"""Application configuration."""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Settings from environment variables."""

    # ✅ LLM / Ollama configuration (OpenAI-compatible)
    openai_api_key: str = "ollama"
    openai_api_base: str = "http://localhost:11434/v1"

    # ✅ Models (Ollama)
    llm_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"

    # ✅ Paths
    database_path: Path = Path("./data/qna.db")
    knowledge_path: Path = Path("./knowledge")

    # ✅ RAG settings
    max_context_docs: int = 2
    chunk_size: int = 2000

    # ✅ Pydantic v2 configuration (replaces class Config)
    model_config = ConfigDict(
        env_file=".env"
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()