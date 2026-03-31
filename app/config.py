"""Application configuration."""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings from environment variables."""
    
    # OpenAI API
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    
    # Models
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    
    # Paths
    database_path: Path = Path("./data/qna.db")
    knowledge_path: Path = Path("./knowledge")
    
    # RAG settings
    max_context_docs: int = 2
    chunk_size: int = 2000
    
    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()