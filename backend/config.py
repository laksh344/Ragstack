"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM Providers ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- LangSmith ---
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "ragstack"

    # --- Cohere ---
    cohere_api_key: str = ""

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ragstack"

    # --- Elasticsearch ---
    es_host: str = "http://localhost:9200"
    es_index: str = "ragstack"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
