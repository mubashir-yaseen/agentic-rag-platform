"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_platform.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Central configuration management for RAG Platform.

    Loads environment variables from .env file or environment,
    with type-validation and default fallbacks.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App & Logging
    APP_NAME: str = Field(default="RAG Chatbot Platform")
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_JSON_FORMAT: bool = Field(default=False)

    # Supabase Configuration
    SUPABASE_URL: str = Field(default="")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(default="")
    SUPABASE_ANON_KEY: str = Field(default="")
    SUPABASE_KEY: str = Field(default="")

    # LLM API Keys & Endpoints
    OPENAI_API_KEY: str = Field(default="")
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    LLM_MODEL: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b:free")
    LLM_MODEL_MODE: str = Field(default="fixed")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo")

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    HUGGINGFACEHUB_API_TOKEN: str = Field(default="")

    # RAG Tuning Parameters
    RAG_CHUNK_SIZE: int = Field(default=1000)
    RAG_CHUNK_OVERLAP: int = Field(default=200)
    RAG_K_RESULTS: int = Field(default=3)
    RAG_TEMPERATURE: float = Field(default=0.2)

    # Vector Store Local Configuration
    VECTOR_STORE_DIR: str = Field(default="vector_stores")
    VECTOR_STORE_BACKEND: str = Field(default="faiss")

    # Controlled output directory for agent-generated files (e.g. Excel exports).
    # Files are always saved under a UUID-derived name inside this directory;
    # never derived directly from user-supplied paths.
    GENERATED_FILES_DIR: str = Field(default="generated_files")

    # Observability & Tracing (Langfuse / LangSmith)
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="https://cloud.langfuse.com")
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: str = Field(default="")
    LANGSMITH_PROJECT: str = Field(default="rag-platform")

    @property
    def effective_api_key(self) -> str:
        """Return the active LLM API key (OpenRouter or OpenAI)."""
        return self.OPENROUTER_API_KEY or self.OPENAI_API_KEY

    @property
    def effective_supabase_key(self) -> str:
        """Return the active Supabase key (service role preferred, anon key, or generic key)."""
        return self.SUPABASE_SERVICE_ROLE_KEY or self.SUPABASE_ANON_KEY or self.SUPABASE_KEY

    def validate_llm_credentials(self) -> None:
        """Ensure an API key is available for LLM inference."""
        if not self.effective_api_key:
            raise ConfigurationError(
                "Missing LLM API Key: please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env"
            )

    def validate_supabase_credentials(self) -> None:
        """Ensure Supabase URL and key are configured."""
        if not self.SUPABASE_URL or not self.effective_supabase_key:
            raise ConfigurationError(
                "Missing Supabase credentials: please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY/SUPABASE_KEY"
            )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached singleton instance of application settings."""
    return Settings()
