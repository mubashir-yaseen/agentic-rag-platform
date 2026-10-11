"""Backwards-compatible configuration shim importing from rag_platform.config."""

from rag_platform.config import Settings, get_settings

_settings = get_settings()

OPENAI_API_KEY = _settings.OPENAI_API_KEY
OPENAI_MODEL = _settings.OPENAI_MODEL
EMBEDDING_MODEL = _settings.EMBEDDING_MODEL
CHUNK_SIZE = _settings.RAG_CHUNK_SIZE
CHUNK_OVERLAP = _settings.RAG_CHUNK_OVERLAP
K_RESULTS = _settings.RAG_K_RESULTS
TEMPERATURE = _settings.RAG_TEMPERATURE
VECTOR_STORE_DIR = _settings.VECTOR_STORE_DIR
VECTOR_STORE_BACKEND = _settings.VECTOR_STORE_BACKEND
SUPABASE_URL = _settings.SUPABASE_URL
SUPABASE_ANON_KEY = _settings.SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY = _settings.SUPABASE_SERVICE_ROLE_KEY
SUPABASE_KEY = _settings.effective_supabase_key


def validate_config() -> bool:
    """Validate that required LLM API configuration is set."""
    _settings.validate_llm_credentials()
    return True


if __name__ == "__main__":
    print("RAG System Configuration:")
    print(f"  OpenAI Model: {OPENAI_MODEL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"  K Results: {K_RESULTS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Vector Store Dir: {VECTOR_STORE_DIR}")
