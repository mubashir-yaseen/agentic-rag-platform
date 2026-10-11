"""Embeddings factory and singleton management for dense vector models."""

from functools import lru_cache
from typing import Any, Optional

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_cached_embedding_model(model_name: str, api_token: Optional[str]) -> Any:
    """Cached instantiation of a HuggingFace embeddings client.

    Calls HuggingFace's hosted Inference API remotely instead of loading the
    model's weights (and torch/transformers) into local process memory, to
    keep the backend's memory footprint small enough for constrained hosts.
    """
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    logger.info("Initializing remote HuggingFaceEndpointEmbeddings model: %s", model_name)
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=api_token,
    )


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Retrieve or initialize a cached embeddings client.

    Args:
        model_name: Optional model identifier. Defaults to settings.EMBEDDING_MODEL.

    Returns:
        An embeddings client exposing embed_query()/embed_documents(), matching
        the interface the rest of the codebase already relies on.
    """
    settings = get_settings()
    target_model = model_name or settings.EMBEDDING_MODEL
    api_token = getattr(settings, "HUGGINGFACEHUB_API_TOKEN", None)
    return _get_cached_embedding_model(target_model, api_token)


def warmup_embedding_model(model_name: Optional[str] = None) -> Any:
    """Preload the configured embedding model once per process.

    This intentionally reuses the existing cache layer so the first user request
    does not pay the lazy Hugging Face initialization penalty again.
    """
    return get_embedding_model(model_name)
