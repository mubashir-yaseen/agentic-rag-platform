"""Embeddings factory and singleton management for dense vector models."""

from functools import lru_cache
from typing import Any, Optional

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def _get_cached_embedding_model(model_name: str) -> Any:
    """Cached instantiation of HuggingFace embeddings model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("Initializing HuggingFaceEmbeddings model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Retrieve or initialize a cached HuggingFaceEmbeddings instance.

    Args:
        model_name: Optional model identifier. Defaults to settings.EMBEDDING_MODEL.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    settings = get_settings()
    target_model = model_name or settings.EMBEDDING_MODEL
    return _get_cached_embedding_model(target_model)


def warmup_embedding_model(model_name: Optional[str] = None) -> Any:
    """Preload the configured embedding model once per process.

    This intentionally reuses the existing cache layer so the first user request
    does not pay the lazy Hugging Face initialization penalty again.
    """
    return get_embedding_model(model_name)
