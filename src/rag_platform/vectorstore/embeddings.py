"""Embeddings factory using Google's free Gemini embedding API.

Switched from Hugging Face (which now requires a payment method on file for
any Inference Providers usage, even free-tier) and from local sentence-
transformers (which exceeded Render's free-tier RAM/startup-time budget and
caused deploys to hang and time out).

Google's Generative Language API offers a genuinely free tier that requires
only an API key from https://aistudio.google.com/app/apikey -- no credit
card needed. This module calls it directly over HTTP (no extra heavy
dependency), producing 768-dimensional vectors (text-embedding-004).
"""

from functools import lru_cache
from typing import Any, Optional

import requests

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)

GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBEDDING_DIMENSIONS = 768


class GoogleEmbeddings:
    """Drop-in replacement exposing the same embed_documents/embed_query
    interface the rest of the codebase (RAGSystem, ingestion, app.py local
    fallback) already expects from a LangChain-style embeddings object.
    """

    def __init__(self, api_key: str, model: str = GOOGLE_EMBEDDING_MODEL) -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleEmbeddings")
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Google's batchEmbedContents endpoint."""
        if not texts:
            return []
        url = f"{GOOGLE_API_BASE}/{self.model}:batchEmbedContents?key={self.api_key}"
        payload = {
            "requests": [
                {
                    "model": self.model,
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["values"] for item in data.get("embeddings", [])]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Google embedding API returned {len(embeddings)} vectors for {len(texts)} inputs"
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using Google's embedContent endpoint."""
        url = f"{GOOGLE_API_BASE}/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]


@lru_cache(maxsize=4)
def _get_cached_embedding_model(api_key: str, model: str) -> Any:
    logger.info("Initializing GoogleEmbeddings model: %s", model)
    return GoogleEmbeddings(api_key=api_key, model=model)


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Retrieve or initialize a cached GoogleEmbeddings instance.

    Args:
        model_name: Optional model identifier (ignored; kept for signature
            compatibility with earlier HuggingFace-based factory). Google's
            model is selected via GOOGLE_EMBEDDING_MODEL above.

    Returns:
        GoogleEmbeddings instance.
    """
    settings = get_settings()
    api_key = getattr(settings, "GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Get a free key (no credit card required) at "
            "https://aistudio.google.com/app/apikey and set it in your environment."
        )
    return _get_cached_embedding_model(api_key, GOOGLE_EMBEDDING_MODEL)


def warmup_embedding_model(model_name: Optional[str] = None) -> Any:
    """Preload the embedding client once per process.

    This is now cheap (no local model weights to load), so it no longer adds
    meaningful time to app startup.
    """
    return get_embedding_model(model_name)
