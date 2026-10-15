"""Embeddings factory using Google's free Gemini embedding API.

Switched from Hugging Face (which now requires a payment method on file for
any Inference Providers usage, even free-tier) and from local sentence-
transformers (which exceeded Render's free-tier RAM/startup-time budget and
caused deploys to hang and time out).

Google's Generative Language API offers a genuinely free tier that requires
only an API key from https://aistudio.google.com/app/apikey -- no credit
card needed. This module calls it directly over HTTP (no extra heavy
dependency).

IMPORTANT: "text-embedding-004" was retired by Google in January 2026 and
now returns 404/400 errors. The current model is "gemini-embedding-001".
It outputs 3072 dimensions by default; we request output_dimensionality=768
to match the existing Supabase vector(768) column, and L2-normalize the
result ourselves since Google only auto-normalizes the full 3072-dim output.
"""

import math
from functools import lru_cache
from typing import Any, Optional

import requests

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)

GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OUTPUT_DIMENSIONALITY = 768


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Required for gemini-embedding-001 whenever the
    output is truncated below its native 3072 dimensions (per Google's docs);
    the full-size output is already normalized, but a truncated one is not.
    """
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


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

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Google's batchEmbedContents endpoint."""
        if not texts:
            return []
        url = f"{GOOGLE_API_BASE}/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": self.model,
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "output_dimensionality": OUTPUT_DIMENSIONALITY,
                }
                for text in texts
            ]
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        if resp.status_code != 200:
            logger.warning("Google embed_documents failed: %s %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        embeddings = [_l2_normalize(item["values"]) for item in data.get("embeddings", [])]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Google embedding API returned {len(embeddings)} vectors for {len(texts)} inputs"
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string using Google's embedContent endpoint."""
        url = f"{GOOGLE_API_BASE}/{self.model}:embedContent"
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "output_dimensionality": OUTPUT_DIMENSIONALITY,
        }
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
        if resp.status_code != 200:
            logger.warning("Google embed_query failed: %s %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        return _l2_normalize(data["embedding"]["values"])


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

    This is cheap (no local model weights to load), so it does not add
    meaningful time to app startup.
    """
    return get_embedding_model(model_name)
