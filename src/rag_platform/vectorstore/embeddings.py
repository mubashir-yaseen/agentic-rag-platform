"""Embeddings factory using Google's free Gemini embedding API.

Why this module looks the way it does
-------------------------------------
* Hugging Face's hosted inference now asks for a payment method, and running
  sentence-transformers inside the Render free tier (512 MB RAM) made deploys
  hang. Google's Generative Language API has a free tier that only needs an API
  key from https://aistudio.google.com/app/apikey (no credit card).
* The API is called directly over HTTP, so there is no heavy dependency and no
  model weights to load at start-up.
* ``gemini-embedding-001`` returns 3072 numbers per text by default. We ask for
  768 (``output_dimensionality``) to match the ``vector(768)`` column in
  Supabase, and L2-normalise the result ourselves, because Google only
  normalises the full-size output.
* Texts are sent in small batches and rate-limit / server errors are retried
  with back-off. Failures raise ``EmbeddingServiceError`` with Google's own
  message, which the API returns to the UI so you can see what went wrong.

The public functions (``get_embedding_model`` and ``warmup_embedding_model``)
keep the signatures the rest of the code base already uses.
"""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from typing import Any, Optional

import requests

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)

GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Must match the vector(N) column and the match_chunks() function in Supabase
# (see migration 002_switch_to_google_embeddings_768.sql).
OUTPUT_DIMENSIONALITY = 768
EMBEDDING_DIMENSIONS = OUTPUT_DIMENSIONALITY

BATCH_SIZE = 25  # texts per batchEmbedContents call (kept well below any documented cap)
MAX_ATTEMPTS = 5  # per HTTP call, for 429 / 5xx / network errors
MAX_TEXT_CHARS = 8000  # ~2000 tokens, the model's input limit; longer text is cut


class EmbeddingServiceError(RuntimeError):
    """The embedding service failed or rejected a request (message is user-readable)."""


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale a vector to unit length (needed for truncated gemini-embedding-001 output)."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _clean(text: str) -> str:
    """Google rejects empty inputs and over-long ones; make every text acceptable."""
    text = (text or "").strip()
    if not text:
        return "(empty)"
    return text[:MAX_TEXT_CHARS]


def _explain(status: int, body: str) -> str:
    """Turn a Google error response into one readable sentence."""
    message = body[:300]
    try:
        err = json.loads(body).get("error", {})
        message = err.get("message") or message
    except Exception:
        pass
    if status in (400, 403) and ("API key" in message or "api key" in message or "PERMISSION_DENIED" in body):
        return (
            f"Google rejected the API key (HTTP {status}). Check that GOOGLE_API_KEY on Render is a valid "
            f"key from https://aistudio.google.com/app/apikey. Google said: {message}"
        )
    if status == 404:
        return f"Google does not know the embedding model '{GOOGLE_EMBEDDING_MODEL}' (HTTP 404): {message}"
    if status == 429:
        return (
            "Google's free embedding quota is used up for now (HTTP 429). Wait a minute and try again, "
            f"or upload a smaller PDF. Google said: {message}"
        )
    return f"Google embedding request failed (HTTP {status}): {message}"


class GoogleEmbeddings:
    """Exposes ``embed_documents`` / ``embed_query`` like a LangChain embeddings object."""

    def __init__(self, api_key: str, model: str = GOOGLE_EMBEDDING_MODEL) -> None:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for GoogleEmbeddings")
        self.api_key = api_key
        self.model = model

    # -- HTTP -----------------------------------------------------------------
    def _post(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        last_problem = "no response"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as exc:  # network trouble: retry
                last_problem = f"network error: {exc.__class__.__name__}"
                logger.warning("Google embeddings %s (attempt %d/%d)", last_problem, attempt, MAX_ATTEMPTS)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code != 429 and resp.status_code < 500:
                    # a real client error (bad key, bad request): retrying will not help
                    logger.warning("Google embeddings failed: %s %s", resp.status_code, resp.text[:500])
                    raise EmbeddingServiceError(_explain(resp.status_code, resp.text))
                last_problem = _explain(resp.status_code, resp.text)
                logger.warning(
                    "Google embeddings HTTP %s (attempt %d/%d)", resp.status_code, attempt, MAX_ATTEMPTS
                )
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < MAX_ATTEMPTS:
                    time.sleep(min(int(retry_after), 30))
                    continue
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 20))
        raise EmbeddingServiceError(f"Google embeddings kept failing after {MAX_ATTEMPTS} tries: {last_problem}")

    # -- public API -----------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts (document side), in batches."""
        if not texts:
            return []
        url = f"{GOOGLE_API_BASE}/{self.model}:batchEmbedContents"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            payload = {
                "requests": [
                    {
                        "model": self.model,
                        "content": {"parts": [{"text": _clean(text)}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "output_dimensionality": OUTPUT_DIMENSIONALITY,
                    }
                    for text in batch
                ]
            }
            data = self._post(url, payload, timeout=60)
            got = data.get("embeddings", [])
            if len(got) != len(batch):
                raise EmbeddingServiceError(
                    f"Google returned {len(got)} vectors for {len(batch)} texts (batch starting at {start})."
                )
            vectors.extend(_l2_normalize(item["values"]) for item in got)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""
        url = f"{GOOGLE_API_BASE}/{self.model}:embedContent"
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": _clean(text)}]},
            "taskType": "RETRIEVAL_QUERY",
            "output_dimensionality": OUTPUT_DIMENSIONALITY,
        }
        data = self._post(url, payload, timeout=30)
        return _l2_normalize(data["embedding"]["values"])


@lru_cache(maxsize=4)
def _get_cached_embedding_model(api_key: str, model: str) -> GoogleEmbeddings:
    logger.info("Initializing GoogleEmbeddings model: %s", model)
    return GoogleEmbeddings(api_key=api_key, model=model)


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Return the shared GoogleEmbeddings client.

    ``model_name`` is accepted (and ignored) so existing callers that still pass
    the old Hugging Face model name keep working.
    """
    api_key = getattr(get_settings(), "GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Create a free key (no credit card) at "
            "https://aistudio.google.com/app/apikey and add it as an environment variable on the API server."
        )
    return _get_cached_embedding_model(api_key, GOOGLE_EMBEDDING_MODEL)


def warmup_embedding_model(model_name: Optional[str] = None) -> Any:
    """Create the client once at start-up (cheap: no weights, no network call)."""
    return get_embedding_model(model_name)
