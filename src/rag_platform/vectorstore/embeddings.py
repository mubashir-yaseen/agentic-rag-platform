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

Optional second provider: Cloudflare Workers AI (``@cf/baai/bge-base-en-v1.5``, also
768 numbers). Google's free tier allows only 1,000 embedded texts per DAY, which is
not enough to index annual reports (about 2,000 chunks each). Cloudflare's free plan
allows 10,000 "neurons" per day, roughly 5,000+ chunks. If CLOUDFLARE_ACCOUNT_ID and
CLOUDFLARE_API_TOKEN are set they are used, otherwise Google. Vectors from the two
providers are NOT interchangeable: index and search with the same one.

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


CLOUDFLARE_EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5"  # 768 numbers, English retrieval model
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"
CLOUDFLARE_BATCH_SIZE = 50  # texts per request (Cloudflare accepts up to 100)
CLOUDFLARE_MAX_TEXT_CHARS = 1800  # the model reads ~512 tokens; longer text is cut for the vector (the stored text is not)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for bge search queries


class CloudflareEmbeddings:
    """Same interface as GoogleEmbeddings, backed by Cloudflare Workers AI (free: 10,000 neurons/day)."""

    def __init__(self, account_id: str, api_token: str, model: str = CLOUDFLARE_EMBEDDING_MODEL) -> None:
        if not account_id or not api_token:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for CloudflareEmbeddings")
        self.account_id = account_id
        self.api_token = api_token
        self.model = model

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip()
        return text[:CLOUDFLARE_MAX_TEXT_CHARS] if text else "(empty)"

    def _post(self, texts: list[str], timeout: float) -> list[list[float]]:
        url = f"{CLOUDFLARE_API_BASE}/{self.account_id}/ai/run/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        last_problem = "no response"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(url, headers=headers, json={"text": texts}, timeout=timeout)
            except requests.RequestException as exc:
                last_problem = f"network error: {exc.__class__.__name__}"
                logger.warning("Cloudflare embeddings %s (attempt %d/%d)", last_problem, attempt, MAX_ATTEMPTS)
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                        data = body["result"]["data"]
                    except Exception:
                        raise EmbeddingServiceError(f"Cloudflare returned an unexpected answer: {resp.text[:300]}")
                    if len(data) != len(texts):
                        raise EmbeddingServiceError(f"Cloudflare returned {len(data)} vectors for {len(texts)} texts.")
                    return [_l2_normalize(list(vec)) for vec in data]
                message = resp.text[:300]
                try:
                    errors = resp.json().get("errors") or []
                    if errors:
                        message = "; ".join(str(e.get("message", e)) for e in errors)[:300]
                except Exception:
                    pass
                if resp.status_code in (401, 403):
                    raise EmbeddingServiceError(
                        f"Cloudflare rejected the credentials (HTTP {resp.status_code}). Check CLOUDFLARE_ACCOUNT_ID and that "
                        f"CLOUDFLARE_API_TOKEN has the 'Workers AI' permission. Cloudflare said: {message}"
                    )
                if resp.status_code != 429 and resp.status_code < 500:
                    raise EmbeddingServiceError(f"Cloudflare embedding request failed (HTTP {resp.status_code}): {message}")
                last_problem = (
                    f"Cloudflare quota or capacity limit (HTTP {resp.status_code}): {message}"
                    if resp.status_code == 429
                    else f"Cloudflare server error (HTTP {resp.status_code}): {message}"
                )
                logger.warning("Cloudflare embeddings HTTP %s (attempt %d/%d)", resp.status_code, attempt, MAX_ATTEMPTS)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 20))
        raise EmbeddingServiceError(f"Cloudflare embeddings kept failing after {MAX_ATTEMPTS} tries: {last_problem}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), CLOUDFLARE_BATCH_SIZE):
            batch = [self._clean(t) for t in texts[start : start + CLOUDFLARE_BATCH_SIZE]]
            vectors.extend(self._post(batch, timeout=60))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._post([BGE_QUERY_PREFIX + self._clean(text)], timeout=30)[0]


def cloudflare_configured(settings: Any = None) -> bool:
    settings = settings or get_settings()
    return bool(getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "") and getattr(settings, "CLOUDFLARE_API_TOKEN", ""))


def embedding_configured(settings: Any = None) -> bool:
    """True when either embedding provider has its credentials."""
    settings = settings or get_settings()
    return cloudflare_configured(settings) or bool(getattr(settings, "GOOGLE_API_KEY", ""))


@lru_cache(maxsize=4)
def _get_cached_embedding_model(api_key: str, model: str) -> GoogleEmbeddings:
    logger.info("Initializing GoogleEmbeddings model: %s", model)
    return GoogleEmbeddings(api_key=api_key, model=model)


@lru_cache(maxsize=4)
def _get_cached_cloudflare_model(account_id: str, api_token: str, model: str) -> CloudflareEmbeddings:
    logger.info("Initializing CloudflareEmbeddings model: %s", model)
    return CloudflareEmbeddings(account_id=account_id, api_token=api_token, model=model)


def get_embedding_model(model_name: Optional[str] = None) -> Any:
    """Return the shared embeddings client: Cloudflare if its credentials are set, otherwise Google.

    ``model_name`` is accepted (and ignored) so existing callers that still pass
    the old Hugging Face model name keep working.
    """
    settings = get_settings()
    if cloudflare_configured(settings):
        return _get_cached_cloudflare_model(
            settings.CLOUDFLARE_ACCOUNT_ID, settings.CLOUDFLARE_API_TOKEN, CLOUDFLARE_EMBEDDING_MODEL
        )
    api_key = getattr(settings, "GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "No embedding provider configured. Add GOOGLE_API_KEY (free, no credit card: "
            "https://aistudio.google.com/app/apikey) or CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN "
            "as environment variables on the API server."
        )
    return _get_cached_embedding_model(api_key, GOOGLE_EMBEDDING_MODEL)


def warmup_embedding_model(model_name: Optional[str] = None) -> Any:
    """Create the client once at start-up (cheap: no weights, no network call)."""
    return get_embedding_model(model_name)
