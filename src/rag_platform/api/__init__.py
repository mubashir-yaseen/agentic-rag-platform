"""API subpackage providing production FastAPI endpoints, schemas, and middleware."""

from rag_platform.api.main import app, create_app
from rag_platform.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    ReadyResponse,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "HealthResponse",
    "IngestResponse",
    "ReadyResponse",
    "app",
    "create_app",
]
