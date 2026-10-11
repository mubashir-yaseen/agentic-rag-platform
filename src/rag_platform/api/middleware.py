"""Middleware and exception handling layer for FastAPI."""

import time
import uuid
from collections import defaultdict
from typing import Callable
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rag_platform.api.schemas import ErrorResponse
from rag_platform.exceptions import (
    AgentError,
    ConfigurationError,
    IngestionError,
    McpError,
    RagPlatformError,
    RetrievalError,
)
from rag_platform.logging_config import get_correlation_id, get_logger, set_correlation_id

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensure every HTTP request has a valid correlation ID set in contextvars and response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        corr_id = (
            request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        set_correlation_id(corr_id)
        request.state.correlation_id = corr_id

        start_time = time.time()
        logger.info(
            "HTTP %s %s [correlation_id=%s]",
            request.method,
            request.url.path,
            corr_id,
        )

        try:
            response = await call_next(request)
        except Exception:
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "HTTP %s %s finished in %.2fms [correlation_id=%s]",
                request.method,
                request.url.path,
                duration_ms,
                corr_id,
            )

        response.headers["X-Correlation-ID"] = corr_id
        return response


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per client IP address."""

    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Window duration in seconds.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: dict[str, list[float]] = defaultdict(list)

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check whether client IP has exceeded allowed request rate."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old timestamps
        timestamps = [ts for ts in self.clients[client_ip] if ts > cutoff]
        self.clients[client_ip] = timestamps

        if len(timestamps) >= self.max_requests:
            return True

        self.clients[client_ip].append(now)
        return False


rate_limiter = InMemoryRateLimiter(max_requests=120, window_seconds=60.0)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enforce rate limits on high-load API endpoints (/chat, /ingest)."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if "/api/v1/chat" in path or "/api/v1/ingest" in path:
            client_ip = request.client.host if request.client else "unknown"
            if rate_limiter.is_rate_limited(client_ip):
                corr_id = get_correlation_id()
                logger.warning("Rate limit exceeded for client %s on %s", client_ip, path)
                err = ErrorResponse(
                    error="Rate limit exceeded. Please wait before sending more requests.",
                    error_type="RateLimitExceeded",
                    correlation_id=corr_id,
                    details={"client_ip": client_ip, "limit": rate_limiter.max_requests},
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content=err.model_dump(),
                    headers={"X-Correlation-ID": corr_id, "Retry-After": "60"},
                )

        return await call_next(request)


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized custom exception handlers with FastAPI application."""

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.warning("Configuration error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="ConfigurationError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )

    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(request: Request, exc: IngestionError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.warning("Ingestion error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="IngestionError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.error("Retrieval error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="RetrievalError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.error("Agent execution error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="AgentError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )

    @app.exception_handler(McpError)
    async def mcp_error_handler(request: Request, exc: McpError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.error("MCP server error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="McpError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )

    @app.exception_handler(RagPlatformError)
    async def generic_platform_error_handler(request: Request, exc: RagPlatformError) -> JSONResponse:
        corr_id = get_correlation_id()
        logger.error("RAG Platform error: %s (correlation_id=%s)", exc.message, corr_id)
        err = ErrorResponse(
            error=exc.message,
            error_type="RagPlatformError",
            correlation_id=corr_id,
            details=exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err.model_dump(),
            headers={"X-Correlation-ID": corr_id},
        )
