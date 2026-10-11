"""FastAPI Application factory and entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_platform.api.middleware import (
    CorrelationIdMiddleware,
    RateLimitingMiddleware,
    register_exception_handlers,
)
from rag_platform.api.routes.v1 import router as v1_router
from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger, setup_logging
from rag_platform.vectorstore.embeddings import warmup_embedding_model

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create, configure, and assemble the FastAPI application instance."""
    import logging
    settings = get_settings()
    log_lvl = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    setup_logging(level=log_lvl, json_format=settings.LOG_JSON_FORMAT)

    app = FastAPI(
        title="RAG Platform API",
        description="Production-grade Multimodal Agentic RAG Platform with MCP Interoperability",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Correlation-ID"],
    )

    # 2. Rate Limiting Middleware
    app.add_middleware(RateLimitingMiddleware)

    # 3. Correlation ID Tracking Middleware
    app.add_middleware(CorrelationIdMiddleware)

    # 4. Exception Handlers
    register_exception_handlers(app)

    # 5. Route Controllers
    app.include_router(v1_router)

    @app.on_event("startup")
    async def _startup_embedding_warmup() -> None:
        """Preload the embedding model once during app startup to avoid first-request latency."""
        try:
            warmup_embedding_model()
            logger.info("Embedding model preloaded during FastAPI startup")
        except Exception as exc:
            logger.warning("Embedding model startup warmup failed: %s", exc)

    logger.info("FastAPI application created with environment: %s", settings.ENVIRONMENT)
    return app


app = create_app()


def run() -> None:
    """Run uvicorn server for local development."""
    import uvicorn

    uvicorn.run(
        "rag_platform.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
