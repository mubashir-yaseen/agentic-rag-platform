"""Custom exception hierarchy for the RAG platform.

Provides domain-specific exceptions to avoid catching and re-raising raw exceptions.
"""

from typing import Any, Optional


class RagPlatformError(Exception):
    """Base exception class for all domain errors in the RAG platform."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(RagPlatformError):
    """Raised when application configuration is missing, incomplete, or invalid."""


class IngestionError(RagPlatformError):
    """Raised when document extraction, parsing, or chunking fails."""


class RetrievalError(RagPlatformError):
    """Raised when vector search, database queries, or storage operations fail."""


class AgentError(RagPlatformError):
    """Raised when agent graph execution, routing, or tool calls fail."""


class McpError(RagPlatformError):
    """Raised when MCP server or client operations fail."""
