"""RAG Platform Package.

Multimodal Agentic RAG Platform with MCP, structured logging, and robust configuration.
"""

from rag_platform.agent import (
    AgentResult,
    McpClient,
    McpServerConfig,
    McpToolDefinition,
    calculator_tool,
    retrieval_tool,
    run_agent,
    web_search_tool,
)
from rag_platform.api import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    ReadyResponse,
    app,
    create_app,
)
from rag_platform.config import Settings, get_settings
from rag_platform.eval import EvalReport, EvalSample, SampleEvalScore, run_evaluation
from rag_platform.exceptions import (
    AgentError,
    ConfigurationError,
    IngestionError,
    McpError,
    RagPlatformError,
    RetrievalError,
)
from rag_platform.ingestion import ChunkMetadata, ContentType, ExtractedChunk, IngestionResult, ingest_pdf
from rag_platform.logging_config import get_correlation_id, get_logger, set_correlation_id, setup_logging
from rag_platform.mcp import McpServer, create_mcp_app
from rag_platform.observability import AgentTrace, ObservabilityTracer, TraceSpan, get_tracer
from rag_platform.rag_system import RAGSystem
from rag_platform.vectorstore import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
    SupabaseVectorStore,
    get_embedding_model,
)

__version__ = "1.0.0"

__all__ = [
    "AgentError",
    "AgentResult",
    "AgentTrace",
    "ChatRequest",
    "ChatResponse",
    "ChunkMetadata",
    "ConfigurationError",
    "ContentType",
    "DocumentRecord",
    "ErrorResponse",
    "EvalReport",
    "EvalSample",
    "ExtractedChunk",
    "HealthResponse",
    "IngestResponse",
    "IngestionError",
    "IngestionResult",
    "McpClient",
    "McpError",
    "McpServer",
    "McpServerConfig",
    "McpToolDefinition",
    "ObservabilityTracer",
    "RagPlatformError",
    "ReadyResponse",
    "RetrievalError",
    "RAGSystem",
    "SampleEvalScore",
    "SearchFilter",
    "SearchResult",
    "Settings",
    "SupabaseVectorStore",
    "TraceSpan",
    "app",
    "calculator_tool",
    "create_app",
    "create_mcp_app",
    "get_correlation_id",
    "get_embedding_model",
    "get_logger",
    "get_settings",
    "get_tracer",
    "ingest_pdf",
    "retrieval_tool",
    "run_agent",
    "run_evaluation",
    "set_correlation_id",
    "setup_logging",
    "web_search_tool",
]
