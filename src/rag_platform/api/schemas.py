"""Pydantic schemas defining request and response models for the FastAPI service."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health status response."""

    status: str = Field(default="healthy", description="Service health indicator")
    app_name: str = Field(default="RAG Chatbot Platform", description="Application name")
    version: str = Field(default="1.0.0", description="API version")
    environment: str = Field(default="production", description="Runtime environment")


class ReadyResponse(BaseModel):
    """Readiness probe response indicating configuration and service readiness."""

    status: str = Field(default="ready", description="Readiness status (ready/degraded)")
    database_configured: bool = Field(..., description="Whether Supabase pgvector store is reachable")
    llm_configured: bool = Field(..., description="Whether LLM provider credentials are valid")
    checks: dict[str, bool] = Field(default_factory=dict, description="Detailed readiness checks")


class IngestResponse(BaseModel):
    """Response returned upon multimodal PDF ingestion."""

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Source filename")
    total_pages: int = Field(..., description="Total pages processed")
    total_chunks: int = Field(..., description="Total extracted chunks (text + tables + images)")
    tables_count: int = Field(default=0, description="Extracted table chunk count")
    images_count: int = Field(default=0, description="Extracted image chunk count")
    status: str = Field(default="completed", description="Ingestion processing status")


class ChatRequest(BaseModel):
    """Request payload submitted to /api/v1/chat endpoint."""

    query: str = Field(..., min_length=1, description="User question or prompt")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID filter scope")
    content_type: Optional[str] = Field(default=None, description="Optional chunk type filter (text/table/image)")
    mode: Optional[str] = Field(default=None, description="Client-selected UI mode: 'Knowledge Base'|'Research'|'Q&A'")
    company_id: Optional[str] = Field(default=None, description="Optional company identifier for research mode")
    doc_filters: Optional[dict[str, Any]] = Field(default=None, description="Arbitrary metadata filters")
    stream: bool = Field(default=False, description="Whether to stream response via Server-Sent Events (SSE)")


class ChatResponse(BaseModel):
    """Response returned from synchronous chat execution."""

    query: str = Field(..., description="User query submitted")
    answer: str = Field(..., description="Final synthesized agent response")
    routing_decision: str = Field(..., description="Router path (retrieve, tool, mcp, hybrid, direct)")
    reasoning_path: list[str] = Field(default_factory=list, description="Reasoning and action trace")
    sources: list[str] = Field(default_factory=list, description="Source documents and citations")
    tool_outputs: dict[str, Any] = Field(default_factory=dict, description="Outputs collected from tools")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation trace identifier")


class ErrorResponse(BaseModel):
    """Standardized error payload returned across all API exceptions."""

    error: str = Field(..., description="Human-readable error summary")
    error_type: str = Field(..., description="Error category class name")
    correlation_id: Optional[str] = Field(default=None, description="Associated correlation ID")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional context parameters")
