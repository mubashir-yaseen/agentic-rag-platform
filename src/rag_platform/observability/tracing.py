"""Observability and tracing layer supporting Langfuse, LangSmith, and structured local logging."""

import os
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from rag_platform.config import get_settings
from rag_platform.logging_config import get_correlation_id, get_logger

logger = get_logger(__name__)


class TraceSpan(BaseModel):
    """Sub-operation span within an execution trace."""

    name: str = Field(..., description="Span name (e.g., router, retriever, tool_caller, responder)")
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    latency_ms: float = 0.0
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, output: Optional[Any] = None, error: Optional[str] = None) -> None:
        """Mark the span as finished and compute duration."""
        self.end_time = time.time()
        self.latency_ms = round((self.end_time - self.start_time) * 1000, 2)
        if output is not None:
            self.output_data = output
        if error is not None:
            self.error = error


class AgentTrace(BaseModel):
    """Structured execution trace representing an agent run."""

    trace_id: str = Field(..., description="Unique correlation / trace ID")
    name: str = Field(default="agent_run", description="Name of the trace")
    query: str = Field(..., description="User query")
    routing_decision: Optional[str] = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[TraceSpan] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None


class ObservabilityTracer:
    """Manages recording and exporting traces to Langfuse, LangSmith, or structured logs."""

    def __init__(self) -> None:
        """Initialize tracer with configuration from Settings."""
        self.settings = get_settings()
        self.active_traces: dict[str, AgentTrace] = {}
        self._langfuse_client = None
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize remote tracing backends if credentials are configured."""
        # 1. Langfuse
        if self.settings.LANGFUSE_PUBLIC_KEY and self.settings.LANGFUSE_SECRET_KEY:
            try:
                from langfuse import Langfuse

                self._langfuse_client = Langfuse(
                    public_key=self.settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=self.settings.LANGFUSE_SECRET_KEY,
                    host=self.settings.LANGFUSE_HOST,
                )
                logger.info("Langfuse remote tracing initialized (host=%s)", self.settings.LANGFUSE_HOST)
            except Exception as exc:
                logger.warning("Langfuse client initialization skipped: %s", exc)

        # 2. LangSmith
        if self.settings.LANGSMITH_TRACING and self.settings.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.settings.LANGSMITH_PROJECT
            logger.info("LangSmith tracing enabled (project=%s)", self.settings.LANGSMITH_PROJECT)

    def start_trace(self, query: str, trace_id: Optional[str] = None, name: str = "agent_run") -> AgentTrace:
        """Create and register a new active trace."""
        t_id = trace_id or get_correlation_id()
        trace = AgentTrace(trace_id=t_id, query=query, name=name)
        self.active_traces[t_id] = trace
        logger.info("Observability trace started: trace_id=%s, name=%s", t_id, name)
        return trace

    def record_span(
        self,
        trace_id: str,
        name: str,
        start_time: float,
        output_data: Optional[Any] = None,
        input_data: Optional[Any] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceSpan:
        """Add a completed span to an active trace."""
        span = TraceSpan(
            name=name,
            start_time=start_time,
            input_data=input_data,
            metadata=metadata or {},
        )
        span.finish(output=output_data, error=error)

        trace = self.active_traces.get(trace_id)
        if trace:
            trace.spans.append(span)

        return span

    def end_trace(
        self,
        trace_id: str,
        final_answer: str,
        routing_decision: Optional[str] = None,
        retrieved_chunks: Optional[list[dict[str, Any]]] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentTrace:
        """Complete an active trace, compute metrics, and export."""
        trace = self.active_traces.get(trace_id)
        if not trace:
            trace = AgentTrace(trace_id=trace_id, query="")

        trace.end_time = time.time()
        trace.latency_ms = round((trace.end_time - trace.start_time) * 1000, 2)
        trace.final_answer = final_answer
        trace.routing_decision = routing_decision
        if retrieved_chunks is not None:
            trace.retrieved_chunks = retrieved_chunks
        if tool_calls is not None:
            trace.tool_calls = tool_calls
        if metadata:
            trace.metadata.update(metadata)

        total_tok = prompt_tokens + completion_tokens
        trace.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tok,
        }

        # Export to Langfuse if available
        if self._langfuse_client is not None:
            try:
                lf_trace = self._langfuse_client.trace(
                    id=trace.trace_id,
                    name=trace.name,
                    input={"query": trace.query},
                    output={"answer": trace.final_answer},
                    metadata={
                        "routing_decision": trace.routing_decision,
                        "latency_ms": trace.latency_ms,
                        "token_usage": trace.token_usage,
                        **trace.metadata,
                    },
                )
                for s in trace.spans:
                    lf_trace.span(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        input=s.input_data,
                        output=s.output_data,
                        status_message=s.error,
                    )
            except Exception as exc:
                logger.warning("Failed to export trace to Langfuse: %s", exc)

        logger.info(
            "Observability trace completed: trace_id=%s, decision=%s, latency=%.2fms",
            trace_id,
            routing_decision,
            trace.latency_ms,
        )

        return trace


_global_tracer: Optional[ObservabilityTracer] = None


def get_tracer() -> ObservabilityTracer:
    """Get or create singleton ObservabilityTracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = ObservabilityTracer()
    return _global_tracer
