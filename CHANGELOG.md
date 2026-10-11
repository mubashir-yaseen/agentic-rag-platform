# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-29 - Phase 10: Documentation & Polish

### Added
- Completed flagship `README.md` with visual Mermaid architecture diagrams, detailed feature guides, UI walkthroughs, and a "Why This Project?" principal-engineering portfolio summary.
- Comprehensive `CONTRIBUTING.md` developer guide covering local environment setup, pre-commit hooks, testing guidelines, and pull request workflows.
- Standard MIT `LICENSE` file.
- Final consistency review verifying `.env.example`, `pyproject.toml`, structured exceptions, logging correlation IDs, and test suites across all 10 project upgrade phases.

## [0.10.0] - 2026-08-29 - Phase 9: Docker & CI/CD Pipeline

### Added
- Multi-stage `Dockerfile` for the FastAPI backend service with non-root user security and health checks.
- Multi-stage `Dockerfile.ui` for the Streamlit web client.
- Orchestration configurations in `docker-compose.yml` and `docker/docker-compose.yml` for unified local multi-service runtime.
- GitHub Actions continuous integration pipeline in `.github/workflows/ci.yml` running:
  - Ruff linting and formatting validation.
  - Mypy static type checking across `src/`.
  - Pytest unit test execution with coverage reporting across Python 3.11 and 3.12.
  - Multi-stage Docker image build validation for backend API and frontend UI.
- CI status, coverage, Python version, and MIT license badges added to `README.md`.

## [0.9.0] - 2026-08-29 - Phase 8: Production FastAPI Backend

### Added
- Production FastAPI service in `rag_platform.api.main` with versioned routing under `/api/v1`:
  - `POST /api/v1/chat`: Multi-document agentic chat supporting synchronous JSON response and real-time Server-Sent Events (SSE) streaming.
  - `POST /api/v1/ingest`: Multimodal PDF file ingestion endpoint returning structured `IngestResponse`.
  - `GET /api/v1/eval`: Endpoint executing automated RAGAS evaluation benchmarks on demand.
  - `GET /api/v1/health`: Health status endpoint returning version and environment info.
  - `GET /api/v1/ready`: Readiness probe checking Supabase and LLM API configurations.
- Pydantic validation schemas in `rag_platform.api.schemas` (`ChatRequest`, `ChatResponse`, `IngestResponse`, `HealthResponse`, `ReadyResponse`, `ErrorResponse`).
- Correlation ID tracking middleware (`CorrelationIdMiddleware`) propagating request IDs through logs, contextvars, and HTTP headers (`X-Correlation-ID`).
- Sliding-window in-memory rate limiting middleware (`RateLimitingMiddleware`).
- Centralized exception handlers mapping domain exceptions (`ConfigurationError`, `IngestionError`, `RetrievalError`, `AgentError`, `McpError`) to structured JSON error responses with appropriate HTTP status codes.
- Updated Streamlit UI (`ui/app.py`) to connect to the FastAPI backend with seamless local fallback.
- Comprehensive integration test suite in `tests/unit/test_api.py` using `TestClient` covering all endpoints, streaming SSE, validation errors, and middleware.

## [0.8.0] - 2026-08-29 - Phase 7: Evaluation & Observability Layer

### Added
- Observability and tracing module in `rag_platform.observability.tracing`:
  - `AgentTrace` and `TraceSpan` models for structured trace capture.
  - `ObservabilityTracer` singleton capturing router decisions, retriever spans, tool calls, responder outputs, latency metrics, and token usage.
  - Native export support for Langfuse (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) and LangSmith (`LANGSMITH_TRACING`).
- RAGAS-compatible evaluation harness in `rag_platform.eval.run_eval`:
  - `compute_faithfulness()`: Measures factual grounding of generated answers in retrieved context.
  - `compute_answer_relevance()`: Measures question alignment and reference similarity.
  - `compute_context_precision()`: Evaluates source citation recall against expected ground truth.
  - `run_evaluation()`: Importable function and CLI command generating comprehensive `EvalReport`.
- Benchmark evaluation dataset in `src/rag_platform/eval/testset.json` with 12 labeled multi-document Q&A pairs.
- Automatic trace span generation hooked into `run_agent()` execution flow.
- Unit test suite in `tests/unit/test_eval_observability.py` testing tracing, metric calculation accuracy, testset evaluation, and limit handling.

## [0.7.0] - 2026-08-29 - Phase 6: MCP Client Integration

### Added
- MCP Client layer in `rag_platform.agent.mcp_client.McpClient`:
  - `McpServerConfig` and `McpToolDefinition` data models.
  - JSON configuration loader supporting `mcp_config.json`.
  - Tool discovery (`discover_tools()`) and dynamic invocation (`call_tool()`) with retry mechanisms and graceful fallback.
  - Support for `filesystem` (`read_file`) and `web_search` (`mcp_web_search`) servers.
- LangGraph agent graph updates in `rag_platform.agent.graph`:
  - Router node extended to identify and dispatch external MCP tool requests.
  - Tool caller node updated to execute discovered MCP client tools alongside local arithmetic and search tools.
- External configuration file `mcp_config.json` defining standard MCP servers.
- Unit test suite in `tests/unit/test_mcp_client.py` covering config loading, tool discovery, tool execution, fallback handling, and agent router dispatch.

## [0.6.0] - 2026-08-29 - Phase 5: MCP Server Integration

### Added
- Model Context Protocol (MCP) Server in `rag_platform.mcp.server.McpServer`:
  - `query_knowledge_base`: Standard MCP tool exposing multi-document multimodal search and agent synthesis.
  - `list_documents`: Tool returning structured metadata for all registered documents.
  - Tool execution dispatcher and JSON-RPC stdio transport interface (`create_mcp_app()`, `main()`).
- Robust error boundary formatting domain errors cleanly for external MCP clients without raw stack traces.
- Unit test suite in `tests/unit/test_mcp_server.py` covering tool schema registration, `query_knowledge_base` execution, `list_documents` serialization, and error handling.
- Claude Desktop and Claude Code configuration guide with sample `mcpServers` JSON block in `README.md`.

## [0.5.0] - 2026-08-29 - Phase 4: LangGraph Agent Layer

### Added
- Modular agent tools in `rag_platform.agent.tools`:
  - `retrieval_tool()`: similarity search with metadata filtering against Supabase pgvector.
  - `calculator_tool()`: safe AST-evaluated math computation without python `eval`/`exec`.
  - `web_search_tool()`: real-time web search via DuckDuckGo with resilient fallback.
- State graph architecture in `rag_platform.agent.graph` using LangGraph (`StateGraph`, `AgentState`, `AgentResult`):
  - `router`: Dynamic LLM classification routing between `retrieve`, `tool`, `hybrid`, and `direct`.
  - `retriever`: Fetches and formats context chunks and document sources.
  - `tool_caller`: Executes non-retrieval tools (calculator, web search).
  - `responder`: Synthesizes final answer grounded in context/tool outputs with citations.
- Public agent runner `run_agent(query: str, doc_filters: dict | None, ...)` returning structured `AgentResult`.
- Unit tests in `tests/unit/test_agent.py` testing pure retrieval queries, tool-only calculations, hybrid retrieval+calculation flows, direct answering, and empty query error validation.

## [0.4.0] - 2026-08-29 - Phase 3: Multi-Document Knowledge Base on Supabase

### Added
- SQL migration `001_create_documents_and_chunks.sql` creating `documents` table, `chunks` table with `pgvector(384)`, HNSW vector indexes, and `match_chunks` RPC function with multi-attribute filtering (document ID, content type, upload timestamp range).
- Persistent multi-document store implementation in `rag_platform.vectorstore.supabase_store.SupabaseVectorStore` providing `upsert_document()`, `upsert_chunks()`, `similarity_search()`, `delete_document()`, `list_documents()`, and `get_document()`.
- Exponential backoff retry utility `with_retry()` for resilient database transactions on transient network failures.
- Typed schemas in `rag_platform.vectorstore`: `DocumentRecord`, `SearchFilter`, and `SearchResult`.
- Multi-document selection UI in `ui/app.py` allowing users to filter queries to specific documents or perform global knowledge-base search, alongside content type filters (text, tables, images).
- Unit tests in `tests/unit/test_vectorstore.py` with mocked Supabase client covering document upsert, chunk batch upsert, filtered similarity search, deletion, listing, retrieval error handling, and retry exhaustion.

## [0.3.0] - 2026-08-29 - Phase 2: Multimodal Ingestion

### Added
- Strongly-typed Pydantic schemas in `rag_platform.ingestion.models`: `ContentType`, `ChunkMetadata`, `ExtractedChunk`, and `IngestionResult`.
- Tabular data extraction module `rag_platform.ingestion.tables` transforming detected PDF tables into structured Markdown chunks tagged `content_type=ContentType.TABLE`.
- Image and figure extraction module `rag_platform.ingestion.images` supporting image disk persistence, dimensions filtering, and vision LLM caption generation with offline fallbacks.
- Optical Character Recognition (OCR) fallback module `rag_platform.ingestion.ocr` automatically identifying scanned/image-only pages.
- Pure-function multimodal ingestion coordinator `rag_platform.ingestion.pipeline.ingest_pdf` handling error validation, structured step logging, and metadata tagging.
- Comprehensive unit test suite in `tests/unit/test_ingestion.py` covering prose text PDFs, tabular structures, scanned image OCR fallback, image captioning, corrupt file exceptions, and empty/missing files.

## [0.2.0] - 2026-08-29 - Phase 1: Repository Foundations

### Added
- Standardized `src/rag_platform/` package architecture with subpackages for `ingestion/`, `vectorstore/`, `agent/`, `mcp/`, `api/`, `eval/`, and `observability/`.
- Centralized configuration management using `pydantic-settings` in `rag_platform.config.Settings`.
- Domain-specific exception hierarchy in `rag_platform.exceptions` (`RagPlatformError`, `ConfigurationError`, `IngestionError`, `RetrievalError`, `AgentError`, `McpError`).
- Structured logging configuration with JSON formatting and asynchronous correlation ID propagation in `rag_platform.logging_config`.
- Project metadata, dependency definitions, `ruff` linter/formatter, `mypy`, and `pytest` configurations in `pyproject.toml`.
- Pre-commit hook configuration in `.pre-commit-config.yaml`.
- Comprehensive unit and integration test scaffolding under `tests/unit/` and `tests/integration/`.
- Dedicated UI module in `ui/app.py` with backward-compatible shims for root entrypoints.
- Initial Architecture documentation and updated Setup guide in `README.md`.
