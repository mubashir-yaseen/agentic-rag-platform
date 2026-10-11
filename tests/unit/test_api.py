"""Integration and unit tests for FastAPI endpoints, middleware, and error handling."""

import io
import json
from unittest.mock import MagicMock, patch
import fitz
import pytest
from fastapi.testclient import TestClient

from rag_platform.agent.graph import AgentResult
from rag_platform.api.main import app
from rag_platform.api.middleware import InMemoryRateLimiter, RateLimitingMiddleware


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    """Create in-memory single-page PDF bytes for upload testing."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Annual Financial Report 2023. Total revenue was $124.5 million.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


# --- HEALTH & READINESS ENDPOINTS ---


@pytest.mark.unit
def test_health_check_endpoint(client):
    """Verify /api/v1/health returns 200 OK and valid health payload."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "app_name" in data
    assert "X-Correlation-ID" in resp.headers


@pytest.mark.unit
def test_readiness_check_endpoint(client):
    """Verify /api/v1/ready returns status and dependency checks."""
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ready", "degraded")
    assert "database_configured" in data
    assert "llm_configured" in data
    assert "checks" in data


# --- CHAT ENDPOINTS ---


@pytest.mark.unit
def test_chat_endpoint_sync_calculation(client):
    """Verify synchronous chat endpoint processes math calculation query."""
    mock_calc_result = AgentResult(
        query="What is 15 + 25?",
        answer="The result of 15 + 25 is 40.",
        routing_decision="tool",
        reasoning_path=["Router: tool (calculator)", "Calculator output: 40"],
        sources=[],
        tool_outputs={"calculator": "40"},
    )
    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_calc_result):
        payload = {"query": "What is 15 + 25?", "stream": False}
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is 15 + 25?"
        assert data["routing_decision"] in ("tool", "direct")
        assert "40" in data["answer"] or "calculator" in data["tool_outputs"]
        assert "correlation_id" in data


@pytest.mark.unit
def test_chat_endpoint_streaming_sse(client):
    """Verify streaming chat endpoint returns Server-Sent Events (SSE)."""
    mock_calc_result = AgentResult(
        query="Calculate 100 * 5",
        answer="500",
        routing_decision="tool",
        reasoning_path=["Router: tool", "Calculator output: 500"],
        sources=[],
        tool_outputs={"calculator": "500"},
    )
    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_calc_result):
        payload = {"query": "Calculate 100 * 5", "stream": True}
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        content = resp.text
        assert "data:" in content
        assert "start" in content
        assert "end" in content


@pytest.mark.unit
def test_chat_endpoint_empty_query_validation(client):
    """Verify empty query returns 422 Unprocessable Entity."""
    resp = client.post("/api/v1/chat", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.unit
def test_chat_endpoint_grounded_answer_no_placeholder(client):
    """Regression test ensuring chat endpoint returns real grounded answer and not placeholder."""
    mock_result = AgentResult(
        query="What is the proposed research problem?",
        answer="The proposed research problem addresses contextual financial table reconstruction using multimodal vision-language models.",
        routing_decision="retrieve",
        reasoning_path=["Router: retrieve", "Retriever executed", "Responder synthesized"],
        sources=["thesis_proposal.pdf (Page 2)"],
        tool_outputs={"retriever": "Sample context"},
    )
    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_result):
        payload = {"query": "What is the proposed research problem?", "stream": False}
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert not data["answer"].startswith("Response to:")
        assert "financial table reconstruction" in data["answer"]
        assert data["routing_decision"] == "retrieve"


# --- INGESTION ENDPOINTS ---


@pytest.mark.unit
def test_ingest_pdf_happy_path(client, sample_pdf_bytes):
    """Verify uploading a valid PDF returns 201 Created with document metadata."""
    files = {"file": ("financial_report.pdf", sample_pdf_bytes, "application/pdf")}
    mock_settings = MagicMock()
    mock_settings.SUPABASE_URL = ""
    mock_settings.effective_supabase_key = ""
    with patch("rag_platform.api.routes.v1.get_settings", return_value=mock_settings):
        resp = client.post("/api/v1/ingest", files=files, data={"chunk_size": 500, "chunk_overlap": 100})
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "financial_report.pdf"
        assert data["total_pages"] == 1
        assert data["total_chunks"] >= 1
        assert data["status"] == "completed"


@pytest.mark.unit
def test_ingest_pdf_generates_embeddings_and_upserts_to_supabase(client, sample_pdf_bytes):
    """Verify uploading a valid PDF generates dense embeddings and calls upsert_chunks with vectors."""
    files = {"file": ("financial_report.pdf", sample_pdf_bytes, "application/pdf")}
    
    mock_settings = MagicMock()
    mock_settings.SUPABASE_URL = "https://mock.supabase.co"
    mock_settings.effective_supabase_key = "mock-key"
    mock_settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    mock_hf = MagicMock()
    # Mock embed_documents to return 384-dimensional fake embedding vector for each chunk
    mock_hf.embed_documents.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    
    with patch("rag_platform.api.routes.v1.get_settings", return_value=mock_settings), \
         patch("rag_platform.api.routes.v1.get_embedding_model", return_value=mock_hf), \
         patch("rag_platform.api.routes.v1.SupabaseVectorStore") as mock_vstore_cls:
        
        mock_vstore_instance = MagicMock()
        mock_vstore_cls.return_value = mock_vstore_instance
        
        resp = client.post("/api/v1/ingest", files=files, data={"chunk_size": 500, "chunk_overlap": 100})
        assert resp.status_code == 201
        
        # Verify Supabase upsert_document called
        assert mock_vstore_instance.upsert_document.called
        
        # Verify Supabase upsert_chunks called and every chunk has non-None embedding
        assert mock_vstore_instance.upsert_chunks.called
        args, kwargs = mock_vstore_instance.upsert_chunks.call_args
        upserted_chunks = kwargs.get("chunks") or args[0]
        assert len(upserted_chunks) > 0
        for chunk in upserted_chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384


@pytest.mark.unit
def test_ingest_pdf_embedding_failure_raises_error_and_does_not_upsert_null(client, sample_pdf_bytes):
    """Verify failure in embedding generation fails ingestion and does not upsert NULL embeddings."""
    files = {"file": ("financial_report.pdf", sample_pdf_bytes, "application/pdf")}
    
    mock_settings = MagicMock()
    mock_settings.SUPABASE_URL = "https://mock.supabase.co"
    mock_settings.effective_supabase_key = "mock-key"
    mock_settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    mock_hf = MagicMock()
    mock_hf.embed_documents.side_effect = RuntimeError("Embedding model runtime failure")
    
    with patch("rag_platform.api.routes.v1.get_settings", return_value=mock_settings), \
         patch("rag_platform.api.routes.v1.get_embedding_model", return_value=mock_hf), \
         patch("rag_platform.api.routes.v1.SupabaseVectorStore") as mock_vstore_cls:
        
        mock_vstore_instance = MagicMock()
        mock_vstore_cls.return_value = mock_vstore_instance
        
        resp = client.post("/api/v1/ingest", files=files, data={"chunk_size": 500, "chunk_overlap": 100})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error_type"] == "IngestionError"
        assert "Failed to generate dense vector embeddings" in data["error"]
        
        # Ensure chunks were NOT upserted with NULL vectors
        assert not mock_vstore_instance.upsert_chunks.called


@pytest.mark.unit
def test_ingest_pdf_zero_chunks_safe_handling(client, sample_pdf_bytes):
    """Verify uploading a PDF that produces zero chunks handles gracefully without error."""
    files = {"file": ("empty_content.pdf", sample_pdf_bytes, "application/pdf")}
    
    mock_settings = MagicMock()
    mock_settings.SUPABASE_URL = "https://mock.supabase.co"
    mock_settings.effective_supabase_key = "mock-key"
    mock_settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Mock ingest_pdf returning 0 chunks
    from rag_platform.ingestion.models import IngestionResult
    mock_result = IngestionResult(
        source_doc="empty_content.pdf",
        total_pages=1,
        chunks=[],
        text_chunks_count=0,
        tables_count=0,
        images_count=0,
    )
    
    with patch("rag_platform.api.routes.v1.get_settings", return_value=mock_settings), \
         patch("rag_platform.api.routes.v1.ingest_pdf", return_value=mock_result), \
         patch("rag_platform.api.routes.v1.SupabaseVectorStore") as mock_vstore_cls:
        
        mock_vstore_instance = MagicMock()
        mock_vstore_cls.return_value = mock_vstore_instance
        
        resp = client.post("/api/v1/ingest", files=files, data={"chunk_size": 500, "chunk_overlap": 100})
        assert resp.status_code == 201
        data = resp.json()
        assert data["total_chunks"] == 0
        assert data["status"] == "completed"
        
        # upsert_document should be called, but upsert_chunks should NOT be called for 0 chunks
        assert mock_vstore_instance.upsert_document.called
        assert not mock_vstore_instance.upsert_chunks.called


@pytest.mark.unit
def test_ingest_non_pdf_file_error(client):
    """Verify uploading a non-PDF file returns 400 Bad Request with IngestionError."""
    files = {"file": ("readme.txt", b"Hello world text file", "text/plain")}
    resp = client.post("/api/v1/ingest", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_type"] == "IngestionError"
    assert "Only PDF files are supported" in data["error"]


@pytest.mark.unit
def test_ingest_empty_pdf_error(client):
    """Verify uploading an empty 0-byte file returns 400 Bad Request."""
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    resp = client.post("/api/v1/ingest", files=files)
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_type"] == "IngestionError"
    assert "empty" in data["error"].lower()


# --- EVALUATION BENCHMARK ENDPOINT ---


@pytest.mark.unit
def test_eval_benchmark_endpoint(client):
    """Verify /api/v1/eval executes evaluation and returns aggregated metrics."""
    resp = client.get("/api/v1/eval?limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 2
    assert "mean_faithfulness" in data
    assert "mean_answer_relevance" in data
    assert "mean_context_precision" in data
    assert "mean_overall_score" in data
    assert len(data["sample_scores"]) == 2


# --- MIDDLEWARE & HEADERS TESTS ---


@pytest.mark.unit
def test_correlation_id_forwarding(client):
    """Verify custom X-Correlation-ID header is propagated through responses."""
    custom_id = "test-corr-custom-abc123"
    resp = client.get("/api/v1/health", headers={"X-Correlation-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Correlation-ID") == custom_id


@pytest.mark.unit
def test_rate_limiter_triggers_429():
    """Verify InMemoryRateLimiter detects when threshold is exceeded."""
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60.0)
    client_ip = "127.0.0.1"

    assert limiter.is_rate_limited(client_ip) is False
    assert limiter.is_rate_limited(client_ip) is False
    assert limiter.is_rate_limited(client_ip) is False
    assert limiter.is_rate_limited(client_ip) is True
