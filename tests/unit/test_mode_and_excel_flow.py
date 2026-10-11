"""Focused regression tests for mode isolation (Q&A, Knowledge Base, Research) and Excel routing."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from rag_platform.api.main import app
from rag_platform.agent.mcp_client import McpClient, McpServerConfig
from rag_platform.agent.graph import create_agent_graph, run_agent, AgentResult
from rag_platform.vectorstore.supabase_store import SearchResult, SupabaseVectorStore


@pytest.fixture
def mock_vectorstore():
    """Mock SupabaseVectorStore returning canned SearchResult."""
    vs = MagicMock(spec=SupabaseVectorStore)
    vs.client = None
    vs.similarity_search.return_value = [
        SearchResult(
            chunk_id="chunk_1",
            doc_id="report_2025.pdf",
            content="Q3 Revenue reached $94.9 Billion with a gross margin of 46.2%.",
            metadata={"page_number": 4, "content_type": "text"},
            similarity=0.92,
            filename="report_2025.pdf",
            uploaded_at="2026-08-29T10:00:00Z",
        )
    ]
    return vs


@pytest.fixture
def mock_embeddings():
    """Mock embeddings callable returning a constant float vector."""
    return lambda q: [0.1] * 384


# ==============================================================================
# FIX 1: STRICT MODE ISOLATION TESTS
# ==============================================================================


def test_api_chat_qna_ignores_doc_filters_and_disables_retrieval():
    """Q&A mode must ignore doc_id, content_type, company_id, doc_filters and disable retrieval."""
    client = TestClient(app)
    mock_result = AgentResult(
        query="What is machine learning?",
        answer="Machine learning is a field of artificial intelligence.",
        routing_decision="direct",
        reasoning_path=["Router forced to 'direct' due to allow_retrieval=False"],
        sources=[],
        tool_outputs={},
    )

    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_result) as mock_run:
        payload = {
            "query": "What is machine learning?",
            "doc_id": "accidental-doc-id",
            "content_type": "table",
            "company_id": "accidental-company",
            "doc_filters": {"some_field": "val"},
            "mode": "Q&A",
            "stream": False,
        }
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Check API response
        assert data["routing_decision"] == "direct"
        assert not data.get("sources")

        # Verify run_agent was called with allow_retrieval=False and doc_filters=None
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("allow_retrieval") is False
        assert kwargs.get("doc_filters") is None
        assert kwargs.get("mode") == "Q&A"


def test_qna_mode_execution_isolation(mock_vectorstore, mock_embeddings):
    """Direct agent run in Q&A mode must not call vectorstore and must use general knowledge prompt."""
    captured_messages = []

    def mock_llm(messages, **kwargs):
        captured_messages.extend(messages)
        return "General knowledge answer."

    result = run_agent(
        query="What is photosynthesis?",
        doc_filters={"doc_id": "private_doc.pdf", "company_id": "confidential_corp"},
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        llm_call_fn=mock_llm,
        mode="Q&A",
    )

    # 1. Vectorstore similarity_search MUST NOT be called
    mock_vectorstore.similarity_search.assert_not_called()

    # 2. Routing decision must be direct and sources must be empty
    assert result.routing_decision == "direct"
    assert result.sources == []
    assert result.answer == "General knowledge answer."

    # 3. System prompt must NOT contain company identity or document-grounding constraints
    all_content = " ".join(m.get("content", "") for m in captured_messages)
    assert "confidential_corp" not in all_content
    assert "private_doc.pdf" not in all_content
    assert "general-purpose assistant" in all_content
    assert "without referencing any internal document context or company identity" in all_content


def test_knowledge_base_mode_retrieves_and_grounds(mock_vectorstore, mock_embeddings):
    """Knowledge Base mode must invoke retrieval and use document-grounded context."""
    captured_messages = []

    def mock_llm(messages, **kwargs):
        captured_messages.extend(messages)
        return "Q3 Revenue reached $94.9 Billion."

    result = run_agent(
        query="What was the revenue in the report?",
        doc_filters={"doc_id": "report_2025.pdf"},
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        llm_call_fn=mock_llm,
        mode="Knowledge Base",
    )

    # Similarity search MUST have been called
    mock_vectorstore.similarity_search.assert_called()
    assert result.routing_decision in ("retrieve", "hybrid")
    assert len(result.sources) > 0
    assert "report_2025.pdf" in result.sources[0]

    # Grounded evidence prompt must be present
    all_content = " ".join(m.get("content", "") for m in captured_messages)
    assert "grounded, evidence-first assistant" in all_content


def test_api_chat_research_requires_company():
    """Research mode requests without company_id must be rejected with 400 Bad Request."""
    client = TestClient(app)
    payload = {"query": "Summarize findings", "mode": "Research", "stream": False}
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 400
    assert "must include a company_id" in resp.json()["detail"]


def test_api_chat_research_scoped_retrieval():
    """Research mode requests with company_id must pass company_id in doc_filters."""
    client = TestClient(app)
    mock_result = AgentResult(
        query="Summarize findings",
        answer="Company findings summary.",
        routing_decision="retrieve",
        reasoning_path=["Company-scoped Research request"],
        sources=["doc.pdf (Page 1)"],
        tool_outputs={},
    )

    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_result) as mock_run:
        payload = {
            "query": "Summarize findings",
            "company_id": "comp-789",
            "mode": "Research",
            "stream": False,
        }
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("allow_retrieval") is True
        assert kwargs.get("doc_filters", {}).get("company_id") == "comp-789"
        assert kwargs.get("mode") == "Research"


def test_sse_streaming_preserves_qna_isolation():
    """SSE streaming in Q&A mode must yield events without retrieval or sources."""
    client = TestClient(app)
    mock_result = AgentResult(
        query="Hello AI",
        answer="Hello! How can I help you?",
        routing_decision="direct",
        reasoning_path=["Initialized agent run", "Router forced to 'direct'"],
        sources=[],
        tool_outputs={},
    )

    with patch("rag_platform.api.routes.v1.run_agent", return_value=mock_result) as mock_run:
        payload = {
            "query": "Hello AI",
            "doc_id": "accidental-doc",
            "mode": "Q&A",
            "stream": True,
        }
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "event" in text
        assert "token" in text
        assert "end" in text

        # Verify run_agent was called with allow_retrieval=False and doc_filters=None
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("allow_retrieval") is False
        assert kwargs.get("doc_filters") is None


# ==============================================================================
# FIX 2: EXCEL RUNTIME TRIGGER TESTS
# ==============================================================================


@pytest.mark.parametrize(
    "excel_query",
    [
        "give me all this in an excel file",
        "create an Excel file",
        "give me this in Excel",
        "export this to Excel",
        "make an xlsx",
        "put all this into an Excel file",
        "put these results in Excel",
        "make an Excel of the above",
    ],
)
def test_excel_queries_route_to_hybrid(excel_query, mock_vectorstore, mock_embeddings):
    """Natural-language Excel export requests must route to 'hybrid' so retrieval is run first."""
    def mock_llm(messages, **kwargs):
        # LLM router output simulates returning create_excel with natural-language tool_input
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return json.dumps({"decision": "mcp", "tool_name": "create_excel", "tool_input": excel_query})
        return ""

    graph = create_agent_graph(
        llm_call_fn=mock_llm,
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        allow_retrieval=True,
    )

    state = graph.invoke({
        "query": excel_query,
        "doc_filters": None,
        "routing_decision": "direct",
        "tool_name": None,
        "tool_input": None,
        "retrieval_context": None,
        "tool_output": None,
        "tool_data": None,
        "final_answer": None,
        "reasoning_path": [],
        "sources": [],
        "error": None,
    })

    assert state["routing_decision"] == "hybrid"
    assert state["tool_name"] == "create_excel"


def test_excel_direct_mcp_when_structured_data_present(mock_vectorstore, mock_embeddings):
    """If tool_input contains explicit structured JSON with non-empty data list, keep direct 'mcp' path."""
    structured_json = json.dumps({
        "filename": "custom_report.xlsx",
        "sheet_name": "Sheet1",
        "data": [{"Metric": "Revenue", "Value": 100}, {"Metric": "Margin", "Value": 0.45}],
    })

    def mock_llm(messages, **kwargs):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return json.dumps({"decision": "mcp", "tool_name": "create_excel", "tool_input": structured_json})
        return ""

    graph = create_agent_graph(
        llm_call_fn=mock_llm,
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        allow_retrieval=True,
    )

    state = graph.invoke({
        "query": "Export structured rows to excel",
        "doc_filters": None,
        "routing_decision": "direct",
        "tool_name": None,
        "tool_input": None,
        "retrieval_context": None,
        "tool_output": None,
        "tool_data": None,
        "final_answer": None,
        "reasoning_path": [],
        "sources": [],
        "error": None,
    })

    assert state["routing_decision"] == "mcp"
    assert state["tool_name"] == "create_excel"


def test_excel_blank_or_empty_request_preserves_mcp_without_retrieval(mock_vectorstore, mock_embeddings):
    """If user simply asks for an empty or blank Excel file, do not force retrieval."""
    def mock_llm(messages, **kwargs):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return json.dumps({"decision": "mcp", "tool_name": "create_excel", "tool_input": "blank excel file"})
        return ""

    graph = create_agent_graph(
        llm_call_fn=mock_llm,
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        allow_retrieval=True,
    )

    state = graph.invoke({
        "query": "Create a blank excel spreadsheet",
        "doc_filters": None,
        "routing_decision": "direct",
        "tool_name": None,
        "tool_input": None,
        "retrieval_context": None,
        "tool_output": None,
        "tool_data": None,
        "final_answer": None,
        "reasoning_path": [],
        "sources": [],
        "error": None,
    })

    assert state["routing_decision"] == "mcp"
    assert state["tool_name"] == "create_excel"


def test_agent_create_excel_full_flow_with_retrieval(mock_vectorstore, mock_embeddings):
    """Verify full end-to-end execution of 'Give me all this in an excel file' generates file output."""
    def mock_llm(messages, **kwargs):
        for m in messages:
            content = m.get("content", "")
            if "Classify the user query" in content:
                return json.dumps({"decision": "mcp", "tool_name": "create_excel", "tool_input": "make an excel file"})
            if "Extract tabular data for an Excel export" in content:
                return json.dumps({
                    "filename": "revenue_summary.xlsx",
                    "sheet_name": "Financials",
                    "data": [{"Metric": "Q3 Revenue", "Value": "$94.9 Billion"}, {"Metric": "Gross Margin", "Value": "46.2%"}],
                })
        return "I have generated the Excel file with your financial data."

    client = McpClient(servers=[McpServerConfig(name="excel")])
    result = run_agent(
        query="Give me all this in an excel file",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings,
        llm_call_fn=mock_llm,
        mcp_client=client,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "hybrid"
    assert "generated_file" in result.tool_outputs
    file_info = result.tool_outputs["generated_file"]
    assert file_info["filename"] == "revenue_summary.xlsx"
    assert file_info["row_count"] == 2
