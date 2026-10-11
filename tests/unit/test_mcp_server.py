"""Unit tests for the MCP Server implementation."""

from datetime import datetime
import json
from unittest.mock import MagicMock
import pytest

from rag_platform.exceptions import RetrievalError
from rag_platform.mcp.server import McpServer, create_mcp_app
from rag_platform.vectorstore.supabase_store import DocumentRecord, SearchResult, SupabaseVectorStore


@pytest.fixture
def mock_vectorstore():
    """Mock SupabaseVectorStore for MCP testing."""
    vs = MagicMock(spec=SupabaseVectorStore)
    vs.list_documents.return_value = [
        DocumentRecord(
            doc_id="doc_apple_2025",
            filename="AAPL_2025_10K.pdf",
            uploaded_at=datetime(2026, 8, 29, 10, 0, 0),
            doc_type="pdf",
            metadata={"pages": 45, "tables": 12},
        ),
        DocumentRecord(
            doc_id="doc_nvidia_2025",
            filename="NVDA_2025_Annual.pdf",
            uploaded_at=datetime(2026, 8, 28, 10, 0, 0),
            doc_type="pdf",
            metadata={"pages": 50, "tables": 18},
        ),
    ]
    vs.similarity_search.return_value = [
        SearchResult(
            chunk_id="chunk_aapl_1",
            doc_id="doc_apple_2025",
            content="Apple reported quarterly revenue of $94.9 billion, up 6 percent year over year.",
            metadata={"page_number": 2, "content_type": "text"},
            similarity=0.94,
            filename="AAPL_2025_10K.pdf",
            uploaded_at="2026-08-29T10:00:00Z",
        )
    ]
    return vs


@pytest.fixture
def mock_embeddings_fn():
    """Mock embeddings callable returning a constant vector."""
    return lambda q: [0.1] * 384


@pytest.fixture
def mock_llm_fn():
    """Mock LLM callable returning structured answers."""
    def _call(messages):
        for m in messages:
            content = m.get("content", "")
            if "Classify the user query" in content:
                return '{"decision": "retrieve", "tool_name": null, "tool_input": null}'
        return "Apple reported quarterly revenue of $94.9 billion, representing a 6% increase."
    return _call


# --- MCP TESTS ---


@pytest.mark.unit
def test_mcp_tool_definitions():
    """Verify MCP server exposes standard tool schemas with query_knowledge_base and list_documents."""
    server = McpServer()
    tools = server.get_tool_definitions()

    tool_names = [t["name"] for t in tools]
    assert "query_knowledge_base" in tool_names
    assert "list_documents" in tool_names

    q_tool = next(t for t in tools if t["name"] == "query_knowledge_base")
    assert "question" in q_tool["inputSchema"]["required"]
    assert "doc_filter" in q_tool["inputSchema"]["properties"]
    assert "content_type" in q_tool["inputSchema"]["properties"]


@pytest.mark.unit
def test_mcp_list_documents(mock_vectorstore):
    """Verify list_documents tool returns formatted metadata list."""
    server = McpServer(vectorstore=mock_vectorstore)
    docs = server.list_documents()

    assert len(docs) == 2
    assert docs[0]["doc_id"] == "doc_apple_2025"
    assert docs[0]["filename"] == "AAPL_2025_10K.pdf"
    assert docs[1]["doc_id"] == "doc_nvidia_2025"
    mock_vectorstore.list_documents.assert_called_once()


@pytest.mark.unit
def test_mcp_query_knowledge_base(mock_vectorstore, mock_embeddings_fn, mock_llm_fn):
    """Verify query_knowledge_base tool routes query to agent and returns formatted response."""
    server = McpServer(
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm_fn,
    )
    res = server.query_knowledge_base(
        question="What was Apple's reported quarterly revenue?",
        doc_filter="doc_apple_2025",
    )

    assert "94.9 billion" in res
    assert "Sources:" in res
    assert "AAPL_2025_10K.pdf" in res


@pytest.mark.unit
def test_mcp_query_empty_question():
    """Verify query_knowledge_base rejects empty question string."""
    server = McpServer()
    res = server.query_knowledge_base(question="")
    assert "Error:" in res


@pytest.mark.unit
def test_mcp_handle_tool_call_query(mock_vectorstore, mock_embeddings_fn, mock_llm_fn):
    """Verify handle_tool_call formats standard MCP JSON response."""
    server = McpServer(
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm_fn,
    )
    result = server.handle_tool_call(
        tool_name="query_knowledge_base",
        arguments={"question": "Revenue details?"},
    )

    assert result["isError"] is False
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert "94.9 billion" in result["content"][0]["text"]


@pytest.mark.unit
def test_mcp_handle_tool_call_list_docs(mock_vectorstore):
    """Verify handle_tool_call formats list_documents JSON response."""
    server = McpServer(vectorstore=mock_vectorstore)
    result = server.handle_tool_call(
        tool_name="list_documents",
        arguments={},
    )

    assert result["isError"] is False
    parsed_docs = json.loads(result["content"][0]["text"])
    assert len(parsed_docs) == 2
    assert parsed_docs[0]["doc_id"] == "doc_apple_2025"


@pytest.mark.unit
def test_mcp_handle_unknown_tool():
    """Verify handle_tool_call returns error flag for unknown tool names."""
    server = McpServer()
    result = server.handle_tool_call(
        tool_name="unknown_tool_xyz",
        arguments={},
    )
    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]


@pytest.mark.unit
def test_mcp_vectorstore_error_handling():
    """Verify domain errors from vectorstore surface cleanly without raw stack traces."""
    failing_vs = MagicMock(spec=SupabaseVectorStore)
    failing_vs.list_documents.side_effect = RetrievalError("Database unreachable")

    server = McpServer(vectorstore=failing_vs)
    docs = server.list_documents()

    assert len(docs) == 1
    assert "error" in docs[0]
    assert "Database unreachable" in docs[0]["error"]


@pytest.mark.unit
def test_create_mcp_app_instantiation():
    """Verify create_mcp_app returns an initialized server or FastMCP instance."""
    app = create_mcp_app()
    assert app is not None
