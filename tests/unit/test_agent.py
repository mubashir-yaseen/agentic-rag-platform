"""Unit tests for LangGraph agent routing, tools, and execution flows."""

import math
from unittest.mock import MagicMock
import pytest

from rag_platform.agent.graph import AgentResult, create_agent_graph, run_agent
from rag_platform.agent.tools import (
    ToolResult,
    _compute_keyword_relevance,
    _expand_adjacent_context_chunks,
    _phrase_equivalent_match,
    calculator_tool,
    retrieval_tool,
    web_search_tool,
)
from rag_platform.exceptions import AgentError
from rag_platform.vectorstore.supabase_store import SearchResult, SupabaseVectorStore


@pytest.fixture
def mock_vectorstore():
    """Create a mock SupabaseVectorStore returning predefined SearchResult items."""
    vs = MagicMock(spec=SupabaseVectorStore)
    vs.client = None  # Disable live lexical scan in unit tests
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
def mock_embeddings_fn():
    """Mock embeddings callable returning a constant float vector."""
    return lambda q: [0.1] * 384


# --- TOOL TESTS ---


@pytest.mark.unit
def test_calculator_tool_valid_expressions():
    """Verify calculator tool accurately parses and evaluates safe arithmetic."""
    res1 = calculator_tool("12 * 8 + 4")
    assert res1.success is True
    assert res1.output == "100"

    res2 = calculator_tool("(250 - 50) / 4")
    assert res2.success is True
    assert res2.output == "50"

    res3 = calculator_tool("What is 15.5 * 2?")
    assert res3.success is True
    assert res3.output == "31"


@pytest.mark.unit
def test_calculator_tool_invalid_expression():
    """Verify calculator tool handles syntax errors gracefully."""
    res = calculator_tool("import os; os.system('ls')")
    assert res.success is False
    assert "error" in res.output.lower()


@pytest.mark.unit
def test_web_search_tool_execution():
    """Verify web search tool executes cleanly."""
    res = web_search_tool("latest AI developments 2026")
    assert res.success is True
    assert isinstance(res.output, str)
    assert len(res.output) > 0


@pytest.mark.unit
def test_retrieval_tool_execution(mock_vectorstore, mock_embeddings_fn):
    """Verify retrieval tool executes similarity search and returns formatted context."""
    res = retrieval_tool(
        query="What was the Q3 Revenue?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        k=2,
    )
    assert res.success is True
    assert "94.9 Billion" in res.output
    assert res.data["count"] == 1
    mock_vectorstore.similarity_search.assert_called_once()


@pytest.mark.unit
def test_retrieval_tool_lexical_prefilter_uses_any_keyword_or_semantics(mock_embeddings_fn):
    """The lexical candidate filter must be recall-safe and keyword-OR rather than AND-based."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()
    table = MagicMock()
    vectorstore.client.table.return_value = table
    table.select.return_value = table
    table.eq.return_value = table
    table.or_.return_value = table
    table.range.return_value = MagicMock(data=[{
        "chunk_id": "candidate_chunk",
        "doc_id": "doc_1",
        "content": "This section discusses research methodology and data collection.",
        "metadata": {"page_number": 5, "content_type": "text"},
    }])
    vectorstore.similarity_search.return_value = [
        SearchResult(
            chunk_id="candidate_chunk",
            doc_id="doc_1",
            content="This section discusses research methodology and data collection.",
            metadata={"page_number": 5, "content_type": "text"},
            similarity=0.91,
            filename="report.pdf",
            uploaded_at="2026-08-29T10:00:00Z",
        )
    ]

    res = retrieval_tool(
        query="What is the research problem?",
        vectorstore=vectorstore,
        embeddings_fn=mock_embeddings_fn,
        k=1,
        filters={"doc_id": "doc_1"},
    )

    assert res.success is True
    table.or_.assert_called_once()
    or_clause = table.or_.call_args[0][0]
    assert "research" in or_clause.lower()
    assert "problem" in or_clause.lower()


@pytest.mark.unit
def test_retrieval_tool_lexical_prefilter_falls_back_when_empty(mock_embeddings_fn):
    """If the server-side prefilter has zero hits, retrieval must fall back to the full lexical pool."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()
    table = MagicMock()
    vectorstore.client.table.return_value = table
    table.select.return_value = table
    table.eq.return_value = table
    table.or_.return_value = table
    table.range.side_effect = [
        MagicMock(data=[]),
        MagicMock(data=[{
            "chunk_id": "fallback_chunk",
            "doc_id": "doc_1",
            "content": "The specific research problem is to reduce page-level table ambiguity.",
            "metadata": {"page_number": 7, "content_type": "text"},
        }]),
    ]
    vectorstore.similarity_search.return_value = [
        SearchResult(
            chunk_id="fallback_chunk",
            doc_id="doc_1",
            content="The specific research problem is to reduce page-level table ambiguity.",
            metadata={"page_number": 7, "content_type": "text"},
            similarity=0.88,
            filename="report.pdf",
            uploaded_at="2026-08-29T10:00:00Z",
        )
    ]

    res = retrieval_tool(
        query="What is the proposed research problem?",
        vectorstore=vectorstore,
        embeddings_fn=mock_embeddings_fn,
        k=1,
        filters={"doc_id": "doc_1"},
    )

    assert res.success is True
    assert res.data["count"] == 1
    assert "research problem" in res.output.lower()
    assert table.range.call_count >= 2


@pytest.mark.unit
def test_expand_adjacent_context_chunks_includes_same_page_continuations():
    """The first chunk in a page sequence should pull in immediately adjacent continuation chunks from the same document/page."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()

    page_rows = [
        {
            "chunk_id": "doc_1_p10_txt0_100",
            "doc_id": "doc_1",
            "content": "part one",
            "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 0}},
        },
        {
            "chunk_id": "doc_1_p10_txt1_101",
            "doc_id": "doc_1",
            "content": "part two",
            "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 1}},
        },
        {
            "chunk_id": "doc_1_p10_txt2_102",
            "doc_id": "doc_1",
            "content": "part three",
            "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 2}},
        },
        {
            "chunk_id": "doc_2_p10_txt0_200",
            "doc_id": "doc_2",
            "content": "other document",
            "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 0}},
        },
        {
            "chunk_id": "doc_1_p11_txt0_103",
            "doc_id": "doc_1",
            "content": "next page",
            "metadata": {"page_number": 11, "content_type": "text", "extra": {"split_index": 0}},
        },
    ]
    chunk_query = MagicMock()
    chunk_query.execute.return_value = MagicMock(data=page_rows)
    vectorstore.client.table.return_value.select.return_value.eq.return_value.eq.return_value = chunk_query

    initial = [
        SearchResult(
            chunk_id="doc_1_p10_txt0_100",
            doc_id="doc_1",
            content="part one",
            metadata={"page_number": 10, "content_type": "text"},
            similarity=0.99,
            filename="doc_1.pdf",
            uploaded_at="2025-01-01",
        )
    ]

    expanded = _expand_adjacent_context_chunks(vectorstore, initial, max_neighbors=2)
    ids = [chunk.chunk_id for chunk in expanded]
    assert "doc_1_p10_txt0_100" in ids
    assert "doc_1_p10_txt1_101" in ids
    assert "doc_1_p10_txt2_102" in ids
    assert "doc_2_p10_txt0_200" not in ids
    assert "doc_1_p11_txt0_103" not in ids


@pytest.mark.unit
def test_expand_adjacent_context_chunks_ignores_distant_or_duplicate_neighbors():
    """Only immediate same-page neighbors should be included, and duplicate chunks should be deduplicated."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()
    page_rows = [
        {"chunk_id": "doc_1_p10_txt0_100", "doc_id": "doc_1", "content": "a", "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 0}}},
        {"chunk_id": "doc_1_p10_txt1_101", "doc_id": "doc_1", "content": "b", "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 1}}},
        {"chunk_id": "doc_1_p10_txt2_102", "doc_id": "doc_1", "content": "c", "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 2}}},
        {"chunk_id": "doc_1_p10_txt4_104", "doc_id": "doc_1", "content": "d", "metadata": {"page_number": 10, "content_type": "text", "extra": {"split_index": 4}}},
    ]
    chunk_query = MagicMock()
    chunk_query.execute.return_value = MagicMock(data=page_rows)
    vectorstore.client.table.return_value.select.return_value.eq.return_value.eq.return_value = chunk_query

    initial = [
        SearchResult(chunk_id="doc_1_p10_txt0_100", doc_id="doc_1", content="a", metadata={"page_number": 10, "content_type": "text"}, similarity=0.99, filename="doc_1.pdf", uploaded_at="2025-01-01"),
        SearchResult(chunk_id="doc_1_p10_txt1_101", doc_id="doc_1", content="b", metadata={"page_number": 10, "content_type": "text"}, similarity=0.90, filename="doc_1.pdf", uploaded_at="2025-01-01"),
    ]

    expanded = _expand_adjacent_context_chunks(vectorstore, initial, max_neighbors=1)
    ids = [chunk.chunk_id for chunk in expanded]
    assert ids.count("doc_1_p10_txt1_101") == 1
    assert "doc_1_p10_txt2_102" in ids
    assert "doc_1_p10_txt4_104" not in ids


@pytest.mark.unit
def test_run_agent_company_scoped_context_includes_selected_company_identity():
    """Company-scoped requests must include the selected company name and symbol in the responder context."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()
    documents_query = MagicMock()
    documents_query.limit.return_value = documents_query
    documents_query.execute.return_value = MagicMock(data=[{
        "doc_id": "doc_1",
        "metadata": {"name": "Bank Al-Habib Limited", "symbol": "BAHL", "company_id": "company-1", "scope": "company", "report_type": "annual_report"},
    }])
    vectorstore.client.table.return_value.select.return_value.filter.return_value = documents_query
    vectorstore.similarity_search.return_value = [
        SearchResult(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            content="Annual report details",
            metadata={"page_number": 5, "content_type": "text"},
            similarity=0.88,
            filename="doc_1.pdf",
            uploaded_at="2025-01-01",
        )
    ]

    captured = {}

    def mock_llm(messages):
        captured["prompt"] = messages[0]["content"]
        return "The selected company is Bank Al-Habib Limited (BAHL)."

    result = run_agent(
        query="What is the 2025 deposit split?",
        doc_filters={"company_id": "company-1"},
        vectorstore=vectorstore,
        embeddings_fn=lambda q: [0.1] * 384,
        llm_call_fn=mock_llm,
    )

    assert result.routing_decision == "retrieve"
    assert "Bank Al-Habib Limited" in captured["prompt"]
    assert "BAHL" in captured["prompt"]
    assert "authoritative" in captured["prompt"].lower()


@pytest.mark.unit
def test_run_agent_non_company_requests_do_not_add_company_identity():
    """Non-company requests should not gain unintended company identity context."""
    vectorstore = MagicMock(spec=SupabaseVectorStore)
    vectorstore.client = MagicMock()
    vectorstore.similarity_search.return_value = [
        SearchResult(
            chunk_id="doc_1_chunk_1",
            doc_id="doc_1",
            content="Annual report details",
            metadata={"page_number": 5, "content_type": "text"},
            similarity=0.88,
            filename="doc_1.pdf",
            uploaded_at="2025-01-01",
        )
    ]

    captured = {}

    def mock_llm(messages):
        captured["prompt"] = messages[0]["content"]
        return "The report mentions 2025 activity."

    run_agent(
        query="What is in the annual report?",
        vectorstore=vectorstore,
        embeddings_fn=lambda q: [0.1] * 384,
        llm_call_fn=mock_llm,
    )

    assert "Selected company" not in captured["prompt"]
    assert "Bank Al-Habib Limited" not in captured["prompt"]


@pytest.mark.unit
def test_phrase_equivalence_accumulates_all_applicable_matches():
    """A query containing both phrase pairs should accumulate both bonuses exactly once each."""
    query = "current account and saving account"
    text = "current deposits and savings deposits"
    expected = 2 * (3.5 + math.log(2))
    score = _phrase_equivalent_match(query, text)
    assert score == pytest.approx(expected)
    relevance = _compute_keyword_relevance(query, text)
    assert relevance == pytest.approx(expected)


@pytest.mark.unit
def test_phrase_equivalence_current_account_only():
    """The current-account equivalence should still apply when only that phrase is present."""
    query = "current account"
    text = "current deposits"
    score = _phrase_equivalent_match(query, text)
    assert score == pytest.approx(3.5 + math.log(2))
    assert _compute_keyword_relevance(query, text) == pytest.approx(score)


@pytest.mark.unit
def test_phrase_equivalence_saving_account_only():
    """The saving-account equivalence should still apply when only that phrase is present."""
    query = "saving account"
    text = "savings deposits"
    score = _phrase_equivalent_match(query, text)
    assert score == pytest.approx(3.5 + math.log(2))
    assert _compute_keyword_relevance(query, text) == pytest.approx(score)


@pytest.mark.unit
def test_phrase_equivalence_ignores_unrelated_phrases():
    """Generic bank/account context should not trigger account/deposit phrase equivalence."""
    query = "current account"
    for text in [
        "account balance",
        "accounts with other banks",
        "term deposits",
        "cash and balances with treasury banks",
    ]:
        assert _phrase_equivalent_match(query, text) == 0.0


@pytest.mark.unit
def test_lexical_relevance_matches_current_deposits_equivalent_phrase():
    """Current-account queries should also match current-deposit wording without equating every account/deposit token."""
    query = "current account"
    text = "Current deposits"
    score = _compute_keyword_relevance(query, text)
    assert score > 0
    assert score > _compute_keyword_relevance("bank 2025", text)


@pytest.mark.unit
def test_lexical_relevance_matches_savings_deposits_equivalent_phrase():
    """Saving-account queries should also match savings-deposit wording without hard-coding a single report."""
    query = "saving account"
    text = "Savings deposits"
    score = _compute_keyword_relevance(query, text)
    assert score > 0
    assert score > _compute_keyword_relevance("bank 2025", text)


@pytest.mark.unit
def test_lexical_relevance_preserves_direct_current_account_match():
    """A direct current-account wording still scores strongly as before."""
    query = "current account"
    text = "current account"
    score = _compute_keyword_relevance(query, text)
    assert score > 5


@pytest.mark.unit
def test_lexical_relevance_preserves_direct_saving_account_match():
    """A direct saving-account wording should remain strong."""
    query = "saving account"
    text = "saving account"
    score = _compute_keyword_relevance(query, text)
    assert score > 5


@pytest.mark.unit
def test_lexical_relevance_preserves_saving_account_plural_morphology():
    """Simple singular/plural handling should still work for saving-account wording."""
    query = "saving account"
    text = "savings account"
    score = _compute_keyword_relevance(query, text)
    assert score > 0


@pytest.mark.unit
def test_lexical_relevance_does_not_double_count_repeated_generic_account_term():
    """The same generic account term should not count twice as a high-value keyword when it appears twice in the query."""
    query = "current account and saving account"
    text = "current account and saving account"
    direct_score = _compute_keyword_relevance(query, text)
    generic_only_score = _compute_keyword_relevance("account", text)
    assert direct_score > generic_only_score
    assert generic_only_score < 4


@pytest.mark.unit
def test_lexical_relevance_caps_repeated_generic_account_occurrences():
    """Repeated generic account occurrences should have diminishing returns instead of dominating the score."""
    query = "current account"
    text = "account account account account account account account account"
    score = _compute_keyword_relevance(query, text)
    assert score < 5


@pytest.mark.unit
def test_lexical_relevance_does_not_boost_account_balance_as_deposit_equivalent():
    """The account/deposit equivalence must be phrase-aware, not a global token alias."""
    query = "current account"
    direct_equivalent = _compute_keyword_relevance(query, "current deposits")
    balance_match = _compute_keyword_relevance(query, "account balance")
    assert direct_equivalent > balance_match
    assert balance_match < 3


@pytest.mark.unit
def test_lexical_relevance_does_not_boost_other_bank_accounts_as_deposit_equivalent():
    """A generic accounts-with-other-banks phrase should not get current-account deposit equivalence."""
    query = "current account"
    direct_equivalent = _compute_keyword_relevance(query, "current deposits")
    other_bank_match = _compute_keyword_relevance(query, "accounts with other banks")
    assert direct_equivalent > other_bank_match
    assert other_bank_match < 3


@pytest.mark.unit
def test_lexical_relevance_does_not_treat_term_deposits_as_current_account():
    """Term deposits should not be conflated with current-account phrases."""
    query = "current account"
    term_deposit_match = _compute_keyword_relevance(query, "term deposits")
    current_deposit_match = _compute_keyword_relevance(query, "current deposits")
    assert term_deposit_match < current_deposit_match
    assert term_deposit_match < 3


@pytest.mark.unit
def test_lexical_relevance_does_not_boost_treasury_bank_summary_for_current_account_phrase():
    """Treasury or bank-balance summary text should not receive the current-account phrase boost."""
    query = "current account"
    treasury_match = _compute_keyword_relevance(query, "cash and balances with treasury banks")
    current_deposit_match = _compute_keyword_relevance(query, "current deposits")
    assert treasury_match < current_deposit_match
    assert treasury_match < 3


@pytest.mark.unit
def test_lexical_relevance_avoids_bank_2025_noise():
    """Generic terms alone should not produce a large lexical score."""
    query = "current account"
    text = "bank 2025 general reporting summary"
    score = _compute_keyword_relevance(query, text)
    assert score < 2


# --- AGENT GRAPH & RUN_AGENT TESTS ---


@pytest.mark.unit
def test_default_llm_completion_fixed_mode_uses_primary_model_only(monkeypatch):
    """In fixed mode, automatic fallback candidates must not be attempted."""
    from rag_platform.agent.graph import default_llm_completion
    from rag_platform.config import get_settings

    settings = get_settings()
    original_mode = settings.LLM_MODEL_MODE
    settings.LLM_MODEL_MODE = "fixed"
    monkeypatch.setattr(settings, "LLM_MODEL", "fixed-model", raising=False)
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", raising=False)

    calls = []

    class FakeResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers, json, timeout):
        calls.append(json["model"])
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    result = default_llm_completion([{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert calls == ["fixed-model"]
    settings.LLM_MODEL_MODE = original_mode


@pytest.mark.unit
def test_default_llm_completion_auto_mode_tries_candidates_in_order(monkeypatch):
    """Auto mode should try the configured model and then the known OpenRouter fallback sequence."""
    from rag_platform.agent.graph import default_llm_completion
    from rag_platform.config import get_settings

    settings = get_settings()
    original_mode = settings.LLM_MODEL_MODE
    settings.LLM_MODEL_MODE = "auto"
    monkeypatch.setattr(settings, "LLM_MODEL", "primary-model", raising=False)
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", raising=False)

    attempts = []

    class FailResponse:
        status_code = 500
        def json(self):
            return {"error": "bad"}

    class SuccessResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "success from fallback"}}]}

    def fake_post(url, headers, json, timeout):
        attempts.append(json["model"])
        if json["model"] == "primary-model":
            return FailResponse()
        return SuccessResponse()

    monkeypatch.setattr("requests.post", fake_post)

    result = default_llm_completion([{"role": "user", "content": "hi"}])

    assert result == "success from fallback"
    assert attempts[0] == "primary-model"
    assert attempts[1] == "nvidia/nemotron-3.5-lightning:free"
    settings.LLM_MODEL_MODE = original_mode


@pytest.mark.unit
def test_run_agent_pure_retrieval_query(mock_vectorstore, mock_embeddings_fn):
    """Verify agent routes document/report queries through the retriever node."""
    def mock_llm(messages):
        # Router classification response
        for m in messages:
            if "Routing classification" in m.get("content", "") or "Classify the user query" in m.get("content", ""):
                return '{"decision": "retrieve", "tool_name": null, "tool_input": null}'
        return "The company's Q3 revenue was $94.9 Billion."

    result = run_agent(
        query="What is the financial revenue mentioned in the annual report?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "retrieve"
    assert len(result.reasoning_path) >= 3
    assert "report_2025.pdf" in str(result.sources)
    assert "94.9 Billion" in result.answer


@pytest.mark.unit
def test_run_agent_tool_only_query(mock_vectorstore, mock_embeddings_fn):
    """Verify agent routes calculation queries through the tool_caller node."""
    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "tool", "tool_name": "calculator", "tool_input": "45 * 20"}'
        return "The calculated product is 900."

    result = run_agent(
        query="Calculate 45 * 20",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "tool"
    assert "calculator" in result.tool_outputs
    assert result.tool_outputs["calculator"] == "900"
    assert "900" in result.answer


@pytest.mark.unit
def test_run_agent_hybrid_query(mock_vectorstore, mock_embeddings_fn):
    """Verify agent executes both retrieval and tool computation for hybrid requests."""
    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "hybrid", "tool_name": "calculator", "tool_input": "94.9 * 0.462"}'
        return "Based on the report revenue of $94.9B and margin of 46.2%, the gross profit is approx $43.84B."

    result = run_agent(
        query="From the report, what is the gross profit if revenue is 94.9 and margin is 46.2%?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "hybrid"
    assert "retriever" in result.tool_outputs
    assert "calculator" in result.tool_outputs
    assert len(result.reasoning_path) >= 4


@pytest.mark.unit
def test_run_agent_direct_query(mock_vectorstore, mock_embeddings_fn):
    """Verify agent handles general questions directly without unnecessary retrieval or tool calls."""
    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "direct", "tool_name": null, "tool_input": null}'
        return "Python is an interpreted, high-level programming language."

    result = run_agent(
        query="Explain what Python is in one sentence.",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "direct"
    assert "Python is an interpreted" in result.answer


@pytest.mark.unit
def test_run_agent_retries_when_llm_emits_tool_call_markup(mock_vectorstore, mock_embeddings_fn):
    """The responder must reject leaked tool-call markup and retry once before falling back."""
    responses = iter([
        '{"decision": "tool", "tool_name": "calculator", "tool_input": "12 * 3"}',
        '<tool_call><arg_key>query</arg_key><arg_value>current weather in Karachi</arg_value></tool_call>',
        'The calculator result is 36.',
    ])

    def mock_llm(messages):
        content = next(responses)
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "tool", "tool_name": "calculator", "tool_input": "12 * 3"}'
        return content

    result = run_agent(
        query="What is 12 * 3?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert result.routing_decision == "tool"
    assert "<tool_call" not in result.answer.lower()
    assert "36" in result.answer


@pytest.mark.unit
def test_run_agent_empty_query_raises_agent_error():
    """Verify run_agent raises AgentError on empty input."""
    with pytest.raises(AgentError, match="Query cannot be empty"):
        run_agent("")


@pytest.mark.unit
def test_run_agent_does_not_return_placeholder_response(mock_vectorstore, mock_embeddings_fn):
    """Regression test ensuring agent synthesizes real grounded answers instead of placeholder."""
    def mock_llm(messages):
        for m in messages:
            content = m.get("content", "")
            if "Classify the user query" in content:
                return '{"decision": "retrieve", "tool_name": null, "tool_input": null}'
            if "=== KNOWLEDGE BASE CONTEXT ===" in content:
                return "The Q3 Revenue was $94.9 Billion with a gross margin of 46.2%."
        return "Generic synthesized answer"

    result = run_agent(
        query="What is the proposed research problem?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert isinstance(result, AgentResult)
    assert not result.answer.startswith("Response to:")
    assert "94.9 Billion" in result.answer
    assert result.routing_decision == "retrieve"


@pytest.mark.unit
def test_retrieval_tool_hybrid_scoring_outranks_toc_chunks(mock_vectorstore, mock_embeddings_fn):
    """Verify hybrid scoring penalizes Table of Contents dot chunks and prioritizes body text."""
    toc_chunk = SearchResult(
        chunk_id="chunk_toc",
        doc_id="thesis.pdf",
        content="2. Literature Review.........................................................................3 3. Proposed Research Work..................................................3",
        similarity=0.50,
        metadata={"page_number": 1, "content_type": "text"},
        filename="thesis.pdf",
    )
    body_chunk = SearchResult(
        chunk_id="chunk_body",
        doc_id="thesis.pdf",
        content="The specific research problem is: given two consecutive document pages containing a table that continues across the page boundary, can a compact structured representation of page N improve the reconstruction of the table segment on page N+1 compared with processing page N+1 independently?",
        similarity=0.45,
        metadata={"page_number": 3, "content_type": "text"},
        filename="thesis.pdf",
    )

    mock_vectorstore.similarity_search.return_value = [toc_chunk, body_chunk]
    res = retrieval_tool(
        query="What is the proposed research problem?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        k=2,
    )

    assert res.success is True
    chunks = res.data["chunks"]
    assert len(chunks) == 2
    # Body chunk should outrank TOC chunk due to keyword relevance and TOC penalty
    assert chunks[0]["chunk_id"] == "chunk_body"
    assert chunks[1]["chunk_id"] == "chunk_toc"


@pytest.mark.unit
def test_retrieval_tool_prefers_problem_statement_over_methodology_heading(
    mock_vectorstore, mock_embeddings_fn
):
    """Regression: 'proposed research problem' must prefer problem-statement body over methodology."""
    methodology = SearchResult(
        chunk_id="chunk_method",
        doc_id="thesis.pdf",
        content=(
            "inference cost? 3. Research plan a) Proposed Research Methodology "
            "Research design. The study will use a controlled comparative experiment. "
            "Every test instance will consist of two consecutive pages containing a table."
        ),
        similarity=0.48,
        metadata={"page_number": 4, "content_type": "text"},
        filename="thesis.pdf",
    )
    problem_body = SearchResult(
        chunk_id="chunk_problem",
        doc_id="thesis.pdf",
        content=(
            "a) Problem statement When a financial table continues from page N to page N+1, "
            "processing page N+1 independently can produce an internally plausible table. "
            "The specific research problem is: given two consecutive document pages containing "
            "a table that continues across the page boundary, can a compact structured "
            "representation of page N improve the reconstruction of the table segment on page N+1 "
            "compared with processing page N+1 independently?"
        ),
        similarity=0.40,
        metadata={"page_number": 3, "content_type": "text"},
        filename="thesis.pdf",
    )
    mock_vectorstore.similarity_search.return_value = [methodology, problem_body]

    res = retrieval_tool(
        query="What is the proposed research problem?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        k=2,
    )
    assert res.success is True
    assert res.data["chunks"][0]["chunk_id"] == "chunk_problem"
    assert "specific research problem" in res.data["chunks"][0]["content"].lower()


@pytest.mark.unit
def test_run_agent_uses_default_retrieval_k(mock_vectorstore, mock_embeddings_fn, monkeypatch):
    """Ensure production run_agent path requests DEFAULT_RETRIEVAL_K (not legacy k=4)."""
    from rag_platform.agent import graph as graph_mod
    from rag_platform.agent.tools import DEFAULT_RETRIEVAL_K

    captured = {}

    def fake_retrieval_tool(*, query, vectorstore, embeddings_fn, filters, k):
        captured["k"] = k
        return ToolResult(
            tool_name="retriever",
            output="--- Source [1] ---\nThe specific research problem is page-context conditioning.",
            data={
                "count": 1,
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "doc_id": "proposal.pdf",
                        "filename": "proposal.pdf",
                        "content": "The specific research problem is page-context conditioning.",
                        "metadata": {"page_number": 3, "content_type": "text"},
                        "similarity": 0.9,
                    }
                ],
            },
            success=True,
        )

    monkeypatch.setattr(graph_mod, "retrieval_tool", fake_retrieval_tool)

    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "retrieve", "tool_name": null, "tool_input": null}'
        return "Grounded answer about the research problem."

    result = run_agent(
        query="What is the proposed research problem?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )
    assert captured.get("k") == DEFAULT_RETRIEVAL_K
    assert DEFAULT_RETRIEVAL_K >= 6
    assert result.routing_decision == "retrieve"


@pytest.mark.unit
def test_run_agent_retrieval_grounded_research_problem_and_datasets(mock_vectorstore, mock_embeddings_fn):
    """Regression test verifying agent receives grounded document content for research problem & datasets."""
    body_chunk = SearchResult(
        chunk_id="chunk_body_p3",
        doc_id="proposal.pdf",
        content="The specific research problem is: given two consecutive document pages containing a table that continues across page boundary. Datasets used include FinTabNet, PubTables-1M, and PubTables-v2.",
        similarity=0.85,
        metadata={"page_number": 3, "content_type": "text"},
        filename="proposal.pdf",
    )
    mock_vectorstore.similarity_search.return_value = [body_chunk]

    def mock_llm(messages):
        for m in messages:
            content = m.get("content", "")
            if "Classify the user query" in content:
                return '{"decision": "retrieve", "tool_name": null, "tool_input": null}'
            if "=== KNOWLEDGE BASE CONTEXT ===" in content:
                assert "PubTables-v2" in content
                assert "consecutive document pages" in content
                return "The proposed research problem investigates two-page financial table reconstruction using PubTables-v2 and FinTabNet."
        return "Generic response"

    result = run_agent(
        query="What is the proposed research problem?",
        vectorstore=mock_vectorstore,
        embeddings_fn=mock_embeddings_fn,
        llm_call_fn=mock_llm,
    )

    assert result.routing_decision == "retrieve"
    assert "PubTables-v2" in result.answer
    assert not result.answer.startswith("Response to:")
    assert "table of contents" not in result.answer.lower()
    assert "unavailable" not in result.answer.lower()

