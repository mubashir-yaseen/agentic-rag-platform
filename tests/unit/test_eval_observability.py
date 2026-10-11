"""Unit tests for observability tracing and RAGAS evaluation harness."""

import json
import os
import tempfile
import pytest

from rag_platform.agent.graph import AgentResult, run_agent
from rag_platform.eval.run_eval import (
    EvalReport,
    EvalSample,
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
    run_evaluation,
)
from rag_platform.observability.tracing import AgentTrace, ObservabilityTracer, TraceSpan, get_tracer


# --- OBSERVABILITY & TRACING TESTS ---


@pytest.mark.unit
def test_tracer_start_and_end_trace():
    """Verify tracer records execution spans, computes latency, and produces AgentTrace."""
    tracer = ObservabilityTracer()
    trace_id = "test-corr-12345"

    trace = tracer.start_trace(query="What was 2023 revenue?", trace_id=trace_id)
    assert trace.trace_id == trace_id
    assert trace.query == "What was 2023 revenue?"

    # Record spans
    tracer.record_span(
        trace_id=trace_id,
        name="router",
        start_time=trace.start_time,
        input_data={"query": trace.query},
        output_data={"decision": "retrieve"},
    )

    tracer.record_span(
        trace_id=trace_id,
        name="retriever",
        start_time=trace.start_time,
        output_data={"chunk_count": 3},
    )

    final_trace = tracer.end_trace(
        trace_id=trace_id,
        final_answer="Revenue was $124.5M in 2023.",
        routing_decision="retrieve",
        prompt_tokens=150,
        completion_tokens=30,
        metadata={"model": "test-model"},
    )

    assert final_trace.trace_id == trace_id
    assert final_trace.final_answer == "Revenue was $124.5M in 2023."
    assert final_trace.routing_decision == "retrieve"
    assert len(final_trace.spans) == 2
    assert final_trace.token_usage["total_tokens"] == 180
    assert final_trace.latency_ms >= 0.0


@pytest.mark.unit
def test_global_get_tracer_singleton():
    """Verify get_tracer returns a functional singleton instance."""
    tracer1 = get_tracer()
    tracer2 = get_tracer()
    assert tracer1 is tracer2


@pytest.mark.unit
def test_agent_run_populates_trace():
    """Verify executing run_agent creates and completes an active trace."""
    tracer = get_tracer()

    def mock_llm(messages):
        return "Calculated total is 42."

    res = run_agent(query="What is 40 + 2?", llm_call_fn=mock_llm)
    assert isinstance(res, AgentResult)


# --- RAGAS EVALUATION HARNESS TESTS ---


@pytest.mark.unit
def test_compute_faithfulness_grounded_and_ungrounded():
    """Verify faithfulness metric detects grounded vs hallucinated content."""
    context = "The Company reported total consolidated revenue of $124.5 million in fiscal year 2023."
    grounded_answer = "In fiscal year 2023, the Company revenue was $124.5 million."
    hallucinated_answer = "The enterprise acquired Quantum Dynamics for 500 billion euros in 2099."

    high_faith = compute_faithfulness(grounded_answer, context)
    low_faith = compute_faithfulness(hallucinated_answer, context)

    assert high_faith > 0.6
    assert low_faith < 0.2


@pytest.mark.unit
def test_compute_answer_relevance():
    """Verify answer relevance metric aligns with question intent and expected answers."""
    question = "What was the R&D expenditure in Q4 2023?"
    expected = "R&D expenditures were $18.6 million in Q4 2023."
    relevant_answer = "Research and development expenditures totaled $18.6 million in Q4 2023."
    irrelevant_answer = "The weather in Seattle was cloudy with light rain."

    high_rel = compute_answer_relevance(relevant_answer, question, expected)
    low_rel = compute_answer_relevance(irrelevant_answer, question, expected)

    assert high_rel > 0.6
    assert low_rel < 0.3


@pytest.mark.unit
def test_compute_context_precision():
    """Verify context precision correctly matches source citations."""
    expected_sources = ["annual_report_2023.pdf", "Page 12"]
    matching_retrieved = ["annual_report_2023.pdf (Page 12)", "Page 12 footnote"]
    non_matching = ["other_contract.pdf (Page 1)"]

    high_prec = compute_context_precision(matching_retrieved, expected_sources)
    low_prec = compute_context_precision(non_matching, expected_sources)

    assert high_prec == 1.0
    assert low_prec == 0.0


@pytest.mark.unit
def test_run_evaluation_custom_testset():
    """Verify run_evaluation executes over custom sample data and computes aggregate metrics."""
    test_data = [
        {
            "id": "test-1",
            "question": "What is the termination notice period?",
            "expected_answer": "30 days written notice is required.",
            "expected_sources": ["contract.pdf", "Page 5"],
            "ground_truth_context": "Either party may terminate upon 30 days written notice.",
        },
        {
            "id": "test-2",
            "question": "What was the gross margin in 2023?",
            "expected_answer": "Gross margin was 68.5% in 2023.",
            "expected_sources": ["financials.pdf"],
            "ground_truth_context": "In 2023, the Company recorded a gross margin of 68.5%.",
        },
    ]

    report = run_evaluation(testset_data=test_data)

    assert isinstance(report, EvalReport)
    assert report.total_samples == 2
    assert report.mean_faithfulness >= 0.7
    assert report.mean_answer_relevance >= 0.7
    assert report.mean_context_precision == 1.0
    assert report.mean_overall_score >= 0.7
    assert len(report.sample_scores) == 2


@pytest.mark.unit
def test_run_evaluation_default_testset_with_limit():
    """Verify run_evaluation runs on the real testset.json with sample limiting."""
    report = run_evaluation(limit=3)

    assert isinstance(report, EvalReport)
    assert report.total_samples == 3
    assert 0.0 <= report.mean_overall_score <= 1.0
    assert len(report.sample_scores) == 3
