"""Tests for eval/run_live_eval.py (the scoring logic and the runner, with a fake API)."""

import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "run_live_eval.py")
_spec = importlib.util.spec_from_file_location("run_live_eval", _PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


@pytest.mark.unit
def test_numbers_match_with_or_without_thousand_separators():
    q = {"id": "a", "expected": ["12,345.6"], "source_contains": "UBL", "pages": [7]}
    ok = ev.score_one(q, "Profit after tax was 12345.6 million.", ["UBL_AR_2025.pdf (Page 7)"])
    assert ok["correct"] and ok["doc_hit"] and ok["page_hit"]


@pytest.mark.unit
def test_all_expected_facts_are_required():
    q = {"id": "b", "expected": ["100", "200"]}
    assert not ev.score_one(q, "Deposits were 100.", [])["correct"]
    assert ev.score_one(q, "Deposits were 100 and assets 200.", [])["correct"]


@pytest.mark.unit
def test_expected_any_accepts_alternative_spellings():
    q = {"id": "c", "expected_any": [["Yousuf Adil", "Yousuf Adil Chartered"]]}
    assert ev.score_one(q, "Signed by Yousuf Adil.", [])["correct"]
    assert not ev.score_one(q, "Signed by someone else.", [])["correct"]


@pytest.mark.unit
def test_wrong_page_or_report_is_not_a_citation_hit():
    q = {"id": "d", "expected": ["1"], "source_contains": "UBL", "pages": [7]}
    wrong_page = ev.score_one(q, "1", ["UBL_AR_2025.pdf (Page 9)"])
    assert wrong_page["doc_hit"] and not wrong_page["page_hit"]
    wrong_doc = ev.score_one(q, "1", ["HBL_AR_2025.pdf (Page 7)"])
    assert not wrong_doc["doc_hit"]


@pytest.mark.unit
def test_unanswerable_question_is_correct_only_when_the_app_abstains():
    q = {"id": "e", "should_abstain": True}
    assert ev.score_one(q, "The provided documents do not contain this information.", [])["correct"]
    assert not ev.score_one(q, "Revenue in Japan was 55 million.", [])["correct"]


@pytest.mark.unit
def test_infrastructure_problems_are_errors_not_wrong_answers():
    q = {"id": "f", "expected": ["1"]}
    row = ev.score_one(q, "I can't search the documents right now because the free daily limit was reached.", [])
    assert "error" in row and "correct" not in row


@pytest.mark.unit
def test_runner_summarises_accuracy_citations_abstention_and_latency():
    questions = [
        {"id": "1", "question": "q1", "expected": ["42"], "source_contains": "UBL", "pages": [3]},
        {"id": "2", "question": "q2", "expected": ["7"], "source_contains": "UBL"},
        {"id": "3", "question": "q3", "should_abstain": True},
        {"id": "4", "question": "q4", "expected": ["9"]},
        {"id": "REPLACE", "question": "skip me", "expected": ["REPLACE_me"]},
    ]
    replies = {
        "q1": {"answer": "It is 42.", "sources": ["UBL_AR_2025.pdf (Page 3)"]},
        "q2": {"answer": "I think 8.", "sources": ["UBL_AR_2025.pdf (Page 1)"]},
        "q3": {"answer": "That is not available in the documents.", "sources": []},
    }

    def fake_api(payload):
        if payload["query"] == "q4":
            raise RuntimeError("server timeout")
        return replies[payload["query"]]

    clock = iter(range(0, 1000, 2))  # every call takes 2 virtual seconds
    real = [q for q in questions if not ev.is_template(q)]
    out = ev.evaluate(real, fake_api, clock=lambda: next(clock), log=lambda *_: None)

    assert out["questions"] == 4 and out["errors"] == 1
    assert out["answer_accuracy_pct"] == 50.0 and out["answerable_scored"] == 2
    assert out["report_cited_pct"] == 100.0 and out["page_cited_pct"] == 100.0
    assert out["abstention_pct"] == 100.0
    assert out["latency_p50_s"] == 2.0
    assert "Answer accuracy" in ev.format_report(out)
    assert ev.is_template(questions[-1])
