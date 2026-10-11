"""Evaluation subpackage providing RAGAS metrics, testsets, and evaluation runner."""

from rag_platform.eval.run_eval import (
    EvalReport,
    EvalSample,
    SampleEvalScore,
    compute_answer_relevance,
    compute_context_precision,
    compute_faithfulness,
    run_evaluation,
)

__all__ = [
    "EvalReport",
    "EvalSample",
    "SampleEvalScore",
    "compute_answer_relevance",
    "compute_context_precision",
    "compute_faithfulness",
    "run_evaluation",
]
