"""Measure the DEPLOYED app on questions you wrote from your real annual reports.

What it measures (honest, simple, reproducible):
  * answer accuracy   - the answer contains every expected fact (numbers are compared without commas / spaces)
  * citation hit rate - the sources include the expected report (and, if given, the expected page)
  * abstention rate   - for questions the report cannot answer, the app says so instead of inventing something
  * latency           - median (p50) and 95th percentile (p95) seconds per question

Questions that fail because of infrastructure (daily quota used up, server asleep, timeout) are reported as
"errors" and are NOT counted as wrong answers.

Usage:
    python eval/run_live_eval.py --base-url https://<your-api>.onrender.com --questions eval/questions.json
    python eval/run_live_eval.py --base-url ... --questions eval/questions.json --out eval/results.json

Question file format: see eval/questions.example.json and eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from typing import Any, Callable, Optional

ABSTAIN_PHRASES = (
    "not available", "does not contain", "do not contain", "not contain", "cannot find", "can't find", "could not find",
    "no information", "not mentioned", "not provided", "not specified", "insufficient", "not supported",
    "does not provide", "do not provide", "no mention", "unable to find", "not found in",
)
INFRA_PHRASES = (
    "daily limit", "i could not search your documents", "didn't respond", "did not respond", "overloaded",
    "kept failing", "http 429", "timed out", "temporarily unavailable", "heavy traffic",
)


def _norm(text: str) -> str:
    """Lower-case and drop thousands separators / spaces so '1,234.5' and '1234.5' compare equal."""
    text = (text or "").lower()
    text = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text)


def is_template(q: dict[str, Any]) -> bool:
    blob = json.dumps(q)
    return "REPLACE" in blob


def score_one(q: dict[str, Any], answer: str, sources: list[str]) -> dict[str, Any]:
    """Score a single answer. Pure function, easy to unit-test."""
    text, norm_answer = answer or "", _norm(answer or "")
    result: dict[str, Any] = {"id": q.get("id"), "kind": "abstain" if q.get("should_abstain") else "answer"}

    if any(p in text.lower() for p in INFRA_PHRASES):
        result["error"] = "service problem (quota, timeout or model overloaded)"
        return result

    if q.get("should_abstain"):
        result["correct"] = any(p in text.lower() for p in ABSTAIN_PHRASES)
    else:
        expected = [_norm(str(x)) for x in q.get("expected", [])]
        result["correct"] = bool(expected) and all(e in norm_answer for e in expected)
        anyof = [[_norm(str(x)) for x in grp] for grp in q.get("expected_any", [])]
        if anyof:
            result["correct"] = bool(result["correct"] or not expected) and all(
                any(opt in norm_answer for opt in grp) for grp in anyof
            )

    # Citation check: right report, and (optionally) right page.
    want_doc = (q.get("source_contains") or "").lower()
    want_pages = [int(p) for p in q.get("pages", [])]
    result["cited"] = bool(sources)
    if want_doc and not q.get("should_abstain"):
        doc_hits = [s for s in sources if want_doc in s.lower()]
        result["doc_hit"] = bool(doc_hits)
        if want_pages:
            pages = {int(m) for s in doc_hits for m in re.findall(r"page\s*(\d+)", s.lower())}
            result["page_hit"] = bool(pages & set(want_pages))
    return result


def evaluate(
    questions: list[dict[str, Any]],
    call_api: Callable[[dict[str, Any]], dict[str, Any]],
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        payload = {"query": q["question"], "mode": q.get("mode", "Research")}
        if q.get("company_id"):
            payload["company_id"] = q["company_id"]
        started = clock()
        try:
            resp = call_api(payload)
            answer, sources = resp.get("answer", ""), [str(s) for s in resp.get("sources", [])]
            row = score_one(q, answer, sources)
            row.update(answer=answer[:600], sources=sources[:6])
        except Exception as exc:  # network error, 5xx, timeout
            row = {"id": q.get("id"), "kind": "abstain" if q.get("should_abstain") else "answer", "error": str(exc)[:200]}
        row["question"] = q["question"]
        row["seconds"] = round(clock() - started, 2)
        rows.append(row)
        status = "ERROR" if row.get("error") else ("ok" if row.get("correct") else "WRONG")
        log(f"[{i}/{len(questions)}] {status:5} {row['seconds']:6.1f}s  {q['question'][:70]}")
        if delay and i < len(questions):
            sleep(delay)
    return summarize(rows)


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, 1) if den else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if not r.get("error")]
    answerable = [r for r in scored if r["kind"] == "answer"]
    abstain = [r for r in scored if r["kind"] == "abstain"]
    with_doc = [r for r in answerable if "doc_hit" in r]
    with_page = [r for r in answerable if "page_hit" in r]
    times = sorted(r["seconds"] for r in scored)
    p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))] if times else None
    return {
        "questions": len(rows),
        "errors": len(rows) - len(scored),
        "answer_accuracy_pct": _pct(sum(bool(r.get("correct")) for r in answerable), len(answerable)),
        "answerable_scored": len(answerable),
        "report_cited_pct": _pct(sum(r["doc_hit"] for r in with_doc), len(with_doc)),
        "page_cited_pct": _pct(sum(r["page_hit"] for r in with_page), len(with_page)),
        "abstention_pct": _pct(sum(bool(r.get("correct")) for r in abstain), len(abstain)),
        "abstain_scored": len(abstain),
        "latency_p50_s": round(statistics.median(times), 1) if times else None,
        "latency_p95_s": round(p95, 1) if p95 is not None else None,
        "rows": rows,
    }


def format_report(s: dict[str, Any]) -> str:
    def f(v, unit=""):
        return "n/a" if v is None else f"{v}{unit}"

    return "\n".join([
        "| Metric | Result |",
        "|---|---|",
        f"| Questions run | {s['questions']} ({s['errors']} infrastructure errors excluded from scores) |",
        f"| Answer accuracy (answerable questions) | {f(s['answer_accuracy_pct'], '%')} of {s['answerable_scored']} |",
        f"| Correct report cited | {f(s['report_cited_pct'], '%')} |",
        f"| Correct page cited | {f(s['page_cited_pct'], '%')} |",
        f"| Abstains on unanswerable questions | {f(s['abstention_pct'], '%')} of {s['abstain_scored']} |",
        f"| Latency p50 / p95 | {f(s['latency_p50_s'], 's')} / {f(s['latency_p95_s'], 's')} |",
    ])


def make_http_caller(base_url: str, timeout: float, retries: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
    import requests

    url = base_url.rstrip("/") + "/api/v1/chat"

    def call(payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception = RuntimeError("no attempt made")
        for attempt in range(1, retries + 2):
            try:
                resp = requests.post(url, json={**payload, "stream": False}, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()
                last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
            except requests.RequestException as exc:
                last = exc
            time.sleep(min(5 * attempt, 20))  # free hosting may still be waking up
        raise last

    return call


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the deployed RAG app on your own questions.")
    ap.add_argument("--base-url", required=True, help="e.g. https://agentic-rag-platform-xxxx.onrender.com")
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between questions (free LLMs are rate limited)")
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args(argv)

    with open(args.questions, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    questions = [q for q in (data["questions"] if isinstance(data, dict) else data) if not is_template(q)]
    if not questions:
        print("No real questions found. Replace the REPLACE_... placeholders in your questions file first.")
        return 2

    print(f"Running {len(questions)} questions against {args.base_url} (first call may take ~60s if the server is asleep)\n")
    summary = evaluate(questions, make_http_caller(args.base_url, args.timeout, args.retries), delay=args.delay)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("\n" + format_report(summary))
    print(f"\nFull results saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
