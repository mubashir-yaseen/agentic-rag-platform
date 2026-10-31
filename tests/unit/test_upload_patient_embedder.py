"""Tests for the free-tier-safe embedder used by upload_company_report.py (pacing, waiting, resume, daily budget)."""

import json

import pytest

from rag_platform.vectorstore.embeddings import EmbeddingServiceError
from upload_company_report import PatientEmbedder


class FakeClock:
    """Virtual time so the tests never really sleep."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class FakeInner:
    model = "fake-model"

    def __init__(self, clock, fail_first=0, message="quota used up (HTTP 429). Please retry in 43.7s."):
        self.clock, self.fail_first, self.message, self.calls = clock, fail_first, message, []

    def embed_documents(self, texts):
        self.clock.t += 0.5  # a request takes time
        self.calls.append((self.clock.t, len(texts)))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise EmbeddingServiceError(self.message)
        return [[float(len(t))] * 3 for t in texts]


def make(inner, clock, **kw):
    return PatientEmbedder(inner, sleep=clock.sleep, clock=clock.now, log=lambda *_: None, **kw)


@pytest.mark.unit
def test_pacing_never_exceeds_the_per_minute_limit():
    clock = FakeClock()
    inner = FakeInner(clock)
    vectors = make(inner, clock, per_minute=80).embed_documents([f"t{i}" for i in range(400)])
    assert len(vectors) == 400 and vectors[5] == [2.0] * 3
    worst = max(sum(n for tt, n in inner.calls if t - 60 < tt <= t) for t, _ in inner.calls)
    assert worst <= 80
    assert clock.t >= 240  # 400 texts at <=80/minute cannot finish faster


@pytest.mark.unit
def test_waits_the_time_google_asks_for_then_continues():
    clock = FakeClock()
    inner = FakeInner(clock, fail_first=2)
    vectors = make(inner, clock).embed_documents(["a", "bb"])
    assert vectors == [[1.0] * 3, [2.0] * 3]
    assert clock.t >= 90  # two waits of ~46.7 seconds ("retry in 43.7s" + margin)


@pytest.mark.unit
def test_gives_up_with_a_resume_message_instead_of_looping_forever():
    clock = FakeClock()
    inner = FakeInner(clock, fail_first=10**6)
    with pytest.raises(EmbeddingServiceError, match="Run the same command again"):
        make(inner, clock, max_waits=3).embed_documents(["a"])
    assert len(inner.calls) == 4


@pytest.mark.unit
def test_other_errors_are_not_swallowed():
    clock = FakeClock()
    inner = FakeInner(clock, fail_first=1, message="rejected the API key (HTTP 400)")
    with pytest.raises(EmbeddingServiceError, match="API key"):
        make(inner, clock).embed_documents(["a"])
    assert len(inner.calls) == 1


@pytest.mark.unit
def test_progress_is_saved_and_a_second_run_only_embeds_the_rest(tmp_path):
    clock = FakeClock()
    cache = str(tmp_path / "cache.jsonl")
    texts = [f"chunk number {i}" for i in range(50)]

    class Flaky(FakeInner):
        def embed_documents(self, texts):
            if self.calls:
                raise EmbeddingServiceError("HTTP 429 quota. Please retry in 1s.")
            return super().embed_documents(texts)

    with pytest.raises(EmbeddingServiceError):
        make(Flaky(clock), clock, max_waits=1, slice_size=20, cache_path=cache).embed_documents(texts)
    assert sum(1 for _ in open(cache)) == 20  # first slice survived the failure

    second = FakeInner(clock)
    out = make(second, clock, slice_size=20, cache_path=cache).embed_documents(texts)
    assert len(out) == 50 and sum(n for _, n in second.calls) == 30

    third = FakeInner(clock)
    make(third, clock, cache_path=cache).embed_documents(texts)
    assert third.calls == []  # fully cached: no requests at all

    with open(cache, "a") as fh:
        fh.write('{"k": "abc", "v": [1,')  # half-written last line (crash while saving)
    assert len(make(FakeInner(clock), clock, cache_path=cache).embed_documents(texts)) == 50


@pytest.mark.unit
def test_vectors_of_different_models_never_share_a_cache_entry():
    clock = FakeClock()
    a = make(type("A", (), {"model": "models/gemini-embedding-001"})(), clock)
    b = make(type("B", (), {"model": "@cf/baai/bge-base-en-v1.5"})(), clock)
    assert a._key("same text") != b._key("same text")


@pytest.mark.unit
def test_daily_budget_stops_cleanly_and_resets_the_next_day(tmp_path):
    clock = FakeClock()
    usage = str(tmp_path / "usage.json")
    day = ["2026-10-24"]

    def build(inner):
        return make(inner, clock, daily_budget=100, usage_path=usage, today=lambda: day[0], slice_size=20)

    first = FakeInner(clock)
    with pytest.raises(EmbeddingServiceError, match="after 00:00 UTC"):
        build(first).embed_documents([f"b{i}" for i in range(150)])
    assert sum(n for _, n in first.calls) == 100
    assert json.load(open(usage)) == {"date": "2026-10-24", "chunks": 100}

    same_day = FakeInner(clock)
    with pytest.raises(EmbeddingServiceError):
        build(same_day).embed_documents([f"b{i}" for i in range(150)])
    assert same_day.calls == []

    day[0] = "2026-10-25"
    next_day = FakeInner(clock)
    with pytest.raises(EmbeddingServiceError):
        build(next_day).embed_documents([f"c{i}" for i in range(150)])
    assert sum(n for _, n in next_day.calls) == 100  # a fresh budget
