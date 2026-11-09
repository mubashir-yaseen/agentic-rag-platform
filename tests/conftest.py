import pytest
"""Shared test setup: the unit tests must pass on a clean machine (and in CI) with no real secrets."""

import os

# Dummy values so settings that "require" credentials exist. Real services are always mocked in unit tests.
# setdefault never overrides a value that is already set in the environment.
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("SUPABASE_URL", "https://test-project.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")


@pytest.fixture(autouse=True)
def _no_live_web_search(monkeypatch):
    """Unit tests must never hit DuckDuckGo: CI has no reliable network for it."""
    try:
        import duckduckgo_search
    except ImportError:
        return

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=5, **kwargs):
            return [{"title": "AI news results", "href": "https://example.com/ai", "body": "AI results for testing."}]

    monkeypatch.setattr(duckduckgo_search, "DDGS", _FakeDDGS)
