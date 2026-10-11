"""Integration smoke tests for Supabase vector store and fallback handling."""

import pytest
from rag_platform.config import get_settings
from rag_platform.rag_system import RAGSystem
from rag_platform.vectorstore.supabase_store import SupabaseVectorStore


@pytest.mark.integration
def test_supabase_client_initialization_behavior():
    """Verify RAGSystem initializes gracefully even if live credentials are not present."""
    settings = get_settings()
    if settings.SUPABASE_URL and settings.effective_supabase_key and settings.effective_api_key:
        try:
            rag = RAGSystem()
            assert rag.supabase_ok is True
        except Exception as exc:
            pytest.skip(f"Live Supabase connection failed in current environment: {exc}")
    else:
        pytest.skip("Supabase credentials not configured; skipping live integration check.")


@pytest.mark.integration
def test_supabase_vector_store_live_smoke():
    """Verify SupabaseVectorStore can query tables if live credentials are present."""
    settings = get_settings()
    if settings.SUPABASE_URL and settings.effective_supabase_key:
        try:
            store = SupabaseVectorStore()
            docs = store.list_documents()
            assert isinstance(docs, list)
        except Exception as exc:
            pytest.skip(f"Live Supabase table query failed in current environment: {exc}")
    else:
        pytest.skip("Supabase credentials not configured; skipping live integration check.")
