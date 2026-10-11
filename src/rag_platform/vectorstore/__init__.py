"""Vectorstore subpackage providing persistent multi-document storage in Supabase."""

from rag_platform.vectorstore.embeddings import get_embedding_model
from rag_platform.vectorstore.supabase_store import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
    SupabaseVectorStore,
    with_retry,
)

__all__ = [
    "DocumentRecord",
    "SearchFilter",
    "SearchResult",
    "SupabaseVectorStore",
    "get_embedding_model",
    "with_retry",
]
