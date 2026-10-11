"""Unit smoke tests for RAGSystem text processing and imports."""

from unittest.mock import MagicMock, patch
import pytest
from rag_platform.exceptions import IngestionError
from rag_platform.rag_system import RAGSystem


@pytest.mark.unit
def test_rag_system_chunking():
    """Verify chunk_text splits content accurately into Document objects."""
    with patch.object(RAGSystem, "__init__", lambda self, *args, **kwargs: None):
        rag = RAGSystem()
        sample_text = (
            "Paragraph 1 contains introductory concepts about RAG platforms.\n\n"
            "Paragraph 2 discusses embedding models and retrieval techniques.\n\n"
            "Paragraph 3 covers multi-document indexing and agent architectures."
        )
        chunks = rag.chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 3
        assert all(hasattr(c, "page_content") for c in chunks)
        assert all("chunk_id" in c.metadata for c in chunks)


@pytest.mark.unit
def test_extract_text_nonexistent_file_raises_ingestion_error():
    """Verify extract_text_from_pdf raises IngestionError on missing files."""
    with patch.object(RAGSystem, "__init__", lambda self, *args, **kwargs: None):
        rag = RAGSystem()
        with pytest.raises(IngestionError, match="File not found"):
            rag.extract_text_from_pdf("non_existent_file.pdf")


@pytest.mark.unit
def test_rag_system_init_from_centralized_settings():
    """Verify RAGSystem initializes from centralized settings without st.secrets."""
    mock_settings = MagicMock()
    mock_settings.EMBEDDING_MODEL = "test-model"
    mock_settings.LLM_MODEL = "test-llm"
    mock_settings.RAG_TEMPERATURE = 0.2
    mock_settings.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    mock_settings.effective_api_key = "test-api-key"
    mock_settings.SUPABASE_URL = "https://test.supabase.co"
    mock_settings.effective_supabase_key = "test-supabase-key"

    with patch("rag_platform.rag_system.get_settings", return_value=mock_settings), \
         patch("rag_platform.rag_system.create_client") as mock_create_client, \
         patch("rag_platform.vectorstore.embeddings.get_embedding_model") as mock_get_emb:
        rag = RAGSystem()
        assert rag.api_key == "test-api-key"
        assert rag.supabase_url == "https://test.supabase.co"
        assert rag.supabase_key == "test-supabase-key"
        assert rag.supabase_ok is True
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-supabase-key")
