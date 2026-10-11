"""Unit tests for SupabaseVectorStore with mocked Supabase client."""

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

from rag_platform.exceptions import RetrievalError
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk, IngestionResult
from rag_platform.vectorstore.supabase_store import (
    DocumentRecord,
    SearchFilter,
    SearchResult,
    SupabaseVectorStore,
    with_retry,
)

from upload_company_report import build_company_document_metadata, upload_company_annual_report


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client mimicking table queries and RPC calls."""
    client = MagicMock()
    return client


@pytest.mark.unit
def test_upsert_document(mock_supabase_client):
    """Verify upsert_document executes table upsert and returns DocumentRecord."""
    mock_table = MagicMock()
    mock_supabase_client.table.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(
        data=[{
            "doc_id": "doc_123",
            "filename": "annual_report.pdf",
            "uploaded_at": "2026-08-29T12:00:00Z",
            "doc_type": "pdf",
            "metadata": {"pages": 10},
        }]
    )

    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    record = store.upsert_document(
        doc_id="doc_123",
        filename="annual_report.pdf",
        doc_type="pdf",
        metadata={"pages": 10},
    )

    assert isinstance(record, DocumentRecord)
    assert record.doc_id == "doc_123"
    assert record.filename == "annual_report.pdf"
    mock_supabase_client.table.assert_called_with("documents")


@pytest.mark.unit
def test_upsert_chunks_with_models(mock_supabase_client):
    """Verify upsert_chunks handles ExtractedChunk Pydantic models."""
    mock_table = MagicMock()
    mock_supabase_client.table.return_value = mock_table
    mock_table.upsert.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"chunk_id": "c1"}, {"chunk_id": "c2"}])

    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    chunks = [
        ExtractedChunk(
            content="Sample text chunk 1",
            metadata=ChunkMetadata(
                source_doc="doc1.pdf",
                page_number=1,
                content_type=ContentType.TEXT,
                chunk_id="doc1_p1_0",
            ),
            embedding=[0.1, 0.2, 0.3],
        ),
        ExtractedChunk(
            content="| Col1 | Col2 |\n|---|---|\n| 10 | 20 |",
            metadata=ChunkMetadata(
                source_doc="doc1.pdf",
                page_number=1,
                content_type=ContentType.TABLE,
                chunk_id="doc1_p1_tbl1",
            ),
            embedding=[0.4, 0.5, 0.6],
        ),
    ]

    count = store.upsert_chunks(chunks, doc_id="doc_123")
    assert count == 2
    mock_supabase_client.table.assert_called_with("chunks")


@pytest.mark.unit
def test_similarity_search_with_metadata_filters(mock_supabase_client):
    """Verify similarity_search passes correct parameters to RPC match_chunks."""
    mock_supabase_client.rpc.return_value = MagicMock()
    mock_supabase_client.rpc.return_value.execute.return_value = MagicMock(
        data=[
            {
                "chunk_id": "doc1_p1_0",
                "doc_id": "doc_123",
                "content": "Relevant matched paragraph",
                "metadata": {"content_type": "text"},
                "similarity": 0.89,
                "filename": "doc1.pdf",
                "uploaded_at": "2026-08-29T12:00:00Z",
            }
        ]
    )

    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    filters = SearchFilter(doc_id="doc_123", content_type="text")
    results = store.similarity_search(query_embedding=[0.1, 0.2, 0.3], filters=filters, k=3)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].similarity == 0.89
    assert results[0].content == "Relevant matched paragraph"

    mock_supabase_client.rpc.assert_called_with(
        "match_chunks",
        {
            "query_embedding": [0.1, 0.2, 0.3],
            "filter_doc_id": "doc_123",
            "filter_content_type": "text",
            "filter_start_date": None,
            "filter_end_date": None,
            "match_count": 3,
        },
    )


@pytest.mark.unit
def test_delete_document(mock_supabase_client):
    """Verify delete_document calls table delete filtering by doc_id."""
    mock_table = MagicMock()
    mock_supabase_client.table.return_value = mock_table
    mock_table.delete.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[{"doc_id": "doc_123"}])

    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    success = store.delete_document("doc_123")

    assert success is True
    mock_table.delete.assert_called_once()
    mock_table.eq.assert_called_with("doc_id", "doc_123")


@pytest.mark.unit
def test_list_documents(mock_supabase_client):
    """Verify list_documents returns ordered list of DocumentRecord objects."""
    mock_table = MagicMock()
    mock_supabase_client.table.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.execute.return_value = MagicMock(
        data=[
            {"doc_id": "doc1", "filename": "doc1.pdf", "uploaded_at": "2026-08-29T10:00:00Z", "doc_type": "pdf", "metadata": {}},
            {"doc_id": "doc2", "filename": "doc2.pdf", "uploaded_at": "2026-08-28T10:00:00Z", "doc_type": "pdf", "metadata": {}},
        ]
    )

    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    docs = store.list_documents()

    assert len(docs) == 2
    assert docs[0].doc_id == "doc1"
    assert docs[1].doc_id == "doc2"


@pytest.mark.unit
def test_get_document_found_and_not_found(mock_supabase_client):
    """Verify get_document handles found and not found outcomes."""
    mock_table = MagicMock()
    mock_supabase_client.table.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.limit.return_value = mock_table

    # Found case
    mock_table.execute.return_value = MagicMock(
        data=[{"doc_id": "doc1", "filename": "doc1.pdf", "uploaded_at": "2026-08-29T10:00:00Z", "doc_type": "pdf", "metadata": {}}]
    )
    store = SupabaseVectorStore(supabase_client=mock_supabase_client)
    doc = store.get_document("doc1")
    assert doc is not None
    assert doc.filename == "doc1.pdf"

    # Not found case
    mock_table.execute.return_value = MagicMock(data=[])
    doc_none = store.get_document("doc_non_existent")
    assert doc_none is None


@pytest.mark.unit
def test_with_retry_exhaustion_raises_retrieval_error():
    """Verify with_retry raises RetrievalError when max retries are exhausted."""
    def failing_fn():
        raise ConnectionError("Network unreachable")

    with pytest.raises(RetrievalError, match="Operation 'test_op' failed after 2 attempts"):
        with_retry(failing_fn, max_retries=2, initial_backoff=0.01, operation_name="test_op")


@pytest.mark.unit
def test_get_embedding_model_singleton_caching():
    """Verify get_embedding_model caches and reuses the HuggingFaceEmbeddings instance."""
    from unittest.mock import patch
    from rag_platform.vectorstore.embeddings import _get_cached_embedding_model, get_embedding_model

    mock_instance = MagicMock()
    with patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_instance, create=True) as mock_cls:
        # Clear lru_cache for testing
        _get_cached_embedding_model.cache_clear()

        emb1 = get_embedding_model("test-model-name")
        emb2 = get_embedding_model("test-model-name")

        assert emb1 is emb2
        assert mock_cls.call_count == 1


@pytest.mark.unit
def test_warmup_embedding_model_uses_singleton_cache():
    """Warmup must reuse the existing singleton cache and avoid multiple initialization calls."""
    from rag_platform.vectorstore.embeddings import _get_cached_embedding_model, warmup_embedding_model

    mock_instance = MagicMock()
    with patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_instance, create=True) as mock_cls:
        _get_cached_embedding_model.cache_clear()

        emb1 = warmup_embedding_model("warmup-model-name")
        emb2 = warmup_embedding_model("warmup-model-name")

        assert emb1 is emb2
        assert mock_cls.call_count == 1


@pytest.mark.unit
def test_embedding_generation_dimension_and_count():
    """Verify embedding model generates exactly one 384-dimensional vector per input text."""
    from unittest.mock import MagicMock, patch
    from rag_platform.vectorstore.embeddings import _get_cached_embedding_model, get_embedding_model

    mock_instance = MagicMock()
    sample_texts = ["Quarterly financial results chunk 1", "Balance sheet assets chunk 2"]
    mock_instance.embed_documents.return_value = [[0.05] * 384 for _ in sample_texts]

    with patch("langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_instance, create=True):
        _get_cached_embedding_model.cache_clear()
        model = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        vectors = model.embed_documents(sample_texts)

        assert len(vectors) == len(sample_texts)
        for vec in vectors:
            assert len(vec) == 384


@pytest.mark.unit
def test_build_company_document_metadata_preserves_existing_fields():
    """Verify company metadata is merged into the active documents.metadata without clobbering generic fields."""
    metadata = build_company_document_metadata(
        company_id="c_42",
        symbol="aapl",
        name="Apple Inc.",
        year=2024,
        report_type="annual_report",
        storage_path="annual-reports/AAPL/2024_annual_report.pdf",
        extra_metadata={"total_pages": 14, "source": "uploaded"},
    )

    assert metadata["company_id"] == "c_42"
    assert metadata["symbol"] == "AAPL"
    assert metadata["name"] == "Apple Inc."
    assert metadata["scope"] == "company"
    assert metadata["report_type"] == "annual_report"
    assert metadata["year"] == 2024
    assert metadata["storage_path"] == "annual-reports/AAPL/2024_annual_report.pdf"
    assert metadata["total_pages"] == 14
    assert metadata["source"] == "uploaded"


@pytest.mark.unit
def test_upload_company_annual_report_uses_active_documents_and_chunks_pipeline(tmp_path):
    """Verify company reports are stored in the active documents/chunks schema with metadata and embeddings."""
    pdf_path = tmp_path / "annual_report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    chunks = [
        ExtractedChunk(
            content="Apple revenue grew strongly in 2024.",
            metadata=ChunkMetadata(
                source_doc="annual_report.pdf",
                page_number=1,
                content_type=ContentType.TEXT,
                chunk_id="annual_report.pdf_p1_0",
            ),
        ),
        ExtractedChunk(
            content="| Metric | Value |\n| --- | --- |\n| Revenue | $100B |",
            metadata=ChunkMetadata(
                source_doc="annual_report.pdf",
                page_number=2,
                content_type=ContentType.TABLE,
                chunk_id="annual_report.pdf_p2_tbl0",
            ),
        ),
    ]
    ingest_result = IngestionResult(
        source_doc="annual_report.pdf",
        total_pages=2,
        chunks=chunks,
        text_chunks_count=1,
        tables_count=1,
        images_count=0,
    )

    mock_supabase = MagicMock()
    mock_storage = MagicMock()
    mock_supabase.storage.from_.return_value = mock_storage
    mock_vectorstore = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]

    with patch("upload_company_report.create_client", return_value=mock_supabase), \
         patch("upload_company_report.ingest_pdf", return_value=ingest_result), \
         patch("upload_company_report.SupabaseVectorStore", return_value=mock_vectorstore), \
         patch("upload_company_report.get_embedding_model", return_value=mock_embedding_model):
        result = upload_company_annual_report(
            pdf_path=str(pdf_path),
            company_id="company_42",
            company_symbol="AAPL",
            company_name="Apple Inc.",
            year=2024,
            report_type="annual_report",
            file_name="annual_report.pdf",
            storage_bucket="annual-reports",
        )

    assert result["storage_path"] == "annual-reports/AAPL/annual_report.pdf"
    assert result["metadata"]["company_id"] == "company_42"
    assert result["metadata"]["symbol"] == "AAPL"
    assert result["metadata"]["scope"] == "company"
    assert result["metadata"]["year"] == 2024
    assert result["metadata"]["report_type"] == "annual_report"

    mock_supabase.storage.from_.assert_called_once_with("annual-reports")
    mock_storage.upload.assert_called_once()
    upload_call_kwargs = mock_storage.upload.call_args.kwargs
    assert upload_call_kwargs["path"] == "AAPL/annual_report.pdf"
    assert upload_call_kwargs["file_options"]["content-type"] == "application/pdf"

    mock_vectorstore.upsert_document.assert_called_once()
    document_kwargs = mock_vectorstore.upsert_document.call_args.kwargs
    assert document_kwargs["doc_type"] == "pdf"
    assert document_kwargs["filename"] == "annual_report.pdf"
    assert document_kwargs["metadata"]["company_id"] == "company_42"
    assert document_kwargs["metadata"]["storage_path"] == "annual-reports/AAPL/annual_report.pdf"

    mock_vectorstore.upsert_chunks.assert_called_once()
    assert mock_vectorstore.upsert_chunks.call_args.kwargs["doc_id"] == result["doc_id"]
    assert len(mock_vectorstore.upsert_chunks.call_args.kwargs["chunks"]) == 2
    assert all(chunk.embedding is not None for chunk in mock_vectorstore.upsert_chunks.call_args.kwargs["chunks"])
