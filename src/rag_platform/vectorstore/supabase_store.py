"""Persistent multi-document vector store using Supabase and pgvector."""

from datetime import datetime
import time
from typing import Any, Callable, Optional, TypeVar
from pydantic import BaseModel, Field
from supabase import Client, create_client

from rag_platform.config import get_settings
from rag_platform.exceptions import ConfigurationError, RetrievalError
from rag_platform.ingestion.models import ExtractedChunk
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DocumentRecord(BaseModel):
    """Metadata schema representing a registered document in Supabase."""

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of upload")
    doc_type: str = Field(default="pdf", description="Document type / format")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom document-level metadata")


class SearchFilter(BaseModel):
    """Filter parameters for similarity search queries."""

    doc_id: Optional[str] = Field(default=None, description="Scope query to a specific document ID")
    content_type: Optional[str] = Field(default=None, description="Filter by text, table, or image")
    start_date: Optional[str] = Field(default=None, description="ISO-8601 start timestamp filter")
    end_date: Optional[str] = Field(default=None, description="ISO-8601 end timestamp filter")


class SearchResult(BaseModel):
    """Individual chunk search match returned by vector similarity search."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Document ID this chunk belongs to")
    content: str = Field(..., description="Text/markdown content of chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")
    similarity: float = Field(..., description="Cosine similarity score (0 to 1)")
    filename: Optional[str] = Field(default=None, description="Parent document filename")
    uploaded_at: Optional[str] = Field(default=None, description="Document upload timestamp")


def with_retry(
    operation: Callable[[], T],
    max_retries: int = 3,
    initial_backoff: float = 0.5,
    backoff_multiplier: float = 2.0,
    operation_name: str = "database_operation",
) -> T:
    """Execute a callable with exponential backoff retry on transient connection errors.

    Args:
        operation: Zero-argument callable to execute.
        max_retries: Number of retry attempts before giving up.
        initial_backoff: Initial sleep duration in seconds.
        backoff_multiplier: Multiplier applied to backoff after each failure.
        operation_name: Descriptive name for logging.

    Returns:
        The return value of the operation.

    Raises:
        RetrievalError: If all retry attempts fail or unrecoverable error occurs.
    """
    delay = initial_backoff
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "Attempt %d/%d failed for '%s': %s",
                attempt,
                max_retries,
                operation_name,
                exc,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff_multiplier

    logger.exception("All %d attempts failed for '%s'", max_retries, operation_name)
    raise RetrievalError(
        f"Operation '{operation_name}' failed after {max_retries} attempts: {last_exception}",
        details={"operation": operation_name, "error": str(last_exception)},
    ) from last_exception


class SupabaseVectorStore:
    """Manages multi-document storage, chunk vector indexing, and filtered similarity search."""

    def __init__(
        self,
        supabase_client: Optional[Client] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ) -> None:
        """Initialize the Supabase vector store client.

        Args:
            supabase_client: Optional pre-configured Supabase Client (useful for mocking).
            supabase_url: Supabase API URL.
            supabase_key: Supabase service role or anon key.

        Raises:
            ConfigurationError: If Supabase credentials are missing.
        """
        if supabase_client is not None:
            self.client: Client = supabase_client
            return

        settings = get_settings()
        url = supabase_url or settings.SUPABASE_URL
        key = supabase_key or settings.effective_supabase_key

        if not url or not key:
            raise ConfigurationError(
                "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )

        try:
            self.client = create_client(url.rstrip("/"), key)
            logger.info("SupabaseVectorStore successfully connected to %s", url)
        except Exception as exc:
            logger.exception("Failed to initialize Supabase client")
            raise RetrievalError(f"Supabase client initialization failed: {exc}") from exc

    def upsert_document(
        self,
        doc_id: str,
        filename: str,
        doc_type: str = "pdf",
        metadata: Optional[dict[str, Any]] = None,
    ) -> DocumentRecord:
        """Register or update a document in the knowledge base documents table.

        Args:
            doc_id: Unique identifier for the document.
            filename: Original name of the document file.
            doc_type: Document classification format.
            metadata: Custom metadata dictionary.

        Returns:
            Created DocumentRecord instance.

        Raises:
            RetrievalError: If the insert or update fails.
        """
        doc_data = {
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "metadata": metadata or {},
        }

        def _do_upsert() -> DocumentRecord:
            resp = self.client.table("documents").upsert(doc_data).execute()
            rows = getattr(resp, "data", None)
            if not rows:
                raise RetrievalError(
                    f"No data returned when upserting document {doc_id}",
                    details={"doc_id": doc_id},
                )
            row = rows[0]
            return DocumentRecord(
                doc_id=row["doc_id"],
                filename=row["filename"],
                uploaded_at=row.get("uploaded_at") or datetime.utcnow(),
                doc_type=row.get("doc_type", "pdf"),
                metadata=row.get("metadata", {}),
            )

        return with_retry(_do_upsert, operation_name=f"upsert_document_{doc_id}")

    def upsert_chunks(
        self,
        chunks: list[ExtractedChunk] | list[dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> int:
        """Batch insert or update chunk embeddings and metadata into the chunks table.

        Args:
            chunks: List of ExtractedChunk objects or structured chunk dictionaries.
            doc_id: Default document ID if not explicitly specified in chunk metadata.

        Returns:
            Number of successfully inserted/updated chunks.

        Raises:
            RetrievalError: If chunk upsertion fails.
        """
        if not chunks:
            return 0

        payload: list[dict[str, Any]] = []
        for c in chunks:
            if isinstance(c, ExtractedChunk):
                target_doc_id = c.metadata.source_doc if not doc_id else doc_id
                payload.append({
                    "chunk_id": c.metadata.chunk_id,
                    "doc_id": target_doc_id,
                    "content": c.content,
                    "embedding": c.embedding,
                    "metadata": c.metadata.model_dump(),
                })
            elif isinstance(c, dict):
                c_id = c.get("chunk_id") or c.get("id") or f"{doc_id}_{c.get('index', 0)}"
                meta = c.get("metadata", {})
                payload.append({
                    "chunk_id": c_id,
                    "doc_id": c.get("doc_id", doc_id),
                    "content": c.get("content", ""),
                    "embedding": c.get("embedding"),
                    "metadata": meta,
                })

        def _do_chunks_upsert() -> int:
            resp = self.client.table("chunks").upsert(payload).execute()
            rows = getattr(resp, "data", None)
            return len(rows) if rows is not None else len(payload)

        return with_retry(_do_chunks_upsert, operation_name=f"upsert_chunks_batch_{len(payload)}")

    def similarity_search(
        self,
        query_embedding: list[float],
        filters: Optional[dict[str, Any] | SearchFilter] = None,
        k: int = 5,
    ) -> list[SearchResult]:
        """Perform semantic similarity search using pgvector with multi-attribute metadata filtering.

        Args:
            query_embedding: Dense embedding vector of the query.
            filters: Optional dictionary or SearchFilter with doc_id, content_type, start_date, end_date.
            k: Maximum number of top matching chunks to retrieve.

        Returns:
            List of SearchResult objects ordered by descending cosine similarity.

        Raises:
            RetrievalError: If the RPC call fails.
        """
        filter_dict: dict[str, Any] = {}
        if isinstance(filters, SearchFilter):
            filter_dict = filters.model_dump(exclude_none=True)
        elif isinstance(filters, dict):
            filter_dict = {k_: v_ for k_, v_ in filters.items() if v_ is not None}

        rpc_params = {
            "query_embedding": query_embedding,
            "filter_doc_id": filter_dict.get("doc_id"),
            "filter_content_type": filter_dict.get("content_type"),
            "filter_start_date": filter_dict.get("start_date"),
            "filter_end_date": filter_dict.get("end_date"),
            "match_count": k,
        }

        def _do_search() -> list[SearchResult]:
            resp = self.client.rpc("match_chunks", rpc_params).execute()
            rows = resp.data or []
            results: list[SearchResult] = []
            for row in rows:
                results.append(
                    SearchResult(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        content=row["content"],
                        metadata=row.get("metadata", {}),
                        similarity=float(row.get("similarity", 0.0)),
                        filename=row.get("filename"),
                        uploaded_at=str(row.get("uploaded_at")) if row.get("uploaded_at") else None,
                    )
                )
            return results

        return with_retry(_do_search, operation_name="similarity_search_match_chunks")

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its cascaded chunks from Supabase.

        Args:
            doc_id: Identifier of the document to delete.

        Returns:
            True if deletion succeeded.

        Raises:
            RetrievalError: If deletion fails.
        """
        def _do_delete() -> bool:
            resp = self.client.table("documents").delete().eq("doc_id", doc_id).execute()
            logger.info("Successfully deleted document %s and cascaded chunks", doc_id)
            return True

        return with_retry(_do_delete, operation_name=f"delete_document_{doc_id}")

    def list_documents(self) -> list[DocumentRecord]:
        """Fetch all registered documents ordered by most recent upload.

        Returns:
            List of DocumentRecord items.

        Raises:
            RetrievalError: If query fails.
        """
        def _do_list() -> list[DocumentRecord]:
            resp = (
                self.client.table("documents")
                .select("*")
                .order("uploaded_at", desc=True)
                .execute()
            )
            rows = resp.data or []
            return [
                DocumentRecord(
                    doc_id=r["doc_id"],
                    filename=r["filename"],
                    uploaded_at=r.get("uploaded_at") or datetime.utcnow(),
                    doc_type=r.get("doc_type", "pdf"),
                    metadata=r.get("metadata", {}),
                )
                for r in rows
            ]

        return with_retry(_do_list, operation_name="list_documents")

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Fetch metadata for a single document by doc_id.

        Args:
            doc_id: Identifier of document to retrieve.

        Returns:
            DocumentRecord if found, None otherwise.
        """
        def _do_get() -> Optional[DocumentRecord]:
            resp = (
                self.client.table("documents")
                .select("*")
                .eq("doc_id", doc_id)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                return None
            r = rows[0]
            return DocumentRecord(
                doc_id=r["doc_id"],
                filename=r["filename"],
                uploaded_at=r.get("uploaded_at") or datetime.utcnow(),
                doc_type=r.get("doc_type", "pdf"),
                metadata=r.get("metadata", {}),
            )

        return with_retry(_do_get, operation_name=f"get_document_{doc_id}")
