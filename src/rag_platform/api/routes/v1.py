"""Version 1 API Routes (/api/v1)."""

import asyncio
import json
import os
import tempfile
import time
import uuid
from typing import Any, AsyncGenerator, Iterator, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse

from rag_platform.agent.excel_tool import resolve_generated_file_path, sanitize_excel_filename
from rag_platform.agent.graph import AgentResult, run_agent
from rag_platform.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    ReadyResponse,
)
from rag_platform.config import get_settings
from rag_platform.eval.run_eval import EvalReport, run_evaluation
from rag_platform.exceptions import IngestionError, RagPlatformError
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.logging_config import get_correlation_id, get_logger
from rag_platform.vectorstore import SupabaseVectorStore, get_embedding_model
from rag_platform.vectorstore.embeddings import EMBEDDING_DIMENSIONS

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check() -> HealthResponse:
    """Check service operational health."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="1.0.0",
        environment=settings.ENVIRONMENT,
    )


@router.get("/ready", response_model=ReadyResponse, summary="Service Readiness Probe")
async def readiness_check() -> ReadyResponse:
    """Check whether external dependencies (Supabase, LLM) are configured and ready."""
    settings = get_settings()
    db_ok = bool(settings.SUPABASE_URL and settings.effective_supabase_key)
    llm_ok = bool(settings.effective_api_key)
    embed_ok = bool(settings.GOOGLE_API_KEY)

    is_ready = db_ok and llm_ok and embed_ok
    return ReadyResponse(
        status="ready" if is_ready else "degraded",
        database_configured=db_ok,
        llm_configured=llm_ok,
        checks={
            "supabase_credentials": db_ok,
            "llm_api_key": llm_ok,
            "google_api_key": embed_ok,
        },
    )


# --------------------------------------------------------------------------- #
# Self-test: open /api/v1/diag in a browser to see what is (not) working.
# It reports only yes/no flags, counts and scrubbed error text, never secrets.
# --------------------------------------------------------------------------- #
_DIAG_CACHE: dict[str, Any] = {"at": 0.0, "report": None}
_DIAG_MIN_INTERVAL_SECONDS = 15.0  # protects the free embedding quota from being hammered


def _scrub(text: Any, settings: Any) -> str:
    """Shorten an error message and blank out any secret that might appear in it."""
    out = str(text)
    for name in (
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_ANON_KEY",
        "SUPABASE_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
    ):
        secret = getattr(settings, name, "")
        if secret and len(secret) > 6:
            out = out.replace(secret, "***")
    return out[:400]


def _run_diagnostics() -> dict[str, Any]:
    """Try each dependency once and describe the result (blocking; run in a thread)."""
    settings = get_settings()
    hints: list[str] = []
    report: dict[str, Any] = {
        "keys_present": {
            "google_api_key": bool(settings.GOOGLE_API_KEY),
            "supabase_url": bool(settings.SUPABASE_URL),
            "supabase_key": bool(settings.effective_supabase_key),
            "llm_api_key": bool(settings.effective_api_key),
        },
        "llm": {"model_mode": settings.LLM_MODEL_MODE, "model": settings.LLM_MODEL},
        "expected_embedding_dimensions": EMBEDDING_DIMENSIONS,
    }
    if not settings.GOOGLE_API_KEY:
        hints.append("Add GOOGLE_API_KEY (free, from aistudio.google.com/app/apikey) in Render > Environment.")
    if not settings.effective_api_key:
        hints.append("Add OPENROUTER_API_KEY in Render > Environment, or answers cannot be generated.")

    query_vec = None
    try:
        query_vec = get_embedding_model().embed_query("diagnostic ping")
        report["embedding"] = {"ok": len(query_vec) == EMBEDDING_DIMENSIONS, "dimension": len(query_vec)}
        if len(query_vec) != EMBEDDING_DIMENSIONS:
            hints.append("The embedding size does not match the database column size.")
    except Exception as exc:
        report["embedding"] = {"ok": False, "error": _scrub(exc, settings)}

    store = None
    try:
        store = SupabaseVectorStore()
        documents = store.list_documents()
        database: dict[str, Any] = {"ok": True, "documents": len(documents)}
        try:
            resp = store.client.table("chunks").select("chunk_id", count="exact").limit(1).execute()
            database["chunks"] = resp.count
            if documents and resp.count == 0:
                hints.append(
                    "Documents exist but none has searchable content. Delete them (sidebar > Documents) and upload again."
                )
        except Exception as exc:
            database["chunks_error"] = _scrub(exc, settings)
        report["database"] = database
    except Exception as exc:
        report["database"] = {"ok": False, "error": _scrub(exc, settings)}
        hints.append("Check SUPABASE_URL and the Supabase key in Render > Environment.")

    if query_vec is not None and store is not None and report["database"]["ok"]:
        try:
            store.similarity_search(query_embedding=query_vec, k=1)
            report["vector_search"] = {"ok": True}
        except Exception as exc:
            message = _scrub(exc, settings)
            report["vector_search"] = {"ok": False, "error": message}
            hints.append(
                "Vector search failed. If the message mentions dimensions or match_chunks, run "
                "002_switch_to_google_embeddings_768.sql in the Supabase SQL editor."
            )
    else:
        report["vector_search"] = {"ok": None, "note": "skipped because embeddings or the database failed"}

    report["ok"] = bool(report["embedding"]["ok"] and report["database"]["ok"] and report["vector_search"].get("ok"))
    report["hints"] = hints
    return report


@router.get("/diag", summary="Self-test: keys, embeddings, database, vector search")
async def diagnostics() -> dict[str, Any]:
    """Run a quick end-to-end check of everything ingestion and retrieval depend on."""
    cached = _DIAG_CACHE["report"]
    if cached is not None and time.monotonic() - _DIAG_CACHE["at"] < _DIAG_MIN_INTERVAL_SECONDS:
        return {**cached, "cached": True}
    report = await run_in_threadpool(_run_diagnostics)
    _DIAG_CACHE.update(at=time.monotonic(), report=report)
    return report


UPSERT_BATCH_SIZE = 100  # chunks per database write (keeps each request small)


def _ingest_pdf_sync(
    tmp_path: str, filename: str, doc_id: str, chunk_size: int, chunk_overlap: int
) -> IngestResponse:
    """Parse a PDF, embed its chunks and store them. Blocking: call from a worker thread."""
    ingest_result = ingest_pdf(
        pdf_path=tmp_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extract_tables=True,
        extract_images=True,
        allow_ocr=True,
        caption_llm=False,
    )
    if not ingest_result.chunks:
        raise IngestionError(
            f"No text could be extracted from '{filename}'. It may be a scanned PDF (pictures of pages) "
            "and OCR is not available on this server."
        )

    settings = get_settings()
    if settings.SUPABASE_URL and settings.effective_supabase_key:
        try:
            embedder = get_embedding_model()
            contents = [c.content for c in ingest_result.chunks]
            embeddings = embedder.embed_documents(contents)
            if len(embeddings) != len(ingest_result.chunks):
                raise IngestionError(
                    f"Mismatch in embeddings count: generated {len(embeddings)} for {len(ingest_result.chunks)} chunks"
                )
            for i, emb in enumerate(embeddings):
                if not emb or len(emb) != EMBEDDING_DIMENSIONS:
                    raise IngestionError(
                        f"Invalid embedding dimension for chunk {i}: expected {EMBEDDING_DIMENSIONS}, "
                        f"got {len(emb) if emb else 0}"
                    )
                ingest_result.chunks[i].embedding = emb
        except IngestionError:
            raise
        except Exception as exc:
            logger.exception("Failed to generate chunk embeddings during ingestion: %s", exc)
            raise IngestionError(
                f"Failed to generate dense vector embeddings for document '{filename}': {exc}"
            ) from exc

        vstore = SupabaseVectorStore()
        vstore.upsert_document(
            doc_id=doc_id,
            filename=filename,
            doc_type="pdf",
            metadata={
                "total_pages": ingest_result.total_pages,
                "tables": ingest_result.tables_count,
                "images": ingest_result.images_count,
            },
        )
        try:
            for start in range(0, len(ingest_result.chunks), UPSERT_BATCH_SIZE):
                vstore.upsert_chunks(
                    chunks=ingest_result.chunks[start : start + UPSERT_BATCH_SIZE],
                    doc_id=doc_id,
                )
        except Exception:
            # Do not leave a "ghost" document (listed, but with nothing to search).
            try:
                vstore.delete_document(doc_id)
            except Exception:
                logger.exception("Could not remove partially stored document %s", doc_id)
            raise

    return IngestResponse(
        doc_id=doc_id,
        filename=filename,
        total_pages=ingest_result.total_pages,
        total_chunks=len(ingest_result.chunks),
        tables_count=ingest_result.tables_count,
        images_count=ingest_result.images_count,
        status="completed",
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Multimodal PDF Document",
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF document file to ingest"),
    chunk_size: int = Form(default=1000, description="Target chunk size in characters"),
    chunk_overlap: int = Form(default=200, description="Chunk overlap size in characters"),
) -> IngestResponse:
    """Process and persist a multimodal PDF document (extracting text, tables, images, and OCR)."""
    filename = file.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise IngestionError("Only PDF files are supported for ingestion.")

    content = await file.read()
    if not content:
        raise IngestionError("Uploaded file is empty.")

    doc_id = str(uuid.uuid4())[:8] + "_" + filename

    # Save to temporary file for PDF processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Heavy, blocking work (PDF parsing, embedding calls, database writes) runs in a worker
        # thread, so /health and other requests stay responsive while a PDF is being indexed.
        return await run_in_threadpool(
            _ingest_pdf_sync, tmp_path, filename, doc_id, chunk_size, chunk_overlap
        )
    except RagPlatformError:
        raise  # already carries a readable message and the right HTTP status
    except Exception as exc:
        logger.exception("Unexpected failure while indexing %s", filename)
        raise IngestionError(f"Unexpected error while indexing '{filename}': {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


STREAM_PIECE_CHARS = 24  # characters per streamed event (one event per character was very slow)


def _answer_pieces(text: str, size: int = STREAM_PIECE_CHARS) -> Iterator[str]:
    """Split a finished answer into small pieces for the typing effect."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def _sse_generator(request: ChatRequest, doc_filters: Optional[dict], allow_retrieval: bool) -> AsyncGenerator[str, None]:
    """Stream agent execution steps and final answer tokens via SSE without disturbing the JSON path."""
    request_start = time.perf_counter()
    corr_id = get_correlation_id()
    logger.info("PERF: api_request_start")
    try:
        yield f"data: {json.dumps({'event': 'start', 'query': request.query, 'correlation_id': corr_id})}\n\n"
        await asyncio.sleep(0.01)

        # The agent is slow and blocking: run it in a worker thread, and send an SSE comment line
        # every few seconds while waiting so proxies do not drop an apparently idle connection.
        agent_task = asyncio.ensure_future(
            run_in_threadpool(
                run_agent,
                query=request.query,
                doc_filters=doc_filters,
                allow_retrieval=allow_retrieval,
                mode=request.mode,
            )
        )
        while True:
            finished, _ = await asyncio.wait({agent_task}, timeout=8.0)
            if finished:
                break
            yield ": keep-alive\n\n"
        result = agent_task.result()
        for step in result.reasoning_path:
            yield f"data: {json.dumps({'event': 'reasoning', 'step': step})}\n\n"
            await asyncio.sleep(0.01)

        for piece in _answer_pieces(result.answer):
            yield f"data: {json.dumps({'event': 'token', 'text': piece})}\n\n"
            await asyncio.sleep(0.01)

        yield f"data: {json.dumps({'event': 'end', 'answer': result.answer, 'routing_decision': result.routing_decision, 'sources': result.sources, 'tool_outputs': result.tool_outputs})}\n\n"
    except Exception as exc:
        logger.exception("Streaming chat failed for correlation_id=%s: %s", corr_id, exc)
        yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"
    finally:
        logger.info("PERF: api_request_total=%.2fs", time.perf_counter() - request_start)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Agentic Multi-Document Chat",
)
async def chat_endpoint(request: ChatRequest):
    """Execute LangGraph agent workflow across internal knowledge base and external MCP tools."""
    request_start = time.perf_counter()
    logger.info("PERF: api_request_start")

    # Respect client-declared UI mode to enforce strict isolation server-side.
    mode = (request.mode or "").strip()
    is_qna = bool(mode and mode.lower() in ("q&a", "qa", "qna", "general q&a", "general q&a mode"))

    # If client indicates Q&A mode, enforce no document/company scope regardless
    # of what the client supplied; this prevents accidental retrieval leakage.
    if is_qna:
        doc_filters = None
        allow_retrieval_flag = False
    else:
        # Start with any explicit doc_filters provided, but we'll modify according to mode
        doc_filters = dict(request.doc_filters or {})
        if request.doc_id:
            doc_filters["doc_id"] = request.doc_id
        if request.content_type and request.content_type != "all":
            doc_filters["content_type"] = request.content_type
        if request.company_id:
            doc_filters["company_id"] = request.company_id
        allow_retrieval_flag = True

    # Research mode requires an explicit selected company; reject otherwise.
    if mode and mode.lower() in ("research",):
        if not (request.company_id or (doc_filters and doc_filters.get("company_id"))):
            raise HTTPException(status_code=400, detail="Research mode requests must include a company_id or select a company.")

    if request.stream:
        return StreamingResponse(
            _sse_generator(request, doc_filters if doc_filters else None, allow_retrieval=allow_retrieval_flag),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Correlation-ID": get_correlation_id(),
            },
        )

    result = await run_in_threadpool(
        run_agent,
        query=request.query,
        doc_filters=doc_filters if doc_filters else None,
        allow_retrieval=allow_retrieval_flag,
        mode=mode,
    )
    logger.info("PERF: api_request_total=%.2fs", time.perf_counter() - request_start)
    return ChatResponse(
        query=request.query,
        answer=result.answer,
        routing_decision=result.routing_decision,
        reasoning_path=result.reasoning_path,
        sources=result.sources,
        tool_outputs=result.tool_outputs,
        correlation_id=get_correlation_id(),
    )


@router.get("/eval", response_model=EvalReport, summary="Run RAGAS Evaluation Benchmark")
async def run_eval_endpoint(limit: Optional[int] = Query(default=None, ge=1, le=50)) -> EvalReport:
    """Run automated RAGAS evaluation over labeled benchmark test set."""
    logger.info("Triggering API evaluation benchmark (limit=%s)", limit)
    return run_evaluation(limit=limit)


@router.get("/files/{file_id}", summary="Download an Agent-Generated File")
async def download_generated_file(file_id: str, filename: Optional[str] = Query(default=None)):
    """Serve a previously agent-generated file (e.g. an Excel export).

    file_id is an opaque UUID-hex token; resolve_generated_file_path validates its
    shape and confirms the resolved path stays inside the controlled output directory
    before anything is served, so this endpoint cannot be used for path traversal.
    """
    path = resolve_generated_file_path(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Generated file not found or has expired.")

    display_name = sanitize_excel_filename(filename) if filename else path.name
    return FileResponse(
        path=str(path),
        filename=display_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
