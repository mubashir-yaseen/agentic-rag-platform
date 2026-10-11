"""Version 1 API Routes (/api/v1)."""

import asyncio
import json
import os
import tempfile
import time
import uuid
from typing import AsyncGenerator, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
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
from rag_platform.exceptions import IngestionError
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.logging_config import get_correlation_id, get_logger
from rag_platform.vectorstore import SupabaseVectorStore, get_embedding_model

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

    is_ready = db_ok and llm_ok
    return ReadyResponse(
        status="ready" if is_ready else "degraded",
        database_configured=db_ok,
        llm_configured=llm_ok,
        checks={
            "supabase_credentials": db_ok,
            "llm_api_key": llm_ok,
            "huggingface_token": bool(settings.HUGGINGFACEHUB_API_TOKEN),
        },
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
        ingest_result = ingest_pdf(
            pdf_path=tmp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extract_tables=True,
            extract_images=True,
            allow_ocr=True,
            caption_llm=False,
        )

        settings = get_settings()
        if settings.SUPABASE_URL and settings.effective_supabase_key:
            if ingest_result.chunks:
                try:
                    hf = get_embedding_model()
                    contents = [c.content for c in ingest_result.chunks]
                    embeddings = hf.embed_documents(contents)
                    if len(embeddings) != len(ingest_result.chunks):
                        raise IngestionError(
                            f"Mismatch in embeddings count: generated {len(embeddings)} for {len(ingest_result.chunks)} chunks"
                        )
                    for i, emb in enumerate(embeddings):
                        if not emb or len(emb) != 384:
                            raise IngestionError(
                                f"Invalid embedding dimension for chunk {i}: expected 384, got {len(emb) if emb else 0}"
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
            if ingest_result.chunks:
                vstore.upsert_chunks(
                    chunks=ingest_result.chunks,
                    doc_id=doc_id,
                )

        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            total_pages=ingest_result.total_pages,
            total_chunks=len(ingest_result.chunks),
            tables_count=ingest_result.tables_count,
            images_count=ingest_result.images_count,
            status="completed",
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _sse_generator(request: ChatRequest, doc_filters: Optional[dict], allow_retrieval: bool) -> AsyncGenerator[str, None]:
    """Stream agent execution steps and final answer tokens via SSE without disturbing the JSON path."""
    request_start = time.perf_counter()
    corr_id = get_correlation_id()
    logger.info("PERF: api_request_start")
    try:
        yield f"data: {json.dumps({'event': 'start', 'query': request.query, 'correlation_id': corr_id})}\n\n"
        await asyncio.sleep(0.01)

        result = run_agent(
            query=request.query,
            doc_filters=doc_filters,
            allow_retrieval=allow_retrieval,
            mode=request.mode,
        )
        for step in result.reasoning_path:
            yield f"data: {json.dumps({'event': 'reasoning', 'step': step})}\n\n"
            await asyncio.sleep(0.01)

        for token in result.answer:
            yield f"data: {json.dumps({'event': 'token', 'text': token})}\n\n"
            await asyncio.sleep(0.005)

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

    result = run_agent(
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
