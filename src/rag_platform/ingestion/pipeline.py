"""Unified multimodal PDF ingestion pipeline coordinating text, table, and image extraction."""

import os
from typing import Optional
import fitz as pymupdf

from rag_platform.config import get_settings
from rag_platform.exceptions import IngestionError
from rag_platform.ingestion.images import extract_images_from_page
from rag_platform.ingestion.models import ExtractedChunk, IngestionResult
from rag_platform.ingestion.pdf_text import extract_page_text_chunks
from rag_platform.ingestion.tables import extract_tables_from_page
from rag_platform.logging_config import get_correlation_id, get_logger

logger = get_logger(__name__)


def ingest_pdf(
    pdf_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    extract_tables: bool = True,
    extract_images: bool = True,
    allow_ocr: bool = True,
    output_images_dir: str = "data/extracted_images",
    caption_llm: bool = True,
) -> IngestionResult:
    """Execute the end-to-end multimodal ingestion pipeline on a PDF file.

    Extracts text, structured tables as markdown, and figures/diagrams with captions,
    returning structured typed chunks with consistent metadata.

    Args:
        pdf_path: Local path to the PDF document.
        chunk_size: Maximum characters per prose text chunk.
        chunk_overlap: Overlap between consecutive prose text chunks.
        extract_tables: Whether to detect and extract tabular data.
        extract_images: Whether to extract and caption embedded figures.
        allow_ocr: Whether to run OCR on scanned/image-only pages.
        output_images_dir: Directory where extracted images will be saved.
        caption_llm: Whether to invoke vision LLM to caption extracted images.

    Returns:
        IngestionResult containing all multimodal chunks and execution metrics.

    Raises:
        IngestionError: If the file is missing, corrupt, unreadable, or extraction fails.
    """
    settings = get_settings()
    c_size = chunk_size or settings.RAG_CHUNK_SIZE
    c_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
    corr_id = get_correlation_id()

    source_doc = os.path.basename(pdf_path)
    logger.info(
        "Starting multimodal ingestion for %s (correlation_id=%s)",
        source_doc,
        corr_id,
        extra={"file_path": pdf_path, "chunk_size": c_size, "chunk_overlap": c_overlap},
    )

    if not os.path.exists(pdf_path):
        raise IngestionError(f"PDF file not found at path: '{pdf_path}'")

    if os.path.getsize(pdf_path) == 0:
        raise IngestionError(
            f"PDF file '{source_doc}' is empty (0 bytes).",
            details={"file_path": pdf_path},
        )

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.exception("Corrupt or invalid PDF file: %s", pdf_path)
        raise IngestionError(
            f"Failed to open or parse PDF '{source_doc}'. The file may be corrupt or not a valid PDF: {exc}",
            details={"file_path": pdf_path},
        ) from exc

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise IngestionError(f"PDF '{source_doc}' contains 0 pages.")

        all_chunks: list[ExtractedChunk] = []
        ocr_pages_used: list[int] = []
        total_tables = 0
        total_images = 0
        total_text_chunks = 0

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Prose Text Extraction (+ OCR fallback if scanned)
            text_chunks, used_ocr = extract_page_text_chunks(
                page=page,
                source_doc=source_doc,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                allow_ocr_fallback=allow_ocr,
                start_chunk_idx=len(all_chunks),
            )
            all_chunks.extend(text_chunks)
            total_text_chunks += len(text_chunks)
            if used_ocr:
                ocr_pages_used.append(page_num)

            # 2. Table Extraction
            if extract_tables:
                table_chunks = extract_tables_from_page(
                    page=page,
                    source_doc=source_doc,
                    start_chunk_idx=len(all_chunks),
                )
                all_chunks.extend(table_chunks)
                total_tables += len(table_chunks)

            # 3. Image & Figure Extraction
            if extract_images:
                image_chunks = extract_images_from_page(
                    doc=doc,
                    page=page,
                    source_doc=source_doc,
                    output_dir=output_images_dir,
                    caption_llm=caption_llm,
                    start_chunk_idx=len(all_chunks),
                )
                all_chunks.extend(image_chunks)
                total_images += len(image_chunks)

        logger.info(
            "Ingestion completed for %s: %d total chunks (%d text, %d tables, %d images, %d OCR pages)",
            source_doc,
            len(all_chunks),
            total_text_chunks,
            total_tables,
            total_images,
            len(ocr_pages_used),
            extra={
                "total_chunks": len(all_chunks),
                "text_chunks": total_text_chunks,
                "tables_count": total_tables,
                "images_count": total_images,
                "ocr_pages": ocr_pages_used,
            },
        )

        return IngestionResult(
            source_doc=source_doc,
            total_pages=total_pages,
            chunks=all_chunks,
            text_chunks_count=total_text_chunks,
            tables_count=total_tables,
            images_count=total_images,
            ocr_pages=ocr_pages_used,
        )

    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during multimodal ingestion of %s", source_doc)
        raise IngestionError(
            f"Multimodal ingestion encountered an unrecoverable failure: {exc}",
            details={"file_path": pdf_path},
        ) from exc
    finally:
        doc.close()
