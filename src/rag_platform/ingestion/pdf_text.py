"""Prose text extraction and chunking from PDF documents."""

import fitz as pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_platform.exceptions import IngestionError
from rag_platform.ingestion import ocr
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


def extract_page_text_chunks(
    page: pymupdf.Page,
    source_doc: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    allow_ocr_fallback: bool = True,
    start_chunk_idx: int = 0,
) -> tuple[list[ExtractedChunk], bool]:
    """Extract prose text from a PDF page, with OCR fallback for scanned pages.

    Args:
        page: PyMuPDF Page instance.
        source_doc: Name of the source PDF.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        allow_ocr_fallback: Whether to perform OCR if digital text is absent.
        start_chunk_idx: Offset for chunk indexing.

    Returns:
        Tuple of (list of ExtractedChunk objects, bool indicating if OCR was used).

    Raises:
        IngestionError: If text extraction fails unexpectedly.
    """
    page_num = page.number + 1
    used_ocr = False

    try:
        raw_text = page.get_text()

        if ocr.is_scanned_page(page) and allow_ocr_fallback:
            logger.info("Page %d of %s appears scanned; initiating OCR fallback", page_num, source_doc)
            ocr_text = ocr.extract_page_ocr_text(page)
            if ocr_text.strip():
                raw_text = ocr_text
                used_ocr = True

        cleaned_text = raw_text.strip()
        if not cleaned_text:
            return [], used_ocr

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = splitter.split_text(cleaned_text)

        chunks: list[ExtractedChunk] = []
        for idx, split_text in enumerate(splits):
            chunk_id = f"{source_doc}_p{page_num}_txt{idx}_{start_chunk_idx + idx}"
            chunk = ExtractedChunk(
                content=split_text,
                metadata=ChunkMetadata(
                    source_doc=source_doc,
                    page_number=page_num,
                    content_type=ContentType.TEXT,
                    chunk_id=chunk_id,
                    extra={"is_ocr": used_ocr, "split_index": idx},
                ),
            )
            chunks.append(chunk)

        return chunks, used_ocr

    except Exception as exc:
        logger.exception("Text extraction failed on page %d of %s", page_num, source_doc)
        raise IngestionError(
            f"Prose text extraction failed on page {page_num}: {exc}",
            details={"page": page_num, "source_doc": source_doc},
        ) from exc
