"""Multimodal ingestion package for extracting text, tables, figures, and scanned PDFs."""

from rag_platform.ingestion.images import extract_images_from_page, generate_image_caption
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk, IngestionResult
from rag_platform.ingestion.ocr import extract_page_ocr_text, is_scanned_page
from rag_platform.ingestion.pdf_text import extract_page_text_chunks
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.ingestion.tables import extract_tables_from_page, table_to_markdown

__all__ = [
    "ChunkMetadata",
    "ContentType",
    "ExtractedChunk",
    "IngestionResult",
    "extract_images_from_page",
    "extract_page_ocr_text",
    "extract_page_text_chunks",
    "extract_tables_from_page",
    "generate_image_caption",
    "ingest_pdf",
    "is_scanned_page",
    "table_to_markdown",
]
