"""Table extraction from PDF documents using PyMuPDF and fallback table engines."""

from typing import Any, Optional
import fitz as pymupdf

from rag_platform.exceptions import IngestionError
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


def table_to_markdown(headers: list[str], rows: list[list[Any]]) -> str:
    """Convert table headers and rows into clean GitHub-flavored markdown.

    Args:
        headers: List of column header names.
        rows: List of row lists containing cell data.

    Returns:
        Formatted markdown table string.
    """
    cleaned_headers = [str(h or "").strip().replace("\n", " ") for h in headers]
    if not any(cleaned_headers):
        cleaned_headers = [f"Col {i + 1}" for i in range(len(rows[0]) if rows else 1)]

    header_line = "| " + " | ".join(cleaned_headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(cleaned_headers)) + " |"

    body_lines = []
    for row in rows:
        cleaned_row = [str(cell or "").strip().replace("\n", " ") for cell in row]
        # Pad row if shorter than headers
        while len(cleaned_row) < len(cleaned_headers):
            cleaned_row.append("")
        body_lines.append("| " + " | ".join(cleaned_row[:len(cleaned_headers)]) + " |")

    return "\n".join([header_line, separator_line] + body_lines)


def extract_tables_from_page(
    page: pymupdf.Page,
    source_doc: str,
    start_chunk_idx: int = 0,
) -> list[ExtractedChunk]:
    """Extract tabular structures from a single PDF page and return typed chunks.

    Args:
        page: PyMuPDF Page instance.
        source_doc: Name or identifier of the source PDF.
        start_chunk_idx: Index offset for generated chunk IDs.

    Returns:
        List of ExtractedChunk objects with content_type=ContentType.TABLE.

    Raises:
        IngestionError: If table parsing encounters an unexpected failure.
    """
    table_chunks: list[ExtractedChunk] = []
    page_num = page.number + 1

    try:
        # PyMuPDF built-in high-accuracy table finder
        tables = page.find_tables()
        if not tables or len(tables.tables) == 0:
            return []

        for idx, tbl in enumerate(tables.tables):
            raw_table_data = tbl.extract()
            if not raw_table_data or len(raw_table_data) < 2:
                continue

            headers = [str(cell or "") for cell in raw_table_data[0]]
            rows = raw_table_data[1:]

            md_content = table_to_markdown(headers, rows)
            chunk_id = f"{source_doc}_p{page_num}_tbl{idx}_{start_chunk_idx + idx}"

            chunk = ExtractedChunk(
                content=f"### Table (Page {page_num})\n{md_content}",
                metadata=ChunkMetadata(
                    source_doc=source_doc,
                    page_number=page_num,
                    content_type=ContentType.TABLE,
                    chunk_id=chunk_id,
                    extra={
                        "table_index": idx,
                        "row_count": len(rows),
                        "col_count": len(headers),
                    },
                ),
            )
            table_chunks.append(chunk)

        if table_chunks:
            logger.info(
                "Extracted %d tables from page %d of %s",
                len(table_chunks),
                page_num,
                source_doc,
            )
        return table_chunks

    except Exception as exc:
        logger.exception("Failed to extract tables from page %d of %s", page_num, source_doc)
        raise IngestionError(
            f"Table extraction failed on page {page_num}: {exc}",
            details={"page": page_num, "source_doc": source_doc},
        ) from exc
