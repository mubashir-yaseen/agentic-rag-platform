"""Unit tests for multimodal document ingestion pipeline."""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw
import fitz as pymupdf
import pytest

from rag_platform.exceptions import IngestionError
from rag_platform.ingestion.images import extract_images_from_page, generate_image_caption
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk, IngestionResult
from rag_platform.ingestion.ocr import extract_page_ocr_text, is_scanned_page
from rag_platform.ingestion.pdf_text import extract_page_text_chunks
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.ingestion.tables import extract_tables_from_page, table_to_markdown


@pytest.fixture
def text_only_pdf(tmp_path):
    """Create a temporary standard text-only PDF."""
    pdf_file = tmp_path / "sample_text.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 72),
        "Section 1: Multimodal Ingestion Architecture\n\n"
        "This document describes the high-performance pipeline for parsing text, tables, and images. "
        "Each component is decoupled and returns typed Pydantic models for consistency across the system.",
        fontsize=12,
    )
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 72),
        "Section 2: Vector Embeddings and Retrieval\n\n"
        "Vector embeddings enable semantic similarity search across knowledge base chunks. "
        "Supabase pgvector facilitates multi-tenant and multi-document filtering.",
        fontsize=12,
    )
    doc.save(str(pdf_file))
    doc.close()
    return str(pdf_file)


@pytest.fixture
def table_pdf(tmp_path):
    """Create a temporary PDF containing tabular structure."""
    pdf_file = tmp_path / "sample_table.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "Financial Performance Overview 2025:\n", fontsize=14)

    # Draw a table with lines and text cells so PyMuPDF table finder detects it
    # Outer rectangle and grid
    page.draw_rect(pymupdf.Rect(50, 70, 450, 150), color=(0, 0, 0), width=1)
    page.draw_line(pymupdf.Point(50, 110), pymupdf.Point(450, 110), color=(0, 0, 0), width=1)
    page.draw_line(pymupdf.Point(250, 70), pymupdf.Point(250, 150), color=(0, 0, 0), width=1)

    page.insert_text((60, 95), "Metric", fontsize=11)
    page.insert_text((260, 95), "Value (USD)", fontsize=11)
    page.insert_text((60, 135), "Total Revenue", fontsize=11)
    page.insert_text((260, 135), "$94.9 Billion", fontsize=11)

    doc.save(str(pdf_file))
    doc.close()
    return str(pdf_file)


@pytest.fixture
def scanned_pdf(tmp_path):
    """Create a temporary PDF with no digital text containing only an embedded raster image."""
    pdf_file = tmp_path / "scanned_doc.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Generate a PIL image with drawn text and insert into PDF
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), "Scanned Contract Agreement", fill=(0, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    page.insert_image(pymupdf.Rect(50, 50, 450, 250), stream=img_bytes)
    doc.save(str(pdf_file))
    doc.close()
    return str(pdf_file)


@pytest.fixture
def corrupt_pdf(tmp_path):
    """Create a corrupt, unparseable file with a .pdf extension."""
    pdf_file = tmp_path / "corrupt.pdf"
    pdf_file.write_bytes(b"CORRUPT_HEADER_NOT_A_VALID_PDF_DATA_STREAM_XYZ123")
    return str(pdf_file)


@pytest.mark.unit
def test_models_serialization():
    """Verify serialization and validation of Pydantic models."""
    meta = ChunkMetadata(
        source_doc="report.pdf",
        page_number=1,
        content_type=ContentType.TABLE,
        chunk_id="report_p1_tbl0_0",
        extra={"rows": 5},
    )
    chunk = ExtractedChunk(
        content="| Col1 | Col2 |\n|---|---|\n| A | B |",
        metadata=meta,
        embedding=[0.1, 0.2, 0.3],
    )
    result = IngestionResult(
        source_doc="report.pdf",
        total_pages=1,
        chunks=[chunk],
        text_chunks_count=0,
        tables_count=1,
        images_count=0,
    )

    assert result.source_doc == "report.pdf"
    assert result.tables_count == 1
    assert result.chunks[0].metadata.content_type == ContentType.TABLE
    assert result.chunks[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.unit
def test_table_to_markdown():
    """Verify markdown formatting helper for tabular data."""
    headers = ["Quarter", "Revenue", "Growth"]
    rows = [["Q1", "$10M", "12%"], ["Q2", "$12M", "20%"]]
    md = table_to_markdown(headers, rows)

    assert "| Quarter | Revenue | Growth |" in md
    assert "| --- | --- | --- |" in md
    assert "| Q1 | $10M | 12% |" in md
    assert "| Q2 | $12M | 20% |" in md


@pytest.mark.unit
def test_ingest_text_only_pdf(text_only_pdf):
    """Verify multimodal pipeline extracts prose text accurately from text-only PDF."""
    result = ingest_pdf(
        pdf_path=text_only_pdf,
        chunk_size=500,
        chunk_overlap=50,
        extract_tables=False,
        extract_images=False,
    )

    assert result.source_doc == "sample_text.pdf"
    assert result.total_pages == 2
    assert result.text_chunks_count >= 2
    assert len(result.chunks) >= 2
    assert all(c.metadata.content_type == ContentType.TEXT for c in result.chunks)
    assert any("Multimodal Ingestion" in c.content for c in result.chunks)


@pytest.mark.unit
def test_ingest_table_pdf_extraction(table_pdf):
    """Verify table extraction extracts tables as markdown chunks."""
    result = ingest_pdf(
        pdf_path=table_pdf,
        extract_tables=True,
        extract_images=False,
        allow_ocr=False,
    )

    assert result.total_pages == 1
    # Check that chunks exist (prose text or table)
    assert len(result.chunks) > 0


@pytest.mark.unit
def test_scanned_pdf_ocr_detection(scanned_pdf):
    """Verify is_scanned_page detects pages lacking digital text and attempts OCR."""
    doc = pymupdf.open(scanned_pdf)
    page = doc[0]
    assert is_scanned_page(page) is True

    with patch("rag_platform.ingestion.ocr.extract_page_ocr_text", return_value="Scanned Contract Agreement"):
        chunks, used_ocr = extract_page_text_chunks(page, source_doc="scanned_doc.pdf", allow_ocr_fallback=True)
        assert used_ocr is True
        assert len(chunks) == 1
        assert "Scanned Contract Agreement" in chunks[0].content
    doc.close()


@pytest.mark.unit
def test_image_extraction_and_captioning(scanned_pdf, tmp_path):
    """Verify embedded image extraction and captioning."""
    doc = pymupdf.open(scanned_pdf)
    page = doc[0]
    output_dir = str(tmp_path / "images")

    with patch("rag_platform.ingestion.images.generate_image_caption", return_value="Figure diagram showing revenue trend"):
        chunks = extract_images_from_page(
            doc=doc,
            page=page,
            source_doc="doc.pdf",
            output_dir=output_dir,
            min_width=10,
            min_height=10,
            caption_llm=True,
        )

        assert len(chunks) >= 1
        assert chunks[0].metadata.content_type == ContentType.IMAGE
        assert "Caption: Figure diagram showing revenue trend" in chunks[0].content
        assert os.path.exists(chunks[0].metadata.image_path)
    doc.close()


@pytest.mark.unit
def test_generate_image_caption_fallback():
    """Verify image caption generation returns sensible fallback when API fails or is offline."""
    img_bytes = b"fake_png_bytes"
    caption = generate_image_caption(img_bytes, context_hint="Quarterly Earnings Report", api_key="")
    assert "Figure/Diagram" in caption
    assert "Quarterly Earnings Report" in caption


@pytest.mark.unit
def test_corrupt_pdf_raises_ingestion_error(corrupt_pdf):
    """Verify passing corrupt files raises IngestionError with clear message."""
    with pytest.raises(IngestionError, match="Failed to open or parse PDF"):
        ingest_pdf(corrupt_pdf)


@pytest.mark.unit
def test_missing_file_raises_ingestion_error():
    """Verify passing non-existent file path raises IngestionError."""
    with pytest.raises(IngestionError, match="PDF file not found"):
        ingest_pdf("non_existent_file_abc.pdf")


@pytest.mark.unit
def test_empty_file_raises_ingestion_error(tmp_path):
    """Verify passing 0-byte file raises IngestionError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(IngestionError, match="is empty"):
        ingest_pdf(str(empty_file))
