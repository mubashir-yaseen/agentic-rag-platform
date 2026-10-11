"""Pydantic schemas and data models for multimodal document ingestion."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Supported content types within the multimodal ingestion pipeline."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ChunkMetadata(BaseModel):
    """Metadata attributes associated with an ingested chunk."""

    source_doc: str = Field(..., description="Name or relative path of the source document")
    page_number: int = Field(..., description="1-indexed page number in the original PDF")
    content_type: ContentType = Field(default=ContentType.TEXT, description="Type of extracted content")
    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    image_path: Optional[str] = Field(default=None, description="Path to saved image file if content_type is image")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata fields")


class ExtractedChunk(BaseModel):
    """A discrete unit of text, table markdown/HTML, or image caption with metadata."""

    content: str = Field(..., description="Searchable textual content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Structured metadata for the chunk")
    embedding: Optional[list[float]] = Field(default=None, description="Dense vector embedding")


class IngestionResult(BaseModel):
    """Summary and chunk payload resulting from processing a document."""

    source_doc: str = Field(..., description="Name of processed file")
    total_pages: int = Field(default=0, description="Total pages analyzed")
    chunks: list[ExtractedChunk] = Field(default_factory=list, description="Extracted multimodal chunks")
    text_chunks_count: int = Field(default=0, description="Number of prose text chunks")
    tables_count: int = Field(default=0, description="Number of table chunks extracted")
    images_count: int = Field(default=0, description="Number of image chunks captioned")
    ocr_pages: list[int] = Field(default_factory=list, description="List of page numbers processed via OCR")
