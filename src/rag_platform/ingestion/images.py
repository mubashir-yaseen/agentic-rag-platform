"""Image and figure extraction from PDFs with vision LLM captioning."""

import base64
import os
from typing import Optional
import fitz as pymupdf
import requests

from rag_platform.config import get_settings
from rag_platform.exceptions import IngestionError
from rag_platform.ingestion.models import ChunkMetadata, ContentType, ExtractedChunk
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


def generate_image_caption(
    image_bytes: bytes,
    context_hint: str = "",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate a descriptive caption for an extracted image using a vision-capable LLM.

    Args:
        image_bytes: Raw binary bytes of the image (PNG/JPEG).
        context_hint: Optional surrounding textual context from the PDF page.
        api_key: LLM API key. If unset, loads from application settings.
        base_url: Base endpoint URL.
        model: Model name to use for vision inference.

    Returns:
        Generated natural language caption describing the image/diagram.
    """
    settings = get_settings()
    key = api_key if api_key is not None else settings.effective_api_key
    url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
    vision_model = model or "openai/gpt-4o-mini"

    if not key:
        logger.warning("No API key available for vision captioning; using fallback caption")
        return f"Figure/Diagram extracted from document. Context: {context_hint[:120]}"

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this image or figure from a document in concise detail. "
                        "Highlight key data, chart trends, diagram labels, or visual components "
                        "so it is effectively searchable. "
                        f"Surrounding page context: {context_hint[:200]}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ],
        }
    ]

    try:
        resp = requests.post(
            f"{url}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"model": vision_model, "messages": messages, "max_tokens": 300},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            caption = data["choices"][0]["message"]["content"].strip()
            return caption
        logger.warning("Vision API returned status %s: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.warning("Vision LLM call failed: %s; using descriptive placeholder", exc)

    return f"Extracted figure/diagram from document. Context: {context_hint[:120]}"


def extract_images_from_page(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    source_doc: str,
    output_dir: str = "extracted_images",
    min_width: int = 100,
    min_height: int = 100,
    caption_llm: bool = True,
    start_chunk_idx: int = 0,
) -> list[ExtractedChunk]:
    """Extract embedded images from a PDF page, save them to disk, and caption them.

    Args:
        doc: PyMuPDF Document object.
        page: PyMuPDF Page instance.
        source_doc: Name of the PDF file.
        output_dir: Local filesystem directory to persist extracted image files.
        min_width: Minimum pixel width to filter out icons or decorative rules.
        min_height: Minimum pixel height to filter out icons.
        caption_llm: Whether to invoke vision LLM for caption generation.
        start_chunk_idx: Index offset for chunk ID assignment.

    Returns:
        List of ExtractedChunk objects with content_type=ContentType.IMAGE.

    Raises:
        IngestionError: If image extraction or file writing fails unexpectedly.
    """
    image_chunks: list[ExtractedChunk] = []
    page_num = page.number + 1

    try:
        os.makedirs(output_dir, exist_ok=True)
        image_list = page.get_images(full=True)
        if not image_list:
            return []

        page_text_snippet = page.get_text()[:300].strip()

        for idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]

            if width < min_width or height < min_height:
                continue  # Skip small decorative graphics/icons

            safe_source = os.path.splitext(os.path.basename(source_doc))[0]
            image_filename = f"{safe_source}_p{page_num}_img{idx}.{image_ext}"
            saved_image_path = os.path.join(output_dir, image_filename)

            with open(saved_image_path, "wb") as f:
                f.write(image_bytes)

            caption = (
                generate_image_caption(image_bytes, context_hint=page_text_snippet)
                if caption_llm
                else f"Figure on page {page_num}: {page_text_snippet[:100]}"
            )

            chunk_id = f"{source_doc}_p{page_num}_img{idx}_{start_chunk_idx + idx}"
            chunk = ExtractedChunk(
                content=f"### Image/Figure (Page {page_num})\nCaption: {caption}",
                metadata=ChunkMetadata(
                    source_doc=source_doc,
                    page_number=page_num,
                    content_type=ContentType.IMAGE,
                    chunk_id=chunk_id,
                    image_path=saved_image_path,
                    extra={
                        "image_index": idx,
                        "width": width,
                        "height": height,
                        "format": image_ext,
                    },
                ),
            )
            image_chunks.append(chunk)

        if image_chunks:
            logger.info(
                "Extracted and captioned %d images from page %d of %s",
                len(image_chunks),
                page_num,
                source_doc,
            )
        return image_chunks

    except Exception as exc:
        logger.exception("Failed to extract images from page %d of %s", page_num, source_doc)
        raise IngestionError(
            f"Image extraction failed on page {page_num}: {exc}",
            details={"page": page_num, "source_doc": source_doc},
        ) from exc
