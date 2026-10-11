"""OCR fallback processing for scanned or image-only PDF pages."""

import io
from typing import Optional
from PIL import Image
import fitz as pymupdf

from rag_platform.exceptions import IngestionError
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


def is_scanned_page(page: pymupdf.Page, min_char_threshold: int = 40) -> bool:
    """Determine if a PDF page lacks digital text and likely requires OCR.

    Args:
        page: PyMuPDF page instance.
        min_char_threshold: Minimum character count to consider text present.

    Returns:
        True if page has fewer characters than threshold, False otherwise.
    """
    text = page.get_text().strip()
    return len(text) < min_char_threshold


def extract_page_ocr_text(page: pymupdf.Page, language: str = "eng") -> str:
    """Perform optical character recognition on a rasterized PDF page image.

    Renders the page to a pixmap and attempts OCR via pytesseract,
    with graceful fallback to EasyOCR or PyMuPDF OCR when available.

    Args:
        page: PyMuPDF Page instance to perform OCR on.
        language: Language code for OCR recognition (default 'eng').

    Returns:
        Recognized text from the page image.

    Raises:
        IngestionError: If OCR processing encounters an unrecoverable failure.
    """
    try:
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        logger.exception("Failed to rasterize PDF page for OCR")
        raise IngestionError(f"Page rasterization failed during OCR: {exc}") from exc

    # Try pytesseract first
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(pil_img, lang=language)
        if ocr_text.strip():
            logger.info("OCR successfully performed via pytesseract on page %s", page.number + 1)
            return ocr_text.strip()
    except (ImportError, Exception) as exc:
        logger.debug("Pytesseract not available or failed: %s", exc)

    # Fallback to easyocr if available
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(img_bytes, detail=0)
        if results:
            logger.info("OCR successfully performed via EasyOCR on page %s", page.number + 1)
            return "\n".join(results)
    except (ImportError, Exception) as exc:
        logger.debug("EasyOCR not available or failed: %s", exc)

    # Fallback: PyMuPDF built-in OCR if tessdata is installed
    try:
        ocr_page_text = page.get_text("text")
        if ocr_page_text.strip():
            return ocr_page_text.strip()
    except Exception:
        pass

    logger.warning("No OCR backend succeeded for page %s; returning empty OCR string", page.number + 1)
    return ""
