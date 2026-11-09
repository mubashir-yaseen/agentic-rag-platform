"""Remove characters Postgres cannot store (NUL, lone surrogates) from extracted PDF text."""

from __future__ import annotations

from typing import Any


def scrub_text(text: str) -> str:
    if not text:
        return text
    return text.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")


def scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {(scrub_text(k) if isinstance(k, str) else k): scrub_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_value(v) for v in value]
    return value


def scrub_chunks(chunks: list[Any]) -> int:
    """Clean chunk content and metadata in place; return how many chunks' content changed."""
    changed = 0
    for chunk in chunks:
        cleaned = scrub_text(chunk.content)
        if cleaned != chunk.content:
            changed += 1
            chunk.content = cleaned
        meta = getattr(chunk, "metadata", None)
        if meta is None:
            continue
        meta.extra = scrub_value(meta.extra)
        for field in ("source_doc", "chunk_id", "image_path"):
            val = getattr(meta, field, None)
            if isinstance(val, str):
                setattr(meta, field, scrub_text(val))
    return changed
