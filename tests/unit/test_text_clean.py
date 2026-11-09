from rag_platform.ingestion.models import ChunkMetadata, ExtractedChunk
from rag_platform.ingestion.text_clean import scrub_chunks, scrub_text


def _chunk(text, extra=None):
    return ExtractedChunk(content=text, metadata=ChunkMetadata(source_doc="a\x00.pdf", page_number=1, chunk_id="c1", extra=extra or {}))


def test_scrub_text_removes_nul():
    assert scrub_text("ab\x00cd") == "abcd"
    assert scrub_text("") == ""


def test_scrub_chunks_counts_and_cleans_metadata():
    chunks = [_chunk("x\x00y", {"k": ["a\x00b"]}), _chunk("clean")]
    assert scrub_chunks(chunks) == 1
    assert chunks[0].content == "xy"
    assert chunks[0].metadata.extra == {"k": ["ab"]}
    assert chunks[0].metadata.source_doc == "a.pdf"
