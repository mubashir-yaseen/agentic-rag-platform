"""Upload and index company annual reports with the active documents + chunks pipeline."""

import argparse
import os
import uuid
from typing import Any, Optional

from supabase import create_client

from rag_platform.config import get_settings
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.vectorstore import SupabaseVectorStore, get_embedding_model


def build_company_document_metadata(
    company_id: str,
    symbol: str,
    name: str,
    year: Optional[int] = None,
    report_type: str = "annual_report",
    storage_path: Optional[str] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build document metadata for an indexed annual report without overwriting active schema fields."""
    metadata = dict(extra_metadata or {})
    metadata.update(
        {
            "company_id": str(company_id),
            "symbol": str(symbol).upper(),
            "name": str(name),
            "scope": "company",
            "report_type": report_type,
            "year": int(year) if year is not None else None,
            "storage_path": storage_path,
        }
    )
    return {key: value for key, value in metadata.items() if value is not None}


def upload_company_annual_report(
    pdf_path: str,
    company_id: str,
    company_symbol: str,
    company_name: str,
    year: Optional[int] = None,
    report_type: str = "annual_report",
    file_name: Optional[str] = None,
    storage_bucket: str = "annual-reports",
    supabase_client=None,
    vectorstore: Optional[SupabaseVectorStore] = None,
    embedding_model: Optional[Any] = None,
) -> dict[str, Any]:
    """Upload a PDF to annual-reports/<SYMBOL>/<FILE>.pdf and index it using the active documents/chunks pipeline."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    settings = get_settings()
    supabase_url = settings.SUPABASE_URL
    supabase_key = settings.effective_supabase_key
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or Supabase API key in configuration.")

    resolved_file_name = file_name or os.path.basename(pdf_path)
    symbol = str(company_symbol).upper()
    storage_rel_path = f"{symbol}/{resolved_file_name}"
    storage_path = f"{storage_bucket}/{symbol}/{resolved_file_name}"

    supabase = supabase_client or create_client(supabase_url, supabase_key)
    with open(pdf_path, "rb") as doc_file:
        content = doc_file.read()

    supabase.storage.from_(storage_bucket).upload(
        path=storage_rel_path,
        file=content,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )

    doc_id = str(uuid.uuid4())[:8] + "_" + resolved_file_name
    ingest_result = ingest_pdf(
        pdf_path=pdf_path,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        extract_tables=True,
        extract_images=True,
        allow_ocr=True,
        output_images_dir="data/extracted_images",
        caption_llm=False,
    )

    if ingest_result.chunks:
        embedding_fn = embedding_model or get_embedding_model()
        embeddings = embedding_fn.embed_documents([chunk.content for chunk in ingest_result.chunks])
        if len(embeddings) != len(ingest_result.chunks):
            raise ValueError(
                f"Embedding mismatch for '{resolved_file_name}': generated {len(embeddings)} embeddings for {len(ingest_result.chunks)} chunks"
            )
        for idx, embedding in enumerate(embeddings):
            ingest_result.chunks[idx].embedding = embedding

    active_vectorstore = vectorstore or SupabaseVectorStore()
    metadata = build_company_document_metadata(
        company_id=company_id,
        symbol=symbol,
        name=company_name,
        year=year,
        report_type=report_type,
        storage_path=storage_path,
    )
    active_vectorstore.upsert_document(
        doc_id=doc_id,
        filename=resolved_file_name,
        doc_type="pdf",
        metadata=metadata,
    )
    if ingest_result.chunks:
        active_vectorstore.upsert_chunks(chunks=ingest_result.chunks, doc_id=doc_id)

    return {
        "doc_id": doc_id,
        "filename": resolved_file_name,
        "storage_path": storage_path,
        "total_pages": ingest_result.total_pages,
        "total_chunks": len(ingest_result.chunks),
        "tables_count": ingest_result.tables_count,
        "images_count": ingest_result.images_count,
        "status": "completed",
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload and index a company annual report PDF using the active vectorstore pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--company-id", required=True, help="Company identifier to store in documents.metadata")
    parser.add_argument("--company-symbol", required=True, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--company-name", required=True, help="Company legal name")
    parser.add_argument("--year", default=None, help="Fiscal year")
    parser.add_argument("--report-type", default="annual_report", help="Report classification")
    parser.add_argument("--file-name", default=None, help="Override destination file name within the annual-reports bucket")
    parser.add_argument("--storage-bucket", default="annual-reports", help="Supabase storage bucket")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    result = upload_company_annual_report(
        pdf_path=args.pdf,
        company_id=args.company_id,
        company_symbol=args.company_symbol,
        company_name=args.company_name,
        year=int(args.year) if args.year is not None else None,
        report_type=args.report_type,
        file_name=args.file_name,
        storage_bucket=args.storage_bucket,
    )

    print("UPLOAD OK")
    print("doc_id:", result["doc_id"])
    print("storage_path:", result["storage_path"])
    print("metadata:", result["metadata"])


if __name__ == "__main__":
    main()
