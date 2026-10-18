"""Upload and index company annual reports (run by the site owner on their own computer).

What it does, in order:
  1. checks the company exists in the Supabase ``companies`` table,
  2. stores the PDF in the private Storage bucket ``annual-reports`` as <SYMBOL>/<file>.pdf,
  3. extracts text / tables / images, embeds the chunks (Google Gemini embeddings) and
     saves them in ``documents`` + ``chunks`` tagged with the company, so Research mode finds them.

Running it again for the same company and file replaces the old copy (no duplicates).
Visitors cannot do any of this: the public /ingest endpoint only creates personal documents.

Setup (once, in the project folder, with a .env file next to this script):
    SUPABASE_URL=https://<project>.supabase.co
    SUPABASE_SERVICE_ROLE_KEY=<service_role key>
    GOOGLE_API_KEY=<key from https://aistudio.google.com/app/apikey>
    pip install -e .

Example (PowerShell):
    python upload_company_report.py --pdf "D:\\reports\\Apple_2024.pdf" --company-id <id from companies table> --year 2024
"""

import argparse
import os
import re
import uuid
from typing import Any, Optional

from supabase import create_client

from rag_platform.config import get_settings
from rag_platform.ingestion.pipeline import ingest_pdf
from rag_platform.vectorstore import SupabaseVectorStore, get_embedding_model

UPSERT_BATCH_SIZE = 100  # chunks per database request (a whole annual report in one request can time out)


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


def _safe_storage_name(name: str) -> str:
    """Storage keys reject many characters (spaces, brackets, accents...); keep only safe ones."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "report.pdf"


def _find_company(supabase: Any, company_id: str) -> Optional[dict[str, Any]]:
    """Look the company up in the companies table (the same table the Research dropdown reads)."""
    try:
        rows = supabase.table("companies").select("id, symbol, name").eq("id", company_id).limit(1).execute().data
    except Exception:
        return None
    return rows[0] if isinstance(rows, list) and rows else None


def _ensure_bucket(supabase: Any, bucket: str) -> None:
    """Create the private Storage bucket if it does not exist yet."""
    try:
        supabase.storage.get_bucket(bucket)
        return
    except Exception:
        pass
    try:
        supabase.storage.create_bucket(bucket, options={"public": False})
    except Exception as exc:
        # "already exists" is fine; anything else surfaces when the upload below fails with a clear message.
        if "exist" not in str(exc).lower():
            print(f"Note: could not create bucket '{bucket}' automatically ({exc}). Create it in the Supabase dashboard.")


def _remove_previous_copies(vectorstore: Any, company_id: str, file_name: str) -> int:
    """Delete earlier indexed copies of the same report so a re-run does not create duplicates."""
    try:
        rows = (
            vectorstore.client.table("documents")
            .select("doc_id")
            .eq("filename", file_name)
            .filter("metadata->>company_id", "eq", str(company_id))
            .execute()
            .data
        )
    except Exception:
        return 0
    if not isinstance(rows, list):
        return 0
    removed = 0
    for row in rows:
        try:
            vectorstore.delete_document(row["doc_id"])
            removed += 1
        except Exception as exc:
            print(f"Warning: could not remove old copy {row.get('doc_id')}: {exc}")
    return removed


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
        raise ValueError("Missing SUPABASE_URL or Supabase API key in configuration (.env).")

    resolved_file_name = file_name or os.path.basename(pdf_path)
    symbol = str(company_symbol).upper()
    storage_name = _safe_storage_name(resolved_file_name)
    storage_rel_path = f"{symbol}/{storage_name}"
    storage_path = f"{storage_bucket}/{symbol}/{storage_name}"

    supabase = supabase_client or create_client(supabase_url, supabase_key)
    with open(pdf_path, "rb") as doc_file:
        content = doc_file.read()

    _ensure_bucket(supabase, storage_bucket)
    try:
        supabase.storage.from_(storage_bucket).upload(
            path=storage_rel_path,
            file=content,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not upload to the Storage bucket '{storage_bucket}': {exc}. "
            "Check that the bucket exists (Supabase > Storage) and that SUPABASE_SERVICE_ROLE_KEY is the service_role key."
        ) from exc

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
    if not ingest_result.chunks:
        raise ValueError(f"No text could be extracted from '{resolved_file_name}' (scanned PDF without OCR?).")

    embedding_fn = embedding_model or get_embedding_model()
    embeddings = embedding_fn.embed_documents([chunk.content for chunk in ingest_result.chunks])
    if len(embeddings) != len(ingest_result.chunks):
        raise ValueError(
            f"Embedding mismatch for '{resolved_file_name}': generated {len(embeddings)} embeddings for {len(ingest_result.chunks)} chunks"
        )
    for idx, embedding in enumerate(embeddings):
        ingest_result.chunks[idx].embedding = embedding

    active_vectorstore = vectorstore or SupabaseVectorStore()
    replaced = _remove_previous_copies(active_vectorstore, str(company_id), resolved_file_name)

    metadata = build_company_document_metadata(
        company_id=company_id,
        symbol=symbol,
        name=company_name,
        year=year,
        report_type=report_type,
        storage_path=storage_path,
        extra_metadata={
            "total_pages": ingest_result.total_pages,
            "tables": ingest_result.tables_count,
            "images": ingest_result.images_count,
        },
    )
    active_vectorstore.upsert_document(
        doc_id=doc_id,
        filename=resolved_file_name,
        doc_type="pdf",
        metadata=metadata,
    )
    try:
        for start in range(0, len(ingest_result.chunks), UPSERT_BATCH_SIZE):
            active_vectorstore.upsert_chunks(
                chunks=ingest_result.chunks[start : start + UPSERT_BATCH_SIZE],
                doc_id=doc_id,
            )
    except Exception:
        # Never leave a listed report without searchable content.
        try:
            active_vectorstore.delete_document(doc_id)
        except Exception:
            pass
        raise

    return {
        "doc_id": doc_id,
        "filename": resolved_file_name,
        "storage_path": storage_path,
        "total_pages": ingest_result.total_pages,
        "total_chunks": len(ingest_result.chunks),
        "tables_count": ingest_result.tables_count,
        "images_count": ingest_result.images_count,
        "replaced_previous": replaced,
        "status": "completed",
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload and index a company annual report PDF for Research mode.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--company-id", required=True, help="The id of the company in the Supabase 'companies' table")
    parser.add_argument("--company-symbol", default=None, help="Ticker (default: taken from the companies table)")
    parser.add_argument("--company-name", default=None, help="Company name (default: taken from the companies table)")
    parser.add_argument("--year", default=None, help="Fiscal year, e.g. 2024")
    parser.add_argument("--report-type", default="annual_report", help="Report classification")
    parser.add_argument("--file-name", default=None, help="Override the stored file name")
    parser.add_argument("--storage-bucket", default="annual-reports", help="Supabase Storage bucket")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    settings = get_settings()
    if not settings.SUPABASE_URL or not settings.effective_supabase_key:
        raise SystemExit("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY. Put them in the .env file next to this script.")

    # The company must exist in the table the Research dropdown reads, with exactly this id.
    client = create_client(settings.SUPABASE_URL, settings.effective_supabase_key)
    company = _find_company(client, str(args.company_id))
    if company is None:
        available = []
        try:
            available = client.table("companies").select("id, symbol, name").order("name").execute().data or []
        except Exception:
            pass
        listing = "\n".join(f"  id={c.get('id')}  {c.get('symbol')}  {c.get('name')}" for c in available) or "  (table is empty or unreadable)"
        raise SystemExit(f"No company with id '{args.company_id}' in the companies table. Available companies:\n{listing}")

    result = upload_company_annual_report(
        pdf_path=args.pdf,
        company_id=str(company["id"]),
        company_symbol=args.company_symbol or company.get("symbol") or "UNKNOWN",
        company_name=args.company_name or company.get("name") or "Unknown",
        year=int(args.year) if args.year is not None else None,
        report_type=args.report_type,
        file_name=args.file_name,
        storage_bucket=args.storage_bucket,
        supabase_client=client,
    )

    print("UPLOAD OK")
    print("doc_id:        ", result["doc_id"])
    print("storage_path:  ", result["storage_path"])
    print("pages/chunks:  ", result["total_pages"], "/", result["total_chunks"])
    if result["replaced_previous"]:
        print("replaced older copies:", result["replaced_previous"])
    print("company:       ", result["metadata"].get("name"), f"(id {result['metadata'].get('company_id')})")


if __name__ == "__main__":
    main()
