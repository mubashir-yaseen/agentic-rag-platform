"""Core RAG System implementation for document processing, embedding, and retrieval."""

import os
from typing import Any, Optional
import requests

try:
    import torch
    torch.classes.__path__ = []  # Prevent PyTorch directory watcher issue on Windows
except (ImportError, Exception):
    torch = None  # Graceful fallback if torch native C++ runtime is unavailable

import fitz as pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

from rag_platform.config import get_settings
from rag_platform.exceptions import ConfigurationError, IngestionError, RetrievalError
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """Orchestrates document extraction, vector indexing, and QA generation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.llm_model = llm_model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.RAG_TEMPERATURE
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")

        # Initialize HuggingFace embeddings
        try:
            from rag_platform.vectorstore.embeddings import get_embedding_model

            self.embeddings = get_embedding_model(self.model_name)
        except Exception as exc:
            logger.warning("HuggingFaceEmbeddings initialization warning: %s", exc)
            self.embeddings = None

        self.api_key = api_key or settings.effective_api_key
        if not self.api_key:
            raise ConfigurationError(
                "Missing LLM API key. Provide OPENROUTER_API_KEY or OPENAI_API_KEY via config or environment."
            )

        self.supabase_url = (supabase_url or settings.SUPABASE_URL).rstrip("/")
        self.supabase_key = supabase_key or settings.effective_supabase_key

        self.supabase: Optional[Client] = None
        self.supabase_ok = False
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
                self.supabase_ok = True
                logger.info("Supabase client initialized successfully.")
            except Exception as exc:
                logger.warning("Supabase client initialization failed: %s", exc)
                self.supabase_ok = False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path: Local filesystem path to the PDF document.

        Returns:
            Extracted text content across all pages.

        Raises:
            IngestionError: If the PDF file cannot be opened or parsed.
        """
        if not os.path.exists(pdf_path):
            raise IngestionError(f"File not found: {pdf_path}")

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as exc:
            logger.exception("Failed to open PDF at %s", pdf_path)
            raise IngestionError(f"Failed to open PDF: {exc}") from exc

        try:
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(f"\n--- Page {page_num + 1} ---\n")
                text_parts.append(page.get_text())
            return "".join(text_parts)
        except Exception as exc:
            logger.exception("Failed to extract text from PDF")
            raise IngestionError(f"Text extraction failed: {exc}") from exc
        finally:
            doc.close()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[Document]:
        """Split raw text into structured Document chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata={"chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

    def chunk_and_embed_document(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Split text, generate vector embeddings, and return structured chunk dictionaries."""
        if not text:
            return []

        docs = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = [d.page_content for d in docs]

        embeddings = [None] * len(texts)
        if self.embeddings is not None:
            try:
                embeddings = self.embeddings.embed_documents(texts)
            except Exception as exc:
                logger.exception("Vector embedding generation failed: %s", exc)
                embeddings = [None] * len(texts)

        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "index": i,
                "content": doc.page_content,
                "embedding": embeddings[i] if i < len(embeddings) else None,
                "metadata": metadata or {},
            })
        return chunks

    def _openrouter_chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Call OpenRouter/OpenAI chat completion endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Chat Platform",
        }
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _openrouter_web_search(
        self,
        question: str,
        max_results: int = 5,
        model: Optional[str] = None,
    ) -> str:
        """Execute a web-augmented chat completion query."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with web search access. "
                    "Use the web search results to answer accurately and cite sources with markdown links when possible."
                ),
            },
            {"role": "user", "content": question},
        ]
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Chat Platform",
        }
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": 0.2,
            "plugins": [{"id": "web", "max_results": max_results}],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=150)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def query_general_question(self, question: str) -> dict[str, Any]:
        """Answer a general-knowledge question using the LLM directly."""
        prompt = (
            "You are a helpful general-purpose assistant.\n\n"
            "Answer the user's question clearly and directly using your own knowledge.\n"
            "If the question is ambiguous, explain the most likely interpretation.\n"
            "If you are uncertain, say so briefly.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        try:
            answer = self._openrouter_chat([{"role": "user", "content": prompt}])
            return {"answer": answer, "source_documents": []}
        except Exception as exc:
            logger.exception("General QA generation failed")
            return {"answer": f"Failed to generate answer: {exc}", "source_documents": []}

    def query_web_search(self, question: str) -> dict[str, Any]:
        """Perform a web-search query for real-time information."""
        prompt = (
            f"Answer the following using current web information only when needed.\n\n"
            f"Question: {question}\n\n"
            f"Return a clear answer with concise supporting details."
        )
        try:
            answer = self._openrouter_web_search(prompt, max_results=5, model=self.llm_model)
            return {"answer": answer, "source_documents": []}
        except Exception as exc:
            logger.exception("Web search failed")
            return {"answer": f"Web search failed: {exc}", "source_documents": []}

    def query_financial_analysis(self, question: str) -> dict[str, Any]:
        """Perform domain-specific financial market analysis."""
        prompt = (
            "You are a finance and stock market analyst.\n\n"
            "Use current market/news context when needed.\n"
            "Explain:\n"
            "1. what the news means,\n"
            "2. likely impact on the broader market,\n"
            "3. likely impact on relevant sectors/stocks,\n"
            "4. whether the effect is bullish, bearish, or mixed,\n"
            "5. any uncertainty.\n\n"
            f"Question:\n{question}\n\n"
            "Answer in a structured, investor-friendly way."
        )
        try:
            answer = self._openrouter_web_search(prompt, max_results=5, model=self.llm_model)
            return {"answer": answer, "source_documents": []}
        except Exception as exc:
            logger.exception("Financial analysis failed")
            return {"answer": f"Financial analysis failed: {exc}", "source_documents": []}

    def _build_answer(self, context: str, question: str) -> str:
        """Prompt the LLM to generate an answer grounded strictly in the provided context."""
        prompt = (
            "You are a helpful assistant that answers questions ONLY based on the provided context.\n\n"
            "IMPORTANT RULES:\n"
            "1. ONLY answer using the provided context.\n"
            "2. If the answer cannot be found in the context, say: "
            '"I cannot answer this question based on the provided documents."\n'
            "3. Do NOT use external knowledge.\n"
            "4. If the question is unclear, ask for clarification.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        try:
            return self._openrouter_chat([{"role": "user", "content": prompt}])
        except Exception:
            logger.exception("Contextual answer generation failed")
            return "Failed to generate answer."

    def get_company_by_symbol_or_name(self, identifier: str) -> Optional[dict[str, Any]]:
        """Lookup company record from Supabase by ticker symbol or name."""
        if not self.supabase:
            return None
        try:
            if not identifier:
                return None
            ident = identifier.strip()
            symbol = ident.upper()

            resp = self.supabase.table("companies").select("*").eq("symbol", symbol).limit(1).execute()
            if resp.data:
                return resp.data[0]

            resp = self.supabase.table("companies").select("*").ilike("name", f"%{ident}%").limit(1).execute()
            if resp.data:
                return resp.data[0]

            return None
        except Exception:
            logger.exception("Failed to lookup company: %s", identifier)
            return None

    def upload_pdf_to_storage(self, bucket_name: str, file_path: str, storage_path: str) -> bool:
        """Upload a PDF file to Supabase storage bucket."""
        if not self.supabase:
            return False
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=content,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )
            logger.info("Uploaded to storage: %s", storage_path)
            return True
        except Exception:
            logger.exception("Failed to upload PDF to Supabase storage")
            return False

    def insert_document_record(
        self,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
        scope: str = "user",
        year: Optional[int] = None,
        report_type: str = "annual_report",
        file_name: str = "",
        storage_path: str = "",
        source_url: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a document record into Supabase documents table."""
        if not self.supabase:
            raise RetrievalError("Supabase client is not configured")
        try:
            data = {
                "user_id": user_id,
                "company_id": company_id,
                "scope": scope,
                "year": year,
                "report_type": report_type,
                "file_name": file_name,
                "storage_path": storage_path,
                "source_url": source_url,
            }
            resp = self.supabase.table("documents").insert(data).execute()
            if getattr(resp, "data", None):
                return resp.data[0].get("id")
            return None
        except Exception as exc:
            logger.exception("Failed to insert document record: %s", exc)
            raise RetrievalError(f"Insert document record failed: {exc}") from exc

    def insert_chunks_record(
        self,
        document_id: int,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
        scope: str = "user",
        chunks_data: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Bulk insert chunk records into Supabase document_chunks table."""
        if not chunks_data or not self.supabase:
            return 0
        try:
            payload = []
            for chunk in chunks_data:
                payload.append({
                    "document_id": document_id,
                    "user_id": user_id,
                    "company_id": company_id,
                    "scope": scope,
                    "chunk_index": int(chunk.get("index", 0)),
                    "content": chunk.get("content"),
                    "metadata": chunk.get("metadata", {}),
                    "embedding": chunk.get("embedding"),
                })
            resp = self.supabase.table("document_chunks").insert(payload).execute()
            if getattr(resp, "data", None):
                return len(resp.data)
            return 0
        except Exception as exc:
            logger.exception("Failed to insert chunk records: %s", exc)
            raise RetrievalError(f"Insert chunk records failed: {exc}") from exc

    def query_user_document(
        self,
        document_id: int,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve context chunks from a user document and generate answer."""
        if not self.supabase:
            return {"answer": "Supabase client not configured.", "source_documents": []}

        if not self.embeddings:
            return {"answer": "Embeddings model not initialized.", "source_documents": []}

        try:
            question_emb = self.embeddings.embed_query(question)
        except Exception:
            return {"answer": "Embedding failed.", "source_documents": []}

        try:
            response = self.supabase.rpc("similar_chunks", {
                "query_embedding": question_emb,
                "document_id_filter": document_id,
                "company_id_filter": None,
                "scope_filter": "user",
                "match_count": top_k,
            }).execute()
            chunks = response.data or []
        except Exception:
            logger.exception("RPC similar_chunks failed")
            return {"answer": "Search failed (RPC error).", "source_documents": []}

        if not chunks:
            return {"answer": "No relevant information found in the document.", "source_documents": []}

        context = "\n\n".join([chunk["content"] for chunk in chunks])
        answer = self._build_answer(context, question)

        source_docs = [
            Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]
        return {"answer": answer, "source_documents": source_docs}

    def query_company_documents(
        self,
        company_id: int,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve context chunks from company documents and generate answer."""
        if not self.supabase:
            return {"answer": "Supabase client not configured.", "source_documents": []}

        if not self.embeddings:
            return {"answer": "Embeddings model not initialized.", "source_documents": []}

        try:
            question_emb = self.embeddings.embed_query(question)
        except Exception:
            return {"answer": "Embedding failed.", "source_documents": []}

        try:
            response = self.supabase.rpc("similar_chunks", {
                "query_embedding": question_emb,
                "document_id_filter": None,
                "company_id_filter": company_id,
                "scope_filter": "company",
                "match_count": top_k,
            }).execute()
            chunks = response.data or []
        except Exception:
            logger.exception("RPC similar_chunks failed")
            return {"answer": "Search failed (RPC error).", "source_documents": []}

        if not chunks:
            return {"answer": "No relevant information found in the company documents.", "source_documents": []}

        context = "\n\n".join([chunk["content"] for chunk in chunks])
        answer = self._build_answer(context, question)

        source_docs = [
            Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]
        return {"answer": answer, "source_documents": source_docs}

    def get_documents_for_company(self, company_id: int) -> list[dict[str, Any]]:
        """Fetch all documents for a specific company."""
        if not self.supabase:
            return []
        try:
            resp = (
                self.supabase.table("documents")
                .select("*")
                .eq("company_id", company_id)
                .eq("scope", "company")
                .order("year", desc=True)
                .execute()
            )
            return resp.data or []
        except Exception:
            logger.exception("Failed to get documents for company")
            return []

    def get_documents_for_user(self, user_id: int) -> list[dict[str, Any]]:
        """Fetch all documents for a specific user."""
        if not self.supabase:
            return []
        try:
            resp = (
                self.supabase.table("documents")
                .select("*")
                .eq("user_id", user_id)
                .eq("scope", "user")
                .order("id", desc=True)
                .execute()
            )
            return resp.data or []
        except Exception:
            logger.exception("Failed to get documents for user")
            return []
