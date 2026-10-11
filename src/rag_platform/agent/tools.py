"""Tools for the LangGraph agent layer (retrieval, calculator, web search)."""

import ast
import operator
import re
import time
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field

from rag_platform.config import get_settings
from rag_platform.exceptions import AgentError
from rag_platform.logging_config import get_logger
from rag_platform.vectorstore.supabase_store import SearchFilter, SearchResult, SupabaseVectorStore

logger = get_logger(__name__)


class ToolResult(BaseModel):
    """Result schema returned by agent tools."""

    tool_name: str = Field(..., description="Name of the invoked tool")
    output: str = Field(..., description="Stringified result of the tool execution")
    data: Optional[dict[str, Any]] = Field(default=None, description="Structured raw data if applicable")
    success: bool = Field(default=True, description="Whether the tool succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# Safe math operators for calculator AST evaluation
_ALLOWED_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# Production retrieval window used by run_agent / FastAPI chat path.
DEFAULT_RETRIEVAL_K = 6


def _is_table_of_contents_chunk(text: str) -> bool:
    """Detect if chunk consists predominantly of Table of Contents dotted leaders."""
    dot_sequences = len(re.findall(r"\.{4,}\s*\d+", text))
    if dot_sequences >= 2:
        return True
    if text.count("...") > 4 and any(d in text for d in ["1.", "2.", "3.", "4.", "5."]):
        return True
    # Outline-only headings with leader dots but little prose
    if dot_sequences >= 1 and len(re.findall(r"[.!?]\s+[A-Z]", text)) == 0 and len(text) < 500:
        return True
    return False


def _extract_query_keywords(query: str) -> list[str]:
    """Extract content keywords from a user query (stopwords removed)."""
    q_words = re.findall(r"\w+", query.lower())
    stopwords = {
        "what", "is", "the", "for", "in", "of", "and", "a", "an", "to", "are",
        "mentioned", "does", "this", "on", "with", "from", "how", "why", "which",
        "when", "where", "who", "be", "by", "as", "at", "or",
    }
    keywords = [w for w in q_words if w not in stopwords and len(w) > 1]
    return keywords or q_words


def _canonicalize_lexical_term(term: str) -> str:
    """Normalize simple inflections without globally equating unrelated finance terms."""
    token = re.sub(r"[^a-z0-9]+", "", term.lower())
    if not token:
        return token

    if token in {"accounts", "account"}:
        return "account"
    if token in {"deposits", "deposit"}:
        return "deposit"
    if token in {"saving", "savings"}:
        return "saving"
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _phrase_equivalent_match(query: str, text: str) -> float:
    """Allow phrase-level financial terminology equivalence without global account/deposit aliasing."""
    import math

    text_lower = text.lower()
    query_lower = query.lower()

    equivalents: list[tuple[str, str]] = [
        ("current account", "current deposits"),
        ("saving account", "savings deposits"),
    ]

    total = 0.0
    for q_phrase, t_phrase in equivalents:
        if q_phrase in query_lower and t_phrase in text_lower:
            total += 3.5 + math.log(2)

    # Keep the direct-match path intact for the same phrase.
    for phrase in ("current account", "saving account"):
        if phrase in query_lower and phrase in text_lower:
            total += 2.5

    return total


def _compute_keyword_relevance(query: str, text: str) -> float:
    """Compute keyword, bigram, and exact phrase overlap score for hybrid retrieval.

    Important: repeated generic query terms (such as the same "account" appearing twice in a
    query) should not be treated as multiple independent high-value keyword hits. Distinct
    conceptual terms should score, but repeated generic terms should be capped.
    """
    import math

    keywords = _extract_query_keywords(query)
    canonical_keywords = [_canonicalize_lexical_term(kw) for kw in keywords]
    unique_keywords: list[str] = []
    seen_terms: set[str] = set()
    for kw in canonical_keywords:
        if not kw or kw in seen_terms:
            continue
        unique_keywords.append(kw)
        seen_terms.add(kw)

    canonical_text_tokens = [_canonicalize_lexical_term(tok) for tok in re.findall(r"[a-z0-9]+", text.lower())]
    text_lower = " ".join(canonical_text_tokens)
    score = 0.0
    generic_terms = {"account", "bank", "current", "saving", "deposit", "percentage", "2025"}

    phrase_bonus = _phrase_equivalent_match(query, text)
    matched_phrase_terms: set[str] = set()
    if phrase_bonus > 0:
        for phrase in ("current account", "saving account"):
            if phrase in query.lower() and phrase in text.lower():
                matched_phrase_terms.add(phrase)
        for q_phrase, t_phrase in [
            ("current account", "current deposits"),
            ("saving account", "savings deposits"),
        ]:
            if q_phrase in query.lower() and t_phrase in text.lower():
                matched_phrase_terms.add(q_phrase)

    for kw in unique_keywords:
        if not kw:
            continue
        count = text_lower.split().count(kw)
        if count <= 0:
            continue

        if kw in generic_terms:
            # Generic tokens should contribute a small capped amount; repeated occurrences of the
            # same generic term must not overwhelm a genuine phrase/concept match.
            # If the explicit phrase-equivalence logic already matched the concept, avoid letting the
            # same generic token dominate the score by double-counting the same concept.
            if kw in {"account", "current", "saving", "deposit"} and any(
                phrase.startswith(kw) or phrase.endswith(kw) for phrase in matched_phrase_terms
            ):
                score += 0.0
            else:
                score += min(0.5 + 0.35 * math.log(min(count, 2)), 1.25)
        else:
            score += 1.0 + math.log(min(count, 4))

    for i in range(len(unique_keywords) - 1):
        if not unique_keywords[i] or not unique_keywords[i + 1]:
            continue
        bigram = f"{unique_keywords[i]} {unique_keywords[i + 1]}"
        if bigram in text_lower:
            score += 4.0 if i == len(unique_keywords) - 2 else 2.0

    if phrase_bonus > 0:
        score += phrase_bonus

    clean_q = " ".join(unique_keywords)
    if clean_q and clean_q in text_lower:
        if all(term in generic_terms for term in clean_q.split() if term):
            score += 1.0
        else:
            score += 3.5

    if {"research", "problem"}.issubset(set(unique_keywords)) or "problem" in unique_keywords:
        for alias in ("problem statement", "research problem", "specific research problem"):
            if alias in text_lower:
                score += 3.0
                break

    return score


def _prose_quality_bonus(text: str, content_type: Optional[str] = None) -> float:
    """Prefer substantive body prose / tables over short captions and outlines."""
    bonus = 0.0
    sentences = len(re.findall(r"[.!?]\s+[A-Z]", text))
    words = len(text.split())
    if words >= 80 and sentences >= 2:
        bonus += 1.2
    elif words >= 40 and sentences >= 1:
        bonus += 0.5
    ctype = (content_type or "").lower()
    if ctype == "image":
        bonus -= 0.8
    elif ctype == "table":
        bonus += 0.4
    return bonus


def _fetch_lexical_chunk_pool(
    vectorstore: SupabaseVectorStore,
    filters: Optional[dict[str, Any] | SearchFilter],
    query: Optional[str] = None,
    page_size: int = 1000,
    max_rows: int = 5000,
) -> list[dict[str, Any]]:
    """Paginate chunks table, with a recall-safe keyword OR prefilter and a full-pool fallback."""
    if not hasattr(vectorstore, "client") or not vectorstore.client:
        return []

    start = time.perf_counter()
    try:
        filter_dict: dict[str, Any] = {}
        if isinstance(filters, SearchFilter):
            filter_dict = filters.model_dump(exclude_none=True)
        elif isinstance(filters, dict):
            filter_dict = {k_: v_ for k_, v_ in filters.items() if v_ is not None}

        def _run_full_scan() -> list[dict[str, Any]]:
            all_rows: list[dict[str, Any]] = []
            offset = 0
            while offset < max_rows:
                query_builder = vectorstore.client.table("chunks").select(
                    "chunk_id, doc_id, content, metadata"
                )
                if filter_dict.get("doc_id"):
                    query_builder = query_builder.eq("doc_id", filter_dict["doc_id"])
                end = min(offset + page_size - 1, max_rows - 1)
                resp = query_builder.range(offset, end).execute()
                batch = resp.data if isinstance(getattr(resp, "data", None), list) else []
                all_rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
            return all_rows

        keywords = _extract_query_keywords(query or "")
        if not keywords:
            return _run_full_scan()

        filtered_rows: list[dict[str, Any]] = []
        offset = 0
        while offset < max_rows:
            query_builder = vectorstore.client.table("chunks").select(
                "chunk_id, doc_id, content, metadata"
            )
            if filter_dict.get("doc_id"):
                query_builder = query_builder.eq("doc_id", filter_dict["doc_id"])
            # Recall-oriented candidate reducer: a chunk is eligible if it matches any meaningful keyword.
            or_clause = ",".join(f"content.ilike.%{kw}%" for kw in keywords)
            end = min(offset + page_size - 1, max_rows - 1)
            resp = query_builder.or_(or_clause).range(offset, end).execute()
            batch = resp.data if isinstance(getattr(resp, "data", None), list) else []
            filtered_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        if filtered_rows:
            return filtered_rows

        logger.info("Lexical prefilter produced zero rows; falling back to full lexical pool for recall safety.")
        return _run_full_scan()
    finally:
        logger.info("PERF: lexical_retrieval=%.2fs", time.perf_counter() - start)


def _normalize_filters(filters: Optional[dict[str, Any] | SearchFilter]) -> dict[str, Any]:
    """Return a plain dict from SearchFilter or mapping without mutating the caller input."""
    if isinstance(filters, SearchFilter):
        return filters.model_dump(exclude_none=True)
    if isinstance(filters, dict):
        return {key: value for key, value in filters.items() if value is not None}
    return {}


def _get_company_document_ids(vectorstore: SupabaseVectorStore, company_id: str) -> list[str]:
    """Resolve a company_id to the active indexed document IDs for company-scoped annual reports."""
    if not hasattr(vectorstore, "client") or not vectorstore.client:
        return []

    start = time.perf_counter()
    try:
        response = (
            vectorstore.client.table("documents")
            .select("doc_id, metadata")
            .filter("metadata->>company_id", "eq", company_id)
            .execute()
        )
        rows = response.data if isinstance(getattr(response, "data", None), list) else []
    except Exception as exc:
        logger.exception("Company document lookup failed for company_id=%s: %s", company_id, exc)
        raise RuntimeError(f"Failed to resolve indexed documents for company_id '{company_id}': {exc}") from exc
    finally:
        logger.info("PERF: company_document_lookup=%.2fs", time.perf_counter() - start)

    company_doc_ids: list[str] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if metadata.get("company_id") != company_id:
            continue
        if "scope" in metadata and metadata.get("scope") != "company":
            continue
        if "report_type" in metadata and metadata.get("report_type") != "annual_report":
            continue
        if "scope" not in metadata or "report_type" not in metadata:
            continue
        doc_id = row.get("doc_id")
        if doc_id:
            company_doc_ids.append(doc_id)
    return company_doc_ids


def _chunk_split_index(chunk_id: Optional[str], metadata: Optional[dict[str, Any]] = None) -> Optional[int]:
    """Best-effort extraction of the chunk order within a page using existing metadata or chunk IDs."""
    if isinstance(metadata, dict):
        extra = metadata.get("extra") or {}
        if isinstance(extra, dict):
            split_index = extra.get("split_index")
            if split_index is not None:
                try:
                    return int(split_index)
                except (TypeError, ValueError):
                    pass

    if not chunk_id:
        return None

    match = re.search(r"(?:txt|tbl|img)(\d+)", chunk_id, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"_(\d+)(?:_|$)", chunk_id)
    if not match:
        return None

    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _expand_adjacent_context_chunks(
    vectorstore: SupabaseVectorStore,
    search_results: list[SearchResult],
    max_neighbors: int = 2,
) -> list[SearchResult]:
    """Expand a retrieved chunk with immediately adjacent same-document/page chunks without altering ranking order."""
    if not search_results or not hasattr(vectorstore, "client") or not vectorstore.client:
        return search_results

    expanded: list[SearchResult] = []
    seen_chunk_ids: set[str] = set()

    def _build_result(raw_row: dict[str, Any], fallback_similarity: float) -> SearchResult:
        metadata = raw_row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = raw_row.get("chunk_id") or ""
        return SearchResult(
            chunk_id=chunk_id,
            doc_id=raw_row.get("doc_id") or "",
            content=raw_row.get("content") or "",
            metadata=metadata,
            similarity=float(raw_row.get("similarity", fallback_similarity)),
            filename=fallback_similarity and (raw_row.get("filename") or None),
            uploaded_at=raw_row.get("uploaded_at"),
        )

    for result in search_results:
        if result.chunk_id not in seen_chunk_ids:
            expanded.append(result)
            seen_chunk_ids.add(result.chunk_id)

        page_number = (result.metadata or {}).get("page_number")
        doc_id = result.doc_id
        if not doc_id or page_number is None:
            continue

        try:
            page_value = int(page_number)
        except (TypeError, ValueError):
            continue

        try:
            response = (
                vectorstore.client.table("chunks")
                .select("chunk_id, doc_id, content, metadata")
                .eq("doc_id", doc_id)
                .eq("metadata->>page_number", str(page_value))
                .execute()
            )
            rows = response.data if isinstance(getattr(response, "data", None), list) else []
        except Exception:
            logger.debug("Adjacent chunk expansion skipped for doc_id=%s page=%s", doc_id, page_number)
            continue

        if not rows:
            continue

        page_rows = []
        for row in rows:
            row_doc_id = row.get("doc_id")
            row_meta = row.get("metadata") or {}
            if not isinstance(row_meta, dict):
                continue
            if row_doc_id != doc_id:
                continue
            row_page_number = row_meta.get("page_number")
            if row_page_number is None:
                continue
            try:
                if int(row_page_number) != page_value:
                    continue
            except (TypeError, ValueError):
                continue
            page_rows.append(row)

        if not page_rows:
            continue

        ordered_rows = sorted(
            page_rows,
            key=lambda row: (
                _chunk_split_index(row.get("chunk_id"), row.get("metadata") or {}),
                str(row.get("chunk_id") or ""),
            ),
        )

        seed_index = None
        for idx, row in enumerate(ordered_rows):
            if row.get("chunk_id") == result.chunk_id:
                seed_index = idx
                break

        if seed_index is None:
            continue

        start_idx = max(0, seed_index - max_neighbors)
        end_idx = min(len(ordered_rows), seed_index + max_neighbors + 1)
        for index in range(start_idx, end_idx):
            row = ordered_rows[index]
            row_chunk_id = row.get("chunk_id")
            if row_chunk_id in seen_chunk_ids:
                continue
            if index == seed_index:
                continue
            neighbor = _build_result(row, result.similarity)
            expanded.append(neighbor)
            seen_chunk_ids.add(row_chunk_id)

    return expanded


def _resolve_company_identity(vectorstore: SupabaseVectorStore, company_id: str) -> Optional[str]:
    """Return the canonical company name and symbol from the existing document metadata."""
    if not company_id or not hasattr(vectorstore, "client") or not vectorstore.client:
        return None

    try:
        response = (
            vectorstore.client.table("documents")
            .select("metadata")
            .filter("metadata->>company_id", "eq", str(company_id))
            .limit(1)
            .execute()
        )
        rows = response.data if isinstance(getattr(response, "data", None), list) else []
    except Exception:
        return None

    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("name")
        symbol = metadata.get("symbol")
        if name:
            if symbol:
                return f"{name} ({symbol})"
            return str(name)

    return None


def _embed_query(embeddings_fn: Any, query: str) -> list[float]:
    """Support both callable embedding functions and HuggingFaceEmbeddings-like objects."""
    start = time.perf_counter()
    try:
        if callable(embeddings_fn):
            return embeddings_fn(query)
        if hasattr(embeddings_fn, "embed_query"):
            return embeddings_fn.embed_query(query)
        raise TypeError("Embeddings provider must be callable or expose embed_query().")
    finally:
        logger.info("PERF: query_embedding=%.2fs", time.perf_counter() - start)


def _collect_hybrid_search_results(
    vectorstore: SupabaseVectorStore,
    query: str,
    query_emb: list[float],
    filters: Optional[dict[str, Any] | SearchFilter],
    k: int,
) -> list[SearchResult]:
    """Run the existing dense + lexical + hybrid scoring pipeline for a given filter set."""
    start = time.perf_counter()
    candidate_k = max(30, k * 5)
    raw_results: list[SearchResult] = vectorstore.similarity_search(
        query_embedding=query_emb,
        filters=filters,
        k=candidate_k,
    )
    logger.info("PERF: dense_retrieval=%.2fs", time.perf_counter() - start)

    all_doc_chunks: list[dict[str, Any]] = []
    try:
        all_doc_chunks = _fetch_lexical_chunk_pool(vectorstore, filters, query=query)
    except Exception as exc:
        logger.warning("Lexical chunk scan failed; continuing with dense-only candidates: %s", exc)

    sim_map = {r.chunk_id: r for r in raw_results}
    scored_candidates: list[SearchResult] = []
    seen_texts: set[str] = set()

    pool_items: list[tuple[str, str, str, dict[str, Any], float, Optional[str], Optional[str]]] = []
    for r in raw_results:
        pool_items.append(
            (r.chunk_id, r.doc_id, r.content, r.metadata or {}, r.similarity, r.filename, r.uploaded_at)
        )

    for c in all_doc_chunks:
        cid = c.get("chunk_id")
        if cid and cid not in sim_map:
            pool_items.append(
                (
                    cid,
                    c.get("doc_id", ""),
                    c.get("content", ""),
                    c.get("metadata") or {},
                    0.0,
                    None,
                    None,
                )
            )

    rank_start = time.perf_counter()
    for cid, doc_id, content, meta, vec_sim, fname, up_at in pool_items:
        norm_key = re.sub(r"\s+", " ", content[:120].strip().lower())
        if not norm_key or norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)

        kw_score = _compute_keyword_relevance(query, content)
        is_toc = _is_table_of_contents_chunk(content)
        toc_factor = 0.15 if is_toc else 1.0
        prose_bonus = 0.0 if is_toc else _prose_quality_bonus(content, meta.get("content_type"))

        hybrid_score = (vec_sim * 1.2 + kw_score * 0.75 + prose_bonus) * toc_factor

        scored_candidates.append(
            SearchResult(
                chunk_id=cid,
                doc_id=doc_id,
                content=content,
                metadata=meta,
                similarity=float(hybrid_score),
                filename=fname,
                uploaded_at=up_at,
            )
        )

    scored_candidates.sort(key=lambda x: x.similarity, reverse=True)
    ranked = scored_candidates[:k]
    logger.info("PERF: hybrid_ranking=%.2fs", time.perf_counter() - rank_start)
    return ranked


def _safe_eval_node(node: ast.AST) -> float | int:
    """Safely evaluate arithmetic AST nodes without eval/exec."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _ALLOWED_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _ALLOWED_OPERATORS[op_type](operand)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator_tool(expression: str) -> ToolResult:
    """Safely evaluate arithmetic expressions without invoking python eval/exec.

    Args:
        expression: Math expression string (e.g., '145 * 24 + 18.5 / 2').

    Returns:
        ToolResult with the computed number or error.
    """
    logger.info("Executing calculator tool with expression: %s", expression)
    try:
        # Clean expression of common natural language wrapper noise
        clean_expr = expression.strip()
        clean_expr = re.sub(r"^[^\d\(\)\-\+\.]*", "", clean_expr)
        clean_expr = re.sub(r"[^\d\(\)\-\+\.\*\/\%]*$", "", clean_expr)

        if not clean_expr:
            return ToolResult(
                tool_name="calculator",
                output="Invalid expression",
                success=False,
                error="Empty or non-arithmetic expression",
            )

        parsed_ast = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_node(parsed_ast.body)
        formatted_result = f"{result:.6g}" if isinstance(result, float) else str(result)
        return ToolResult(
            tool_name="calculator",
            output=formatted_result,
            data={"result": result, "expression": clean_expr},
            success=True,
        )
    except Exception as exc:
        logger.warning("Calculator tool evaluation failed: %s", exc)
        return ToolResult(
            tool_name="calculator",
            output=f"Calculation error: {exc}",
            success=False,
            error=str(exc),
        )


def web_search_tool(query: str, max_results: int = 5) -> ToolResult:
    """Execute a web search query for real-time information via DuckDuckGo or OpenRouter plugin.

    Args:
        query: Search query string.
        max_results: Max items to return.

    Returns:
        ToolResult with web search snippets.
    """
    logger.info("Executing web search tool for query: %s", query)
    try:
        # Try duckduckgo_search if installed
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                formatted_snippets = []
                for i, item in enumerate(results, 1):
                    title = item.get("title", "No Title")
                    snippet = item.get("body", "")
                    link = item.get("href", "")
                    formatted_snippets.append(f"[{i}] **{title}**\n{snippet}\nURL: {link}")
                output_text = "\n\n".join(formatted_snippets)
                return ToolResult(
                    tool_name="web_search",
                    output=output_text,
                    data={"results_count": len(results)},
                    success=True,
                )
        except Exception as ddg_exc:
            logger.debug("DuckDuckGo search fallback: %s", ddg_exc)

        # Fallback stub if offline or module missing
        return ToolResult(
            tool_name="web_search",
            output=f"Web search results for '{query}': No live external network connectivity in current environment.",
            data={"query": query},
            success=True,
        )
    except Exception as exc:
        logger.warning("Web search tool failed: %s", exc)
        return ToolResult(
            tool_name="web_search",
            output=f"Web search failed: {exc}",
            success=False,
            error=str(exc),
        )


def retrieval_tool(
    query: str,
    vectorstore: Optional[SupabaseVectorStore] = None,
    embeddings_fn: Optional[Callable[[str], list[float]]] = None,
    filters: Optional[dict[str, Any] | SearchFilter] = None,
    k: int = DEFAULT_RETRIEVAL_K,
) -> ToolResult:
    """Retrieve relevant multi-document chunks from Supabase pgvector store using hybrid re-ranking.

    Args:
        query: Search query text.
        vectorstore: SupabaseVectorStore instance.
        embeddings_fn: Callable returning dense vector embedding for query text.
        filters: Metadata filter (e.g., doc_id, content_type).
        k: Number of chunks to retrieve.

    Returns:
        ToolResult with formatted chunks and source citations.
    """
    logger.info("Executing retrieval tool with query: '%s', k=%d", query, k)
    if vectorstore is None:
        return ToolResult(
            tool_name="retriever",
            output="Vectorstore is not configured or available.",
            success=False,
            error="No vectorstore provided",
        )

    try:
        if embeddings_fn is None:
            return ToolResult(
                tool_name="retriever",
                output="Embeddings generator is not available.",
                success=False,
                error="No embeddings_fn provided",
            )

        query_emb = _embed_query(embeddings_fn, query)
        filter_dict = _normalize_filters(filters)
        company_id = filter_dict.get("company_id")

        if company_id is not None:
            try:
                matching_doc_ids = _get_company_document_ids(vectorstore, str(company_id))
            except Exception as exc:
                return ToolResult(
                    tool_name="retriever",
                    output=f"Company research lookup failed for company_id '{company_id}'.",
                    success=False,
                    error=str(exc),
                )

            if not matching_doc_ids:
                return ToolResult(
                    tool_name="retriever",
                    output=f"No indexed documents were found for company_id '{company_id}'.",
                    data={"company_id": str(company_id), "count": 0, "chunks": []},
                    success=True,
                )

            working_filters = {key: value for key, value in filter_dict.items() if key != "company_id"}
            merged_results: dict[str, SearchResult] = {}
            for doc_id in matching_doc_ids:
                per_doc_filters = {**working_filters, "doc_id": doc_id}
                doc_results = _collect_hybrid_search_results(vectorstore, query, query_emb, per_doc_filters, k)
                for result in doc_results:
                    merged_results[result.chunk_id] = result

            search_results = sorted(merged_results.values(), key=lambda x: x.similarity, reverse=True)[:k]
        else:
            search_results = _collect_hybrid_search_results(vectorstore, query, query_emb, filters, k)

        logger.info(
            "Hybrid retrieval complete: returning=%d",
            len(search_results),
        )

        if not search_results:
            return ToolResult(
                tool_name="retriever",
                output="No matching document chunks found.",
                data={"chunks": [], "count": 0},
                success=True,
            )

        expanded_results = _expand_adjacent_context_chunks(vectorstore, search_results, max_neighbors=2)
        formatted_chunks = []
        for i, res in enumerate(expanded_results, 1):
            doc_label = res.filename or res.doc_id
            page_num = res.metadata.get("page_number", "?")
            c_type = res.metadata.get("content_type", "text")
            formatted_chunks.append(
                f"--- Source [{i}] (Doc: {doc_label}, Page: {page_num}, Type: {c_type}, Score: {res.similarity:.2f}) ---\n"
                f"{res.content}"
            )

        output_text = "\n\n".join(formatted_chunks)
        return ToolResult(
            tool_name="retriever",
            output=output_text,
            data={
                "count": len(expanded_results),
                "chunks": [r.model_dump() for r in expanded_results],
            },
            success=True,
        )
    except Exception as exc:
        logger.exception("Retrieval tool execution failed")
        return ToolResult(
            tool_name="retriever",
            output=f"Retrieval failed: {exc}",
            success=False,
            error=str(exc),
        )
