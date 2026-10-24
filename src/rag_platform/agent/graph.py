"""LangGraph-based agent routing between retrieval, tool calling, MCP clients, and direct LLM answering."""

import json
import re
import time
from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from rag_platform.agent.mcp_client import McpClient
from rag_platform.agent.excel_tool import create_excel_tool
from rag_platform.agent.tools import (
    DEFAULT_RETRIEVAL_K,
    ToolResult,
    _resolve_company_identity,
    calculator_tool,
    retrieval_tool,
    web_search_tool,
)
from rag_platform.config import get_settings
from rag_platform.exceptions import AgentError
from rag_platform.logging_config import get_correlation_id, get_logger
from rag_platform.observability.tracing import get_tracer
from rag_platform.vectorstore.supabase_store import SearchFilter, SupabaseVectorStore

logger = get_logger(__name__)


class AgentResult(BaseModel):
    """Structured response model returned by run_agent."""

    query: str = Field(..., description="User query submitted to the agent")
    answer: str = Field(..., description="Final synthesized answer")
    routing_decision: str = Field(..., description="Chosen execution path (retrieve, tool, mcp, direct, hybrid)")
    reasoning_path: list[str] = Field(default_factory=list, description="Sequence of reasoning steps and actions")
    sources: list[str] = Field(default_factory=list, description="Citations and context sources used in answer")
    tool_outputs: dict[str, Any] = Field(default_factory=dict, description="Outputs collected from invoked tools")


class AgentState(TypedDict):
    """Internal state threaded through LangGraph nodes."""

    query: str
    doc_filters: Optional[dict[str, Any]]
    routing_decision: str  # "retrieve", "tool", "mcp", "hybrid", "direct"
    tool_name: Optional[str]
    tool_input: Optional[str]
    retrieval_context: Optional[str]
    tool_output: Optional[str]
    final_answer: Optional[str]
    reasoning_path: list[str]
    sources: list[str]
    error: Optional[str]
    mode: Optional[str]
    allow_retrieval: bool
    generated_file: Optional[dict[str, Any]]


# Total time one LLM call (including model fallbacks) may take before we give up.
LLM_TOTAL_BUDGET_SECONDS = 100.0
LLM_REQUEST_TIMEOUT_SECONDS = 60
LLM_MAX_TOKENS = 1500  # caps answer length so slow free models finish sooner

_EXCEL_WORDS = ("excel", "xlsx", "spreadsheet", "workbook")
_EXCEL_VERBS = ("create", "make", "generate", "build", "export", "prepare", "give me", "produce", "draft", "download")



def _retrieval_failure_message(output: Any) -> str:
    """Explain a failed document search; quota problems get a plain-language message."""
    text = str(output)
    lowered = text.lower()
    if "429" in text or "daily free allocation" in lowered or "quota" in lowered:
        return (
            "I can't search the documents right now because the free daily limit of the search service has been "
            "reached. It resets every day at 00:00 UTC (05:00 Pakistan time). Please try again then."
        )
    return f"I could not search your documents: {text}"

def _is_excel_request(query: str) -> bool:
    lowered = query.lower()
    return any(w in lowered for w in _EXCEL_WORDS) and any(v in lowered for v in _EXCEL_VERBS)


def default_llm_completion(
    messages: list[dict[str, str]],
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    stream: bool = False,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> str:
    """Execute LLM chat completion using centralized settings (OpenRouter/OpenAI) with optional streaming."""
    import requests

    start = time.perf_counter()
    settings = get_settings()
    if not settings.effective_api_key:
        logger.warning("No LLM API key configured for default_llm_completion")
        logger.info("PERF: llm=%.2fs", time.perf_counter() - start)
        return ""

    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.effective_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG Chat Platform",
    }

    primary_model = model or settings.LLM_MODEL
    model_mode = getattr(settings, "LLM_MODEL_MODE", "fixed")

    if model_mode == "auto" and "openrouter" in settings.OPENROUTER_BASE_URL.lower():
        fallback_models = [
            "openrouter/free",
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3.5-super-vl:free",
            "google/gemma-4-26b-a4b-it:free",
        ]
        candidate_models = [primary_model]
        for fm in fallback_models:
            if fm not in candidate_models:
                candidate_models.append(fm)
    else:
        candidate_models = [primary_model]

    for cand in candidate_models:
        if time.perf_counter() - start > LLM_TOTAL_BUDGET_SECONDS:
            logger.warning("LLM time budget of %.0fs used up; giving up", LLM_TOTAL_BUDGET_SECONDS)
            break
        payload = {
            "model": cand,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.RAG_TEMPERATURE,
            "stream": stream,
            "max_tokens": LLM_MAX_TOKENS,
        }
        requested_max_tokens = payload.get("max_tokens")
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict))
        try:
            request_start = time.perf_counter()
            if stream:
                with requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS, stream=True) as resp:
                    if resp.status_code != 200:
                        logger.warning("LLM model %s returned status %s after %.3fs", cand, resp.status_code, time.perf_counter() - request_start)
                        continue
                    collected: list[str] = []
                    for line in resp.iter_lines():
                        if not line or not line.startswith(b"data:"):
                            continue
                        data = line.decode("utf-8", errors="replace")[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            payload_obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = payload_obj.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        chunk = delta.get("content") or delta.get("reasoning") or ""
                        if not chunk:
                            continue
                        if on_chunk is not None:
                            on_chunk(chunk)
                        collected.append(chunk)
                    content = "".join(collected)
                    if content:
                        logger.info(
                            "PERF: llm_model=%s llm_wait=%.3fs prompt_chars=%d max_tokens=%s output_chars=%d total=%.2fs",
                            cand,
                            time.perf_counter() - request_start,
                            prompt_chars,
                            requested_max_tokens,
                            len(content),
                            time.perf_counter() - start,
                        )
                        logger.info("PERF: llm=%.2fs", time.perf_counter() - start)
                        return content
                    continue

            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)
            request_wait = time.perf_counter() - request_start
            if resp.status_code != 200:
                logger.warning("LLM model %s returned status %s after %.3fs", cand, resp.status_code, request_wait)
                continue
            parse_start = time.perf_counter()
            data = resp.json()
            parse_elapsed = time.perf_counter() - parse_start
            if "error" in data and not data.get("choices"):
                logger.warning("LLM model %s returned error: %s after %.3fs", cand, data.get("error"), request_wait)
                continue
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                msg = choices[0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                if content:
                    output_chars = len(content)
                    logger.info(
                        "PERF: llm_model=%s llm_wait=%.3fs prompt_chars=%d max_tokens=%s output_chars=%d response_parse=%.3fs total=%.2fs",
                        cand,
                        request_wait,
                        prompt_chars,
                        requested_max_tokens,
                        output_chars,
                        parse_elapsed,
                        time.perf_counter() - start,
                    )
                    logger.info("PERF: llm=%.2fs", time.perf_counter() - start)
                    return content
        except Exception as exc:
            logger.warning("Default LLM completion attempt with %s failed after %.3fs: %s", cand, time.perf_counter() - request_start if 'request_start' in locals() else 0.0, exc)
            continue

    logger.info("PERF: llm=%.2fs", time.perf_counter() - start)
    return ""


def create_agent_graph(
    llm_call_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
    vectorstore: Optional[SupabaseVectorStore] = None,
    embeddings_fn: Optional[Callable[[str], list[float]]] = None,
    mcp_client: Optional[McpClient] = None,
    token_streamer: Optional[Callable[[str], None]] = None,
) -> StateGraph:
    """Build and compile the LangGraph agent state machine.

    Nodes:
    1. router: Classifies whether query requires internal retrieval, local tool, MCP tool, hybrid, or direct response.
    2. retriever: Calls retrieval tool against Supabase pgvector store.
    3. tool_caller: Executes tools (calculator, web_search, or external MCP client tools like read_file).
    4. responder: Synthesizes retrieved context and tool outputs into a grounded final answer with citations.

    Args:
        llm_call_fn: Callable taking OpenAI/OpenRouter messages and returning completion text.
        vectorstore: SupabaseVectorStore instance for retrieval.
        embeddings_fn: Callable taking text query and returning dense embedding vector.
        mcp_client: Optional McpClient instance for consuming external MCP servers.

    Returns:
        Compiled LangGraph state graph.
    """
    client = mcp_client or McpClient()
    discovered_mcp_tools = client.discover_tools()
    tracer = get_tracer()
    active_llm = llm_call_fn or default_llm_completion

    def _call_llm(messages: list[dict[str, str]], fallback: str = "", stream: bool = False, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        try:
            try:
                res = active_llm(messages, stream=stream, on_chunk=on_chunk)
            except TypeError:
                if stream:
                    res = active_llm(messages, on_chunk=on_chunk)
                else:
                    res = active_llm(messages)
            return res if res else fallback
        except Exception as exc:
            logger.warning("Agent LLM call failed: %s; using fallback", exc)
            return fallback

    def _contains_tool_call_markup(text: Optional[str]) -> bool:
        if not isinstance(text, str):
            return False
        markup_tokens = ("<tool_call", "</tool_call>", "<arg_key>", "<arg_value>")
        lowered = text.lower()
        return any(token in lowered for token in markup_tokens)

    # --- NODE 1: ROUTER ---
    # Rule-based on purpose: a free-tier LLM call just to pick a route costs 5-30 s and often
    # fails. Rules decide in microseconds, and the choice is shown in "How this was answered".
    def router_node(state: AgentState) -> dict[str, Any]:
        node_start = time.perf_counter()
        query = state["query"]
        doc_filters = state.get("doc_filters")
        allow_retrieval = state.get("allow_retrieval", True)
        mode = (state.get("mode") or "").strip()
        lowered = query.lower()
        logger.info("Agent Router analyzing query: '%s' (mode=%s)", query, mode or "-")
        reasoning = list(state.get("reasoning_path", []))

        company_id = None
        if isinstance(doc_filters, dict):
            company_id = doc_filters.get("company_id")
        elif isinstance(doc_filters, SearchFilter):
            company_id = getattr(doc_filters, "company_id", None)

        math_match = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*\/\%]\s*\d+(?:\.\d+)?)", query)
        calc_words = any(k in lowered for k in ("calculate", "sum of", "product of", "divided by", "multiplied by"))
        file_path_match = re.search(r"([a-zA-Z]:[\\\/][^\s]+|[a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]{1,5})", query)
        wants_read_file = bool(file_path_match) and any(k in lowered for k in ("read file", "open file", "contents of", "inspect file", "read "))
        wants_live = any(k in lowered for k in ("news", "latest", "current price", "today", "weather", "right now"))
        wants_excel = _is_excel_request(query)

        decision = "direct"
        tool_name: Optional[str] = None
        tool_input: Optional[str] = query
        why = ""

        if not allow_retrieval:
            # Q&A mode: your documents are never searched.
            if wants_excel:
                decision, tool_name, why = "tool", "create_excel", "Q&A mode: spreadsheet requested, documents are not searched"
            elif math_match or calc_words:
                decision, tool_name = "tool", "calculator"
                tool_input = math_match.group(0) if math_match else query
                why = "Q&A mode: arithmetic detected, using the calculator"
            elif wants_live:
                decision, tool_name, why = "mcp", "mcp_web_search", "Q&A mode: live information requested, using web search"
            else:
                why = "Q&A mode: answering from general knowledge, documents are not searched"
        elif company_id is not None:
            decision = "hybrid" if wants_excel else "retrieve"
            tool_name = "create_excel" if wants_excel else None
            why = f"Research mode: searching documents of the selected company (company_id={company_id})"
        else:
            if wants_excel:
                decision, tool_name, why = "hybrid", "create_excel", "Spreadsheet requested: searching the documents first, then building the Excel file"
            elif wants_read_file:
                decision, tool_name = "mcp", "read_file"
                tool_input = file_path_match.group(0)
                why = "File path detected, reading it with the file tool"
            elif (math_match or calc_words) and not doc_filters and not any(
                k in lowered for k in ("document", "pdf", "report", "table", "page", "revenue", "filing")
            ):
                decision, tool_name = "tool", "calculator"
                tool_input = math_match.group(0) if math_match else query
                why = "Arithmetic detected, using the calculator"
            elif wants_live and not doc_filters:
                decision, tool_name, why = "mcp", "mcp_web_search", "Live information requested, using web search"
            else:
                decision, why = "retrieve", "Searching your documents"

        reasoning.append(f"Router decision: '{decision}' (tool: {tool_name or 'none'}) - {why}")
        logger.info("Router decision finalized: %s", decision)
        logger.info("PERF: router_decision=%.2fs", time.perf_counter() - node_start)

        tracer.record_span(
            trace_id=get_correlation_id(),
            name="router",
            start_time=node_start,
            input_data={"query": query, "mode": mode, "company_id": company_id},
            output_data={"decision": decision, "tool_name": tool_name},
        )

        return {
            "routing_decision": decision,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "reasoning_path": reasoning,
        }

    # --- NODE 2: RETRIEVER ---
    def retriever_node(state: AgentState) -> dict[str, Any]:
        node_start = time.time()
        query = state["query"]
        doc_filters = state.get("doc_filters")
        reasoning = list(state.get("reasoning_path", []))
        sources = list(state.get("sources", []))

        logger.info(
            "Agent Retriever node fetching context for: '%s' (k=%d)",
            query,
            DEFAULT_RETRIEVAL_K,
        )
        res = retrieval_tool(
            query=query,
            vectorstore=vectorstore,
            embeddings_fn=embeddings_fn,
            filters=doc_filters,
            k=DEFAULT_RETRIEVAL_K,
        )

        context_text = res.output
        data = res.data or {}
        zero_company_docs = (
            res.success
            and isinstance(data, dict)
            and data.get("count") == 0
            and data.get("chunks") == []
            and doc_filters is not None
            and isinstance(doc_filters, dict)
            and doc_filters.get("company_id") is not None
        )

        if not res.success:
            reasoning.append(f"Retriever failed: {res.output}")
            tracer.record_span(
                trace_id=get_correlation_id(),
                name="retriever",
                start_time=node_start,
                input_data={"query": query, "filters": doc_filters},
                output_data={"error": res.output},
            )
            return {
                "retrieval_context": None,
                "reasoning_path": reasoning,
                "sources": [],
                "final_answer": _retrieval_failure_message(res.output),
            }

        no_hits = bool(res.success and isinstance(data, dict) and data.get("count") == 0 and not zero_company_docs)
        if no_hits and state.get("routing_decision") != "hybrid":
            reasoning.append("Retriever found no matching chunks; no LLM call needed")
            return {
                "retrieval_context": None,
                "reasoning_path": reasoning,
                "sources": [],
                "final_answer": (
                    "I could not find anything relevant in the indexed documents. "
                    "Check that a PDF has been uploaded and indexed, or choose a different document or company."
                ),
            }

        if zero_company_docs:
            reasoning.append(
                "Retriever found zero company-scoped indexed documents; skipping LLM synthesis to avoid unrelated retrieval leakage"
            )
            corr_id = get_correlation_id()
            tracer.record_span(
                trace_id=corr_id,
                name="retriever",
                start_time=node_start,
                input_data={"query": query, "filters": doc_filters},
                output_data={"chunk_count": 0, "company_scoped_zero_result": True},
            )
            return {
                "retrieval_context": None,
                "reasoning_path": reasoning,
                "sources": [],
                "final_answer": "No indexed documents were found for the selected company.",
            }

        if res.success and data and "chunks" in data:
            for c in data["chunks"]:
                doc_name = c.get("filename") or c.get("doc_id", "doc")
                sources.append(f"{doc_name} (Page {c.get('metadata', {}).get('page_number', '?')})")

        reasoning.append(f"Retriever executed: found {data.get('count', 0) if data else 0} matching chunks")

        corr_id = get_correlation_id()
        tracer.record_span(
            trace_id=corr_id,
            name="retriever",
            start_time=node_start,
            input_data={"query": query, "filters": doc_filters},
            output_data={"chunk_count": data.get("count", 0) if data else 0},
        )

        return {
            "retrieval_context": context_text,
            "reasoning_path": reasoning,
            "sources": sources,
        }

    def _build_excel_spec(query: str, context: Optional[str]) -> Optional[dict[str, Any]]:
        """Ask the LLM to turn the request (plus retrieved facts) into rows for create_excel."""
        evidence = (context or "")[:6000]
        messages = [
            {
                "role": "system",
                "content": (
                    "You turn a request into spreadsheet data. Reply with ONLY one JSON object, no prose, in this shape: "
                    '{"filename": "short_name", "sheet_name": "Sheet1", "data": [{"Column A": "value", "Column B": 1}]}. '
                    "Every row object must use the same column names. Use at most 200 rows. "
                    "If context is provided, take facts and numbers ONLY from it and never invent values; "
                    "leave a cell empty when the context does not say. Without context, build the table the user describes "
                    "(for example a template with headers and a few example rows)."
                ),
            },
            {"role": "user", "content": f"Request: {query}\n\nContext:\n{evidence or '(none)'}"},
        ]
        raw = _call_llm(messages)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return None
        try:
            spec = json.loads(match.group(0))
        except Exception:
            return None
        rows = spec.get("data")
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            return None
        return {
            "filename": spec.get("filename") or "generated_file",
            "sheet_name": spec.get("sheet_name") or "Sheet1",
            "data": rows[:200],
        }

    # --- NODE 3: TOOL CALLER (Supports Local Tools + MCP Client Tools) ---
    def tool_caller_node(state: AgentState) -> dict[str, Any]:
        node_start = time.time()
        tool_name = state.get("tool_name") or "calculator"
        tool_input = state.get("tool_input") or state["query"]
        reasoning = list(state.get("reasoning_path", []))

        logger.info("Agent Tool Caller node executing tool: %s with input: '%s'", tool_name, tool_input)

        if tool_name == "create_excel":
            spec = _build_excel_spec(state["query"], state.get("retrieval_context"))
            if spec is None:
                res = ToolResult(
                    tool_name="create_excel",
                    output="the language model did not return usable table data (it may be busy), please try again",
                    success=False,
                    error="no table data",
                )
            else:
                res = create_excel_tool(
                    filename=spec["filename"], sheet_name=spec["sheet_name"], data=spec["data"]
                )
        elif tool_name == "calculator":
            res = calculator_tool(tool_input)
        elif tool_name == "web_search":
            res = web_search_tool(tool_input)
        elif tool_name in ("read_file", "mcp_web_search") or tool_name in client.discovered_tools:
            # Delegate to MCP Client with automatic retry and graceful fallback
            args = {"path": tool_input} if tool_name == "read_file" else {"query": tool_input}
            res = client.call_tool(tool_name, arguments=args, max_retries=1)
        else:
            res = ToolResult(
                tool_name=tool_name,
                output=f"Tool '{tool_name}' is not recognized.",
                success=False,
                error="Unknown tool",
            )

        reasoning.append(f"Tool '{tool_name}' executed -> output: {res.output[:100]}")

        corr_id = get_correlation_id()
        tracer.record_span(
            trace_id=corr_id,
            name=f"tool:{tool_name}",
            start_time=node_start,
            input_data={"tool": tool_name, "input": tool_input},
            output_data={"success": res.success, "output_preview": res.output[:150]},
        )

        update: dict[str, Any] = {
            "tool_output": res.output,
            "reasoning_path": reasoning,
        }
        if tool_name == "create_excel":
            generated = (res.data or {}).get("generated_file") if res.success else None
            update["generated_file"] = generated
            if generated:
                update["final_answer"] = (
                    f"I created the Excel file **{generated['filename']}** "
                    f"({generated['row_count']} rows, {generated['column_count']} columns). "
                    "Use the download button below this message."
                )
            else:
                update["final_answer"] = f"I could not create the Excel file: {res.output}"
        return update

    # --- NODE 4: RESPONDER ---
    def responder_node(state: AgentState) -> dict[str, Any]:
        node_start = time.time()
        query = state["query"]
        decision = state.get("routing_decision", "direct")
        context = state.get("retrieval_context")
        tool_out = state.get("tool_output")
        reasoning = list(state.get("reasoning_path", []))

        logger.info("Agent Responder synthesizing final answer for decision: %s", decision)

        context_blocks = []
        if context:
            context_blocks.append(f"=== KNOWLEDGE BASE CONTEXT ===\n{context}")
        if tool_out:
            context_blocks.append(f"=== TOOL / MCP OUTPUT ===\n{tool_out}")

        selected_company_identity = None
        if isinstance(state.get("doc_filters"), dict) and state["doc_filters"].get("company_id"):
            selected_company_identity = _resolve_company_identity(vectorstore, str(state["doc_filters"]["company_id"]))
        elif isinstance(state.get("doc_filters"), SearchFilter) and getattr(state["doc_filters"], "company_id", None):
            selected_company_identity = _resolve_company_identity(vectorstore, str(getattr(state["doc_filters"], "company_id")))

        full_context = "\n\n".join(context_blocks)
        if selected_company_identity:
            authority_block = (
                "Selected company identity (authoritative): "
                f"{selected_company_identity}. Do not substitute a different company name in the final answer. "
                "Use this company identity as the canonical company for the answer."
            )
            full_context = f"{authority_block}\n\n{full_context}" if full_context else authority_block

        tool_guard_instruction = (
            "You are now in the final synthesis step. "
            "The external/MCP tool has already been executed and the provided tool output is the result/context from that execution. "
            "Do not attempt another tool call, do not request another lookup, and do not issue any additional agent/tool loop. "
            "Do not output <tool_call>, </tool_call>, <arg_key>, <arg_value>, or similar tool-invocation markup. "
            "Return only the final user-facing answer in plain prose. "
            "Synthesize the answer from the provided context and tool output only. "
            "If the available evidence is insufficient, clearly say that rather than inventing information."
        )

        system_instruction = (
            "You are a grounded, evidence-first assistant. "
            "Use ONLY the provided retrieved context and tool outputs to answer the user's question. "
            "Do not use outside knowledge, assumptions, general world knowledge, or inferred facts to fill gaps. "
            "Every factual claim must be explicitly supported by the retrieved context. "
            "If the retrieved context does not contain enough evidence to answer, explicitly say that the information is not available in the provided/indexed documents. "
            "Do not infer a numeric value such as zero simply because something is not mentioned. "
            "For hypothetical, impossible, fabricated, or unsupported premises, do not invent an answer; state that the provided documents do not support the premise. "
            "When multiple figures are present, preserve their distinctions exactly (for example standalone vs consolidated), and do not silently collapse them into one. "
            "Prefer direct evidence over loosely related context. "
            "Do not cite a source merely because it is related to the company; only cite sources that actually support the claim being made. "
            "Keep the answer concise and directly answer the question. "
            "If the evidence is insufficient, abstain rather than hallucinate. "
            "Do not browse the web or introduce external information unless the current routing/tool path explicitly requested web research. For company-scoped Research mode, stay within the retrieved company documents. "
            "Preserve the application's existing citation/source format and do not invent sources."
        )

        if not state.get("allow_retrieval", True):
            system_instruction = (
                "You are a helpful, knowledgeable assistant in general Q&A mode. "
                "The user's uploaded documents are NOT available in this mode, so do not claim to have read them. "
                "Answer clearly and concisely from general knowledge. "
                "If a tool output is provided, rely on it for the facts it contains."
            )

        if tool_out:
            system_instruction = f"{system_instruction} {tool_guard_instruction}"

        prompt_messages = [
            {"role": "system", "content": f"{system_instruction}\n\n{full_context}" if full_context else system_instruction},
            {"role": "user", "content": query},
        ]

        if state.get("final_answer") is not None:
            answer = state["final_answer"]
            reasoning.append("Responder skipped LLM synthesis because company-scoped retrieval returned zero indexed documents")
        else:
            answer = _call_llm(prompt_messages, stream=bool(token_streamer), on_chunk=token_streamer)
            if _contains_tool_call_markup(answer):
                retry_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "The previous final answer contained tool-call markup. "
                            "This is invalid for the final user-facing response. "
                            "The external/MCP tool has already been executed and the provided tool output/context is the source of truth. "
                            "Do not call tools again. Do not emit any <tool_call>, </tool_call>, <arg_key>, <arg_value>, or similar markup. "
                            "Return only a concise final answer in plain prose based on the already-provided context and tool output. "
                            "If the available evidence is insufficient, say that clearly instead of inventing information.\n\n"
                            f"{full_context}"
                            if full_context else ""
                        ),
                    },
                    {"role": "user", "content": query},
                ]
                retry_answer = _call_llm(retry_prompt)
                if retry_answer and not _contains_tool_call_markup(retry_answer):
                    answer = retry_answer
                else:
                    answer = (
                        "The available external tool result and retrieved context do not contain enough information "
                        "to answer this request accurately."
                    )
            if not answer:
                if tool_out and not context:
                    answer = f"Result from {state.get('tool_name', 'tool')}:\n\n{tool_out}"
                elif context and not tool_out:
                    # Prefer the highest-ranked retrieved excerpt (already hybrid-reranked),
                    # not a short prefix that often truncates into TOC headings.
                    answer = (
                        "LLM synthesis was temporarily unavailable. "
                        "Highest-ranked retrieved excerpts:\n\n"
                        f"{context[:2500]}"
                    )
                elif context and tool_out:
                    answer = (
                        "LLM synthesis was temporarily unavailable.\n\n"
                        f"Retrieved context:\n{context[:1500]}\n\nTool result:\n{tool_out}"
                    )
                else:
                    answer = f"Response to: {query}"
            reasoning.append("Responder generated final synthesized response with citations")

        corr_id = get_correlation_id()
        tracer.record_span(
            trace_id=corr_id,
            name="responder",
            start_time=node_start,
            input_data={"has_context": bool(context), "has_tool_output": bool(tool_out)},
            output_data={"answer_length": len(answer)},
        )

        return {
            "final_answer": answer,
            "reasoning_path": reasoning,
        }

    # --- ROUTING CONDITION FUNCTIONS ---
    def select_next_node(state: AgentState) -> Literal["retriever", "tool_caller", "responder"]:
        decision = state.get("routing_decision", "direct")
        if decision in ("retrieve", "hybrid"):
            return "retriever"
        if decision in ("tool", "mcp"):
            return "tool_caller"
        return "responder"

    def select_after_retriever(state: AgentState) -> Literal["tool_caller", "responder"]:
        if state.get("final_answer") is not None:
            return "responder"
        decision = state.get("routing_decision", "retrieve")
        if decision == "hybrid":
            return "tool_caller"
        return "responder"

    # --- BUILD GRAPH ---
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tool_caller", tool_caller_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        select_next_node,
        {
            "retriever": "retriever",
            "tool_caller": "tool_caller",
            "responder": "responder",
        },
    )

    workflow.add_conditional_edges(
        "retriever",
        select_after_retriever,
        {
            "tool_caller": "tool_caller",
            "responder": "responder",
        },
    )

    workflow.add_edge("tool_caller", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


def run_agent(
    query: str,
    doc_filters: Optional[dict[str, Any]] = None,
    vectorstore: Optional[SupabaseVectorStore] = None,
    embeddings_fn: Optional[Callable[[str], list[float]]] = None,
    llm_call_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
    mcp_client: Optional[McpClient] = None,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    allow_retrieval: bool = True,
    mode: Optional[str] = None,
    **_ignored: Any,
) -> AgentResult:
    """Execute the compiled LangGraph agent pipeline with MCP client tools and observability tracing.

    Args:
        query: User question or command.
        doc_filters: Metadata filters for document scope (e.g., {'doc_id': '...'}).
        vectorstore: SupabaseVectorStore instance.
        embeddings_fn: Callable generating query dense vector embeddings.
        llm_call_fn: Callable executing LLM completions.
        mcp_client: Optional McpClient instance.

    Returns:
        AgentResult with answer, reasoning path, sources, and tool outputs.

    Raises:
        AgentError: If execution fails fatally.
    """
    corr_id = get_correlation_id()
    tracer = get_tracer()
    tracer.start_trace(query=query, trace_id=corr_id)
    start = time.perf_counter()

    logger.info("run_agent started (correlation_id=%s) for query: '%s'", corr_id, query)

    if not query or not query.strip():
        raise AgentError("Query cannot be empty")

    settings = get_settings()

    active_vectorstore = vectorstore
    if active_vectorstore is None and settings.SUPABASE_URL and settings.effective_supabase_key:
        try:
            active_vectorstore = SupabaseVectorStore()
        except Exception as exc:
            logger.warning("SupabaseVectorStore auto-init failed: %s", exc)

    active_embeddings_fn = embeddings_fn
    if active_embeddings_fn is None:
        try:
            from rag_platform.vectorstore.embeddings import get_embedding_model

            hf = get_embedding_model()
            active_embeddings_fn = hf.embed_query
        except Exception as exc:
            logger.warning("Embeddings auto-init failed: %s", exc)
            _embeddings_problem = str(exc)

            def active_embeddings_fn(_text: str, _problem: str = _embeddings_problem) -> list[float]:
                raise RuntimeError(_problem)

    active_llm_call_fn = llm_call_fn or default_llm_completion

    try:
        app = create_agent_graph(
            llm_call_fn=active_llm_call_fn,
            vectorstore=active_vectorstore,
            embeddings_fn=active_embeddings_fn,
            mcp_client=mcp_client,
            token_streamer=on_token if stream else None,
        )

        initial_state: AgentState = {
            "query": query.strip(),
            "doc_filters": doc_filters,
            "routing_decision": "direct",
            "tool_name": None,
            "tool_input": None,
            "retrieval_context": None,
            "tool_output": None,
            "final_answer": None,
            "reasoning_path": [f"Initialized agent run (correlation_id={corr_id})"],
            "sources": [],
            "error": None,
            "mode": mode,
            "allow_retrieval": allow_retrieval,
            "generated_file": None,
        }

        final_state = app.invoke(initial_state)

        tool_outputs = {}
        if final_state.get("tool_output"):
            tool_outputs[final_state.get("tool_name") or "tool"] = final_state.get("tool_output")
        if final_state.get("retrieval_context"):
            tool_outputs["retriever"] = final_state.get("retrieval_context")
        if final_state.get("generated_file"):
            tool_outputs["generated_file"] = final_state["generated_file"]

        final_answer = final_state.get("final_answer") or "No answer generated."
        unique_sources = list(dict.fromkeys(final_state.get("sources", [])))

        tracer.end_trace(
            trace_id=corr_id,
            final_answer=final_answer,
            routing_decision=final_state.get("routing_decision"),
            tool_calls=[{"tool": final_state.get("tool_name"), "output": final_state.get("tool_output")}] if final_state.get("tool_name") else [],
            metadata={"sources": unique_sources},
        )
        logger.info("PERF: final_agent_completion=%.2fs", time.perf_counter() - start)

        return AgentResult(
            query=query,
            answer=final_answer,
            routing_decision=final_state.get("routing_decision", "direct"),
            reasoning_path=final_state.get("reasoning_path", []),
            sources=unique_sources,
            tool_outputs=tool_outputs,
        )
    except AgentError:
        raise
    except Exception as exc:
        logger.exception("Agent graph execution encountered an unexpected error")
        raise AgentError(
            f"Agent execution failed for query '{query}': {exc}",
            details={"query": query, "correlation_id": corr_id},
        ) from exc
