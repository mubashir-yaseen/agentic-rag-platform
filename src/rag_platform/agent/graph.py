"""LangGraph-based agent routing between retrieval, tool calling, MCP clients, and direct LLM answering."""

import json
import re
import time
from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from rag_platform.agent.mcp_client import McpClient
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
    tool_data: Optional[dict[str, Any]]
    final_answer: Optional[str]
    reasoning_path: list[str]
    sources: list[str]
    error: Optional[str]


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
            "nvidia/nemotron-3.5-lightning:free",
            "inclusionai/ling-3.0-flash-fin:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-4-31b-it:free",
            "liquid/lfm-2.5-2.6b:free",
        ]
        candidate_models = [primary_model]
        for fm in fallback_models:
            if fm not in candidate_models:
                candidate_models.append(fm)
    else:
        candidate_models = [primary_model]

    for cand in candidate_models:
        payload = {
            "model": cand,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.RAG_TEMPERATURE,
            "stream": stream,
        }
        requested_max_tokens = payload.get("max_tokens")
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict))
        try:
            request_start = time.perf_counter()
            if stream:
                with requests.post(url, headers=headers, json=payload, timeout=90, stream=True) as resp:
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

            resp = requests.post(url, headers=headers, json=payload, timeout=90)
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
    allow_retrieval: bool = True,
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

    def _try_parse_json_object(text: Optional[str]) -> Optional[dict[str, Any]]:
        """Best-effort extraction of a JSON object embedded in LLM output text."""
        if not isinstance(text, str) or not text.strip():
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # --- NODE 1: ROUTER ---
    def router_node(state: AgentState) -> dict[str, Any]:
        node_start = time.perf_counter()
        query = state["query"]
        doc_filters = state.get("doc_filters")
        logger.info("Agent Router analyzing query: '%s'", query)
        reasoning = list(state.get("reasoning_path", []))

        # If retrieval is disabled for this run (e.g., Q&A strict isolation),
        # force a direct decision and bypass retrieval/tool routing.
        if not allow_retrieval:
            reasoning.append("Router forced to 'direct' due to allow_retrieval=False")
            corr_id = get_correlation_id()
            tracer.record_span(
                trace_id=corr_id,
                name="router",
                start_time=node_start,
                input_data={"query": query, "allow_retrieval": False},
                output_data={"decision": "direct", "tool_name": None},
            )
            return {
                "routing_decision": "direct",
                "tool_name": None,
                "tool_input": query,
                "reasoning_path": reasoning,
            }

        company_id = None
        if isinstance(doc_filters, dict):
            company_id = doc_filters.get("company_id")
        elif isinstance(doc_filters, SearchFilter):
            company_id = getattr(doc_filters, "company_id", None)

        if company_id is not None:
            decision = "retrieve"
            tool_name = None
            tool_input = query
            reasoning.append(f"Company-scoped Research request bypassed LLM router for company_id={company_id}")
            logger.info("Bypassing LLM router for company-scoped request with company_id=%s", company_id)
            logger.info("PERF: router_decision=%.2fs", time.perf_counter() - node_start)
            corr_id = get_correlation_id()
            tracer.record_span(
                trace_id=corr_id,
                name="router",
                start_time=node_start,
                input_data={"query": query, "company_id": company_id},
                output_data={"decision": decision, "tool_name": tool_name},
            )
            return {
                "routing_decision": decision,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "reasoning_path": reasoning,
            }

        # Check for obvious math expressions first
        math_match = re.search(r"(\d+\s*[\+\-\*\/\%]\s*\d+)", query)

        # Check for file path patterns for MCP filesystem tool (support Windows/POSIX paths)
        file_path_match = re.search(r"([a-zA-Z]:[\\\/][^\s]+|[a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]{1,5})", query)

        routing_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a routing classification engine for an AI assistant. "
                    "Classify the user query into exactly ONE of the following action categories:\n"
                    "- 'retrieve': The user asks about indexed knowledge base documents, research proposals, thesis, reports, financial data, tables, figures, or domain facts.\n"
                    "- 'tool': The user asks to perform arithmetic calculations or local tool execution.\n"
                    "- 'mcp': The user asks to read a specific local file, generate an Excel file from data already available, or query live external MCP servers.\n"
                    "- 'hybrid': The user query requires retrieving document facts AND performing calculations/external actions (including generating an Excel file FROM retrieved data).\n"
                    "- 'direct': General greetings, chit-chat, or queries completely unrelated to documents or tools.\n\n"
                    "Available MCP tools: " + ", ".join([f"{t.tool_name} ({t.description})" for t in discovered_mcp_tools]) + "\n\n"
                    "Respond with JSON format strictly:\n"
                    '{"decision": "retrieve"|"tool"|"mcp"|"hybrid"|"direct", "tool_name": "calculator"|"web_search"|"read_file"|"mcp_web_search"|"create_excel"|null, "tool_input": "<args>", "reason": "<short explanation>"}'
                ),
            },
            {"role": "user", "content": f"Query: {query}"},
        ]

        llm_resp = _call_llm(routing_prompt, fallback="")
        decision = "direct"
        tool_name = None
        tool_input = None

        try:
            json_match = re.search(r"\{.*\}", llm_resp, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                decision = parsed.get("decision", "direct")
                tool_name = parsed.get("tool_name")
                tool_input = parsed.get("tool_input")
        except Exception:
            pass

        # If LLM didn't specify tool_input or if heuristics apply
        if not tool_input or decision == "direct":
            if any(k in query.lower() for k in ["read file", "open file", "contents of", "inspect file"]) or (file_path_match and "read" in query.lower()):
                decision = "mcp"
                tool_name = "read_file"
                tool_input = file_path_match.group(0) if file_path_match else query
            elif math_match or any(k in query.lower() for k in ["calculate", "sum of", "product of", "what is 2+", "divided by"]):
                decision = "tool" if decision == "direct" else decision
                tool_name = "calculator"
                tool_input = math_match.group(0) if math_match else query
            elif any(k in query.lower() for k in ["document", "pdf", "table", "report", "filing", "revenue", "financial", "sec", "page", "proposal", "thesis", "vlm", "baseline", "dataset", "research", "author"]):
                decision = "retrieve" if decision == "direct" else decision
            elif any(k in query.lower() for k in ["news", "latest", "current price", "today", "weather"]):
                decision = "mcp" if decision == "direct" else decision
                tool_name = "mcp_web_search"
                tool_input = query
            elif any(k in query.lower() for k in ["excel", "xlsx", "spreadsheet"]):
                tool_name = "create_excel"
                tool_input = query

        # Fallback heuristic: If query mentions Excel and tool_name wasn't set, set it
        if not tool_name and any(k in query.lower() for k in ["excel", "xlsx", "spreadsheet"]):
            tool_name = "create_excel"
            tool_input = tool_input or query

        # Fix 2: Explicit Excel routing logic
        if tool_name == "create_excel":
            parsed = _try_parse_json_object(tool_input)
            has_structured_rows = False
            if isinstance(parsed, dict):
                d = parsed.get("data")
                if isinstance(d, list) and len(d) > 0:
                    has_structured_rows = True

            if has_structured_rows:
                decision = "mcp"
            else:
                lowered_query = query.lower()
                is_empty_or_blank = any(k in lowered_query for k in ["empty excel", "blank excel", "empty spreadsheet", "blank spreadsheet"])
                if is_empty_or_blank:
                    decision = "mcp"
                else:
                    decision = "hybrid"

        if doc_filters and decision == "direct":
            decision = "retrieve"

        reasoning.append(f"Router decision: '{decision}' (Tool: {tool_name or 'none'})")
        logger.info("Router decision finalized: %s", decision)
        logger.info("PERF: router_decision=%.2fs", time.perf_counter() - node_start)

        corr_id = get_correlation_id()
        tracer.record_span(
            trace_id=corr_id,
            name="router",
            start_time=node_start,
            input_data={"query": query},
            output_data={"decision": decision, "tool_name": tool_name},
        )

        return {
            "routing_decision": decision,
            "tool_name": tool_name,
            "tool_input": tool_input or query,
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

    # --- NODE 3: TOOL CALLER (Supports Local Tools + MCP Client Tools) ---
    def tool_caller_node(state: AgentState) -> dict[str, Any]:
        node_start = time.time()
        tool_name = state.get("tool_name") or "calculator"
        tool_input = state.get("tool_input") or state["query"]
        reasoning = list(state.get("reasoning_path", []))

        logger.info("Agent Tool Caller node executing tool: %s with input: '%s'", tool_name, tool_input)

        if tool_name == "calculator":
            res = calculator_tool(tool_input)
        elif tool_name == "web_search":
            res = web_search_tool(tool_input)
        elif tool_name == "create_excel":
            # The excel tool needs structured data, not a natural-language instruction.
            # 1. If the router already produced a JSON object as tool_input, use it.
            # 2. Otherwise, run a small dedicated extraction call grounded ONLY in the
            #    retrieval context gathered earlier in this run (if any) plus the user's
            #    query, so we don't invent figures that were never retrieved.
            excel_args = _try_parse_json_object(tool_input)
            has_rows = isinstance(excel_args, dict) and isinstance(excel_args.get("data"), list) and excel_args["data"]
            if not has_rows:
                extraction_context = state.get("retrieval_context") or ""
                extraction_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Extract tabular data for an Excel export as strict JSON only, with no prose "
                            "and no markdown code fences. Schema: "
                            '{"filename": string, "sheet_name": string, "data": [{"<column>": <value>, ...}, ...]}. '
                            "Use ONLY figures explicitly present in the provided context and user request; "
                            "never invent or estimate numbers. If no concrete tabular data is available, "
                            'return {"data": []}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"User request: {state['query']}\n\nAvailable context:\n{extraction_context[:4000]}",
                    },
                ]
                extraction_resp = _call_llm(extraction_prompt, fallback="")
                excel_args = _try_parse_json_object(extraction_resp) or {}

            res = client.call_tool(
                "create_excel",
                arguments={
                    "filename": excel_args.get("filename") if isinstance(excel_args, dict) else None,
                    "sheet_name": excel_args.get("sheet_name") if isinstance(excel_args, dict) else None,
                    "data": excel_args.get("data") if isinstance(excel_args, dict) else None,
                },
                max_retries=0,
            )
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

        return {
            "tool_output": res.output,
            "tool_data": res.data,
            "reasoning_path": reasoning,
        }

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

        if not allow_retrieval:
            general_qna_instruction = (
                "You are a helpful general-purpose assistant. "
                "Answer the user's question clearly and directly using your general knowledge "
                "without referencing any internal document context or company identity."
            )
            prompt_messages = [
                {"role": "system", "content": general_qna_instruction},
                {"role": "user", "content": query},
            ]
        else:
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

    # Enforce strict Q&A isolation: never permit retrieval or document scope in Q&A mode
    if mode and mode.strip().lower() in ("q&a", "qa", "qna", "general q&a", "general q&a mode"):
        allow_retrieval = False
        doc_filters = None

    if not allow_retrieval:
        doc_filters = None

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

    active_llm_call_fn = llm_call_fn or default_llm_completion

    try:
        app = create_agent_graph(
            llm_call_fn=active_llm_call_fn,
            vectorstore=active_vectorstore,
            embeddings_fn=active_embeddings_fn,
            mcp_client=mcp_client,
            token_streamer=on_token if stream else None,
            allow_retrieval=allow_retrieval,
        )

        initial_state: AgentState = {
            "query": query.strip(),
            "doc_filters": doc_filters if allow_retrieval else None,
            "routing_decision": "direct",
            "tool_name": None,
            "tool_input": None,
            "retrieval_context": None,
            "tool_output": None,
            "tool_data": None,
            "final_answer": None,
            "reasoning_path": [f"Initialized agent run (correlation_id={corr_id})"],
            "sources": [],
            "error": None,
        }

        final_state = app.invoke(initial_state)

        tool_outputs = {}
        if final_state.get("tool_output"):
            tool_outputs[final_state.get("tool_name") or "tool"] = final_state.get("tool_output")
        if final_state.get("retrieval_context"):
            tool_outputs["retriever"] = final_state.get("retrieval_context")

        tool_data = final_state.get("tool_data")
        if isinstance(tool_data, dict) and tool_data.get("generated_file"):
            tool_outputs["generated_file"] = tool_data["generated_file"]

        final_answer = final_state.get("final_answer") or "No answer generated."
        unique_sources = list(set(final_state.get("sources", [])))

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
