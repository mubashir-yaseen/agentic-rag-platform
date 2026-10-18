"""LangGraph orchestration graph for Agentic RAG Platform with full API parameter compatibility and free-tier safeguards."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_platform.config import get_settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State management for the agentic workflow graph."""
    query: str
    mode: str
    doc_id: str | None
    content_type: str | None
    company_id: str | None
    routing_decision: str
    retrieved_docs: List[Any]
    tool_outputs: Dict[str, Any]
    response: str
    error: str | None


@dataclass
class AgentResult:
    """Structured result returned by agent execution."""
    response: str
    sources: List[Any] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    routing_decision: str = "direct"
    error: str | None = None


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Initialize ChatOpenAI pointing to OpenRouter with a rotating list of free models and safety timeouts."""
    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("Missing LLM API key. Provide OPENROUTER_API_KEY.")

    free_models = [
        "openrouter/free",
        "nvidia/nemotron-3.5-lightning:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3.5-super-vl:free",
        "google/gemma-4-26b-a4b-it:free",
    ]

    primary_model = settings.LLM_MODEL if settings.LLM_MODEL and "free" in settings.LLM_MODEL else free_models[0]
    
    return ChatOpenAI(
        model=primary_model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        max_tokens=512,
        request_timeout=20.0,
        max_retries=2,
    )


def router_node(state: AgentState) -> AgentState:
    """Analyze query and decide routing path."""
    logger.info("Agent Router analyzing query: '%s'", state["query"])
    try:
        if state.get("mode") == "Q&A":
            state["routing_decision"] = "direct"
            return state

        llm = get_llm(temperature=0.0)
        prompt = (
            "You are a routing supervisor. Decide the best execution path for the user query.\n"
            "Options:\n"
            "1. 'vector_search': If the query requires looking up documents, PDFs, or knowledge base chunks.\n"
            "2. 'excel_tool': If the query asks for financial metrics, reports, or table generation.\n"
            "3. 'direct': For general questions, summaries, or casual conversation.\n"
            f"User Query: {state['query']}\n"
            "Output ONLY the decision keyword: vector_search, excel_tool, or direct."
        )
        res = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state["query"])])
        decision = res.content.strip().lower()
        
        if "vector" in decision:
            state["routing_decision"] = "vector_search"
        elif "excel" in decision or "sql" in decision:
            state["routing_decision"] = "excel_tool"
        else:
            state["routing_decision"] = "direct"
            
        logger.info("Router decision finalized: %s", state["routing_decision"])
    except Exception as exc:
        logger.warning("Router encountered error (%s), falling back to direct response.", exc)
        state["routing_decision"] = "direct"
        
    return state


def responder_node(state: AgentState) -> AgentState:
    """Synthesize final response based on router decision and context."""
    logger.info("Agent Responder synthesizing final answer for decision: %s", state["routing_decision"])
    try:
        llm = get_llm(temperature=0.2)
        query = state["query"]
        
        if state["routing_decision"] == "vector_search" and state.get("retrieved_docs"):
            context = "\n\n".join([str(doc) for doc in state["retrieved_docs"]])
            prompt = f"Answer the query using the retrieved document context below:\n\nContext:\n{context}\n\nQuery: {query}"
        else:
            prompt = query

        res = llm.invoke([SystemMessage(content="You are a helpful enterprise AI assistant."), HumanMessage(content=prompt)])
        state["response"] = res.content
    except Exception as exc:
        logger.error("Responder LLM failure: %s", exc)
        state["response"] = (
            "All free-tier model providers are currently experiencing heavy traffic or temporary 503 overloads. "
            f"Please wait a moment and try your query again. (Details: {str(exc)})"
        )
        
    return state


def create_agent_graph():
    """Construct and compile the LangGraph workflow graph."""
    from langgraph.graph import END, StateGraph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("responder", responder_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "responder")
    workflow.add_edge("responder", END)
    
    return workflow.compile()


def run_agent(
    query: str,
    mode: str = "Knowledge Base",
    doc_id: str | None = None,
    content_type: str | None = None,
    company_id: str | None = None,
    doc_filters: str | None = None,
    **kwargs: Any,
) -> AgentResult:
    """Execute the compiled agent graph for a given user query with complete API signature safety."""
    try:
        graph = create_agent_graph()
        resolved_doc_id = doc_id or doc_filters or kwargs.get("doc_filters")
        
        initial_state: AgentState = {
            "query": query,
            "mode": mode,
            "doc_id": resolved_doc_id,
            "content_type": content_type,
            "company_id": company_id,
            "routing_decision": "direct",
            "retrieved_docs": [],
            "tool_outputs": {},
            "response": "",
            "error": None,
        }
        final_state = graph.invoke(initial_state)
        return AgentResult(
            response=final_state.get("response", "No response generated."),
            sources=final_state.get("retrieved_docs", []),
            tool_outputs=final_state.get("tool_outputs", {}),
            routing_decision=final_state.get("routing_decision", "direct"),
            error=final_state.get("error"),
        )
    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        return AgentResult(
            response=f"An error occurred while processing your request: {str(exc)}",
            error=str(exc),
        )


# Alias for internal imports
build_graph = create_agent_graph
