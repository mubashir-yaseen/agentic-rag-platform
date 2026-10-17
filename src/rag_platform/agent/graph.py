"""LangGraph orchestration graph for Agentic RAG Platform with explicit timeout safeguards."""

import json
import logging
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from rag_platform.config import get_settings
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


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


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Initialize ChatOpenAI pointing to OpenRouter with hardcoded safety timeouts."""
    settings = get_settings()
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("Missing LLM API key. Provide OPENROUTER_API_KEY.")

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        max_tokens=1024,  # Prevents runaway generation loops
        request_timeout=30.0,  # Fails fast instead of hanging for 400+ seconds
        max_retries=1,  # Avoids compounding fallback delays
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
            "2. 'sql_tool' or 'excel_tool': If the query asks for financial metrics, reports, or table generation.\n"
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
        state["response"] = f"I encountered a temporary provider timeout while generating your response. Please try again. (Error: {str(exc)})"
        
    return state


def build_graph():
    """Construct the compiled LangGraph workflow."""
    from langgraph.graph import END, StateGraph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("responder", responder_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "responder")
    workflow.add_edge("responder", END)
    
    return workflow.compile()
