from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.tools import create_excel, web_search


@dataclass
class AgentResponse:
    answer: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


KNOWLEDGE_BASE = [
    "This platform supports MCP-style tool calling.",
    "Available tools: create_excel and web_search.",
]


TOOL_REGISTRY = {
    "create_excel": create_excel,
    "web_search": web_search,
}


def _retrieve_context(user_message: str) -> list[str]:
    message = user_message.lower()
    return [item for item in KNOWLEDGE_BASE if any(word in item.lower() for word in message.split())]


def handle_user_query(message: str, output_dir: str | None = None) -> AgentResponse:
    text = message.lower()
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    context_hits = _retrieve_context(message)

    if "excel" in text or "spreadsheet" in text:
        target_dir = Path(output_dir).resolve() if output_dir else Path.cwd()
        file_path = target_dir / "agent_output.xlsx"
        rows = [
            {"item": "task", "value": "agentic rag chatbot"},
            {"item": "capability", "value": "mcp tool calling"},
        ]
        tool_calls.append({"name": "create_excel", "arguments": {"filename": str(file_path), "rows": rows}})
        try:
            result = TOOL_REGISTRY["create_excel"](filename=str(file_path), rows=rows)
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive path
            result = {"tool": "create_excel", "error": str(exc)}
        tool_results.append(result)

    if "search" in text or "web" in text:
        query = message.replace("search", "").strip() or "agentic rag"
        tool_calls.append({"name": "web_search", "arguments": {"query": query, "max_results": 3}})
        try:
            result = TOOL_REGISTRY["web_search"](query=query, max_results=3)
        except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - defensive path
            result = {"tool": "web_search", "query": query, "results": [], "count": 0, "error": str(exc)}
        tool_results.append(result)

    if tool_results:
        answer = "Completed requested MCP tool call(s)."
    else:
        answer = "I can help with RAG chat, web search, and Excel file creation via MCP tools."

    if context_hits:
        answer = f"{answer} Context: {' | '.join(context_hits)}"

    return AgentResponse(answer=answer, tool_calls=tool_calls, tool_results=tool_results)
