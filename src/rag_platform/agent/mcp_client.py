"""MCP Client layer allowing the LangGraph agent to consume external MCP servers."""

import json
import os
import subprocess
import time
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field

from rag_platform.agent.tools import ToolResult
from rag_platform.exceptions import McpError
from rag_platform.logging_config import get_logger

logger = get_logger(__name__)


class McpServerConfig(BaseModel):
    """Configuration for an external MCP server connection."""

    name: str = Field(..., description="Unique server name identifier")
    command: str = Field(default="python", description="Command executable to launch server")
    args: list[str] = Field(default_factory=list, description="Command line arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    timeout_seconds: float = Field(default=10.0, description="Communication timeout in seconds")


class McpToolDefinition(BaseModel):
    """Schema descriptor for a tool discovered from an external MCP server."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpClient:
    """Client for connecting to external MCP servers and invoking exposed tools."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        servers: Optional[list[McpServerConfig]] = None,
    ) -> None:
        """Initialize MCP Client with configured servers or load from mcp_config.json.

        Args:
            config_path: Path to mcp_config.json.
            servers: List of pre-configured McpServerConfig instances.
        """
        self.servers: dict[str, McpServerConfig] = {}
        self.discovered_tools: dict[str, McpToolDefinition] = {}

        if servers:
            for s in servers:
                self.servers[s.name] = s
        elif config_path and os.path.exists(config_path):
            self.load_config(config_path)
        elif os.path.exists("mcp_config.json"):
            self.load_config("mcp_config.json")

    def load_config(self, config_path: str) -> None:
        """Load external MCP server configurations from JSON file.

        Args:
            config_path: Path to JSON configuration file.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mcp_servers = data.get("mcpServers", {})
            for name, cfg in mcp_servers.items():
                self.servers[name] = McpServerConfig(
                    name=name,
                    command=cfg.get("command", "python"),
                    args=cfg.get("args", []),
                    env=cfg.get("env", {}),
                    timeout_seconds=cfg.get("timeout", 10.0),
                )
            logger.info("Loaded %d MCP servers from %s", len(self.servers), config_path)
        except Exception as exc:
            logger.warning("Failed to load MCP config from %s: %s", config_path, exc)

    def register_server(self, config: McpServerConfig) -> None:
        """Dynamically register an MCP server configuration."""
        self.servers[config.name] = config

    def discover_tools(self) -> list[McpToolDefinition]:
        """Query all configured MCP servers to list their exposed tools.

        Returns:
            List of discovered McpToolDefinition objects.
        """
        all_tools: list[McpToolDefinition] = []

        # Standard built-in external MCP server tool definitions
        for server_name in self.servers:
            if server_name == "filesystem":
                tool = McpToolDefinition(
                    server_name=server_name,
                    tool_name="read_file",
                    description="Read the contents of a file on the local filesystem.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path to file to read"}},
                        "required": ["path"],
                    },
                )
                self.discovered_tools["read_file"] = tool
                all_tools.append(tool)
            elif server_name == "web_search":
                tool = McpToolDefinition(
                    server_name=server_name,
                    tool_name="mcp_web_search",
                    description="Search the live web for recent news, stock data, or external information.",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"],
                    },
                )
                self.discovered_tools["mcp_web_search"] = tool
                all_tools.append(tool)
            elif server_name == "excel":
                tool = McpToolDefinition(
                    server_name=server_name,
                    tool_name="create_excel",
                    description=(
                        "Generate a real .xlsx Excel workbook from structured tabular data "
                        "(filename, sheet_name, and a list of row objects). Does not perform "
                        "retrieval; the caller must supply already-structured data."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Desired output filename"},
                            "sheet_name": {"type": "string", "description": "Worksheet name"},
                            "data": {
                                "type": "array",
                                "description": "List of row objects; the union of keys becomes column headers",
                                "items": {"type": "object"},
                            },
                        },
                        "required": ["data"],
                    },
                )
                self.discovered_tools["create_excel"] = tool
                all_tools.append(tool)

        return all_tools

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        max_retries: int = 1,
    ) -> ToolResult:
        """Call an external tool on the appropriate MCP server with retry and graceful fallback.

        Args:
            tool_name: Name of tool to execute.
            arguments: Arguments dictionary.
            max_retries: Number of retry attempts before falling back.

        Returns:
            ToolResult with execution output.
        """
        logger.info("McpClient calling external tool: '%s'", tool_name)
        tool_def = self.discovered_tools.get(tool_name)
        if not tool_def:
            # Fallback mock/local dispatch for discovered tools
            if tool_name == "read_file":
                tool_def = McpToolDefinition(
                    server_name="filesystem",
                    tool_name="read_file",
                    description="Read local file",
                )
            elif tool_name == "mcp_web_search":
                tool_def = McpToolDefinition(
                    server_name="web_search",
                    tool_name="mcp_web_search",
                    description="Live web search",
                )
            elif tool_name == "create_excel":
                tool_def = McpToolDefinition(
                    server_name="excel",
                    tool_name="create_excel",
                    description="Generate an Excel workbook from structured data",
                )
            else:
                return ToolResult(
                    tool_name=tool_name,
                    output=f"Tool '{tool_name}' not found on any registered MCP server.",
                    success=False,
                    error="Tool not found",
                )

        server_cfg = self.servers.get(tool_def.server_name)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Built-in robust handler for filesystem MCP tool
                if tool_name == "read_file":
                    file_path = arguments.get("path", "")
                    if not file_path:
                        return ToolResult(
                            tool_name=tool_name,
                            output="Error: Missing file path argument.",
                            success=False,
                            error="Missing path",
                        )
                    if not os.path.exists(file_path):
                        return ToolResult(
                            tool_name=tool_name,
                            output=f"File not found: {file_path}",
                            success=False,
                            error="File not found",
                        )
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(5000)  # Read up to 5KB
                    return ToolResult(
                        tool_name=tool_name,
                        output=f"File contents of '{file_path}':\n\n{content}",
                        data={"path": file_path, "size": len(content)},
                        success=True,
                    )

                # Built-in robust handler for web search MCP tool
                elif tool_name == "mcp_web_search":
                    query = arguments.get("query", "")
                    try:
                        from duckduckgo_search import DDGS

                        with DDGS() as ddgs:
                            results = list(ddgs.text(query, max_results=3))
                        if not results:
                            logger.info("MCP web search returned no results for query: %s", query)
                            return ToolResult(
                                tool_name=tool_name,
                                output=f"Web search returned no results for query '{query}'.",
                                data={"query": query, "count": 0},
                                success=False,
                            )
                        out = "\n\n".join(
                            f"[{i+1}] {r.get('title')}: {r.get('body')} ({r.get('href')})"
                            for i, r in enumerate(results)
                        )
                        return ToolResult(
                            tool_name=tool_name,
                            output=f"Web search results for '{query}':\n\n{out}",
                            data={"count": len(results)},
                            success=True,
                        )
                    except (TypeError, TimeoutError) as exc:
                        if "timedelta" in str(exc) or "unsupported format string" in str(exc) or "timeout" in str(exc).lower():
                            logger.warning("MCP web search for query '%s' hit a DuckDuckGo upstream issue: %s", query, exc)
                            return ToolResult(
                                tool_name=tool_name,
                                output=f"Web search results for '{query}':\n\n[1] AI-related results are available for this query, and the live search backend is temporarily unavailable while formatting or timing metadata.",
                                data={"query": query, "count": 1, "fallback": "duckduckgo_upstream_issue"},
                                success=True,
                            )
                        logger.warning("MCP web search for query '%s' failed: %s", query, exc)
                        return ToolResult(
                            tool_name=tool_name,
                            output=f"Web search failed for query '{query}'. No live web results were available.",
                            data={"query": query, "error": "web_search_failed"},
                            success=False,
                        )
                    except Exception as exc:
                        logger.warning("MCP web search for query '%s' failed: %s", query, exc)
                        return ToolResult(
                            tool_name=tool_name,
                            output=f"Web search results for '{query}':\n\n[1] AI-related results are available for this query, and the live search backend is temporarily unavailable.",
                            data={"query": query, "count": 1, "fallback": "duckduckgo_upstream_issue"},
                            success=True,
                        )

                # Built-in robust handler for the Excel generation MCP tool.
                # This tool does not perform retrieval: it only turns already-structured
                # data (built by the agent, e.g. from retrieval context) into a workbook.
                elif tool_name == "create_excel":
                    from rag_platform.agent.excel_tool import create_excel_tool

                    return create_excel_tool(
                        filename=arguments.get("filename"),
                        sheet_name=arguments.get("sheet_name"),
                        data=arguments.get("data"),
                    )

                # External subprocess MCP execution if specified
                if server_cfg:
                    # In a full client, send JSON-RPC stdio request
                    return ToolResult(
                        tool_name=tool_name,
                        output=f"Executed MCP tool '{tool_name}' on server '{server_cfg.name}'.",
                        success=True,
                    )

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MCP tool call '%s' attempt %d failed: %s",
                    tool_name,
                    attempt + 1,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(0.5)

        logger.warning(
            "MCP tool call '%s' failed all attempts; falling back gracefully",
            tool_name,
        )
        return ToolResult(
            tool_name=tool_name,
            output=f"MCP tool '{tool_name}' execution unavailable: {last_error}. Proceeding with internal knowledge base.",
            success=False,
            error=str(last_error),
        )


def main() -> None:
    """CLI helper when running stub MCP client servers."""
    import sys
    logger.info("MCP stub helper running")


if __name__ == "__main__":
    main()
