"""MCP (Model Context Protocol) subpackage for exposing the RAG platform as a server and consuming tools as a client."""

from rag_platform.mcp.server import McpServer, create_mcp_app, main

__all__ = [
    "McpServer",
    "create_mcp_app",
    "main",
]
