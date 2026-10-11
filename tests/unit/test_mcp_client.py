"""Unit tests for McpClient and agent MCP tool integrations."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from rag_platform.agent.graph import AgentResult, run_agent
from rag_platform.agent.mcp_client import McpClient, McpServerConfig, McpToolDefinition


@pytest.fixture
def sample_mcp_config(tmp_path):
    """Create a temporary mcp_config.json file."""
    config_file = tmp_path / "mcp_config.json"
    data = {
        "mcpServers": {
            "filesystem": {
                "command": "python",
                "args": ["-m", "rag_platform.agent.mcp_client", "--server", "filesystem"],
                "description": "Local filesystem MCP server",
            },
            "web_search": {
                "command": "python",
                "args": ["-m", "rag_platform.agent.mcp_client", "--server", "web_search"],
                "description": "Web search MCP server",
            },
        }
    }
    config_file.write_text(json.dumps(data))
    return str(config_file)


# --- MCP CLIENT UNIT TESTS ---


@pytest.mark.unit
def test_mcp_client_load_config(sample_mcp_config):
    """Verify McpClient parses server configurations from JSON file."""
    client = McpClient(config_path=sample_mcp_config)
    assert len(client.servers) == 2
    assert "filesystem" in client.servers
    assert "web_search" in client.servers


@pytest.mark.unit
def test_mcp_client_discover_tools(sample_mcp_config):
    """Verify discover_tools registers exposed tools from configured servers."""
    client = McpClient(config_path=sample_mcp_config)
    tools = client.discover_tools()

    tool_names = [t.tool_name for t in tools]
    assert "read_file" in tool_names
    assert "mcp_web_search" in tool_names


@pytest.mark.unit
def test_mcp_client_call_read_file_tool(tmp_path):
    """Verify calling read_file MCP tool retrieves local file content."""
    test_file = tmp_path / "sample_doc.txt"
    test_file.write_text("Confidential NDA Agreement content for testing MCP tool.")

    client = McpClient(servers=[McpServerConfig(name="filesystem")])
    client.discover_tools()

    res = client.call_tool("read_file", {"path": str(test_file)})
    assert res.success is True
    assert "Confidential NDA Agreement" in res.output


@pytest.mark.unit
def test_mcp_client_call_read_file_missing_path():
    """Verify read_file tool returns error when path does not exist."""
    client = McpClient(servers=[McpServerConfig(name="filesystem")])
    client.discover_tools()

    res = client.call_tool("read_file", {"path": "non_existent_file_xyz.txt"})
    assert res.success is False
    assert "File not found" in res.output


@pytest.mark.unit
def test_mcp_client_call_web_search_tool():
    """Verify calling mcp_web_search tool executes with clean output."""
    client = McpClient(servers=[McpServerConfig(name="web_search")])
    client.discover_tools()

    res = client.call_tool("mcp_web_search", {"query": "latest AI news 2026"})
    assert res.success is True
    assert "AI" in res.output or "results" in res.output.lower()


@pytest.mark.unit
def test_mcp_client_graceful_fallback_on_failure():
    """Verify McpClient returns graceful fallback ToolResult when tool fails."""
    client = McpClient()
    res = client.call_tool("unknown_tool", {"arg": "val"})
    assert res.success is False
    assert "not found" in res.output.lower()


# --- AGENT MCP INTEGRATION TESTS ---


@pytest.mark.unit
def test_agent_routes_to_mcp_filesystem_tool(tmp_path):
    """Verify agent routes file reading queries to MCP read_file tool."""
    test_doc = tmp_path / "contracts.txt"
    test_doc.write_text("Standard Consulting Terms and Conditions 2026.")
    doc_path = str(test_doc).replace("\\", "/")

    client = McpClient(servers=[McpServerConfig(name="filesystem")])

    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return json.dumps({
                    "decision": "mcp",
                    "tool_name": "read_file",
                    "tool_input": str(test_doc),
                })
        return "The document contains the standard consulting terms."

    result = run_agent(
        query=f"Please read file {doc_path} and summarize it",
        llm_call_fn=mock_llm,
        mcp_client=client,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "mcp"
    assert "read_file" in result.tool_outputs
    assert "Standard Consulting Terms" in result.tool_outputs["read_file"]


@pytest.mark.unit
def test_agent_routes_to_mcp_web_search_tool():
    """Verify agent routes live external queries to MCP web search tool."""
    client = McpClient(servers=[McpServerConfig(name="web_search")])

    def mock_llm(messages):
        for m in messages:
            if "Classify the user query" in m.get("content", ""):
                return '{"decision": "mcp", "tool_name": "mcp_web_search", "tool_input": "NVIDIA stock price today"}'
        return "NVIDIA is currently trading with strong volume."

    result = run_agent(
        query="What is the latest news about NVIDIA stock price today?",
        llm_call_fn=mock_llm,
        mcp_client=client,
    )

    assert isinstance(result, AgentResult)
    assert result.routing_decision == "mcp"
    assert "mcp_web_search" in result.tool_outputs
