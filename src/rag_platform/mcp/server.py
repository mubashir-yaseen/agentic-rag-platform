"""Model Context Protocol (MCP) Server for the RAG platform.

Exposes the knowledge base and agent retrieval capabilities as standardized
MCP tools for clients like Claude Desktop, Claude Code, and Cursor.
"""

import json
from typing import Any, Callable, Optional

from rag_platform.agent.graph import run_agent
from rag_platform.config import get_settings
from rag_platform.exceptions import RagPlatformError
from rag_platform.logging_config import get_logger
from rag_platform.vectorstore.supabase_store import SupabaseVectorStore

logger = get_logger(__name__)


class McpServer:
    """Manages MCP tool definitions, execution dispatch, and protocol handling."""

    def __init__(
        self,
        name: str = "rag-platform-server",
        vectorstore: Optional[SupabaseVectorStore] = None,
        embeddings_fn: Optional[Callable[[str], list[float]]] = None,
        llm_call_fn: Optional[Callable[[list[dict[str, str]]], str]] = None,
    ) -> None:
        """Initialize the MCP server instance.

        Args:
            name: Identifier for the server instance.
            vectorstore: Optional SupabaseVectorStore instance.
            embeddings_fn: Optional embeddings generator callable.
            llm_call_fn: Optional LLM completion callable.
        """
        self.name = name
        self.vectorstore = vectorstore
        self.embeddings_fn = embeddings_fn
        self.llm_call_fn = llm_call_fn

    def _ensure_vectorstore(self) -> SupabaseVectorStore:
        """Lazy-initialize SupabaseVectorStore if not injected."""
        if self.vectorstore is None:
            self.vectorstore = SupabaseVectorStore()
        return self.vectorstore

    def _ensure_embeddings(self) -> Callable[[str], list[float]]:
        """Lazy-initialize embeddings callable if not injected."""
        if self.embeddings_fn is None:
            from rag_platform.vectorstore.embeddings import get_embedding_model

            hf = get_embedding_model()
            self.embeddings_fn = hf.embed_query
        return self.embeddings_fn

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP tool schemas adhering to the Model Context Protocol specification."""
        return [
            {
                "name": "query_knowledge_base",
                "description": (
                    "Query the multimodal multi-document RAG knowledge base. "
                    "Searches indexed PDF text, structured tables, and figure captions in Supabase pgvector, "
                    "and synthesizes a grounded answer with citations."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question or search query to run against the knowledge base.",
                        },
                        "doc_filter": {
                            "type": "string",
                            "description": "Optional document ID or filename to scope the search to.",
                        },
                        "content_type": {
                            "type": "string",
                            "enum": ["text", "table", "image"],
                            "description": "Optional filter for content type (text prose, tabular data, or images).",
                        },
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "list_documents",
                "description": (
                    "List all documents registered and indexed in the RAG knowledge base, "
                    "including document IDs, filenames, upload timestamps, and page metadata."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def query_knowledge_base(
        self,
        question: str,
        doc_filter: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """Execute knowledge base search and agent synthesis.

        Args:
            question: User question.
            doc_filter: Optional document ID or filename filter.
            content_type: Optional content type filter (text, table, image).

        Returns:
            Grounded textual answer with citations.
        """
        logger.info("MCP query_knowledge_base called with question: '%s'", question)
        if not question or not question.strip():
            return "Error: Question parameter cannot be empty."

        try:
            vs = self._ensure_vectorstore()
            emb_fn = self._ensure_embeddings()

            filters: dict[str, Any] = {}
            if doc_filter:
                filters["doc_id"] = doc_filter
            if content_type:
                filters["content_type"] = content_type

            agent_result = run_agent(
                query=question.strip(),
                doc_filters=filters if filters else None,
                vectorstore=vs,
                embeddings_fn=emb_fn,
                llm_call_fn=self.llm_call_fn,
            )

            sources_text = ""
            if agent_result.sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in agent_result.sources)

            return f"{agent_result.answer}{sources_text}"

        except RagPlatformError as exc:
            logger.warning("MCP query_knowledge_base domain error: %s", exc)
            return f"Error querying knowledge base: {exc.message}"
        except Exception as exc:
            logger.exception("Unexpected error in MCP query_knowledge_base")
            return f"Internal error during knowledge base query: {exc}"

    def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents with metadata.

        Returns:
            List of document metadata dictionaries.
        """
        logger.info("MCP list_documents called")
        try:
            vs = self._ensure_vectorstore()
            records = vs.list_documents()
            return [
                {
                    "doc_id": r.doc_id,
                    "filename": r.filename,
                    "uploaded_at": str(r.uploaded_at),
                    "doc_type": r.doc_type,
                    "metadata": r.metadata,
                }
                for r in records
            ]
        except RagPlatformError as exc:
            logger.warning("MCP list_documents domain error: %s", exc)
            return [{"error": exc.message}]
        except Exception as exc:
            logger.exception("Unexpected error in MCP list_documents")
            return [{"error": str(exc)}]

    def handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an MCP tool call request and format the standard MCP JSON response.

        Args:
            tool_name: Name of tool to execute.
            arguments: Tool arguments dictionary.

        Returns:
            MCP tool result response dictionary.
        """
        logger.info("Handling MCP tool call: '%s'", tool_name)
        if tool_name == "query_knowledge_base":
            question = arguments.get("question", "")
            doc_filter = arguments.get("doc_filter")
            content_type = arguments.get("content_type")
            res_str = self.query_knowledge_base(question, doc_filter=doc_filter, content_type=content_type)
            return {
                "content": [{"type": "text", "text": res_str}],
                "isError": res_str.startswith("Error:"),
            }
        elif tool_name == "list_documents":
            docs = self.list_documents()
            return {
                "content": [{"type": "text", "text": json.dumps(docs, indent=2)}],
                "isError": len(docs) == 1 and "error" in docs[0],
            }
        else:
            return {
                "content": [{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}],
                "isError": True,
            }


def create_mcp_app() -> Any:
    """Build official MCP FastMCP/Server application when official SDK is used."""
    try:
        from mcp.server.fastmcp import FastMCP

        mcp_app = FastMCP("rag-platform-server")
        server = McpServer()

        @mcp_app.tool()
        def query_knowledge_base(
            question: str,
            doc_filter: Optional[str] = None,
            content_type: Optional[str] = None,
        ) -> str:
            """Query the multimodal multi-document RAG knowledge base."""
            return server.query_knowledge_base(question, doc_filter=doc_filter, content_type=content_type)

        @mcp_app.tool()
        def list_documents() -> str:
            """List all documents registered in the RAG knowledge base."""
            docs = server.list_documents()
            return json.dumps(docs, indent=2)

        return mcp_app
    except ImportError:
        logger.warning("Official 'mcp' SDK not installed; using pure McpServer class.")
        return McpServer()


def main() -> None:
    """Entry point to run the MCP Server over standard I/O (stdio)."""
    import sys

    logger.info("Starting RAG Platform MCP Server over stdio")
    try:
        from mcp.server.fastmcp import FastMCP

        app = create_mcp_app()
        if isinstance(app, FastMCP):
            app.run(transport="stdio")
            return
    except Exception as exc:
        logger.debug("FastMCP stdio transport not started via SDK: %s", exc)

    # Pure stdio JSON-RPC fallback handler
    server = McpServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": server.get_tool_definitions()}}
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                result = server.handle_tool_call(tool_name, args)
                resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
            elif method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "rag-platform-server", "version": "0.5.0"},
                        "capabilities": {"tools": {}},
                    },
                }
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
