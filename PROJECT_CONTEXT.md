\# RAG Platform — Project Context



\## Purpose



This project is an evolution of an older RAG chatbot into a production-style

Agentic RAG platform.



The goal is not simply document Q\&A. The platform provides:



\- Agentic query routing

\- Hybrid retrieval

\- Supabase/pgvector

\- LLM-based grounded responses

\- MCP tool integration

\- FastAPI backend

\- Streamlit UI

\- Observability

\- Testing

\- Docker/CI/CD readiness



\## Current Architecture



User

&#x20; ↓

Streamlit UI

&#x20; ↓

FastAPI

&#x20; ↓

LangGraph Agent

&#x20; ↓

Router

&#x20; ↓

Retrieval Tool

&#x20; ↓

Hybrid Retrieval

&#x20; ├── Dense vector search

&#x20; ├── Lexical matching

&#x20; ├── TOC-aware ranking

&#x20; ├── Prose heuristics

&#x20; └── Cross-document deduplication

&#x20; ↓

LLM

&#x20; ↓

Grounded Answer + Sources



MCP is also part of the platform architecture and needs final end-to-end

validation.



\## Important Existing Implementation



Project source is under:



src/rag\_platform/



Important areas include:



\- src/rag\_platform/config.py

\- src/rag\_platform/agent/

\- src/rag\_platform/api/

\- src/rag\_platform/...

\- ui/app.py

\- tests/



\## Configuration



Configuration is centralized using pydantic-settings and .env.



Do NOT reintroduce Streamlit secrets such as:



st.secrets



Backend configuration must continue using the centralized configuration.



Supabase key fallback currently supports:



SUPABASE\_SERVICE\_ROLE\_KEY

→ SUPABASE\_ANON\_KEY

→ SUPABASE\_KEY



\## Current Working Backend



Correct FastAPI application:



src/rag\_platform/api/main.py



The application is:



app = create\_app()



Correct command from project root:



$env:PYTHONPATH="src;."

python -m uvicorn rag\_platform.api.main:app --reload --port 8000



Health endpoint:



GET http://127.0.0.1:8000/api/v1/health



Current health endpoint has been verified successfully.



Chat endpoint:



POST http://127.0.0.1:8000/api/v1/chat



This endpoint has also been verified successfully.



Example:



{

&#x20; "query": "What is the proposed research problem?",

&#x20; "stream": false

}



The API currently correctly routes the question to retrieval, retrieves

relevant chunks, calls the LLM, and returns a grounded answer with sources,

reasoning\_path, routing\_decision, tool\_outputs, and correlation\_id.



\## Retrieval



The retrieval system was significantly improved because pure vector search

was returning irrelevant table-of-contents chunks.



Current hybrid retrieval includes:



\- dense similarity

\- lexical relevance

\- TOC detection/penalty

\- prose heuristics

\- cross-document deduplication

\- larger lexical scanning

\- configurable retrieval k



Current default retrieval k is 6.



The thesis research-problem query has been verified to retrieve the correct

Page 3 content.



\## Verified Queries



These have been tested successfully:



1\. What is the proposed research problem?

2\. What datasets are mentioned in the proposal?

3\. What is the starting date mentioned for baseline VLM implementation?



\## Testing



Agent/API scoped tests currently pass:



29 tests passed.



Earlier broader testing also reached 91 passed with 7 warnings before later

retrieval refinements.



Do not remove existing tests.



\## Current Status



Completed/core working:



\- Baseline audit

\- Engineering/configuration cleanup

\- Core ingestion

\- Supabase/pgvector

\- Hybrid retrieval

\- LangGraph agent/router

\- Core FastAPI API

\- Basic observability

\- Automated tests



Partially complete / remaining:



1\. Streamlit end-to-end verification

2\. MCP end-to-end verification

3\. Formal RAG evaluation

4\. Docker/containerization

5\. CI/CD

6\. Final documentation and cleanup



\## Important Development Rule



DO NOT rewrite the existing architecture.



DO NOT replace working retrieval logic.



DO NOT make broad refactors unless explicitly requested.



Before changing code:



1\. Inspect the existing implementation.

2\. Explain what is already present.

3\. Identify the smallest required change.

4\. Make only the required change.

5\. Run relevant tests.

6\. Report exactly what changed and what was verified.



Prefer minimal, backwards-compatible changes.



Never claim something is implemented or working without actually inspecting

and/or testing it.

