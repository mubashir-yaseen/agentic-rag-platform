# agentic-rag-platform

Agentic RAG chatbot with MCP tool calling for Excel file creation and web search.

## Setup

```bash
pip install -e .[dev]
```

## Run

```bash
uvicorn app.main:app --reload
```

## API

- `GET /health`
- `POST /chat` with JSON body:

```json
{
  "message": "create an excel file"
}
```

## Test and Lint

```bash
pytest -q
ruff check .
```
