from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import handle_user_query

app = FastAPI(title="Agentic RAG Platform")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    response = handle_user_query(request.message)
    return {
        "answer": response.answer,
        "tool_calls": response.tool_calls,
        "tool_results": response.tool_results,
    }
