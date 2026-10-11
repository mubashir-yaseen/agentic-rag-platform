from fastapi.testclient import TestClient

from app import agent
from app.main import app

client = TestClient(app)


def test_chat_invokes_excel_tool() -> None:
    response = client.post("/chat", json={"message": "Please create an excel file"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_calls"]
    assert payload["tool_calls"][0]["name"] == "create_excel"


def test_chat_invokes_web_search_tool(monkeypatch) -> None:
    def fake_web_search(query: str, max_results: int = 5):
        return {
            "tool": "web_search",
            "query": query,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": "demo"}],
            "count": 1,
        }

    monkeypatch.setitem(agent.TOOL_REGISTRY, "web_search", fake_web_search)

    response = client.post("/chat", json={"message": "search latest rag frameworks"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_calls"]
    assert payload["tool_calls"][0]["name"] == "web_search"
    assert payload["tool_results"][0]["count"] == 1


def test_chat_handles_web_search_failure(monkeypatch) -> None:
    def failing_web_search(query: str, max_results: int = 5):
        raise RuntimeError("network unavailable")

    monkeypatch.setitem(agent.TOOL_REGISTRY, "web_search", failing_web_search)

    response = client.post("/chat", json={"message": "search agentic rag"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_results"][0]["count"] == 0
    assert payload["tool_results"][0]["error"] == "network unavailable"
