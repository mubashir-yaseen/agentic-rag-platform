def call_api_chat(
    query: str,
    doc_id: str = None,
    content_type: str = None,
    company_id: str = None,
    mode: str = None,
    stream: bool = False,
    result_sink: dict = None,
) -> dict:
    """Send query to FastAPI /chat endpoint with a safe 120-second timeout."""
    # ... existing payload setup ...
    
    if not stream:
        resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()

    resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120.0, stream=True)
    resp.raise_for_status()
    # ... rest of stream handling ...
