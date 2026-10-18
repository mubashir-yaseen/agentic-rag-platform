"""Streamlit UI Application for Multimodal Multi-Document RAG Platform with FastAPI backend support."""

import base64
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
import torch
torch.classes.__path__ = []

from dotenv import find_dotenv, load_dotenv
import streamlit as st

from rag_platform import RAGSystem, get_settings
from rag_platform.ingestion import ingest_pdf
from rag_platform.vectorstore import SearchFilter, SupabaseVectorStore

load_dotenv(find_dotenv())
settings = get_settings()

SUPABASE_OK = bool(settings.SUPABASE_URL and settings.effective_supabase_key)

try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000/api/v1"))
except Exception:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Multimodal RAG Platform | Developed by Mubashir",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_image_base64(path: str) -> str:
    """Convert local image to Base64 for inline HTML rendering."""
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""


logo_b64 = get_image_base64("logo.png")

USER_AVATAR = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='%23ef4444'><path d='M12 2C6.48 2 2 6.48 2 "
    "12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 4c1.93 0 3.5 1.57 3.5 "
    "3.5S13.93 13 12 13s-3.5-1.57-3.5-3.5S10.07 6 12 6zm0 14c-2.03 0-3.8-.85-5.05-2.2.1-.17 "
    "2.05-1.25 5.05-1.25s4.95 1.08 5.05 1.25C15.8 19.15 14.03 20 12 20z'/></svg>"
)
ASSISTANT_AVATAR = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
    "viewBox='0 0 24 24' fill='%2338bdf8'><path d='M12 2a2 2 0 0 1 2 2v1h1a3 "
    "3 0 0 1 3 3v2h1a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-1v1a3 3 0 0 1-3 3H9a3 3 "
    "0 0 1-3-3v-1H5a2 2 0 0 1-2-2v-6a2 2 0 0 1 2-2h1V8a3 3 0 0 1 3-3h1V4a2 "
    "2 0 0 1 2-2zm-3 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm6 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z'/></svg>"
)

st.markdown(
    """<style>
:root{
    --bg:#0b0f14;
    --card:#11161d;
    --text:#ffffff;
    --muted:#cbd5e1;
    --border:#273244;
    --accent:#ef4444;
}
html, body, [class*="css"]{background: var(--bg) !important;color: var(--text) !important;}
.stApp{background: linear-gradient(180deg, #0b0f14 0%, #111827 45%, #0b0f14 100%);}
section[data-testid="stSidebar"]{background: #0f172a;border-right: 1px solid var(--border);}
.block-container {max-width: 1000px !important; padding-top: 0rem !important; padding-bottom: 0rem !important;}
header[data-testid="stHeader"] {display: none !important;}
div[data-testid="stVerticalBlock"] > div:has(div.sticky-header-marker) {
    position: sticky; top: 0; z-index: 9999; background-color: var(--bg);
}
.header-nav {display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 0.4rem;}
.brand-left {display: flex; align-items: center; gap: 8px; flex: 1;}
.brand-left img {width: 18px; height: 18px; border-radius: 6px; object-fit: cover; border: 1px solid var(--border);}
.brand-author {font-size: 0.75rem; color: var(--muted); white-space: nowrap;}
.header-center-title {font-size: 0.95rem; font-weight: 700; color: var(--text); text-align: center; flex: 2;}
.header-right-empty {flex: 1;}
.stButton > button {
    background: #11161d !important; color: var(--text) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.8rem !important; padding: 0.2rem 0.4rem !important;
}
</style>""",
    unsafe_allow_html=True,
)

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "Knowledge Base"
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None
if "selected_content_type" not in st.session_state:
    st.session_state.selected_content_type = "all"
if "current_company" not in st.session_state:
    st.session_state.current_company = None

mode_mapping = {
    "Knowledge Base": "Multi-Document Knowledge Base",
    "Research": "Company Research Mode",
    "Q&A": "General Q&A Mode",
}


def check_api_server() -> bool:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def call_api_chat(
    query: str,
    doc_id: str = None,
    content_type: str = None,
    company_id: str = None,
    mode: str = None,
    stream: bool = False,
    result_sink: dict = None,
) -> dict:
    try:
        current_mode = mode or st.session_state.get("mode")
    except Exception:
        current_mode = mode

    payload = {"query": query, "stream": stream}
    if current_mode:
        payload["mode"] = current_mode
    if current_mode != "Q&A":
        if doc_id:
            payload["doc_id"] = doc_id
        if content_type and content_type != "all":
            payload["content_type"] = content_type
        if company_id is not None:
            payload["company_id"] = company_id

    try:
        if not stream:
            resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=90.0)
            resp.raise_for_status()
            data = resp.json()
            if result_sink is not None:
                result_sink.update({
                    "sources": data.get("sources", []),
                    "tool_outputs": data.get("tool_outputs", {}),
                    "routing_decision": data.get("routing_decision"),
                })
            return data

        resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=90.0, stream=True)
        resp.raise_for_status()
        collected = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                event = json.loads(line.replace("data:", "", 1).strip())
            except json.JSONDecodeError:
                continue
            if event.get("event") == "token":
                chunk = event.get("text", "")
                collected.append(chunk)
                yield chunk
            elif event.get("event") in {"end", "error"}:
                if event.get("event") == "end":
                    if result_sink is not None:
                        result_sink.update({
                            "sources": event.get("sources", []),
                            "tool_outputs": event.get("tool_outputs", {}),
                            "routing_decision": event.get("routing_decision"),
                        })
                    break
                raise RuntimeError(event.get("error", "Stream ended with an error"))
        if not collected:
            raise RuntimeError("No streamed content was received from the API.")
    except requests.exceptions.Timeout:
        raise RuntimeError("The backend server took too long to respond. Please try again.")


def main():
    st.markdown('<div class="sticky-header-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="header-nav">
            <div class="brand-left">
                {"<img src='data:image/jpeg;base64," + logo_b64 + "' />" if logo_b64 else ""}
                <span class="brand-author">Mubashir</span>
            </div>
            <div class="header-center-title">Multimodal Agentic RAG Platform</div>
            <div class="header-right-empty"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    modes = ["Knowledge Base", "Research", "Q&A"]
    mode_cols = st.columns([1, 1, 0.8])
    for idx, m in enumerate(modes):
        with mode_cols[idx]:
            is_active = st.session_state.mode == m
            if st.button(m, key=f"mode_tab_{m}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.mode = m
                st.rerun()

    st.divider()
    chat_container = st.container(height=500, border=False)

    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
            with st.chat_message(role, avatar=avatar):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                placeholder = st.empty()
                try:
                    stream_meta: dict = {}
                    stream_payload = call_api_chat(query=prompt, stream=True, result_sink=stream_meta)
                    answer_parts = []
                    for chunk in stream_payload:
                        answer_parts.append(chunk)
                        placeholder.markdown("".join(answer_parts))
                    answer = "".join(answer_parts)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as exc:
                    placeholder.error(f"Error generating response: {exc}")

    with st.sidebar:
        st.markdown("### Model Controls")
        st.selectbox(
            "LLM Model",
            [
                "openrouter/free",
                "nvidia/nemotron-3.5-lightning:free",
                "nvidia/nemotron-3-ultra-550b-a55b:free",
                "nvidia/nemotron-3-super-120b-a12b:free",
                "nvidia/nemotron-3.5-super-vl:free",
                "google/gemma-4-26b-a4b-it:free",
            ],
            key="llm_model",
        )


if __name__ == "__main__":
    main()
