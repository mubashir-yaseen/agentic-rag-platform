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

# Robustly read API_BASE_URL from Streamlit Secrets first, then environment variables, falling back to localhost
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

.block-container {
    max-width: 1000px !important;
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}

div[data-testid="stVerticalBlock"] > div:has(div.sticky-header-marker) {
    position: sticky;
    top: 0;
    z-index: 9999;
    background-color: var(--bg);
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    margin-bottom: 0rem !important;
}

.header-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 0.4rem;
}
.brand-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}
.brand-left img {
    width: 18px;
    height: 18px;
    border-radius: 6px;
    object-fit: cover;
    border: 1px solid var(--border);
}
.brand-author {
    font-size: 0.75rem;
    color: var(--muted);
    white-space: nowrap;
}
.header-center-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    text-align: center;
    flex: 2;
    letter-spacing: 0.5px;
}
.header-right-empty {
    flex: 1;
}

.stButton > button {
    background: #11161d !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.2rem 0.4rem !important;
    min-height: 30px !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {border-color: var(--accent) !important;background: #1c2330 !important;}
div.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border-color: #991b1b !important;
}
div.stButton > button[kind="primary"]:hover {background: #dc2626 !important;border-color: #991b1b !important;}

[data-testid="stFileUploader"] {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    padding: 0.2rem 0.5rem !important;
    min-height: 40px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    padding: 0.15rem 0.4rem !important;
    font-size: 0.75rem !important;
    min-height: 26px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    font-size: 0.75rem !important;
}

[data-testid="stSelectbox"] {
    margin-top: 0.2rem !important;
    margin-bottom: 0rem !important;
}
[data-testid="stSelectbox"] > div > div {
    min-height: 30px !important;
    padding-top: 0px !important;
    padding-bottom: 0px !important;
    font-size: 0.8rem !important;
}

hr {
    margin-top: 0.25rem !important;
    margin-bottom: 0.25rem !important;
}

div[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border-bottom: 1px solid rgba(39, 50, 68, 0.4);
    padding: 12px 8px !important;
    border-radius: 0px !important;
}
div[data-testid="stChatMessageContent"] {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    font-weight: 400 !important;
}
div[data-testid="stChatMessageContent"] p,
div[data-testid="stChatMessageContent"] li,
div[data-testid="stChatMessageContent"] span {
    color: #f8fafc !important;
}
div[data-testid="stChatMessageContent"] strong {
    color: #ffffff !important;
    font-weight: 700 !important;
}

@media (max-width: 640px) {
    div[data-testid="stChatMessageContent"] {
        font-size: 0.9rem !important;
        line-height: 1.45 !important;
    }
    .header-center-title {
        font-size: 0.85rem !important;
    }
    .brand-author {
        font-size: 0.65rem !important;
    }
}

[data-testid="stChatInput"] {border-top: none !important;background: transparent !important;}
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
if "backend_awake" not in st.session_state:
    st.session_state.backend_awake = False

mode_mapping = {
    "Knowledge Base": "Multi-Document Knowledge Base",
    "Research": "Company Research Mode",
    "Q&A": "General Q&A Mode",
}


def check_api_server(timeout: float = 8.0) -> bool:
    """Check if FastAPI backend service is reachable.

    Render's free tier spins the backend down after ~15 min idle; the next
    request has to wait through a cold start (can take 30-60s), so a short
    timeout here would wrongly report the backend as unreachable and fall
    back to local initialization (which deliberately lacks LLM API keys).
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def wake_backend_if_needed() -> bool:
    """Ping the backend once per session, waiting out a cold start if needed.

    Returns True once the backend responds, False if it never comes up.
    Result is cached in session_state so this only happens once per session,
    not on every single chat message.
    """
    if st.session_state.get("backend_awake"):
        return True

    if check_api_server(timeout=3.0):
        st.session_state.backend_awake = True
        return True

    with st.spinner("Waking up backend service (this can take up to a minute on first use)..."):
        if check_api_server(timeout=75.0):
            st.session_state.backend_awake = True
            return True

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
    """Send query to FastAPI /chat endpoint, optionally consuming Server-Sent Events incrementally."""
    try:
        current_mode = mode or st.session_state.get("mode")
    except Exception:
        current_mode = mode

    payload = {
        "query": query,
        "stream": stream,
    }
    if current_mode:
        payload["mode"] = current_mode

    if current_mode != "Q&A":
        if doc_id:
            payload["doc_id"] = doc_id
        if content_type and content_type != "all":
            payload["content_type"] = content_type
        if company_id is not None:
            payload["company_id"] = company_id

    if not stream:
        resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=90.0)
        resp.raise_for_status()
        data = resp.json()
        if result_sink is not None:
            result_sink.update(
                {
                    "sources": data.get("sources", []),
                    "tool_outputs": data.get("tool_outputs", {}),
                    "routing_decision": data.get("routing_decision"),
                }
            )
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
                    result_sink.update(
                        {
                            "sources": event.get("sources", []),
                            "tool_outputs": event.get("tool_outputs", {}),
                            "routing_decision": event.get("routing_decision"),
                        }
                    )
                break
            raise RuntimeError(event.get("error", "Stream ended with an error"))
    if not collected:
        raise RuntimeError("No streamed content was received from the API.")


def fetch_generated_file_bytes(generated_file: dict) -> Optional[bytes]:
    """Fetch bytes for an agent-generated file for download_button."""
    if not generated_file or not generated_file.get("file_id"):
        return None
    try:
        params = {"filename": generated_file.get("filename")} if generated_file.get("filename") else None
        resp = requests.get(
            f"{API_BASE_URL}/files/{generated_file['file_id']}",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def render_generated_file_download(generated_file: dict, key: str) -> None:
    """Render a download button for an agent-generated file, if fetchable."""
    if not generated_file:
        return
    file_bytes = fetch_generated_file_bytes(generated_file)
    if file_bytes is None:
        st.caption("Generated file is no longer available for download.")
        return
    st.download_button(
        label=f"Download {generated_file.get('filename', 'file.xlsx')}",
        data=file_bytes,
        file_name=generated_file.get("filename", "file.xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def call_api_ingest(file_buffer, filename: str) -> dict:
    """Upload document to FastAPI /ingest endpoint."""
    files = {"file": (filename, file_buffer, "application/pdf")}
    resp = requests.post(f"{API_BASE_URL}/ingest", files=files, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def initialize_services() -> bool:
    """Initialize RAGSystem and SupabaseVectorStore."""
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = RAGSystem(
                model_name=st.session_state.get("embedding_model", settings.EMBEDDING_MODEL),
                llm_model=st.session_state.get("llm_model", settings.LLM_MODEL),
                temperature=st.session_state.get("temperature", settings.RAG_TEMPERATURE),
                base_url=st.session_state.get("base_url", settings.OPENROUTER_BASE_URL),
            )
        if st.session_state.vector_store is None and SUPABASE_OK:
            st.session_state.vector_store = SupabaseVectorStore()
        return True
    except Exception as exc:
        st.error(f"Service initialization error: {exc}")
        return False


def process_multimodal_pdf(uploaded_file) -> bool:
    """Run multimodal ingestion pipeline and persist chunks to Supabase."""
    try:
        api_available = wake_backend_if_needed()
        if api_available:
            with st.spinner("Uploading and processing multimodal PDF via FastAPI service..."):
                res = call_api_ingest(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.selected_doc_id = res.get("doc_id")
                return True

        if not initialize_services():
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        doc_id = str(uuid.uuid4())[:8] + "_" + uploaded_file.name

        with st.spinner("Extracting text, tables, and images with OCR..."):
            ingest_result = ingest_pdf(
                pdf_path=tmp_path,
                chunk_size=st.session_state.get("chunk_size", settings.RAG_CHUNK_SIZE),
                chunk_overlap=st.session_state.get("chunk_overlap", settings.RAG_CHUNK_OVERLAP),
                extract_tables=True,
                extract_images=True,
                allow_ocr=True,
                caption_llm=False,
            )

        if not ingest_result.chunks:
            st.error("No content could be extracted from the document.")
            os.unlink(tmp_path)
            return False

        rag_sys = st.session_state.rag_system
        if rag_sys and rag_sys.embeddings:
            with st.spinner(f"Computing dense vector embeddings for {len(ingest_result.chunks)} chunks..."):
                contents = [c.content for c in ingest_result.chunks]
                embeddings = rag_sys.embeddings.embed_documents(contents)
                for i, emb in enumerate(embeddings):
                    ingest_result.chunks[i].embedding = emb

        if st.session_state.vector_store:
            with st.spinner("Storing document and vector chunks in Supabase..."):
                st.session_state.vector_store.upsert_document(
                    doc_id=doc_id,
                    filename=uploaded_file.name,
                    doc_type="pdf",
                    metadata={
                        "total_pages": ingest_result.total_pages,
                        "tables": ingest_result.tables_count,
                        "images": ingest_result.images_count,
                    },
                )
                st.session_state.vector_store.upsert_chunks(
                    chunks=ingest_result.chunks,
                    doc_id=doc_id,
                )

        st.session_state.selected_doc_id = doc_id
        os.unlink(tmp_path)
        return True

    except Exception as exc:
        st.error(f"Error processing multimodal document: {exc}")
        return False


def main():
    """Main Streamlit application entry point."""
    header_container = st.container()
    with header_container:
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
                if st.button(
                    m,
                    key=f"mode_tab_{m}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    if m == "Q&A":
                        st.session_state.selected_doc_id = None
                        st.session_state.selected_content_type = "all"
                        st.session_state.current_company = None
                    st.session_state.mode = m
                    st.rerun()

        backend_mode_str = mode_mapping[st.session_state.mode]

        if st.session_state.mode == "Knowledge Base":
            initialize_services()
            available_docs = []
            if st.session_state.vector_store:
                try:
                    available_docs = st.session_state.vector_store.list_documents()
                except Exception:
                    pass

            doc_options = ["All Documents (Global Search)"] + [f"{d.filename} ({d.doc_id[:8]})" for d in available_docs]
            doc_id_map = {f"{d.filename} ({d.doc_id[:8]})": d.doc_id for d in available_docs}

            col_doc, col_filter, col_upload = st.columns([1.5, 1, 1.2])

            with col_doc:
                selected_label = st.selectbox(
                    "Scope Query Target:",
                    options=doc_options,
                    key="kb_doc_selector",
                    label_visibility="collapsed",
                )
                if selected_label == "All Documents (Global Search)":
                    st.session_state.selected_doc_id = None
                else:
                    st.session_state.selected_doc_id = doc_id_map.get(selected_label)

            with col_filter:
                content_types = ["All Content", "Text Only", "Tables Only", "Images/Figures Only"]
                type_map = {
                    "All Content": None,
                    "Text Only": "text",
                    "Tables Only": "table",
                    "Images/Figures Only": "image",
                }
                selected_type_label = st.selectbox(
                    "Filter Content Type:",
                    options=content_types,
                    key="kb_type_selector",
                    label_visibility="collapsed",
                )
                st.session_state.selected_content_type = type_map[selected_type_label]

            with col_upload:
                uploaded_file = st.file_uploader(
                    "Upload PDF to Knowledge Base:",
                    type="pdf",
                    label_visibility="collapsed",
                    key="kb_file_uploader",
                )
                if uploaded_file and st.button("Ingest Document", use_container_width=True):
                    if process_multimodal_pdf(uploaded_file):
                        st.success(f"Indexed {uploaded_file.name} successfully!")
                        st.rerun()

        elif st.session_state.mode == "Research":
            initialize_services()
            companies = []
            vector_store = st.session_state.vector_store
            if vector_store and getattr(vector_store, "client", None):
                try:
                    response = (
                        vector_store.client.table("companies")
                        .select("id, symbol, name")
                        .order("name")
                        .execute()
                    )
                    companies = response.data or []
                except Exception as exc:
                    st.error(f"Unable to load companies from the database: {exc}")
                    companies = []
            else:
                st.error("Company data is unavailable because the Supabase client is not configured.")

            if not companies:
                st.warning("No companies are available in the companies table.")
                st.session_state.current_company = None
            else:
                company_options = [f"{company.get('name', '')} ({company.get('symbol', '')})" for company in companies]
                company_lookup = {label: company for label, company in zip(company_options, companies)}
                selected_label = st.selectbox(
                    "Select Target Company Dossier:",
                    options=company_options,
                    key="header_research_selectbox",
                    label_visibility="collapsed",
                )
                st.session_state.current_company = company_lookup.get(selected_label)

        st.divider()

    chat_container = st.container(height=500, border=False)

    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
            with st.chat_message(role, avatar=avatar):
                if role == "user":
                    st.markdown(f"**You**\n\n{message['content']}")
                else:
                    st.markdown(message["content"])
                    if message.get("sources"):
                        with st.expander("Retrieved Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(f"**Source {i}**")
                                st.write(getattr(source, "page_content", str(source)))
                    if message.get("generated_file"):
                        render_generated_file_download(
                            message["generated_file"],
                            key=f"download_{message.get('id', id(message))}",
                        )

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(f"**You**\n\n{prompt}")

            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                placeholder = st.empty()
                try:
                    generated_file = None
                    if wake_backend_if_needed():
                        stream_meta: dict = {}
                        if st.session_state.mode == "Research" and st.session_state.current_company:
                            stream_payload = call_api_chat(
                                query=prompt,
                                doc_id=st.session_state.selected_doc_id,
                                content_type=st.session_state.selected_content_type,
                                company_id=st.session_state.current_company.get("id"),
                                mode="Research",
                                stream=True,
                                result_sink=stream_meta,
                            )
                        elif st.session_state.mode == "Q&A":
                            stream_payload = call_api_chat(
                                query=prompt,
                                mode="Q&A",
                                stream=True,
                                result_sink=stream_meta,
                            )
                        else:
                            stream_payload = call_api_chat(
                                query=prompt,
                                doc_id=st.session_state.selected_doc_id,
                                content_type=st.session_state.selected_content_type,
                                mode="Knowledge Base",
                                stream=True,
                                result_sink=stream_meta,
                            )
                        answer_parts = []
                        for chunk in stream_payload:
                            answer_parts.append(chunk)
                            placeholder.markdown("".join(answer_parts))
                        answer = "".join(answer_parts)
                        sources = []
                        generated_file = (stream_meta.get("tool_outputs") or {}).get("generated_file")
                    else:
                        initialize_services()
                        rag_sys = st.session_state.rag_system
                        vstore = st.session_state.vector_store

                        if backend_mode_str == "Multi-Document Knowledge Base" and vstore and rag_sys and rag_sys.embeddings:
                            query_emb = rag_sys.embeddings.embed_query(prompt)
                            filters = SearchFilter(
                                doc_id=st.session_state.selected_doc_id,
                                content_type=st.session_state.selected_content_type,
                            )
                            search_results = vstore.similarity_search(
                                query_embedding=query_emb,
                                filters=filters,
                                k=st.session_state.get("k_results", settings.RAG_K_RESULTS),
                            )
                            if not search_results:
                                answer = "I cannot find relevant information in the selected document scope."
                                sources = []
                            else:
                                context = "\n\n".join([r.content for r in search_results])
                                answer = rag_sys._build_answer(context, prompt)
                                sources = [r.content for r in search_results]
                        elif backend_mode_str == "Company Research Mode":
                            answer = "Company research mode ready. Query company reports via Knowledge Base."
                            sources = []
                        else:
                            res = rag_sys.query_general_question(prompt)
                            answer = res["answer"]
                            sources = res.get("source_documents", [])

                    placeholder.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources, "generated_file": generated_file}
                    )
                    st.rerun()

                except Exception as exc:
                    placeholder.error(f"Error generating response: {exc}")

    with st.sidebar:
        st.markdown("### Knowledge Base & Model Controls")
        with st.expander("Model Framework Tuning"):
            st.session_state["embedding_model"] = st.selectbox(
                "Embedding Model",
                ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            )
            st.session_state["llm_model"] = st.selectbox(
                "LLM Model",
                ["nvidia/nemotron-3-ultra-550b-a55b:free", "openai/gpt-4o-mini"],
            )
            st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
            st.session_state["k_results"] = st.slider("Retrieved Chunks Count", 1, 10, 4)

        if st.button("Clear Chat Session", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
