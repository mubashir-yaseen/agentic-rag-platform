"""Streamlit chat UI for the Multimodal Agentic RAG Platform.

Layout (modelled on ChatGPT / Claude):
  * Left sidebar  - New chat, search, saved conversations, documents, admin.
  * Main area     - a slim toolbar (mode + scope), the conversation, and the
                    chat input. Documents can be attached with the paperclip
                    in the input box.

All heavy work happens in the FastAPI backend (``API_BASE_URL``). This file
only talks to the backend over HTTP and to Supabase for the document list,
company list and chat history.
"""

from __future__ import annotations

import base64
import hmac
import html
import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from rag_platform import get_settings  # noqa: E402
from rag_platform.vectorstore import SupabaseVectorStore  # noqa: E402

try:  # works for `streamlit run app.py` (package import) ...
    from ui.chat_store import (  # noqa: E402
        MemoryChatStore,
        SupabaseChatStore,
        is_valid_visitor_id,
        make_title,
        new_visitor_id,
        parse_ts,
        probe_supabase,
    )
except ImportError:  # ... and for `streamlit run ui/app.py`
    from chat_store import (  # type: ignore  # noqa: E402
        MemoryChatStore,
        SupabaseChatStore,
        is_valid_visitor_id,
        make_title,
        new_visitor_id,
        parse_ts,
        probe_supabase,
    )

settings = get_settings()
SUPABASE_OK = bool(settings.SUPABASE_URL and settings.effective_supabase_key)

try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000/api/v1"))
except Exception:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_BASE_URL = str(API_BASE_URL).rstrip("/")

APP_TITLE = "Multimodal Agentic RAG"
MODES = ["Knowledge Base", "Research", "Q&A"]
MODE_HELP = {
    "Knowledge Base": "Upload a PDF, then ask questions about it.",
    "Research": "Pick a company and research its filings.",
    "Q&A": "General assistant with tools. Your documents are not used.",
}
ALL_DOCS = "__all__"
MAX_DOCS_PER_VISITOR = 5
MAX_UPLOAD_MB = 15
NO_DOCS_MESSAGE = "Please upload a document first, then ask any question related to your uploaded document."
CONTENT_TYPES = {"All content": None, "Text": "text", "Tables": "table", "Images / figures": "image"}

SUGGESTIONS = {
    "Knowledge Base": [
        "Summarize the key points of this document",
        "What tables or figures does it contain?",
        "What are the main risks mentioned?",
        "Compare the key numbers across years",
    ],
    "Research": [
        "Give me a dossier summary of this company",
        "What are the key financial highlights?",
        "What risks does management call out?",
        "Any notable strategic changes?",
    ],
    "Q&A": [
        "Explain retrieval-augmented generation simply",
        "What is 15% of 2,480?",
        "Search the web for the latest AI news",
        "Create an Excel monthly budget template",
    ],
}



# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def _logo_b64(path: str = "logo.png") -> str:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except OSError:
        return ""


ASSISTANT_AVATAR = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='%23d97757'><path d='M12 2l1.9 6.1L20 10l-6.1 1.9L12 18l-1.9-6.1L4 10l6.1-1.9z'/>"
    "<path d='M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z'/></svg>"
)

CSS = """
<style>
:root{
  --bg:#212121; --sidebar:#171717; --surface:#2f2f2f; --surface-2:#3a3a3a;
  --border:#3b3b3b; --text:#ececec; --muted:#9b9b9b; --accent:#d97757;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--bg) !important; color:var(--text);
  font-family:-apple-system,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
}
[data-testid="stHeader"]{background:var(--bg) !important;}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > .st-key-toolbar){
  position:sticky; top:3.75rem; z-index:90; background:var(--bg); margin-bottom:1rem;
}
.st-key-toolbar{padding:.35rem 0 .5rem 0;}
/* Hide the Deploy button, menu and footer, but NOT the whole toolbar: newer Streamlit keeps the
   "open sidebar" arrow inside it, and hiding it made a collapsed sidebar impossible to reopen. */
[data-testid="stDecoration"], #MainMenu, footer, [data-testid="stMainMenu"],
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], .stAppDeployButton{display:none !important;}
[data-testid="stToolbar"]{background:transparent !important;}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"]{visibility:visible !important; color:var(--text) !important;}

/* ---------- main column ---------- */
[data-testid="stMainBlockContainer"], .block-container{
  max-width:820px !important; padding:3.75rem 1rem 7rem 1rem !important;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:var(--sidebar) !important; border-right:1px solid var(--border);}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1rem !important;}
.brand{display:flex; align-items:center; gap:10px; margin:0 0 14px 2px;}
.brand img{width:28px; height:28px; border-radius:8px; object-fit:cover;}
.brand .t{font-weight:650; font-size:1rem; line-height:1.1; color:var(--text);}
.brand .s{font-size:.72rem; color:var(--muted);}
[data-testid="stSidebar"] .stButton > button{
  width:100%; justify-content:flex-start; text-align:left;
  background:transparent !important; color:var(--text) !important;
  border:1px solid transparent !important; border-radius:10px !important;
  padding:.42rem .65rem !important; font-size:.88rem !important; font-weight:450 !important;
  min-height:36px; box-shadow:none !important;
}
[data-testid="stSidebar"] .stButton > button p{
  text-align:left; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; width:100%;
}
[data-testid="stSidebar"] .stButton > button:hover{background:var(--surface) !important;}
[data-testid="stSidebar"] .stButton > button[kind="primary"]{
  background:var(--surface) !important; border-color:transparent !important;
}
[data-testid="stSidebar"] .st-key-new_chat button{
  border:1px solid var(--border) !important; font-weight:600 !important;
}
[data-testid="stSidebar"] .st-key-new_chat button:hover{background:var(--surface) !important;}
.group-label{
  font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); margin:14px 0 4px 6px;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"]{
  min-height:36px; padding:.3rem .4rem !important; justify-content:center;
  background:transparent !important; border-color:transparent !important; color:var(--muted) !important;
}
[data-testid="stSidebar"] [data-testid="stPopoverButton"] > div:last-child{display:none !important;}
[data-testid="stSidebar"] [data-testid="stPopoverButton"]:hover{background:var(--surface) !important; color:var(--text) !important;}
[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"]{overflow:hidden; width:100%;}
[data-testid="stSidebar"] input{
  background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--text) !important; font-size:.85rem !important;
}
[data-testid="stSidebar"] hr{margin:.8rem 0 !important; border-color:var(--border) !important;}
.side-note{font-size:.72rem; color:var(--muted); line-height:1.35; margin:6px 4px;}

/* ---------- toolbar (mode + scope) ---------- */
[data-testid="stSegmentedControl"] button, [data-testid="stPills"] button{
  border-radius:999px !important; font-size:.82rem !important;
}
.st-key-scope_menu button{
  border-radius:999px !important; font-size:.82rem !important;
  background:var(--surface) !important; border:1px solid var(--border) !important;
}
.toolbar-hint{font-size:.78rem; color:var(--muted); margin:2px 2px 0 2px; padding-bottom:1rem;}

/* ---------- empty state ---------- */
.hero{text-align:center; margin:11vh 0 1.4rem 0;}
.hero h1{font-size:1.9rem; font-weight:650; letter-spacing:-.01em; margin:.2rem 0 .35rem 0; color:var(--text);}
.hero p{color:var(--muted); font-size:.95rem; margin:0;}
.hero .spark{font-size:2rem; color:var(--accent);}
[class*="st-key-sugg_"] button{
  width:100%; min-height:64px; text-align:left; justify-content:flex-start;
  background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:14px !important; color:var(--text) !important;
  font-size:.88rem !important; font-weight:450 !important; padding:.7rem .9rem !important;
}
[class*="st-key-sugg_"] button:hover{border-color:var(--accent) !important; background:var(--surface-2) !important;}

/* ---------- messages ---------- */
[data-testid="stChatMessage"]{
  background:transparent !important; border:none !important; padding:.55rem 0 !important; gap:.85rem;
}
[data-testid="stChatMessageAvatarUser"]{display:none !important;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse; justify-content:flex-start;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto !important; width:fit-content !important; max-width:82%;
  background:var(--surface); border-radius:20px; padding:.65rem 1.05rem;
  margin-left:auto !important; margin-right:0 !important;
}
[data-testid="stChatMessageAvatarAssistant"]{background:transparent !important;}
[data-testid="stChatMessageContent"]{font-size:.98rem; line-height:1.62; color:var(--text);}
[data-testid="stChatMessageContent"] p{margin-bottom:.6rem;}
[data-testid="stChatMessageContent"] table{font-size:.88rem;}
.chips{display:flex; flex-wrap:wrap; gap:6px; margin:.35rem 0 .2rem 0;}
.chip{
  font-size:.74rem; color:var(--muted); background:var(--surface);
  border:1px solid var(--border); border-radius:999px; padding:2px 10px;
}
.route{
  display:inline-block; font-size:.7rem; letter-spacing:.04em; text-transform:uppercase;
  color:var(--accent); border:1px solid var(--accent); border-radius:6px; padding:1px 7px; margin-bottom:6px;
}
[data-testid="stExpander"]{border:1px solid var(--border) !important; border-radius:12px !important; background:transparent !important;}
[data-testid="stExpander"] summary{font-size:.82rem; color:var(--muted);}

/* ---------- chat input ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div{background:var(--bg) !important;}
[data-testid="stChatInput"]{
  background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:26px !important; box-shadow:0 2px 14px rgba(0,0,0,.25);
}
[data-testid="stChatInput"] > div{border:none !important; box-shadow:none !important; background:transparent !important;}
[data-testid="stChatInput"] textarea{color:var(--text) !important; font-size:.98rem !important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent) !important;}

/* ---------- misc ---------- */
.stButton > button[kind="secondary"], .stDownloadButton > button{border-radius:10px;}
.stAlert{border-radius:12px;}

@media (max-width: 640px){
  [data-testid="stMainBlockContainer"], .block-container{padding:3.75rem .7rem 7rem .7rem !important;}
  .hero{margin:2vh 0 1rem 0;} .hero h1{font-size:1.45rem;} .hero .spark{font-size:1.5rem;}
  .toolbar-hint{display:none;}
  [class*="st-key-sugg_2"], [class*="st-key-sugg_3"]{display:none;}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{max-width:92%;}
}
</style>
"""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _secret(name: str) -> str:
    """Read a value from Streamlit secrets, falling back to the environment."""
    try:
        value = st.secrets.get(name)  # type: ignore[no-untyped-call]
    except Exception:
        value = None
    return str(value or os.getenv(name, "") or "")


def _supports_chat_files() -> bool:
    try:
        import inspect

        return "accept_file" in inspect.signature(st.chat_input).parameters
    except Exception:
        return False


def scroll_to_latest() -> None:
    """Scroll the conversation to the newest message.

    Streamlit does not follow new chat messages on its own, so without this a
    long conversation leaves the latest answer below the fold. The tiny
    zero-height component lives in the sidebar so it adds no gap to the chat.
    """
    with st.sidebar:
        components.html(
            """<script>
            const go = () => {
              try {
                const doc = window.parent.document;
                const el = doc.querySelector('[data-testid="stAppScrollToBottomContainer"]')
                        || doc.querySelector('section.main') || doc.querySelector('[data-testid="stMain"]');
                if (el) el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});
              } catch (e) {}
            };
            setTimeout(go, 80); setTimeout(go, 450); setTimeout(go, 1200);
            </script>""",
            height=0,
        )


class BackendError(Exception):
    """A readable error coming back from the FastAPI service."""


def _raise_for_backend(resp: requests.Response) -> None:
    if resp.ok:
        return
    try:
        body = resp.json()
        msg = body.get("error") or body.get("detail") or resp.text[:300]
        if not isinstance(msg, str):
            msg = json.dumps(msg)[:300]
    except Exception:
        msg = resp.text[:300] or resp.reason
    raise BackendError(f"{resp.status_code}: {msg}")


# --------------------------------------------------------------------------- #
# Backend calls
# --------------------------------------------------------------------------- #
def check_api_server(timeout: float = 8.0) -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def ensure_backend(slot) -> bool:
    """Wake the backend if it is asleep (Render free tier sleeps when idle)."""
    if st.session_state.get("backend_awake"):
        return True
    if check_api_server(timeout=3.0):
        st.session_state.backend_awake = True
        return True
    slot.info(
        "Waking up the server. Free hosting sleeps when idle, so the first reply "
        "can take about a minute. Later replies are fast."
    )
    ok = check_api_server(timeout=95.0)
    slot.empty()
    if ok:
        st.session_state.backend_awake = True
    return ok


def stream_chat(payload: dict[str, Any], sink: dict[str, Any]) -> Iterator[str]:
    """POST to /chat and yield answer text as it arrives (Server-Sent Events)."""
    resp = requests.post(f"{API_BASE_URL}/chat", json={**payload, "stream": True}, timeout=120.0, stream=True)
    _raise_for_backend(resp)
    sink.setdefault("reasoning", [])
    got_any = False
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        kind = event.get("event")
        if kind == "reasoning":
            sink["reasoning"].append(event.get("step", ""))
        elif kind == "token":
            got_any = True
            yield event.get("text", "")
        elif kind == "end":
            sink["sources"] = event.get("sources", []) or []
            sink["tool_outputs"] = event.get("tool_outputs", {}) or {}
            sink["routing"] = event.get("routing_decision")
            sink["answer"] = event.get("answer", "")
            break
        elif kind == "error":
            raise BackendError(event.get("error", "The server reported an error."))
    if not got_any and not sink.get("answer"):
        raise BackendError("The server returned an empty response.")


def call_api_ingest(data: bytes, filename: str, owner: Optional[str] = None) -> dict[str, Any]:
    resp = requests.post(
        f"{API_BASE_URL}/ingest",
        files={"file": (filename, data, "application/pdf")},
        data={"owner": owner} if owner else None,
        timeout=300.0,
    )
    _raise_for_backend(resp)
    return resp.json()


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_generated_file(base_url: str, file_id: str, filename: str) -> Optional[bytes]:
    try:
        resp = requests.get(f"{base_url}/files/{file_id}", params={"filename": filename}, timeout=20.0)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Supabase (documents, companies, chat history)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore()


def get_vector_store() -> Optional[SupabaseVectorStore]:
    if not SUPABASE_OK:
        return None
    try:
        return _vector_store()
    except Exception:
        return None


def load_documents(force: bool = False) -> list[Any]:
    if force or st.session_state.get("docs") is None:
        vs = get_vector_store()
        docs: list[Any] = []
        if vs is not None:
            try:
                docs = vs.list_documents()
            except Exception as exc:
                st.session_state.docs_error = str(exc)[:200]
        st.session_state.docs = docs
    return st.session_state.docs


def _doc_owner(doc: Any) -> Optional[str]:
    meta = getattr(doc, "metadata", None)
    return meta.get("owner") if isinstance(meta, dict) else None


def my_documents() -> list[Any]:
    """Documents uploaded by this visitor (newest first). Annual reports and other people's files are excluded."""
    vid = st.session_state.get("visitor_id")
    return [d for d in load_documents() if _doc_owner(d) == vid]


def managed_documents() -> list[Any]:
    """What the sidebar lists: everything for the admin, otherwise only this visitor's own files."""
    return load_documents() if is_admin() else my_documents()


def can_delete(doc: Any) -> bool:
    return is_admin() or _doc_owner(doc) == st.session_state.get("visitor_id")


def sync_scope_doc() -> None:
    """Always keep a valid document selected: the newest one of this visitor, or none."""
    ids = [d.doc_id for d in my_documents()]
    if not ids:
        st.session_state.scope_doc = None
    elif st.session_state.get("scope_doc") not in ids:
        st.session_state.scope_doc = ids[0]


def load_companies() -> list[dict[str, Any]]:
    if st.session_state.get("companies") is None:
        rows: list[dict[str, Any]] = []
        vs = get_vector_store()
        if vs is not None:
            try:
                rows = (
                    vs.client.table("companies").select("id, symbol, name").order("name").execute().data or []
                )
            except Exception as exc:
                st.session_state.companies_error = str(exc)[:200]
        st.session_state.companies = rows
    return st.session_state.companies


def get_visitor_id() -> str:
    """Anonymous id kept in the page URL (?u=...) so a refresh keeps your chats."""
    vid = st.query_params.get("u")
    if not is_valid_visitor_id(vid):
        vid = new_visitor_id()
        st.query_params["u"] = vid
    return str(vid)


def current_store():
    """Supabase-backed history when available, in-memory otherwise."""
    vid = st.session_state.visitor_id
    if "history_probe" not in st.session_state:
        vs = get_vector_store()
        st.session_state.history_probe = probe_supabase(vs.client if vs else None)
    if st.session_state.history_probe is None:
        return SupabaseChatStore(get_vector_store().client, vid)  # type: ignore[union-attr]
    return MemoryChatStore(st.session_state.memory_history, vid)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("visitor_id", get_visitor_id())
    ss.setdefault("memory_history", {})
    ss.setdefault("conv_id", None)
    ss.setdefault("messages", [])
    ss.setdefault("mode_sel", "Knowledge Base")
    ss.setdefault("_last_mode", "Knowledge Base")
    ss.setdefault("scope_doc", None)
    ss.setdefault("scope_type", "All content")
    ss.setdefault("scope_company", None)
    ss.setdefault("backend_awake", False)
    ss.setdefault("pending_prompt", None)
    ss.setdefault("regen", False)
    ss.setdefault("docs", None)
    ss.setdefault("companies", None)
    ss.setdefault("admin_ok", False)


def apply_deferred_state() -> None:
    """Apply widget-state changes queued during the previous run."""
    deferred = st.session_state.pop("_deferred", None)
    if deferred:
        st.session_state.update(deferred)


def current_mode() -> str:
    mode = st.session_state.get("mode_sel") or st.session_state.get("_last_mode") or MODES[0]
    st.session_state._last_mode = mode
    return mode


def _keep_mode() -> None:
    if st.session_state.get("mode_sel") is None:
        st.session_state.mode_sel = st.session_state.get("_last_mode", MODES[0])


# --------------------------------------------------------------------------- #
# Conversation actions (sidebar callbacks)
# --------------------------------------------------------------------------- #
def start_new_chat() -> None:
    st.session_state.conv_id = None
    st.session_state.messages = []


def open_conversation(conv_id: str, mode: str) -> None:
    store = current_store()
    st.session_state.conv_id = conv_id
    st.session_state.messages = store.get_messages(conv_id)
    st.session_state._scroll = True
    if mode in MODES:
        st.session_state.mode_sel = mode
        st.session_state._last_mode = mode


def delete_conversation(conv_id: str) -> None:
    current_store().delete_conversation(conv_id)
    if st.session_state.conv_id == conv_id:
        start_new_chat()


def rename_conversation(conv_id: str, key: str) -> None:
    title = (st.session_state.get(key) or "").strip()
    if title:
        current_store().rename_conversation(conv_id, make_title(title, 60))


def queue_prompt(text: str) -> None:
    st.session_state.pending_prompt = text


def queue_regenerate() -> None:
    """Drop the last assistant reply and ask the same question again."""
    msgs = st.session_state.messages
    if msgs and msgs[-1]["role"] == "assistant":
        last = msgs.pop()
        if st.session_state.conv_id and last.get("id"):
            current_store().delete_message(st.session_state.conv_id, last["id"])
    st.session_state.regen = True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _group_label(ts: Any) -> str:
    now = datetime.now(timezone.utc)
    age = now - parse_ts(ts)
    if age < timedelta(days=1) and parse_ts(ts).date() == now.date():
        return "Today"
    if age < timedelta(days=2):
        return "Yesterday"
    if age < timedelta(days=8):
        return "Previous 7 days"
    return "Older"


def is_admin() -> bool:
    """Admin = ADMIN_PASSWORD is configured and was entered. Admins can see and delete every document.
    Everyone can upload (Knowledge Base only) and delete their own files."""
    return bool(_secret("ADMIN_PASSWORD")) and bool(st.session_state.get("admin_ok"))


def _try_unlock() -> None:
    expected = _secret("ADMIN_PASSWORD")
    given = st.session_state.get("admin_pw_input", "")
    if expected and hmac.compare_digest(str(given).encode(), expected.encode()):
        st.session_state.admin_ok = True
    else:
        st.session_state.admin_fail = True


def _delete_document(doc_id: str) -> None:
    vs = get_vector_store()
    if vs is None:
        return
    target = next((d for d in load_documents() if d.doc_id == doc_id), None)
    if target is None or not can_delete(target):
        st.toast("You can only delete documents you uploaded.", icon=":material/lock:")
        return
    try:
        vs.delete_document(doc_id)
        if st.session_state.get("scope_doc") == doc_id:
            st.session_state.scope_doc = None
        st.session_state.docs = None
        st.toast("Document deleted.", icon=":material/delete:")
    except Exception as exc:
        st.toast(f"Could not delete: {exc}", icon=":material/error:")


def render_sidebar(store) -> None:
    logo = _logo_b64()
    with st.sidebar:
        img = f"<img src='data:image/jpeg;base64,{logo}' />" if logo else ""
        st.markdown(
            f"<div class='brand'>{img}<div><div class='t'>{html.escape(APP_TITLE)}</div>"
            "<div class='s'>by Mubashir</div></div></div>",
            unsafe_allow_html=True,
        )
        st.button(
            "New chat",
            key="new_chat",
            icon=":material/edit_square:",
            on_click=start_new_chat,
            use_container_width=True,
        )
        query = st.text_input("Search chats", key="chat_search", placeholder="Search chats", label_visibility="collapsed")

        convs = store.list_conversations(limit=60)
        if query:
            convs = [c for c in convs if query.lower() in (c.get("title") or "").lower()]

        if not convs:
            st.markdown("<div class='side-note'>No chats yet. Start by asking a question.</div>", unsafe_allow_html=True)

        groups: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
        for conv in convs:
            groups.setdefault(_group_label(conv.get("updated_at")), []).append(conv)

        for label, items in groups.items():
            st.markdown(f"<div class='group-label'>{label}</div>", unsafe_allow_html=True)
            for conv in items:
                cid = conv["id"]
                active = cid == st.session_state.conv_id
                row_main, row_menu = st.columns([0.82, 0.18], vertical_alignment="center")
                with row_main:
                    st.button(
                        conv.get("title") or "New chat",
                        key=f"conv_{cid}",
                        type="primary" if active else "secondary",
                        on_click=open_conversation,
                        args=(cid, conv.get("mode", "")),
                        use_container_width=True,
                    )
                with row_menu:
                    with st.popover("", icon=":material/more_horiz:", use_container_width=True):
                        rename_key = f"rename_{cid}"
                        st.text_input("Rename", value=conv.get("title") or "", key=rename_key)
                        st.button(
                            "Save name",
                            key=f"save_{cid}",
                            on_click=rename_conversation,
                            args=(cid, rename_key),
                            use_container_width=True,
                        )
                        st.button(
                            "Delete chat",
                            key=f"delchat_{cid}",
                            icon=":material/delete:",
                            on_click=delete_conversation,
                            args=(cid,),
                            use_container_width=True,
                        )

        st.divider()
        render_documents_panel()
        render_admin_panel()

        if store.persistent:
            st.markdown(
                "<div class='side-note'>Chats are saved. Bookmark this page's link to come back to them.</div>",
                unsafe_allow_html=True,
            )
        else:
            reason = st.session_state.get("history_probe") or ""
            st.markdown(
                "<div class='side-note'>Chats are kept only until you close this tab. "
                "Run <code>003_chat_history.sql</code> in Supabase to save them.</div>",
                unsafe_allow_html=True,
            )
            if reason:
                with st.expander("Why?"):
                    st.caption(reason)


def render_documents_panel() -> None:
    docs = managed_documents()
    with st.expander(f"Documents ({len(docs)})", expanded=False, icon=":material/description:"):
        if st.session_state.get("docs_error"):
            st.caption(f"Could not load documents: {st.session_state.docs_error}")
        if not docs:
            st.caption("No documents yet. Switch to Knowledge Base and attach a PDF.")
        for doc in docs:
            c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
            c1.markdown(f"<div style='font-size:.82rem'>{html.escape(doc.filename)}</div>", unsafe_allow_html=True)
            if can_delete(doc):
                with c2:
                    with st.popover("", icon=":material/delete:", use_container_width=True):
                        st.caption("Delete this document and its chunks?")
                        st.button(
                            "Delete",
                            key=f"deldoc_{doc.doc_id}",
                            on_click=_delete_document,
                            args=(doc.doc_id,),
                            use_container_width=True,
                        )
        if not is_admin():
            st.caption(f"Only you can see the files you upload (up to {MAX_DOCS_PER_VISITOR}).")
        if st.button("Refresh list", key="refresh_docs", icon=":material/refresh:", use_container_width=True):
            load_documents(force=True)
            st.rerun()
        if current_mode() == "Knowledge Base" and not _supports_chat_files():
            up = st.file_uploader("Upload PDF", type="pdf", key="side_uploader")
            if up is not None and st.button("Index this PDF", key="side_index", use_container_width=True):
                ingest_uploaded(up)
                st.rerun()


def render_admin_panel() -> None:
    if not _secret("ADMIN_PASSWORD"):
        return
    if st.session_state.get("admin_ok"):
        st.markdown("<div class='side-note'>Admin unlocked: you can see and delete every document.</div>", unsafe_allow_html=True)
        return
    with st.expander("Admin", icon=":material/lock:"):
        st.text_input("Password", type="password", key="admin_pw_input", label_visibility="collapsed", placeholder="Admin password")
        st.button("Unlock", key="admin_unlock", on_click=_try_unlock, use_container_width=True)
        if st.session_state.pop("admin_fail", False):
            st.caption("Wrong password.")


# --------------------------------------------------------------------------- #
# Toolbar: mode + scope
# --------------------------------------------------------------------------- #
def render_toolbar() -> None:
    with st.container(key="toolbar"):
        left, right = st.columns([3, 2], vertical_alignment="center")
        with left:
            if hasattr(st, "segmented_control"):
                st.segmented_control(
                    "Mode",
                    MODES,
                    selection_mode="single",
                    key="mode_sel",
                    on_change=_keep_mode,
                    label_visibility="collapsed",
                )
            else:  # older Streamlit
                st.radio("Mode", MODES, key="mode_sel", horizontal=True, label_visibility="collapsed")
        mode = current_mode()

        with right:
            label, icon = scope_label(mode)
            with st.container(key="scope_menu"):
                with st.popover(label, icon=icon, use_container_width=True):
                    render_scope_controls(mode)
        st.markdown(f"<div class='toolbar-hint'>{html.escape(MODE_HELP[mode])}</div>", unsafe_allow_html=True)


def scope_label(mode: str) -> tuple[str, str]:
    if mode == "Q&A":
        return "No documents", ":material/block:"
    if mode == "Research":
        company = _selected_company()
        return (company.get("name", "Select company") if company else "Select company"), ":material/domain:"
    doc_id = st.session_state.get("scope_doc")
    if doc_id:
        name = next((d.filename for d in my_documents() if d.doc_id == doc_id), "Selected document")
        return (name if len(name) <= 26 else name[:25] + "…"), ":material/description:"
    return "No document yet", ":material/upload_file:"


def _selected_company() -> Optional[dict[str, Any]]:
    """The company chosen in the scope menu (the first one until the visitor picks another)."""
    companies = load_companies()
    cid = st.session_state.get("scope_company")
    if cid is None and companies:
        cid = companies[0]["id"]
        st.session_state.scope_company = cid
    return next((c for c in companies if c.get("id") == cid), None)


def render_scope_controls(mode: str) -> None:
    if mode == "Q&A":
        st.caption("General assistant. Calculator, web search and Excel export are available; your documents are not searched.")
        return
    if mode == "Research":
        companies = load_companies()
        if not companies:
            err = st.session_state.get("companies_error")
            st.warning(
                "No companies found. " + (f"({err})" if err else "Add rows to the `companies` table in Supabase.")
            )
            return
        ids = [c["id"] for c in companies]
        names = {c["id"]: f"{c.get('name', '')} ({c.get('symbol', '')})" for c in companies}
        if st.session_state.get("scope_company") not in ids:
            st.session_state.scope_company = ids[0]
        st.selectbox("Company", ids, key="scope_company", format_func=lambda i: names.get(i, str(i)))
        return

    docs = my_documents()
    if not docs:
        st.caption("No documents yet. Attach a PDF with the + button in the message box.")
        return
    ids = [d.doc_id for d in docs]
    names = {d.doc_id: d.filename for d in docs}
    if st.session_state.get("scope_doc") not in ids:
        st.session_state.scope_doc = ids[0]
    st.selectbox("Search in", ids, key="scope_doc", format_func=lambda i: names.get(i, str(i)))
    st.selectbox("Content type", list(CONTENT_TYPES.keys()), key="scope_type")


def build_payload(prompt: str, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": prompt, "mode": mode}
    if mode == "Q&A":
        return payload
    ctype = CONTENT_TYPES.get(st.session_state.get("scope_type", "All content"))
    if mode == "Research":
        company = _selected_company()
        if company:
            payload["company_id"] = company.get("id")
    else:
        # A PDF attached together with a question is searched immediately, even
        # though the scope widget only updates on the next run.
        doc_id = st.session_state.pop("_scope_override", None) or st.session_state.get("scope_doc")
        if doc_id:
            payload["doc_id"] = doc_id
        # (the owner id is the visitor's own anonymous id: only their documents are ever listed)
    if ctype:
        payload["content_type"] = ctype
    return payload


# --------------------------------------------------------------------------- #
# Rendering the conversation
# --------------------------------------------------------------------------- #
def render_chips(sources: list[str]) -> None:
    seen: list[str] = []
    for src in sources:
        text = str(src).strip()
        if text and text not in seen:
            seen.append(text)
    if not seen:
        return
    chips = "".join(f"<span class='chip'>{html.escape(s)}</span>" for s in sorted(seen)[:12])
    st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)


def render_trace(meta: dict[str, Any]) -> None:
    routing = meta.get("routing")
    steps = [s for s in (meta.get("reasoning") or []) if s]
    if not routing and not steps:
        return
    with st.expander("How this was answered"):
        if routing:
            st.markdown(f"<span class='route'>{html.escape(str(routing))}</span>", unsafe_allow_html=True)
        for step in steps:
            st.markdown(f"- {step}")


def render_file_download(gen: dict[str, Any], key: str) -> None:
    file_id = gen.get("file_id")
    if not file_id:
        return
    name = gen.get("filename") or "file.xlsx"
    data = _fetch_generated_file(API_BASE_URL, file_id, name)
    if data is None:
        st.caption("The generated file has expired and can no longer be downloaded.")
        return
    st.download_button(
        f"Download {name}",
        data=data,
        file_name=name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
        key=key,
    )


def render_message(msg: dict[str, Any], idx: int, is_last: bool) -> None:
    role = msg["role"]
    with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else None):
        meta = msg.get("meta") or {}
        if meta.get("error"):
            st.warning(msg["content"])
            if is_last and not meta.get("notice"):
                st.button("Try again", key=f"retry_{idx}", icon=":material/refresh:", on_click=queue_regenerate)
            return
        st.markdown(msg["content"])
        if role != "assistant":
            return
        render_chips(msg.get("sources") or [])
        render_trace(meta)
        gen = meta.get("generated_file")
        if gen:
            render_file_download(gen, key=f"dl_{idx}_{gen.get('file_id', '')}")
        if is_last and not meta.get("notice"):
            st.button("Regenerate", key=f"regen_{idx}", icon=":material/refresh:", on_click=queue_regenerate)


def render_empty_state(mode: str) -> None:
    if mode == "Knowledge Base" and not my_documents():
        attach_hint = (
            f"Use the + button in the message box to attach a PDF (up to {MAX_UPLOAD_MB} MB)."
            if _supports_chat_files()
            else "Open Documents in the left sidebar to upload a PDF."
        )
        st.markdown(
            "<div class='hero'><div class='spark'>✦</div>"
            "<h1>Upload a document to get started</h1>"
            f"<p>{html.escape(NO_DOCS_MESSAGE)}</p>"
            f"<p style='font-size:.85rem;opacity:.8'>{html.escape(attach_hint)}</p></div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        "<div class='hero'><div class='spark'>✦</div>"
        f"<h1>{'What should we look into?' if mode != 'Q&A' else 'How can I help today?'}</h1>"
        f"<p>{html.escape(MODE_HELP[mode])}</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, text in enumerate(SUGGESTIONS[mode]):
        with cols[i % 2]:
            st.button(text, key=f"sugg_{i}", on_click=queue_prompt, args=(text,), use_container_width=True)


# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
def ensure_conversation(store, first_text: str, mode: str) -> str:
    if st.session_state.conv_id is None:
        conv = store.create_conversation(make_title(first_text), mode)
        st.session_state.conv_id = conv["id"]
    return st.session_state.conv_id


def ingest_uploaded(file) -> bool:
    """Send a PDF to the backend and report the result. Returns True on success."""
    store = current_store()
    mode = current_mode()
    conv_id = ensure_conversation(store, f"Uploaded {file.name}", mode)
    slot = st.empty()
    try:
        if not file.name.lower().endswith(".pdf"):
            raise BackendError("Only PDF files are supported.")
        if file.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise BackendError(f"The file is {file.size / 1024 / 1024:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.")
        if not is_admin() and len(my_documents()) >= MAX_DOCS_PER_VISITOR:
            raise BackendError(
                f"You already have {MAX_DOCS_PER_VISITOR} documents. Delete one in the sidebar (Documents) to upload another."
            )
        if not ensure_backend(slot):
            raise BackendError("The server did not wake up in time. Please try again in a minute.")
        with st.spinner(f"Indexing {file.name} (extracting text, tables and images, then embedding)..."):
            res = call_api_ingest(file.getvalue(), file.name, st.session_state.get("visitor_id"))
    except Exception as exc:
        text = f"I couldn't index **{file.name}**: {exc}"
        _append(store, conv_id, "assistant", text, meta={"error": True, "notice": True})
        return False

    # Widget keys can't be changed after the widget is drawn in this run, so
    # hand the new values to the next run (applied in apply_deferred_state).
    st.session_state.docs = None
    st.session_state._scope_override = res.get("doc_id")
    st.session_state._deferred = {
        "scope_doc": res.get("doc_id"),
        "mode_sel": "Knowledge Base",
        "_last_mode": "Knowledge Base",
    }
    summary = (
        f"Indexed **{res.get('filename', file.name)}**: {res.get('total_pages', '?')} pages, "
        f"{res.get('total_chunks', '?')} chunks ({res.get('tables_count', 0)} tables, "
        f"{res.get('images_count', 0)} images). Ask me anything about it."
    )
    _append(store, conv_id, "assistant", summary, meta={"notice": True})
    store.touch_conversation(conv_id, "Knowledge Base")
    return True


def _append(store, conv_id: str, role: str, content: str, sources=None, meta=None, persist: bool = True) -> dict[str, Any]:
    meta = meta or {}
    if persist and not meta.get("error"):
        saved = store.add_message(conv_id, role, content, sources=sources, meta=meta)
    else:
        saved = {"id": None, "role": role, "content": content, "sources": list(sources or []), "meta": meta}
    st.session_state.messages.append(saved)
    return saved


def answer_turn(prompt: str, add_user: bool = True) -> None:
    """Run one question through the agent, stream the reply and save it."""
    store = current_store()
    mode = current_mode()
    conv_id = ensure_conversation(store, prompt, mode)

    st.session_state._scroll = True  # after the rerun that follows this turn
    if add_user:
        _append(store, conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
    scroll_to_latest()

    if mode == "Knowledge Base" and not st.session_state.get("_scope_override") and not my_documents():
        _append(store, conv_id, "assistant", NO_DOCS_MESSAGE, meta={"notice": True}, persist=False)
        return

    if mode == "Research" and not _selected_company():
        _append(store, conv_id, "assistant", "Pick a company in the scope menu (top right) to start researching.", meta={"error": True, "notice": True})
        return

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        slot = st.empty()
        body = st.empty()
        sink: dict[str, Any] = {}
        try:
            if not ensure_backend(slot):
                raise BackendError("The server isn't responding. Please try again in a minute.")
            slot.markdown("<span style='color:#9b9b9b'>Thinking…</span>", unsafe_allow_html=True)
            parts: list[str] = []
            last_paint = 0.0
            for chunk in stream_chat(build_payload(prompt, mode), sink):
                parts.append(chunk)
                now = time.monotonic()
                if now - last_paint > 0.05:
                    slot.empty()
                    body.markdown("".join(parts) + " ▌")
                    last_paint = now
            answer = "".join(parts).strip() or (sink.get("answer") or "").strip()
            slot.empty()
            if not answer:
                raise BackendError("The model returned an empty answer.")
            if answer.startswith("Response to:"):
                raise BackendError(
                    "The language model didn't respond. The free model is sometimes overloaded, so please try again."
                )
        except Exception as exc:
            slot.empty()
            body.empty()
            message = str(exc)
            if isinstance(exc, requests.exceptions.Timeout):
                message = "The request timed out. The server may be busy or waking up. Please try again."
            elif isinstance(exc, requests.exceptions.ConnectionError):
                message = "Couldn't reach the server. Please try again in a moment."
            _append(store, conv_id, "assistant", message, meta={"error": True})
            return

        body.markdown(answer)
        meta = {
            "routing": sink.get("routing"),
            "reasoning": sink.get("reasoning", []),
        }
        gen = (sink.get("tool_outputs") or {}).get("generated_file")
        if gen:
            meta["generated_file"] = gen
        _append(store, conv_id, "assistant", answer, sources=sink.get("sources", []), meta=meta)
        store.touch_conversation(conv_id, mode)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    # Must run on every session (module-level code only runs for the first visitor, because
    # Streamlit caches imports), and must be the first Streamlit call of the run.
    st.set_page_config(
        page_title=f"{APP_TITLE} | Mubashir",
        page_icon="logo.png" if os.path.exists("logo.png") else None,
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    apply_deferred_state()
    st.session_state.pop("_scope_override", None)
    store = current_store()

    sync_scope_doc()
    render_sidebar(store)
    render_toolbar()
    mode = current_mode()

    messages = st.session_state.messages
    if not messages and not st.session_state.pending_prompt and not st.session_state.regen:
        render_empty_state(mode)
    for idx, msg in enumerate(messages):
        render_message(msg, idx, is_last=(idx == len(messages) - 1))
    if st.session_state.pop("_scroll", False) and messages:
        scroll_to_latest()

    can_attach = _supports_chat_files() and mode == "Knowledge Base"  # uploads: Knowledge Base only
    has_docs = bool(my_documents())
    placeholder = {
        "Knowledge Base": "Ask about your document" if has_docs else "Attach a PDF to begin, then ask about it",
        "Research": "Ask about the selected company",
        "Q&A": "Ask anything",
    }[mode]

    if can_attach:
        submitted = st.chat_input(placeholder, accept_file=True, file_type=["pdf"], key="chat_box")
        text, files = "", []
        if submitted:
            text = (submitted["text"] if "text" in submitted else "") or ""
            files = list(submitted["files"]) if "files" in submitted else []
    else:
        submitted = st.chat_input(placeholder, key="chat_box")
        text, files = (submitted or ""), []

    pending = st.session_state.pending_prompt
    regen = st.session_state.regen
    st.session_state.pending_prompt = None
    st.session_state.regen = False

    for f in files:
        ingest_uploaded(f)
    if files:
        if text.strip():
            answer_turn(text.strip())
        st.rerun()

    if text.strip():
        answer_turn(text.strip())
        st.rerun()
    elif pending:
        answer_turn(pending)
        st.rerun()
    elif regen:
        last_user = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user:
            answer_turn(last_user, add_user=False)
            st.rerun()


if __name__ == "__main__":
    main()
