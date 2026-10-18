"""Chat history persistence for the Streamlit UI.

Two interchangeable stores share one interface:

* ``SupabaseChatStore`` keeps conversations in the ``conversations`` and
  ``messages`` tables (see migration ``003_chat_history.sql``).
* ``MemoryChatStore`` keeps them in a plain dict (held in Streamlit's session
  state) so the app still works, without saving, if Supabase or the tables
  are not available.

Visitors are anonymous: every conversation is tagged with a random
``visitor_id`` and every query filters on it, so one visitor never sees
another visitor's chats.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

TABLE_CONVERSATIONS = "conversations"
TABLE_MESSAGES = "messages"
LOCAL_PREFIX = "local-"

_VISITOR_ID_RE = re.compile(r"^[a-f0-9]{8,40}$")


def new_visitor_id() -> str:
    """Create a random anonymous visitor id."""
    return uuid.uuid4().hex[:24]


def is_valid_visitor_id(value: Any) -> bool:
    """Accept only ids we generated ourselves (lowercase hex)."""
    return isinstance(value, str) and bool(_VISITOR_ID_RE.match(value))


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: Any) -> datetime:
    """Parse a Supabase/ISO timestamp into an aware datetime (UTC fallback)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def make_title(text: str, max_len: int = 48) -> str:
    """Derive a short conversation title from the first user message."""
    first_line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    collapsed = " ".join(first_line.split())
    if not collapsed:
        return "New chat"
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def _message_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "role": row.get("role", "assistant"),
        "content": row.get("content", ""),
        "sources": row.get("sources") or [],
        "meta": row.get("meta") or {},
    }


class MemoryChatStore:
    """Non-persistent store backed by a caller-owned dict (session state)."""

    persistent = False
    last_error: Optional[str] = None

    def __init__(self, backing: dict[str, Any], visitor_id: str) -> None:
        self._d = backing
        self._visitor_id = visitor_id
        self._d.setdefault("conversations", {})
        self._d.setdefault("messages", {})

    def list_conversations(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = list(self._d["conversations"].values())
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        return rows[:limit]

    def create_conversation(self, title: str, mode: str) -> dict[str, Any]:
        now = utcnow_iso()
        conv = {
            "id": LOCAL_PREFIX + uuid.uuid4().hex,
            "title": title,
            "mode": mode,
            "created_at": now,
            "updated_at": now,
        }
        self._d["conversations"][conv["id"]] = conv
        self._d["messages"][conv["id"]] = []
        return conv

    def rename_conversation(self, conv_id: str, title: str) -> None:
        conv = self._d["conversations"].get(conv_id)
        if conv:
            conv["title"] = title

    def touch_conversation(self, conv_id: str, mode: Optional[str] = None) -> None:
        conv = self._d["conversations"].get(conv_id)
        if conv:
            conv["updated_at"] = utcnow_iso()
            if mode:
                conv["mode"] = mode

    def delete_conversation(self, conv_id: str) -> None:
        self._d["conversations"].pop(conv_id, None)
        self._d["messages"].pop(conv_id, None)

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        return [dict(m) for m in self._d["messages"].get(conv_id, [])]

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[list[str]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        msg = {
            "id": uuid.uuid4().hex,
            "role": role,
            "content": content,
            "sources": list(sources or []),
            "meta": dict(meta or {}),
        }
        self._d["messages"].setdefault(conv_id, []).append(msg)
        return dict(msg)

    def delete_message(self, conv_id: str, message_id: str) -> None:
        msgs = self._d["messages"].get(conv_id, [])
        self._d["messages"][conv_id] = [m for m in msgs if m["id"] != message_id]


class SupabaseChatStore:
    """Persistent store. Never raises on a database hiccup: it logs, records
    ``last_error`` and returns an empty/neutral value so the chat keeps working.
    """

    persistent = True

    def __init__(self, client: Any, visitor_id: str) -> None:
        self._client = client
        self._visitor_id = visitor_id
        self.last_error: Optional[str] = None

    def _fail(self, action: str, exc: Exception) -> None:
        self.last_error = f"{action}: {exc}"
        logger.warning("Chat history %s failed: %s", action, exc)

    def _owns(self, conv_id: str) -> bool:
        """True if the conversation belongs to this visitor."""
        resp = (
            self._client.table(TABLE_CONVERSATIONS)
            .select("id")
            .eq("id", conv_id)
            .eq("visitor_id", self._visitor_id)
            .limit(1)
            .execute()
        )
        return bool(resp.data)

    def list_conversations(self, limit: int = 60) -> list[dict[str, Any]]:
        try:
            resp = (
                self._client.table(TABLE_CONVERSATIONS)
                .select("id,title,mode,created_at,updated_at")
                .eq("visitor_id", self._visitor_id)
                .order("updated_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(resp.data or [])
        except Exception as exc:
            self._fail("list_conversations", exc)
            return []

    def create_conversation(self, title: str, mode: str) -> dict[str, Any]:
        now = utcnow_iso()
        row = {
            "visitor_id": self._visitor_id,
            "title": title,
            "mode": mode,
            "created_at": now,
            "updated_at": now,
        }
        try:
            resp = self._client.table(TABLE_CONVERSATIONS).insert(row).execute()
            data = resp.data or []
            if data:
                return data[0]
            raise RuntimeError("insert returned no row")
        except Exception as exc:
            self._fail("create_conversation", exc)
            return {
                "id": LOCAL_PREFIX + uuid.uuid4().hex,
                "title": title,
                "mode": mode,
                "created_at": now,
                "updated_at": now,
            }

    def rename_conversation(self, conv_id: str, title: str) -> None:
        if conv_id.startswith(LOCAL_PREFIX):
            return
        try:
            (
                self._client.table(TABLE_CONVERSATIONS)
                .update({"title": title})
                .eq("id", conv_id)
                .eq("visitor_id", self._visitor_id)
                .execute()
            )
        except Exception as exc:
            self._fail("rename_conversation", exc)

    def touch_conversation(self, conv_id: str, mode: Optional[str] = None) -> None:
        if conv_id.startswith(LOCAL_PREFIX):
            return
        values: dict[str, Any] = {"updated_at": utcnow_iso()}
        if mode:
            values["mode"] = mode
        try:
            (
                self._client.table(TABLE_CONVERSATIONS)
                .update(values)
                .eq("id", conv_id)
                .eq("visitor_id", self._visitor_id)
                .execute()
            )
        except Exception as exc:
            self._fail("touch_conversation", exc)

    def delete_conversation(self, conv_id: str) -> None:
        if conv_id.startswith(LOCAL_PREFIX):
            return
        try:
            (
                self._client.table(TABLE_CONVERSATIONS)
                .delete()
                .eq("id", conv_id)
                .eq("visitor_id", self._visitor_id)
                .execute()
            )
        except Exception as exc:
            self._fail("delete_conversation", exc)

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        if conv_id.startswith(LOCAL_PREFIX):
            return []
        try:
            if not self._owns(conv_id):
                return []
            resp = (
                self._client.table(TABLE_MESSAGES)
                .select("id,role,content,sources,meta,created_at")
                .eq("conversation_id", conv_id)
                .order("created_at")
                .execute()
            )
            return [_message_row(r) for r in (resp.data or [])]
        except Exception as exc:
            self._fail("get_messages", exc)
            return []

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        sources: Optional[list[str]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        msg = {
            "id": None,
            "role": role,
            "content": content,
            "sources": list(sources or []),
            "meta": dict(meta or {}),
        }
        if conv_id.startswith(LOCAL_PREFIX):
            return msg
        try:
            if not self._owns(conv_id):
                raise PermissionError("conversation does not belong to this visitor")
            resp = (
                self._client.table(TABLE_MESSAGES)
                .insert(
                    {
                        "conversation_id": conv_id,
                        "role": role,
                        "content": content,
                        "sources": msg["sources"],
                        "meta": msg["meta"],
                    }
                )
                .execute()
            )
            data = resp.data or []
            if data:
                msg["id"] = data[0].get("id")
        except Exception as exc:
            self._fail("add_message", exc)
        return msg

    def delete_message(self, conv_id: str, message_id: Optional[str]) -> None:
        if not message_id or conv_id.startswith(LOCAL_PREFIX):
            return
        try:
            if not self._owns(conv_id):
                return
            (
                self._client.table(TABLE_MESSAGES)
                .delete()
                .eq("id", message_id)
                .eq("conversation_id", conv_id)
                .execute()
            )
        except Exception as exc:
            self._fail("delete_message", exc)


def probe_supabase(client: Any) -> Optional[str]:
    """Check that the history tables exist and are reachable.

    Returns None when everything is fine, otherwise a short reason string.
    """
    if client is None:
        return "Supabase is not configured for the UI."
    try:
        client.table(TABLE_CONVERSATIONS).select("id").limit(1).execute()
        client.table(TABLE_MESSAGES).select("id").limit(1).execute()
        return None
    except Exception as exc:
        return str(exc)[:300]
