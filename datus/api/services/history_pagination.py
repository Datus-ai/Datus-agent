"""Stateless, user-scoped history windows and on-demand tool content.

Cursors address persisted SQLite rows, never process-local caches, so requests
can move between replicas sharing the session volume. Model input is untouched.
"""

import ast
import base64
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from datus.api.models.cli_models import ChatHistoryPageData
from datus.models.session_manager import SessionManager, extract_user_input, is_compact_resume_text
from datus.utils.config_utils import coerce_positive_int

if TYPE_CHECKING:
    from datus.api.services.chat_service import ChatService


class HistoryChangedError(ValueError):
    """A compact/rewind invalidated the cursor or stored content."""


def _encode(value: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).decode().rstrip("=")


def _decode(token: str) -> dict:
    if len(token) > 2048:
        raise ValueError("Invalid history reference")
    try:
        value = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid history reference") from exc
    if not isinstance(value, dict) or value.get("v") != 1:
        raise ValueError("Invalid history reference")
    return value


def _db_path(manager: SessionManager, session_id: str) -> Path:
    manager._validate_session_id(session_id)
    root = Path(manager.session_dir).resolve()
    path = (root / f"{session_id}.db").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("Session not found")
    return path


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _parse_output(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw


def subagent_run_offsets(manager: SessionManager, session_id: str, start_id: int) -> dict[str, int]:
    """Skip a reused child's runs belonging to task calls on earlier pages."""
    with _connect(_db_path(manager, session_id)) as conn:
        rows = conn.execute(
            "SELECT json_extract(o.message_data, '$.output') FROM agent_messages o "
            "JOIN agent_messages c ON c.session_id=o.session_id "
            "AND json_extract(c.message_data, '$.call_id')=json_extract(o.message_data, '$.call_id') "
            "WHERE o.session_id=? AND o.id<? AND json_extract(o.message_data, '$.type')='function_call_output' "
            "AND json_extract(c.message_data, '$.type')='function_call' "
            "AND json_extract(c.message_data, '$.name')='task' ORDER BY o.id",
            (session_id, start_id),
        )
        native_rows = conn.execute(
            "SELECT json_extract(r.value,'$.content') FROM agent_messages m, json_each(m.message_data,'$.content') r "
            "WHERE m.session_id=? AND m.id<? AND json_extract(m.message_data,'$.role')='user' "
            "AND r.type='object' AND json_extract(r.value,'$.type')='tool_result' "
            "AND json_extract(r.value,'$.tool_use_id') IN ("
            "SELECT json_extract(t.value,'$.id') FROM agent_messages c,json_each(c.message_data,'$.content') t "
            "WHERE c.session_id=? AND c.id<? AND t.type='object' "
            "AND json_extract(t.value,'$.name')='task')",
            (session_id, start_id, session_id, start_id),
        )
        counts: Counter[str] = Counter()
        for records in (rows, native_rows):
            for (raw,) in records:
                output = _parse_output(raw)
                if isinstance(output, list):
                    output = _parse_output(
                        "\n".join(str(block.get("text", "")) for block in output if isinstance(block, dict))
                    )
                if isinstance(output, dict):
                    result = _parse_output(output.get("result", output))
                    child = result.get("session_id") if isinstance(result, dict) else None
                    if isinstance(child, str) and SessionManager._SESSION_ID_RE.fullmatch(child) and ".." not in child:
                        counts[child] += 1
        return dict(counts)


def _window(manager: SessionManager, session_id: str, limit: int, cursor: Optional[str]):
    path = _db_path(manager, session_id)
    with _connect(path) as conn:
        snapshot = conn.execute(
            "SELECT COALESCE(MAX(id),0) FROM agent_messages WHERE session_id=?", (session_id,)
        ).fetchone()[0]
        before = snapshot + 1
        if cursor:
            saved = _decode(cursor)
            if saved.get("s") != session_id:
                raise ValueError("Cursor belongs to another session")
            before, snapshot = saved.get("before"), saved.get("snapshot")
            if type(before) is not int or type(snapshot) is not int or not 1 <= before <= snapshot:
                raise ValueError("Invalid history cursor")
            anchor = conn.execute(
                "SELECT message_data FROM agent_messages WHERE session_id=? AND id=?", (session_id, before)
            ).fetchone()
            if not anchor or hashlib.sha256(anchor[0].encode()).hexdigest() != saved.get("anchor"):
                raise HistoryChangedError("History changed; reload the latest messages")
        # Anthropic tool_result-only user rows are not user turns. Filter in
        # SQLite before returning content; a single tool result can be megabytes.
        rows = conn.execute(
            "SELECT id,message_data FROM agent_messages WHERE session_id=? AND id<? AND id<=? "
            "AND json_extract(message_data,'$.role')='user' AND ("
            "json_type(message_data,'$.content')!='array' OR "
            "NOT EXISTS(SELECT 1 FROM json_each(message_data,'$.content') WHERE json_extract(value,'$.type')='tool_result') "
            "OR EXISTS(SELECT 1 FROM json_each(message_data,'$.content') "
            "WHERE json_extract(value,'$.type') IN ('text','input_text','output_text'))) ORDER BY id DESC",
            (session_id, before, snapshot),
        )
        turns = []
        for row_id, raw in rows:
            content = extract_user_input(json.loads(raw).get("content", ""))
            if is_compact_resume_text(content):
                continue
            turns.append((row_id, raw))
            if len(turns) == limit + 1:
                break
        has_more = len(turns) > limit
        selected = turns[:limit]
        if not selected:
            return (1, snapshot) if not cursor and snapshot else None, False, None
        first_id, first_raw = selected[-1]
        next_cursor = (
            _encode(
                {
                    "v": 1,
                    "s": session_id,
                    "before": first_id,
                    "snapshot": snapshot,
                    "anchor": hashlib.sha256(first_raw.encode()).hexdigest(),
                }
            )
            if has_more
            else None
        )
        return (first_id, min(snapshot, before - 1)), has_more, next_cursor


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _preview(value: Any, budget: int) -> Any:
    """Bound preview traversal as well as serialized size."""
    nodes = 200
    remaining = budget

    def visit(item: Any, depth: int):
        nonlocal nodes, remaining
        nodes -= 1
        if isinstance(item, str):
            text = item[: min(remaining, budget)]
            remaining -= len(text)
            return text + ("…" if len(text) < len(item) else "")
        if not isinstance(item, (dict, list)):
            return item
        if depth >= 6 or nodes <= 0 or remaining <= 0:
            return "…"
        if isinstance(item, list):
            result = []
            for child in item:
                if nodes <= 0 or remaining <= 0:
                    break
                result.append(visit(child, depth + 1))
            return result
        result = {}
        for key, child in item.items():
            if nodes <= 0 or remaining <= 0:
                break
            result[str(key)[:100]] = visit(child, depth + 1)
        return result

    return visit(value, 0)


def _value(part, field: str) -> Any:
    if field == "params" and part.type == "call-tool":
        return part.payload.get("toolParams")
    if field == "result" and part.type == "call-tool-result":
        result = part.payload.get("result") or {}
        return result.get("error") if not result.get("success") else result.get("result")
    raise ValueError("Invalid tool content field")


def get_history_page(
    service: "ChatService", session_id: str, user_id: Optional[str], limit: Optional[int], cursor: Optional[str]
):
    config = getattr(service.agent_config, "api_config", {}) or {}
    max_turns = min(coerce_positive_int(config.get("max_history_turns"), 50), 100)
    page_size = min(coerce_positive_int(limit, coerce_positive_int(config.get("default_history_turns"), 10)), max_turns)
    inline_limit = min(coerce_positive_int(config.get("history_inline_chars"), 8000), 16000)
    manager = SessionManager(session_dir=service._session_dir, scope=user_id)
    try:
        window, has_more, next_cursor = _window(manager, session_id, page_size, cursor)
    except FileNotFoundError:
        return ChatHistoryPageData()
    if not window:
        return ChatHistoryPageData()
    result = service.get_history(session_id, user_id, window=window)
    if not result.success or result.data is None:
        raise ValueError(result.errorMessage or "Unable to read history")
    messages = result.data.messages
    for message in messages:
        for index, part in enumerate(message.content):
            if part.type == "call-tool":
                part.payload["historyOnly"] = True
                # Review diffs belong to live calls, not read-only history.
                part.payload.pop("preview", None)
                field = "params"
            elif part.type == "call-tool-result":
                field = "result"
            else:
                continue
            original = _value(part, field)
            text = _text(original)
            if len(text) <= inline_limit:
                continue
            metadata = {
                "ref": _encode(
                    {
                        "v": 1,
                        "s": session_id,
                        "a": window[0],
                        "b": window[1],
                        "m": message.message_id,
                        "i": index,
                        "f": field,
                        "h": hashlib.sha256(text.encode()).hexdigest(),
                    }
                ),
                "preview": text[:inline_limit],
                "size_bytes": len(text.encode()),
            }
            if field == "params":
                part.payload["toolParams"] = _preview(original, inline_limit)
                part.payload["toolParamsHistoryContent"] = metadata
            else:
                envelope = part.payload["result"]
                if isinstance(part.payload.get("error"), str):
                    part.payload["error"] = part.payload["error"][:inline_limit]
                part.payload["shortDesc"] = str(part.payload.get("shortDesc", ""))[:500]
                envelope["historyContent"] = metadata
                if envelope.get("success"):
                    envelope["result"] = _preview(original, inline_limit)
                else:
                    envelope["error"] = text[:inline_limit]
                    # Failed envelopes sometimes also retain the same raw result.
                    envelope["result"] = None
    return ChatHistoryPageData(messages=messages, has_more=has_more, next_cursor=next_cursor)


def get_history_content(service: "ChatService", session_id: str, user_id: Optional[str], reference: str) -> str:
    locator = _decode(reference)
    start, end, index = locator.get("a"), locator.get("b"), locator.get("i")
    if locator.get("s") != session_id or any(type(n) is not int for n in (start, end, index)):
        raise ValueError("Invalid tool content reference")
    if not 1 <= start <= end or not 0 <= index < 100000:
        raise ValueError("Invalid tool content reference")
    manager = SessionManager(session_dir=service._session_dir, scope=user_id)
    _db_path(manager, session_id)
    history = service.get_history(session_id, user_id, window=(start, end))
    if not history.success or history.data is None:
        raise FileNotFoundError("Tool output is no longer available")
    for message in history.data.messages:
        if message.message_id == locator.get("m") and index < len(message.content):
            text = _text(_value(message.content[index], locator.get("f")))
            if hashlib.sha256(text.encode()).hexdigest() != locator.get("h"):
                raise HistoryChangedError("Tool output changed; reload the latest messages")
            return text
    raise FileNotFoundError("Tool output is no longer available")
