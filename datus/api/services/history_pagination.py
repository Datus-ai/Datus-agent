"""Stateless, user-scoped history windows and on-demand tool content.

Cursors address persisted SQLite rows, never process-local caches, so requests
can move between replicas sharing the session volume. Model input is untouched.
"""

import base64
import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

from datus.api.models.cli_models import ChatHistoryPageData
from datus.models.history_archive import Anchor, HistoryArchive, child_session_id, prompt_anchor
from datus.models.session_manager import SessionManager, extract_user_input, is_compact_resume_text
from datus.utils.artifact_output import session_user_scope

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


def subagent_run_offsets(manager: SessionManager, session_id: str, start_id: int) -> dict[str, int]:
    """Count earlier task results using the archive index and the live SQLite tail."""
    path = _db_path(manager, session_id)
    with _connect(path) as conn:
        conn.execute("BEGIN")
        archive = HistoryArchive(conn, path, session_id)
        counts: Counter[str] = Counter()
        indexed_calls = ""
        if archive.hidden_through:
            counts.update(
                dict(
                    conn.execute(
                        "SELECT child_id,COUNT(*) FROM chat_history_tasks "
                        "WHERE session_id=:session AND message_id<:before AND child_id IS NOT NULL GROUP BY child_id",
                        {"session": session_id, "before": start_id},
                    )
                )
            )
            indexed_calls = (
                " UNION SELECT call_id FROM chat_history_tasks "
                "WHERE session_id=:session AND message_id<:before AND child_id IS NULL"
            )

        rows = conn.execute(
            "WITH live AS (SELECT message_data FROM agent_messages "
            "WHERE session_id=:session AND id>:hidden AND id<:before), "
            "task_calls(call_id) AS ("
            "SELECT json_extract(message_data,'$.call_id') FROM live "
            "WHERE json_extract(message_data,'$.type')='function_call' "
            "AND json_extract(message_data,'$.name')='task' "
            "UNION SELECT json_extract(t.value,'$.id') FROM live,json_each(message_data,'$.content') t "
            "WHERE t.type='object' AND json_extract(t.value,'$.type')='tool_use' "
            "AND json_extract(t.value,'$.name')='task'" + indexed_calls + ") "
            "SELECT json_extract(message_data,'$.output') FROM live "
            "WHERE json_extract(message_data,'$.type')='function_call_output' "
            "AND json_extract(message_data,'$.call_id') IN task_calls "
            "UNION ALL SELECT json_extract(r.value,'$.content') FROM live,json_each(message_data,'$.content') r "
            "WHERE json_extract(message_data,'$.role')='user' AND r.type='object' "
            "AND json_extract(r.value,'$.type')='tool_result' "
            "AND json_extract(r.value,'$.tool_use_id') IN task_calls",
            {"session": session_id, "hidden": archive.hidden_through or 0, "before": start_id},
        )
        counts.update(child for (raw,) in rows if (child := child_session_id(raw)))
        return dict(counts)


def _user_turn_rows(
    conn: sqlite3.Connection, session_id: str, snapshot: int, *, after: int = 0
) -> Iterator[tuple[int, str]]:
    # Exclude native tool-result bodies in SQLite before returning user prompts.
    rows = conn.execute(
        "SELECT id,message_data FROM agent_messages WHERE session_id=? AND id>? AND id<=? "
        "AND json_extract(message_data,'$.role')='user' AND ("
        "json_type(message_data,'$.content')!='array' OR "
        "NOT EXISTS(SELECT 1 FROM json_each(message_data,'$.content') "
        "WHERE type='object' AND json_extract(value,'$.type')='tool_result') "
        "OR EXISTS(SELECT 1 FROM json_each(message_data,'$.content') "
        "WHERE type='object' AND json_extract(value,'$.type') IN ('text','input_text','output_text'))) ORDER BY id",
        (session_id, after, snapshot),
    )
    for row_id, raw in rows:
        content = extract_user_input(json.loads(raw).get("content", ""))
        if not is_compact_resume_text(content):
            yield row_id, raw


def _turn_anchors(archive: HistoryArchive, snapshot: int) -> Iterator[Anchor]:
    yield from archive.archived_anchors(snapshot)
    for row_id, raw in _user_turn_rows(archive.conn, archive.session_id, snapshot, after=archive.hidden_through or 0):
        anchor = prompt_anchor(row_id, raw)
        if anchor is not None:
            yield anchor


def subagent_run_window(manager: SessionManager, session_id: str, skip: int, count: int) -> Optional[tuple[int, int]]:
    """Resolve prompt boundaries without fetching off-page tool bodies."""
    path = _db_path(manager, session_id)
    with _connect(path) as conn:
        conn.execute("BEGIN")
        archive = HistoryArchive(conn, path, session_id)
        snapshot = archive.snapshot()
        if archive.hidden_through:
            anchors = _turn_anchors(archive, snapshot)
            first = next(anchors, None)
            stop = first[0] - 1 if first else snapshot
            prefix = next((row[0] for row in archive.rows(1, stop) if _starts_implicit_run(json.loads(row[1]))), None)
            starts = chain(
                [prefix] if prefix is not None else [],
                [first[0]] if first else [],
                (anchor[0] for anchor in anchors),
            )
            markers = list(islice(starts, skip, skip + count + 1))
            if not markers:
                return None
            return markers[0], markers[count] - 1 if len(markers) > count else snapshot
        turns = _user_turn_rows(conn, session_id, snapshot)
        first = next(turns, None)
        # Legacy/compacted sessions can start with an implicit assistant run.
        prefix = conn.execute(
            "SELECT id FROM agent_messages WHERE session_id=? AND id<? AND ("
            "json_extract(message_data,'$.type')='function_call' OR ("
            "json_extract(message_data,'$.role')='assistant' AND EXISTS("
            "SELECT 1 FROM json_each(message_data,'$.content') b WHERE b.type='object' AND ("
            "json_extract(b.value,'$.type') IN ('tool_use','server_tool_use') OR ("
            "json_extract(b.value,'$.type') IN ('text','output_text') "
            "AND json_extract(b.value,'$.text')!=''))))) ORDER BY id LIMIT 1",
            (session_id, first[0] if first else snapshot + 1),
        ).fetchone()
        starts = chain([(prefix[0], "")] if prefix else [], [first] if first else [], turns)
        markers = [row_id for row_id, _ in islice(starts, skip, skip + count + 1)]
        if not markers:
            return None
        return markers[0], markers[count] - 1 if len(markers) > count else snapshot


def _starts_implicit_run(item: dict) -> bool:
    if item.get("type") == "function_call":
        return True
    content = item.get("content")
    return (
        item.get("role") == "assistant"
        and isinstance(content, list)
        and any(
            isinstance(block, dict)
            and (
                block.get("type") in ("tool_use", "server_tool_use")
                or (block.get("type") in ("text", "output_text") and block.get("text"))
            )
            for block in content
        )
    )


def _window(manager: SessionManager, session_id: str, limit: int, cursor: Optional[str], turn: Optional[int]):
    path = _db_path(manager, session_id)
    with _connect(path) as conn:
        conn.execute("BEGIN")
        archive = HistoryArchive(conn, path, session_id)
        snapshot = archive.snapshot()
        before = snapshot + 1
        if cursor and turn is not None:
            raise ValueError("Use either a cursor or a target turn")
        if cursor:
            saved = _decode(cursor)
            if saved.get("s") != session_id:
                raise ValueError("Cursor belongs to another session")
            before, snapshot = saved.get("before"), saved.get("snapshot")
            if type(before) is not int or type(snapshot) is not int or not 1 <= before <= snapshot:
                raise ValueError("Invalid history cursor")
        turns = list(reversed(list(_turn_anchors(archive, snapshot))))
        if cursor and not any(row_id == before and digest == saved.get("anchor") for row_id, digest, _, _ in turns):
            raise HistoryChangedError("History changed; reload the latest messages")
        total = len(turns)
        if turn is not None:
            if not 1 <= turn <= total:
                raise HistoryChangedError("The requested turn is no longer available")
            first_turn = ((turn - 1) // limit) * limit + 1
            last_turn = min(total, first_turn + limit - 1)
            selected = turns[total - last_turn : total - first_turn + 1]
            before = turns[total - last_turn - 1][0] if last_turn < total else snapshot + 1
        else:
            selected = [row for row in turns if row[0] < before][:limit]
        if not selected:
            return ((1, snapshot) if not cursor and snapshot else None), False, None, total, []
        first_id, first_digest, _, _ = selected[-1]
        indices = {row[0]: total - index for index, row in enumerate(turns)}
        has_more = indices[first_id] > 1
        next_cursor = (
            _encode(
                {
                    "v": 1,
                    "s": session_id,
                    "before": first_id,
                    "snapshot": snapshot,
                    "anchor": first_digest,
                }
            )
            if has_more
            else None
        )
        anchors = [
            {
                "message_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{row_id}")),
                "index": indices[row_id],
                "text": text,
            }
            for row_id, _, text, _ in reversed(selected)
        ]
        return (first_id, min(snapshot, before - 1)), has_more, next_cursor, total, anchors


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
        body = result.get("result")
        if result.get("success"):
            return body
        error = result.get("error")
        if body is None or body == error or (isinstance(body, str) and body.strip() == error):
            return error
        # Failed tools can retain a separate, large body despite a short error.
        # Keep both in the downloadable content so neither bypasses truncation.
        return {"error": error, "result": body}
    raise ValueError("Invalid tool content field")


def get_history_page(
    service: "ChatService",
    session_id: str,
    user_id: Optional[str],
    limit: Optional[int],
    cursor: Optional[str],
    turn: Optional[int] = None,
    inline_chars: int = 8000,
):
    page_size = limit if limit is not None else 10
    inline_limit = inline_chars
    manager = SessionManager(session_dir=service._session_dir, scope=session_user_scope(service.agent_config, user_id))
    try:
        window, has_more, next_cursor, total_turns, turns = _window(manager, session_id, page_size, cursor, turn)
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
                    if isinstance(envelope.get("error"), str):
                        envelope["error"] = envelope["error"][:inline_limit]
                    # Failed envelopes sometimes also retain the same raw result.
                    envelope["result"] = None
    return ChatHistoryPageData(
        messages=messages,
        execution_context=result.data.execution_context,
        has_more=has_more,
        next_cursor=next_cursor,
        total_turns=total_turns,
        turns=turns,
    )


def get_history_content(service: "ChatService", session_id: str, user_id: Optional[str], reference: str) -> str:
    locator = _decode(reference)
    start, end, index = locator.get("a"), locator.get("b"), locator.get("i")
    if locator.get("s") != session_id or any(type(n) is not int for n in (start, end, index)):
        raise ValueError("Invalid tool content reference")
    if not 1 <= start <= end or not 0 <= index < 100000:
        raise ValueError("Invalid tool content reference")
    manager = SessionManager(session_dir=service._session_dir, scope=session_user_scope(service.agent_config, user_id))
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
