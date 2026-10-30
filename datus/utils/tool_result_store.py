# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Immutable, session-owned tool-result snapshots with bounded indexed reads."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import struct
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable

from datus.models.request_capacity import ContextCapacityError
from datus.storage.semantic_model.artifact_file import atomic_write_text, path_mutation_lock
from datus.utils.image_content import contains_images

ACTIVE_RESULT_STORE: ContextVar["ToolResultStore | None"] = ContextVar("datus_tool_result_store", default=None)
_RESULT_ID = re.compile(r"^[a-f0-9]{64}$")
RESULT_REFERENCES_OPEN = "<tool_result_references>"
RESULT_REFERENCES_CLOSE = "</tool_result_references>"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_result(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        if value.lstrip().startswith(("{", "[")):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                pass
        return value


class ToolResultStore:
    """One store per active session; result IDs never resolve outside its directory."""

    def __init__(self, archive: Any, counter: Callable[[Any], int] | None = None, *, page_tokens: int = 2048):
        self.root = Path(archive.dir) / "tool_results"
        self.counter = counter or (lambda value: len(json_text(value).encode("utf-8")))
        self.page_tokens = page_tokens
        self._registered: dict[str, Any] = {}
        self._views: dict[tuple[str, str], dict] = {}

    def register(self, call_id: str, result: Any) -> None:
        """Retain a structured result until the next input preparation; do not mutate it."""
        self._registered[call_id] = result

    def release(self, call_id: str) -> None:
        """Release registrations whose outputs were split into media/text blocks."""
        self._registered.pop(call_id, None)

    def _fits(self, value: Any, tokens: int, size: int) -> bool:
        return len(json_text(value).encode("utf-8")) <= size and self.counter(value) <= tokens

    def project(self, call_id: str, raw: Any, *, force: bool = False) -> Any:
        registered = self._registered.pop(call_id, None)
        if contains_images(raw):
            return raw
        preview_tokens = max(4096, self.page_tokens * 2)
        preview_bytes = max(16384, preview_tokens * 4)
        if not force and self._fits(raw, preview_tokens, preview_bytes):
            return raw
        parsed = parse_result(raw)
        if isinstance(parsed, dict):
            body = parsed.get("result")
            extra = body.get("extra") if isinstance(body, dict) else None
            if isinstance(extra, dict) and extra.get("read_with") == "read_tool_result" and extra.get("result_id"):
                return self.read(
                    extra["result_id"],
                    extra.get("section", ""),
                    extra.get("cursor"),
                    limit=1 if force else max(1, len(body.get("items", []))),
                    max_tokens=preview_tokens,
                    max_bytes=preview_bytes,
                    text_char_limit=1000 if force else None,
                )
        wire = raw if isinstance(raw, str) else json_text(raw)
        digest = hashlib.sha256(wire.encode("utf-8")).hexdigest()
        key = (call_id, digest + (":compact" if force else ""))
        if key in self._views:
            return self._views[key]
        result_id = hashlib.sha256(f"{call_id}:{digest}".encode()).hexdigest()
        try:
            if not (self.root / f"{result_id}.json").exists():
                self._save(result_id, wire, registered if registered is not None else parse_result(raw))
            view = self.read(
                result_id,
                limit=1 if force else 200,
                max_tokens=preview_tokens,
                max_bytes=preview_bytes,
                text_char_limit=1000 if force else None,
            )
        except ContextCapacityError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise ContextCapacityError(
                "RESULT_ARCHIVE_FAILED", f"Cannot preserve tool result {call_id}: {exc}"
            ) from exc
        self._views[key] = view
        return view

    def recovery_references(self, items: list[Any], limit: int = 8, max_tokens: int = 4096) -> list[dict]:
        """Keep a bounded set of exact recent references across repeated summaries.

        Only tool outputs and our previous recovery block are considered. Old
        references remain recoverable through the archived session history.
        """
        refs: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(value: Any) -> None:
            if not isinstance(value, dict) or len(refs) >= limit:
                return
            rid, section = value.get("result_id"), value.get("section", "")
            if not isinstance(rid, str) or not isinstance(section, str) or (rid, section) in seen:
                return
            try:
                self._metadata(rid)
            except (OSError, ValueError):
                return
            ref = {key: value[key] for key in ("result_id", "section", "cursor", "next_cursor") if key in value}
            if self.counter([*refs, ref]) <= max_tokens:
                seen.add((rid, section))
                refs.append(ref)

        def output(value: Any) -> None:
            value = parse_result(value)
            if isinstance(value, list):
                for block in value:
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
                        output(block.get("text", ""))
            elif isinstance(value, dict):
                body = value.get("result")
                if isinstance(body, dict):
                    extra = body.get("extra")
                    if isinstance(extra, dict) and extra.get("read_with") == "read_tool_result":
                        add(extra)
                        for row in body.get("items", []):
                            if isinstance(row, dict):
                                add(row.get("record_ref"))

        def previous(text: Any) -> None:
            if not isinstance(text, str) or RESULT_REFERENCES_OPEN not in text:
                return
            encoded = text.rsplit(RESULT_REFERENCES_OPEN, 1)[1].split(RESULT_REFERENCES_CLOSE, 1)[0]
            try:
                values = json.loads(encoded)
            except ValueError:
                return
            if isinstance(values, list):
                for value in values:
                    add(value)

        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output":
                output(item.get("output"))
            content = item.get("content")
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        output(block.get("content"))
                    elif isinstance(block, dict) and item.get("role") == "assistant":
                        previous(block.get("text"))
            elif item.get("role") == "assistant":
                previous(content)
            if len(refs) >= limit:
                break
        return refs

    def _save(self, result_id: str, wire: str, payload: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta_path = self.root / f"{result_id}.json"
        with path_mutation_lock(meta_path):
            if meta_path.exists():
                return
            envelope = (
                payload
                if isinstance(payload, dict) and "success" in payload and ("result" in payload or "error" in payload)
                else {}
            )
            body = envelope.get("result") if envelope else payload
            extra = body.get("extra") if isinstance(body, dict) else None
            extra = extra if isinstance(extra, dict) else {}
            sections: dict[str, Iterable[Any]] = {}
            meta = {
                "sha256": hashlib.sha256(wire.encode("utf-8")).hexdigest(),
                "success": envelope.get("success", 1),
                "error": str(envelope["error"]).encode("utf-8")[:256].decode("utf-8", errors="ignore")
                if envelope.get("error")
                else None,
                "total": body.get("total") if isinstance(body, dict) and isinstance(body.get("items"), list) else None,
                "source_truncated": body.get("truncated", extra.get("source_truncated"))
                if isinstance(body, dict)
                else None,
                "source_complete": body.get("complete", extra.get("source_complete"))
                if isinstance(body, dict)
                else None,
                "source_has_more": body.get("has_more") if isinstance(body, dict) else None,
                "sections": {},
            }
            if isinstance(body, dict) and isinstance(body.get("lineage"), list):
                for name in (
                    "lineage",
                    "nodes",
                    "files",
                    "components",
                    "stale",
                    "incomplete",
                    "resolved",
                    "ambiguous",
                    "not_found",
                    "stats",
                    "copies",
                ):
                    if name not in body:
                        continue
                    value = body[name]
                    sections[name] = self._records(value)
            elif isinstance(body, dict) and isinstance(body.get("items"), list):
                sections["items"] = self._records(body["items"])
            elif isinstance(body, list):
                sections["items"] = self._records(body)
            elif isinstance(body, dict):
                for name in ("matches", "columns", "files"):
                    if isinstance(body.get(name), list):
                        sections[name] = self._records(body[name])
                fields = {key: value for key, value in body.items() if key not in sections}
                if fields:
                    sections["fields"] = self._records(fields)
                if not sections:
                    sections["fields"] = []
            else:
                sections["text"] = []
            atomic_write_text(self.root / f"{result_id}.raw", wire, mode=0o600)
            for name, records in sections.items():
                meta["sections"][name] = self._write_section(result_id, name, records)
            meta["default_section"] = next(iter(sections))
            # Publishing metadata is the commit: an interrupted write cannot
            # expose an index or raw file that has not been completed.
            atomic_write_text(meta_path, json_text(meta), mode=0o600)

    @staticmethod
    def _records(value: Any) -> Iterable[dict]:
        if isinstance(value, dict):
            return ({"key": key, "value": item} for key, item in value.items())
        return (item if isinstance(item, dict) else {"value": item} for item in value)

    def _write_section(self, result_id: str, section: str, records: Iterable[Any]) -> int:
        paths = [self.root / f"{result_id}.{section}.{suffix}" for suffix in ("jsonl", "idx")]
        temporary: list[Path] = []
        count = 0
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as data:
                temporary.append(Path(data.name))
                with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as index:
                    temporary.append(Path(index.name))
                    for record in records:
                        index.write(struct.pack(">Q", data.tell()))
                        data.write(json_text(record).encode("utf-8") + b"\n")
                        count += 1
                    index.flush()
                    os.fsync(index.fileno())
                data.flush()
                os.fsync(data.fileno())
            for tmp, target in zip(temporary, paths, strict=True):
                os.replace(tmp, target)
        finally:
            for tmp in temporary:
                tmp.unlink(missing_ok=True)
        return count

    def _metadata(self, result_id: str) -> dict:
        if not _RESULT_ID.fullmatch(result_id):
            raise ValueError("Invalid result_id")
        path = self.root / f"{result_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("Tool result is unavailable or expired in this session")
        return json.loads(path.read_text(encoding="utf-8"))

    def read(
        self,
        result_id: str,
        section: str = "",
        cursor: str | None = None,
        limit: int = 100,
        *,
        max_tokens: int | None = None,
        max_bytes: int | None = None,
        text_char_limit: int | None = None,
    ) -> dict:
        max_tokens = self.page_tokens if max_tokens is None else max_tokens
        max_bytes = max(8192, self.page_tokens * 4) if max_bytes is None else max_bytes
        meta = self._metadata(result_id)
        section = section or meta["default_section"]
        parent, _, fragment = section.partition("/")
        if parent not in meta["sections"] and parent != "raw":
            raise ValueError("Unknown tool-result section")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, 200)
        position = self._decode_cursor(cursor, result_id, section) if cursor else 0
        extra = {
            "result_id": result_id,
            "section": section,
            "cursor": cursor,
            "next_cursor": None,
            "captured_count": meta["sections"].get(parent),
            "source_truncated": meta["source_truncated"],
            "source_complete": meta["source_complete"],
            "source_has_more": meta.get("source_has_more"),
            "sections": list(meta["sections"]),
            "sha256": meta["sha256"],
            "read_with": "read_tool_result",
            "limited_by": [],
        }
        view = {
            "success": meta["success"],
            "error": meta["error"],
            "result": {
                "items": [],
                "total": meta["total"],
                "has_more": False,
                "extra": extra,
            },
        }
        if parent in {"raw", "text"} or fragment:
            path = self.root / f"{result_id}.raw" if parent in {"raw", "text"} else self._data_path(result_id, parent)
            start, end = self._record_bounds(result_id, parent, int(fragment)) if fragment else (0, path.stat().st_size)
            if position < 0 or position > end - start:
                raise ValueError("Invalid text cursor")
            with self._open(path) as data:
                data.seek(start + position)
                content = data.read(min(max_bytes, end - start - position))
            # Trim an incomplete final UTF-8 character, never split it across pages.
            text = content.decode("utf-8", errors="ignore")
            if text_char_limit is not None:
                text = text[:text_char_limit]
            while True:
                next_position = position + len(text.encode("utf-8"))
                more = next_position < end - start
                view["result"]["items"] = [{"text": text, "offset": position}]
                view["result"]["has_more"] = more
                extra["next_cursor"] = self._cursor(result_id, section, next_position) if more else None
                extra["limited_by"] = ["bytes_or_tokens"] if more else []
                if self._fits(view, max_tokens, max_bytes):
                    return view
                if not text:
                    raise ValueError("Page metadata exceeds the result budget")
                text = text[: max(0, len(text) // 2)]
        count = meta["sections"][parent]
        if position < 0 or position > count:
            raise ValueError("Invalid record cursor")
        for record_index in range(position, min(count, position + limit)):
            start, end = self._record_bounds(result_id, parent, record_index)
            with self._open(self._data_path(result_id, parent)) as data:
                data.seek(start)
                content = data.read(min(end - start, max_bytes + 1))
            if len(content) < end - start:
                record = {
                    "record_ref": {"result_id": result_id, "section": f"{parent}/{record_index}"},
                    "preview": content[:512].decode("utf-8", errors="ignore"),
                    "preview_complete": False,
                }
            else:
                record = json.loads(content)
                if parent == "matches" and isinstance(record.get("content"), str) and len(record["content"]) > 512:
                    record = {
                        **{key: value for key, value in record.items() if key != "content"},
                        "record_ref": {"result_id": result_id, "section": f"{parent}/{record_index}"},
                        "preview": record["content"][:512],
                        "preview_complete": False,
                    }
            view["result"]["items"].append(record)
            next_position = record_index + 1
            more = next_position < count
            view["result"]["has_more"] = more
            extra["next_cursor"] = self._cursor(result_id, section, next_position) if more else None
            extra["limited_by"] = ["records"] if more else []
            if not self._fits(view, max_tokens, max_bytes):
                view["result"]["items"].pop()
                if not view["result"]["items"]:
                    # A complete record can be smaller than the byte cap but
                    # larger than the token cap. Read it as a referenced fragment.
                    view["result"]["items"] = [
                        {
                            "record_ref": {"result_id": result_id, "section": f"{parent}/{record_index}"},
                            "preview_complete": False,
                        }
                    ]
                    if not self._fits(view, max_tokens, max_bytes):
                        raise ValueError("Page metadata exceeds the result budget")
                else:
                    next_position = record_index
                view["result"]["has_more"] = next_position < count
                extra["next_cursor"] = (
                    self._cursor(result_id, section, next_position) if next_position < count else None
                )
                extra["limited_by"] = ["bytes_or_tokens"]
                break
        # Cursor and limiter fields can grow after a rejected candidate.
        # Recheck the final envelope, not only the last candidate record.
        while not self._fits(view, max_tokens, max_bytes):
            rows = view["result"]["items"]
            if len(rows) > 1:
                rows.pop()
                next_position = position + len(rows)
                view["result"]["has_more"] = next_position < count
                extra["next_cursor"] = (
                    self._cursor(result_id, section, next_position) if next_position < count else None
                )
            elif rows and "record_ref" not in rows[0]:
                rows[0] = {
                    "record_ref": {"result_id": result_id, "section": f"{parent}/{position}"},
                    "preview_complete": False,
                }
            else:
                raise ValueError("Page metadata exceeds the result budget")
        return view

    @staticmethod
    def _open(path: Path):
        if path.is_symlink():
            raise ValueError("Invalid archive file")
        return path.open("rb")

    def _data_path(self, result_id: str, section: str) -> Path:
        return self.root / f"{result_id}.{section}.jsonl"

    def _record_bounds(self, result_id: str, section: str, position: int) -> tuple[int, int]:
        with self._open(self.root / f"{result_id}.{section}.idx") as index:
            index.seek(position * 8)
            raw = index.read(16)
        if len(raw) < 8:
            raise ValueError("Invalid record index")
        start = struct.unpack(">Q", raw[:8])[0]
        end = struct.unpack(">Q", raw[8:])[0] if len(raw) == 16 else self._data_path(result_id, section).stat().st_size
        return start, end

    @staticmethod
    def _cursor(result_id: str, section: str, position: int) -> str:
        return base64.urlsafe_b64encode(json_text([result_id, section, position]).encode()).decode()

    @staticmethod
    def _decode_cursor(cursor: str, result_id: str, section: str) -> int:
        try:
            rid, name, position = json.loads(base64.urlsafe_b64decode(cursor))
            if rid != result_id or name != section or isinstance(position, bool) or not isinstance(position, int):
                raise ValueError
            return position
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid cursor for this result and section") from exc
