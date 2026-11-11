# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Session-owned snapshot context, native continuation references and legacy recovery."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from datus.storage.semantic_model.artifact_file import atomic_write_text

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

    def _archive_text(self, key: str, wire: str) -> dict:
        """Keep long instructions/legacy results lossless and readable by read_file.

        JSONL text fragments avoid a single serialized JSON line exceeding the
        existing file tool's per-read cap. Offsets refer to original characters.
        """
        digest = hashlib.sha256((key + wire).encode("utf-8")).hexdigest()
        path = self.root.parent / "recovered_results" / f"{digest}.jsonl"
        if not path.exists():
            fragments = (json_text({"offset": i, "text": wire[i : i + 1000]}) for i in range(0, len(wire), 1000))
            atomic_write_text(path, "\n".join(fragments) + "\n", mode=0o600)
        return {
            "archive_path": str(path),
            "read_with": "read_file",
            "characters": len(wire),
            "format": "JSONL text fragments; read in line ranges and concatenate text by offset",
        }

    def archive_context(self, key: str, raw: Any) -> Any:
        """Preserve oversized init requests/clarifications without a reader tool."""
        if self._fits(raw, max(4096, self.page_tokens * 2), max(16384, self.page_tokens * 8)):
            return raw
        wire = json_text(raw)
        return {"success": 1, "result": {**self._archive_text(key, wire), "preview": wire[:1000]}}

    def recover_legacy(self, raw: Any) -> Any:
        """Only rewrite old archive references, never new tool results."""
        parsed = parse_result(raw)
        body = parsed.get("result") if isinstance(parsed, dict) else None
        extra = body.get("extra") if isinstance(body, dict) else None
        if not isinstance(extra, dict) or extra.get("read_with") != "read_tool_result":
            return raw
        rid = extra.get("result_id", "")
        try:
            metadata = self._metadata(rid)
            path = self.root / f"{rid}.raw"
            with self._open(path) as handle:
                wire = handle.read().decode("utf-8")
            if metadata.get("sha256") != hashlib.sha256(wire.encode("utf-8")).hexdigest():
                raise ValueError("Archived tool result is corrupt")
        except (OSError, ValueError, TypeError) as exc:
            return {
                "success": 0,
                "error": f"Cannot recover historical tool result: {exc}; rerun the original tool",
                "result": None,
            }
        return {
            "success": parsed.get("success", 1),
            "error": parsed.get("error"),
            "result": self._archive_text(rid, wire),
        }

    def _fits(self, value: Any, tokens: int, size: int) -> bool:
        return len(json_text(value).encode("utf-8")) <= size and self.counter(value) <= tokens

    def recovery_references(self, items: list[Any], limit: int = 8, max_tokens: int = 4096) -> list[dict]:
        """Retain exact native-tool continuation arguments across compaction."""
        from datus.utils.query_result_text import parse_query_result_text

        calls = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                calls[item.get("call_id")] = item.get("name")
            for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    calls[block.get("id")] = block.get("name")
        refs, seen = [], set()

        def add(ref):
            if not isinstance(ref, dict) or len(refs) >= limit:
                return
            if "result_id" in ref and "tool" not in ref:
                # Previous versions retained opaque archive identifiers/cursors.
                try:
                    recovered = self.recover_legacy({"result": {"extra": {**ref, "read_with": "read_tool_result"}}})
                    ref = recovered["result"]
                except (OSError, ValueError, TypeError, KeyError):
                    return
                if not isinstance(ref, dict):
                    return
            key = (
                ref.get("tool"),
                ref.get("snapshot_id") or ref.get("result_id") or ref.get("archive_path"),
                ref.get("section"),
            )
            if key not in seen and self.counter([*refs, ref]) <= max_tokens:
                seen.add(key)
                refs.append(ref)

        def output(raw, tool):
            parsed = parse_result(raw)
            if isinstance(parsed, list):
                for block in reversed(parsed):
                    if isinstance(block, dict) and block.get("type") in {"text", "input_text", "output_text"}:
                        output(block.get("text"), tool)
                return
            if isinstance(parsed, str):
                page = parse_query_result_text(parsed)
                if page and page.result_id and tool in {"query_metrics", "select_records"}:
                    info = next((line[6:] for line in page.lines if line.startswith("page: ")), None)
                    if info:
                        try:
                            data = json.loads(info)
                        except (TypeError, ValueError):
                            return
                        if not isinstance(data, dict):
                            return
                        add(
                            {
                                "tool": tool,
                                "result_id": page.result_id,
                                "offset": data.get("next_offset") if data.get("has_more") else data.get("offset", 0),
                            }
                        )
                return
            if not isinstance(parsed, dict):
                return
            body = parsed.get("result")
            if isinstance(body, str):
                output(body, tool)
                return
            if not isinstance(body, dict):
                return
            if body.get("read_with") == "read_file":
                add(body)
                return
            extra = body.get("extra") or {}
            if extra.get("read_with") == "read_tool_result":
                add(extra)
            elif extra.get("snapshot_id") and tool:
                ref = {
                    "tool": tool,
                    "snapshot_id": extra["snapshot_id"],
                    "offset": extra.get("next_offset") if body.get("has_more") else extra.get("offset", 0),
                }
                if body.get("section"):
                    ref["section"] = body["section"]
                add(ref)

        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output":
                output(item.get("output"), calls.get(item.get("call_id")))
            content = item.get("content")
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        output(block.get("content"), calls.get(block.get("tool_use_id")))
                    elif isinstance(block, dict) and item.get("role") == "assistant":
                        content_text = block.get("text", "")
                        if RESULT_REFERENCES_OPEN in content_text:
                            try:
                                for ref in json.loads(
                                    content_text.rsplit(RESULT_REFERENCES_OPEN, 1)[1].split(RESULT_REFERENCES_CLOSE)[0]
                                ):
                                    add(ref)
                            except (ValueError, TypeError):
                                pass
            elif item.get("role") == "assistant" and isinstance(content, str) and RESULT_REFERENCES_OPEN in content:
                try:
                    for ref in json.loads(
                        content.rsplit(RESULT_REFERENCES_OPEN, 1)[1].split(RESULT_REFERENCES_CLOSE)[0]
                    ):
                        add(ref)
                except (ValueError, TypeError):
                    pass
        return refs

    def _metadata(self, result_id: str) -> dict:
        if not isinstance(result_id, str) or not _RESULT_ID.fullmatch(result_id):
            raise ValueError("Invalid result_id")
        path = self.root / f"{result_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("Tool result is unavailable or expired in this session")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _open(path: Path):
        if path.is_symlink():
            raise ValueError("Invalid archive file")
        return path.open("rb")
