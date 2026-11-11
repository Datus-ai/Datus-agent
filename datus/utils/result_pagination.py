# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tool-owned immutable result pages. Page sizes have defaults, never a hard cap."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from pathlib import Path
from typing import Any, Callable

from datus.storage.semantic_model.artifact_file import atomic_write_text
from datus.utils.tool_result_store import ACTIVE_RESULT_STORE, json_text


def native_pagination(default_page_size: int = 100) -> Callable:
    """Default model invocations to pages, preserving direct Python/MCP calls."""

    def declare(method: Callable) -> Callable:
        method.__datus_default_page_size__ = default_page_size
        return method

    return declare


def validate_page(page_size: int, offset: int) -> None:
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")


def scope_digest(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


class SnapshotStore:
    """Persist in the active session when available; otherwise own an instance cache.

    A resume in another session or with a different tool/policy scope cannot resolve
    a snapshot. Refreshing a directory never changes an existing snapshot.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, dict] = {}

    @staticmethod
    def _path(snapshot_id: str) -> Path | None:
        active = ACTIVE_RESULT_STORE.get()
        if active is None:
            return None
        return active.root.parent / "result_snapshots" / f"{scope_digest(snapshot_id)}.json"

    def save(self, tool: str, payload: Any, scope: Any, query: dict) -> str:
        snapshot_id = "snapshot:" + secrets.token_hex(16)
        record = {"tool": tool, "scope": scope_digest(scope), "query": query, "payload": payload}
        # A serialized copy owns the snapshot; mutations of caller objects cannot change it.
        wire = json_text(record)
        path = self._path(snapshot_id)
        if path is None:
            self._snapshots[snapshot_id] = json.loads(wire)
        else:
            atomic_write_text(path, wire, mode=0o600)
        return snapshot_id

    def read(self, snapshot_id: str, tool: str, scope: Any, query: dict) -> Any:
        if not isinstance(snapshot_id, str) or re.fullmatch(r"snapshot:[a-f0-9]{32}", snapshot_id) is None:
            raise ValueError("Invalid snapshot_id")
        path = self._path(snapshot_id)
        if path is None:
            record = self._snapshots.get(snapshot_id)
        elif path.is_file() and not path.is_symlink():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = None
        if (
            not isinstance(record, dict)
            or record.get("tool") != tool
            or record.get("scope") != scope_digest(scope)
            or not isinstance(record.get("query"), dict)
            or "payload" not in record
        ):
            raise ValueError("Snapshot is unavailable or expired in this tool/session scope; start a new query")
        for name, value in query.items():
            if value is not None and value != "" and value != record["query"].get(name):
                raise ValueError(f"Cannot change {name} while reading a snapshot; start a new query")
        return record["payload"]


def list_page(items: list, snapshot_id: str, page_size: int, offset: int) -> dict:
    validate_page(page_size, offset)
    if offset > len(items):
        raise ValueError("offset is beyond this snapshot")
    rows = items[offset : offset + page_size]
    next_offset = offset + len(rows)
    more = next_offset < len(items)
    return {
        "items": rows,
        "total": len(items),
        "has_more": more,
        "extra": {
            "snapshot_id": snapshot_id,
            "offset": offset,
            "returned": len(rows),
            "next_offset": next_offset if more else None,
        },
    }
