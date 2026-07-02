# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Disk-backed archive for long tool I/O, shared across the codebase.

Two producers write the same ``[DATUS_ARCHIVED]`` marker so a single parser
(:func:`parse_archived_marker`) understands both:

1. The rule-based minor compact pass (``datus/agent/node/compact_archive.py``)
   offloads oversized ``function_call_output.output`` items during compaction.
2. The bash tool (``datus/tools/func_tool/bash_tool.py``) offloads a command's
   output at execution time when it crosses the tool's threshold.

Both write under ``path_manager.session_data_dir(session_id)`` and replace the
in-context payload with a single-line marker carrying the absolute file path and
a short preview, so the LLM can ``read_file(<path>)`` to recover the original.

This module lives under ``datus/utils`` (its only dependencies are
``path_manager`` and ``exceptions``) so ``datus/tools`` can import it without an
inverted ``tools -> agent.node`` dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger
from datus.utils.path_manager import DatusPathManager, get_path_manager

logger = get_logger(__name__)

#: Fixed prefix written into an offloaded payload whenever an item has been
#: moved to disk. The bracketed all-caps keeps the marker visually distinct from
#: real tool output and unlikely to collide with any legitimate JSON payload.
ARCHIVED_MARKER = "[DATUS_ARCHIVED]"

_ERROR_FALLBACK_MARKERS = ('"success": 0', "'success': 0", '"error":', "'error':", "Traceback")

#: Error-output preview is widened to this multiple of ``preview_chars`` so the
#: LLM sees the full traceback / error message inline without needing a
#: round-trip through ``read_file``. Internal constant — not user-tunable
#: because the relationship is dictated by error semantics, not policy.
_ERROR_PREVIEW_MULTIPLIER = 2


def build_archived_marker(path: Any, preview: str) -> str:
    """Compose the canonical single-line ``[DATUS_ARCHIVED]`` marker.

    Keeping the format in one place guarantees every producer (compact pass,
    bash tool) emits a string that :func:`parse_archived_marker` can split. The
    `` preview=`` delimiter must appear exactly once and be preceded by a single
    space, so callers pass an already-single-line ``preview``.
    """
    return f"{ARCHIVED_MARKER} path={path} preview={preview}"


def is_archived_output(output_str: Any) -> bool:
    """Detect whether an offloaded payload is already a marker.

    ``output`` is a free-form string on the wire (real tool outputs are
    typically JSON-encoded FuncToolResult envelopes), so prefix matching is
    enough — no nested decode required.
    """
    return isinstance(output_str, str) and output_str.startswith(ARCHIVED_MARKER)


def parse_archived_marker(text: Any) -> Optional[Dict[str, str]]:
    """Parse ``[DATUS_ARCHIVED] path=<p> preview=<t>`` into its parts.

    Returns ``{"path": <abs>, "preview": <text>}`` when ``text`` is a marker
    string, ``None`` otherwise. Pure string parsing — the archive file at
    ``path`` is never opened, so callers can surface the inline preview on
    display surfaces (e.g. ``/chat/history``) without any disk I/O or
    reverse archival.

    The split uses the literal `` preview=`` delimiter, which appears exactly
    once and is preceded by a single space in markers produced by
    :func:`build_archived_marker`. This is robust to spaces inside the path's
    fallback form ``<unavailable: archive write failed>`` because that token
    contains no ``preview=`` substring.
    """
    if not isinstance(text, str) or not text.startswith(ARCHIVED_MARKER):
        return None
    body = text[len(ARCHIVED_MARKER) :].strip()
    path = ""
    preview = ""
    if " preview=" in body:
        head, preview = body.split(" preview=", 1)
        if head.startswith("path="):
            path = head[len("path=") :].strip()
    elif body.startswith("path="):
        path = body[len("path=") :].strip()
    return {"path": path, "preview": preview}


def is_error_output(output_text: str) -> bool:
    """Detect FuncToolResult-shaped errors so the preview can be widened.

    FuncToolResult (datus/tools/func_tool/base.py) wraps every tool return in
    ``{"success": 0/1, "error": ..., "result": ...}``. We try a JSON parse
    first because that is the authoritative envelope; only if the payload is
    not valid JSON do we fall back to substring matching, which catches things
    like raw tracebacks or partial JSON written by a crashing tool.

    The substring fallback can false-positive on legitimate non-JSON tool
    output that happens to contain ``"Traceback"`` or ``"error":`` as data
    (e.g. a SQL string literal, a log-mining query result). The only
    consequence is a wider preview for that one entry; correctness of the
    archived content is unaffected, so we accept the false positive in
    exchange for not missing real tracebacks emitted by crashing tools.
    """
    try:
        obj = json.loads(output_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return any(m in output_text for m in _ERROR_FALLBACK_MARKERS)
    if isinstance(obj, dict):
        if obj.get("success") == 0:
            return True
        err = obj.get("error")
        if isinstance(err, str) and err:
            return True
    return False


def make_single_line_preview(content: str, preview_chars: int) -> str:
    """Return a single-line, length-bounded preview of ``content``.

    Newlines/carriage-returns are flattened to spaces because multi-line
    previews would break the ``[DATUS_ARCHIVED] ... preview=<text>`` marker's
    single-line contract and the LLM cares only about the gist anyway.
    """
    return content[:preview_chars].replace("\n", " ").replace("\r", " ").strip()


class ToolArchive:
    """Writes long tool I/O to disk and returns a plain-text marker.

    The archive directory is resolved through :class:`DatusPathManager` so the
    storage layout stays in one place (``sessions/{project}/{session_id}/data``).
    Tests pass ``base_dir`` explicitly to keep writes hermetic.
    """

    def __init__(
        self,
        project_name: str,
        session_id: str,
        *,
        base_dir: Optional[Path] = None,
        preview_chars: int = 1000,
        path_manager: Optional[DatusPathManager] = None,
    ) -> None:
        if base_dir is not None:
            self.dir = Path(base_dir)
        else:
            pm = path_manager or get_path_manager()
            # session_data_dir() requires a project-bound path manager; if the
            # caller passed a generic one we rebuild with project_name.
            if not pm.project_name or pm.project_name != project_name:
                pm = DatusPathManager(datus_home=pm.datus_home, project_name=project_name)
            self.dir = pm.session_data_dir(session_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.preview_chars = preview_chars

    def archive(self, content: str, message_idx: int, kind: str) -> str:
        """Persist ``content`` to disk and return a single-line marker string.

        Args:
            content: Original argument or output text. Always written verbatim;
                no transformation is performed.
            message_idx: Zero-padded into the filename so a directory listing
                sorts in original session order, which makes manual debugging
                much easier.
            kind: ``"args"`` (extension ``.json``) or ``"output"`` (extension
                ``.txt``). Drives both the file extension and the preview
                strategy (error outputs get a longer preview).

        Returns:
            A single-line string of the form
            ``"[DATUS_ARCHIVED] path=<abs> preview=<text>"``. Callers store
            this verbatim in ``function_call_output.output``.
        """
        if kind not in ("args", "output"):
            raise DatusException(
                ErrorCode.COMMON_FIELD_INVALID,
                message_args={
                    "field_name": "kind",
                    "except_values": "'args' or 'output'",
                    "your_value": repr(kind),
                },
            )
        ext = "json" if kind == "args" else "txt"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        fname = f"{message_idx:06d}_{kind}_{digest[:8]}.{ext}"
        path = self.dir / fname
        # Idempotent write: if the same content lands here (same idx + hash)
        # we just overwrite — the bytes are identical.
        path.write_text(content, encoding="utf-8")
        preview_n = (
            self.preview_chars * _ERROR_PREVIEW_MULTIPLIER
            if kind == "output" and is_error_output(content)
            else self.preview_chars
        )
        return build_archived_marker(path, make_single_line_preview(content, preview_n))

    # Backwards-compat alias: the previous private helper name.
    @staticmethod
    def _is_error(output_text: str) -> bool:
        return is_error_output(output_text)
