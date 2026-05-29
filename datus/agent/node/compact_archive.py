# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Disk-backed archive for compact tool output.

When the rule-based minor compact pass scans the eligible region (everything
older than ``keep_recent_user_turns`` user-message turns), any
``function_call_output.output`` text whose length crosses ``archive_threshold``
is written verbatim to a file under ``path_manager.session_data_dir(session_id)``
and replaced in-session with a single-line plain-text marker. The marker carries
the absolute file path and a short preview so the LLM can ``read_file(<path>)``
to recover the original.

``function_call.arguments`` is intentionally left untouched: it must stay a
well-formed tool-call payload, and substituting a marker string for it can
make the LLM service reject the turn or imitate the marker shape in later
tool calls. Only outputs are archived.

Two design properties matter:

1. **Zero information loss** — the LLM can always ``read_file(<path>)`` to
   reconstruct the original, including for execute_ddl SQL, write_file
   content, or 50KB task subagent outputs.
2. **Idempotent re-scan** — the in-session replacement starts with a fixed
   prefix (:data:`ARCHIVED_MARKER`) so a second compact pass detects the
   already-archived item and skips it. The ``compacted_until`` state is only
   a performance hint; correctness does not depend on it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.path_manager import DatusPathManager, get_path_manager

logger = logging.getLogger(__name__)

#: Fixed prefix written into ``function_call_output.output`` whenever an item
#: has been offloaded to disk. The choice of bracketed all-caps keeps the
#: marker visually distinct from real tool output and unlikely to collide with
#: any legitimate JSON payload.
ARCHIVED_MARKER = "[DATUS_ARCHIVED]"

_ERROR_FALLBACK_MARKERS = ('"success": 0', "'success': 0", '"error":', "'error':", "Traceback")


def is_archived_output(output_str: Any) -> bool:
    """Detect whether ``function_call_output.output`` is already a marker.

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
    :meth:`ToolArchive.archive`. This is robust to spaces inside the path's
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


#: Error-output preview is widened to this multiple of ``preview_chars`` so
#: the LLM sees the full traceback / error message inline without needing a
#: round-trip through ``read_file``. Internal constant — not user-tunable
#: because the relationship is dictated by error semantics, not policy.
_ERROR_PREVIEW_MULTIPLIER = 2


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
            if kind == "output" and self._is_error(content)
            else self.preview_chars
        )
        # Preview is single-line — multi-line previews would break log parsing
        # and the LLM cares only about the gist anyway.
        preview = content[:preview_n].replace("\n", " ").replace("\r", " ").strip()
        return f"{ARCHIVED_MARKER} path={path} preview={preview}"

    @staticmethod
    def _is_error(output_text: str) -> bool:
        """Detect FuncToolResult-shaped errors so the preview is widened.

        FuncToolResult (datus/tools/func_tool/base.py) wraps every tool return
        in ``{"success": 0/1, "error": ..., "result": ...}``. We try a JSON
        parse first because that is the authoritative envelope; only if the
        payload is not valid JSON do we fall back to substring matching,
        which catches things like raw tracebacks or partial JSON written by a
        crashing tool.

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


def maybe_truncate_item(
    item: Dict[str, Any],
    archive: ToolArchive,
    threshold: int,
    idx: int,
) -> Dict[str, Any]:
    """Apply the archive rule to a single session item.

    Only ``function_call_output.output`` is eligible. ``function_call.arguments``
    is deliberately never archived — it must stay a valid tool-call payload (see
    the module docstring) — so ``function_call`` items always pass through.

    Returns the input ``item`` unchanged when nothing was archived (already a
    marker, below threshold, or not an eligible type) so callers can
    identity-compare to detect modifications. Otherwise returns a new dict with
    ``output`` replaced by the marker.

    Idempotency: outputs whose text already begins with :data:`ARCHIVED_MARKER`
    are skipped — re-running the pass over a partially-archived prefix produces
    no extra writes and no double-encoded markers.
    """
    item_type = item.get("type")
    if item_type == "function_call_output":
        out_text = item.get("output", "")
        if not isinstance(out_text, str) or len(out_text) < threshold:
            return item
        if is_archived_output(out_text):
            return item
        try:
            marker = archive.archive(out_text, idx, "output")
        except OSError as exc:
            logger.warning("compact archive write failed (idx=%s, kind=output): %s", idx, exc)
            preview_n = (
                archive.preview_chars * _ERROR_PREVIEW_MULTIPLIER
                if archive._is_error(out_text)
                else archive.preview_chars
            )
            return _inline_truncated(item, out_text, preview_n)
        return {**item, "output": marker}
    return item


def _inline_truncated(item: Dict[str, Any], text: str, preview_n: int) -> Dict[str, Any]:
    """Degrade gracefully when the disk archive write fails.

    Drops back to an inline preview-only placeholder in ``output`` so the rest
    of the compact pass can still proceed. Information is lost on this branch,
    but only when the disk itself is broken — the alternative (raising) would
    crash a session over a transient ENOSPC. The placeholder still carries
    :data:`ARCHIVED_MARKER` so a later pass treats it as already-handled.
    """
    preview = text[:preview_n].replace("\n", " ").replace("\r", " ").strip()
    marker = f"{ARCHIVED_MARKER} path=<unavailable: archive write failed> preview={preview}"
    return {**item, "output": marker}
