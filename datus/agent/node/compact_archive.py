# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Compact-pass archiving glue.

The disk-archive primitive (:class:`ToolArchive`) and the ``[DATUS_ARCHIVED]``
marker helpers now live in :mod:`datus.utils.tool_archive` so both the minor
compact pass and the bash tool can share them without an inverted
``tools -> agent.node`` dependency. This module re-exports them for existing
call sites and keeps the compact-pass-specific item rewriting logic
(:func:`maybe_truncate_item`).

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
   reconstruct the original, including for execute_sql SQL, write_file
   content, or 50KB task subagent outputs.
2. **Idempotent re-scan** — the in-session replacement starts with a fixed
   prefix (:data:`ARCHIVED_MARKER`) so a second compact pass detects the
   already-archived item and skips it. The ``compacted_until`` state is only
   a performance hint; correctness does not depend on it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

# Re-exported for backwards compatibility with existing import sites
# (datus/api/services/action_sse_converter.py, tests, etc.).
from datus.utils.tool_archive import (  # noqa: F401
    _ERROR_FALLBACK_MARKERS,
    _ERROR_PREVIEW_MULTIPLIER,
    ARCHIVED_MARKER,
    ToolArchive,
    build_archived_marker,
    is_archived_output,
    is_error_output,
    make_single_line_preview,
    parse_archived_marker,
)

logger = logging.getLogger(__name__)


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
                if is_error_output(out_text)
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
    preview = make_single_line_preview(text, preview_n)
    marker = build_archived_marker("<unavailable: archive write failed>", preview)
    return {**item, "output": marker}
