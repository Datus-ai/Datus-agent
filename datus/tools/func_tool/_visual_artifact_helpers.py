# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for the visual-artifact subagents (report + dashboard)
and the matching artifact tool implementations.

Both ``GenVisualReportAgenticNode`` / ``GenVisualDashboardAgenticNode``
*and* the underlying ``ReportArtifactTools`` / ``DashboardArtifactTools``
need a small set of byte-identical helpers:

* Detect inline ``rpt_<id>`` / ``dash_<id>`` mentions in the user message
  so the LLM can decide between "edit existing" and "create new that
  references existing".
* Walk the recorded :class:`ActionHistory.output` envelope produced by
  artifact tool calls to pull out fields like ``app_jsx_path`` or
  ``render_files``.
* Allocate a fresh artifact id collision-free under the project root,
  and stamp an ISO-8601 UTC timestamp for ``executed_at`` / ``saved_at``
  fields.

Keeping all of the above in one module so the two subagents stay
byte-identical on the logic the LLM observes.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import uuid
from pathlib import Path
from typing import Any, List, Optional

from datus.schemas.action_history import ActionHistory


def detect_referenced_artifact_ids(
    *,
    user_message: str,
    project_root: Path,
    root_dir_name: str,
    id_inline_regex: re.Pattern[str],
    id_full_regex: re.Pattern[str],
    app_jsx_relpath: str = "render/app.jsx",
) -> List[str]:
    """Return artifact ids the user mentioned that already exist on disk.

    Used purely as an awareness hint for the LLM — the model decides
    whether to bind/edit them or to start a fresh artifact that
    references them. Deduplicates while preserving first-mention order
    so the hint reads naturally.

    Parameters
    ----------
    user_message:
        Raw user prompt text. Matched case-insensitively because LLMs
        sometimes uppercase artifact ids in conversation.
    project_root:
        Resolved project root; artifact ids resolve to
        ``project_root/<root_dir_name>/<id>/``.
    root_dir_name:
        ``"reports"`` for ``rpt_``, ``"dashboards"`` for ``dash_``.
    id_inline_regex:
        Pattern that matches the id inside the message body (e.g. the
        loose ``rpt_<chars>`` form with a non-alnum guard at the front).
    id_full_regex:
        Strict pattern that the candidate must additionally pass before
        we treat it as a real artifact id (e.g. ``REPORT_ID_RE``).
    app_jsx_relpath:
        Path inside the artifact directory that must exist before we
        consider the directory a valid artifact. Defaults to the
        ``render/app.jsx`` contract both subagents enforce.
    """
    root = project_root / root_dir_name
    if not root.is_dir():
        return []
    seen: set[str] = set()
    found: List[str] = []
    for match in id_inline_regex.finditer(user_message.lower()):
        candidate = match.group(0)
        if candidate in seen or not id_full_regex.fullmatch(candidate):
            continue
        candidate_dir = root / candidate
        if candidate_dir.is_dir() and (candidate_dir / app_jsx_relpath).is_file():
            seen.add(candidate)
            found.append(candidate)
    return found


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp at second precision (``YYYY-MM-DDTHH:MM:SSZ``).

    Used for ``executed_at`` (report queries) and ``saved_at`` (dashboard
    template metadata). Stamped to the second so two saves within the
    same minute don't appear identical.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SLUG_NON_ASCII_RE = re.compile(r"[^a-z0-9]+")


def slugify_title(title: str, max_len: int = 32) -> str:
    """Best-effort ASCII slug from an LLM-supplied artifact title.

    Non-ASCII characters are stripped, runs of separators collapse to a
    single underscore, leading/trailing underscores are trimmed, and the
    result is truncated to ``max_len`` chars. Returns an empty string
    when the title contains no usable characters — callers fall back to
    a literal default (``"report"`` / ``"dashboard"``).
    """
    ascii_only = title.encode("ascii", errors="ignore").decode("ascii")
    slug = _SLUG_NON_ASCII_RE.sub("_", ascii_only.lower()).strip("_")
    return slug[:max_len]


def allocate_artifact_id(
    *,
    title: str,
    project_root: Path,
    prefix: str,
    root_dir_name: str,
    default_base_slug: str,
    max_total_len: int,
    attempts: int = 8,
) -> str:
    """Generate ``<prefix><title-slug>_<yymmdd>_<rand6>`` not colliding on disk.

    Parameters
    ----------
    title:
        Raw LLM-supplied title. Slugified via :func:`slugify_title`; empty
        result falls back to ``default_base_slug``.
    project_root:
        Resolved project root; the candidate is checked against
        ``project_root/<root_dir_name>/<candidate>``.
    prefix:
        ``"rpt_"`` for reports, ``"dash_"`` for dashboards.
    root_dir_name:
        ``"reports"`` for reports, ``"dashboards"`` for dashboards.
    default_base_slug:
        Slug to use when ``title`` reduces to an empty string after
        ASCII filtering.
    max_total_len:
        Hard cap on the final id length. Mirrors the legacy values
        (report = 83, dashboard = 84) so we don't shorten existing ids.
    attempts:
        Number of unique-id rolls before giving up. Default 8.

    Raises
    ------
    RuntimeError
        If no collision-free candidate is found within ``attempts``.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%y%m%d")
    base_slug = slugify_title(title) or default_base_slug
    artifact_root = project_root / root_dir_name
    for _ in range(attempts):
        suffix = uuid.uuid4().hex[:6]
        candidate = f"{prefix}{base_slug}_{stamp}_{suffix}"[:max_total_len]
        if not (artifact_root / candidate).exists():
            return candidate
    raise RuntimeError(f"Failed to allocate a unique {root_dir_name.rstrip('s')} id after {attempts} attempts")


def extract_artifact_result_field(action: ActionHistory, field: str) -> Optional[str]:
    """Pull a string-valued field out of a recorded artifact tool call.

    Tool outputs land in :pyattr:`ActionHistory.output` under a few
    possible shapes depending on which dispatcher recorded them — see
    the agent framework's tool harness and the mock-LLM test harness.
    ``FuncToolResult`` is always serialized as
    ``{success, error, result}``, so we recursively scan for that
    envelope. JSON-string payloads (some dispatchers store tool output
    as a serialized string) are parsed on the fly. Empty strings are
    treated as "not found" so callers don't have to disambiguate.
    """
    output = action.output
    if not isinstance(output, dict):
        return None

    def _scan(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            if field in obj and isinstance(obj[field], str):
                return obj[field]
            for key in ("result", "raw_output", "output", "data"):
                if key in obj:
                    found = _scan(obj[key])
                    if found:
                        return found
            for value in obj.values():
                found = _scan(value)
                if found:
                    return found
        elif isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except (TypeError, json.JSONDecodeError):
                return None
            return _scan(parsed)
        return None

    return _scan(output)


def extract_artifact_result_list(action: ActionHistory, field: str) -> Optional[List[Any]]:
    """Pull a list-valued field out of a recorded artifact tool call.

    Same scanning rules as :func:`extract_artifact_result_field`. Unlike
    the string variant, an empty list IS treated as a hit — callers may
    legitimately observe a zero-row payload and we should not paper over
    that by continuing to scan siblings.
    """
    output = action.output
    if not isinstance(output, dict):
        return None

    def _scan(obj: Any) -> Optional[List[Any]]:
        if isinstance(obj, dict):
            if field in obj and isinstance(obj[field], list):
                return obj[field]
            for key in ("result", "raw_output", "output", "data"):
                if key in obj:
                    found = _scan(obj[key])
                    if found is not None:
                        return found
            for value in obj.values():
                found = _scan(value)
                if found is not None:
                    return found
        elif isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except (TypeError, json.JSONDecodeError):
                return None
            return _scan(parsed)
        return None

    return _scan(output)
