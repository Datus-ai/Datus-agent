# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for the visual-artifact subagents (report + dashboard).

Both ``GenVisualReportAgenticNode`` and ``GenVisualDashboardAgenticNode``
need to:

* Detect inline ``rpt_<id>`` / ``dash_<id>`` mentions in the user message
  so the LLM can decide between "edit existing" and "create new that
  references existing".
* Walk the recorded :class:`ActionHistory.output` envelope produced by
  artifact tool calls to pull out fields like ``app_jsx_path`` or
  ``render_files``.

Keeping these in one module so the two subagents stay byte-identical on
the logic the LLM observes.
"""

from __future__ import annotations

import json
import re
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
