# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared helpers for the visual-artifact subagents (report + dashboard)
and the matching artifact tool implementations.

Both ``GenVisualReportAgenticNode`` / ``GenVisualDashboardAgenticNode``
*and* the underlying ``ReportArtifactTools`` / ``DashboardArtifactTools``
need a tiny shared toolbox:

* ``utc_now_iso()`` — ISO-8601 UTC timestamp at second precision used
  for ``executed_at`` / ``saved_at`` / ``created_at`` fields.
* ``extract_artifact_result_field`` / ``extract_artifact_result_list`` —
  walk a recorded :class:`ActionHistory.output` envelope to pull out
  fields like ``app_jsx_path`` or ``render_files``.

The earlier ``rpt_<slug>_<yymmdd>_<rand>`` allocator and the matching
``detect_referenced_artifact_ids`` inline-scan helper are gone: the LLM
now picks a bare ``slug`` directly (the system prompt forces a ``glob``
of the kind root for uniqueness), so there's nothing to allocate and
nothing to inline-detect.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, List, Optional

from datus.schemas.action_history import ActionHistory


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp at second precision (``YYYY-MM-DDTHH:MM:SSZ``).

    Used for ``executed_at`` (report queries) and ``saved_at`` (dashboard
    template metadata) and ``created_at`` (artifact manifest).
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
