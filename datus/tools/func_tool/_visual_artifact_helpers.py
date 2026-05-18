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
import os
import re
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from datus.schemas.action_history import ActionHistory
from datus.schemas.analysis_artifacts import QueryBrief, SubjectRefIds
from datus.schemas.artifact_manifest import ArtifactManifest
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


# Append-time noise filter for ``intent.md``. We only block patterns
# that are mechanically obvious and have ZERO chance of being real user
# intent: renderer / compiler error reports forwarded into the prompt
# loop. These are structural — pure pattern matching catches them
# precisely with no risk of dropping real intent.
_INTENT_NOISE_PATTERNS = (
    re.compile(r"^\s*Error:\s", re.IGNORECASE),
    re.compile(r"\bTraceback\b", re.IGNORECASE),
    re.compile(r"\bReferenceError\b"),
    re.compile(r"\bSyntaxError\b"),
    re.compile(r"\bTypeError\b"),
    # JS-style stack frame: ``    at functionName (file:line)``.
    re.compile(r"^\s*at [A-Za-z_$][\w$.]*\s*\(.*\)", re.MULTILINE),
)


def _is_meaningful_intent(message: str) -> bool:
    """Decide whether a prompt should be recorded in ``analysis/intent.md``.

    Returns ``False`` for empty / whitespace-only prompts and renderer
    error reports (mechanically obvious noise); ``True`` for everything
    else. Semantic "placeholder vs real intent" judgment runs later, in
    the finalize-stage LLM call.
    """
    text = (message or "").strip()
    if not text:
        return False
    for pattern in _INTENT_NOISE_PATTERNS:
        if pattern.search(text):
            return False
    return True


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


# --------------------------------------------------------------------------- #
# Analysis-artifact filesystem helpers                                        #
# --------------------------------------------------------------------------- #
#
# These wrap the three filesystem mutations the report / dashboard artifact
# tools both need to perform once we landed the analysis/ directory:
#
#   * ``append_intent_section`` — append-only writes to ``intent.md``,
#     filtered through ``_is_meaningful_intent`` so renderer error reports
#     and "continue" placeholders never reach the file.
#   * ``upsert_manifest_after_save`` — bump ``manifest.updated_at`` and add a
#     datasource to ``manifest.datasources`` if it isn't already there.
#   * ``write_query_brief`` — write ``queries/<name>.brief.json``.
#
# Each helper is best-effort: failures are logged and surfaced as a string
# error message but never raise, so the caller can decide whether to bubble
# the issue up as a hard FuncToolResult error (e.g. for save_query, where
# missing brief metadata makes the artifact incomplete) or treat it as
# a soft warning (e.g. for intent.md, where the SQL is the load-bearing
# artifact and the prompt log is bonus).


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomic file write — same implementation duplicated in the report /
    dashboard tool modules. Exposed here so the analysis helpers below can
    use it without forcing a cross-import."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_intent_section(
    analysis_dir: Path,
    *,
    user_message: str,
    mode: str,
    timestamp: str,
) -> Optional[str]:
    """Append a timestamped ``> ...`` blockquote section to ``analysis/intent.md``.

    The file is the raw record of every user prompt that drove a
    start_new / bind_existing call against this artifact — see
    ``docs/analysis_artifacts.md`` §3.3. Always append; never rewrite.

    Returns an error string on failure (so the caller can include it in
    the FuncToolResult), ``None`` on success.

    Mechanically obvious noise is dropped at append time
    (:func:`_is_meaningful_intent` — renderer / compiler error reports
    forwarded into the loop). Semantic "placeholder vs real intent"
    curation runs later, in the finalize-stage LLM call, which has
    the multi-prompt context needed to tell short directives like
    ``聚焦风控`` apart from operational nudges like ``继续吧`` /
    ``下一步``. The follow-up consultant only reads the
    post-finalize intent.md, so any append-time pass-through that the
    finalize curator later drops is invisible to it.
    """
    if not _is_meaningful_intent(user_message):
        if user_message and user_message.strip():
            logger.debug(
                "Skipping non-intent prompt for %s/intent.md: %r",
                analysis_dir,
                user_message.strip()[:80],
            )
        return None
    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        path = analysis_dir / "intent.md"
        # One blank line as a separator between sections — keeps the file
        # readable in any markdown renderer and easy to diff in git.
        section = _format_intent_section(user_message=user_message, mode=mode, timestamp=timestamp)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_text = existing + ("\n" if existing else "") + section
        _atomic_write_text(path, new_text)
        return None
    except Exception as exc:
        # Honour the "best-effort: never raise" contract documented above.
        # Broad catch covers OSError plus UnicodeDecodeError from a corrupt
        # existing intent.md and any other surprise the formatter raises.
        logger.warning("Failed to append to %s: %s", analysis_dir / "intent.md", exc)
        return f"Failed to append intent section: {exc}"


def _format_intent_section(*, user_message: str, mode: str, timestamp: str) -> str:
    """Format a single ``### [timestamp] mode: ...`` block with the user
    message rendered as a blockquote. Leading / trailing whitespace on
    the message is trimmed; internal newlines become continuation
    blockquote lines so multi-paragraph prompts stay legible."""
    body_lines = [f"> {line}" if line.strip() else ">" for line in user_message.strip().splitlines()]
    return f"### [{timestamp}] mode: {mode}\n" + "\n".join(body_lines) + "\n"


def upsert_manifest_after_save(
    manifest_path: Path,
    *,
    datasource: Optional[str],
    timestamp: str,
) -> Optional[str]:
    """Bump ``updated_at`` and union-add ``datasource`` into ``manifest.datasources``.

    Called by ``save_query`` / ``save_query_template`` after the query
    has been persisted. Reads the existing manifest, validates it,
    mutates the two fields, and writes back atomically. Older manifests
    without the new fields deserialize cleanly (defaults kick in).

    Returns an error string on failure, ``None`` on success.

    Missing / corrupt manifest is a hard error here — every artifact
    must have a valid manifest by the time ``save_query`` runs (the
    tool's ``_require_active`` check already guards against the
    no-active-artifact case, so reaching here without a file means
    something genuinely went wrong).
    """
    try:
        if not manifest_path.is_file():
            return f"manifest missing at {manifest_path.name} — cannot upsert datasources"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"manifest is corrupt: {exc}"
        try:
            manifest = ArtifactManifest.model_validate(raw)
        except Exception as exc:
            return f"manifest schema validation failed: {exc}"

        changed = False
        if datasource:
            label = datasource.strip()
            if label and label not in manifest.datasources:
                manifest.datasources.append(label)
                changed = True
        # Always bump updated_at — even if datasources didn't change, the
        # mutation that triggered this call is meaningful (a query was
        # rewritten in place against the same datasource).
        manifest.updated_at = timestamp
        changed = True

        if changed:
            _atomic_write_text(
                manifest_path,
                json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2) + "\n",
            )
        return None
    except Exception as exc:
        # Honour the "best-effort: never raise" contract documented above.
        logger.warning("Failed to upsert %s: %s", manifest_path, exc)
        return f"Failed to upsert manifest: {exc}"


def write_query_brief(
    queries_dir: Path,
    *,
    name: str,
    hypothesis: str,
    uses: SubjectRefIds,
    caveats: str,
) -> Optional[str]:
    """Write the per-query brief sidecar ``queries/<name>.brief.json``.

    Sibling to ``<name>.sql`` (report) / ``<name>.sql.j2`` (dashboard).
    Write-once-overwrite — rerunning a same-named query replaces this
    file along with the SQL / data files.

    Returns an error string on failure, ``None`` on success.
    """
    try:
        brief = QueryBrief(
            name=name,
            hypothesis=hypothesis,
            uses=uses,
            caveats=caveats,
        )
    except Exception as exc:
        return f"query brief schema validation failed: {exc}"
    try:
        path = queries_dir / f"{name}.brief.json"
        _atomic_write_text(
            path,
            json.dumps(brief.model_dump(), ensure_ascii=False, indent=2) + "\n",
        )
        return None
    except Exception as exc:
        # Honour the "best-effort: never raise" contract documented above.
        logger.warning("Failed to write %s: %s", queries_dir / f"{name}.brief.json", exc)
        return f"Failed to write query brief: {exc}"


def coerce_uses_arg(uses: Any) -> SubjectRefIds:
    """Normalize an LLM-supplied ``uses`` argument into a :class:`SubjectRefIds`.

    Tool framework deserializes function args from the LLM as plain
    JSON-compatible Python, so by the time we see ``uses`` it's almost
    always a ``dict`` (or ``None`` if the LLM omitted the field). We
    accept either and let Pydantic catch obviously-malformed shapes.

    Unknown subject kinds (anything outside ``metrics`` /
    ``reference_sql`` / ``ext_knowledge``) are dropped silently with a
    warning so a forward-looking LLM that learns about a future bucket
    doesn't block the call.
    """
    if uses is None:
        return SubjectRefIds()
    if isinstance(uses, SubjectRefIds):
        return uses
    if not isinstance(uses, dict):
        raise ValueError(f"uses must be a JSON object with kind→ids; got {type(uses).__name__}")
    known_kinds = {"metrics", "reference_sql", "ext_knowledge"}
    cleaned: dict[str, List[str]] = {}
    for kind, ids in uses.items():
        if kind not in known_kinds:
            logger.warning("Dropping unknown uses kind %r (not in %s)", kind, sorted(known_kinds))
            continue
        if ids is None:
            cleaned[kind] = []
            continue
        if not isinstance(ids, list):
            raise ValueError(f"uses.{kind} must be a list of strings; got {type(ids).__name__}")
        for item in ids:
            if not isinstance(item, str):
                raise ValueError(f"uses.{kind} entries must be strings; got {type(item).__name__}")
        cleaned[kind] = list(ids)
    return SubjectRefIds(**cleaned)


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
