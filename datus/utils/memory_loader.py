# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Memory loader utilities for persistent agent memory.

Provides functions to determine memory eligibility and load memory content
for agentic nodes. Memory files are stored under {workspace_root}/.datus/memory/{subagent}/.

MEMORY.md is an INDEX — each line should be a pointer to a topic file:
    - [Title](topic.md) — one-line hook

Long-form content belongs in the sibling topic files, not in MEMORY.md itself.
Load-time enforcement here is advisory (logger.warning); the prompt template
teaches the model the format, and the hard line/byte caps bound the blast
radius when the model forgets.
"""

import re
from pathlib import Path

from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MEMORY_LINE_LIMIT = 200
# ~125 chars/line × 200 lines. Bounds index-entry bloat that slips past the
# line cap when a single line is pathologically long.
MEMORY_BYTE_LIMIT = 25_000
MEMORY_FILENAME = "MEMORY.md"
MEMORY_BASE_DIR = ".datus/memory"

# Allowlist: the only built-in node that gets memory.
# Any name NOT found in _ALL_BUILTIN_NODES is treated as a custom subagent and also gets memory.
# Note: 'feedback' intentionally omitted — the feedback node's purpose is to update the
# caller's memory (e.g. chat), not to maintain a memory file of its own.
_MEMORY_ENABLED_BUILTINS = frozenset({"chat"})
_ALL_BUILTIN_NODES = SYS_SUB_AGENTS | {"explore", "compare"}

_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
# Any single line beyond this length is guaranteed not to be a concise memory.
_LONG_LINE_CHARS = 200
# A section (content under one `##` heading) above this many lines should be
# split out into its own topic file. Short memories stay inline in MEMORY.md;
# only large memories graduate to their own file with frontmatter.
_SECTION_MAX_LINES = 50


def has_memory(node_name: str) -> bool:
    """Determine if a node should have persistent memory.

    Enabled for 'chat' and custom subagents only.
    Built-in system subagents (gen_sql, gen_report, feedback, etc.), explore, and
    compare do not get their own memory file. The feedback node updates the caller
    node's memory instead of maintaining its own.
    """
    if node_name in _MEMORY_ENABLED_BUILTINS:
        return True
    return node_name not in _ALL_BUILTIN_NODES


def _truncate_entrypoint(raw: str) -> tuple[str, bool, bool]:
    """Apply line-then-byte truncation with a guidance-style warning.

    Returns (content, was_line_truncated, was_byte_truncated). Mirrors the
    Claude Code memdir behavior: line cap first (natural boundary), byte cap
    second (cuts at the last newline before the cap so we don't slice
    mid-entry). The appended warning tells the model which cap fired so it
    knows whether to shorten entries or prune the index.
    """
    trimmed = raw.rstrip("\r\n")
    lines = trimmed.split("\n")
    line_count = len(lines)
    byte_count = len(trimmed.encode("utf-8"))

    was_line_truncated = line_count > MEMORY_LINE_LIMIT
    was_byte_truncated = byte_count > MEMORY_BYTE_LIMIT

    if not was_line_truncated and not was_byte_truncated:
        return trimmed, False, False

    content = "\n".join(lines[:MEMORY_LINE_LIMIT]) if was_line_truncated else trimmed

    if len(content.encode("utf-8")) > MEMORY_BYTE_LIMIT:
        # Walk back to a newline so we don't slice mid-entry. We measure in
        # bytes but slice by characters — cut_index falls back to a hard
        # character cap if there's no newline in range.
        encoded = content.encode("utf-8")[:MEMORY_BYTE_LIMIT]
        # Decode ignoring a possibly-split trailing multi-byte char.
        safe = encoded.decode("utf-8", errors="ignore")
        cut_at = safe.rfind("\n")
        content = safe[:cut_at] if cut_at > 0 else safe

    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit: {MEMORY_BYTE_LIMIT}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MEMORY_LINE_LIMIT})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"

    warning = (
        f"\n\n> WARNING: {MEMORY_FILENAME} truncated — {reason}. "
        f"Keep index entries to one line under ~150 chars; move detail into topic files."
    )
    return content + warning, was_line_truncated, was_byte_truncated


def _advise_on_structure(content: str) -> None:
    """Emit logger.warning hints when MEMORY.md entries are too large to stay inline.

    Advisory only — never raises, never mutates. Two signals:
      1. Any single line longer than _LONG_LINE_CHARS → a concise memory was
         crushed into one line, suggest moving detail into a topic file.
      2. Any section (content between two `##`+ headings) longer than
         _SECTION_MAX_LINES → the memory has outgrown inline storage and
         should be split into its own file with a link left in MEMORY.md.
    """
    lines = content.splitlines()
    section_lines = 0
    reported_section = False
    reported_long_line = False

    for line in lines:
        if not reported_long_line and len(line) > _LONG_LINE_CHARS:
            logger.warning(
                f"{MEMORY_FILENAME} has a line over {_LONG_LINE_CHARS} chars; "
                f"move long content into a topic file and link from MEMORY.md."
            )
            reported_long_line = True

        if _HEADING_RE.match(line):
            section_lines = 0
            continue

        section_lines += 1
        if section_lines > _SECTION_MAX_LINES and not reported_section:
            logger.warning(
                f"{MEMORY_FILENAME} has a section longer than {_SECTION_MAX_LINES} lines; "
                f"split that memory into its own topic file and leave a link in MEMORY.md."
            )
            reported_section = True

        if reported_section and reported_long_line:
            return


def load_memory_context(workspace_root: str, subagent_name: str) -> str:
    """Load and truncate MEMORY.md for a subagent. Returns empty string if not found.

    Applies line + byte caps (200 lines / 25 KB) and emits advisory warnings
    when the file looks like a notes buffer rather than an index.
    """
    memory_file = Path(workspace_root) / MEMORY_BASE_DIR / subagent_name / MEMORY_FILENAME
    if not memory_file.exists():
        return ""

    try:
        raw = memory_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        logger.warning(f"Failed to load memory file {memory_file}: {exc}")
        return ""

    if not raw.strip():
        return ""

    content, _, _ = _truncate_entrypoint(raw)
    _advise_on_structure(content)
    return content


def get_memory_dir(workspace_root: str, subagent_name: str) -> str:
    """Get relative memory directory path (relative to workspace_root)."""
    return f"{MEMORY_BASE_DIR}/{subagent_name}"
