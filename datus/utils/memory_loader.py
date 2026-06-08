# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Memory loader utilities for persistent agent memory.

Provides functions to determine memory eligibility and to read/write memory
content for agentic nodes. Memory is a single file stored at
``{workspace_root}/.datus/memory/{subagent}/MEMORY.md``.

There is no progressive disclosure: memory is one flat file bounded by a hard
2000-byte cap, so the whole file always fits inside the system prompt. The
dedicated ``add_memory`` / ``edit_memory`` tools (see
``datus/tools/func_tool/memory_tools.py``) are the only writers and enforce the
cap on write; the load-time truncation here is a defensive guard for files that
were edited externally or predate the cap.
"""

from pathlib import Path
from typing import Optional, Tuple

from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Single hard byte cap for the whole MEMORY.md file. The add_memory/edit_memory
# tools reject writes that would exceed it; load_memory_context truncates as a
# last-resort defense.
MEMORY_BYTE_LIMIT = 2000
MEMORY_FILENAME = "MEMORY.md"
MEMORY_BASE_DIR = ".datus/memory"

# Allowlist: the only built-in node that gets memory.
# Any name NOT found in _ALL_BUILTIN_NODES is treated as a custom subagent and also gets memory.
# Note: 'feedback' intentionally omitted — the feedback node's purpose is to update the
# caller's memory (e.g. chat), not to maintain a memory file of its own.
_MEMORY_ENABLED_BUILTINS = frozenset({"chat"})
_ALL_BUILTIN_NODES = SYS_SUB_AGENTS | {"explore", "compare"}


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


def resolve_memory_node(node_name: str) -> str:
    """Map a node to the memory file it owns when it runs as a *main* agent.

    Built-in system nodes (gen_sql, gen_report, explore, …) do not keep their own
    memory — when one runs as the top-level interactive agent it reads/writes the
    shared ``chat`` memory. Custom agents own a memory file under their own name.
    Subagents never write memory, so this is only consulted on the main-agent path.
    """
    return node_name if node_name not in _ALL_BUILTIN_NODES else "chat"


def _truncate_to_byte_limit(raw: str) -> str:
    """Truncate ``raw`` to ``MEMORY_BYTE_LIMIT`` bytes, on a newline boundary.

    Defensive load-time guard only — the write tools already enforce the cap.
    Cuts at the last newline before the byte cap so an entry is not sliced
    mid-line, falling back to a hard byte cut when there is no newline in range.
    A short warning is appended so the model knows the file was clipped.
    """
    trimmed = raw.rstrip("\r\n")
    if len(trimmed.encode("utf-8")) <= MEMORY_BYTE_LIMIT:
        return trimmed

    # Reserve room for the appended notice so the final string still fits the
    # cap. If the warning itself somehow exceeds the budget, fall back to a bare
    # byte cut with no notice rather than breach the limit.
    warning = f"\n\n> WARNING: {MEMORY_FILENAME} exceeded {MEMORY_BYTE_LIMIT} bytes and was truncated at load time."
    warning_bytes = len(warning.encode("utf-8"))
    budget = MEMORY_BYTE_LIMIT - warning_bytes
    if budget <= 0:
        return trimmed.encode("utf-8")[:MEMORY_BYTE_LIMIT].decode("utf-8", errors="ignore")

    encoded = trimmed.encode("utf-8")[:budget]
    # Decode ignoring a possibly-split trailing multi-byte char.
    safe = encoded.decode("utf-8", errors="ignore")
    cut_at = safe.rfind("\n")
    content = safe[:cut_at] if cut_at > 0 else safe

    return content + warning


def get_memory_file_path(workspace_root: str, memory_node: str) -> Path:
    """Absolute path to a node's single MEMORY.md file."""
    return Path(workspace_root) / MEMORY_BASE_DIR / memory_node / MEMORY_FILENAME


def read_memory_raw(workspace_root: str, memory_node: str) -> str:
    """Read a node's raw MEMORY.md content. Returns '' when the file is absent."""
    memory_file = get_memory_file_path(workspace_root, memory_node)
    if not memory_file.exists():
        return ""
    try:
        return memory_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        logger.warning(f"Failed to read memory file {memory_file}: {exc}")
        return ""


def write_memory_raw(workspace_root: str, memory_node: str, content: str) -> None:
    """Write a node's MEMORY.md content, creating parent directories as needed."""
    memory_file = get_memory_file_path(workspace_root, memory_node)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(content, encoding="utf-8")


def apply_single_replacement(content: str, old_string: str, new_string: str) -> Tuple[Optional[str], Optional[str]]:
    """Replace a single unique occurrence of ``old_string`` with ``new_string``.

    Pure (no IO). Returns ``(new_content, None)`` on success or
    ``(None, error_message)`` when the replacement cannot be applied uniquely.
    An empty ``new_string`` deletes the matched text. Shared by
    ``MemoryFuncTool.edit_memory`` and ``FilesystemFuncTool.edit_file`` so both
    enforce identical match semantics.
    """
    if not old_string:
        return None, "old_string must not be empty"

    match_count = content.count(old_string)
    if match_count == 0:
        preview = old_string[:100] + "..." if len(old_string) > 100 else old_string
        return None, f"old_string not found. Looking for: {preview}"
    if match_count > 1:
        return None, (
            f"old_string matches {match_count} times. It must match exactly once. "
            "Provide more surrounding context to make the match unique."
        )

    return content.replace(old_string, new_string, 1), None


def load_memory_context(workspace_root: str, subagent_name: str) -> str:
    """Load MEMORY.md for a subagent. Returns empty string if not found.

    Applies the defensive 2000-byte cap; the write tools already enforce it on
    write, so truncation only fires for externally-edited or legacy files.
    """
    raw = read_memory_raw(workspace_root, subagent_name)
    if not raw.strip():
        return ""
    return _truncate_to_byte_limit(raw)


def get_memory_dir(workspace_root: str, subagent_name: str) -> str:
    """Get relative memory directory path (relative to workspace_root)."""
    return f"{MEMORY_BASE_DIR}/{subagent_name}"
