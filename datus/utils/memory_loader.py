# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Memory loader utilities for persistent agent memory.

Provides functions to determine memory eligibility and to read/write memory
content for agentic nodes. Memory is a single file stored at
``{workspace_root}/.datus/memory/{subagent}/MEMORY.md``.

Above the 20 KB soft limit the agent is asked to consolidate. Agent writes that
would grow the file past the 25 KB prompt limit are rejected, so the file stays
within what the prompt shows. Files edited by hand can still exceed it; the
prompt then shows the leading complete lines, and edits that shrink such a file
are allowed even though they may touch content beyond the excerpt. Agent writes
use a per-file lock and atomic replacement.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple

from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Soft limit: writes above it succeed and ask the agent to consolidate.
MEMORY_BYTE_LIMIT = 20_000
# The system prompt shows at most this many bytes, so agent writes that would
# grow the file past it are rejected.
MEMORY_CONTEXT_BYTE_LIMIT = 25_000
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
    Built-in system subagents (gen_sql, gen_visual_report, feedback, etc.), explore, and
    compare do not get their own memory file. The feedback node updates the caller
    node's memory instead of maintaining its own.
    """
    if node_name in _MEMORY_ENABLED_BUILTINS:
        return True
    return node_name not in _ALL_BUILTIN_NODES


def resolve_memory_node(node_name: str) -> str:
    """Map a node to the memory file it owns when it runs as a *main* agent.

    Built-in system nodes (gen_sql, gen_visual_report, explore, …) do not keep their own
    memory — when one runs as the top-level interactive agent it reads/writes the
    shared ``chat`` memory. Custom agents own a memory file under their own name.
    Subagents never write memory, so this is only consulted on the main-agent path.
    """
    return node_name if node_name not in _ALL_BUILTIN_NODES else "chat"


def memory_size_warning(used_bytes: int) -> str:
    """Return consolidation guidance for a file above the soft limit."""
    if used_bytes <= MEMORY_BYTE_LIMIT:
        return ""
    return (
        f"{MEMORY_FILENAME} is {used_bytes} UTF-8 bytes, above the {MEMORY_BYTE_LIMIT}-byte soft limit. "
        f"Consolidate stale or duplicate entries with edit_memory until it is below {MEMORY_BYTE_LIMIT} bytes. "
        f"Writes that would grow it past {MEMORY_CONTEXT_BYTE_LIMIT} bytes are rejected."
    )


def memory_limit_error(projected_bytes: int, current_bytes: int) -> str:
    """Explain a write rejected by the prompt limit."""
    return (
        f"Memory not saved: {MEMORY_FILENAME} would be {projected_bytes} UTF-8 bytes, above the "
        f"{MEMORY_CONTEXT_BYTE_LIMIT}-byte hard limit (currently {current_bytes} bytes). "
        f"Consolidate stale or duplicate entries with edit_memory until it is below {MEMORY_BYTE_LIMIT} bytes, "
        "then retry."
    )


def _leading_complete_lines(data: bytes, limit: int) -> str:
    """Decode the longest run of whole lines from ``data`` that fits in ``limit`` bytes.

    Falls back to a cut at a character boundary when the first line alone is too long.
    """
    # One extra byte shows whether the limit falls exactly at the end of a line.
    prefix = data[: limit + 1].decode("utf-8", errors="ignore")
    cut_at = prefix.rfind("\n")
    if cut_at > 0:
        return prefix[:cut_at]
    return data[:limit].decode("utf-8", errors="ignore")


def format_memory_context(content: str, *, read_only: bool = False) -> tuple[str, str]:
    """Return the prompt excerpt and a notice for the agent.

    Newlines are normalized before measuring, so the size matches the one the memory
    tools report. Read-only readers cannot consolidate, so they only get told when
    the excerpt is partial.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    data = content.encode("utf-8")
    used_bytes = len(data)
    notices = [] if read_only else [memory_size_warning(used_bytes)]
    if used_bytes > MEMORY_CONTEXT_BYTE_LIMIT:
        content = _leading_complete_lines(data, MEMORY_CONTEXT_BYTE_LIMIT)
        visible_bytes = len(content.encode("utf-8"))
        notices.append(
            f"This excerpt shows the first {visible_bytes} of {used_bytes} bytes; later entries are hidden. "
            "Treat the excerpt as partial."
        )
    return content, " ".join(notice for notice in notices if notice)


def get_memory_file_path(workspace_root: str, memory_node: str) -> Path:
    """Absolute path to a node's single MEMORY.md file."""
    return Path(workspace_root) / MEMORY_BASE_DIR / memory_node / MEMORY_FILENAME


@contextmanager
def memory_mutation_lock(workspace_root: str, memory_node: str) -> Iterator[None]:
    """Serialize the entire read-modify-write across threads and processes."""
    memory_file = get_memory_file_path(workspace_root, memory_node).resolve()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode stable across atomic replacement of MEMORY.md.
    # Never unlink it on exit: another process may already be waiting on it.
    lock_path = memory_file.with_name(f".{memory_file.name}.lock")
    if os.name != "posix":
        from datus.storage.semantic_model.artifact_file import windows_file_lock

        with windows_file_lock(lock_path):
            yield
        return

    import fcntl

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def read_memory_raw(workspace_root: str, memory_node: str, *, strict: bool = False) -> str:
    """Read MEMORY.md with universal newlines, returning empty text when absent.

    Mutations use strict reads to avoid replacing unreadable memory with an empty file.
    """
    memory_file = get_memory_file_path(workspace_root, memory_node)
    try:
        with memory_file.open(encoding="utf-8") as stream:
            return stream.read()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as exc:
        if strict:
            raise
        logger.warning(f"Failed to read memory file {memory_file}: {exc}")
        return ""


def write_memory_raw(workspace_root: str, memory_node: str, content: str) -> None:
    """Replace MEMORY.md atomically. Hold the mutation lock for read-modify-write.

    A new file gets the mode a plain write would give it under the process umask;
    an existing file keeps its mode.
    """
    from datus.storage.semantic_model.artifact_file import atomic_write_text

    memory_file = get_memory_file_path(workspace_root, memory_node).resolve()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The kernel applies the umask here; the atomic replacement then keeps this mode.
        os.close(os.open(memory_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    except FileExistsError:
        pass
    atomic_write_text(memory_file, content)


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
    """Load the full MEMORY.md for a subagent. ``format_memory_context`` bounds it for the prompt."""
    return read_memory_raw(workspace_root, subagent_name)


def get_memory_dir(workspace_root: str, subagent_name: str) -> str:
    """Get relative memory directory path (relative to workspace_root)."""
    return f"{MEMORY_BASE_DIR}/{subagent_name}"
