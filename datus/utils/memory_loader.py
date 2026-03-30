# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Memory loader utilities for persistent agent memory.

Provides functions to determine memory eligibility and load memory content
for agentic nodes. Memory files are stored under {workspace_root}/.datus/memory/{subagent}/.
"""

from pathlib import Path

from datus.utils.constants import SYS_SUB_AGENTS
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MEMORY_LINE_LIMIT = 200
MEMORY_FILENAME = "MEMORY.md"
MEMORY_BASE_DIR = ".datus/memory"

# Allowlist: the only built-in node that gets memory.
# Any name NOT found in _ALL_BUILTIN_NODES is treated as a custom subagent and also gets memory.
_MEMORY_ENABLED_BUILTINS = frozenset({"chat"})
_ALL_BUILTIN_NODES = SYS_SUB_AGENTS | {"explore", "compare"}


def has_memory(node_name: str) -> bool:
    """Determine if a node should have persistent memory.

    Enabled for 'chat' and custom subagents only.
    Built-in system subagents (gen_sql, gen_report, etc.), explore, and compare do not get memory.
    """
    if node_name in _MEMORY_ENABLED_BUILTINS:
        return True
    return node_name not in _ALL_BUILTIN_NODES


def load_memory_context(workspace_root: str, subagent_name: str) -> str:
    """Load and truncate MEMORY.md for a subagent. Returns empty string if not found."""
    memory_file = Path(workspace_root) / MEMORY_BASE_DIR / subagent_name / MEMORY_FILENAME
    if not memory_file.exists():
        return ""
    lines: list[str] = []
    try:
        with memory_file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no >= MEMORY_LINE_LIMIT:
                    lines.append("")
                    lines.append(f"... (truncated at {MEMORY_LINE_LIMIT} lines, move details to sub-files)")
                    break
                lines.append(line.rstrip("\r\n"))
    except (OSError, UnicodeError) as exc:
        logger.warning(f"Failed to load memory file {memory_file}: {exc}")
        return ""
    return "\n".join(lines)


def get_memory_dir(workspace_root: str, subagent_name: str) -> str:
    """Get relative memory directory path (relative to workspace_root)."""
    return f"{MEMORY_BASE_DIR}/{subagent_name}"
