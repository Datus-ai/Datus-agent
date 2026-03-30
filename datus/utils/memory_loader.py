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

MEMORY_LINE_LIMIT = 200
MEMORY_FILENAME = "MEMORY.md"
MEMORY_BASE_DIR = ".datus/memory"

# Nodes that should NOT have persistent memory (builtin system subagents + explore)
_NO_MEMORY_NODES = SYS_SUB_AGENTS | {"explore"}


def has_memory(node_name: str) -> bool:
    """Determine if a node should have persistent memory."""
    return node_name not in _NO_MEMORY_NODES


def load_memory_context(workspace_root: str, subagent_name: str) -> str:
    """Load and truncate MEMORY.md for a subagent. Returns empty string if not found."""
    memory_file = Path(workspace_root) / MEMORY_BASE_DIR / subagent_name / MEMORY_FILENAME
    if not memory_file.exists():
        return ""
    lines = memory_file.read_text(encoding="utf-8").splitlines()
    if len(lines) > MEMORY_LINE_LIMIT:
        lines = lines[:MEMORY_LINE_LIMIT]
        lines.append(f"\n... (truncated at {MEMORY_LINE_LIMIT} lines, move details to sub-files)")
    return "\n".join(lines)


def get_memory_dir(workspace_root: str, subagent_name: str) -> str:
    """Get relative memory directory path (relative to workspace_root)."""
    return f"{MEMORY_BASE_DIR}/{subagent_name}"
