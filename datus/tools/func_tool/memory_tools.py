# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Dedicated persistent-memory tool.

Exposes ``add_memory`` / ``edit_memory`` as the *only* writers of a node's
single ``MEMORY.md`` file (``{workspace_root}/.datus/memory/{node}/MEMORY.md``).
The file is a flat, hard-capped (2000-byte) document — no progressive
disclosure, no topic files — so its whole content is always inlined into the
system prompt. Generic filesystem tools are blocked from the memory subtree
(see ``fs_path_policy``), so this tool is the single entry point and the byte
cap cannot be bypassed.
"""

from __future__ import annotations

from typing import List

from agents import Tool

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.utils.loggings import get_logger
from datus.utils.memory_loader import (
    MEMORY_BYTE_LIMIT,
    apply_single_replacement,
    read_memory_raw,
    write_memory_raw,
)

logger = get_logger(__name__)


class MemoryFuncTool(BaseTool):
    """Read/append/edit a single node's persistent ``MEMORY.md`` with a hard byte cap.

    The tool is bound to exactly one ``memory_node`` (the memory owner): ``chat``
    for the chat node, the subagent's own name for custom subagents, or the
    caller node for the feedback node. There is no read-only inheritance here —
    a parent's memory reaches a child purely by being inlined into the child's
    system prompt; children that may not write simply do not mount this tool.
    """

    tool_name = "memory"
    tool_description = "Append to or edit the agent's persistent memory (single MEMORY.md, hard byte cap)."

    def __init__(self, *, root_path: str, memory_node: str, **kwargs):
        """
        Args:
            root_path: Workspace root the memory subtree is anchored under.
            memory_node: The memory owner node name. Resolves to
                ``{root_path}/.datus/memory/{memory_node}/MEMORY.md``.
        """
        super().__init__(**kwargs)
        self._root_path = root_path
        self._memory_node = memory_node

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def memory_node(self) -> str:
        return self._memory_node

    def available_tools(self) -> List[Tool]:
        from datus.tools.func_tool import trans_to_function_tool

        return [
            trans_to_function_tool(self.add_memory),
            trans_to_function_tool(self.edit_memory),
        ]

    @staticmethod
    def all_tools_name() -> List[str]:
        return ["add_memory", "edit_memory"]

    # ----------------------------------------------------------------- helpers

    def _result_payload(self, content: str, message: str) -> dict:
        used = len(content.encode("utf-8"))
        return {
            "message": message,
            "memory": content,
            "used_bytes": used,
            "remaining_budget": MEMORY_BYTE_LIMIT - used,
        }

    # ------------------------------------------------------------------- tools

    def add_memory(self, content: str) -> FuncToolResult:
        """
        Append a new entry to persistent memory.

        Args:
            content: The memory text to append. Keep it to one concise fact per
                call; it is appended on its own line(s) after existing memory.

        Returns:
            FuncToolResult. On success, ``result`` carries the updated memory,
            ``used_bytes`` and ``remaining_budget``. When appending would exceed
            the 2000-byte hard cap, nothing is written and ``success=0`` with
            guidance to free space via ``edit_memory`` first.
        """
        try:
            if not content or not content.strip():
                return FuncToolResult(success=0, error="content must not be empty")

            existing = read_memory_raw(self._root_path, self._memory_node)
            addition = content.rstrip("\n")
            new_raw = f"{existing}\n{addition}" if existing else addition

            projected = len(new_raw.encode("utf-8"))
            if projected > MEMORY_BYTE_LIMIT:
                overflow = projected - MEMORY_BYTE_LIMIT
                current = len(existing.encode("utf-8"))
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Memory is full: adding this content would make MEMORY.md {projected} bytes, "
                        f"exceeding the {MEMORY_BYTE_LIMIT}-byte hard limit by {overflow} bytes "
                        f"(currently {current} bytes). The content was NOT saved. "
                        f'First call edit_memory(old_string=<an existing stale entry>, new_string="") '
                        f"to free at least {overflow} bytes, then retry add_memory. "
                        f"The current memory is shown in your system prompt."
                    ),
                )

            write_memory_raw(self._root_path, self._memory_node, new_raw)
            return FuncToolResult(result=self._result_payload(new_raw, "Memory saved."))
        except Exception as exc:
            logger.error(f"MemoryFuncTool.add_memory failed for node '{self._memory_node}': {exc}")
            return FuncToolResult(success=0, error=str(exc))

    def edit_memory(self, old_string: str, new_string: str) -> FuncToolResult:
        """
        Replace a unique occurrence of ``old_string`` in memory with ``new_string``.

        Use this to update or prune existing memory. ``old_string`` must match
        exactly once; pass an empty ``new_string`` to delete the matched text
        (the primary way to free space when add_memory reports the file is full).

        Args:
            old_string: Exact existing text to find. Must match exactly once.
            new_string: Replacement text. Empty string deletes the match.

        Returns:
            FuncToolResult. On success, ``result`` carries the updated memory,
            ``used_bytes`` and ``remaining_budget``. Rejected (nothing written)
            when memory is empty, the match is not unique, or the edit would
            push the file past the 2000-byte cap.
        """
        try:
            if not old_string:
                return FuncToolResult(success=0, error="old_string must not be empty")

            existing = read_memory_raw(self._root_path, self._memory_node)
            if not existing.strip():
                return FuncToolResult(success=0, error="Memory is empty; nothing to edit.")

            new_content, error = apply_single_replacement(existing, old_string, new_string)
            if error is not None:
                return FuncToolResult(success=0, error=error)

            projected = len(new_content.encode("utf-8"))
            if projected > MEMORY_BYTE_LIMIT:
                overflow = projected - MEMORY_BYTE_LIMIT
                return FuncToolResult(
                    success=0,
                    error=(
                        f"Edit rejected: the result would be {projected} bytes, exceeding the "
                        f"{MEMORY_BYTE_LIMIT}-byte hard limit by {overflow} bytes. Nothing was written. "
                        f"Use a shorter new_string or delete other entries first."
                    ),
                )

            write_memory_raw(self._root_path, self._memory_node, new_content)
            return FuncToolResult(result=self._result_payload(new_content, "Memory updated."))
        except Exception as exc:
            logger.error(f"MemoryFuncTool.edit_memory failed for node '{self._memory_node}': {exc}")
            return FuncToolResult(success=0, error=str(exc))
