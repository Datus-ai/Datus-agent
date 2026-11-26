# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Dedicated persistent-memory tool.

Appends and edits a node's single ``MEMORY.md`` file
(``{workspace_root}/.datus/memory/{node}/MEMORY.md``) through serialized mutations.
"""

from __future__ import annotations

from typing import List

from agents import Tool

from datus.tools import BaseTool
from datus.tools.func_tool import FuncToolResult
from datus.tools.registry.contracts import tool_contract
from datus.utils.loggings import get_logger
from datus.utils.memory_loader import (
    MEMORY_BYTE_LIMIT,
    apply_single_replacement,
    memory_mutation_lock,
    memory_size_warning,
    read_memory_raw,
    write_memory_raw,
)

logger = get_logger(__name__)


class MemoryFuncTool(BaseTool):
    """Append to or edit a single node's persistent ``MEMORY.md``.

    The tool is bound to exactly one ``memory_node`` (the memory owner): ``chat``
    for the chat node, the subagent's own name for custom subagents, or the
    caller node for the feedback node. There is no read-only inheritance here —
    a parent's memory reaches a child purely by being inlined into the child's
    system prompt; children that may not write simply do not mount this tool.
    """

    permission_category: str = "memory_tools"

    tool_name = "memory"
    tool_description = "Append to or edit the agent's persistent MEMORY.md with a 20 KB soft limit."

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
        result = {
            "message": message,
            "memory": content,
            "used_bytes": used,
            "remaining_budget": max(0, MEMORY_BYTE_LIMIT - used),
        }
        warning = memory_size_warning(used)
        if warning:
            result["warning"] = warning
        return result

    # ------------------------------------------------------------------- tools

    @tool_contract(access="restricted", direct_call=False)
    def add_memory(self, content: str) -> FuncToolResult:
        """
        Append a new entry to persistent memory.

        Args:
            content: The memory text to append. Keep it to one concise fact per
                call; it is appended on its own line(s) after existing memory.

        Returns:
            FuncToolResult. On success, ``result`` carries the updated memory,
            ``used_bytes`` and ``remaining_budget``. Writes above the 20 KB soft
            limit succeed and include a size warning.
        """
        try:
            if not content or not content.strip():
                return FuncToolResult(success=0, error="content must not be empty")

            addition = content.rstrip("\n")
            with memory_mutation_lock(self._root_path, self._memory_node):
                existing = read_memory_raw(self._root_path, self._memory_node, strict=True)
                new_raw = f"{existing}\n{addition}" if existing else addition
                write_memory_raw(self._root_path, self._memory_node, new_raw)
            return FuncToolResult(result=self._result_payload(new_raw, "Memory saved."))
        except Exception as exc:
            logger.error(f"MemoryFuncTool.add_memory failed for node '{self._memory_node}': {exc}")
            return FuncToolResult(success=0, error=str(exc))

    @tool_contract(access="restricted", direct_call=False)
    def edit_memory(self, old_string: str, new_string: str) -> FuncToolResult:
        """
        Replace a unique occurrence of ``old_string`` in memory with ``new_string``.

        Use this to update or prune existing memory. ``old_string`` must match
        exactly once in the current file; pass an empty ``new_string`` to delete
        the matched text.

        Args:
            old_string: Exact existing text to find. Must match exactly once.
            new_string: Replacement text. Empty string deletes the match.

        Returns:
            FuncToolResult. On success, ``result`` carries the updated memory,
            ``used_bytes`` and ``remaining_budget``. Edits above the 20 KB soft
            limit succeed and include a size warning. An absent or ambiguous
            match leaves the file unchanged.
        """
        try:
            if not old_string:
                return FuncToolResult(success=0, error="old_string must not be empty")

            with memory_mutation_lock(self._root_path, self._memory_node):
                existing = read_memory_raw(self._root_path, self._memory_node, strict=True)
                if not existing.strip():
                    return FuncToolResult(success=0, error="Memory is empty; nothing to edit.")
                new_content, error = apply_single_replacement(existing, old_string, new_string)
                if error is not None:
                    return FuncToolResult(success=0, error=error)
                write_memory_raw(self._root_path, self._memory_node, new_content)
            return FuncToolResult(result=self._result_payload(new_content, "Memory updated."))
        except Exception as exc:
            logger.error(f"MemoryFuncTool.edit_memory failed for node '{self._memory_node}': {exc}")
            return FuncToolResult(success=0, error=str(exc))
