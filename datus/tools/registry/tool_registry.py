# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Shared tool_name -> category registry.

ToolRegistry lives at the AgenticNode level and is shared with PermissionHooks
and proxy_tool so that every consumer uses a single source of truth.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Maps tool_name -> category for all tools registered on a node."""

    def __init__(self, initial: Optional[Dict[str, str]] = None):
        self._registry: Dict[str, str] = dict(initial) if initial else {}

    # ── mutation ──────────────────────────────────────────────────────

    def register_tools(self, category: str, tools: List[Any]) -> None:
        """Register a list of tool objects under *category*.

        Each tool must expose a ``.name`` attribute (or be stringifiable).
        """
        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            self._registry[tool_name] = category
            logger.debug(f"Registered tool '{tool_name}' with category '{category}'")

    # ── read access ───────────────────────────────────────────────────

    def get(self, tool_name: str, default: Optional[str] = None) -> Optional[str]:
        return self._registry.get(tool_name, default)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def items(self) -> list[Tuple[str, str]]:
        return list(self._registry.items())

    def to_dict(self) -> Dict[str, str]:
        """Return a plain-dict copy (useful for serialisation / proxy_tool)."""
        return dict(self._registry)

    # ── dunder helpers ────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ToolRegistry({self._registry!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ToolRegistry):
            return self._registry == other._registry
        if isinstance(other, dict):
            return self._registry == other
        return NotImplemented
