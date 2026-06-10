# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Runtime overrides that let a built-in sub-agent inherit its parent's memory.

When ``SubAgentTaskTool`` launches a built-in sub-agent (gen_sql, gen_report,
explore, ...), that child owns no memory of its own (``has_memory`` is False)
and would otherwise see no MEMORY.md context. This module installs the parent's memory
node name under the child's subagent name for the duration of the child's
execution; ``AgenticNode._inject_memory_context`` consults the override and
renders the parent's memory in read-only mode.

Backed by ``contextvars.ContextVar`` so concurrent ``asyncio.Task`` siblings
remain isolated (each Task copies the current context at creation).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

_OVERRIDES: ContextVar[Optional[Dict[str, str]]] = ContextVar("datus_inherited_memory", default=None)


def _current_map() -> Dict[str, str]:
    return _OVERRIDES.get() or {}


def get_inherited_memory(sub_agent_name: str) -> Optional[str]:
    """Return the parent memory node name pushed for ``sub_agent_name``, if any."""
    return _current_map().get(sub_agent_name)


@contextmanager
def inherited_memory(sub_agent_name: str, parent_memory_node_name: str) -> Iterator[None]:
    new_map = {**_current_map(), sub_agent_name: parent_memory_node_name}
    token = _OVERRIDES.set(new_map)
    try:
        yield
    finally:
        _OVERRIDES.reset(token)
