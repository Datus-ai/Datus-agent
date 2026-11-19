# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool names a sub-agent's ``tools`` list may still use after the tool was replaced.

Every node resolves its ``tools`` list its own way, and each drops a name it
does not know (``ask_metrics`` logs it, ``chat`` does not). A configuration
written for a replaced tool would quietly lose the capability. So a replaced
name is expanded to its successors where the list is read — ``AgentConfig``
loading ``agentic_nodes`` and ``SubAgentConfig.tools`` — before any node sees
it, with one warning per sub-agent. The configuration file is not rewritten on
load. A permission rule written for a replaced name applies to its successors
too (``replaced_names``), so a denied tool is not allowed again under its new
names.

Remove an entry one release after it is added: the old name then stops
working like any other unknown tool.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Set, Tuple

from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Old name -> its successors, as they are written in a ``tools`` list.
REPLACED_TOOLS: Dict[str, Tuple[str, ...]] = {
    "semantic_tools.attribution_analyze": (
        "semantic_tools.attribute_metrics",
        "semantic_tools.attribute_dimensions",
    ),
}

_warned: Set[Tuple[str, str]] = set()
_warned_lock = threading.Lock()


def _warn_once(node_name: str, old: str, new: Tuple[str, ...]) -> None:
    key = (node_name, old)
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    successors = " and ".join(f"`{name}`" for name in new)
    logger.warning(
        f"agentic_nodes.{node_name}.tools: `{old}` has been replaced by {successors}; "
        "both are enabled for this run. Rename it in agent.yml — the old name stops working in the next release."
    )


def expand_replaced_tools(node_name: str, tools: Any) -> Any:
    """``tools`` with every replaced name swapped for its successors.

    ``tools`` is a ``tools`` value as configured: a comma-separated string, or
    a list whose strings may each hold several comma-separated names; a string
    comes back as a string, a list as a list of single names. Entries that are
    not strings are kept where they were, for the caller to handle as before.
    Anything else, and a value without a replaced name, is returned as is.
    """
    if isinstance(tools, str):
        entries: List[Any] = [tools]
    elif isinstance(tools, (list, tuple)):
        entries = list(tools)
    else:
        return tools
    names = [name.strip() for entry in entries if isinstance(entry, str) for name in entry.split(",")]
    if not any(name in REPLACED_TOOLS for name in names):
        return tools
    expanded: List[Any] = []
    for entry in entries:
        if not isinstance(entry, str):
            expanded.append(entry)
            continue
        for item in entry.split(","):
            name = item.strip()
            successors = REPLACED_TOOLS.get(name)
            if successors is None:
                if name:
                    expanded.append(name)
                continue
            _warn_once(node_name or "<unnamed>", name, successors)
            expanded.extend(successors)
    seen: Set[str] = set()
    unique: List[Any] = []
    for item in expanded:
        if isinstance(item, str):
            if item in seen:
                continue
            seen.add(item)
        unique.append(item)
    if isinstance(tools, str):
        return ", ".join(unique)
    return tuple(unique) if isinstance(tools, tuple) else unique


def replaced_names(category: str, name: str) -> Tuple[str, ...]:
    """The names ``category.name`` replaced, within its category:
    ``("attribution_analyze",)`` for ``semantic_tools.attribute_metrics``."""
    tool = f"{category}.{name}"
    old_names = []
    for old, successors in REPLACED_TOOLS.items():
        old_category, _, old_name = old.partition(".")
        if tool in successors and old_category == category:
            old_names.append(old_name)
    return tuple(old_names)


def _reset_warnings_for_tests() -> None:
    """Forget which sub-agents were warned (tests only)."""
    with _warned_lock:
        _warned.clear()
