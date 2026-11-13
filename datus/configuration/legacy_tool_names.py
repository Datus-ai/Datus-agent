# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tool names a sub-agent's ``tools`` list may still use after the tool was replaced.

Every node resolves its ``tools`` list its own way, and each drops a name it
does not know (``ask_metrics`` logs it, ``chat`` does not). A configuration
written for a replaced tool would quietly lose the capability. So a replaced
name is expanded to its successors where the list is read — ``AgentConfig``
loading ``agentic_nodes`` and ``SubAgentConfig.tools`` — before any node sees
it, with one warning per sub-agent. The configuration file is not rewritten.

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

    ``tools`` is a ``tools`` value as configured: a comma-separated string or a
    list of strings; the same type comes back. Anything else, and a list
    without a replaced name, is returned as is.
    """
    if isinstance(tools, str):
        items = tools.split(",")
    elif isinstance(tools, (list, tuple)) and all(isinstance(item, str) for item in tools):
        items = list(tools)
    else:
        return tools
    if not any(item.strip() in REPLACED_TOOLS for item in items):
        return tools
    expanded: List[str] = []
    for item in items:
        name = item.strip()
        successors = REPLACED_TOOLS.get(name)
        if successors is None:
            if name:
                expanded.append(name)
            continue
        _warn_once(node_name or "<unnamed>", name, successors)
        expanded.extend(successors)
    expanded = list(dict.fromkeys(expanded))
    if isinstance(tools, str):
        return ", ".join(expanded)
    return type(tools)(expanded) if isinstance(tools, tuple) else expanded


def _reset_warnings_for_tests() -> None:
    """Forget which sub-agents were warned (tests only)."""
    with _warned_lock:
        _warned.clear()
