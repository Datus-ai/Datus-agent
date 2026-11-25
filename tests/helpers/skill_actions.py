"""Inspect ``load_skill`` tool actions recorded by any model adapter."""

import json
from typing import Any, Iterable, List


def loaded_skill_name(action: Any) -> str | None:
    """Return the ``skill_name`` argument of a ``load_skill`` action, if any.

    OpenAI-compatible adapters record arguments as a JSON string; Claude records a dict.
    """
    payload = action.input if isinstance(action.input, dict) else {}
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    name = arguments.get("skill_name")
    return name if isinstance(name, str) else None


def successful_skill_loads(actions: Iterable[Any], skill_name: str) -> List[Any]:
    """Successful ``load_skill`` tool actions that loaded exactly ``skill_name``."""
    return [
        action
        for action in actions
        if action.role == "tool"
        and action.action_type == "load_skill"
        and action.status == "success"
        and loaded_skill_name(action) == skill_name
    ]
