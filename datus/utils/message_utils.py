"""Utilities for structured user message content.

The structured format stores user messages as a JSON array:
[
  {"type": "user", "content": "原始用户问题"},
  {"type": "enhanced", "content": "Context: ...\\n\\nNow based on the rules above, answer the user question: 原始用户问题"}
]

Callers decide the order; display / session-restore logic always picks the
first element whose ``type`` is ``"user"``.
"""

import json
import logging
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)


class MessagePart(TypedDict):
    """A single part of a structured user message."""

    type: str  # e.g. "user", "enhanced"
    content: str


def build_structured_content(parts: List[MessagePart]) -> str:
    """Serialize a list of message parts into a JSON string.

    Callers are responsible for constructing the parts list and deciding
    order.  The first part with ``type == "user"`` is treated as the
    original user input when the message is later displayed or restored.

    Args:
        parts: Ordered list of message parts.

    Returns:
        A JSON string representing the structured content.
    """
    return json.dumps(parts, ensure_ascii=False)


def is_structured_content(content: str) -> bool:
    """Check whether *content* is in the structured JSON format.

    Returns ``True`` when the content is a JSON array that contains at
    least one element with ``"type": "user"``.
    """
    if not isinstance(content, str) or not content.strip().startswith("["):
        return False
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0:
            return any(isinstance(part, dict) and part.get("type") == "user" for part in parsed)
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return False


def extract_user_input(content: str) -> str:
    """Extract the original user input from *content*.

    If the content is in the structured format, returns the **first**
    ``"user"`` part.  Otherwise returns *content* unchanged (backward-
    compatible with legacy flat-text messages).
    """
    if not is_structured_content(content):
        return content
    try:
        parsed = json.loads(content)
        for part in parsed:
            if isinstance(part, dict) and part.get("type") == "user":
                return part.get("content", content)
    except (json.JSONDecodeError, TypeError):
        pass
    return content


def extract_enhanced_context(content: str) -> Optional[str]:
    """Extract the enhanced context from *content*.

    Returns ``None`` if the content is not in the structured format or
    no ``"enhanced"`` part is found.
    """
    if not is_structured_content(content):
        return None
    try:
        parsed = json.loads(content)
        for part in parsed:
            if isinstance(part, dict) and part.get("type") == "enhanced":
                return part.get("content")
    except (json.JSONDecodeError, TypeError):
        pass
    return None
