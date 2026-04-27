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
from typing import Any, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# Anthropic / OpenAI content-block ``type`` values that carry plain text
# inside ``text`` (Anthropic) or ``text``/``content`` (OpenAI) fields.
_TEXT_BLOCK_TYPES = ("text", "output_text", "input_text")


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


def extract_user_input(content: Any) -> str:
    """Extract the original user input from *content*.

    Supports three input shapes:

    - **List of provider content blocks** (Anthropic ``[{"type":"text","text":"..."}]``
      or OpenAI ``output_text``/``input_text`` blocks). Persisted by
      ``ClaudeModel._generate_with_mcp_stream`` for OAuth multi-turn sessions.
      Concatenates the text fields with newlines.
    - **JSON-encoded Datus structured string** ``[{"type":"user","content":"..."}]``.
      Returns the first ``"user"`` part's ``content``.
    - **Plain string** (legacy flat-text messages). Returned unchanged.

    Always returns a ``str`` so downstream pydantic models / display layers
    never receive a list.
    """
    # Anthropic-style content blocks (list of dicts) — produced by the native
    # Claude OAuth path. Keep the legacy str-only branches below intact.
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            block_type = part.get("type")
            if block_type in _TEXT_BLOCK_TYPES:
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    texts.append(extract_user_input(text) if is_structured_content(text) else text)
            elif block_type == "user":
                # Datus structured part stored unencoded as a list element.
                user_text = part.get("content")
                if isinstance(user_text, str):
                    return user_text
        return "\n".join(texts)

    if not is_structured_content(content):
        return content if isinstance(content, str) else ("" if content is None else str(content))
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
