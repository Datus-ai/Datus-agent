"""Utilities for the enhanced user-message envelope.

A user message that carries extra context is wrapped as plain text::

    <system_reminder>{enhanced context}</system_reminder>
    {original user input}

The opening tag must appear at the very start of the string and the closing
tag is immediately followed by a newline.  When there is no enhanced context
the message is sent as a plain string without any tag.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_REMINDER_OPEN = "<system_reminder>"
SYSTEM_REMINDER_CLOSE = "</system_reminder>\n"

# Anthropic / OpenAI content-block ``type`` values that carry plain text
# inside ``text`` (Anthropic) or ``text``/``content`` (OpenAI) fields.
_TEXT_BLOCK_TYPES = ("text", "output_text", "input_text")


def build_structured_content(enhanced: str, user_input: str) -> str:
    """Wrap ``enhanced`` and ``user_input`` into the system-reminder envelope.

    Callers are responsible for skipping this helper when ``enhanced`` is empty;
    the helper itself unconditionally emits the tag. ``enhanced`` must not
    contain a literal ``SYSTEM_REMINDER_OPEN`` / ``SYSTEM_REMINDER_CLOSE``
    substring — ``extract_*`` splits on the first occurrence of the close
    marker, so a nested tag corrupts round-tripping. ``enhanced`` is system-
    generated today (knowledge bases, schema context, plan-mode workflow), so
    the warning is diagnostic rather than fatal.
    """
    if SYSTEM_REMINDER_CLOSE in enhanced or SYSTEM_REMINDER_OPEN in enhanced:
        logger.warning(
            "build_structured_content: 'enhanced' contains a system_reminder "
            "tag; round-tripping via extract_* will truncate at the inner "
            "occurrence."
        )
    return f"{SYSTEM_REMINDER_OPEN}{enhanced}{SYSTEM_REMINDER_CLOSE}{user_input}"


def is_structured_content(content: Any) -> bool:
    """Return ``True`` when *content* is a complete system-reminder envelope."""
    return isinstance(content, str) and content.startswith(SYSTEM_REMINDER_OPEN) and SYSTEM_REMINDER_CLOSE in content


def extract_user_input(content: Any) -> str:
    """Extract the original user input from *content*.

    Supports three input shapes:

    - **List of provider content blocks** (Anthropic ``[{"type":"text","text":"..."}]``
      or OpenAI ``output_text``/``input_text`` blocks). Persisted by
      ``ClaudeModel._generate_with_mcp_stream`` for OAuth multi-turn sessions.
      Concatenates the text fields with newlines.
    - **System-reminder envelope** ``<system_reminder>...</system_reminder>\\n{user}``.
      Returns the text after the closing tag.
    - **Plain string**. Returned unchanged.

    Always returns a ``str`` so downstream pydantic models / display layers
    never receive a list.
    """
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
        return "\n".join(texts)

    if not isinstance(content, str):
        return "" if content is None else str(content)

    if is_structured_content(content):
        return content.split(SYSTEM_REMINDER_CLOSE, 1)[1]
    return content


def extract_enhanced_context(content: Any) -> Optional[str]:
    """Extract the enhanced context from *content*.

    Returns ``None`` if the content is not a system-reminder envelope.
    """
    if not is_structured_content(content):
        return None
    start = len(SYSTEM_REMINDER_OPEN)
    end = content.index(SYSTEM_REMINDER_CLOSE)
    return content[start:end]
