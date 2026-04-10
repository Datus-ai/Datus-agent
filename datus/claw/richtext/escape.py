# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Smart Slack escape that preserves native Slack tokens."""

from __future__ import annotations

import re

_SLACK_TOKEN_RE = re.compile(r"<[^>\n]+>")

_SLACK_PREFIXES = ("@", "#", "!", "http://", "https://", "mailto:", "tel:", "slack://")


def slack_escape(text: str) -> str:
    """Escape ``&``, ``<``, ``>`` but preserve Slack native tokens.

    Slack tokens such as ``<@U123>``, ``<#C123>``, ``<!here>``,
    ``<https://example.com|label>`` are left untouched.
    """
    parts: list[str] = []
    last_end = 0

    for match in _SLACK_TOKEN_RE.finditer(text):
        inner = match.group()[1:-1]  # strip < >
        if any(inner.startswith(prefix) for prefix in _SLACK_PREFIXES):
            # Preserve this Slack token — escape text before it
            parts.append(_escape_plain(text[last_end : match.start()]))
            parts.append(match.group())
            last_end = match.end()

    # Escape remaining text after the last token
    parts.append(_escape_plain(text[last_end:]))
    return "".join(parts)


def _escape_plain(text: str) -> str:
    """Escape &, <, > in plain text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
