# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Utilities for handling hierarchical reference paths used in @Table/@Metrics/@ReferenceSql completions.
"""

from typing import List

REFERENCE_PATH_REGEX = r'(?:(?:"[^"@\r\n]*"|[^@\s".]+)(?:\.(?:"[^"@\r\n]*"|[^@\s".]+))*)(?:\.)?'


def normalize_reference_path(path: str) -> str:
    """
    Normalize a hierarchical reference path by trimming whitespace, removing trailing punctuation,
    and unquoting the final component when wrapped in double quotes.
    """
    if not path:
        return ""

    text = path.strip()
    if '"' not in text:
        # Already normalized (or no quoted segments). Strip trailing punctuation in a lightweight way.
        return text.rstrip(".,;:!?)]}")

    buffer: List[str] = []
    in_quotes = False
    for ch in text:
        if ch == '"':
            in_quotes = not in_quotes
            buffer.append(ch)
        elif ch.isspace() and not in_quotes:
            # Stop once we hit whitespace outside of a quoted segment
            break
        else:
            buffer.append(ch)

    cleaned = "".join(buffer).rstrip(".,;:!?)]}")
    if not cleaned:
        return ""

    # Split on '.' only when not inside double quotes
    segments: List[str] = []
    seg_buf: List[str] = []
    in_quotes = False
    for ch in cleaned:
        if ch == '"':
            in_quotes = not in_quotes
            seg_buf.append(ch)
        elif ch == "." and not in_quotes:
            segments.append("".join(seg_buf).strip())
            seg_buf = []
        else:
            seg_buf.append(ch)
    if seg_buf:
        segments.append("".join(seg_buf).strip())
    segments = [s for s in segments if s]  # drop empty parts
    if not segments:
        return ""
    # Unquote only the final component if enclosed in double quotes
    last = segments[-1]
    if last.startswith('"') and last.endswith('"') and len(last) >= 2:
        segments[-1] = last[1:-1]
    return ".".join(segments)


def split_reference_path(path: str) -> List[str]:
    """Split a hierarchical reference path into individual unquoted components.

    Unlike :func:`normalize_reference_path` (which only unquotes the last
    segment), this function unquotes **all** double-quoted segments so the
    returned list can be used directly as a ``subject_path`` for store lookups.

    Examples::

        split_reference_path('domain.layer1."name with spaces"')
        # => ["domain", "layer1", "name with spaces"]

        split_reference_path("domain..name")
        # => ["domain", "name"]
    """
    if not path:
        return []
    text = path.strip()
    if not text:
        return []

    text = text.rstrip(".,;:!?)]}")

    if '"' not in text:
        return [p.strip() for p in text.split(".") if p.strip()]

    # Quote-aware: stop at whitespace outside quotes (consistent with normalize).
    # IMPORTANT: preserve quote characters so the split pass can use them.
    buffer: List[str] = []
    in_quotes = False
    for ch in text:
        if ch == '"':
            in_quotes = not in_quotes
            buffer.append(ch)
        elif ch.isspace() and not in_quotes:
            break
        else:
            buffer.append(ch)

    cleaned = "".join(buffer)
    if not cleaned:
        return []

    # Split on '.' only outside quotes; strip quotes from each segment (unquote).
    segments: List[str] = []
    seg_buf: List[str] = []
    in_quotes = False
    for ch in cleaned:
        if ch == '"':
            in_quotes = not in_quotes
            # Do NOT append — this effectively unquotes all segments.
        elif ch == "." and not in_quotes:
            seg = "".join(seg_buf).strip()
            if seg:
                segments.append(seg)
            seg_buf = []
        else:
            seg_buf.append(ch)
    if seg_buf:
        seg = "".join(seg_buf).strip()
        if seg:
            segments.append(seg)

    return segments


def quote_path_segment(segment: str) -> str:
    """Quote a path segment if it contains characters that are special in a
    reference path.

    An unquoted segment in ``REFERENCE_PATH_REGEX`` matches ``[^@\\s".]+``,
    so any segment containing **whitespace, ``@``, ``"`` or ``.``** must be
    wrapped in double-quotes.

    * Strips any existing surrounding double-quotes first.
    * Removes internal double-quotes (the ``"[^"]*"`` format cannot represent
      them).
    * Returns the segment unchanged when quoting is unnecessary.
    """
    if not segment:
        return ""
    cleaned = segment.strip().strip('"').strip()
    if not cleaned:
        return ""
    # Remove internal double-quotes (the path format cannot represent them)
    cleaned = cleaned.replace('"', "")
    # Characters that require the segment to be quoted (per REFERENCE_PATH_REGEX)
    _NEEDS_QUOTE_CHARS = set(" \t\n\r.@")
    if _NEEDS_QUOTE_CHARS.intersection(cleaned):
        return f'"{cleaned}"'
    return cleaned
