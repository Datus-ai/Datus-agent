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
