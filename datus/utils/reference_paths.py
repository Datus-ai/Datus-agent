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

    segments = [segment.strip() for segment in cleaned.split(".")]
    if not segments:
        return ""

    last = segments[-1]
    if last.startswith('"') and last.endswith('"') and len(last) >= 2:
        last = last[1:-1]
    segments[-1] = last
    return ".".join(segments)
