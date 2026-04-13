# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Split long text into Slack-safe chunks."""

from __future__ import annotations

SLACK_TEXT_LIMIT = 3000


def chunk_text(text: str, max_length: int = SLACK_TEXT_LIMIT) -> list[str]:
    """Split *text* into chunks of at most *max_length* characters.

    Splitting strategy (in order of preference):
    1. Paragraph boundaries (``\\n\\n``)
    2. Line boundaries (``\\n``)
    3. Hard truncation at *max_length*
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_length:
            current = candidate
            continue

        # Flush what we have so far
        if current:
            chunks.append(current)
            current = ""

        # If the paragraph itself fits, use it directly
        if len(para) <= max_length:
            current = para
            continue

        # Paragraph too long — split by lines
        for line in para.split("\n"):
            line_candidate = f"{current}\n{line}" if current else line
            if len(line_candidate) <= max_length:
                current = line_candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            # Single line too long — hard truncate
            if len(line) <= max_length:
                current = line
            else:
                while line:
                    chunks.append(line[:max_length])
                    line = line[max_length:]

    if current:
        chunks.append(current)

    return chunks
