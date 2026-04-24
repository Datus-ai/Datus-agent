# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Incremental markdown stream buffer for TUI pinned-region rendering.

The buffer accepts LLM-generated ``thinking_delta`` text chunks and decides
which prefix is "stable enough" to be flushed to the Rich scrollback area
(final monokai-highlighted markdown) while the rest stays in the pinned
live-render region as ``tail`` — the area users see being rewritten as new
tokens arrive.

Stable-boundary heuristic:

1.  Hard boundary is ``"\\n\\n"`` (CommonMark blank line). Everything up to
    the last blank line is eligible to flush.
2.  Fence (``\\`\\`\\`\\`\\`\\``) guard: when the number of triple-backticks in
    the whole text is odd, an unclosed fenced code block is in flight and
    *nothing* is flushed — we must never hand a half-open code block to
    Rich's markdown renderer or its styling would leak into subsequent
    content.
3.  Oversize guard: when the tail (the portion still pending) grows past
    :data:`MAX_TAIL_BYTES`, we force a flush at the last ``"\\n"`` so the
    Rich markdown renderer is not asked to re-parse a multi-KB string on
    every token.

This class is deliberately synchronous and pure-Python (no Rich / no
prompt_toolkit); the streaming context runs it on its daemon refresh thread
so external locking is the caller's responsibility.
"""

from __future__ import annotations

from typing import List, Tuple

# Tail byte budget before triggering the oversize commit path. 4 KiB keeps
# per-token Rich re-parsing latency well below one frame at 4 Hz.
MAX_TAIL_BYTES = 4096


class MarkdownStreamBuffer:
    """Accumulate streaming markdown deltas and emit stable segments.

    The buffer holds one growing string (``tail``). Each :meth:`append` call
    returns the list of segments that have crossed the stability boundary
    during that call, in order. Callers print those segments to the
    permanent Rich scrollback area and render the remaining ``tail`` into
    the pinned live region until the next delta arrives or the stream
    terminates (in which case :meth:`flush` drains it).
    """

    def __init__(self) -> None:
        self._tail: str = ""

    def append(self, delta: str) -> List[str]:
        """Append ``delta`` and return any newly stable segments.

        Stable segments are always suffixed with the ``"\\n\\n"`` that
        separated them from the next block, so printing ``"".join(segments)``
        reproduces the original text exactly.
        """
        if not delta:
            return []
        self._tail += delta
        stable, new_tail = self._split(self._tail)
        # Oversize guard: if we still have a huge tail and fences are
        # balanced, commit up to the last newline so Rich isn't asked to
        # re-render a multi-KB chunk on every subsequent delta.
        if len(new_tail) > MAX_TAIL_BYTES and new_tail.count("```") % 2 == 0:
            nl = new_tail.rfind("\n")
            if nl > 0:
                stable.append(new_tail[: nl + 1])
                new_tail = new_tail[nl + 1 :]
        self._tail = new_tail
        return stable

    def append_raw(self, delta: str) -> None:
        """Append ``delta`` without any stability detection.

        Used by the accumulator-only flow: the pinned live region shows the
        growing body token-by-token and the whole text is flushed to the
        scrollback **only** when :meth:`flush` is called at message
        boundaries. Nothing is ever pushed mid-stream, so there is no risk
        of doubling up on a final one-shot render.
        """
        if not delta:
            return
        self._tail += delta

    def flush(self) -> str:
        """Return whatever is left in ``tail`` and clear the buffer.

        Used on stream termination (final response arrived) and on user
        interruption (Ctrl+C / ESC) so the scrollback preserves the exact
        text the user saw.
        """
        pending = self._tail
        self._tail = ""
        return pending

    def clear(self) -> None:
        self._tail = ""

    def get_tail(self) -> str:
        return self._tail

    def has_tail(self) -> bool:
        return bool(self._tail)

    @staticmethod
    def _split(text: str) -> Tuple[List[str], str]:
        """Split ``text`` into ``(stable_segments, new_tail)``.

        Returns an empty ``stable_segments`` list when the text still has
        an unclosed fence or no blank-line boundary has appeared yet.
        """
        if not text:
            return [], ""
        # Unclosed fenced code block: hold everything.
        if text.count("```") % 2 == 1:
            return [], text
        # Commit up to the last blank-line boundary.
        idx = text.rfind("\n\n")
        if idx < 0:
            return [], text
        cut = idx + 2
        stable = text[:cut]
        tail = text[cut:]
        if not stable.strip():
            # Leading whitespace only — nothing meaningful to flush.
            return [], text
        return [stable], tail
