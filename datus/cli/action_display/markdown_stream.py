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
    the last blank line is eligible to flush — this is the "markdown
    segment ended" trigger.
2.  Fence (``\\`\\`\\`\\`\\`\\``) guard: when the number of triple-backticks in
    the whole text is odd, an unclosed fenced code block is in flight and
    *nothing* is flushed — we must never hand a half-open code block to
    Rich's markdown renderer or its styling would leak into subsequent
    content.
3.  Overflow guard: when the tail (the portion still pending) grows past
    :data:`MAX_TAIL_BYTES` bytes **or** :data:`MAX_TAIL_LINES` newlines
    (i.e. exceeds the pinned-region budget), we force a commit at the
    last ``"\\n"`` so everything above the live region's bottom line lands
    in the scrollback instead of being silently trimmed.

This class is deliberately synchronous and pure-Python (no Rich / no
prompt_toolkit); the streaming context runs it on its daemon refresh thread
so external locking is the caller's responsibility.
"""

from __future__ import annotations

from typing import List, Tuple

# Tail byte budget before triggering the oversize commit path. 4 KiB keeps
# per-token Rich re-parsing latency well below one frame at 4 Hz.
MAX_TAIL_BYTES = 4096

# Line-count budget for the tail before the overflow guard fires. Matches
# the pinned-region row budget (terminal_rows - reserved rows for status
# bar + input). When the tail crosses this, everything up to the last
# newline is committed to the scrollback so the live region stays in its
# displayable window.
MAX_TAIL_LINES = 20


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
        # Latched once any stable segment has been handed back to the caller
        # for the scrollback. Downstream code (e.g. the Ctrl+O verbose
        # snapshot) uses this to avoid re-painting the whole accumulated
        # body and duplicating the already-committed prefix.
        self._spilled: bool = False

    def append(self, delta: str) -> List[str]:
        """Append ``delta`` and return any newly stable segments.

        Two independent triggers can harvest a segment on each call:

        * **Markdown segment end** — whenever ``"\\n\\n"`` appears in the
          tail, everything up to the last such boundary is handed back
          (via :meth:`_split`). The returned segment is ``"\\n\\n"``-
          suffixed so ``"".join(segments) + tail`` reproduces the input.
        * **Pinned-region overflow** — if the residual tail is still
          taller than :data:`MAX_TAIL_LINES` newlines or larger than
          :data:`MAX_TAIL_BYTES` bytes (and fences are balanced, and
          we're not inside an unterminated markdown table), we cut
          at the last ``"\\n"`` so the pinned region can actually render
          what remains. The suffix kept in the tail is the unfinished
          current line.
        """
        if not delta:
            return []
        self._tail += delta
        stable, new_tail = self._split(self._tail)
        # Overflow guard: when the residual tail would overflow the
        # pinned region (either by byte budget for Rich re-parse cost or
        # by line budget for actual screen height), commit up to the
        # last newline so the live area stays within its displayable
        # window. Fences must be balanced — we never split a half-open
        # code block away from its opener. Open markdown tables get the
        # same protection: Rich's table renderer needs the header +
        # separator lines to lay out any row, so a header-less fragment
        # would paint the scrollback as a single mashed-pipe line.
        over_budget = len(new_tail) > MAX_TAIL_BYTES or new_tail.count("\n") >= MAX_TAIL_LINES
        if over_budget and new_tail.count("```") % 2 == 0 and not self._tail_is_in_open_table(new_tail):
            nl = new_tail.rfind("\n")
            if nl > 0:
                forced = new_tail[: nl + 1]
                if forced.count("```") % 2 == 0 and forced.strip():
                    stable.append(forced)
                    new_tail = new_tail[nl + 1 :]
        self._tail = new_tail
        if stable:
            self._spilled = True
        return stable

    def append_raw(self, delta: str) -> None:
        """Append ``delta`` without any commit.

        Kept for callers that genuinely want a plain accumulator (no
        mid-stream scrollback emission). The main CLI streaming path
        uses :meth:`append` instead so long bodies stream upward into
        the scrollback as soon as a markdown segment closes or the
        pinned region overflows.
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
        self._spilled = False
        return pending

    def clear(self) -> None:
        self._tail = ""
        self._spilled = False

    def get_tail(self) -> str:
        return self._tail

    def has_tail(self) -> bool:
        return bool(self._tail)

    def has_spilled(self) -> bool:
        """Whether any prefix paragraph has been handed off to the scrollback.

        Consumed by the Ctrl+O verbose snapshot path so it prints only the
        unspilled tail (instead of the full accumulated text) and avoids
        painting the already-committed prefix a second time.
        """
        return self._spilled

    @staticmethod
    def _tail_is_in_open_table(text: str) -> bool:
        """Return True when the tail looks like an unterminated markdown table.

        A tail is "in an open table" when it is not ``\\n\\n``-terminated
        and carries at least two pipe-bearing lines whose ``|`` count is
        >= 2 — i.e. a header + separator (or more rows) is in flight and
        the closing blank line has not arrived yet. The pipe-per-line
        floor of 2 keeps prose references like an inline ``cmd | grep``
        from being mistaken for a table row. Gives open tables the same
        hold-until-closure protection that fenced code blocks already
        get, so the scrollback never sees a header-less table fragment
        (which Rich would collapse into a single mashed-pipe line).
        """
        if not text or "|" not in text:
            return False
        if text.endswith("\n\n"):
            return False
        pipe_lines = [line for line in text.splitlines() if line.count("|") >= 2]
        return len(pipe_lines) >= 2

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
