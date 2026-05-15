# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Find-in-scrollback state machine and scanner for the TUI output pane.

Bound to ``Ctrl+F`` in :class:`~datus.cli.tui.app.DatusApp`. The UI is a
1-row search bar tucked above the input prompt; this module owns the
data side — a case-insensitive substring scanner over the flattened
buffer rows plus a small state container the renderer / app consult.

Coordinates match :mod:`datus.cli.tui.selection`: ``line`` is the
``_BufferSnapshot.get_line`` index, ``start``/``end`` are character
indices (not visual columns), and CJK glyphs count as one character so
:func:`datus.cli.tui.selection.split_line_for_selection` can re-use the
same offsets to paint the highlight overlay.

The scanner is intentionally simple — substring only, no regex — both
to keep the UX predictable and to avoid the "bad regex hangs the TUI"
class of bugs. Each match within a line allows overlap (search for
``aa`` in ``aaaa`` yields three matches) by advancing one character
between hits, mirroring how editors like VSCode handle ``Find``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# A single styled fragment ``(style, text)`` — the same shape
# :class:`prompt_toolkit.layout.controls.UIContent.get_line` returns.
_StyledToken = Tuple[str, str]


@dataclass(frozen=True)
class SearchMatch:
    """One contiguous run of characters matching the current query.

    ``start`` / ``end`` are half-open character indices, identical to
    the coordinate system used by ``MouseEvent.position.x`` and by
    :class:`datus.cli.tui.selection.SelectionPoint.column`.
    """

    line: int
    start: int
    end: int

    def length(self) -> int:
        return self.end - self.start


@dataclass
class SearchState:
    """Mutable container the renderer + app share for the active search.

    ``current_idx`` indexes into :attr:`matches`; ``-1`` means no
    current match (either because the query is empty or because no
    match was found). ``version`` increments on every state change so
    :class:`BufferedOutputControl`'s ``UIContent`` cache key can
    detect "search state changed" without diffing every field.
    """

    query: str = ""
    matches: List[SearchMatch] = field(default_factory=list)
    current_idx: int = -1
    version: int = 0

    # ── mutators (each bumps ``version``) ────────────────────────

    def clear(self) -> None:
        """Drop query + matches + current index. Idempotent."""
        if not self.query and not self.matches and self.current_idx == -1:
            return
        self.query = ""
        self.matches = []
        self.current_idx = -1
        self.version += 1

    def update(self, query: str, matches: List[SearchMatch], *, current_idx: int) -> None:
        """Replace the whole state in one bump; called after a re-scan."""
        self.query = query
        self.matches = list(matches)
        self.current_idx = current_idx if matches else -1
        self.version += 1

    def set_current(self, idx: int) -> None:
        """Move the focused match. No-op when ``matches`` is empty."""
        if not self.matches:
            return
        idx = idx % len(self.matches)
        if idx == self.current_idx:
            return
        self.current_idx = idx
        self.version += 1

    # ── queries ─────────────────────────────────────────────────

    def is_active(self) -> bool:
        """``True`` when the state holds any results worth painting."""
        return bool(self.matches)

    def current(self) -> Optional[SearchMatch]:
        if 0 <= self.current_idx < len(self.matches):
            return self.matches[self.current_idx]
        return None

    def matches_on_line(self, line_idx: int) -> List[Tuple[int, int, bool]]:
        """Return ``(start, end, is_current)`` tuples for ``line_idx``.

        Sorted by ``start`` so the renderer can apply overlay splits in
        left-to-right order. The ``is_current`` flag lets the renderer
        paint the active match with a distinct style.
        """
        if not self.matches:
            return []
        current = self.current()
        out: List[Tuple[int, int, bool]] = []
        for m in self.matches:
            if m.line != line_idx:
                continue
            out.append((m.start, m.end, current is m))
        out.sort(key=lambda triple: triple[0])
        return out


def find_matches(
    get_line: Callable[[int], List[_StyledToken]],
    line_count: int,
    query: str,
) -> List[SearchMatch]:
    """Scan ``line_count`` rows from ``get_line`` for case-insensitive ``query``.

    Returns the matches in row-major order — by ``line`` ascending, then
    by ``start`` ascending within a line. An empty ``query`` always
    yields ``[]`` (the caller treats that as "no search active").

    Overlapping matches on the same line are emitted independently
    (advance by one character, not by ``len(query)``) so e.g. ``aa`` in
    ``aaaa`` produces three matches. This mirrors editor expectations
    and is essentially free for typical buffer sizes.

    Style information is stripped on the way in — only the concatenated
    plain text of each line is scanned, so CJK glyphs and styled spans
    contribute the same way they do for selection.
    """
    if not query:
        return []
    needle = query.lower()
    needle_len = len(needle)
    out: List[SearchMatch] = []
    for line_idx in range(line_count):
        fragments = get_line(line_idx)
        if not fragments:
            continue
        # Concatenate plain text. Skip any fragment shape that doesn't
        # carry a string in slot 1 (defensive — the buffer's snapshots
        # always emit 2-tuples, but ``BufferedOutputControl`` wrappers
        # may inject 3-tuples with a mouse handler that we still want
        # to scan past).
        haystack = "".join(f[1] for f in fragments if len(f) > 1 and isinstance(f[1], str))
        if not haystack:
            continue
        lowered = haystack.lower()
        start = 0
        while True:
            pos = lowered.find(needle, start)
            if pos == -1:
                break
            out.append(SearchMatch(line=line_idx, start=pos, end=pos + needle_len))
            # Step by one to allow overlapping matches.
            start = pos + 1
    return out


__all__ = ["SearchMatch", "SearchState", "find_matches"]
