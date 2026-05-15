# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Selection state machine and fragment-splicing utilities for the output pane.

The full-screen :class:`~datus.cli.tui.app.DatusApp` captures every mouse
event for its custom sticky-bottom scroll model, which means terminal-native
shift+drag text selection is unavailable inside the scrollback area. This
module brings selection back in software:

* :class:`TranscriptSelection` tracks an anchor and head ``(line,
  char_index)`` pair plus a dragging flag. The app updates it from
  ``MOUSE_DOWN`` / ``MOUSE_MOVE`` / ``MOUSE_UP`` events.
* :class:`SelectionAutoscroll` records the direction the user is dragging
  past a viewport edge so the app can periodically tick the scroll offset
  and grow the selection without the user releasing the button.
* :func:`split_line_for_selection` rewrites the per-row prompt_toolkit
  fragment list so the slice between two char indices carries an extra
  selection style class; the renderer then highlights it like a native
  terminal selection.

Why char indices, not visual columns
------------------------------------
prompt_toolkit's :class:`MouseEvent.position` reports ``x`` as the
**character index** within the line (the per-character counter
``rowcol_to_yx`` populates in :func:`Window._copy_body`), *not* the visual
column. For ASCII content those happen to be equal, but for CJK /
fullwidth glyphs they diverge — a click at visual column 4 (just past
``你好``) arrives as ``x=2`` (the third character). Earlier iterations of
this module treated ``x`` as a visual column, which silently shifted the
highlight left by one visual cell per CJK glyph between the line start
and the click. Tracking char indices end-to-end means we never need to
re-derive visual columns: the renderer naturally pairs the highlight
fragment with the same screen cells the user clicked on.

The module has no prompt_toolkit ``Application`` dependency — it is pure
data manipulation so it can be exercised by unit tests without mounting a
TUI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

_StyledToken = Tuple[str, str]


@dataclass(frozen=True)
class SelectionPoint:
    """A single endpoint of the selection.

    ``line`` is the index into the buffer's logical line stream as exposed
    by :class:`~datus.cli.tui.output_buffer._BufferSnapshot.get_line`.
    ``column`` is the **character index** within the rendered line (the
    same coordinate :class:`prompt_toolkit.mouse_events.MouseEvent`
    delivers in ``position.x``: one unit per character regardless of
    visual cell width, see :func:`prompt_toolkit.layout.containers.Window._copy_body`
    where ``col`` is incremented unconditionally per char).
    """

    line: int
    column: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.line, self.column)

    def __lt__(self, other: "SelectionPoint") -> bool:
        return self.as_tuple() < other.as_tuple()

    def __le__(self, other: "SelectionPoint") -> bool:
        return self.as_tuple() <= other.as_tuple()


@dataclass
class TranscriptSelection:
    """Mutable selection model owned by :class:`DatusApp`.

    The selection is described by an anchor (the point where the user
    pressed the mouse down) and a head (the point the cursor is at right
    now). The two are independent of which one is visually first — call
    :meth:`range` to recover a forward ``(start, end)`` pair for slicing.
    """

    anchor: Optional[SelectionPoint] = None
    head: Optional[SelectionPoint] = None
    dragging: bool = False
    # Bump on every state change so render-side caches keyed on selection
    # can detect "selection changed" without diffing point fields. Bumped
    # by :meth:`begin`, :meth:`update_head`, :meth:`finish`, :meth:`clear`.
    version: int = 0

    def begin(self, point: SelectionPoint) -> None:
        """Start a fresh selection at ``point``. Sets dragging=True."""
        self.anchor = point
        self.head = point
        self.dragging = True
        self.version += 1

    def update_head(self, point: SelectionPoint) -> None:
        """Move the head endpoint. No-op when no anchor exists."""
        if self.anchor is None:
            return
        if self.head == point:
            return
        self.head = point
        self.version += 1

    def finish(self) -> None:
        """Mark drag as finished but keep the selection visible."""
        if self.dragging:
            self.dragging = False
            self.version += 1

    def clear(self) -> None:
        """Drop the selection entirely (anchor + head). Idempotent."""
        if self.anchor is None and self.head is None and not self.dragging:
            return
        self.anchor = None
        self.head = None
        self.dragging = False
        self.version += 1

    def is_empty(self) -> bool:
        """``True`` when no selection is active or anchor == head."""
        if self.anchor is None or self.head is None:
            return True
        return self.anchor == self.head

    def range(self) -> Optional[Tuple[SelectionPoint, SelectionPoint]]:
        """Forward-ordered ``(start, end)`` or ``None`` when empty."""
        if self.is_empty():
            return None
        a, h = self.anchor, self.head
        assert a is not None and h is not None  # guarded by is_empty
        if a <= h:
            return (a, h)
        return (h, a)

    def contains_line(self, line_idx: int) -> bool:
        """Whether ``line_idx`` intersects the selection (inclusive)."""
        rng = self.range()
        if rng is None:
            return False
        start, end = rng
        return start.line <= line_idx <= end.line

    def columns_for_line(self, line_idx: int) -> Optional[Tuple[int, int]]:
        """Selection's char-index bounds for ``line_idx``.

        Returns ``(start_col, end_col)`` in character indices. ``end_col``
        may be :data:`COLUMN_TO_LINE_END` (a sentinel) for non-final
        selected rows, meaning "extend to the end of the rendered line";
        the caller clamps to the actual content length when splicing
        fragments.

        ``None`` when the line is outside the selection range.
        """
        rng = self.range()
        if rng is None:
            return None
        start, end = rng
        if not (start.line <= line_idx <= end.line):
            return None
        if start.line == end.line:
            return (start.column, end.column)
        if line_idx == start.line:
            return (start.column, COLUMN_TO_LINE_END)
        if line_idx == end.line:
            return (0, end.column)
        # Fully-selected middle row.
        return (0, COLUMN_TO_LINE_END)


# Sentinel: caller should clamp to the actual content width of the row.
# Picked far past any plausible terminal column count so naive ``min`` math
# in tests still does the right thing without a special-case.
COLUMN_TO_LINE_END: int = 10**9


@dataclass
class SelectionAutoscroll:
    """Direction-state for "drag past the edge keeps growing the selection".

    The app fires :meth:`arm` from its mouse handler whenever the user is
    dragging and their pointer is at or past the viewport edge. A long-
    lived background task polls :meth:`due` every tick interval; when it
    returns ``True`` the app advances the scroll offset by one row and
    updates the selection head to follow.
    """

    direction: int = 0  # -1 up, 0 idle, +1 down
    next_tick_monotonic: float = 0.0
    # 30 ms keeps motion smooth without saturating the renderer.
    interval_seconds: float = 0.03

    def arm(self, direction: int) -> None:
        """Set drag direction; reset tick clock so the first step is fast."""
        if direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1/0/+1, got {direction}")
        if direction == 0:
            self.direction = 0
            return
        if self.direction != direction:
            # Direction flip: fire immediately so the UX feels responsive.
            self.next_tick_monotonic = 0.0
        self.direction = direction

    def disarm(self) -> None:
        self.direction = 0

    def is_active(self) -> bool:
        return self.direction != 0

    def due(self, now: Optional[float] = None) -> bool:
        """Whether the next autoscroll step should fire by ``now``."""
        if self.direction == 0:
            return False
        ts = time.monotonic() if now is None else now
        if ts >= self.next_tick_monotonic:
            self.next_tick_monotonic = ts + self.interval_seconds
            return True
        return False


@dataclass(frozen=True)
class _Slice:
    """Helper struct returned by :func:`_walk_to_char`."""

    fragment_idx: int  # Index into the original fragment list.
    char_offset: int  # Number of characters of fragment.text already consumed.


def _walk_to_char(fragments: List[_StyledToken], target_char: int) -> _Slice:
    """Locate ``(fragment_idx, char_offset)`` for the ``target_char``-th char.

    Walks per-character (NOT per visual cell), so fullwidth glyphs count
    as a single character — matching :class:`prompt_toolkit.layout.containers.Window`'s
    ``rowcol_to_yx`` accounting where ``col`` increments by one per char
    regardless of cell width.

    A target past the line's last character returns a slice pointing one
    past the final fragment / final character.
    """
    consumed = 0
    for f_idx, fragment in enumerate(fragments):
        text = fragment[1] if len(fragment) > 1 else ""
        if not text:
            continue
        text_len = len(text)
        if target_char < consumed + text_len:
            return _Slice(fragment_idx=f_idx, char_offset=target_char - consumed)
        consumed += text_len
    return _Slice(fragment_idx=len(fragments), char_offset=0)


def _line_char_count(fragments: List[_StyledToken]) -> int:
    return sum(len(f[1]) for f in fragments if len(f) > 1 and f[1])


def split_line_for_selection(
    fragments: List[_StyledToken],
    start_col: int,
    end_col: int,
    selection_style: str = "class:selection",
) -> List[_StyledToken]:
    """Return a new fragment list with chars ``[start_col, end_col)`` highlighted.

    ``start_col`` and ``end_col`` are **character indices** into the
    concatenated text of ``fragments`` — the same coordinate system that
    :class:`prompt_toolkit.mouse_events.MouseEvent.position.x` reports.
    Fragments outside the selection are returned untouched; fragments that
    straddle a boundary are split into prefix / inside / suffix pieces,
    and the inside piece has ``selection_style`` appended to its style
    string so prompt_toolkit's style merger paints a "selected" background
    over the existing styling.

    ``end_col`` may be :data:`COLUMN_TO_LINE_END` for multi-row selections
    where the row should be highlighted to the end of its content; the
    function clamps to the actual character count.

    The function never mutates ``fragments``.
    """
    if start_col >= end_col or not fragments:
        return list(fragments)

    line_len = _line_char_count(fragments)
    end_col = min(end_col, line_len)
    if start_col >= end_col:
        return list(fragments)

    start = _walk_to_char(fragments, start_col)
    end = _walk_to_char(fragments, end_col)

    out: List[_StyledToken] = []
    for f_idx, fragment in enumerate(fragments):
        style = fragment[0] if fragment else ""
        text = fragment[1] if len(fragment) > 1 else ""
        # Carry over any 3rd element (mouse handler) verbatim on the
        # un-split parts so click handling still works.
        trailing = fragment[2:]

        if not text:
            out.append(fragment)
            continue

        if f_idx < start.fragment_idx or f_idx > end.fragment_idx:
            out.append(fragment)
            continue

        prefix_chars = start.char_offset if f_idx == start.fragment_idx else 0
        if f_idx == end.fragment_idx:
            inside_chars = end.char_offset - prefix_chars
        else:
            inside_chars = len(text) - prefix_chars

        if prefix_chars > 0:
            out.append(_make_fragment(style, text[:prefix_chars], trailing))
        if inside_chars > 0:
            inside_text = text[prefix_chars : prefix_chars + inside_chars]
            merged_style = _merge_style(style, selection_style)
            out.append(_make_fragment(merged_style, inside_text, trailing))
        suffix_text = text[prefix_chars + inside_chars :]
        if suffix_text:
            out.append(_make_fragment(style, suffix_text, trailing))
    return out


def line_char_count(fragments: List[_StyledToken]) -> int:
    """Total character count of a fragment list (excludes embedded ``\\n``)."""
    return _line_char_count(fragments)


def extract_plain_text_between(
    fragments: List[_StyledToken],
    start_col: int,
    end_col: int,
) -> str:
    """Return plain text between two character indices of a fragment list."""
    if start_col >= end_col or not fragments:
        return ""
    line_len = _line_char_count(fragments)
    end_col = min(end_col, line_len)
    if start_col >= end_col:
        return ""
    start = _walk_to_char(fragments, start_col)
    end = _walk_to_char(fragments, end_col)
    out: List[str] = []
    for f_idx, fragment in enumerate(fragments):
        text = fragment[1] if len(fragment) > 1 else ""
        if not text:
            continue
        if f_idx < start.fragment_idx or f_idx > end.fragment_idx:
            continue
        prefix_chars = start.char_offset if f_idx == start.fragment_idx else 0
        if f_idx == end.fragment_idx:
            inside_chars = end.char_offset - prefix_chars
        else:
            inside_chars = len(text) - prefix_chars
        if inside_chars > 0:
            out.append(text[prefix_chars : prefix_chars + inside_chars])
    return "".join(out)


def _fragment_text(fragments: List[_StyledToken]) -> str:
    return "".join((f[1] if len(f) > 1 else "") for f in fragments)


def _make_fragment(style: str, text: str, trailing: Tuple) -> _StyledToken:
    if trailing:
        return (style, text, *trailing)
    return (style, text)


def _merge_style(existing: str, addition: str) -> str:
    """Append ``addition`` to a prompt_toolkit style string.

    prompt_toolkit's style parser handles space-separated style strings,
    so concatenation is enough — ``addition`` wins on conflict because it
    is parsed last.
    """
    existing = (existing or "").strip()
    if not existing:
        return addition
    return f"{existing} {addition}"


__all__ = [
    "COLUMN_TO_LINE_END",
    "SelectionAutoscroll",
    "SelectionPoint",
    "TranscriptSelection",
    "extract_plain_text_between",
    "line_char_count",
    "split_line_for_selection",
]
