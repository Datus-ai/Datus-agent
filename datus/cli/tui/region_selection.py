# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Independent mouse text-selection for auxiliary TUI panes.

The scrollback pane already implements software selection (see
:mod:`datus.cli.tui.selection` and ``DatusApp._output_mouse_handler``); the
other :class:`~prompt_toolkit.layout.controls.FormattedTextControl`-backed
panes — status bar, todo sidebar, queue preview — historically ignored
mouse drags, so their text could not be copied at all while the app owns
mouse capture. :class:`SelectableRegion` closes that gap: each region owns
its own :class:`TranscriptSelection` so a drag is always confined to the
region it started in, and releasing the button copies the highlighted text
to the system clipboard exactly like the scrollback pane does.

Coordinate contract
-------------------
prompt_toolkit's ``Window`` rebases mouse positions into UIContent space
before calling the control's handler: ``position.y`` is the **logical line
index** (wrap-aware for ``wrap_lines=True`` windows) and ``position.x`` is
the **character index** within that line — the same coordinates
:func:`split_line_for_selection` and :func:`extract_text_from_lines`
consume. Clicks below the content area are clamped by prompt_toolkit onto
the last painted row, so no scroll/offset accounting is needed here (these
panes never scroll).

Blank rows need one caveat: ``Window._copy_body`` only registers
``rowcol_to_yx`` entries for painted cells, so a fully-empty line would
make prompt_toolkit fall back to the ``Point(0, 0)`` sentinel and warp a
drag to the region top. :meth:`SelectableRegion.render_tokens` pads such
lines with a single space — visually identical, but every visible row
stays mouse-resolvable. The padding never reaches the clipboard because
extraction reads the raw (unpadded) lines.

The class has no prompt_toolkit ``Application`` dependency; interaction
with the rest of the app flows through the ``on_begin`` / ``on_copy`` /
``invalidate`` callbacks so it can be exercised by unit tests without
mounting a TUI.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from prompt_toolkit.formatted_text.utils import split_lines
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from datus.cli.tui.selection import (
    MultiClickTracker,
    SelectionPoint,
    TranscriptSelection,
    extract_text_from_lines,
    fragment_plain_text,
    split_line_for_selection,
    word_bounds_at,
)

_StyledToken = Tuple[str, str]


class SelectableRegion:
    """Mouse-selection + clipboard-copy for one ``tokens_fn``-backed pane.

    ``tokens_fn`` returns the same flat fragment stream the region's
    ``FormattedTextControl`` renders (``\\n``-separated). The owning app
    should point the control's ``text`` at :meth:`render_tokens` (so the
    highlight paints) and route the control's mouse events into
    :meth:`handle_mouse`.

    Callbacks:

    * ``on_begin`` — fired when a fresh drag starts; the app uses it to
      clear every *other* region's selection so exactly one highlight is
      visible at a time.
    * ``on_copy`` — receives the extracted plain text on drag release; the
      app wires it to the clipboard + hint plumbing.
    * ``invalidate`` — request a repaint after any selection state change.
    """

    def __init__(
        self,
        name: str,
        tokens_fn: Callable[[], List[_StyledToken]],
        *,
        on_begin: Optional[Callable[[], None]] = None,
        on_copy: Optional[Callable[[str], None]] = None,
        invalidate: Optional[Callable[[], None]] = None,
    ) -> None:
        self.name = name
        self.selection = TranscriptSelection()
        self._tokens_fn = tokens_fn
        self._on_begin = on_begin or (lambda: None)
        self._on_copy = on_copy or (lambda _text: None)
        self._invalidate = invalidate or (lambda: None)
        self._click_tracker = MultiClickTracker()

    def lines(self) -> List[List[_StyledToken]]:
        """Current region content split into per-line fragment lists."""
        try:
            tokens = list(self._tokens_fn() or [])
        except Exception:
            tokens = []
        return [list(line) for line in split_lines(tokens)]

    def render_tokens(self) -> List[_StyledToken]:
        """Fragment stream for the region's ``FormattedTextControl``.

        Identical to the raw ``tokens_fn`` output except that blank lines
        are padded with a single space (mouse resolvability — see module
        docstring) and lines intersecting the selection get the
        ``class:selection`` splice from :func:`split_line_for_selection`.
        """
        lines = self.lines()
        active = not self.selection.is_empty()
        out: List[_StyledToken] = []
        last = len(lines) - 1
        for idx, line in enumerate(lines):
            if not any(len(f) > 1 and f[1] for f in line):
                line = [("", " ")]
            if active:
                bounds = self.selection.columns_for_line(idx)
                if bounds is not None:
                    line = split_line_for_selection(line, bounds[0], bounds[1])
            out.extend(line)
            if idx != last:
                out.append(("", "\n"))
        return out

    def handle_mouse(self, event: MouseEvent):  # noqa: ANN201
        """Selection state machine: DOWN begins, MOVE extends, UP copies.

        Rapid same-spot clicks widen the selection instead: a double-click
        selects (and copies) the word under the pointer, a triple-click the
        whole line.

        Returns ``None`` when the event was consumed, ``NotImplemented``
        otherwise (scroll wheel, non-left buttons, moves without an active
        drag) so ``Window`` fallback behaviour is preserved.
        """
        et = event.event_type
        if et == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            point = self._point_from_event(event)
            if point is None:
                return NotImplemented
            clicks = self._click_tracker.register(point.line, point.column)
            self._on_begin()
            if clicks >= 2:
                self._select_span(point, whole_line=clicks >= 3)
                return None
            self.selection.begin(point)
            self._invalidate()
            return None
        if et == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT:
            if not self.selection.dragging:
                return NotImplemented
            point = self._point_from_event(event)
            if point is not None:
                self.selection.update_head(point)
            self._invalidate()
            return None
        if et == MouseEventType.MOUSE_UP:
            if not self.selection.dragging:
                return NotImplemented
            self.finish_drag()
            return None
        return NotImplemented

    def _select_span(self, point: SelectionPoint, *, whole_line: bool) -> None:
        """Select (and copy) the word or whole line at ``point``.

        Multi-click path: the selection is created already-finished (no
        drag in flight) so a subsequent MOUSE_MOVE with the button still
        held does not warp it, and :meth:`finish_drag` handles the copy +
        repaint exactly like a drag release would.
        """
        lines = self.lines()
        fragments = lines[point.line] if 0 <= point.line < len(lines) else []
        text = fragment_plain_text(fragments)
        if whole_line:
            bounds = (0, len(text)) if text else None
        else:
            bounds = word_bounds_at(text, point.column)
        if bounds is None or bounds[0] >= bounds[1]:
            self.selection.clear()
            self._invalidate()
            return
        self.selection.begin(SelectionPoint(line=point.line, column=bounds[0]))
        self.selection.update_head(SelectionPoint(line=point.line, column=bounds[1]))
        self.finish_drag()

    def finish_drag(self) -> None:
        """Finalise the drag; fire ``on_copy`` when the selection is non-empty.

        Also the entry point for *cross-region* releases: when the user
        drags out of this region and lets go over another pane, that pane's
        handler calls this so the drag never dangles.
        """
        self.selection.finish()
        if not self.selection.is_empty():
            text = self.extract_text()
            if text:
                self._on_copy(text)
        self._invalidate()

    def extract_text(self) -> str:
        """Plain text covered by the current selection (styles stripped)."""
        lines = self.lines()
        return extract_text_from_lines(lambda idx: lines[idx], len(lines), self.selection)

    def _point_from_event(self, event: MouseEvent) -> Optional[SelectionPoint]:
        """Translate a control-relative event into a region coordinate.

        Lines past the content end are clamped onto the last line so a
        drag below the region still highlights something visible (matches
        the output pane's behaviour).
        """
        line = int(event.position.y)
        column = int(event.position.x)
        if line < 0 or column < 0:
            return None
        total = len(self.lines())
        if total <= 0:
            return None
        if line >= total:
            line = total - 1
        return SelectionPoint(line=line, column=column)


__all__ = ["SelectableRegion"]
