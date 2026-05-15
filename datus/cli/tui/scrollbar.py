# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Click / drag-aware scrollbar for the output pane.

prompt_toolkit ships a :class:`prompt_toolkit.layout.margins.ScrollbarMargin`
but its fragments are passive — the Window's mouse-handler range is set up
to ignore margin columns entirely (see ``Window._write_to_screen``), so
clicks on the margin never reach the control. To get a draggable
scrollbar we therefore render it as its own :class:`Window` placed inside
a :class:`VSplit` next to the output pane. The Window's content is a
:class:`FormattedTextControl` whose fragments are three-tuples ``(style,
char, mouse_handler)`` — prompt_toolkit dispatches the per-cell mouse
handler natively when fragments carry a third element, so we get click
and drag for free.

Geometry
--------
* Track height ``H`` = viewport rows (= height of the parent VSplit row).
* Thumb height ``T = max(1, round(H * H / total))`` when ``total > H``,
  else the thumb fills the track (no overflow → no need to scroll).
* Thumb top ``Y = round(top * (H - T) / (total - H))`` when ``total > H``.
* Click on track row ``r`` sets scroll offset to
  ``round(r * (total - H) / max(H - 1, 1))``, clamped to ``[0, total-H]``.

The math is symmetric: dragging the thumb down by one row advances the
scroll by ``(total - H) / (H - 1)`` rows, so the bottom of the track
always represents the bottom of the content.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

_StyledToken = Tuple


# Glyphs. Unicode block + light vertical line — high contrast on both
# light and dark terminal palettes. No emoji (per CLAUDE.md CLI style
# rules).
_THUMB_CHAR = "\u2588"  # █
_TRACK_CHAR = "\u2502"  # │


class ScrollbarController:
    """Adapter between the scrollbar widget and :class:`DatusApp`.

    The widget never touches the app directly so it stays unit-testable.
    Methods accept an *intended* state ("scroll to this row") and the
    controller applies it through the app's existing sticky-bottom
    machinery — preserving the invariant that wheel events and scrollbar
    drags converge on the same internal model.
    """

    def __init__(
        self,
        *,
        viewport_rows_fn: Callable[[], int],
        total_rows_fn: Callable[[], int],
        get_scroll_fn: Callable[[], int],
        set_scroll_fn: Callable[[int], None],
        invalidate_fn: Callable[[], None],
    ) -> None:
        self._viewport_rows_fn = viewport_rows_fn
        self._total_rows_fn = total_rows_fn
        self._get_scroll_fn = get_scroll_fn
        self._set_scroll_fn = set_scroll_fn
        self._invalidate_fn = invalidate_fn
        # Drag flag is set on the first ``MOUSE_DOWN`` and cleared on
        # ``MOUSE_UP``. The output pane's mouse handler reads it to
        # suppress selection-copy when the user is dragging the
        # scrollbar (so releasing the mouse over the transcript does
        # not unexpectedly overwrite the clipboard with whatever was
        # selected before).
        self._dragging: bool = False

    @property
    def dragging(self) -> bool:
        return self._dragging

    def viewport_rows(self) -> int:
        return max(1, int(self._viewport_rows_fn()))

    def total_rows(self) -> int:
        return max(0, int(self._total_rows_fn()))

    def max_scroll(self) -> int:
        return max(0, self.total_rows() - self.viewport_rows())

    def is_overflow(self) -> bool:
        """Whether the content exceeds the viewport (i.e. scrollbar useful)."""
        return self.total_rows() > self.viewport_rows()

    # ── thumb geometry ────────────────────────────────────────────

    def thumb_geometry(self) -> Tuple[int, int]:
        """Return ``(thumb_top, thumb_height)`` for the current state.

        Both values are in rows within the track. When the content fits
        in the viewport the thumb fills the whole track (top=0,
        height=viewport_rows) so the track still renders as a
        visually-consistent column rather than leaving the gutter empty.
        """
        viewport = self.viewport_rows()
        total = self.total_rows()
        if total <= viewport or total <= 0:
            return (0, viewport)
        # Round-half-up via int(... + 0.5) to keep the thumb at least one
        # row tall for very large transcripts.
        thumb_height = max(1, round(viewport * viewport / total))
        thumb_height = min(thumb_height, viewport)

        max_scroll = total - viewport
        scroll = max(0, min(self._get_scroll_fn(), max_scroll))
        free = viewport - thumb_height
        if free <= 0 or max_scroll <= 0:
            thumb_top = 0
        else:
            thumb_top = round(scroll * free / max_scroll)
            thumb_top = max(0, min(thumb_top, free))
        return (thumb_top, thumb_height)

    # ── input handling ────────────────────────────────────────────

    def scroll_to_track_row(self, row: int) -> None:
        """Snap the viewport so its top aligns proportionally with ``row``.

        ``row`` is a track-relative row index (0..viewport_rows-1). The
        math mirrors the inverse of :meth:`thumb_geometry`'s thumb_top
        calculation so clicking on the same track row twice is stable.
        """
        viewport = self.viewport_rows()
        max_scroll = self.max_scroll()
        if max_scroll <= 0 or viewport <= 1:
            self._set_scroll_fn(0)
            self._invalidate_fn()
            return
        row = max(0, min(row, viewport - 1))
        # Map row ∈ [0, viewport-1] linearly to scroll ∈ [0, max_scroll].
        scroll = round(row * max_scroll / max(viewport - 1, 1))
        scroll = max(0, min(scroll, max_scroll))
        self._set_scroll_fn(scroll)
        self._invalidate_fn()

    def handle_event(self, event: MouseEvent) -> None:
        """Dispatch one prompt_toolkit MouseEvent to drag / click handlers.

        Returns nothing — callers should always treat the event as
        handled (i.e. return ``None`` from the wrapping fragment handler
        so prompt_toolkit invalidates the layout). ``mouse_event.position.y``
        is interpreted as the row within the scrollbar track.
        """
        et = event.event_type
        if et == MouseEventType.MOUSE_DOWN and event.button == MouseButton.LEFT:
            self._dragging = True
            self.scroll_to_track_row(event.position.y)
        elif et == MouseEventType.MOUSE_MOVE and event.button == MouseButton.LEFT:
            if self._dragging:
                self.scroll_to_track_row(event.position.y)
        elif et == MouseEventType.MOUSE_UP:
            self._dragging = False
        elif et == MouseEventType.SCROLL_UP:
            # Forwarding wheel events on the scrollbar to the same
            # offset model keeps the gutter clickable+scrollable —
            # without this the wheel would fall through to the parent
            # which has no scrollbar handler of its own.
            current = max(0, self._get_scroll_fn() - 1)
            self._set_scroll_fn(current)
            self._invalidate_fn()
        elif et == MouseEventType.SCROLL_DOWN:
            current = min(self.max_scroll(), self._get_scroll_fn() + 1)
            self._set_scroll_fn(current)
            self._invalidate_fn()


def build_scrollbar_fragments(controller: ScrollbarController) -> List[_StyledToken]:
    """Render the scrollbar's per-row fragments for the current frame."""
    viewport = controller.viewport_rows()
    thumb_top, thumb_height = controller.thumb_geometry()

    def handler(mouse_event: MouseEvent):  # noqa: ANN202
        controller.handle_event(mouse_event)
        return None

    out: List[_StyledToken] = []
    for row in range(viewport):
        in_thumb = thumb_top <= row < thumb_top + thumb_height
        style = "class:scrollbar.thumb" if in_thumb else "class:scrollbar.track"
        char = _THUMB_CHAR if in_thumb else _TRACK_CHAR
        # 3-tuple: (style, text, mouse_handler). prompt_toolkit's
        # FormattedTextControl.mouse_handler looks for the 3rd element on
        # each fragment under the cursor cell.
        out.append((style, char, handler))
        if row < viewport - 1:
            out.append(("", "\n"))
    return out


def build_scrollbar_window(
    controller: ScrollbarController,
    *,
    visible_filter: Callable[[], bool] | None = None,
) -> ConditionalContainer:
    """Construct the scrollbar :class:`Window` for embedding in a VSplit.

    ``visible_filter`` defaults to "show whenever total > viewport". Pass
    a custom predicate (e.g. ``lambda: True``) for an always-visible
    track. The window's width is hard-pinned at 1 column so neighbouring
    weights don't cause the gutter to jitter when the viewport size
    changes.
    """
    is_visible: Callable[[], bool] = visible_filter or controller.is_overflow

    def get_fragments() -> List[_StyledToken]:
        return build_scrollbar_fragments(controller)

    window = Window(
        content=FormattedTextControl(text=get_fragments, focusable=False, show_cursor=False),
        width=Dimension.exact(1),
        wrap_lines=False,
        style="class:scrollbar",
        # Width is fixed and the content is sized to the row by virtue of
        # the parent VSplit — no scroll math at the Window level.
        always_hide_cursor=True,
    )
    return ConditionalContainer(content=window, filter=Condition(is_visible))


__all__ = [
    "ScrollbarController",
    "build_scrollbar_fragments",
    "build_scrollbar_window",
]
