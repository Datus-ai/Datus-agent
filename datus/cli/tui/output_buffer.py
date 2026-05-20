# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""In-memory output buffer that bridges Rich console → prompt_toolkit tokens.

In ``full_screen=True`` mode the prompt_toolkit Application owns the entire
terminal. There is no scrollback area above the layout that ``patch_stdout``
can inject into, so every byte Rich emits must instead flow into a Window
that lives *inside* our Layout.

:class:`TUIOutputBuffer` satisfies Rich's minimal IO contract (``write``,
``flush``, ``isatty``) so it can be passed as ``Console(file=buffer,
force_terminal=True, color_system="256")``. Each ``write`` call accumulates
bytes, splits them on ``\\n``, and parses every complete line through
:class:`prompt_toolkit.formatted_text.ANSI` so the styled content survives
the round trip into a :class:`FormattedTextControl`.

The buffer also concatenates the live-tail snapshot (kept by
:class:`LiveDisplayState` for streaming markdown / subagent rolling
windows) on top of the committed history. Together they form the single
token stream the scrollable output Window renders each frame.

Thread-safety
-------------
``write`` is called from arbitrary worker threads (the agent runs on a
ThreadPoolExecutor). All mutations are guarded by an internal lock so
concurrent prints don't interleave their characters. After each write the
configured ``on_change`` callback is fired; wire it to
:meth:`DatusApp.invalidate`, which itself dispatches via
``loop.call_soon_threadsafe`` so the main loop wakes safely.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple

from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseEvent

from datus.cli.tui.live_display_state import LiveDisplayLine
from datus.cli.tui.search import SearchState
from datus.cli.tui.selection import (
    TranscriptSelection,
    extract_plain_text_between,
    line_char_count,
    split_line_for_selection,
)

_StyledToken = Tuple[str, str]


class _BufferSnapshot:
    """Frozen view of the buffer captured at a single ``snapshot()`` call.

    Holds references to the committed lines (tuple of per-line fragment
    lists), the live-tail snapshot, and the parsed partial-line fragments.
    Provides ``get_line(idx)`` so :class:`BufferedOutputControl` can hand it
    straight to prompt_toolkit's :class:`UIContent` without materialising the
    full fragment stream. ``UIContent.get_line`` is invoked **only for visible
    rows**, so a multi-thousand-line scrollback no longer pays an O(N)
    ``tuple(fragments)`` hash on every key press.
    """

    __slots__ = ("committed", "live_lines", "partial_fragments", "total")

    def __init__(
        self,
        committed: Tuple[List[_StyledToken], ...],
        live_lines: Tuple[LiveDisplayLine, ...],
        partial_fragments: List[_StyledToken],
        total: int,
    ) -> None:
        self.committed = committed
        self.live_lines = live_lines
        self.partial_fragments = partial_fragments
        self.total = total

    def get_line(self, idx: int) -> List[_StyledToken]:
        committed_n = len(self.committed)
        if idx < committed_n:
            return self.committed[idx]
        idx -= committed_n
        live_n = len(self.live_lines)
        if idx < live_n:
            return self.live_lines[idx].segments
        return self.partial_fragments


class TUIOutputBuffer:
    """Captures Rich console output and surfaces it as prompt_toolkit tokens.

    The buffer is append-only. There is no maximum line cap — the user
    explicitly chose unlimited history retention (see the plan). A future
    iteration may add ``agent.tui.output_max_lines`` config.
    """

    def __init__(
        self,
        live_state_snapshot_fn: Optional[Callable[[], List[LiveDisplayLine]]] = None,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._committed: List[List[_StyledToken]] = []
        self._partial: str = ""
        self._live_state_snapshot_fn = live_state_snapshot_fn or (lambda: [])
        self._on_change = on_change or (lambda: None)
        # Line count cached from the most recent ``tokens()`` call so
        # ``line_count()`` returns a value consistent with the tokens the
        # caller is rendering — without it, a live-state mutation between
        # the two calls could make ``cursor.y`` exceed ``len(fragment_lines)``
        # and crash prompt_toolkit's render loop with an IndexError.
        self._last_line_count: int = 0
        # Render cache. ``FormattedTextControl`` keys ``processed_lines`` on
        # the *identity* of the fragment list, so as long as nothing visible
        # has changed since the last paint we return the same list object
        # and the control skips re-layout entirely. Without this every
        # scroll wheel tick triggers a full O(N) rebuild — the dominant
        # cost when the verbose-mode scrollback contains thousands of lines.
        # ``_committed_version`` is bumped on every committed-history
        # mutation; ``_cache_partial`` / ``_cache_live_lines`` track the
        # other two inputs to ``tokens()``.
        self._committed_version: int = 0
        self._cache_tokens: Optional[List[_StyledToken]] = None
        self._cache_committed_version: int = -1
        self._cache_partial: Optional[str] = None
        self._cache_live_lines: Optional[List[LiveDisplayLine]] = None
        # Separate cache for the lazy-row snapshot consumed by
        # :class:`BufferedOutputControl`. Reusing the same snapshot object
        # across paints lets the control's own ``UIContent`` cache short-
        # circuit re-layout when nothing visible has changed.
        self._cache_snapshot: Optional[_BufferSnapshot] = None
        self._cache_snapshot_version: int = -1
        self._cache_snapshot_partial: Optional[str] = None
        self._cache_snapshot_live: Optional[Tuple[LiveDisplayLine, ...]] = None

    # ── Rich file-like contract ───────────────────────────────────

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._partial += text
            new_lines: List[List[_StyledToken]] = []
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                new_lines.append(list(to_formatted_text(ANSI(line))))
            if new_lines:
                self._committed.extend(new_lines)
                self._committed_version += 1
        self._on_change()
        return len(text)

    def flush(self) -> None:
        # Rich calls flush after every print; we have nothing to flush —
        # the next paint will pick up whatever's in _committed / _partial.
        pass

    def isatty(self) -> bool:
        # Rich respects this via ``force_terminal=True``; returning False
        # ensures Rich doesn't try cursor-movement escapes that would
        # confuse our token consumer.
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        # Some callers (e.g. shutil.get_terminal_size) probe for a real
        # file descriptor. We don't own one, so raise the standard error
        # — callers all fall back to ``shutil`` or terminal-size defaults.
        raise OSError("TUIOutputBuffer has no underlying file descriptor")

    # ── prompt_toolkit token source ───────────────────────────────

    def tokens(self) -> List[_StyledToken]:
        """Build the full token stream rendered by the output Window.

        Order: committed history → live-tail snapshot → unflushed partial.
        Each line is followed by an explicit ``("", "\\n")`` separator so
        the consuming Window splits rows correctly. Trailing newline is
        dropped only when neither a live tail nor a partial line trails
        the committed history — keeps the visual cursor anchored at
        content.

        Also publishes ``_last_line_count`` so a subsequent
        :meth:`line_count` call returns a value consistent with the
        tokens this call produced.
        """
        # Snapshot live state outside the buffer lock — the LiveDisplayState
        # callback acquires its own lock, and holding both at once invites
        # deadlock with writers that flow buffer → live state.
        live_lines = list(self._live_state_snapshot_fn() or [])

        with self._lock:
            partial = self._partial
            committed_version = self._committed_version
            # Cache hit: every input that feeds the token stream is byte-for-
            # byte identical. Return the exact same list object so
            # ``FormattedTextControl`` short-circuits its per-frame layout.
            if (
                self._cache_tokens is not None
                and self._cache_committed_version == committed_version
                and self._cache_partial == partial
                and self._cache_live_lines == live_lines
            ):
                return self._cache_tokens
            committed = list(self._committed)

        out: List[_StyledToken] = []
        last_was_newline = False
        for line in committed:
            out.extend(line)
            out.append(("", "\n"))
            last_was_newline = True
        for live_line in live_lines:
            out.extend(live_line.segments)
            out.append(("", "\n"))
            last_was_newline = True
        if partial:
            out.extend(to_formatted_text(ANSI(partial)))
            last_was_newline = False

        # Drop a trailing pure-newline if we have no in-flight content —
        # avoids an empty row at the very bottom of the pane.
        if last_was_newline and not partial:
            out.pop()

        line_count = len(committed) + len(live_lines) + (1 if partial else 0)
        with self._lock:
            self._last_line_count = line_count
            self._cache_tokens = out
            self._cache_committed_version = committed_version
            self._cache_partial = partial
            self._cache_live_lines = live_lines

        return out

    def line_count(self) -> int:
        """Live row count: ``committed + live_tail + (partial?)``.

        Use this for general-purpose introspection (tests, sticky-bottom
        heuristics, page-size calculations). For cursor positioning fed
        to ``FormattedTextControl.get_cursor_position`` use
        :meth:`render_line_count` instead — that's the only place a
        race between ``tokens()`` and a separate ``line_count()`` snapshot
        can crash prompt_toolkit's render loop.
        """
        with self._lock:
            committed_n = len(self._committed)
            partial_n = 1 if self._partial else 0
        live_n = len(self._live_state_snapshot_fn() or [])
        return committed_n + live_n + partial_n

    def render_line_count(self) -> int:
        """Row count from the most recent ``tokens()`` call.

        prompt_toolkit's render order calls ``tokens()`` first (via
        ``FormattedTextControl.text``) and ``get_cursor_position``
        second, so reading this cached value guarantees the cursor
        index is ≤ ``len(fragment_lines)`` even if ``LiveDisplayState``
        mutates between the two calls.
        """
        with self._lock:
            return self._last_line_count

    def clear(self) -> None:
        """Drop all committed history and any unflushed partial line.

        Used by ``chat_commands._full_screen_reprint`` when the user
        presses Ctrl+O to toggle verbose trace mode in full-screen TUI:
        the old behaviour relied on ``Console.clear()`` blanking the
        terminal viewport, but in full-screen mode that escape sequence
        is just parsed as styled tokens and inserted into the buffer.
        Resetting the buffer in place is the equivalent operation here.

        The live-tail snapshot is owned by :class:`LiveDisplayState`, so
        this does not touch it — callers can clear that separately if
        needed.
        """
        with self._lock:
            had_content = bool(self._committed) or bool(self._partial)
            self._committed = []
            self._partial = ""
            self._last_line_count = 0
            self._committed_version += 1
            # Drop both render caches so the next ``tokens()`` / ``snapshot()``
            # call rebuilds against the empty state instead of handing the
            # renderer a stale list.
            self._cache_tokens = None
            self._cache_committed_version = -1
            self._cache_partial = None
            self._cache_live_lines = None
            self._cache_snapshot = None
            self._cache_snapshot_version = -1
            self._cache_snapshot_partial = None
            self._cache_snapshot_live = None
        if had_content:
            self._on_change()

    # ── wiring helpers ────────────────────────────────────────────

    def set_on_change(self, on_change: Callable[[], None]) -> None:
        """Replace the repaint callback after construction.

        Used by ``DatusCLI._init_tui_app`` to break the chicken-and-egg
        between :class:`TUIOutputBuffer` and :meth:`DatusApp.invalidate`:
        the buffer must exist *before* the app (Rich gets a console
        pointed at it), but only the app can supply a real
        ``invalidate`` callable.
        """
        with self._lock:
            self._on_change = on_change

    def set_live_state_snapshot_fn(self, live_state_snapshot_fn: Callable[[], List[LiveDisplayLine]]) -> None:
        """Replace the live-tail snapshot source post-construction."""
        with self._lock:
            self._live_state_snapshot_fn = live_state_snapshot_fn

    # ── viewport snapshot (lazy-row API) ──────────────────────────

    def snapshot(self) -> _BufferSnapshot:
        """Return an immutable view consumed by :class:`BufferedOutputControl`.

        Unlike :meth:`tokens`, this never builds a flat fragment stream — it
        only freezes the per-line containers so prompt_toolkit's renderer can
        call ``get_line(idx)`` for the rows it actually needs. The snapshot is
        cached and reused whenever the buffer state has not changed, so a
        sequence of paints over a static scrollback returns the exact same
        snapshot object and the control's own ``UIContent`` cache short-
        circuits all further work.
        """
        # Acquire the live snapshot outside our lock — its provider holds its
        # own lock and the two must never nest in opposite orders.
        live_lines = tuple(self._live_state_snapshot_fn() or [])

        with self._lock:
            partial = self._partial
            committed_version = self._committed_version
            cached = self._cache_snapshot
            if (
                cached is not None
                and self._cache_snapshot_version == committed_version
                and self._cache_snapshot_partial == partial
                and self._cache_snapshot_live == live_lines
            ):
                return cached
            committed = tuple(self._committed)

        partial_fragments: List[_StyledToken] = list(to_formatted_text(ANSI(partial))) if partial else []
        total = len(committed) + len(live_lines) + (1 if partial else 0)
        snap = _BufferSnapshot(committed, live_lines, partial_fragments, total)

        with self._lock:
            self._cache_snapshot = snap
            self._cache_snapshot_version = committed_version
            self._cache_snapshot_partial = partial
            self._cache_snapshot_live = live_lines
            # Keep ``render_line_count`` aligned with whichever code path the
            # renderer most recently consulted — both ``tokens`` and
            # ``snapshot`` report the same logical row count.
            self._last_line_count = total
        return snap


class BufferedOutputControl(UIControl):
    """``UIControl`` that renders :class:`TUIOutputBuffer` content lazily.

    Replaces :class:`prompt_toolkit.layout.controls.FormattedTextControl` for
    the scrollable output pane. The stock control hashes
    ``tuple(fragments_with_mouse_handlers)`` into its ``_content_cache`` key on
    every paint and rebuilds ``fragment_lines`` via ``split_lines`` — both
    operations are O(N) over the full scrollback, which dominates render cost
    once the buffer holds thousands of rows. By delegating to a per-line
    ``get_line`` callable backed by :class:`_BufferSnapshot`, prompt_toolkit
    only touches the rows that intersect the viewport, so type-latency stays
    flat regardless of scrollback depth.
    """

    def __init__(
        self,
        buffer: "TUIOutputBuffer",
        *,
        focusable: bool = True,
        show_cursor: bool = False,
        get_cursor_position: Optional[Callable[[], Optional[Point]]] = None,
        selection_provider: Optional[Callable[[], Optional[TranscriptSelection]]] = None,
        search_provider: Optional[Callable[[], Optional[SearchState]]] = None,
    ) -> None:
        self._buffer = buffer
        self._focusable = to_filter(focusable)
        self.show_cursor = show_cursor
        self.get_cursor_position = get_cursor_position
        # Pulls the live :class:`TranscriptSelection` from the owning app
        # at every paint; ``None`` (or returning ``None``) disables the
        # highlight path entirely so the existing fast path stays
        # untouched for tests / non-TUI callers.
        self._selection_provider = selection_provider or (lambda: None)
        # Parallel hook for the Ctrl+F find-in-scrollback overlay (see
        # :mod:`datus.cli.tui.search`). Composed under the selection
        # overlay so user-initiated highlights always win visually.
        self._search_provider = search_provider or (lambda: None)
        # One-slot ``UIContent`` cache. The key threads through every
        # input ``create_content`` consults so a stale ``UIContent`` is
        # never returned when the selection or search state changes mid-
        # paint. ``create_content`` is called multiple times per render
        # run (preferred_height, then the actual paint); returning the
        # same ``UIContent`` lets prompt_toolkit's per-line height cache
        # stay warm.
        self._uicontent_key: Optional[Tuple] = None
        self._uicontent: Optional[UIContent] = None

    def is_focusable(self) -> bool:
        return self._focusable()

    def create_content(self, width: int, height: Optional[int]) -> UIContent:
        snap = self._buffer.snapshot()
        cursor_position: Optional[Point] = None
        if self.get_cursor_position is not None:
            cursor_position = self.get_cursor_position()
        cursor_key = (cursor_position.x, cursor_position.y) if cursor_position is not None else None

        # Cache key incorporates the selection range so a drag invalidates
        # the cached ``UIContent`` even though the snapshot identity is
        # unchanged. ``selection.version`` is bumped on every state
        # transition; including the range tuple as well guards against a
        # bug where two distinct ranges share a version counter (shouldn't
        # happen today but is cheap insurance).
        selection = self._selection_provider() if self._selection_provider else None
        sel_range_key: Optional[Tuple[int, int, int, int]] = None
        sel_version = 0
        if selection is not None and not selection.is_empty():
            rng = selection.range()
            if rng is not None:
                start, end = rng
                sel_range_key = (start.line, start.column, end.line, end.column)
            sel_version = selection.version

        # Search overlay state — same cache discipline as the selection
        # overlay. Threading both ``version`` *and* ``current_idx`` plus
        # a truthy "are there matches" flag means a Ctrl+F user typing a
        # query (matches change) or pressing Enter (current_idx changes)
        # invalidates the cache deterministically.
        search = self._search_provider() if self._search_provider else None
        search_active = search is not None and search.is_active()
        search_key = (search.version, search.current_idx, len(search.matches)) if search_active else None

        key = (id(snap), cursor_key, sel_version, sel_range_key, search_key)
        if self._uicontent_key == key and self._uicontent is not None:
            return self._uicontent

        # Always wrap with the blank-line padder: prompt_toolkit's
        # ``Window._copy_body`` only registers ``rowcol_to_yx`` entries
        # for cells it actually paints. A truly-empty fragment list paints
        # nothing → no rowcol entries → the outer mouse handler can't
        # resolve any (y, x) to a line index and falls back to the
        # sentinel ``Point(0, 0)``. During a selection drag that fallback
        # snaps the head all the way to the top of the buffer and the
        # highlight expands to every line above the anchor. Emitting a
        # single space gives every visible row a clickable cell at col 0
        # without changing the visible output (the cell renders as a
        # space, which a blank row was already showing anyway).
        get_line = _padded_blank_line(snap.get_line)
        # Apply overlays bottom-up: search first, selection on top — a
        # user dragging across search hits gets the selection styling
        # (reverse video) on the same characters as the search hit, and
        # prompt_toolkit's style merger combines them sensibly.
        if search_active:
            get_line = _search_aware_get_line(get_line, search)
        if selection is not None and not selection.is_empty():
            get_line = _selection_aware_get_line(get_line, selection)

        content = UIContent(
            get_line=get_line,
            line_count=snap.total,
            cursor_position=cursor_position or Point(x=0, y=0),
            show_cursor=self.show_cursor,
        )
        self._uicontent_key = key
        self._uicontent = content
        return content

    def mouse_handler(self, mouse_event: MouseEvent):  # noqa: ANN201
        # Default: defer to ``Window`` / key bindings. ``DatusApp`` overrides
        # this attribute in place to wire scroll-wheel handling.
        return NotImplemented


def _padded_blank_line(
    base_get_line: Callable[[int], List[_StyledToken]],
) -> Callable[[int], List[_StyledToken]]:
    """Wrap ``base_get_line`` so empty fragment lists render with a single space.

    Rationale: ``Window._copy_body`` only writes ``rowcol_to_yx`` entries
    when the line contains at least one printable character. An empty
    fragment list paints nothing, so prompt_toolkit's outer mouse handler
    cannot translate ``(screen_y, screen_x)`` back to a line index for
    that row — it falls through to ``Point(0, 0)`` which, during a
    selection drag, snaps the highlight to the very top of the buffer.
    A single space fragment is invisible (it paints as a blank cell, same
    visual as the empty row) but guarantees one rowcol entry per visible
    line. The wrap is applied to the rendered ``get_line`` only —
    :func:`extract_selection_text` reads the raw snapshot directly so the
    padding never leaks into clipboard text.
    """

    def get_line(idx: int) -> List[_StyledToken]:
        line = base_get_line(idx)
        if not line:
            return [("", " ")]
        # A non-empty fragment list with zero total characters (e.g. a
        # single ``("style", "")``) hits the same bug.
        if all(not (len(f) > 1 and f[1]) for f in line):
            return [("", " ")]
        return line

    return get_line


def _selection_aware_get_line(
    base_get_line: Callable[[int], List[_StyledToken]],
    selection: TranscriptSelection,
) -> Callable[[int], List[_StyledToken]]:
    """Wrap ``base_get_line`` so selected rows render with a highlight style.

    Only lines that intersect the selection are rewritten; rows outside the
    range are forwarded untouched (and unchanged identity). The wrapper is
    rebuilt on every cache miss in :meth:`BufferedOutputControl.create_content`
    so a stale ``selection`` reference can never paint over a fresh snapshot.
    """

    def get_line(idx: int) -> List[_StyledToken]:
        line = base_get_line(idx)
        bounds = selection.columns_for_line(idx)
        if bounds is None:
            return line
        start_col, end_col = bounds
        return split_line_for_selection(line, start_col, end_col)

    return get_line


def _search_aware_get_line(
    base_get_line: Callable[[int], List[_StyledToken]],
    search: SearchState,
) -> Callable[[int], List[_StyledToken]]:
    """Wrap ``base_get_line`` so search hits render with the match style.

    Multi-hit rows are processed left-to-right; each region is spliced
    via :func:`split_line_for_selection`. Because that helper splits on
    character indices (not visual columns) and only adds new fragments
    *without* shifting any existing character offsets, applying it
    repeatedly with non-overlapping target ranges is safe. Overlapping
    matches on the same row are merged by emitting them in order — the
    later region's style wins on the overlap, which is exactly the
    behaviour we want for the "current" match (it's always emitted last
    when it shares a row with siblings via :meth:`SearchState.matches_on_line`'s
    start-sorted order, so we make a second pass to paint it after the
    base hits).
    """

    def get_line(idx: int) -> List[_StyledToken]:
        line = base_get_line(idx)
        regions = search.matches_on_line(idx)
        if not regions:
            return line
        # Paint non-current hits first so the current match's distinctive
        # style is applied last and wins any overlap.
        for start, end, is_current in regions:
            if is_current:
                continue
            line = split_line_for_selection(line, start, end, selection_style="class:search-match")
        for start, end, is_current in regions:
            if not is_current:
                continue
            line = split_line_for_selection(line, start, end, selection_style="class:search-match.current")
        return line

    return get_line


def extract_selection_text(
    buffer: "TUIOutputBuffer",
    selection: TranscriptSelection,
) -> str:
    """Return the plain text covered by ``selection`` as a single string.

    Lines are joined with ``\\n``. Style information (ANSI / Rich classes)
    is stripped — the clipboard payload is intentionally plain so it
    pastes cleanly into editors and chat windows. Visual columns at the
    selection boundary are honoured, so a partial click on a CJK glyph
    selects the whole glyph (matches :func:`split_line_for_selection`'s
    snap-past behaviour).

    Trailing whitespace is stripped from each extracted line. Renderables
    with background fill (e.g. the bordered USER message panel) pad each
    row with spaces out to the pane width; without this strip those
    padding spaces would land on the clipboard and force the user to
    clean them up manually after pasting.
    """
    rng = selection.range()
    if rng is None:
        return ""
    snap = buffer.snapshot()
    out_lines: List[str] = []
    start, end = rng
    for line_idx in range(start.line, end.line + 1):
        if line_idx < 0 or line_idx >= snap.total:
            continue
        fragments = snap.get_line(line_idx)
        if start.line == end.line:
            from_col, to_col = start.column, end.column
        elif line_idx == start.line:
            from_col, to_col = start.column, line_char_count(fragments)
        elif line_idx == end.line:
            from_col, to_col = 0, end.column
        else:
            from_col, to_col = 0, line_char_count(fragments)
        out_lines.append(extract_plain_text_between(fragments, from_col, to_col).rstrip())
    return "\n".join(out_lines)
