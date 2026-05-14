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

from prompt_toolkit.formatted_text import ANSI, to_formatted_text

from datus.cli.tui.live_display_state import LiveDisplayLine

_StyledToken = Tuple[str, str]


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
            # Drop the render cache so the next ``tokens()`` rebuilds against
            # the empty state instead of handing the renderer a stale list.
            self._cache_tokens = None
            self._cache_committed_version = -1
            self._cache_partial = None
            self._cache_live_lines = None
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
