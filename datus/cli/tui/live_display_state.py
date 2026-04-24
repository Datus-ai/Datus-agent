# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared state for the pinned live-render region above the status bar.

The :class:`DatusApp` TUI reads this state on its main event loop thread to
paint the rolling subagent / processing-tool region. Writers (the
``InlineStreamingContext`` refresh daemon) mutate the state and then invoke
``invalidate_cb`` so prompt_toolkit schedules a single repaint that covers
the live region, the status bar, and the input bar in one pass — without
ever fighting ``patch_stdout(raw=True)`` for cursor position.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# Per-subagent pinned-window budget.
#
# Each active subagent contributes a block of ``1 + TOOL_LINES_PER_GROUP``
# rows (one header line + its rolling tool tail).
TOOL_LINES_PER_GROUP = 2

# Hard safety cap — the pinned region never holds more than this many rows
# regardless of what the terminal reports, to keep unbounded runaway fan-out
# from eating unlimited memory. The *effective* ceiling per render is much
# lower, set dynamically from the current terminal height via
# :meth:`LiveDisplayState.max_rows`.
MAX_LIVE_LINES_TOTAL = 200

# Rows reserved for the status bar, separators, hint, and a 1-line input
# area. The actual input area may grow up to 15 rows when the user pastes
# multi-line content; prompt_toolkit arbitrates that at render time by
# shrinking us first (``dont_extend_height=True``).
_PINNED_RESERVED_ROWS = 8
# Floor so the pinned area still renders meaningfully on narrow terminals.
_PINNED_MIN_ROWS = 3


def compute_pinned_max_rows(terminal_rows: int) -> int:
    """Return the pinned-region row budget for a terminal of ``terminal_rows``.

    The result is clamped between :data:`_PINNED_MIN_ROWS` and
    :data:`MAX_LIVE_LINES_TOTAL` and accounts for the other bottom-area
    widgets via :data:`_PINNED_RESERVED_ROWS`.
    """
    if terminal_rows <= 0:
        return _PINNED_MIN_ROWS
    budget = terminal_rows - _PINNED_RESERVED_ROWS
    if budget < _PINNED_MIN_ROWS:
        return _PINNED_MIN_ROWS
    if budget > MAX_LIVE_LINES_TOTAL:
        return MAX_LIVE_LINES_TOTAL
    return budget


@dataclass
class LiveDisplayLine:
    """Single pinned line expressed as prompt_toolkit formatted-text fragments."""

    segments: List[Tuple[str, str]] = field(default_factory=list)


class LiveDisplayState:
    """Thread-safe state for the TUI pinned live-render region.

    The state holds at most ``MAX_LIVE_LINES_TOTAL`` rolling lines. All mutations
    acquire an internal lock; reads (``snapshot``, ``is_active``, ``line_count``)
    also acquire it, so the main loop always sees consistent snapshots.

    ``invalidate_cb`` is a zero-arg callable the state invokes after each
    mutation to ask prompt_toolkit to repaint. The callback must itself be
    thread-safe (``prompt_toolkit.Application.invalidate`` qualifies because
    it dispatches via ``loop.call_soon_threadsafe``).
    """

    def __init__(self, invalidate_cb: Optional[Callable[[], None]] = None) -> None:
        self._lock = threading.Lock()
        self._lines: List[LiveDisplayLine] = []
        self._invalidate: Callable[[], None] = invalidate_cb or (lambda: None)
        # Producer of the current pinned-row budget. ``DatusApp`` wires in a
        # callable that reads the live terminal height so the cap shrinks
        # when the user resizes down and grows when they resize up. When
        # unset (non-TUI tests), the hard safety cap is used.
        self._max_rows_cb: Callable[[], int] = lambda: MAX_LIVE_LINES_TOTAL

    def set_invalidate(self, invalidate_cb: Callable[[], None]) -> None:
        """Register or replace the repaint callback.

        Used to break the chicken-and-egg between :class:`LiveDisplayState` and
        ``DatusApp.invalidate``: the state is constructed first with a no-op
        callback and the real one is attached after the app exists.
        """
        with self._lock:
            self._invalidate = invalidate_cb

    def set_max_rows_provider(self, provider: Callable[[], int]) -> None:
        """Install a callable that returns the *current* pinned-row budget.

        Writers call :meth:`max_rows` before shaping their line lists; the
        provider is resolved on every call so terminal-resize events flow
        through without further plumbing.
        """
        with self._lock:
            self._max_rows_cb = provider

    def max_rows(self) -> int:
        """Return the current pinned-row ceiling (terminal-height aware)."""
        try:
            value = int(self._max_rows_cb())
        except Exception:
            value = MAX_LIVE_LINES_TOTAL
        if value < _PINNED_MIN_ROWS:
            return _PINNED_MIN_ROWS
        if value > MAX_LIVE_LINES_TOTAL:
            return MAX_LIVE_LINES_TOTAL
        return value

    def set_lines(self, lines: List[LiveDisplayLine]) -> None:
        """Replace the pinned lines with ``lines`` (trimmed to ``max_rows()``).

        Writers are expected to pre-shape ``lines`` into a flat sequence;
        this method enforces the terminal-aware ceiling so even a writer
        that ignores :meth:`max_rows` cannot overflow the bottom area.
        No-op when the trimmed payload equals the current state, so the
        per-token streaming hot path doesn't wake prompt_toolkit's main
        loop when nothing visible changed.
        """
        if not lines:
            with self._lock:
                changed = bool(self._lines)
                self._lines = []
                cb = self._invalidate
            if changed:
                cb()
            return
        cap = self.max_rows()
        trimmed = list(lines[-cap:])
        with self._lock:
            changed = trimmed != self._lines
            if changed:
                self._lines = trimmed
            cb = self._invalidate
        if changed:
            cb()

    def clear(self) -> None:
        with self._lock:
            changed = bool(self._lines)
            self._lines = []
            cb = self._invalidate
        if changed:
            cb()

    def snapshot(self) -> List[LiveDisplayLine]:
        with self._lock:
            return list(self._lines)

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._lines)

    def line_count(self) -> int:
        with self._lock:
            return len(self._lines)
