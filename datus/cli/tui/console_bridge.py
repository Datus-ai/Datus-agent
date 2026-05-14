# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers bridging Rich output to a running prompt_toolkit Application.

In legacy ``full_screen=False`` mode ``patch_stdout(raw=True)`` replaced
``sys.stdout`` with a proxy that renders new lines above the pinned
status bar + input. ``run_in_terminal_sync`` was the worker-thread entry
that ensured a single print landed in the right place above that pinned
area without fighting the renderer.

In the current ``full_screen=True`` layout there is no scrollback to
inject into: every Rich ``console.print`` goes into the in-memory
:class:`TUIOutputBuffer`, which is itself thread-safe. The wrapper
collapses to a direct ``func()`` call when the Application is running in
full-screen — both because there is no pinned area to escape and because
``in_terminal()`` under ``full_screen=True`` *erases* the layout before
running ``func`` (which would blank the entire output pane).

The legacy off-loop ``run_coroutine_threadsafe(in_terminal())`` branch is
retained for any callers that still construct a ``full_screen=False``
Application (interaction modals, sub-wizards, etc.) so their existing
output behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.application.run_in_terminal import in_terminal, run_in_terminal


def run_in_terminal_sync(func: Callable[[], None]) -> None:
    """Schedule ``func`` to print above the TUI and return after it runs.

    Safe to call from any thread. Four paths:

    * **No Application active**: direct call.
    * **Full-screen Application (the main Datus TUI)**: direct call. The
      output buffer is thread-safe and there is no scrollback region to
      avoid; ``in_terminal()`` would blank the full-screen layout for
      every print, which is destructive.
    * **On the prompt_toolkit event loop of a non-full-screen
      Application** (e.g. legacy modals): dispatch via
      :func:`run_in_terminal` and return immediately.
    * **Off-loop thread with a non-full-screen Application**: submit via
      :func:`asyncio.run_coroutine_threadsafe` and block on completion.
    """
    app = get_app_or_none()
    if app is None:
        func()
        return

    # full_screen mode: the entire terminal is owned by the layout; the
    # output pane is fed by TUIOutputBuffer which handles its own thread
    # safety. ``in_terminal()`` here would erase the layout for every
    # print, so just invoke directly — buffer.write under its lock is the
    # right level of synchronization.
    if getattr(app, "full_screen", False):
        func()
        return

    try:
        asyncio.get_running_loop()
        on_event_loop = True
    except RuntimeError:
        on_event_loop = False

    if on_event_loop:
        # Fire and forget — the callback is scheduled on the same loop and
        # will run before control returns to the key handler.
        run_in_terminal(func)
        return

    loop = getattr(app, "loop", None)
    if loop is None:
        # Application exists but its loop isn't running yet (pre-``run`` or
        # post-shutdown). Safe to invoke directly — there's no pinned area
        # to preserve.
        func()
        return

    async def _wrap() -> None:
        async with in_terminal():
            func()

    try:
        cf = asyncio.run_coroutine_threadsafe(_wrap(), loop)
    except RuntimeError:
        # Loop closed between the getattr and the submit — just run inline.
        func()
        return
    try:
        cf.result()
    except Exception:  # pragma: no cover - defensive
        # Propagating would leak into background threads; callers that care
        # should wrap ``func`` themselves. Here we swallow to keep the TUI
        # responsive.
        pass
