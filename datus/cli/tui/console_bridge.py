# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Helpers bridging Rich output to a running prompt_toolkit Application.

``patch_stdout(raw=True)`` inside :class:`DatusApp` replaces ``sys.stdout``
with a proxy that renders new lines above the pinned status bar + input.
Rich ``Console`` instances that write to ``sys.stdout`` therefore scroll
correctly without fighting the bottom area.

Two concerns remain:

1. Rich ``Live`` must not also patch stdout (otherwise both layers fight).
   Callers construct ``Live(redirect_stdout=False, redirect_stderr=False)``;
   this module does not own that choice but documents it here for reference.
2. Some outputs — e.g. Pygments-highlighted user echo — prefer to appear
   exactly where the cursor currently is. ``run_in_terminal_sync`` wraps
   ``prompt_toolkit.application.run_in_terminal`` so callers can schedule a
   print and block until it is rendered, useful from worker threads.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from prompt_toolkit.application import get_app_or_none
from prompt_toolkit.application.run_in_terminal import in_terminal, run_in_terminal


def run_in_terminal_sync(func: Callable[[], None]) -> None:
    """Schedule ``func`` to print above the TUI and return after it runs.

    Safe to call from any thread. When no Application is active (non-TTY
    fallback path), simply invokes ``func`` directly.

    Three paths:

    * **On the prompt_toolkit event loop** (key-binding callbacks): dispatch
      via :func:`run_in_terminal` and return immediately; blocking on the
      future would deadlock the loop that must run the callback.
    * **Off-loop thread (e.g. the streaming refresh daemon) with a live
      Application**: ``asyncio.get_running_loop()`` raises, so we instead
      submit a coroutine via :func:`asyncio.run_coroutine_threadsafe` against
      ``app.loop`` and block on its completion. Using
      :func:`run_in_terminal` here is unsafe because it calls
      ``asyncio.ensure_future`` which needs a running loop on the calling
      thread (Python 3.12+ no longer auto-creates one).
    * **No Application**: direct call.
    """
    app = get_app_or_none()
    if app is None:
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
