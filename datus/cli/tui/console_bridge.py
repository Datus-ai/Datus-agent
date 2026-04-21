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
from prompt_toolkit.application.run_in_terminal import run_in_terminal


def run_in_terminal_sync(func: Callable[[], None]) -> None:
    """Schedule ``func`` to print above the TUI and return after it runs.

    Safe to call from any thread. When no Application is active (non-TTY
    fallback path), simply invokes ``func`` directly.
    """
    app = get_app_or_none()
    if app is None:
        func()
        return

    future = run_in_terminal(func)

    # Callers invoked from a prompt_toolkit key handler run on the same
    # asyncio loop that schedules the ``run_in_terminal`` callback. Blocking
    # that thread on ``future.result()`` deadlocks the loop. Detect the
    # in-loop case and let the scheduled callback finish asynchronously.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        on_event_loop = False
    else:
        on_event_loop = True

    if on_event_loop:
        return

    # ``run_in_terminal`` returns an asyncio.Future; ``result()`` blocks until
    # the callback completes on the Application event loop.
    try:
        future.result()
    except Exception:  # pragma: no cover - defensive
        # Propagating would leak into background threads; callers that care
        # should wrap ``func`` themselves. Here we swallow to keep the TUI
        # responsive.
        pass
