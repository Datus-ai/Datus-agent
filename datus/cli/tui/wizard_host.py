# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Protocol and helpers for embedding sub-wizards inside the main TUI.

Background
----------
Historically every interactive slash command (``/model``, ``/agent``,
``ask_user``/``confirm``, ``/effort``, ``/language``, ``/skill``,
``/mcp``, ``/datasource``, ``/bootstrap`` …) built its own
``prompt_toolkit.Application(full_screen=False)`` and ran it through
``DatusApp.suspend_input()`` (which uses ``in_terminal()`` to release
stdin). In full-screen mode this clears the entire screen — the user
loses sight of LLM output / sidebar / task list while answering a
prompt.

This module switches to **embedded panels**: each sub-wizard exposes a
container + key bindings + a ``done_future``. The main :class:`DatusApp`
hosts the panel in a ``DynamicContainer`` slot that replaces the
``status bar + input + hint`` rows. The ``top_row`` (output + sidebar)
keeps rendering above, just with fewer rows. When the wizard's key
binding sets ``done_future``, the host unmounts and restores the
input area.

Threading
---------
Worker threads call :meth:`DatusApp.run_wizard` which:

1. Allocates the ``done_future`` on the main loop (via
   ``loop.create_future()``)
2. Builds the panel synchronously on the worker
3. Schedules ``mount_wizard`` on the main loop via
   ``call_soon_threadsafe``
4. Blocks the worker on the future (resolved by the wizard's key
   binding from the main loop)
5. Schedules ``unmount_wizard`` to restore normal layout

No nested ``asyncio.run`` is needed — the wizard's interactivity
shares the parent app's event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import AnyContainer

if TYPE_CHECKING:
    from prompt_toolkit.layout.containers import Container


@dataclass
class EmbeddedWizard:
    """A sub-wizard panel mounted inside ``DatusApp``.

    Attributes
    ----------
    container :
        The root container for the wizard's UI. Replaces the bottom
        section of the main TUI while mounted.
    key_bindings :
        Wizard-scoped key bindings. Attached to ``container`` (or its
        wrapping HSplit) so they only activate while focus is inside
        the wizard subtree — prompt_toolkit's focus model auto-scopes
        them; no manual mount/unmount of the bindings themselves is
        needed.
    first_focus :
        The container to give focus when the wizard mounts. ``None``
        means the host leaves focus wherever it was (rarely useful;
        most wizards point at their first interactive widget).
    done_future :
        Future the wizard resolves when finished. Resolve with the
        wizard's result type, or ``None`` for cancel (ESC). The host
        treats ``done_future.cancelled()`` the same as ``None``.
    """

    container: AnyContainer
    key_bindings: KeyBindings
    first_focus: Optional["Container"]
    done_future: asyncio.Future


def make_panel_factory(
    builder: Callable[[asyncio.Future], EmbeddedWizard],
) -> Callable[[asyncio.Future], EmbeddedWizard]:
    """Trivial passthrough — kept as a named hook for typing & docs.

    Wizards typically expose ``build_embedded_panel(done_future)`` as
    an instance method; ``run_wizard`` accepts any callable matching
    this signature. This helper exists so that callers can write
    ``tui_app.run_wizard(make_panel_factory(self.build_embedded_panel))``
    with explicit intent, but ``tui_app.run_wizard(self.build_embedded_panel)``
    works the same.
    """
    return builder


def resolve_cancel(done_future: asyncio.Future) -> None:
    """Resolve ``done_future`` as a cancel (None result).

    Wizards' ESC / Ctrl+C handlers call this. If the future is already
    done (e.g. duplicate signal), silently no-op so handlers stay
    idempotent.
    """
    if not done_future.done():
        done_future.set_result(None)


def resolve_with(done_future: asyncio.Future, result: Any) -> None:
    """Resolve ``done_future`` with a non-None result.

    Same idempotence guarantee as :func:`resolve_cancel`.
    """
    if not done_future.done():
        done_future.set_result(result)
