# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/profile`` pickers rendered as prompt_toolkit ``Application``s.

Mirrors the dual-mode pattern of the other migrated wizards
(:mod:`datus.cli.effort_app` et al): :meth:`run` constructs a transient
standalone ``Application(full_screen=False)`` for non-TUI callers;
:meth:`build_embedded_panel` returns an :class:`EmbeddedWizard` that the
active :class:`~datus.cli.tui.app.DatusApp` mounts in its bottom slot
(status bar / input row are replaced, output pane stays visible above).

Two apps:
  * ``ProfilePickerApp`` — primary profile selection dialog.
  * ``DangerousConfirmApp`` — second confirmation for entering Dangerous mode.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console

from datus.cli.cli_styles import CLR_CURRENT, CLR_CURSOR, SYM_ARROW, print_error, render_tui_title_bar
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.tui.wizard_host import EmbeddedWizard

logger = get_logger(__name__)


# Short one-line descriptions shown beside each profile.
_PROFILE_DESCRIPTIONS = {
    "normal": "Read-only + confirm every write",
    "auto": "Workspace writes auto; DB/MCP still ask",
    "dangerous": "Nearly all writes auto (see warning)",
}


class ProfilePickerApp:
    """Primary picker for permission profile selection.

    Returns the selected profile name (``"normal"``/``"auto"``/``"dangerous"``)
    on Enter, or ``None`` if the user cancels (Esc / Ctrl-C). Selecting the
    current profile is allowed at this layer — the caller is responsible for
    treating that as a no-op.
    """

    _PROFILES = ("normal", "auto", "dangerous")

    def __init__(self, console: Console, current: str = "normal"):
        self._console = console
        self._current = current if current in self._PROFILES else "normal"
        self._idx = self._PROFILES.index(self._current)
        # Dual-mode finish hook (see ``EffortApp`` for the rationale).
        self._on_done: Optional[Callable[[Optional[str]], None]] = None
        self._app: Optional[Application] = None
        self._list_window: Optional[Window] = None

    # ── Standalone entry point ────────────────────────────────────

    def run(self) -> Optional[str]:
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        self._app = Application(
            layout=Layout(root, focused_element=self._list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._on_done = lambda result: self._app.exit(result=result)
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("ProfilePickerApp crashed: %s", exc)
            print_error(self._console, f"/profile error: {exc}")
            return None
        finally:
            self._on_done = None
            self._app = None

    def build_embedded_panel(self, done_future: "asyncio.Future") -> "EmbeddedWizard":
        from datus.cli.tui.wizard_host import EmbeddedWizard, resolve_cancel, resolve_with

        def _done(result):
            if result is None:
                resolve_cancel(done_future)
            else:
                resolve_with(done_future, result)

        self._on_done = _done
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        return EmbeddedWizard(
            container=root,
            key_bindings=kb,
            first_focus=self._list_window,
            done_future=done_future,
        )

    # ── Internals ────────────────────────────────────────────────

    def _finish(self, result: Optional[str]) -> None:
        if self._on_done is None:
            return
        self._on_done(result)

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):  # noqa: ANN001
            self._idx = max(0, self._idx - 1)

        @kb.add("down")
        def _down(event):  # noqa: ANN001
            self._idx = min(len(self._PROFILES) - 1, self._idx + 1)

        @kb.add("enter")
        def _enter(event):  # noqa: ANN001
            self._finish(self._PROFILES[self._idx])

        @kb.add("escape")
        def _escape(event):  # noqa: ANN001
            self._finish(None)

        @kb.add("c-c")
        def _ctrl_c(event):  # noqa: ANN001
            self._finish(None)

        return kb

    def _build_root_container(self, kb: KeyBindings) -> HSplit:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Permission Profile Selection")),
            height=1,
        )
        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=1, max=2),
        )
        self._list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True, key_bindings=kb),
            always_hide_cursor=True,
            height=Dimension(min=3, max=len(self._PROFILES) + 1),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        return HSplit(
            [
                title_bar,
                header_window,
                Window(height=1, char="\u2500"),
                self._list_window,
                Window(height=1, char="\u2500"),
                hint_window,
            ]
        )

    def _render_header(self) -> List[Tuple[str, str]]:
        return [
            ("bold", "  Select permission profile"),
            ("", f"  [current: {self._current}]"),
        ]

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        for i, name in enumerate(self._PROFILES):
            desc = _PROFILE_DESCRIPTIONS[name]
            label = f"{name:<10}  {desc}"
            is_current = name == self._current
            if is_current:
                label += "  \u2190 current"
            if i == self._idx:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            elif is_current:
                lines.append((CLR_CURRENT, f"    {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  \u2191\u2193 navigate   Enter select   Esc cancel")]


class DangerousConfirmApp:
    """Second confirmation before switching into Dangerous mode.

    Returns ``True`` only if the user explicitly selects the Enable option.
    Default highlight is Cancel to reduce accidental activation.
    """

    _CHOICES = (
        ("cancel", "Cancel (stay on current profile)"),
        ("enable", "Enable Dangerous for this session"),
    )

    def __init__(self, console: Console):
        self._console = console
        self._idx = 0  # default: Cancel
        self._on_done: Optional[Callable[[Optional[str]], None]] = None
        self._app: Optional[Application] = None
        self._list_window: Optional[Window] = None

    def run(self) -> bool:
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        self._app = Application(
            layout=Layout(root, focused_element=self._list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )
        self._on_done = lambda result: self._app.exit(result=result)
        try:
            result = self._app.run()
        except KeyboardInterrupt:
            return False
        except Exception as exc:
            logger.error("DangerousConfirmApp crashed: %s", exc)
            print_error(self._console, f"/profile confirm error: {exc}")
            return False
        finally:
            self._on_done = None
            self._app = None
        return result == "enable"

    def build_embedded_panel(self, done_future: "asyncio.Future") -> "EmbeddedWizard":
        from datus.cli.tui.wizard_host import EmbeddedWizard, resolve_with

        # The host normalises ``None`` to "cancel" (the safer choice for
        # Dangerous), so resolve the future with whichever sentinel the
        # bindings produce.
        self._on_done = lambda result: resolve_with(done_future, result or "cancel")
        kb = self._build_key_bindings()
        root = self._build_root_container(kb)
        return EmbeddedWizard(
            container=root,
            key_bindings=kb,
            first_focus=self._list_window,
            done_future=done_future,
        )

    def _finish(self, result: Optional[str]) -> None:
        if self._on_done is None:
            return
        self._on_done(result)

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):  # noqa: ANN001
            self._idx = max(0, self._idx - 1)

        @kb.add("down")
        def _down(event):  # noqa: ANN001
            self._idx = min(len(self._CHOICES) - 1, self._idx + 1)

        @kb.add("enter")
        def _enter(event):  # noqa: ANN001
            self._finish(self._CHOICES[self._idx][0])

        @kb.add("escape")
        def _escape(event):  # noqa: ANN001
            self._finish("cancel")

        @kb.add("c-c")
        def _ctrl_c(event):  # noqa: ANN001
            self._finish("cancel")

        return kb

    def _build_root_container(self, kb: KeyBindings) -> HSplit:
        title_bar = Window(
            content=FormattedTextControl(lambda: render_tui_title_bar("Dangerous Profile Confirmation")),
            height=1,
        )
        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=8, max=10),
        )
        self._list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True, key_bindings=kb),
            always_hide_cursor=True,
            height=Dimension(min=2, max=2),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        return HSplit(
            [
                title_bar,
                header_window,
                Window(height=1, char="\u2500"),
                self._list_window,
                Window(height=1, char="\u2500"),
                hint_window,
            ]
        )

    def _render_header(self) -> List[Tuple[str, str]]:
        return [
            ("bold fg:ansired", "  WARNING: DANGEROUS PROFILE - Explicit Confirmation Required\n"),
            ("", "\n"),
            ("", "  Switching to Dangerous will auto-execute:\n"),
            ("", "    \u2022 All DB writes (including DDL, DELETE)\n"),
            ("", "    \u2022 All BI/Scheduler writes (including deletes)\n"),
            ("", "    \u2022 All MCP tools\n"),
            ("", "    \u2022 All skills\n"),
            ("", "\n"),
            ("", "  Still protected: writes outside workspace require ASK;\n"),
            ("", "  ~/.datus internals remain hidden."),
        ]

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        for i, (_key, label) in enumerate(self._CHOICES):
            if i == self._idx:
                lines.append((CLR_CURSOR, f"  {SYM_ARROW} {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  \u2191\u2193 navigate   Enter confirm   Esc cancel")]
