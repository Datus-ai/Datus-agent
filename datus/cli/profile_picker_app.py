# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Self-contained ``/profile`` pickers rendered as prompt_toolkit ``Application``s.

Mirrors the lightweight inline picker pattern of :mod:`datus.cli.agent_picker_app`
— avoids ``radiolist_dialog`` / ``button_dialog`` which are fullscreen and
compete with the outer REPL's cursor-position requests, producing sluggish
key response and visual churn.

Two apps:
  * ``ProfilePickerApp`` — primary profile selection dialog.
  * ``DangerousConfirmApp`` — second confirmation for entering Dangerous mode.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from rich.console import Console

from datus.cli.cli_styles import print_error
from datus.utils.loggings import get_logger

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
        self._app = self._build_app()

    def run(self) -> Optional[str]:
        try:
            return self._app.run()
        except KeyboardInterrupt:
            return None
        except Exception as exc:
            logger.error("ProfilePickerApp crashed: %s", exc)
            print_error(self._console, f"/profile error: {exc}")
            return None

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            self._idx = max(0, self._idx - 1)

        @kb.add("down")
        def _down(event):
            self._idx = min(len(self._PROFILES) - 1, self._idx + 1)

        @kb.add("enter")
        def _enter(event):
            event.app.exit(self._PROFILES[self._idx])

        @kb.add("escape")
        def _escape(event):
            event.app.exit(None)

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit(None)

        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=1, max=2),
        )
        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=3, max=len(self._PROFILES) + 1),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        root = HSplit(
            [
                header_window,
                Window(height=1, char="─"),
                list_window,
                Window(height=1, char="─"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
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
            if name == self._current:
                label += "  <- current"
            if i == self._idx:
                lines.append(("reverse", f"  -> {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  up/down navigate   Enter select   Esc cancel")]


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
        self._app = self._build_app()

    def run(self) -> bool:
        try:
            result = self._app.run()
        except KeyboardInterrupt:
            return False
        except Exception as exc:
            logger.error("DangerousConfirmApp crashed: %s", exc)
            print_error(self._console, f"/profile confirm error: {exc}")
            return False
        return result == "enable"

    def _build_app(self) -> Application:
        kb = KeyBindings()

        @kb.add("up")
        def _up(event):
            self._idx = max(0, self._idx - 1)

        @kb.add("down")
        def _down(event):
            self._idx = min(len(self._CHOICES) - 1, self._idx + 1)

        @kb.add("enter")
        def _enter(event):
            event.app.exit(self._CHOICES[self._idx][0])

        @kb.add("escape")
        def _escape(event):
            event.app.exit("cancel")

        @kb.add("c-c")
        def _ctrl_c(event):
            event.app.exit("cancel")

        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(min=8, max=10),
        )
        list_window = Window(
            content=FormattedTextControl(self._render_list, focusable=True),
            always_hide_cursor=True,
            height=Dimension(min=2, max=2),
        )
        hint_window = Window(
            content=FormattedTextControl(self._render_footer_hint, focusable=False),
            height=1,
        )

        root = HSplit(
            [
                header_window,
                Window(height=1, char="─"),
                list_window,
                Window(height=1, char="─"),
                hint_window,
            ]
        )

        return Application(
            layout=Layout(root, focused_element=list_window),
            key_bindings=kb,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
        )

    def _render_header(self) -> List[Tuple[str, str]]:
        return [
            ("fg:ansired", "  WARNING: DANGEROUS PROFILE - Explicit Confirmation Required\n"),
            ("", "\n"),
            ("", "  Switching to Dangerous will auto-execute:\n"),
            ("", "    - All DB writes (including DDL, DELETE)\n"),
            ("", "    - All BI/Scheduler writes (including deletes)\n"),
            ("", "    - All MCP tools\n"),
            ("", "    - All skills\n"),
            ("", "\n"),
            ("", "  Still protected: writes outside workspace require ASK;\n"),
            ("", "  ~/.datus internals remain hidden."),
        ]

    def _render_list(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        for i, (_key, label) in enumerate(self._CHOICES):
            if i == self._idx:
                lines.append(("reverse", f"  -> {label}\n"))
            else:
                lines.append(("", f"    {label}\n"))
        return lines

    def _render_footer_hint(self) -> List[Tuple[str, str]]:
        return [("", "  up/down navigate   Enter confirm   Esc cancel")]
