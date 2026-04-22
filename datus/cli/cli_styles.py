# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Centralised CLI style constants and output helpers.

Every colour, symbol, and message format used by slash-command output
lives here.  Changing a constant in this module propagates to every
command file, ``_render_utils``, and ``_cli_utils`` — one edit, global
effect.

Usage in command modules::

    from datus.cli.cli_styles import print_error, print_success, TABLE_HEADER_STYLE

    print_error(self.console, "Database name is required")
    print_success(self.console, f"Switched to {name}")
"""

from __future__ import annotations

from rich.console import Console

# ── Symbols ──────────────────────────────────────────────────
SYM_CHECK = "\u2713"  # ✓
SYM_CROSS = "\u2717"  # ✗
SYM_ARROW = "\u2192"  # →
SYM_BULLET = "\u2022"  # •

# ── Semantic colour tokens (Rich markup values) ─────────────
CLR_ERROR = "red"
CLR_SUCCESS = "green"
CLR_WARNING = "yellow"
CLR_INFO = "dim"
CLR_USAGE = "cyan"

# prompt_toolkit ANSI names for interactive selectors
CLR_CURSOR = "ansicyan"
CLR_CURRENT = "ansigreen"

# ── Table / code ────────────────────────────────────────────
TABLE_HEADER_STYLE = "green"
CODE_THEME = "monokai"


# ── Helper functions ────────────────────────────────────────


def print_error(console: Console, message: str, *, prefix: bool = True) -> None:
    if prefix:
        console.print(f"[{CLR_ERROR}]Error:[/] {message}")
    else:
        console.print(f"[{CLR_ERROR}]{message}[/]")


def print_success(console: Console, message: str, *, symbol: bool = False) -> None:
    if symbol:
        console.print(f"[{CLR_SUCCESS}]{SYM_CHECK} {message}[/]")
    else:
        console.print(f"[{CLR_SUCCESS}]{message}[/]")


def print_warning(console: Console, message: str) -> None:
    console.print(f"[{CLR_WARNING}]{message}[/]")


def print_info(console: Console, message: str) -> None:
    console.print(f"[{CLR_INFO}]{message}[/]")


def print_status(console: Console, message: str, *, ok: bool) -> None:
    if ok:
        console.print(f"[{CLR_SUCCESS}]{SYM_CHECK} {message}[/]")
    else:
        console.print(f"[{CLR_ERROR}]{SYM_CROSS} {message}[/]")


def print_usage(console: Console, syntax: str) -> None:
    from rich.text import Text

    label = Text("Usage: ", style=CLR_USAGE)
    label.append(syntax)
    console.print(label)


def print_empty_set(console: Console, message: str = "Empty set.") -> None:
    console.print(f"[{CLR_WARNING}]{message}[/]")
