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

from typing import List, Tuple

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

# ── Paste collapse ─────────────────────────────────────────
PASTE_COLLAPSE_THRESHOLD = 10

# ── Table / code ────────────────────────────────────────────
TABLE_HEADER_STYLE = "green"
TABLE_BORDER_STYLE = "blue"
HEADER_BOLD_CYAN = "bold cyan"
HEADER_BOLD_GREEN = "bold green"
CODE_THEME = "monokai"

# ── Hex palette (kept raw; swap centrally to adapt themes) ──
STATUS_BAR_FG_HINT = "#9a9aaa"  # dim/meta text in status bar and hints
STATUS_BAR_BRAND = "#ffd866"  # brand badge
STATUS_BAR_RUNNING = "#ffb86c"  # running indicator / spinner dot
STATUS_BAR_SEP = "#444444"  # horizontal separator
LIVE_SECONDARY = "#6e6e6e"  # subagent-live / processing-live lines
DIALOG_BG = "#444444"
DIALOG_FG = "#ffffff"
DIALOG_SHADOW = "#000000"
DIALOG_STATUS_BG = "#000044"

# ── SQL tag palette (consumed by subject_rich_utils) ────────
SQL_TAG_COLORS: list[str] = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

# ── Autocomplete Pygments token colours ─────────────────────
# Token keys are resolved in ``datus.cli.autocomplete`` so this
# module stays free of Pygments imports.
AUTOCOMPLETE_TOKEN_COLORS: dict[str, str] = {
    "at_tables": "#00CED1",  # Cyan
    "at_metrics": "#FFD700",  # Gold
    "at_reference_sql": "#32CD32",  # Green
    "at_files": "ansiblue",
}

# ── ActionRole colour map, keyed by ActionRole.name ─────────
# Callers (``renderers.py``) materialise the enum-keyed dict.
ACTION_ROLE_COLOR_NAMES: dict[str, str] = {
    "SYSTEM": "bright_magenta",
    "ASSISTANT": "bright_blue",
    "USER": "bright_green",
    "TOOL": "bright_cyan",
    "WORKFLOW": "bright_yellow",
    "INTERACTION": "bright_yellow",
}

# ── prompt_toolkit Style dicts ──────────────────────────────
# Main REPL / TUI status bar + completion menu + pinned rolling window.
STATUS_BAR_STYLE: dict[str, str] = {
    "prompt": "ansigreen bold",
    "input-prompt": "ansigreen bold",
    "input-prompt.busy": "ansibrightblack",
    "input-prompt.hint": f"italic {STATUS_BAR_FG_HINT}",
    "input-area": "",
    "status-bar": STATUS_BAR_FG_HINT,
    "status-bar.brand": f"{STATUS_BAR_BRAND} bold",
    "status-bar.plan": STATUS_BAR_FG_HINT,
    "status-bar.profile": STATUS_BAR_FG_HINT,
    "status-bar.profile.auto": "ansicyan",
    "status-bar.profile.dangerous": "ansired",
    "status-bar.sep": STATUS_BAR_FG_HINT,
    "status-bar.agent": STATUS_BAR_FG_HINT,
    "status-bar.connector": STATUS_BAR_FG_HINT,
    "status-bar.model": STATUS_BAR_FG_HINT,
    "status-bar.tokens": STATUS_BAR_FG_HINT,
    "status-bar.ctx": STATUS_BAR_FG_HINT,
    "status-bar.running": f"{STATUS_BAR_RUNNING} bold",
    "status-bar.dot": f"{STATUS_BAR_RUNNING} bold",
    "separator": STATUS_BAR_SEP,
    # Slash-command autocomplete popup. ``bg:default`` blends into the
    # terminal palette; ``noreverse`` strips prompt_toolkit's default
    # reverse-video highlight so the selection is conveyed by bright
    # cyan text alone.
    "completion-menu": "bg:default",
    "completion-menu.completion": "bg:default fg:default",
    "completion-menu.completion.current": "noreverse bg:default fg:ansibrightcyan",
    "completion-menu.meta.completion": "bg:default fg:ansibrightblack",
    "completion-menu.meta.completion.current": "noreverse bg:default fg:ansibrightcyan",
    "hint": f"{STATUS_BAR_FG_HINT} italic",
    # Pinned subagent/tool rolling-window lines match scrollback [dim].
    "subagent-live": LIVE_SECONDARY,
    "processing-live": LIVE_SECONDARY,
    # Top-level running tool: default foreground (same colour as a completed
    # main-agent tool header) — keeps the PROCESSING row visually in-line
    # with the row that eventually replaces it.
    "processing-live-top": "",
    # Pinned subagent header: plain cyan prefix, default colour goal.
    "subagent-header-live": "ansicyan",
    "subagent-header-goal-live": "",
}

# Sub-agent wizard modal dialog (prompt_toolkit full-screen Application).
SUB_AGENT_WIZARD_STYLE: dict[str, str] = {
    "status-bar": f"bg:{DIALOG_STATUS_BG} {DIALOG_FG}",
    "input-window": "fg:ansigreen",
    "textarea": "fg:ansigreen",
    "label": "fg:ansicyan",
    "tip": "fg:ansiyellow bold",
    "separator": "fg:ansigray",
    "dialog": f"bg:{DIALOG_BG}",
    "dialog frame.label": f"fg:{DIALOG_FG} bg:{DIALOG_SHADOW}",
    "dialog.body": f"bg:{DIALOG_BG} fg:{DIALOG_FG}",
    "dialog shadow": f"bg:{DIALOG_SHADOW}",
    "rule": "",
    "rule.selected": "bg:ansiblue fg:ansiwhite",
    "rule.editing": "bg:ansigreen fg:ansiwhite",
}

# Minimal style used by ad-hoc ``prompt()`` calls in ``_cli_utils``.
PROMPT_ONLY_STYLE: dict[str, str] = {
    "prompt": "ansigreen bold",
}


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


def render_tui_title_bar(title: str) -> List[Tuple[str, str]]:
    dash = "\u2500"
    return [("", dash * 4 + " "), ("bold", title), ("", " " + dash * 200)]
