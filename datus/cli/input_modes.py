# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Input-bar mode model shared by the REPL and the TUI shell.

The TUI input bar cycles through three modes with Tab (on an empty line):
chat (plain input goes to the model), SQL (plain input runs against the
current datasource), and bash (plain input runs through the permission-gated
bash tool). This module owns the mode enum plus the per-mode "chrome" (prompt
label, separator/hint styling) so :mod:`datus.cli.repl` and
:mod:`datus.cli.tui.app` can share it without importing each other.
"""

from enum import Enum
from typing import Dict

from pydantic import BaseModel, ConfigDict


class InputMode(str, Enum):
    """The three input-bar modes, cycled with Tab in the TUI."""

    CHAT = "chat"
    SQL = "sql"
    BASH = "bash"


_CYCLE = (InputMode.CHAT, InputMode.SQL, InputMode.BASH)


def next_input_mode(mode: InputMode) -> InputMode:
    """Return the mode following ``mode`` in the Tab cycle (chat → sql → bash → chat)."""
    idx = _CYCLE.index(InputMode(mode))
    return _CYCLE[(idx + 1) % len(_CYCLE)]


class ModeChrome(BaseModel):
    """Prompt label and style classes the TUI renders for one input mode.

    ``hint`` is the persistent one-line reminder pinned under the input while
    the mode is active; empty for chat (the window collapses to zero rows).
    """

    model_config = ConfigDict(frozen=True)

    prompt: str
    prompt_style: str
    separator_style: str
    hint: str
    hint_style: str


MODE_CHROME: Dict[InputMode, ModeChrome] = {
    InputMode.CHAT: ModeChrome(
        prompt="> ",
        prompt_style="class:input-prompt",
        separator_style="class:separator",
        hint="",
        hint_style="",
    ),
    InputMode.SQL: ModeChrome(
        prompt="sql> ",
        prompt_style="class:input-prompt.sql",
        separator_style="class:separator.sql",
        hint=" SQL mode · Enter run · \\+Enter newline · Tab next mode · Esc exit ",
        hint_style="class:sql-mode-hint",
    ),
    InputMode.BASH: ModeChrome(
        prompt="bash> ",
        prompt_style="class:input-prompt.bash",
        separator_style="class:separator.bash",
        hint=" BASH mode · Enter run · \\+Enter newline · Tab next mode · Esc exit ",
        hint_style="class:bash-mode-hint",
    ),
}
