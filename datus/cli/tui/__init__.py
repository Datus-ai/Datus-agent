# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Persistent prompt_toolkit TUI shell for the Datus REPL.

The TUI keeps the status bar and input TextArea pinned to the bottom of the
terminal across the entire REPL lifetime, including while agent loops run in
a worker thread. This module is only loaded when ``sys.stdin``/``sys.stdout``
are TTYs and ``DATUS_TUI`` is not explicitly disabled; non-TTY paths (CI,
pipes, ``--print`` mode) continue to use the classic ``PromptSession``.
"""

from datus.cli.tui.app import DatusApp, tui_enabled

__all__ = ["DatusApp", "tui_enabled"]
