# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/init`` slash command — generate ``AGENTS.md`` for the current project.

The handler is a thin wrapper around :class:`datus.cli.init_workspace.InitWorkspace`.
``InitWorkspace`` already does all the heavy lifting (directory scan, project
type detection, optional datasource probe, LLM-driven content generation,
template fallback). This module:

  - takes no arguments — the datasource is whichever the REPL is currently
    pinned to (``agent_config.current_datasource`` or the launch
    ``--datasource``); empty string means skip the schema probe.
  - builds the ``argparse``-shaped namespace ``InitWorkspace`` expects.
  - releases ``stdin`` from the outer TUI while ``InitWorkspace`` runs, so
    the embedded ``rich.Prompt.ask`` (overwrite/cancel) and the LLM streaming
    output don't fight the persistent prompt_toolkit Application.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error
from datus.cli.init_workspace import InitWorkspace
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class InitCommands:
    """Handler for the ``/init`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console

    def cmd_init(self, args: str) -> None:
        """Dispatch ``/init`` — no arguments; uses the current datasource."""
        if (args or "").strip():
            print_error(
                self.console,
                "/init takes no arguments; switch datasource via /datasource first.",
                prefix=False,
            )
            return

        cli_args = getattr(self.cli, "args", None)
        config = getattr(cli_args, "config", "") or ""
        debug = bool(getattr(cli_args, "debug", False))
        datasource = (
            getattr(self.cli.agent_config, "current_datasource", "") or getattr(cli_args, "datasource", "") or ""
        )
        ns = SimpleNamespace(config=config, debug=debug, datasource=datasource)

        tui_app = getattr(self.cli, "tui_app", None)
        ctx = tui_app.suspend_input() if tui_app is not None else nullcontext()
        with ctx:
            InitWorkspace(ns).run()
