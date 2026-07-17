# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/plugins`` slash command — interactive plugin manager.

Opens :class:`datus.cli.plugin_app.PluginApp` (a single prompt_toolkit
Application) to configure global plugin profiles and this project's activation.
Mirrors ``/model``: in TUI mode the app is embedded via
:meth:`datus.cli.tui.app.DatusApp.run_wizard`; otherwise it falls back to a
transient ``Application`` via :meth:`PluginApp.run`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_info, print_warning
from datus.cli.plugin_app import PluginApp
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class PluginCommands:
    """Handler for the ``/plugins`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console
        self.agent_config = cli.agent_config

    def cmd_plugins(self, args: str) -> None:
        """Open the interactive plugin manager."""
        if not getattr(self.agent_config, "plugins_enabled", True):
            print_warning(self.console, "Plugins are disabled (`agent.plugins_enabled: false`).")
            return
        # Gate on list_plugins() (manifest-based) rather than entry points:
        # a disabled managed plugin's directory is off sys.path, so an
        # entry-point probe would misreport "no plugins installed" and make
        # the manager — the tool for re-enabling plugins — unreachable.
        try:
            from datus.cli.plugin_service import list_plugins

            if not list_plugins(self.agent_config):
                print_info(
                    self.console,
                    "No plugins installed. Install one with `datus plugin install <source>`.",
                )
                return
        except Exception as exc:  # noqa: BLE001 - discovery must not block the command
            logger.debug("plugin discovery failed for /plugins: %s", exc)

        app = PluginApp(self.agent_config, self.console)
        self._run_app(app)

    def _run_app(self, app: PluginApp) -> None:
        """Run ``app`` embedded in the active TUI, or standalone as fallback."""
        tui_app = getattr(self.cli, "tui_app", None)
        if tui_app is not None and getattr(tui_app, "_loop", None) is not None:
            return tui_app.run_wizard(app.build_embedded_panel)
        return app.run()


__all__ = ["PluginCommands"]
