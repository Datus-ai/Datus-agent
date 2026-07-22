# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/sandbox`` slash command — toggle the OS-level bash sandbox.

Entry points
------------

  - ``/sandbox`` / ``/sandbox status`` — show state, mode, the detected
    mechanism (sandbox-exec / bwrap) and the extra allowlists.
  - ``/sandbox on|off``            — flip for this session only (immediate:
    every BashTool shares the same ``SandboxSettings`` object).
  - ``/sandbox strict|normal``     — enable AND pin the mode. ``strict`` is
    the multi-tenant tier: workspace + tmp + explicit allowlists only
    (``~/.datus`` fully blocked) and a minimal child environment.
  - ``... --project``              — also persist to ``./.datus/config.yml``.
  - ``... --global``               — also persist to ``agent.yml``.
  - ``/sandbox --clear``           — remove the project-level override.

When enabled, bash commands can only write inside the workspace, the session
data dir (normal mode) and tmp, and only read system dirs plus the same
allowlist (``agent.bash.sandbox.allow_read`` / ``allow_write`` extend it).
Fail-closed: if no OS mechanism is available, commands are rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datus.cli.cli_styles import print_error, print_info, print_success, print_warning
from datus.configuration.project_config import ProjectOverride, load_project_override, save_project_override
from datus.tools.func_tool import bash_sandbox
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)

_USAGE = "/sandbox [status|on|off|strict|normal] [--project|--global] | /sandbox --clear"


class SandboxCommands:
    """Handlers for the ``/sandbox`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console
        self.agent_config = cli.agent_config

    def cmd_sandbox(self, args: str) -> None:
        tokens = args.strip().split()
        flag_global = "--global" in tokens
        flag_project = "--project" in tokens
        flag_clear = "--clear" in tokens
        value_tokens = [t for t in tokens if not t.startswith("--")]
        value = value_tokens[0].strip().lower() if value_tokens else ""

        if flag_clear:
            self._clear_project()
            return
        if value in ("", "status"):
            self._status()
            return
        if value not in ("on", "off", "strict", "normal"):
            print_error(self.console, f"Invalid argument '{value}'. Usage: {_USAGE}")
            return

        # Immediate session effect: BashTool holds the same SandboxSettings
        # object and re-reads it on every call. ``on``/``off`` toggle the
        # switch (mode untouched); ``strict``/``normal`` enable and pin mode.
        settings = self.agent_config.bash_sandbox
        if value == "off":
            settings.enabled = False
        else:
            settings.enabled = True
            if value in (bash_sandbox.MODE_STRICT, bash_sandbox.MODE_NORMAL):
                settings.mode = value
        if settings.enabled and not bash_sandbox.is_available():
            print_warning(
                self.console,
                f"No OS sandbox mechanism is available ({bash_sandbox.unavailable_reason()}). "
                "Bash commands will be REJECTED until it is installed or the sandbox is turned off.",
            )

        if flag_project:
            self._save_project(value)
        elif flag_global:
            self._save_global()
        else:
            print_success(self.console, f"Bash sandbox {value} (this session only)")

    def _save_project(self, value: str) -> None:
        override = load_project_override() or ProjectOverride()
        # ``on``/``off`` persist as booleans (mode follows global config);
        # mode names persist as strings (enable + pin, see project_config).
        override.sandbox = {"on": True, "off": False}.get(value, value)
        path = save_project_override(override)
        print_success(self.console, f"Bash sandbox {value} (saved to {path})")

    def _save_global(self) -> None:
        """Persist the whole ``bash.sandbox`` section to agent.yml.

        Serialized from the live settings so ``mode``, ``deny_network`` and
        ``allow_read``/``allow_write`` survive the shallow ``update_item``
        merge (the ``sandbox`` sub-dict is replaced as a unit; sibling
        ``bash.*`` keys are preserved).
        """
        settings = self.agent_config.bash_sandbox
        sandbox_yaml = {"enabled": settings.enabled, "mode": settings.mode}
        if settings.allow_read:
            sandbox_yaml["allow_read"] = list(settings.allow_read)
        if settings.allow_write:
            sandbox_yaml["allow_write"] = list(settings.allow_write)
        if settings.deny_network:
            sandbox_yaml["deny_network"] = True
        self.cli.configuration_manager.update_item("bash", {"sandbox": sandbox_yaml})
        state = f"{'on' if settings.enabled else 'off'} ({settings.mode})"
        print_success(self.console, f"Bash sandbox {state} (saved to agent.yml)")
        override = load_project_override()
        if override is not None and override.sandbox is not None:
            print_info(
                self.console,
                f"Note: project-level override 'sandbox: {str(override.sandbox).lower()}' "
                "still takes precedence in this project on reload.",
            )

    def _clear_project(self) -> None:
        override = load_project_override()
        if override and override.sandbox is not None:
            override.sandbox = None
            save_project_override(override)
        print_success(self.console, "Project sandbox override cleared. agent.yml (or the default: off) applies.")

    def _status(self) -> None:
        settings = self.agent_config.bash_sandbox
        mechanism = bash_sandbox.detect_mechanism()
        print_info(self.console, f"Bash sandbox: {'on' if settings.enabled else 'off'} (mode: {settings.mode})")
        print_info(self.console, f"Mechanism: {mechanism or 'unavailable'}")
        if settings.deny_network:
            print_info(self.console, "Network: denied inside the sandbox")
        if settings.enabled and mechanism is None:
            print_warning(
                self.console,
                f"Fail-closed: bash commands are rejected ({bash_sandbox.unavailable_reason()}).",
            )
        override = load_project_override()
        if override is not None and override.sandbox is not None:
            print_info(self.console, f"Source: project (.datus/config.yml: sandbox: {str(override.sandbox).lower()})")
        if not settings.enabled:
            return
        if settings.is_strict:
            print_info(
                self.console,
                "Writable: workspace, tmp"
                + (f", extra: {', '.join(settings.allow_write)}" if settings.allow_write else ""),
            )
            print_info(
                self.console,
                "Readable: system dirs, python env + all writable (datus home BLOCKED)"
                + (f", extra: {', '.join(settings.allow_read)}" if settings.allow_read else ""),
            )
            print_info(self.console, "Environment: minimal allowlist (process secrets hidden)")
        else:
            print_info(
                self.console,
                "Writable: workspace, session data dir, tmp"
                + (f", extra: {', '.join(settings.allow_write)}" if settings.allow_write else ""),
            )
            print_info(
                self.console,
                "Readable: system dirs, datus home, python env + all writable"
                + (f", extra: {', '.join(settings.allow_read)}" if settings.allow_read else ""),
            )
