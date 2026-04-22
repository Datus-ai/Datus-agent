# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/language`` slash command — set or show the response language.

Entry point
-----------

  - ``/language``                      — open the interactive picker.
  - ``/language zh``                   — set language, confirm scope interactively.
  - ``/language zh --project``         — persist to .datus/config.yml.
  - ``/language zh --global``          — persist to agent.yml.
  - ``/language --clear``              — remove project-level override.

The interactive path delegates to :class:`datus.cli.language_app.LanguageApp`,
a single prompt_toolkit Application that hosts language selection and scope
confirmation. In TUI mode we wrap the run in
:meth:`datus.cli.tui.app.DatusApp.suspend_input` so the outer persistent
Application releases ``stdin`` for the duration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from datus.cli.language_app import LANGUAGE_CHOICES, LanguageApp, LanguageSelection
from datus.configuration.project_config import ProjectOverride, load_project_override, save_project_override
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class LanguageCommands:
    """Handlers for the ``/language`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console
        self.agent_config = cli.agent_config

    def cmd_language(self, args: str) -> None:
        """Dispatch ``/language`` based on its argument shape."""
        tokens = args.strip().split()
        flag_global = "--global" in tokens
        flag_project = "--project" in tokens
        flag_clear = "--clear" in tokens
        code_tokens = [t for t in tokens if not t.startswith("--")]
        code = code_tokens[0] if code_tokens else ""

        global_lang = self.cli.configuration_manager.get("language") or ""
        project_override = load_project_override()
        project_lang = project_override.language if project_override else None

        if flag_clear:
            self._clear_language(project_override, global_lang, scope="project")
            return

        if not code and not flag_global and not flag_project:
            self._run_interactive(global_lang, project_lang, project_override)
            return

        if code == "auto":
            if flag_global:
                self._clear_language(project_override, global_lang, scope="global")
            elif flag_project:
                self._clear_language(project_override, global_lang, scope="project")
            else:
                selection = self._run_scope_picker("auto")
                if selection is not None:
                    self._clear_language(project_override, global_lang, scope=selection.scope)
            return

        if code not in LANGUAGE_CHOICES:
            self.console.print(f"[yellow]Warning:[/] '{code}' is not a well-known language code. Proceeding anyway.")

        if flag_global:
            self._save_global(code)
        elif flag_project:
            self._save_project(code, project_override)
        else:
            selection = self._run_scope_picker(code)
            if selection is None:
                return
            if selection.scope == "global":
                self._save_global(selection.code)
            else:
                self._save_project(selection.code, project_override)

    def _run_interactive(
        self,
        global_lang: str,
        project_lang: Optional[str],
        project_override: Optional[ProjectOverride],
    ) -> None:
        current = getattr(self.agent_config, "language", None) or ""
        source = "project" if project_lang else ("global" if global_lang else "not set")
        app = LanguageApp(
            console=self.console,
            current_language=current,
            current_source=source,
        )
        selection = self._run_app(app)
        if selection is None:
            return
        if selection.code == "auto":
            self._clear_language(project_override, global_lang, scope=selection.scope)
            return
        if selection.code == current:
            self.console.print(f"[dim]Language unchanged: {current}[/]")
            return
        if selection.scope == "global":
            self._save_global(selection.code)
        else:
            self._save_project(selection.code, project_override)

    def _run_scope_picker(self, code: str) -> Optional[LanguageSelection]:
        """Run a scope-only picker for direct ``/language <code>`` invocations."""
        app = LanguageApp(
            console=self.console,
            current_language=code,
            current_source="",
            scope_only=code,
        )
        return self._run_app(app)

    def _run_app(self, app: LanguageApp) -> Optional[LanguageSelection]:
        tui_app = getattr(self.cli, "tui_app", None)
        if tui_app is not None:
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    def _save_global(self, code: str) -> None:
        self.agent_config.language = code
        self.cli.configuration_manager.update_item("language", code)
        self.console.print(f"[bold green]Language set to: {code} (saved to agent.yml)[/]")

    def _save_project(self, code: str, project_override: Optional[ProjectOverride]) -> None:
        self.agent_config.language = code
        override = project_override or ProjectOverride()
        override.language = code
        path = save_project_override(override)
        self.console.print(f"[bold green]Language set to: {code} (saved to {path})[/]")

    def _clear_language(self, project_override: Optional[ProjectOverride], global_lang: str, scope: str) -> None:
        if scope == "global":
            self.cli.configuration_manager.update_item("language", None)
            self.agent_config.language = None
            self.console.print("[bold green]Global language cleared. Language is now unset (model decides).[/]")
        else:
            if project_override and project_override.language is not None:
                project_override.language = None
                save_project_override(project_override)
            self.agent_config.language = global_lang or None
            if global_lang:
                self.console.print(
                    f"[bold green]Project language override cleared. Falling back to global: {global_lang}[/]"
                )
            else:
                self.console.print(
                    "[bold green]Project language override cleared. Language is now unset (model decides).[/]"
                )
