# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""``/effort`` slash command — set or show the reasoning effort level.

Entry point
-----------

  - ``/effort``                      — open the interactive picker.
  - ``/effort high``                 — choose scope interactively (project/global).
  - ``/effort high --project``       — persist to .datus/config.yml.
  - ``/effort high --global``        — persist to agent.yml (top-level).
  - ``/effort off``                  — disable reasoning (shortcut for ``low/medium/high``=off).
  - ``/effort --clear``              — remove project-level override.
  - ``/effort status``               — print the effective level and its source.

Effort values accepted: ``off|minimal|low|medium|high``. The level is mapped
by LiteLLM to each provider's native dialect (OpenAI ``reasoning_effort``,
Anthropic ``thinking.budget_tokens``, Gemini ``thinking_config.thinking_budget``,
etc.), so a single knob covers every provider Datus supports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from datus.cli.cli_styles import print_error, print_info, print_success
from datus.cli.effort_app import EFFORT_CHOICES, EffortApp, EffortSelection
from datus.configuration.project_config import (
    REASONING_EFFORT_CHOICES,
    ProjectOverride,
    load_project_override,
    save_project_override,
)
from datus.utils.loggings import get_logger

if TYPE_CHECKING:
    from datus.cli.repl import DatusCLI

logger = get_logger(__name__)


class EffortCommands:
    """Handlers for the ``/effort`` slash command."""

    def __init__(self, cli: "DatusCLI"):
        self.cli = cli
        self.console = cli.console
        self.agent_config = cli.agent_config

    def cmd_effort(self, args: str) -> None:
        """Dispatch ``/effort`` based on its argument shape."""
        tokens = args.strip().split()
        flag_global = "--global" in tokens
        flag_project = "--project" in tokens
        flag_clear = "--clear" in tokens
        value_tokens = [t for t in tokens if not t.startswith("--")]
        value = value_tokens[0].strip().lower() if value_tokens else ""

        if value == "status":
            self._status()
            return

        if flag_clear:
            self._clear_project()
            return

        if not value and not flag_global and not flag_project:
            self._run_interactive()
            return

        if value and value not in REASONING_EFFORT_CHOICES:
            print_error(
                self.console,
                f"Invalid effort '{value}'. Expected one of {sorted(REASONING_EFFORT_CHOICES)}.",
            )
            return

        if flag_global:
            self._save_global(value)
        elif flag_project:
            self._save_project(value)
        else:
            selection = self._run_scope_picker(value)
            if selection is None:
                return
            if selection.scope == "global":
                self._save_global(selection.code)
            else:
                self._save_project(selection.code)

    def _run_interactive(self) -> None:
        current, source = self._current_effort()
        app = EffortApp(
            console=self.console,
            current_effort=current,
            current_source=source,
        )
        selection = self._run_app(app)
        if selection is None:
            return
        if selection.scope == "global":
            self._save_global(selection.code)
        else:
            self._save_project(selection.code)

    def _run_scope_picker(self, code: str) -> Optional[EffortSelection]:
        """Run a scope-only picker for direct ``/effort <level>`` invocations."""
        app = EffortApp(
            console=self.console,
            current_effort=code,
            current_source="",
            scope_only=code,
        )
        return self._run_app(app)

    def _run_app(self, app: EffortApp) -> Optional[EffortSelection]:
        tui_app = getattr(self.cli, "tui_app", None)
        if tui_app is not None:
            with tui_app.suspend_input():
                return app.run()
        return app.run()

    def _save_global(self, effort: str) -> None:
        """Write the effort level to the top-level ``agent.reasoning_effort``.

        Project-level ``./.datus/config.yml`` still wins at ``active_model()``
        time, so global scope behaves as a default across all projects.
        """
        self.cli.configuration_manager.update_item("reasoning_effort", effort)
        # Sync the in-memory override only if there is no project-level entry
        # already taking precedence; otherwise the project value would appear
        # to be shadowed mid-session even though it still wins on reload.
        override = load_project_override()
        project_effort = override.reasoning_effort if override else None
        if project_effort is None:
            self.agent_config.set_active_reasoning_effort(effort, persist=False)
        print_success(self.console, f"Reasoning effort set to '{effort}' (saved to agent.yml)")
        if project_effort is not None:
            print_info(
                self.console,
                f"Note: project-level override '{project_effort}' still takes precedence in this project.",
            )

    def _save_project(self, effort: str) -> None:
        override = load_project_override() or ProjectOverride()
        override.reasoning_effort = effort
        path = save_project_override(override)
        self.agent_config.set_active_reasoning_effort(effort, persist=False)
        print_success(self.console, f"Reasoning effort set to '{effort}' (saved to {path})")

    def _clear_project(self) -> None:
        override = load_project_override()
        if override and override.reasoning_effort is not None:
            override.reasoning_effort = None
            save_project_override(override)
        # Fall back to the global value from agent.yml, if any.
        global_value = self.cli.configuration_manager.get("reasoning_effort") or None
        self.agent_config.set_active_reasoning_effort(global_value, persist=False)
        if global_value:
            print_success(
                self.console,
                f"Project reasoning_effort cleared. Falling back to global: '{global_value}'.",
            )
        else:
            print_success(
                self.console,
                "Project reasoning_effort cleared. No global default; model-level settings apply.",
            )

    def _status(self) -> None:
        effective, source = self._current_effort()
        if effective:
            label = EFFORT_CHOICES.get(effective, effective)
            print_info(self.console, f"Reasoning effort: {effective} — {label} (source: {source})")
        else:
            print_info(self.console, "Reasoning effort: not set (model-level settings apply).")
        self._print_model_capability()

    def _print_model_capability(self) -> None:
        """Show whether the active model can actually consume a reasoning hint.

        Queries ``litellm.supports_reasoning`` for the active model so users
        immediately see when ``/effort`` will no-op against a non-reasoning
        model. Silent on any failure so ``/effort status`` never crashes on
        a half-configured setup.
        """
        try:
            model_config = self.agent_config.active_model()
            model_name = getattr(model_config, "model", "")
            provider = getattr(model_config, "type", "") or None
        except Exception:
            return
        if not model_name:
            return
        try:
            import litellm

            supports = bool(litellm.supports_reasoning(model=model_name, custom_llm_provider=provider))
        except Exception:
            return
        print_info(
            self.console,
            f"Active model '{model_name}' supports reasoning (per LiteLLM): {'yes' if supports else 'no'}",
        )

    def _current_effort(self) -> tuple[str, str]:
        """Return ``(effort, source)`` where source is project/global/model/off."""
        override = load_project_override()
        project_effort = override.reasoning_effort if override else None
        if project_effort:
            return project_effort, "project"
        global_effort = self.cli.configuration_manager.get("reasoning_effort") or ""
        if global_effort:
            return str(global_effort), "global"
        try:
            model_config = self.agent_config.active_model()
        except Exception:
            return "", "not set"
        if model_config.reasoning_effort:
            return model_config.reasoning_effort, "model"
        if model_config.enable_thinking:
            return "medium", "model (enable_thinking=true)"
        return "", "not set"
