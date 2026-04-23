#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Datus-CLI: Data engineering agent builds evolvable context for your data system.
Main entry point for the CLI application.
"""

import argparse

from datus import __version__
from datus.cli.repl import DatusCLI
from datus.utils.async_utils import setup_windows_policy
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import configure_logging, get_logger

logger = get_logger(__name__)


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Datus: Data engineering agent builds evolvable context for your data system"
        )
        self._setup_arguments()

    def _setup_arguments(self):
        # Add version argument
        self.parser.add_argument("-v", "--version", action="version", version=f"Datus CLI {__version__}")

        # Database connection settings
        self.parser.add_argument(
            "--db_type",
            dest="db_type",
            choices=[DBType.SQLITE, "snowflake", DBType.DUCKDB],
            default=DBType.SQLITE,
            help="Database type to connect to",
        )
        self.parser.add_argument(
            "--db_path", dest="db_path", type=str, help="Path to database file (for SQLite/DuckDB)"
        )

        # General settings
        self.parser.add_argument(
            "--history_file",
            dest="history_file",
            type=str,
            default=None,
            help="Path to history file (default: {agent.home}/history)",
        )
        self.parser.add_argument(
            "--config",
            dest="config",
            type=str,
            help="Path to configuration file (default: ./conf/agent.yml > {agent.home}/conf/agent.yml)",
        )
        self.parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        self.parser.add_argument("--no_color", dest="no_color", action="store_true", help="Disable colored output")
        # storage_path parameter deprecated - data path is now fixed at {agent.home}/data

        self.parser.add_argument(
            "--datasource",
            type=str,
            help="Datasource name to connect",
            default="",
        )

        # LLM trace settings
        self.parser.add_argument(
            "--save_llm_trace",
            action="store_true",
            help="Enable saving LLM input/output traces to YAML files",
        )

        # Filesystem strict mode: fail-closed for paths outside the project
        # root instead of prompting the broker. ``default=None`` preserves
        # ``agent.filesystem.strict`` in YAML when neither flag is passed.
        filesystem_strict_group = self.parser.add_mutually_exclusive_group()
        filesystem_strict_group.add_argument(
            "--filesystem-strict",
            dest="filesystem_strict",
            action="store_true",
            default=None,
            help="Reject filesystem reads/writes outside the project root at the "
            "tool layer (fail-closed; no interactive prompt). Overrides "
            "agent.filesystem.strict from YAML.",
        )
        filesystem_strict_group.add_argument(
            "--no-filesystem-strict",
            dest="filesystem_strict",
            action="store_false",
            help="Force-disable filesystem strict mode even if agent.filesystem.strict is true in YAML.",
        )

        # Execution mode: --web and --print are mutually exclusive
        mode_group = self.parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--web",
            action="store_true",
            help="Launch web-based chatbot interface",
        )
        mode_group.add_argument(
            "-p",
            "--print",
            dest="print_mode",
            type=str,
            default=None,
            help="Run a single prompt and stream MessagePayload JSON lines to stdout",
        )

        # Web interface settings
        self.parser.add_argument(
            "--port",
            type=int,
            default=8501,
            help="Port for web interface (default: 8501)",
        )

        self.parser.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Host for web interface (default: localhost)",
        )

        self.parser.add_argument(
            "--subagent",
            type=str,
            default="",
            help="Subagent name to open directly (for web and print modes)",
        )

        self.parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="Resume an existing session by session_id (for print mode)",
        )

        self.parser.add_argument(
            "--proxy_tools",
            dest="proxy_tools",
            type=str,
            default=None,
            help="Comma-separated tool patterns to proxy in print mode (e.g. 'filesystem_tools.*')",
        )

        self.parser.add_argument(
            "--session-scope",
            dest="session_scope",
            type=str,
            default=None,
            help="Session scope for directory isolation (sessions stored under {session_dir}/{scope}/)",
        )

        self.parser.add_argument(
            "--chatbot-dist",
            dest="chatbot_dist",
            type=str,
            default=None,
            help="Path to @datus/web-chatbot dist directory (for --web mode)",
        )

        self.parser.add_argument(
            "--stream",
            dest="stream_thinking",
            action="store_true",
            default=False,
            help="Enable streaming thinking deltas in print mode (token-by-token output)",
        )

    def parse_args(self):
        return self.parser.parse_args()


class Application:
    def __init__(self):
        self.arg_parser = ArgumentParser()

    def run(self):
        args = self.arg_parser.parse_args()

        configure_logging(args.debug, console_output=False)

        # REPL-only: ensure ./.datus/config.yml exists before anything touches
        # agent config. Must run before _resolve_default_datasource so the
        # project-level default_datasource can win over the base agent.yml's.
        if args.print_mode is None and not args.web:
            self._ensure_project_config(args)

        is_repl = args.print_mode is None and not args.web
        if not args.datasource:
            # Try to auto-select: default datasource or single datasource
            args.datasource = self._resolve_default_datasource(args, allow_empty=is_repl)
            if not args.datasource and not is_repl:
                return

        if args.resume and args.print_mode is None:
            self.arg_parser.parser.error("--resume requires --print mode")

        if args.proxy_tools and args.print_mode is None:
            self.arg_parser.parser.error("--proxy_tools requires --print mode")

        if args.print_mode is not None:
            from datus.cli.print_mode import PrintModeRunner

            PrintModeRunner(args).run()
        elif args.web:
            self._run_web_interface(args)
        else:
            cli = DatusCLI(args)
            cli.run()

    def _resolve_default_datasource(self, args, allow_empty: bool = False) -> str:
        """Auto-select datasource when --datasource is not specified."""
        from rich.console import Console
        from rich.table import Table

        from datus.configuration.agent_config_loader import load_agent_config

        console = Console()
        try:
            config = load_agent_config(config=args.config or "", action="service", reload=True, create_if_missing=True)
        except Exception:
            if not allow_empty:
                self.arg_parser.parser.print_help()
            return ""

        datasources = config.services.datasources
        if not datasources:
            return ""

        # default_datasource reflects the project-level overlay when present — it
        # is applied inside load_agent_config via _apply_project_override which
        # flips datasources[*].default before AgentConfig is built.
        default_db = config.services.default_datasource
        if default_db:
            return default_db

        if allow_empty:
            return ""

        # Multiple datasources, no default — show list and ask user to specify
        console.print("[yellow]Multiple datasources configured. Please specify --datasource <name>[/yellow]\n")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        for name, cfg in datasources.items():
            table.add_row(name, cfg.type)
        console.print(table)
        return ""

    def _ensure_project_config(self, args) -> None:
        """Ensure ``./.datus/config.yml`` exists; create a minimal one when absent.

        Unlike the previous first-run wizard, this no longer prompts the user
        to pick a model — the CLI starts with no active model and the user
        can configure one later via ``/model``.  Only ``default_datasource``
        is auto-selected (first DB from agent.yml) so that the REPL can
        function immediately for database browsing.

        When the file exists but references stale values, repair silently
        (clear the model target instead of prompting).
        """
        from datus.configuration.agent_config_loader import load_agent_config
        from datus.configuration.project_config import (
            ProjectOverride,
            project_config_path,
            save_project_override,
        )

        if not project_config_path().exists():
            try:
                base_config = load_agent_config(config=args.config or "", reload=True, create_if_missing=True)
            except Exception as e:
                logger.error(f"Cannot create project config: base agent.yml failed to load: {e}")
                raise
            default_datasource = base_config.services.default_datasource
            if not default_datasource and base_config.services.datasources:
                default_datasource = next(iter(base_config.services.datasources))
            override = ProjectOverride(default_datasource=default_datasource)
            written = save_project_override(override)
            from rich.console import Console

            console = Console()
            console.print(f"[green]Created project config:[/] {written}")
            console.print("[dim]No model configured yet — use /model inside the CLI to set one.[/]")
            return

        self._repair_project_overrides(args)

    def _repair_project_overrides(self, args) -> None:
        """Clear stale model target silently; re-prompt only for datasource.

        Model target validation is deferred to runtime — if it is stale we
        simply clear it so the CLI starts with no active model. The user
        can pick one later via ``/model``.

        For ``default_datasource`` we still prompt interactively because the
        REPL needs a valid DB connection to be useful.
        """
        import sys

        from rich.console import Console

        from datus.cli._cli_utils import select_choice
        from datus.configuration.agent_config_loader import configuration_manager
        from datus.configuration.project_config import (
            load_project_override,
            project_config_path,
            save_project_override,
        )

        override = load_project_override()
        if override is None or override.is_empty():
            return

        try:
            raw = dict(configuration_manager(config_path=args.config or "", reload=True).data)
        except Exception as e:
            logger.error(f"Cannot validate project overrides: base agent.yml failed to load: {e}")
            raise

        model_names = list((raw.get("models") or {}).keys())
        db_names = list(((raw.get("services") or {}).get("datasources") or {}).keys())

        target_invalid, stale_desc = self._classify_target(override.target, raw, model_names)
        db_invalid = override.default_datasource is not None and override.default_datasource not in db_names
        if not (target_invalid or db_invalid):
            return

        console = Console()
        changed = False

        if target_invalid:
            override.target = None
            console.print(
                f"[yellow]Cleared stale model target ({stale_desc}) from {project_config_path()}. "
                f"Use /model to configure a new one.[/]"
            )
            changed = True

        if db_invalid:
            if not db_names:
                raise DatusException(
                    code=ErrorCode.COMMON_CONFIG_ERROR,
                    message_args={
                        "config_error": (
                            "Base agent.yml has no 'agent.services.datasources' defined; cannot repair "
                            f"default_datasource={override.default_datasource!r} in .datus/config.yml."
                        )
                    },
                )
            if not sys.stdin.isatty():
                raise DatusException(
                    code=ErrorCode.COMMON_CONFIG_ERROR,
                    message_args={
                        "config_error": (
                            f"Project config {project_config_path()} has stale "
                            f"default_datasource={override.default_datasource!r} and stdin is not a TTY. "
                            f"Edit .datus/config.yml manually or rerun in an interactive terminal."
                        )
                    },
                )
            console.print(
                f"[yellow]default_datasource[/] = {override.default_datasource!r} not found in agent.yml "
                f"services.datasources ({sorted(db_names)}). Please pick a replacement:"
            )
            db_types = (raw.get("services") or {}).get("datasources") or {}
            choices = {name: f"{name}  ({(db_types.get(name) or {}).get('type', 'unknown')})" for name in db_names}
            picked = select_choice(console, choices, default=db_names[0])
            override.default_datasource = picked or db_names[0]
            changed = True

        if changed:
            save_project_override(override)

    @staticmethod
    def _classify_target(target, raw, model_names):
        """Return ``(invalid, description)`` for a project-level target.

        Each target shape is validated against the right source:
          - legacy string / ``ProjectTarget(custom=...)`` → ``agent.models``.
          - ``ProjectTarget(provider=..., model=...)`` → credentials must be
            resolvable for that provider via ``agent.providers`` or the
            catalog's ``api_key_env``; otherwise we flag it stale so the
            caller can fall back to a custom model.

        ``description`` is a human-friendly string embedded in the prompt
        and the non-TTY error message.
        """
        from datus.configuration.project_config import ProjectTarget

        if target is None:
            return False, ""
        if isinstance(target, ProjectTarget):
            if target.custom:
                return target.custom not in model_names, f"custom={target.custom!r}"
            if target.provider and target.model:
                provider = target.provider
                desc = f"provider={provider!r} model={target.model!r}"
                if not Application._provider_has_credentials(provider, raw):
                    return True, desc
                return False, desc
            return True, repr(target)
        return target not in model_names, f"target={target!r}"

    @staticmethod
    def _provider_has_credentials(provider: str, raw: dict) -> bool:
        """Lightweight credential check that mirrors
        :meth:`AgentConfig.provider_available` without instantiating the
        full config (which would re-run override validation and defeat the
        whole repair flow).
        """
        import os

        from datus.configuration.agent_config import _load_provider_catalog, resolve_env

        providers_raw = raw.get("providers") or {}
        user_entry = providers_raw.get(provider) if isinstance(providers_raw, dict) else None
        if not isinstance(user_entry, dict):
            user_entry = {}

        try:
            catalog = _load_provider_catalog()
        except Exception:
            catalog = {}
        meta = {}
        if isinstance(catalog, dict):
            providers_meta = catalog.get("providers") or {}
            if isinstance(providers_meta, dict):
                meta = providers_meta.get(provider) or {}
                if not isinstance(meta, dict):
                    meta = {}

        auth_type = meta.get("auth_type") or user_entry.get("auth_type") or "api_key"
        if auth_type == "subscription":
            try:
                from datus.auth.claude_credential import get_claude_subscription_token

                token, _ = get_claude_subscription_token(api_key_from_config=user_entry.get("api_key") or "")
                return bool(token)
            except Exception:
                return False
        if auth_type == "oauth":
            try:
                from datus.auth.oauth_manager import OAuthManager

                return OAuthManager().is_authenticated()
            except Exception:
                return False

        api_key = user_entry.get("api_key")
        if api_key and resolve_env(str(api_key)).strip() and not resolve_env(str(api_key)).startswith("<MISSING:"):
            return True
        env_name = meta.get("api_key_env")
        if env_name and os.getenv(str(env_name), "").strip():
            return True
        return False

    def _run_web_interface(self, args):
        """Launch web chatbot interface"""
        from datus.cli.web import run_web_interface

        run_web_interface(args)


def main():
    """Entry point for console scripts"""
    import sys

    # Intercept 'skill' subcommand and delegate to datus.main's skill handler
    if len(sys.argv) > 1 and sys.argv[1] == "skill":
        from datus.main import create_parser as create_main_parser

        parser = create_main_parser()
        args = parser.parse_args()
        configure_logging(getattr(args, "debug", False), console_output=False)
        from datus.cli.skill_cli import run_skill_command

        sys.exit(run_skill_command(args))

    app = Application()
    app.run()


if __name__ == "__main__":
    setup_windows_policy()
    main()
