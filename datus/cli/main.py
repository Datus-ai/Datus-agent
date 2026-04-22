#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Datus-CLI: An AI-powered SQL command-line interface for data engineers.
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
        self.parser = argparse.ArgumentParser(description="Datus: AI-powered SQL command-line interface")
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
            "--namespace",
            type=str,
            help="Datasource name to connect (use --datasource, --namespace is deprecated)",
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

        if not args.datasource:
            # Try to auto-select: default datasource or single datasource
            args.datasource = self._resolve_default_datasource(args)
            if not args.datasource:
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

    def _resolve_default_datasource(self, args) -> str:
        """Auto-select datasource when --datasource is not specified."""
        from rich.console import Console
        from rich.table import Table

        from datus.configuration.agent_config_loader import load_agent_config

        console = Console()
        try:
            config = load_agent_config(config=args.config or "", action="service", reload=True)
        except Exception:
            self.arg_parser.parser.print_help()
            return ""

        datasources = config.services.datasources
        if not datasources:
            console.print("[yellow]No datasources configured. Run 'datus configure' first.[/yellow]")
            return ""

        # default_datasource reflects the project-level overlay when present — it
        # is applied inside load_agent_config via _apply_project_override which
        # flips datasources[*].default before AgentConfig is built.
        default_db = config.services.default_datasource
        if default_db:
            return default_db

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
        """Trigger the first-run wizard if ``./.datus/config.yml`` is absent,
        or repair it when existing overrides no longer match the base
        ``agent.yml`` (e.g. a ``target`` that was renamed or removed).

        Idempotent: does nothing when the overlay file is valid. Loads the
        base ``agent.yml`` first so the wizard can constrain choices to
        models/datasources that actually exist; when the base config itself
        cannot be loaded, surface that error directly (the wizard has
        nothing to offer in that case).

        Repair is REPL-only: API / print / web paths keep raising on
        invalid overrides, because they have no broker to prompt the user.
        """
        from datus.cli.project_init import run_project_init
        from datus.configuration.agent_config_loader import load_agent_config
        from datus.configuration.project_config import project_config_path

        if not project_config_path().exists():
            try:
                base_config = load_agent_config(config=args.config or "", reload=True)
            except Exception as e:
                logger.error(f"Cannot run project setup wizard: base agent.yml failed to load: {e}")
                raise
            run_project_init(base_config)
            return

        self._repair_project_overrides(args)

    def _repair_project_overrides(self, args) -> None:
        """Re-prompt the user when ``./.datus/config.yml`` references a
        ``target`` or ``default_datasource`` that no longer exists in the base
        agent.yml, and persist the corrected values.

        Reads the raw agent.yml directly via ``configuration_manager`` to
        avoid tripping ``_apply_project_override`` (which is exactly what
        would raise on the stale value we're trying to fix).

        Requires an interactive TTY: ``select_choice`` silently falls back
        to its default when prompt_toolkit cannot run, which would
        otherwise persist an unintended choice. When stdin is not a TTY,
        raise instead of silently auto-writing.
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

        target_invalid = override.target is not None and override.target not in model_names
        db_invalid = override.default_datasource is not None and override.default_datasource not in db_names
        if not (target_invalid or db_invalid):
            return

        if not sys.stdin.isatty():
            stale = []
            if target_invalid:
                stale.append(f"target={override.target!r}")
            if db_invalid:
                stale.append(f"default_datasource={override.default_datasource!r}")
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message_args={
                    "config_error": (
                        f"Project config {project_config_path()} has stale values "
                        f"({', '.join(stale)}) and stdin is not a TTY; cannot prompt "
                        f"for replacements. Edit .datus/config.yml manually or rerun "
                        f"the CLI in an interactive terminal."
                    )
                },
            )

        console = Console()
        console.print()
        console.print(f"[yellow]Project config {project_config_path()} has stale values:[/]")

        if target_invalid:
            if not model_names:
                raise DatusException(
                    code=ErrorCode.COMMON_CONFIG_ERROR,
                    message_args={
                        "config_error": (
                            "Base agent.yml has no 'agent.models' defined; cannot repair "
                            f"target={override.target!r} in .datus/config.yml."
                        )
                    },
                )
            console.print(
                f"  [red]target[/] = {override.target!r} not found in agent.yml models "
                f"({sorted(model_names)}). Please pick a replacement:"
            )
            choices = {name: name for name in model_names}
            picked = select_choice(console, choices, default=model_names[0])
            override.target = picked or model_names[0]

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
            console.print(
                f"  [red]default_datasource[/] = {override.default_datasource!r} not found in agent.yml "
                f"services.datasources ({sorted(db_names)}). Please pick a replacement:"
            )
            db_types = (raw.get("services") or {}).get("datasources") or {}
            choices = {name: f"{name}  ({(db_types.get(name) or {}).get('type', 'unknown')})" for name in db_names}
            picked = select_choice(console, choices, default=db_names[0])
            override.default_datasource = picked or db_names[0]

        written = save_project_override(override)
        console.print(
            f"[green]Updated project config:[/] {written} "
            f"(target={override.target}, default_datasource={override.default_datasource})"
        )

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
