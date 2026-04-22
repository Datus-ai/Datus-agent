#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Interactive configuration manager for Datus Agent database connections.

Incrementally manages database connections in ``~/.datus/conf/agent.yml``
under the ``agent.services.datasources`` section. Shows current state,
supports add/delete of individual database entries, and installs
missing database adapter plugins on demand.

LLM models are now managed exclusively by the ``/model`` slash command
(see ``datus.cli.model_commands``). The init wizard also no longer
writes into ``agent.models``; custom / self-hosted models are left to
hand-edit in ``agent.yml``.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datus.cli._cli_utils import select_choice
from datus.cli.init_util import detect_db_connectivity
from datus.utils.loggings import get_logger, print_rich_exception
from datus.utils.path_manager import get_path_manager
from datus.utils.resource_utils import copy_data_file

logger = get_logger(__name__)

_BACK = "__back__"


def _prompt_with_back(label: str, default: str = "", password: bool = False) -> str:
    """Prompt with ESC to go back. Uses prompt_toolkit for key handling.

    Returns _BACK if ESC pressed, otherwise the entered value.
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("escape")
        def _esc(event):
            event.app.exit(result=_BACK)

        session = PromptSession(key_bindings=kb)
        suffix = f" ({default})" if default else ""
        result = session.prompt(f"{label}{suffix}: ", is_password=password)
        if result == _BACK:
            return _BACK
        return result.strip() if result.strip() else default

    except (KeyboardInterrupt, EOFError):
        return _BACK


class InteractiveConfigure:
    """Incremental configuration manager for database connections."""

    def __init__(self, user_home: Optional[str] = None):
        self.user_home = user_home if user_home else Path.home()
        self.console = Console(log_path=False)

        path_manager = get_path_manager()
        self.conf_dir = path_manager.conf_dir
        self.template_dir = path_manager.template_dir
        self.sample_dir = path_manager.sample_dir
        self.config_path = self.conf_dir / "agent.yml"

        # Working state — datasources only. LLM model management lives in the
        # ``/model`` slash command now; this wizard intentionally leaves
        # ``agent.providers`` / ``agent.models`` alone.
        self.datasources: Dict[str, dict] = {}

    # ── Setup helpers ──────────────────────────────────────────────

    def _init_dirs(self):
        # Bootstrap: no project_name yet, so skip project-scoped dirs
        # (``sessions/{project_name}/`` is created lazily at session runtime).
        get_path_manager().ensure_dirs("conf", "data", "logs", "template", "sample")

    def _copy_files(self):
        try:
            copy_data_file(resource_path="prompts", dest_dir=self.template_dir, overwrite=True)
        except Exception as e:
            logger.debug(f"Error copying template files: {e}")
        try:
            copy_data_file(resource_path="sample_data", dest_dir=self.sample_dir, overwrite=False)
        except Exception as e:
            logger.debug(f"Error copying sample files: {e}")
        # Deploy built-in skills (init, etc.) to ~/.datus/skills/
        try:
            skills_dir = Path(self.user_home) / ".datus" / "skills"
            copy_data_file(resource_path="resources/skills", dest_dir=skills_dir, overwrite=False)
        except Exception as e:
            logger.debug(f"Error deploying built-in skills: {e}")

    def _load_existing_config(self):
        """Load datasources from existing agent.yml.

        Supports the new ``agent.services.datasources`` layout, the legacy
        singular ``agent.service.datasources`` key, and the further-legacy
        ``agent.namespace`` block. ``agent.models`` / ``agent.providers``
        are ignored here — they are not this wizard's responsibility.
        """
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            return

        agent = raw.get("agent", {})

        services = agent.get("services") or {}
        legacy_service = agent.get("service") or {}
        if isinstance(services, dict) and services.get("datasources") is not None:
            self.datasources = services.get("datasources", {})
        elif isinstance(legacy_service, dict) and legacy_service.get("datasources") is not None:
            self.datasources = legacy_service.get("datasources", {})
        elif "namespace" in agent:
            # Auto-migrate legacy namespace format
            from datus.configuration.agent_config import ServicesConfig

            migrated = ServicesConfig.migrate_from_namespace(agent["namespace"])
            self.datasources = migrated.get("datasources", {})

    # ── Display ────────────────────────────────────────────────────

    def _show_current_state(self):
        """Display current datasources.

        LLM configuration is deliberately omitted — users run ``/model``
        inside the CLI to see and change it.
        """
        if self.datasources:
            table = Table(title="Current Datasources", show_header=True, header_style="bold green")
            table.add_column("Name", style="cyan")
            table.add_column("Type")
            table.add_column("Connection")
            table.add_column("Default")
            for name, cfg in self.datasources.items():
                conn = cfg.get("uri", "") or cfg.get("host", "") or cfg.get("account", "")
                is_default = "*" if cfg.get("default") else ""
                table.add_row(name, cfg.get("type", ""), str(conn), is_default)
            self.console.print(table)
        else:
            self.console.print("[yellow]No datasources configured.[/yellow]")

        self.console.print()

    # ── Main flow ──────────────────────────────────────────────────

    def run(self) -> int:
        # Suppress console logging. pytest's live-log plugin attaches a
        # handler whose ``stream`` lacks a ``.name`` attribute; guard the
        # introspection so the wizard keeps working under the test harness.
        root_logger = logging.getLogger()
        original_handler_levels = {}
        for handler in root_logger.handlers:
            stream_name = getattr(getattr(handler, "stream", None), "name", "")
            if stream_name in ("<stdout>", "<stderr>"):
                original_handler_levels[handler] = handler.level
                handler.setLevel(logging.CRITICAL + 1)

        try:
            self._init_dirs()
            self._copy_files()
            self._load_existing_config()

            self.console.print("\n[bold cyan]Datus Configure[/bold cyan]")
            self.console.print("Manage your database connections. Use `/model` inside the CLI to pick an LLM.\n")

            if not self.datasources:
                return self._first_time_setup()
            return self._interactive_menu()

        except KeyboardInterrupt:
            self.console.print("\nConfiguration cancelled by user")
            return 1
        except Exception as e:
            print_rich_exception(self.console, e, "Configuration failed", logger)
            return 1
        finally:
            for handler, level in original_handler_levels.items():
                handler.setLevel(level)

    def _first_time_setup(self) -> int:
        """Guided first-time setup: add at least one database."""
        self.console.print("[dim]First time setup — let's add a database.[/dim]\n")

        while not self._add_database():
            if not Confirm.ask("Re-enter database configuration?", default=True):
                return 1

        self._save()
        self.console.print()
        self._show_current_state()
        self._display_completion()
        return 0

    def _interactive_menu(self) -> int:
        """Show current state + action menu loop (database-only)."""
        while True:
            self._show_current_state()

            actions: Dict[str, str] = {"add_database": "Add a database"}
            if self.datasources:
                actions["delete_database"] = "Delete a database"
            actions["done"] = "Done"

            self.console.print("What would you like to do?")
            action = select_choice(self.console, actions, default="done")

            if action == "done":
                self._display_completion()
                return 0
            if action == "add_database":
                self._add_database()
                self._save()
            elif action == "delete_database":
                self._delete_database()
                self._save()

            self.console.print()

    # ── Add database ───────────────────────────────────────────────

    def _add_database(self) -> bool:
        """Add a new database connection. Supports 'back' at each step."""
        self.console.print("[bold yellow]Add Database[/bold yellow]")

        from datus.tools.db_tools import connector_registry

        db_name = ""
        db_type = ""
        config_data: Dict[str, Any] = {}

        step = 0
        while step < 3:
            if step == 0:
                # Step 0: Database name
                db_name = Prompt.ask("- Database name")
                if not db_name.strip():
                    self.console.print("Database name cannot be empty")
                    continue
                if db_name in self.datasources:
                    self.console.print(f"Database '{db_name}' already exists. Delete it first to re-add.")
                    continue
                step = 1

            elif step == 1:
                # Step 1: Database type (with plugin install)
                available_adapters = connector_registry.list_available_adapters()
                installed_types = set(available_adapters.keys())
                installable_types = {"snowflake", "mysql", "postgresql", "starrocks", "bigquery", "clickhouse"}
                not_installed = sorted(installable_types - installed_types)

                all_choices = {}
                for t in sorted(installed_types):
                    all_choices[t] = t
                for t in not_installed:
                    all_choices[t] = f"{t} (not installed — will install datus-{t})"

                if not all_choices:
                    self.console.print("No database types available.")
                    return False

                default_type = "duckdb" if "duckdb" in all_choices else list(all_choices.keys())[0]
                self.console.print("- Database type:")
                db_type = select_choice(self.console, all_choices, default=default_type)

                # Install plugin if needed
                if db_type not in installed_types:
                    package = f"datus-{db_type}"
                    self.console.print(f"[dim]Installing {package}...[/dim]")
                    if not self._install_plugin(package):
                        return False
                    self.console.print(f"[green]{package} installed successfully.[/green]")
                    self.console.print(
                        "[yellow]Please run `datus configure` again to configure the new database.[/yellow]"
                    )
                    return False

                step = 2

            elif step == 2:
                # Step 2: Connection fields (from adapter schema)
                available_adapters = connector_registry.list_available_adapters()
                adapter_metadata = available_adapters[db_type]
                config_fields = adapter_metadata.get_config_fields()

                if not config_fields:
                    self.console.print(f"Adapter '{db_type}' does not have a configuration schema.")
                    return False

                config_data = {"type": db_type}
                field_list = [(k, v) for k, v in config_fields.items() if k not in ("type", "name")]
                field_idx = 0
                went_back_to_type = False

                while field_idx < len(field_list):
                    field_name, field_info = field_list[field_idx]
                    label = f"- {field_name.replace('_', ' ').capitalize()}"
                    default_value = field_info.get("default")
                    input_type = field_info.get("input_type", "text")
                    required = field_info.get("required", False)

                    if input_type == "password" or field_name == "password":
                        value = _prompt_with_back(label, password=True)
                        if value == _BACK:
                            if field_idx == 0:
                                went_back_to_type = True
                                break
                            field_idx -= 1
                            prev_field = field_list[field_idx][0]
                            config_data.pop(prev_field, None)
                            continue
                    elif input_type == "file_path":
                        sample_file = field_info.get("default_sample")
                        default_path = str(self.sample_dir / sample_file) if sample_file else str(default_value or "")
                        value = _prompt_with_back(label, default=default_path)
                    elif field_info.get("type") == "int" or field_name == "port":
                        value = _prompt_with_back(label, default=str(default_value) if default_value else "")
                        if value == _BACK:
                            if field_idx == 0:
                                went_back_to_type = True
                                break
                            field_idx -= 1
                            prev_field = field_list[field_idx][0]
                            config_data.pop(prev_field, None)
                            continue
                        if value:
                            try:
                                value = int(value)
                                if field_name == "port" and not (1 <= value <= 65535):
                                    self.console.print("[yellow]Port must be between 1 and 65535.[/yellow]")
                                    continue
                            except ValueError:
                                self.console.print("[yellow]Invalid integer value.[/yellow]")
                                continue
                        else:
                            value = default_value
                    elif not required and default_value is not None:
                        value = _prompt_with_back(label, default=str(default_value))
                    elif not required:
                        value = _prompt_with_back(label, default="")
                    else:
                        value = _prompt_with_back(label)

                    if value == _BACK:
                        if field_idx == 0:
                            went_back_to_type = True
                            break
                        field_idx -= 1
                        prev_field = field_list[field_idx][0]
                        config_data.pop(prev_field, None)
                        continue

                    if value != "" and value is not None:
                        config_data[field_name] = value
                    field_idx += 1

                if went_back_to_type:
                    step = 1
                    continue

                step = 3

        # Test connectivity — strip internal fields before passing to adapter
        test_data = {k: v for k, v in config_data.items() if k != "default"}
        self.console.print("Testing database connectivity...")
        success, error_msg = detect_db_connectivity(db_name, test_data)
        if not success:
            self.console.print(f"Database connectivity test failed: {error_msg}\n")
            return False

        self.console.print("Database connection test successful\n")

        # Mark as default if first database
        if not self.datasources:
            config_data["default"] = True
        elif Confirm.ask(f"- Set '{db_name}' as default database?", default=False):
            for cfg in self.datasources.values():
                cfg.pop("default", None)
            config_data["default"] = True

        self.datasources[db_name] = config_data
        return True

    # ── Delete ─────────────────────────────────────────────────────

    def _delete_database(self):
        """Delete a database configuration."""
        name = Prompt.ask("- Database name to delete", choices=list(self.datasources.keys()))
        if Confirm.ask(f"Delete database '{name}'?", default=False):
            del self.datasources[name]
            self.console.print(f"Database '{name}' deleted.")

    # ── Save ───────────────────────────────────────────────────────

    def _save(self):
        """Save datasources to agent.yml, preserving all other sections.

        ``agent.providers`` / ``agent.models`` / ``agent.target`` are
        intentionally untouched here; see :class:`ModelCommands` for the
        LLM write path.
        """
        existing = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                pass

        agent = existing.get("agent", {})

        # Ensure services structure, migrating any leftover legacy singular `service`
        raw_services = agent.get("services")
        services = dict(raw_services) if isinstance(raw_services, dict) else {}
        legacy_service = agent.get("service")
        if isinstance(legacy_service, dict):
            for key, value in legacy_service.items():
                services.setdefault(key, value)
        services["datasources"] = self.datasources
        if "semantic_layer" not in services:
            services["semantic_layer"] = {}
        if "bi_platforms" not in services:
            services["bi_platforms"] = {}
        if "schedulers" not in services:
            services["schedulers"] = {}
        agent["services"] = services

        # Remove legacy namespace and singular service keys
        agent.pop("namespace", None)
        agent.pop("service", None)

        # Set default nodes if not present
        if "nodes" not in agent:
            agent["nodes"] = {"schema_linking": {"matching_rate": "fast"}, "date_parser": {"language": "en"}}

        existing["agent"] = agent

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ── Display ────────────────────────────────────────────────────

    def _display_completion(self):
        default_db = ""
        for name, cfg in self.datasources.items():
            if cfg.get("default"):
                default_db = name
                break
        if not default_db and self.datasources:
            default_db = next(iter(self.datasources))

        if default_db:
            self.console.print(
                f"\nRun `datus init` to initialize your project, or `datus-cli --database {default_db}`."
            )
        else:
            self.console.print("\nRun `datus init` to initialize your project.")

    # ── Plugin install ─────────────────────────────────────────────

    def _install_plugin(self, package: str) -> bool:
        """Install a database adapter plugin into the current Python environment."""
        import shutil
        import subprocess
        import sys

        # Use the current Python's pip to ensure correct environment
        python = sys.executable
        uv_path = shutil.which("uv")
        if uv_path:
            cmd = [uv_path, "pip", "install", "--python", python, package]
        else:
            cmd = [python, "-m", "pip", "install", package]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.console.print(f"[red]Install failed: {result.stderr.strip()}[/red]")
                return False
            return True
        except subprocess.TimeoutExpired:
            self.console.print("[red]Install timed out.[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]Install failed: {e}[/red]")
            return False
