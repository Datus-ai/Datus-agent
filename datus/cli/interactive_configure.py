#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Interactive configuration manager for Datus Agent.

Incrementally manages LLM models and database connections in
~/.datus/conf/agent.yml. Shows current state, supports add/delete
of individual entries. Does not touch other config sections.
"""

import logging
import os
from getpass import getpass
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
from datus.utils.resource_utils import copy_data_file, read_data_file_text

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
    """Incremental configuration manager for LLM models and databases."""

    def __init__(self, user_home: Optional[str] = None):
        self.user_home = user_home if user_home else Path.home()
        self.console = Console(log_path=False)

        path_manager = get_path_manager()
        self.conf_dir = path_manager.conf_dir
        self.template_dir = path_manager.template_dir
        self.sample_dir = path_manager.sample_dir
        self.config_path = self.conf_dir / "agent.yml"

        # Working state — loaded from existing config or empty
        self.target: str = ""
        self.models: Dict[str, dict] = {}
        self.databases: Dict[str, dict] = {}

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
        """Load models and databases from existing agent.yml."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            return

        agent = raw.get("agent", {})
        self.target = agent.get("target", "")
        self.models = agent.get("models", {})

        # Support new services format and legacy singular service / namespace formats
        services = agent.get("services") or {}
        legacy_service = agent.get("service") or {}
        if isinstance(services, dict) and services.get("databases") is not None:
            self.databases = services.get("databases", {})
        elif isinstance(legacy_service, dict) and legacy_service.get("databases") is not None:
            self.databases = legacy_service.get("databases", {})
        elif "namespace" in agent:
            # Auto-migrate legacy namespace format
            from datus.configuration.agent_config import ServicesConfig

            migrated = ServicesConfig.migrate_from_namespace(agent["namespace"])
            self.databases = migrated.get("databases", {})

    def _load_provider_catalog(self) -> dict:
        try:
            text = read_data_file_text(resource_path="conf/providers.yml", encoding="utf-8")
            local_catalog = yaml.safe_load(text) or {}
        except Exception as e:
            logger.error(f"Failed to load providers.yml: {e}")
            return {"providers": {}, "model_overrides": {}}

        from datus.cli.provider_model_catalog import resolve_provider_models

        try:
            return resolve_provider_models(local_catalog)
        except Exception as e:
            logger.debug(f"resolve_provider_models failed, using local catalog: {e}")
            return local_catalog

    # ── Display ────────────────────────────────────────────────────

    def _show_current_state(self):
        """Display current models and databases."""
        # Models table
        if self.models:
            table = Table(title="Current Models", show_header=True, header_style="bold green")
            table.add_column("Name", style="cyan")
            table.add_column("Model")
            table.add_column("Base URL")
            table.add_column("Default")
            for name, cfg in self.models.items():
                is_default = "*" if name == self.target else ""
                table.add_row(name, cfg.get("model", ""), cfg.get("base_url", ""), is_default)
            self.console.print(table)
        else:
            self.console.print("[yellow]No models configured.[/yellow]")

        self.console.print()

        # Databases table
        if self.databases:
            table = Table(title="Current Databases", show_header=True, header_style="bold green")
            table.add_column("Name", style="cyan")
            table.add_column("Type")
            table.add_column("Connection")
            table.add_column("Default")
            for name, cfg in self.databases.items():
                conn = cfg.get("uri", "") or cfg.get("host", "") or cfg.get("account", "")
                is_default = "*" if cfg.get("default") else ""
                table.add_row(name, cfg.get("type", ""), str(conn), is_default)
            self.console.print(table)
        else:
            self.console.print("[yellow]No databases configured.[/yellow]")

        self.console.print()

    # ── Main flow ──────────────────────────────────────────────────

    def run(self) -> int:
        self._init_dirs()
        self._copy_files()
        self._load_existing_config()

        # Suppress console logging
        root_logger = logging.getLogger()
        original_handler_levels = {}
        for handler in root_logger.handlers:
            if hasattr(handler, "stream") and handler.stream.name in ["<stdout>", "<stderr>"]:
                original_handler_levels[handler] = handler.level
                handler.setLevel(logging.CRITICAL + 1)

        try:
            self.console.print("\n[bold cyan]Datus Configure[/bold cyan]")
            self.console.print("Manage your LLM models and database connections.\n")

            if not self.models and not self.databases:
                return self._first_time_setup()
            else:
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
        """Guided first-time setup: add one model + one database."""
        self.console.print("[dim]First time setup — let's add a model and database.[/dim]\n")

        # Add model
        while not self._add_model():
            if not Confirm.ask("Re-enter model configuration?", default=True):
                return 1

        # Add database
        while not self._add_database():
            if not Confirm.ask("Re-enter database configuration?", default=True):
                return 1

        self._save()
        self.console.print()
        self._show_current_state()
        self._display_completion()
        return 0

    def _interactive_menu(self) -> int:
        """Show current state + action menu loop."""
        while True:
            self._show_current_state()

            actions = {}
            actions["add_model"] = "Add a model"
            actions["add_database"] = "Add a database"
            if self.models:
                actions["delete_model"] = "Delete a model"
            if self.databases:
                actions["delete_database"] = "Delete a database"
            if len(self.models) > 1:
                actions["set_default_model"] = "Set default model"
            actions["done"] = "Done"

            self.console.print("What would you like to do?")
            action = select_choice(self.console, actions, default="done")

            if action == "done":
                self._display_completion()
                return 0
            elif action == "add_model":
                self._add_model()
                self._save()
            elif action == "add_database":
                self._add_database()
                self._save()
            elif action == "delete_model":
                self._delete_model()
                self._save()
            elif action == "delete_database":
                self._delete_database()
                self._save()
            elif action == "set_default_model":
                self._set_default_model()
                self._save()

            self.console.print()

    # ── Add model ──────────────────────────────────────────────────

    def _add_model(self) -> bool:
        """Add a new LLM model configuration. Supports 'back' at each step."""
        self.console.print("[bold yellow]Add Model[/bold yellow]")

        catalog = self._load_provider_catalog()
        providers = catalog.get("providers", {})
        model_param_overrides = catalog.get("model_overrides", {})

        if not providers:
            self.console.print("No providers found in conf/providers.yml")
            return False

        # Collected values
        provider = ""
        api_key = ""
        base_url = ""
        model_name = ""

        step = 0
        while step < 4:
            if step == 0:
                # Step 0: Provider
                self.console.print("- Which LLM provider?")
                provider = select_choice(self.console, {k: k for k in providers}, default="openai")
                provider_info = providers[provider]

                # OAuth / subscription — no back support for these special flows
                if provider_info.get("auth_type") == "oauth":
                    return self._configure_codex_oauth(provider, provider_info)
                if provider_info.get("auth_type") == "subscription":
                    return self._configure_claude_subscription(provider, provider_info)
                step = 1

            elif step == 1:
                # Step 1: API key
                provider_info = providers[provider]
                api_key_env = provider_info.get("api_key_env", "")
                env_value = os.environ.get(api_key_env, "") if api_key_env else ""

                if env_value:
                    self.console.print(f"  [dim]Detected ${{{api_key_env}}} in environment[/dim]")
                    use_env = Confirm.ask(f"- Use ${{{api_key_env}}} as API key?", default=True)
                    api_key = f"${{{api_key_env}}}" if use_env else _prompt_with_back("- API key", password=True)
                elif api_key_env:
                    self.console.print(
                        f"  [dim]Hint: set ${{{api_key_env}}} env var to avoid entering key manually[/dim]"
                    )
                    api_key = _prompt_with_back(
                        f"- API key (or env var like ${{{api_key_env}}})", default=f"${{{api_key_env}}}"
                    )
                else:
                    api_key = _prompt_with_back("- API key")

                if api_key == _BACK:
                    step = 0
                    continue
                if not api_key.strip():
                    self.console.print("API key cannot be empty")
                    continue
                step = 2

            elif step == 2:
                # Step 2: Base URL
                provider_info = providers[provider]
                base_url = _prompt_with_back("- Base URL", default=provider_info["base_url"])
                if base_url == _BACK:
                    step = 1
                    continue
                step = 3

            elif step == 3:
                # Step 3: Model
                provider_info = providers[provider]
                models = provider_info.get("models", [])
                if models:
                    self.console.print("- Select model:")
                    model_name = select_choice(
                        self.console,
                        {str(m): str(m) for m in models},
                        default=provider_info.get("default_model", str(models[0])),
                        allow_free_text=True,
                    )
                else:
                    model_name = _prompt_with_back("- Model name", default=provider_info.get("default_model", ""))
                    if model_name == _BACK:
                        step = 2
                        continue
                    model_name = model_name.strip()
                step = 4

        entry = {"type": providers[provider]["type"], "base_url": base_url, "api_key": api_key, "model": model_name}
        if model_name in model_param_overrides:
            entry.update(model_param_overrides[model_name])

        # Test connectivity
        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity(entry)
        if not success:
            self.console.print(f"LLM connectivity test failed: {error_msg}\n")
            return False

        self.console.print("LLM model test successful\n")
        self.models[provider] = entry

        if not self.target:
            self.target = provider
        elif Confirm.ask(f"- Set '{provider}' as default model?", default=False):
            self.target = provider

        return True

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
                if db_name in self.databases:
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
        if not self.databases:
            config_data["default"] = True
        elif Confirm.ask(f"- Set '{db_name}' as default database?", default=False):
            for cfg in self.databases.values():
                cfg.pop("default", None)
            config_data["default"] = True

        self.databases[db_name] = config_data
        return True

    # ── Delete ─────────────────────────────────────────────────────

    def _delete_model(self):
        """Delete a model configuration."""
        name = Prompt.ask("- Model name to delete", choices=list(self.models.keys()))
        if Confirm.ask(f"Delete model '{name}'?", default=False):
            del self.models[name]
            if self.target == name:
                self.target = next(iter(self.models), "")
            self.console.print(f"Model '{name}' deleted.")

    def _delete_database(self):
        """Delete a database configuration."""
        name = Prompt.ask("- Database name to delete", choices=list(self.databases.keys()))
        if Confirm.ask(f"Delete database '{name}'?", default=False):
            del self.databases[name]
            self.console.print(f"Database '{name}' deleted.")

    def _set_default_model(self):
        """Set default model (target)."""
        name = Prompt.ask("- Default model", choices=list(self.models.keys()), default=self.target)
        self.target = name
        self.console.print(f"Default model set to '{name}'.")

    # ── Save ───────────────────────────────────────────────────────

    def _save(self):
        """Save models and databases to agent.yml, preserving other sections."""
        existing = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                pass

        agent = existing.get("agent", {})

        # Only update what we manage
        agent["target"] = self.target
        agent["models"] = self.models

        # Ensure services structure, migrating any leftover legacy singular `service`
        raw_services = agent.get("services")
        services = dict(raw_services) if isinstance(raw_services, dict) else {}
        legacy_service = agent.get("service")
        if isinstance(legacy_service, dict):
            for key, value in legacy_service.items():
                services.setdefault(key, value)
        services["databases"] = self.databases
        if "semantic_layer" not in services:
            services["semantic_layer"] = {}
        if "bi_tools" not in services:
            services["bi_tools"] = {}
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
        for name, cfg in self.databases.items():
            if cfg.get("default"):
                default_db = name
                break
        if not default_db and self.databases:
            default_db = next(iter(self.databases))

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

    # ── LLM connectivity test ──────────────────────────────────────

    def _test_llm_connectivity(self, model_entry: dict) -> tuple[bool, str]:
        try:
            from datus.configuration.agent_config import load_model_config, resolve_env
            from datus.models.base import LLMBaseModel

            resolved = {k: resolve_env(str(v)) if isinstance(v, str) else v for k, v in model_entry.items()}
            model_config = load_model_config(resolved)

            model_type = model_config.type
            model_class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_type)
            if not model_class_name:
                return False, f"Unsupported model type: {model_type}"
            module = __import__(f"datus.models.{model_type}_model", fromlist=[model_class_name])
            model_class = getattr(module, model_class_name)
            llm = model_class(model_config)

            response = llm.generate("Say hello in 5 words")
            return (True, "") if response else (False, "Empty response from model")
        except Exception as e:
            return False, str(e)

    # ── Special auth flows ─────────────────────────────────────────

    def _configure_codex_oauth(self, provider: str, provider_config: dict) -> bool:
        try:
            from datus.auth.codex_credential import get_codex_oauth_token

            token = get_codex_oauth_token()
        except Exception as e:
            self.console.print(f"Failed to get Codex OAuth token: {e}")
            return False

        models = provider_config.get("models", [])
        if models:
            self.console.print("- Select model:")
            model_name = select_choice(
                self.console,
                {m: m for m in models},
                default=provider_config.get("default_model", models[0]),
                allow_free_text=True,
            )
        else:
            model_name = Prompt.ask("- Model name", default=provider_config.get("default_model", "")).strip()

        entry = {
            "type": provider_config["type"],
            "vendor": provider,
            "api_key": token,
            "model": model_name,
            "auth_type": "oauth",
        }

        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity(entry)
        if not success:
            self.console.print(f"LLM connectivity test failed: {error_msg}\n")
            return False

        self.console.print("Codex OAuth model test successful\n")
        self.models[provider] = entry
        if not self.target:
            self.target = provider
        return True

    def _configure_claude_subscription(self, provider: str, provider_config: dict) -> bool:
        models = provider_config.get("models", [])
        if models:
            self.console.print("- Select model:")
            model_name = select_choice(
                self.console,
                {m: m for m in models},
                default=provider_config.get("default_model", models[0]),
                allow_free_text=True,
            )
        else:
            model_name = Prompt.ask("- Model name", default=provider_config.get("default_model", "")).strip()

        self.console.print("  [dim]Detecting Claude subscription token...[/dim]")
        try:
            from datus.auth.claude_credential import get_claude_subscription_token

            token, source = get_claude_subscription_token()
            self.console.print(f"  Subscription token detected (from {source})")
        except Exception:
            self.console.print("  [yellow]Could not auto-detect subscription token[/yellow]")
            token = getpass("- Paste your subscription token (sk-ant-oat01-...): ")
            if not token.strip():
                self.console.print("Token cannot be empty")
                return False

        entry = {
            "type": provider_config["type"],
            "vendor": provider,
            "base_url": provider_config["base_url"],
            "api_key": token,
            "model": model_name,
            "auth_type": "subscription",
        }

        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity(entry)
        if not success:
            self.console.print(f"LLM connectivity test failed: {error_msg}")
            return False

        self.console.print("Claude subscription model test successful\n")
        self.models[provider] = entry
        if not self.target:
            self.target = provider
        return True
