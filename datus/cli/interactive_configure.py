#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Interactive configuration command for Datus Agent.

Configures workspace-level settings: LLM provider, database connections,
and storage paths. Writes to ~/.datus/conf/agent.yml using the new
service.databases format.

This replaces the LLM/DB config part of the old `datus init` command.
Bootstrap (metadata KB, reference SQL) is now separate via `datus bootstrap-kb`.
"""

import logging
import os
from getpass import getpass
from pathlib import Path
from typing import Optional

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


class InteractiveConfigure:
    """Interactive configuration wizard for Datus Agent workspace."""

    def __init__(self, user_home: Optional[str] = None):
        self.db_name = ""
        self.workspace_path = ""
        self.user_home = user_home if user_home else Path.home()
        self.console = Console(log_path=False)

        path_manager = get_path_manager()
        self.conf_dir = path_manager.conf_dir
        self.template_dir = path_manager.template_dir
        self.sample_dir = path_manager.sample_dir

        self.config = {
            "agent": {
                "target": "",
                "models": {},
                "service": {
                    "databases": {},
                    "bi_tools": {},
                    "schedulers": {},
                },
                "storage": {
                    "workspace_root": "~/.datus/workspace",
                    "embedding_device_type": "cpu",
                },
                "nodes": {
                    "schema_linking": {"matching_rate": "fast"},
                    "date_parser": {"language": "en"},
                },
            }
        }

    def _init_dirs(self):
        path_manager = get_path_manager()
        path_manager.ensure_dirs("conf", "data", "logs", "sessions", "template", "sample")

    def _copy_files(self):
        """Copy template and sample files to datus home."""
        try:
            copy_data_file(resource_path="prompts", dest_dir=self.template_dir, overwrite=True)
        except Exception as e:
            logger.debug(f"Error copying template files: {e}")
        try:
            copy_data_file(resource_path="sample_data", dest_dir=self.sample_dir, overwrite=False)
        except Exception as e:
            logger.debug(f"Error copying sample files: {e}")

    def run(self) -> int:
        """Main entry point for the interactive configuration."""
        self._init_dirs()
        self._copy_files()

        config_path = self.conf_dir / "agent.yml"

        if config_path.exists():
            self.console.print(f"\n[yellow]Configuration file already exists at {config_path}[/yellow]")
            if not Confirm.ask("Do you want to overwrite the existing configuration?", default=False):
                self.console.print("Configuration cancelled.")
                return 0
            # Backup with timestamp
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.with_suffix(f".yml.bak.{ts}")
            import shutil

            shutil.copy2(config_path, backup_path)
            self.console.print(f"[dim]Backed up to {backup_path}[/dim]\n")

        # Suppress console logging during configure process
        root_logger = logging.getLogger()
        original_handler_levels = {}
        for handler in root_logger.handlers:
            if hasattr(handler, "stream") and handler.stream.name in ["<stdout>", "<stderr>"]:
                original_handler_levels[handler] = handler.level
                handler.setLevel(logging.CRITICAL + 1)

        try:
            self.console.print("\n[bold cyan]Datus Configure[/bold cyan]")
            self.console.print("Set up your LLM and database connections.\n")

            # Step 1: Configure LLM
            while not self._configure_llm():
                if not Confirm.ask("Re-enter LLM configuration?", default=True):
                    return 1

            # Step 2: Configure Database
            while not self._configure_database():
                if not Confirm.ask("Re-enter database configuration?", default=True):
                    return 1

            # Step 3: Configure Workspace
            while not self._configure_workspace():
                if not Confirm.ask("Re-enter workspace configuration?", default=True):
                    return 1

            if not self._save_configuration():
                return 1

            # Summary
            self.console.print("\n[bold yellow]Configuration Summary[/bold yellow]")
            self._display_summary()
            self._display_completion()
            return 0

        except KeyboardInterrupt:
            self.console.print("\nConfiguration cancelled by user")
            return 1
        except Exception as e:
            print_rich_exception(self.console, e, "Configuration failed", logger)
            return 1
        finally:
            for handler, level in original_handler_levels.items():
                handler.setLevel(level)

    def _load_provider_catalog(self) -> dict:
        try:
            text = read_data_file_text(resource_path="conf/providers.yml", encoding="utf-8")
            return yaml.safe_load(text)
        except Exception as e:
            logger.error(f"Failed to load providers.yml: {e}")
            return {"providers": {}, "model_overrides": {}}

    def _configure_llm(self) -> bool:
        """Step 1: Configure LLM provider and test connectivity."""
        self.console.print("[bold yellow][1/3] Configure LLM[/bold yellow]")

        catalog = self._load_provider_catalog()
        providers = catalog.get("providers", {})
        model_param_overrides = catalog.get("model_overrides", {})

        if not providers:
            self.console.print("No providers found in conf/providers.yml")
            return False

        self.console.print("- Which LLM provider?")
        provider = select_choice(
            self.console,
            {k: k for k in providers.keys()},
            default="openai",
        )

        # OAuth flow for Codex provider
        if providers[provider].get("auth_type") == "oauth":
            return self._configure_codex_oauth(provider, providers[provider])

        # Subscription flow for Claude subscription
        if providers[provider].get("auth_type") == "subscription":
            return self._configure_claude_subscription(provider, providers[provider])

        provider_info = providers[provider]

        # API key: detect env var, offer ${ENV_VAR} as default
        api_key_env = provider_info.get("api_key_env", "")
        env_value = os.environ.get(api_key_env, "") if api_key_env else ""

        if env_value:
            self.console.print(f"  [dim]Detected ${{{api_key_env}}} in environment[/dim]")
            use_env = Confirm.ask(f"- Use ${{{api_key_env}}} as API key?", default=True)
            if use_env:
                api_key = f"${{{api_key_env}}}"
            else:
                api_key = getpass("- Enter your API key: ")
        elif api_key_env:
            self.console.print(f"  [dim]Hint: set ${{{api_key_env}}} env var to avoid entering key manually[/dim]")
            api_key = Prompt.ask(
                f"- API key (or env var like ${{{api_key_env}}})",
                default=f"${{{api_key_env}}}",
            )
        else:
            api_key = getpass("- Enter your API key: ")

        if not api_key.strip():
            self.console.print("API key cannot be empty")
            return False
        base_url = Prompt.ask("- Enter your base URL", default=provider_info["base_url"])

        models = provider_info.get("models", [])
        if models:
            self.console.print("- Select your model:")
            model_name = select_choice(
                self.console,
                {str(m): str(m) for m in models},
                default=provider_info.get("default_model", str(models[0])),
                allow_free_text=True,
            )
        else:
            model_name = Prompt.ask("- Enter your model name", default=provider_info.get("default_model", "")).strip()

        self.config["agent"]["target"] = provider
        model_config_entry = {
            "type": provider_info["type"],
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
        }
        if model_name in model_param_overrides:
            model_config_entry.update(model_param_overrides[model_name])
        self.config["agent"]["models"][provider] = model_config_entry

        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity()
        if success:
            self.console.print("LLM model test successful\n")
            return True
        else:
            self.console.print(f"LLM connectivity test failed: {error_msg}\n")
            return False

    def _configure_database(self) -> bool:
        """Step 2: Configure database connection."""
        self.console.print("[bold yellow][2/3] Configure Database[/bold yellow]")

        self.db_name = Prompt.ask("- Database name")
        if not self.db_name.strip():
            self.console.print("Database name cannot be empty")
            return False

        from datus.tools.db_tools import connector_registry

        available_adapters = connector_registry.list_available_adapters()
        if not available_adapters:
            self.console.print("No database adapters available.")
            return False

        db_types = sorted(available_adapters.keys())
        default_type = "duckdb" if "duckdb" in db_types else db_types[0]
        self.console.print("- Database type:")
        db_type = select_choice(
            self.console,
            {t: t for t in db_types},
            default=default_type,
        )

        adapter_metadata = available_adapters[db_type]
        config_fields = adapter_metadata.get_config_fields()

        config_data = {"type": db_type}

        if not config_fields:
            self.console.print(f"Adapter '{db_type}' does not have a configuration schema registered.")
            return False

        for field_name, field_info in config_fields.items():
            if field_name in ["type", "name"]:
                continue

            label = f"- {field_name.replace('_', ' ').capitalize()}"
            required = field_info.get("required", False)
            default_value = field_info.get("default")
            input_type = field_info.get("input_type", "text")

            if input_type == "password" or field_name == "password":
                value = getpass(f"{label}: ")
            elif input_type == "file_path":
                sample_file = field_info.get("default_sample")
                if sample_file:
                    default_path = str(self.sample_dir / sample_file)
                    value = Prompt.ask(label, default=default_path)
                else:
                    value = Prompt.ask(label, default=str(default_value) if default_value else "")
            elif field_info.get("type") == "int" or field_name == "port":
                while True:
                    value_str = Prompt.ask(label, default=str(default_value) if default_value else "")
                    if not value_str:
                        value = default_value
                        break
                    try:
                        value = int(value_str)
                        if field_name == "port" and not (1 <= value <= 65535):
                            self.console.print("[yellow]Port must be between 1 and 65535.[/yellow]")
                            continue
                        break
                    except ValueError:
                        self.console.print("[yellow]Invalid integer value.[/yellow]")
            elif not required and default_value is not None:
                value = Prompt.ask(label, default=str(default_value))
            elif not required:
                value = Prompt.ask(label, default="")
            else:
                value = Prompt.ask(label)

            if value != "" and value is not None:
                config_data[field_name] = value

        # Mark as default (first database)
        config_data["default"] = True

        self.config["agent"]["service"]["databases"][self.db_name] = config_data

        # Test connectivity
        self.console.print("Testing database connectivity...")
        success, error_msg = detect_db_connectivity(self.db_name, config_data)
        if success:
            self.console.print("Database connection test successful\n")
            return True
        else:
            self.console.print(f"Database connectivity test failed: {error_msg}\n")
            if self.db_name in self.config["agent"]["service"]["databases"]:
                del self.config["agent"]["service"]["databases"][self.db_name]
            return False

    def _configure_workspace(self) -> bool:
        """Step 3: Configure workspace directory."""
        self.console.print("[bold yellow][3/3] Configure Workspace[/bold yellow]")

        default_workspace = str(self.user_home / ".datus" / "workspace")
        self.workspace_path = Prompt.ask("- Workspace path", default=default_workspace)

        self.config["agent"]["storage"]["workspace_root"] = self.workspace_path
        self.config["agent"]["storage"]["base_path"] = str(self.user_home / ".datus" / "data")

        try:
            Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
            self.console.print("Workspace directory created\n")
            return True
        except Exception as e:
            print_rich_exception(self.console, e, "Failed to create workspace directory", logger)
            return False

    def _save_configuration(self) -> bool:
        """Save configuration, merging with existing file to preserve other sections."""
        try:
            config_path = self.conf_dir / "agent.yml"

            # Load existing config to preserve sections we don't touch
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}

            existing_agent = existing.get("agent", {})

            # Merge: only overwrite sections that configure touches
            existing_agent["target"] = self.config["agent"]["target"]
            existing_agent["models"] = self.config["agent"]["models"]
            existing_agent["service"] = self.config["agent"]["service"]
            existing_agent["storage"] = self.config["agent"]["storage"]

            # Set default nodes if not already configured
            if "nodes" not in existing_agent:
                existing_agent["nodes"] = self.config["agent"]["nodes"]

            # Remove legacy namespace key if present
            existing_agent.pop("namespace", None)

            existing["agent"] = existing_agent

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self.console.print(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            self.console.print(f"Failed to save configuration: {e}")
            return False

    def _display_summary(self):
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        provider = self.config["agent"]["target"]
        model = self.config["agent"]["models"][provider]["model"]

        table.add_row("LLM", f"{provider} ({model})")
        table.add_row("Database", self.db_name)
        table.add_row("Workspace", self.workspace_path)

        self.console.print(table)

    def _display_completion(self):
        self.console.print("\nYou can now run `datus init` to initialize your project workspace.")
        self.console.print(f"Or run `datus-cli --database {self.db_name}` to start the CLI.\n")
        self.console.print("Check the document at https://docs.datus.ai/ for more details.")

    def _test_llm_connectivity(self) -> tuple[bool, str]:
        """Test LLM connectivity with the configured model."""
        try:
            from datus.configuration.agent_config import load_model_config, resolve_env
            from datus.models.base import LLMBaseModel

            provider = self.config["agent"]["target"]
            raw = dict(self.config["agent"]["models"][provider])
            # Resolve env vars (e.g. ${DEEPSEEK_API_KEY}) for the test
            resolved = {k: resolve_env(str(v)) if isinstance(v, str) else v for k, v in raw.items()}
            model_config = load_model_config(resolved)

            # Instantiate the model class directly (create_model expects AgentConfig)
            model_type = model_config.type
            model_class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_type)
            if not model_class_name:
                return False, f"Unsupported model type: {model_type}"
            module = __import__(f"datus.models.{model_type}_model", fromlist=[model_class_name])
            model_class = getattr(module, model_class_name)
            llm = model_class(model_config)

            response = llm.generate("Say hello in 5 words")
            if response:
                return True, ""
            return False, "Empty response from model"
        except Exception as e:
            return False, str(e)

    def _configure_codex_oauth(self, provider: str, provider_config: dict) -> bool:
        """Configure Codex with OAuth authentication."""
        try:
            from datus.auth.codex_credential import get_codex_oauth_token

            token = get_codex_oauth_token()
        except Exception as e:
            self.console.print(f"Failed to get Codex OAuth token: {e}")
            return False

        models = provider_config.get("models", [])
        if models:
            self.console.print("- Select your model:")
            model_name = select_choice(
                self.console,
                {m: m for m in models},
                default=provider_config.get("default_model", models[0]),
                allow_free_text=True,
            )
        else:
            model_name = Prompt.ask("- Enter your model name", default=provider_config.get("default_model", "")).strip()

        self.config["agent"]["target"] = provider
        self.config["agent"]["models"][provider] = {
            "type": provider_config["type"],
            "vendor": provider,
            "api_key": token,
            "model": model_name,
            "auth_type": "oauth",
        }

        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity()
        if success:
            self.console.print("Codex OAuth model test successful\n")
            return True
        else:
            self.console.print(f"LLM connectivity test failed: {error_msg}\n")
            return False

    def _configure_claude_subscription(self, provider: str, provider_config: dict) -> bool:
        """Configure Claude with subscription plan (Pro/Max)."""
        models = provider_config.get("models", [])
        if models:
            self.console.print("- Select your model:")
            model_name = select_choice(
                self.console,
                {m: m for m in models},
                default=provider_config.get("default_model", models[0]),
                allow_free_text=True,
            )
        else:
            model_name = Prompt.ask("- Enter your model name", default=provider_config.get("default_model", "")).strip()

        self.console.print("  [dim]Detecting Claude subscription token...[/dim]")
        try:
            from datus.auth.claude_credential import get_claude_subscription_token

            token, source = get_claude_subscription_token()
            self.console.print(f"  Subscription token detected (from {source})")
            auth_type = "subscription"
        except Exception:
            self.console.print("  [yellow]Could not auto-detect subscription token[/yellow]")
            token = getpass("- Paste your subscription token (sk-ant-oat01-...): ")
            if not token.strip():
                self.console.print("Token cannot be empty")
                return False
            auth_type = "subscription"

        self.config["agent"]["target"] = provider
        self.config["agent"]["models"][provider] = {
            "type": provider_config["type"],
            "vendor": provider,
            "base_url": provider_config["base_url"],
            "api_key": token,
            "model": model_name,
            "auth_type": auth_type,
        }

        self.console.print("Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity()
        if success:
            self.console.print("Claude subscription model test successful\n")
            return True
        else:
            self.console.print(f"LLM connectivity test failed: {error_msg}")
            return False
