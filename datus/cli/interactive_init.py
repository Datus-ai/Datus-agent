#!/usr/bin/env python3

# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.
"""
Interactive initialization command for Datus Agent.

This module provides an interactive CLI for setting up the basic configuration
without requiring users to manually write conf/agent.yml files.
"""

import sys
from getpass import getpass
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datus.cli._cli_utils import select_choice
from datus.cli.cli_styles import print_error
from datus.cli.init_util import detect_db_connectivity
from datus.cli.provider_auth_flows import configure_claude_subscription, configure_codex_oauth
from datus.configuration.agent_config import AgentConfig
from datus.configuration.project_config import ProjectOverride, ProjectTarget, save_project_override
from datus.utils.loggings import configure_logging, get_logger, print_rich_exception
from datus.utils.path_manager import get_path_manager
from datus.utils.path_utils import safe_rmtree
from datus.utils.resource_utils import copy_data_file, read_data_file_text

logger = get_logger(__name__)


class InteractiveInit:
    """Interactive initialization wizard for Datus Agent."""

    def __init__(self, user_home: Optional[str] = None):
        self.workspace_path = ""
        self.namespace_name = ""
        self.user_home = user_home if user_home else Path.home()
        self.console = Console(log_path=False)

        # Use path manager for directory paths.
        # Entry-point exemption: interactive init runs *before* any AgentConfig exists,
        # so we fall back to the context-local path manager / default ~/.datus here.
        path_manager = get_path_manager()
        self.conf_dir = path_manager.conf_dir
        self.template_dir = path_manager.template_dir
        self.sample_dir = path_manager.sample_dir
        self.benchmark_dir = path_manager.benchmark_dir
        # ``agent.providers`` holds per-provider credentials (api_key, base_url,
        # auth_type). ``agent.target`` is intentionally absent — the active
        # provider+model selection is persisted per-project to
        # ``{cwd}/.datus/config.yml`` so each project can pin its own model
        # without cross-project bleed. ``agent.models`` is reserved for custom
        # / self-hosted model entries the user adds later; the init wizard no
        # longer writes to it.
        self.config = {
            "agent": {
                "providers": {},
                "namespace": {},
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
        # Staged project-level target; written to ``./.datus/config.yml`` at
        # the end of ``_save_configuration`` so both files update atomically.
        self._pending_target: Optional[ProjectTarget] = None
        # Probe config of the last tested provider, used by ``_test_llm_connectivity``.
        self._pending_probe: Optional[dict] = None

    def _init_dirs(self):
        from datus.utils.path_manager import get_path_manager

        # Entry-point exemption: see __init__ above — no AgentConfig exists yet during init.
        # Skip project-scoped ``sessions/{project_name}/`` — it is created lazily
        # at session runtime; bootstrapping here has no project_name to use.
        path_manager = get_path_manager()
        path_manager.ensure_dirs("conf", "data", "logs", "template", "sample")

    def run(self) -> int:
        """Main entry point for the interactive initialization."""
        # Check if configuration file already exists
        self._init_dirs()

        self._copy_files()

        config_path = self.conf_dir / "agent.yml"

        if config_path.exists():
            self.console.print(f"\n[yellow]⚠️  Configuration file already exists at {config_path}[/yellow]")
            if not Confirm.ask("Do you want to overwrite the existing configuration?", default=False):
                self.console.print("Initialization cancelled.")
                return 0
            self.console.print()

        import logging

        # Suppress console logging during init process, but keep file logging at INFO level
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()
        console_handlers = []
        original_handler_levels = {}

        # Suppress console handlers completely, keep file handlers at INFO level or above
        for handler in original_handlers:
            if hasattr(handler, "stream") and handler.stream.name in ["<stdout>", "<stderr>"]:
                # Console handlers: disable completely
                console_handlers.append(handler)
                original_handler_levels[handler] = handler.level
                handler.setLevel(logging.CRITICAL + 1)  # Effectively disable console output
            else:
                # File handlers: ensure INFO level or above
                original_handler_levels[handler] = handler.level
                if handler.level > logging.INFO:
                    handler.setLevel(logging.INFO)

        # Ensure root logger allows INFO level for file logging
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

        try:
            self.console.print("\n[bold cyan]Welcome to Datus Init 🎉[/bold cyan]")
            self.console.print("Let's set up your environment step by step.\n")

            # Step 1: Configure LLM
            while not self._configure_llm():
                if not Confirm.ask("Re-enter LLM configuration?", default=True):
                    return 1

            # Step 2: Configure Namespace
            while not self._configure_namespace():
                if not Confirm.ask("Re-enter database configuration?", default=True):
                    return 1

            # Step 3: Configure Workspace
            while not self._configure_workspace():
                if not Confirm.ask("Re-enter workspace configuration?", default=True):
                    return 1

            if not self._save_configuration():
                return 1

            # Step 4: Optional Setup (after config is saved)
            self._optional_setup(str(config_path))

            # Step 5: Summary and save configuration first
            self.console.print("[bold yellow][5/5] Configuration Summary[/bold yellow]")

            self._display_summary()

            self._display_completion()
            return 0

        except KeyboardInterrupt:
            self.console.print("\n❌ Initialization cancelled by user")
            return 1
        except Exception as e:
            print_rich_exception(self.console, e, "Initialization failed", logger)
            return 1
        finally:
            # Restore original logging configuration
            root_logger.setLevel(original_level)
            # Restore original handler levels for all handlers
            for handler, original_handler_level in original_handler_levels.items():
                handler.setLevel(original_handler_level)

    def _load_provider_catalog(self) -> dict:
        """Load LLM provider catalog from conf/providers.yml."""
        try:
            text = read_data_file_text(resource_path="conf/providers.yml", encoding="utf-8")
            return yaml.safe_load(text)
        except Exception as e:
            logger.error(f"Failed to load providers.yml: {e}")
            return {"providers": {}, "model_overrides": {}}

    def _configure_llm(self) -> bool:
        """Step 1: Pick a provider, acquire credentials, and select a model.

        Result shape (written to ``self.config``):

          - ``agent.providers.<provider>`` — credentials block:
            ``{api_key, base_url, auth_type}``
          - ``self._pending_target`` — ``ProjectTarget(provider, model)``,
            flushed to ``{cwd}/.datus/config.yml`` in ``_save_configuration``.

        ``agent.target`` and ``agent.models`` are deliberately untouched; the
        init wizard no longer produces per-model entries. Users who need
        custom / self-hosted endpoints hand-edit ``agent.models`` after init.
        """
        self.console.print("[bold yellow][1/5] Configure LLM[/bold yellow]")

        catalog = self._load_provider_catalog()
        providers = catalog.get("providers", {})

        if not providers:
            self.console.print("❌ No providers found in conf/providers.yml")
            return False

        self.console.print("- Which LLM provider?")
        provider = select_choice(
            self.console,
            {k: k for k in providers.keys()},
            default="openai",
        )

        provider_info = providers.get(provider, {})
        auth_type = provider_info.get("auth_type", "api_key")

        if auth_type == "oauth":
            return self._finalize_provider(
                provider, provider_info, configure_codex_oauth(self.console, provider, provider_info)
            )
        if auth_type == "subscription":
            return self._finalize_provider(
                provider, provider_info, configure_claude_subscription(self.console, provider, provider_info)
            )
        return self._configure_api_key_provider(provider, provider_info)

    def _configure_api_key_provider(self, provider: str, provider_info: dict) -> bool:
        """Prompt for API key + base URL, test connectivity, stage credentials."""
        api_key = getpass("- Enter your API key: ")
        if not api_key.strip():
            self.console.print("❌ API key cannot be empty")
            return False

        base_url = Prompt.ask("- Enter your base URL", default=provider_info.get("base_url", ""))

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

        probe = {
            "type": provider_info.get("type", provider),
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
            "auth_type": "api_key",
        }
        self._pending_probe = probe
        self.console.print("→ Testing LLM connectivity...")
        ok, err = self._test_llm_connectivity()
        if not ok:
            self.console.print(f"❌ LLM connectivity test failed: {err}\n")
            return False

        self.console.print(" ✅ LLM model test successful\n")
        self.config["agent"]["providers"][provider] = {
            "api_key": api_key,
            "base_url": base_url,
            "auth_type": "api_key",
        }
        self._pending_target = ProjectTarget(provider=provider, model=model_name)
        return True

    def _finalize_provider(self, provider: str, provider_info: dict, result: Optional[dict]) -> bool:
        """Persist credentials + target staged by the subscription / oauth flows.

        Returns ``False`` when the helper returned ``None`` so the caller can
        re-prompt; ``True`` otherwise.
        """
        if not result:
            return False
        self._pending_probe = {
            "type": result.get("type", provider_info.get("type", provider)),
            "base_url": result.get("base_url", provider_info.get("base_url", "")),
            "api_key": result.get("api_key", ""),
            "model": result["model"],
            "auth_type": result.get("auth_type", "api_key"),
        }
        # Subscription / OAuth helpers already probe connectivity on their
        # own; an extra generate() call here would double-bill the user.
        self.config["agent"]["providers"][provider] = {
            k: v
            for k, v in {
                "api_key": result.get("api_key") or None,
                "base_url": result.get("base_url") or None,
                "auth_type": result.get("auth_type", "api_key"),
            }.items()
            if v is not None
        }
        self._pending_target = ProjectTarget(provider=provider, model=result["model"])
        return True

    def _configure_namespace(self) -> bool:
        """Step 2: Configure namespace and database."""
        self.console.print("[bold yellow][2/5] Configure Namespace[/bold yellow]")

        # Namespace name
        self.namespace_name = Prompt.ask("- Namespace name")
        if not self.namespace_name.strip():
            self.console.print("❌ Namespace name cannot be empty")
            return False

        # Get available adapters dynamically
        from datus.tools.db_tools import connector_registry

        available_adapters = connector_registry.list_available_adapters()
        if not available_adapters:
            self.console.print("❌ No database adapters available. Please install at least one adapter.")
            return False

        # Database type selection
        db_types = sorted(available_adapters.keys())
        default_type = "duckdb" if "duckdb" in db_types else db_types[0]
        self.console.print("- Database type:")
        db_type = select_choice(
            self.console,
            {t: t for t in db_types},
            default=default_type,
        )

        # Get adapter metadata
        adapter_metadata = available_adapters[db_type]

        config_fields = adapter_metadata.get_config_fields()

        # Collect configuration based on adapter's config schema
        config_data = {
            "type": db_type,
            "name": self.namespace_name,
        }

        # If adapter provides config schema, use it to prompt for fields
        if not config_fields:
            self.console.print(f"❌ Adapter '{db_type}' does not have a configuration schema registered.")
            return False

        for field_name, field_info in config_fields.items():
            # Skip type and name fields
            if field_name in ["type", "name"]:
                continue

            # Determine prompt label and default value
            label = f"- {field_name.replace('_', ' ').capitalize()}"
            required = field_info.get("required", False)
            default_value = field_info.get("default")
            input_type = field_info.get("input_type", "text")

            # Handle input based on input_type metadata
            if input_type == "password" or field_name == "password":
                value = getpass(f"{label}: ")
            elif input_type == "file_path":
                # Handle file path inputs
                sample_file = field_info.get("default_sample")
                if sample_file:
                    default_path = str(self.sample_dir / sample_file)
                    value = Prompt.ask(label, default=default_path)
                else:
                    value = Prompt.ask(label, default=str(default_value) if default_value else "")
            elif field_info.get("type") == "int" or field_name == "port":
                # Handle integer inputs with validation
                while True:
                    value_str = Prompt.ask(label, default=str(default_value) if default_value else "")

                    # Use default if empty string
                    if not value_str:
                        value = default_value
                        break

                    # Try to convert to int
                    try:
                        value = int(value_str)

                        # Validate port range
                        if field_name == "port":
                            if not (1 <= value <= 65535):
                                self.console.print(
                                    "[yellow]Port must be between 1 and 65535. Please try again.[/yellow]"
                                )
                                continue

                        break
                    except ValueError:
                        self.console.print("[yellow]Invalid integer value. Please enter a valid number.[/yellow]")
            elif not required and default_value is not None:
                value = Prompt.ask(label, default=str(default_value))
            elif not required:
                value = Prompt.ask(label, default="")
            else:
                value = Prompt.ask(label)

            # Only add non-empty values
            if value != "" and value is not None:
                config_data[field_name] = value

        self.config["agent"]["namespace"][self.namespace_name] = config_data
        # Test database connectivity
        self.console.print("→ Testing database connectivity...")
        success, error_msg = detect_db_connectivity(
            self.namespace_name, self.config["agent"]["namespace"][self.namespace_name]
        )
        if success:
            self.console.print(" ✅ Database connection test successful\n")
            return True
        else:
            self.console.print(f" ❌ Database connectivity test failed: {error_msg}\n")
            # Remove failed database configuration
            if self.namespace_name in self.config["agent"]["namespace"]:
                del self.config["agent"]["namespace"][self.namespace_name]
            return False

    def _configure_workspace(self) -> bool:
        """Step 3: Configure workspace directory."""
        self.console.print("[bold yellow][3/5] Configure Workspace Root (your sql files located here)[/bold yellow]")

        default_workspace = str(self.user_home / ".datus" / "workspace")
        self.workspace_path = Prompt.ask("- Workspace path", default=default_workspace)

        # Store workspace path in storage configuration
        self.config["agent"]["storage"]["workspace_root"] = self.workspace_path
        self.config["agent"]["storage"]["base_path"] = str(self.user_home / ".datus" / "data")

        # Create workspace directory
        try:
            Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
            self.console.print(" ✅ Workspace directory created\n")
            return True
        except Exception as e:
            print_rich_exception(self.console, e, "Failed to create workspace directory", logger)
            return False

    def _optional_setup(self, config_path: str):
        """Step 4: Optional setup for metadata and reference SQL."""
        self.console.print("[bold yellow][4/5] Optional Setup[/bold yellow]")

        # Initialize metadata knowledge base
        if Confirm.ask("- Initialize vector DB for metadata?", default=False):
            init_metadata_and_log_result(self.namespace_name, config_path, self.console)

        # Initialize reference SQL
        if Confirm.ask("- Initialize reference SQL from workspace?", default=False):
            default_sql_dir = str(Path(self.workspace_path) / "reference_sql")
            sql_dir = Prompt.ask("- Enter SQL directory path to scan", default=default_sql_dir)
            overwrite_sql_and_log_result(
                namespace_name=self.namespace_name, sql_dir=sql_dir, config_path=config_path, console=self.console
            )

        self.console.print()

    def _save_configuration(self) -> bool:
        """Persist credentials to ``~/.datus/conf/agent.yml`` and target to
        ``./.datus/config.yml``.

        Writing the two files atomically keeps the CLI's startup happy: the
        global file drives provider credentials (shared across projects) and
        the project overlay pins ``target: {provider, model}`` for this
        specific workspace.
        """
        try:
            config_path = self.conf_dir / "agent.yml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

            self.console.print(f" ✅ Configuration saved to {config_path}")

            if self._pending_target:
                override = ProjectOverride(target=self._pending_target)
                target_path = save_project_override(override)
                self.console.print(f" ✅ Active target saved to {target_path}")

            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            self.console.print(f" ❌ Failed to save configuration: {e}")
            return False

    def _display_summary(self):
        """Display configuration summary using the provider-level schema."""
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        if self._pending_target and self._pending_target.provider and self._pending_target.model:
            table.add_row("LLM", f"{self._pending_target.provider} ({self._pending_target.model})")
        elif self._pending_target and self._pending_target.custom:
            table.add_row("LLM", f"custom:{self._pending_target.custom}")
        else:
            table.add_row("LLM", "(not configured)")
        table.add_row("Namespace", self.namespace_name)
        table.add_row("Workspace", self.workspace_path)

        self.console.print(table)

    def _display_completion(self):
        """Display completion message."""
        self.console.print(f"\nYou are ready to run `datus-cli --namespace {self.namespace_name}` 🚀")
        self.console.print("\nCheck the document at https://docs.datus.ai/ for more details.")

    def _test_llm_connectivity(self) -> tuple[bool, str]:
        """Instantiate the staged probe config and run a single ``generate`` call.

        Reads from :attr:`_pending_probe` (set by the api-key / subscription /
        oauth flows) rather than reconstructing credentials from the YAML
        being written, so a failed probe does not leak partial state into
        ``self.config``.
        """
        if not self._pending_probe:
            return False, "No pending LLM probe (missing credentials)"
        try:
            from datus.configuration.agent_config import ModelConfig

            probe = self._pending_probe
            model_config = ModelConfig(
                type=probe["type"],
                base_url=probe.get("base_url", ""),
                api_key=probe.get("api_key", ""),
                model=probe["model"],
                temperature=probe.get("temperature"),
                top_p=probe.get("top_p"),
                auth_type=probe.get("auth_type", "api_key"),
                default_headers=probe.get("default_headers"),
            )

            from datus.models.base import LLMBaseModel

            model_type = model_config.type
            class_name = LLMBaseModel.MODEL_TYPE_MAP.get(model_type)
            if not class_name:
                error_msg = f"Unsupported model type: {model_type}"
                logger.error(error_msg)
                return False, error_msg

            module = __import__(f"datus.models.{model_type}_model", fromlist=[class_name])
            model_class = getattr(module, class_name)
            llm_model = model_class(model_config=model_config)

            response = llm_model.generate("Hi")
            if response is not None and len(response.strip()) > 0:
                return True, ""
            return False, "Empty response from model"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM connectivity test failed: {error_msg}")
            return False, error_msg

    def _create_agent_with_config(self, args):
        """Create agent instance with loaded configuration."""
        from datus.agent.agent import Agent
        from datus.configuration.agent_config_loader import load_agent_config

        agent_config = load_agent_config(reload=True)
        agent_config.current_datasource = self.namespace_name

        return Agent(args, agent_config)

    def _copy_files(self):
        copy_data_file(
            resource_path="sample_data/duckdb-demo.duckdb",
            target_dir=self.sample_dir,
        )

        copy_data_file(
            resource_path="sample_data/california_schools",
            target_dir=self.benchmark_dir / "california_schools",
        )


def create_agent(namespace_name: str, components: list, config_path: str, **kwargs):
    import argparse

    default_args = {
        "action": "bootstrap-kb",
        "namespace": namespace_name,
        "components": components,
        "kb_update_strategy": "overwrite",
        "storage_path": None,
        "benchmark": None,
        "schema_linking_type": "full",
        "catalog": "",
        "database_name": "",
        "benchmark_path": None,
        "pool_size": 4,
        "config": config_path,
        "debug": False,
        "save_llm_trace": False,
    }

    # Update with any additional kwargs
    default_args.update(kwargs)

    args = argparse.Namespace(**default_args)

    from datus.agent.agent import Agent
    from datus.configuration.agent_config_loader import load_agent_config

    agent_config = load_agent_config(reload=True, config=config_path, **vars(args))

    agent_config.current_datasource = namespace_name

    return Agent(args, agent_config)


def parse_subject_tree(subject_tree: Optional[str]) -> Optional[list]:
    if not subject_tree:
        return None
    return [item.strip() for item in subject_tree.split(",") if item.strip()]


def _format_reference_sql_line(sql_text: str, max_length: int = 80) -> str:
    condensed = " ".join(str(sql_text).split())
    if len(condensed) > max_length:
        return condensed[:max_length] + "..."
    return condensed or "unknown_sql"


class ReferenceSqlStreamHandler:
    """Stream handler for reference SQL initialization using BatchEvent."""

    def __init__(self, output_mgr):
        from datus.schemas.batch_events import BatchStage

        self.output_mgr = output_mgr
        self.BatchStage = BatchStage
        self.sql_counts: dict[str, int] = {}
        self.current_group: Optional[str] = None

    def handle_event(self, event) -> None:
        from datus.schemas.batch_events import BatchStage

        stage = event.stage

        if stage == BatchStage.TASK_STARTED:
            return

        if stage == BatchStage.TASK_VALIDATED:
            payload = event.payload or {}
            valid_items = payload.get("valid_items", 0)
            invalid_items = payload.get("invalid_items", 0)
            # Don't start progress bar here - wait for TASK_PROCESSING which has actual items to process
            if invalid_items > 0:
                self.output_mgr.add_message(f"Validated: {valid_items} valid, {invalid_items} invalid", style="yellow")
            else:
                self.output_mgr.add_message(f"Validated: {valid_items} SQL items", style="cyan")
            return

        if stage == BatchStage.TASK_PROCESSING:
            total_items = event.total_items or 0
            # Start progress bar here with actual number of items to process
            self.output_mgr.start(total_items=total_items, description="Initializing reference SQL")
            return

        if stage == BatchStage.GROUP_STARTED:
            payload = event.payload or {}
            filepath = str(payload.get("filepath") or event.group_id or "unknown_file")
            total_items = event.total_items or 0
            self.current_group = filepath
            self.output_mgr.start_task(f"File: {escape(filepath)} ({total_items} items)")
            return

        if stage == BatchStage.GROUP_COMPLETED:
            self.output_mgr.complete_task(success=True)
            self.current_group = None
            return

        if stage == BatchStage.ITEM_STARTED:
            payload = event.payload or {}
            filepath = str(payload.get("filepath") or event.group_id or "unknown_file")
            count = self.sql_counts.get(filepath, 0) + 1
            self.sql_counts[filepath] = count
            sql_line = _format_reference_sql_line(str(payload.get("sql") or ""))
            self.output_mgr.add_message(f"#{count}: {escape(sql_line)}", style="dim")
            return

        if stage == BatchStage.ITEM_PROCESSING:
            payload = event.payload or {}
            messages = payload.get("output", {}).get("raw_output")
            if messages:
                self.output_mgr.add_llm_output(str(messages))
            return

        if stage == BatchStage.ITEM_COMPLETED:
            self.output_mgr.update_progress(advance=1.0)
            return

        if stage == BatchStage.ITEM_FAILED:
            error = event.error
            if error:
                self.output_mgr.error(str(error))
            self.output_mgr.update_progress(advance=1.0)
            return

        if stage == BatchStage.TASK_COMPLETED:
            completed = event.completed_items or 0
            failed = event.failed_items or 0
            if failed > 0:
                self.output_mgr.warning(f"Completed: {completed} successful, {failed} failed")
            else:
                self.output_mgr.success(f"All {completed} SQL items processed successfully")
            return


def init_metadata_and_log_result(namespace_name: str, config_path: str, console: Console):
    from datus.configuration.agent_config_loader import load_agent_config
    from datus.storage.schema_metadata.local_init import init_local_schema
    from datus.storage.schema_metadata.store import SchemaWithValueRAG
    from datus.tools.db_tools.db_manager import db_manager_instance

    agent_config = load_agent_config(reload=True, config=config_path)
    agent_config.current_datasource = namespace_name
    kb_update_strategy = "overwrite"
    storage_path = agent_config.rag_storage_path()

    with console.status(f"→ Initializing metadata for {namespace_name} with path `{storage_path}`..."):
        try:
            if kb_update_strategy == "overwrite":
                agent_config.save_storage_config("database")
                from datus.storage.backend_holder import create_vector_connection

                db = create_vector_connection(agent_config.project_name)
                try:
                    db.drop_table("schema_metadata", ignore_missing=True)
                    db.drop_table("schema_value", ignore_missing=True)
                    logger.info("Dropped existing schema_metadata and schema_value tables")
                finally:
                    db.close()
            else:
                agent_config.check_init_storage_config("database")

            metadata_store = SchemaWithValueRAG(agent_config)
            db_manager = db_manager_instance(agent_config.namespaces)
            init_local_schema(
                metadata_store,
                agent_config,
                db_manager,
                build_mode=kb_update_strategy,
                table_type="full",
                init_catalog_name="",
                init_database_name="",
                pool_size=4,
            )

            try:
                schema_size = metadata_store.get_schema_size()
                value_size = metadata_store.get_value_size()
                logger.info(f"Metadata bootstrap completed: {schema_size} tables, {value_size} sample records")
                console.print(f"  → Processed {schema_size} tables with {value_size} sample records")
            except Exception as count_e:
                logger.debug(f"Could not get table counts: {count_e}")
            console.print(" ✅ Metadata knowledge base initialized")
        except Exception as e:
            print_rich_exception(console, e, "Metadata initialization failed", logger)


def overwrite_sql_and_log_result(
    namespace_name: str,
    sql_dir: str,
    config_path: str,
    subject_tree: Optional[str] = None,
    console: Optional[Console] = None,
    force: bool = False,
):
    if not console:
        console = Console(log_path=False)
    from datus.configuration.agent_config_loader import load_agent_config

    try:
        agent_config = load_agent_config(reload=True, config=config_path)
        agent_config.current_datasource = namespace_name
        do_init_sql_and_log_result(agent_config, sql_dir, subject_tree, console, force=force)
    except Exception as e:
        print_rich_exception(console, e, "Reference SQL initialization failed", logger)


def do_init_sql_and_log_result(
    agent_config: AgentConfig,
    sql_dir: str,
    subject_tree: Optional[str] = None,
    console: Optional[Console] = None,
    kb_update_strategy: str = "overwrite",
    force: bool = False,
):
    from datus.storage.reference_sql.reference_sql_init import init_reference_sql
    from datus.storage.reference_sql.store import ReferenceSqlRAG
    from datus.utils.stream_output import StreamOutputManager

    try:
        sql_dir_path = Path(sql_dir)
        if not sql_dir_path.exists():
            print_error(console, f"No sql files found in {sql_dir}", prefix=False)
            return
        if sql_dir_path.is_dir():
            sql_files = list(sql_dir_path.rglob("*.sql"))
            if not sql_files:
                print_error(console, f"No sql files found in {sql_dir}", prefix=False)
                return
        elif sql_dir_path.is_file():
            if sql_dir_path.suffix.lower() != ".sql":
                print_error(console, f"{sql_dir} must be a .sql file", prefix=False)
                return
        else:
            print_error(console, "Only SQL directories or files are supported", prefix=False)
            return

        if kb_update_strategy == "overwrite":
            agent_config.save_storage_config("reference_sql")

            from datus.storage.backend_holder import create_vector_connection

            db = create_vector_connection(agent_config.project_name)
            try:
                db.drop_table("reference_sql", ignore_missing=True)
                logger.info("Dropped existing reference_sql table")
            finally:
                db.close()
            # Also clear the sql_summaries directory (YAML files)
            sql_summary_dir = agent_config.path_manager.sql_summary_path()
            if sql_summary_dir.exists() and not safe_rmtree(sql_summary_dir, "SQL summary directory", force=force):
                console.print("[yellow]Cancelled by user[/yellow]")
                return False, None
        else:
            agent_config.check_init_storage_config("reference_sql")

        console.print(
            f"Reference SQL initialization for {agent_config.current_datasource} (dir: {escape(str(sql_dir))})"
        )

        # Create StreamOutputManager
        output_mgr = StreamOutputManager(
            console=console,
            max_message_lines=10,
            show_progress=True,
            title="Reference SQL Initialization",
        )

        # Create stream handler
        stream_handler = ReferenceSqlStreamHandler(output_mgr)

        subject_tree_list = parse_subject_tree(subject_tree)
        sql_rag = ReferenceSqlRAG(agent_config)

        try:
            result = init_reference_sql(
                sql_rag,
                agent_config,
                sql_dir,
                validate_only=False,
                build_mode=kb_update_strategy,
                pool_size=4,
                subject_tree=subject_tree_list,
                emit=stream_handler.handle_event,
            )
        finally:
            output_mgr.stop()

        if isinstance(result, dict):
            if result.get("message"):
                logger.info(f"Reference SQL bootstrap completed: {result['message']}")

            processed_entries = result.get("processed_entries", 0)
            valid_entries = result.get("valid_entries", 0)
            invalid_entries = result.get("invalid_entries", 0)
            validation_errors = result.get("validation_errors")
            process_errors = result.get("process_errors")
            if valid_entries == 0:
                console.print(f" [yellow]Warning:[/] No SQL files processed in the directory `{sql_dir}`.")
                if validation_errors:
                    console.print(f"    Reason: {validation_errors}")
                return
            if invalid_entries > 0:
                console.print(
                    f"  -> Processed {processed_entries} SQL, {valid_entries} valid SQL,"
                    f" {invalid_entries} invalid SQL. Details: \n\n{validation_errors}",
                )
            if processed_entries == 0:
                console.print(
                    f" [yellow]Warning:[/] Processed failed with validation SQL. Details: \n\n{process_errors}."
                )
                return
            elif process_errors:
                console.print(
                    f"  -> Processed {processed_entries} SQL successfully, "
                    f"but there are still some SQL processing failures. Details: \n\n{process_errors}",
                )
            else:
                console.print(f"  -> Processed {processed_entries} SQL successfully")
            console.print(" [green]OK[/] Imported SQL files into reference completed")
        else:
            logger.info(f"Reference SQL bootstrap result: {result}")
    except Exception as e:
        print_rich_exception(console, e, "Reference SQL initialization failed", logger)


def main():
    """Entry point for the interactive init command."""
    configure_logging(console_output=False)
    init = InteractiveInit()
    return init.run()


if __name__ == "__main__":
    sys.exit(main())
