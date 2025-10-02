#!/usr/bin/env python3
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
from rich.prompt import Confirm, Prompt
from rich.table import Table

from datus.utils.loggings import get_logger
from datus.utils.resource_utils import copy_data_file, read_data_file_text

logger = get_logger(__name__)
console = Console()


class InteractiveInit:
    """Interactive initialization wizard for Datus Agent."""

    def __init__(self, user_home: Optional[str] = None):
        self.workspace_path = ""
        self.namespace_name = ""
        self.user_home = user_home if user_home else Path.home()
        self.datus_dir = self.user_home / ".datus"
        self.data_dir = self.datus_dir / "data"
        self.conf_dir = self.datus_dir / "conf"
        self.template_dir = self.datus_dir / "template"
        self.sample_dir = self.datus_dir / "sample"
        try:
            text = read_data_file_text(resource_path="conf/agent.yml.qs", encoding="utf-8")
            self.config = yaml.safe_load(text)
        except Exception:
            console.print("[yellow]Unable to load sample configuration file, using default configuration[/]")
            self.config = {
                "agent": {
                    "target": "",
                    "models": {},
                    "namespace": {},
                    "storage": {"base_path": "data"},
                    "nodes": {
                        "schema_linking": {"matching_rate": "fast"},
                        "generate_sql": {"prompt_version": "1.0"},
                        "reflect": {"prompt_version": "2.1"},
                        "date_parser": {"language": "en"},
                    },
                }
            }

    def _init_dirs(self):
        for dir_name in (self.datus_dir, self.data_dir, self.conf_dir, self.template_dir, self.sample_dir):
            dir_name.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """Main entry point for the interactive initialization."""
        # Check if configuration file already exists
        self._init_dirs()

        self._copy_files()

        config_path = self.conf_dir / "agent.yml"

        if config_path.exists():
            console.print(f"\n[yellow]⚠️  Configuration file already exists at {config_path}[/yellow]")
            if not Confirm.ask("Do you want to overwrite the existing configuration?", default=False):
                console.print("Initialization cancelled.")
                return 0
            console.print()

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
            console.print("\n[bold cyan]Welcome to Datus Init 🎉[/bold cyan]")
            console.print("Let's set up your environment step by step.\n")

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
            self._optional_setup()

            # Step 5: Summary and save configuration first
            console.print("[bold yellow][5/5] Configuration Summary[/bold yellow]")

            self._display_summary()

            self._display_completion()
            return 0

        except KeyboardInterrupt:
            console.print("\n❌ Initialization cancelled by user")
            return 1
        except Exception as e:
            console.print(f"\n❌ Initialization failed: {e}")
            logger.error(f"Initialization failed: {e}")
            return 1
        finally:
            # Restore original logging configuration
            root_logger.setLevel(original_level)
            # Restore original handler levels for all handlers
            for handler, original_handler_level in original_handler_levels.items():
                handler.setLevel(original_handler_level)

    def _configure_llm(self) -> bool:
        """Step 1: Configure LLM provider and test connectivity."""
        console.print("[bold yellow][1/5] Configure LLM[/bold yellow]")

        # Provider selection
        providers = {
            "openai": {"type": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
            "deepseek": {"type": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
            "claude": {"type": "claude", "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-20250514"},
            "kimi": {"type": "openai", "base_url": "https://api.moonshot.cn/v1", "model": "kimi-k2-turbo-preview"},
            "qwen": {
                "type": "openai",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3-coder",
            },
        }

        provider = Prompt.ask("- Which LLM provider?", choices=list(providers.keys()), default="openai")

        # API key input
        api_key = getpass("- Enter your API key: ")
        if not api_key.strip():
            console.print("❌ API key cannot be empty")
            return False

        # Base URL (with default)
        base_url = Prompt.ask("- Enter your base URL", default=providers[provider]["base_url"])

        # Model name (with default)
        model_name = Prompt.ask("- Enter your model name", default=providers[provider]["model"])

        # Store configuration
        self.config["agent"]["target"] = provider
        self.config["agent"]["models"][provider] = {
            "type": providers[provider]["type"],
            "vendor": provider,
            "base_url": base_url,
            "api_key": api_key,
            "model": model_name,
        }

        # Test LLM connectivity
        console.print("→ Testing LLM connectivity...")
        success, error_msg = self._test_llm_connectivity()
        if success:
            console.print("✔ LLM model test successful\n")
            return True
        else:
            console.print(f"❌ LLM connectivity test failed: {error_msg}\n")
            return False

    def _configure_namespace(self) -> bool:
        """Step 2: Configure namespace and database."""
        console.print("[bold yellow][2/5] Configure Namespace[/bold yellow]")

        # Namespace name
        self.namespace_name = Prompt.ask("- Namespace name")
        if not self.namespace_name.strip():
            console.print("❌ Namespace name cannot be empty")
            return False

        # Database type selection
        db_types = ["sqlite", "duckdb", "snowflake", "mysql", "postgresql", "starrocks"]
        db_type = Prompt.ask("- Database type", choices=db_types, default="duckdb")

        # Connection configuration based on database type
        if db_type in ["starrocks", "mysql"]:
            # Host-based database configuration (StarRocks/MySQL)
            host = Prompt.ask("- Host", default="127.0.0.1")
            port = Prompt.ask("- Port", default="9030")
            username = Prompt.ask("- Username")
            password = getpass("- Password: ")
            database = Prompt.ask("- Database")

            # Store configuration
            config_data = {
                "type": db_type,
                "name": self.namespace_name,
                "host": host,
                "port": int(port),
                "username": username,
                "password": password,
                "database": database,
            }

            # Add StarRocks-specific catalog field
            if db_type == "starrocks":
                config_data["catalog"] = "default_catalog"

            self.config["agent"]["namespace"][self.namespace_name] = config_data
        elif db_type == "snowflake":
            # Snowflake specific configuration
            username = Prompt.ask("- Username")
            account = Prompt.ask("- Account")
            warehouse = Prompt.ask("- Warehouse")
            password = getpass("- Password: ")
            database = Prompt.ask("- Database", default="")
            schema = Prompt.ask("- Schema", default="")

            # Store Snowflake-specific configuration
            self.config["agent"]["namespace"][self.namespace_name] = {
                "type": db_type,
                "name": self.namespace_name,
                "account": account,
                "username": username,
                "password": password,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
            }
        else:
            # For other database types (sqlite, duckdb, postgresql), use connection string
            if db_type == "duckdb":
                default_conn_string = str(self.sample_dir / "duckdb-demo.duckdb")
                conn_string = Prompt.ask("- Connection string", default=default_conn_string)
            else:
                conn_string = Prompt.ask("- Connection string")

            self.config["agent"]["namespace"][self.namespace_name] = {
                "type": db_type,
                "name": self.namespace_name,
                "uri": conn_string,
            }
        # Test database connectivity
        console.print("→ Testing database connectivity...")
        success, error_msg = self._test_db_connectivity()
        if success:
            console.print("✔ Database connection test successful\n")
            return True
        else:
            console.print(f"❌ Database connectivity test failed: {error_msg}\n")
            return False

    def _configure_workspace(self) -> bool:
        """Step 3: Configure workspace directory."""
        console.print("[bold yellow][3/5] Configure Workspace Root (your sql files located here)[/bold yellow]")

        default_workspace = str(self.user_home / ".datus" / "workspace")
        self.workspace_path = Prompt.ask("- Workspace path", default=default_workspace)

        # Store workspace path in storage configuration
        self.config["agent"]["storage"]["workspace_root"] = self.workspace_path
        self.config["agent"]["storage"]["base_path"] = str(self.user_home / ".datus" / "data")

        # Create workspace directory
        try:
            Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
            console.print("✔ Workspace directory created\n")
            return True
        except Exception as e:
            console.print(f"❌ Failed to create workspace directory: {e}\n")
            return False

    def _optional_setup(self):
        """Step 4: Optional setup for metadata and SQL history."""
        console.print("[bold yellow][4/5] Optional Setup[/bold yellow]")

        # Initialize metadata knowledge base
        if Confirm.ask("- Initialize vector DB for metadata?", default=False):
            console.print("→ Initializing metadata knowledge base...")
            if self._initialize_metadata():
                console.print("✔ Metadata knowledge base initialized")
            else:
                console.print("❌ Metadata initialization failed")

        # Initialize SQL history
        if Confirm.ask("- Initialize SQL history from workspace?", default=False):
            default_sql_dir = str(Path(self.workspace_path) / "sql_history")
            sql_dir = Prompt.ask("- Enter SQL directory path to scan", default=default_sql_dir)

            if Path(sql_dir).exists():
                console.print(f"→ Scanning {sql_dir} for SQL files...")
                sql_count = self._initialize_sql_history(sql_dir)
                if sql_count > 0:
                    console.print(f"✔ Imported {sql_count} SQL files into history")
                else:
                    console.print("⚠️ No SQL files found in specified directory")
            else:
                console.print(f"❌ Directory {sql_dir} does not exist")

        console.print()

    def _save_configuration(self) -> bool:
        """Save configuration to file."""
        # Use ~/.datus/conf/agent.yml as the configuration path
        conf_dir = Path.home() / ".datus" / "conf"
        conf_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        try:
            config_path = conf_dir / "agent.yml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

            console.print(f"Configuration saved to {config_path} ✅")
            return True
        except Exception as e:
            console.print(f"❌ Failed to save configuration: {e}")
            return False

    def _display_summary(self):
        """Display configuration summary."""
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        provider = self.config["agent"]["target"]
        model = self.config["agent"]["models"][provider]["model"]

        table.add_row("LLM", f"{provider} ({model})")
        table.add_row("Namespace", self.namespace_name)
        table.add_row("Workspace", self.workspace_path)

        console.print(table)

    def _display_completion(self):
        """Display completion message."""
        console.print(f"\nYou are ready to run `datus-cli --namespace {self.namespace_name}` 🚀")
        console.print("\nCheck the document at https://docs.datus.ai/ for more details.")

    def _test_llm_connectivity(self) -> tuple[bool, str]:
        """Test LLM model connectivity."""
        try:
            # Test LLM connectivity by creating the specific model directly
            provider = self.config["agent"]["target"]
            model_config_data = self.config["agent"]["models"][provider]

            # Create model config object
            from datus.configuration.agent_config import ModelConfig

            model_config = ModelConfig(
                type=model_config_data["type"],
                base_url=model_config_data["base_url"],
                api_key=model_config_data["api_key"],
                model=model_config_data["model"],
            )

            # Import and create the specific model class
            model_type = model_config_data["type"]

            # Map model types to class names
            type_map = {
                "deepseek": "DeepSeekModel",
                "openai": "OpenAIModel",
                "claude": "ClaudeModel",
                "qwen": "QwenModel",
                "gemini": "GeminiModel",
            }

            if model_type not in type_map:
                error_msg = f"Unsupported model type: {model_type}"
                logger.error(error_msg)
                return False, error_msg

            class_name = type_map[model_type]
            module = __import__(f"datus.models.{model_type}_model", fromlist=[class_name])
            model_class = getattr(module, class_name)

            # Create model instance
            llm_model = model_class(model_config=model_config)

            # Simple test - try to generate a response
            response = llm_model.generate("Hi")
            if response is not None and len(response.strip()) > 0:
                return True, ""
            else:
                return False, "Empty response from model"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM connectivity test failed: {error_msg}")
            return False, error_msg

    def _test_db_connectivity(self) -> tuple[bool, str]:
        """Test database connectivity."""
        try:
            # Test database connectivity using connector's built-in method
            from datus.configuration.agent_config import DbConfig
            from datus.tools.db_tools.db_manager import DBManager

            # Get database configuration
            db_config_data = self.config["agent"]["namespace"][self.namespace_name]
            db_type = db_config_data["type"]

            # Create DbConfig object with appropriate fields based on database type
            if db_type in ["starrocks", "mysql", "snowflake"]:
                # For connectors that need specific parameters
                db_config = DbConfig(
                    type=db_type,
                    host=db_config_data.get("host", ""),
                    port=db_config_data.get("port", 0),
                    username=db_config_data.get("username", ""),
                    password=db_config_data.get("password", ""),
                    database=db_config_data.get("database", ""),
                    # StarRocks specific
                    catalog=db_config_data.get("catalog", "default_catalog"),
                    # Snowflake specific
                    account=db_config_data.get("account", ""),
                    warehouse=db_config_data.get("warehouse", ""),
                    schema=db_config_data.get("schema", ""),
                )
            else:
                # For URI-based connectors (sqlite, duckdb, postgresql)
                db_config = DbConfig(
                    type=db_type,
                    uri=db_config_data.get("uri", ""),
                    database=db_config_data.get("name", self.namespace_name),
                )

            # Create DB manager with minimal config
            namespaces = {self.namespace_name: {self.namespace_name: db_config}}
            db_manager = DBManager(namespaces)

            # Get connector and test connection using built-in method
            connector = db_manager.get_conn(self.namespace_name, self.namespace_name)
            test_result = connector.test_connection()

            # Handle different return types from different connectors
            if isinstance(test_result, bool):
                return (test_result, "") if test_result else (False, "Connection test failed")
            elif isinstance(test_result, dict):
                success = test_result.get("success", False)
                error_msg = test_result.get("error", "Connection test failed") if not success else ""
                return success, error_msg
            else:
                return False, "Unknown connection test result format"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Database connectivity test failed: {error_msg}")
            return False, error_msg

    def _create_bootstrap_args(self, components: list, **kwargs):
        """Create common args namespace for bootstrap operations."""
        import argparse

        default_args = {
            "action": "bootstrap-kb",
            "namespace": self.namespace_name,
            "components": components,
            "kb_update_strategy": "overwrite",
            "storage_path": None,
            "benchmark": None,
            "schema_linking_type": "full",
            "database_name": "",
            "benchmark_path": None,
            "pool_size": 4,
            "config": None,
            "debug": False,
            "save_llm_trace": False,
        }

        # Update with any additional kwargs
        default_args.update(kwargs)

        return argparse.Namespace(**default_args)

    def _create_agent_with_config(self, args):
        """Create agent instance with loaded configuration."""
        from datus.agent.agent import Agent
        from datus.configuration.agent_config_loader import load_agent_config

        agent_config = load_agent_config()
        agent_config.current_namespace = self.namespace_name

        return Agent(args, agent_config)

    def _initialize_metadata(self) -> bool:
        """Initialize metadata knowledge base."""
        try:
            logger.info(
                f"Metadata initialization...{self.namespace_name},  {self.config['agent']['storage']['base_path']}"
            )
            args = self._create_bootstrap_args(["metadata"])
            agent = self._create_agent_with_config(args)
            result = agent.bootstrap_kb()
            # Log detailed results
            if isinstance(result, dict) and "message" in result:
                logger.info(f"Metadata bootstrap completed: {result['message']}")
            else:
                logger.info(f"Metadata bootstrap result: {result}")

            # Try to get table counts after bootstrap
            try:
                if hasattr(agent, "metadata_store") and agent.metadata_store:
                    schema_size = agent.metadata_store.get_schema_size()
                    value_size = agent.metadata_store.get_value_size()
                    logger.info(f"Bootstrap success: {schema_size} tables processed, {value_size} sample records")
                    console.print(f"  → Processed {schema_size} tables with {value_size} sample records")
            except Exception as count_e:
                logger.debug(f"Could not get table counts: {count_e}")

            return result is not False

        except Exception as e:
            logger.error(f"Metadata initialization failed: {e}")
            return False

    def _initialize_sql_history(self, sql_dir: str) -> int:
        """Initialize SQL history from specified directory."""
        try:
            logger.info(f"SQL history initialization...{self.namespace_name}, dir:{sql_dir}")
            # Count SQL files first
            sql_files = list(Path(sql_dir).rglob("*.sql"))
            if not sql_files:
                return 0

            args = self._create_bootstrap_args(["sql_history"], sql_dir=sql_dir, validate_only=False)
            agent = self._create_agent_with_config(args)
            result = agent.bootstrap_kb()

            # Log detailed results
            if isinstance(result, dict):
                if "message" in result:
                    logger.info(f"SQL history bootstrap completed: {result['message']}")
                if "processed_count" in result:
                    logger.info(f"Bootstrap success: {result['processed_count']} SQL files processed")
                elif "sql_count" in result:
                    logger.info(f"Bootstrap success: {result['sql_count']} SQL files processed")
            else:
                logger.info(f"SQL history bootstrap result: {result}")

            if result is not False:
                return len(sql_files)
            else:
                return 0

        except Exception as e:
            logger.error(f"SQL history initialization failed: {e}")
            return 0

    def _copy_files(self):
        copy_data_file(
            resource_path="tests/duckdb-demo.duckdb",
            target_dir=self.sample_dir,
        )

        copy_data_file(resource_path="prompts/prompt_template", target_dir=self.template_dir)


def main():
    """Entry point for the interactive init command."""
    try:
        init = InteractiveInit()
        return init.run()
    except Exception as e:
        console.print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error during interactive initialization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
