# -*- coding: utf-8 -*-
"""Generation hooks implementation for intercepting generation tool execution flow."""

import asyncio
import os

from agents.lifecycle import AgentHooks
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.cli.yaml_editor import edit_yaml_in_editor
from datus.configuration.agent_config import AgentConfig
from datus.utils.loggings import get_logger
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class GenerationCancelledException(Exception):
    """Exception raised when user cancels generation flow."""


@optional_traceable(name="GenerationHooks", run_type="chain")
class GenerationHooks(AgentHooks):
    """Hooks for handling generation tool results and user interaction."""

    def __init__(self, console: Console, agent_config: AgentConfig = None):
        """
        Initialize generation hooks.

        Args:
            console: Rich console for output
            agent_config: Agent configuration for storage access
        """
        self.console = console
        self.agent_config = agent_config

    async def on_agent_start(self, context, agent) -> None:
        logger.info("Generation agent start")

    async def on_start(self, context, agent) -> None:
        logger.info("Generation start")

    @optional_traceable(name="on_tool_end", run_type="chain")
    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Handle generation tool completion."""
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        # Debug: log all tool calls
        logger.info(f"=== TOOL END: {tool_name}, type: {type(result)} ===")
        logger.debug(f"Result content: {result}")

        # Intercept filesystem tools (native tools only)
        if tool_name in ["write_file", "edit_file"]:
            logger.info(f"=== FILESYSTEM TOOL DETECTED: {tool_name} ===")
            await self._handle_file_generation(tool_name, result)
        else:
            logger.debug(f"Tool '{tool_name}' not in target list [write_file, edit_file], skipping")

    async def on_tool_start(self, context, agent, tool) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_handoff(self, context, agent, source) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_agent_end(self, context, agent, output) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_end(self, context, agent, output) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()
        logger.info("Generation end")

    @optional_traceable(name="on_error", run_type="chain")
    async def on_error(self, context, agent, error) -> None:
        pass

    @optional_traceable(name="_handle_file_generation", run_type="chain")
    async def _handle_file_generation(self, tool_name: str, result):
        """
        Handle file generation (write_file/edit_file) result with user interaction.
        Supports native filesystem tools only.

        Args:
            tool_name: Name of the tool (write_file or edit_file)
            result: Tool result from filesystem tool
        """
        try:
            logger.info(f"=== HANDLE FILE GENERATION START: {tool_name} ===")
            # Extract file path from FuncToolResult object
            file_path = ""

            # Handle FuncToolResult object (native tools)
            if hasattr(result, "result") and hasattr(result, "success"):
                # Access FuncToolResult attributes directly
                result_value = result.result
                if isinstance(result_value, str) and ":" in result_value:
                    # Extract path from message like "File written successfully: /path/to/file.yml"
                    potential_path = result_value.split(":", 1)[1].strip()
                    if potential_path.endswith((".yml", ".yaml")):
                        file_path = potential_path
                        logger.info(f"=== EXTRACTED FILE PATH: {file_path} ===")

            logger.info(f"Extracted file_path: {file_path} from result type: {type(result)}")

            if not file_path:
                logger.warning(f"Could not extract file path from {tool_name} result: {result}")
                return

            # Only handle YAML files
            if not file_path.endswith((".yml", ".yaml")):
                logger.debug(f"Non-YAML file {file_path}, skipping interception")
                return

            # Read the file content that was just written
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist after {tool_name}")
                return

            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            if not yaml_content:
                logger.warning(f"Empty YAML content in {file_path}")
                return

            # Display generated YAML
            self.console.print("\n" + "=" * 60)
            self.console.print(f"[bold green]Generated YAML: {os.path.basename(file_path)}[/]")
            self.console.print(f"[dim]Path: {file_path}[/]")
            self.console.print("=" * 60)

            # Display YAML with syntax highlighting
            syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
            panel = Panel(syntax, title="Generated YAML", border_style="green")
            self.console.print(panel)

            # Get user action
            await self._get_user_action(yaml_content, file_path)

        except GenerationCancelledException:
            self.console.print("[yellow]Generation workflow cancelled[/]")
        except Exception as e:
            logger.error(f"Error handling file generation: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    async def _clear_output_and_show_prompt(self):
        """Clear output buffers and show user input prompt."""
        import sys

        # Ensure all pending output is flushed
        await asyncio.sleep(0.2)
        for _ in range(3):
            sys.stdout.flush()
            sys.stderr.flush()
            await asyncio.sleep(0.05)

        # Show clear separator
        self.console.print("\n" + "=" * 80)
        self.console.print("[bold red]⚠ USER INPUT REQUIRED ⚠[/]")
        self.console.print("=" * 80)

        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]CHOOSE ACTION:[/]")
        self.console.print("")
        self.console.print("  1. Accept - Save to RAG storage")
        self.console.print("  2. Edit - Modify YAML in editor")
        self.console.print("  3. Cancel - Discard changes")
        self.console.print("")

    async def _clear_output_and_show_edit_prompt(self):
        """Clear output buffers and show edit prompt."""
        import sys

        await asyncio.sleep(0.2)
        for _ in range(2):
            sys.stdout.flush()
            sys.stderr.flush()
            await asyncio.sleep(0.05)

        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]What next?[/]")
        self.console.print("  1. Save to RAG storage")
        self.console.print("  2. Edit again")
        self.console.print("  3. Cancel")

    @optional_traceable(name="_get_user_action", run_type="chain")
    async def _get_user_action(self, yaml_content: str, file_path: str):
        """
        Get user action on generated YAML with proper flow control.

        Args:
            yaml_content: Generated YAML content
            file_path: Path where YAML was saved
        """
        try:
            logger.info("=== Starting user interaction for YAML file ===")

            # Stop the live display if active
            live_was_stopped = execution_controller.stop_live_display()
            logger.info(f"=== Live display stopped: {live_was_stopped} ===")

            # Use execution control first, but don't suppress output yet
            async with execution_controller.pause_execution():
                logger.info("=== Execution paused, showing prompt ===")
                # Clear buffers and show prompt normally
                await self._clear_output_and_show_prompt()

                # Now suppress output ONLY during the actual input()
                def get_user_input():
                    logger.info("=== About to call input() ===")
                    result = blocking_input_manager.get_blocking_input(
                        lambda: input("Your choice (1-3) [1]: ").strip() or "1"
                    )
                    logger.info(f"=== Got user input: {result} ===")
                    return result

                choice = await execution_controller.request_user_input(get_user_input)
                logger.info(f"=== User choice processed: {choice} ===")

            if choice == "1":
                # Accept and sync
                await self._sync_to_storage(yaml_content, file_path)

            elif choice == "2":
                # Edit
                await self._handle_edit(yaml_content, file_path)

            elif choice == "3":
                # Cancel
                self.console.print("[yellow]Changes discarded[/]")
                raise GenerationCancelledException("User cancelled")

            else:
                self.console.print("[red]Invalid choice, please try again[/]")
                await self._get_user_action(yaml_content, file_path)

            # Resume live display after user interaction completes
            logger.info("=== Resuming live display ===")
            execution_controller.resume_live_display()

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Generation workflow cancelled[/]")
            # Resume live display even on error
            execution_controller.resume_live_display()
            raise GenerationCancelledException("User interrupted")
        except Exception as e:
            # Resume live display on any error
            execution_controller.resume_live_display()
            raise e

    @optional_traceable(name="_handle_edit", run_type="chain")
    async def _handle_edit(self, yaml_content: str, file_path: str):
        """
        Handle YAML editing flow.

        Args:
            yaml_content: Current YAML content
            file_path: File path
        """
        try:
            self.console.print("[cyan]Opening editor...[/]")
            loop = asyncio.get_event_loop()

            # Run editor in executor
            edited_content, confirmed = await loop.run_in_executor(None, edit_yaml_in_editor, yaml_content)

            if not confirmed:
                self.console.print("[yellow]Edit cancelled, returning to menu[/]")
                await self._get_user_action(yaml_content, file_path)
                return

            # Save edited content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(edited_content)

            self.console.print(f"[green]Changes saved to: {file_path}[/]")

            # Show diff summary
            original_lines = yaml_content.count("\n") + 1
            edited_lines = edited_content.count("\n") + 1
            self.console.print(f"[dim]Lines: {original_lines} → {edited_lines}[/]")

            # Ask to sync or re-edit
            await self._after_edit_action(edited_content, file_path)

        except Exception as e:
            logger.error(f"Error during edit: {e}")
            self.console.print(f"[red]Edit error: {e}[/]")
            await self._get_user_action(yaml_content, file_path)

    @optional_traceable(name="_after_edit_action", run_type="chain")
    async def _after_edit_action(self, edited_content: str, file_path: str):
        """
        Handle action after editing with proper flow control.

        Args:
            edited_content: Edited YAML content
            file_path: File path
        """
        # Use execution control but allow output for the prompt
        async with execution_controller.pause_execution():
            await self._clear_output_and_show_edit_prompt()

            def get_user_input():
                return blocking_input_manager.get_blocking_input(
                    lambda: input("Your choice (1-3) [1]: ").strip() or "1"
                )

            choice = await execution_controller.request_user_input(get_user_input)

        if choice == "1":
            await self._sync_to_storage(edited_content, file_path)
            # Resume live display after sync
            execution_controller.resume_live_display()
        elif choice == "2":
            await self._handle_edit(edited_content, file_path)
            # Live display will be resumed in nested calls
        elif choice == "3":
            self.console.print("[yellow]Changes saved to file but not synced to RAG[/]")
            # Resume live display before raising
            execution_controller.resume_live_display()
            raise GenerationCancelledException("User cancelled sync")
        else:
            self.console.print("[red]Invalid choice[/]")
            await self._after_edit_action(edited_content, file_path)

    @optional_traceable(name="_sync_to_storage", run_type="chain")
    async def _sync_to_storage(self, yaml_content: str, file_path: str):
        """
        Sync YAML content to RAG storage based on file type.

        Args:
            yaml_content: YAML content to sync
            file_path: File path
        """
        if not self.agent_config:
            self.console.print("[red]Agent configuration not available, cannot sync to RAG[/]")
            self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")
            return

        try:
            self.console.print("[cyan]Syncing to LanceDB...[/]")

            # Determine file type based on path and call appropriate sync method
            loop = asyncio.get_event_loop()

            if "semantic_model" in file_path or self._is_semantic_model_yaml(yaml_content):
                result = await loop.run_in_executor(None, self._sync_semantic_models_to_db, file_path)
                item_type = "semantic model"
            elif "metric" in file_path or self._is_metric_yaml(yaml_content):
                result = await loop.run_in_executor(None, self._sync_metrics_to_db, file_path)
                item_type = "metric"
            elif "sql_history" in file_path or self._is_sql_history_yaml(yaml_content):
                result = await loop.run_in_executor(None, self._sync_sql_history_to_db, file_path)
                item_type = "SQL history"
            else:
                self.console.print("[yellow]Unknown YAML type, cannot determine sync method[/]")
                self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")
                return

            if result.get("success"):
                self.console.print(f"[bold green]✓ Successfully synced {item_type} to LanceDB[/]")
                message = result.get("message", "")
                if message:
                    self.console.print(f"[dim]{message}[/]")
                self.console.print(f"[dim]File: {file_path}[/]")
            else:
                error = result.get("error", "Unknown error")
                self.console.print(f"[red]Sync failed: {error}[/]")
                self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")

        except Exception as e:
            logger.error(f"Error syncing to storage: {e}")
            self.console.print(f"[red]Sync error: {e}[/]")
            self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")

    def _is_semantic_model_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content is a semantic model (contains data_source)."""
        import yaml

        try:
            docs = list(yaml.safe_load_all(yaml_content))
            return any("data_source" in doc for doc in docs if doc)
        except Exception:
            return False

    def _is_metric_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content is a metric (contains metric)."""
        import yaml

        try:
            docs = list(yaml.safe_load_all(yaml_content))
            return any("metric" in doc for doc in docs if doc)
        except Exception:
            return False

    def _is_sql_history_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content is SQL history (contains sql_history or has id+sql+summary fields)."""
        import yaml

        try:
            doc = yaml.safe_load(yaml_content)
            if isinstance(doc, dict):
                # Check for explicit sql_history key
                if "sql_history" in doc:
                    return True
                # Check for characteristic fields of SQL history
                has_sql = "sql" in doc
                has_id = "id" in doc
                has_summary = "summary" in doc or "comment" in doc
                return has_sql and has_id and has_summary
            return False
        except Exception:
            return False

    def _sync_semantic_models_to_db(self, file_path: str) -> dict:
        """
        Sync semantic model YAML file to LanceDB.

        Args:
            file_path: Path to the semantic model YAML file

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            import json
            from datetime import datetime

            import yaml

            from datus.storage.metric.init_utils import gen_semantic_model_id
            from datus.storage.metric.store import rag_by_configuration

            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            data_source = None
            for doc in docs:
                if doc and "data_source" in doc:
                    data_source = doc["data_source"]
                    break

            if not data_source:
                return {"success": False, "error": "No data_source found in YAML file"}

            # Get database config
            current_db_config = self.agent_config.current_db_config()

            # Extract table name from sql_table or infer from data_source name
            table_name = data_source.get("name", "")
            if "sql_table" in data_source:
                # Parse table name from sql_table (e.g., "schema.table" or "table")
                sql_table = data_source["sql_table"]
                table_name = sql_table.split(".")[-1] if "." in sql_table else sql_table

            # Build semantic model dict
            semantic_model_dict = {
                "id": gen_semantic_model_id(
                    current_db_config.catalog, current_db_config.database, current_db_config.schema, table_name
                ),
                "catalog_name": current_db_config.catalog or "",
                "database_name": current_db_config.database or "",
                "schema_name": current_db_config.schema or "",
                "table_name": table_name,
                "catalog_database_schema": (
                    f"{current_db_config.catalog}_{current_db_config.database}_{current_db_config.schema}"
                ),
                "domain": "",  # Will be filled if available in config
                "layer1": "",
                "layer2": "",
                "semantic_file_path": file_path,
                "semantic_model_name": data_source.get("name", ""),
                "semantic_model_desc": data_source.get("description", ""),
                "identifiers": json.dumps(data_source.get("identifiers", []), ensure_ascii=False),
                "dimensions": json.dumps(data_source.get("dimensions", []), ensure_ascii=False),
                "measures": json.dumps(data_source.get("measures", []), ensure_ascii=False),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Store to LanceDB
            storage = rag_by_configuration(self.agent_config)
            storage.semantic_model_storage.store([semantic_model_dict])

            logger.info(f"Successfully synced semantic model {table_name} to LanceDB")
            return {"success": True, "message": f"Synced semantic model: {table_name}"}

        except Exception as e:
            logger.error(f"Error syncing semantic model to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _sync_metrics_to_db(self, file_path: str) -> dict:
        """
        Sync metrics YAML file to LanceDB.

        Args:
            file_path: Path to the metrics YAML file

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            from datetime import datetime

            import yaml

            from datus.storage.metric.init_utils import gen_metric_id
            from datus.storage.metric.store import rag_by_configuration

            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            metrics_list = []
            for doc in docs:
                if doc and "metric" in doc:
                    metrics_list.append(doc["metric"])

            if not metrics_list:
                return {"success": False, "error": "No metrics found in YAML file"}

            # Get storage
            storage = rag_by_configuration(self.agent_config)

            # Store each metric
            synced_count = 0
            for metric_doc in metrics_list:
                metric_name = metric_doc.get("name", "")
                metric_dict = {
                    "id": gen_metric_id("", "", "", "", metric_name),  # Simple ID generation
                    "semantic_model_name": "",
                    "domain": "",
                    "layer1": "",
                    "layer2": "",
                    "domain_layer1_layer2": "__",
                    "name": metric_name,
                    "description": metric_doc.get("description", ""),
                    "constraint": metric_doc.get("constraint", ""),
                    "sql_query": metric_doc.get("sql_query", ""),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                storage.metric_storage.store([metric_dict])
                synced_count += 1

            logger.info(f"Successfully synced {synced_count} metrics to LanceDB")
            return {"success": True, "message": f"Synced {synced_count} metric(s)"}

        except Exception as e:
            logger.error(f"Error syncing metrics to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _sync_sql_history_to_db(self, file_path: str) -> dict:
        """
        Sync SQL history YAML file to LanceDB.

        Args:
            file_path: Path to the SQL history YAML file

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            import yaml

            from datus.storage.sql_history.init_utils import gen_sql_history_id
            from datus.storage.sql_history.store import sql_history_rag_by_configuration

            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            # Extract sql_history data
            if "sql_history" in doc:
                sql_history_data = doc["sql_history"]
            elif isinstance(doc, dict) and "sql" in doc:
                # Direct format without sql_history wrapper
                sql_history_data = doc
            else:
                return {"success": False, "error": "No sql_history data found in YAML file"}

            # Generate ID if not present or if it's a placeholder
            sql_query = sql_history_data.get("sql", "")
            comment = sql_history_data.get("comment", "")
            item_id = sql_history_data.get("id", "")

            if not item_id or item_id == "auto_generated":
                item_id = gen_sql_history_id(sql_query, comment)
                sql_history_data["id"] = item_id

            # Ensure all required fields are present
            sql_history_dict = {
                "id": item_id,
                "name": sql_history_data.get("name", ""),
                "sql": sql_query,
                "comment": comment,
                "summary": sql_history_data.get("summary", ""),
                "filepath": sql_history_data.get("filepath", ""),
                "domain": sql_history_data.get("domain", ""),
                "layer1": sql_history_data.get("layer1", ""),
                "layer2": sql_history_data.get("layer2", ""),
                "tags": sql_history_data.get("tags", ""),
            }

            # Store to LanceDB
            storage = sql_history_rag_by_configuration(self.agent_config)
            storage.store_batch([sql_history_dict])

            logger.info(f"Successfully synced SQL history {item_id} to LanceDB")
            return {"success": True, "message": f"Synced SQL history: {sql_history_dict['name']}"}

        except Exception as e:
            logger.error(f"Error syncing SQL history to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
