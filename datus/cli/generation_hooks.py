# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Generation hooks implementation for intercepting generation tool execution flow."""

import asyncio
import os

from agents.lifecycle import AgentHooks
from rich.console import Console
from rich.syntax import Syntax

from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.llm_text_generator import generate_metric_llm_text
from datus.storage.sql_history import SqlHistoryRAG
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
        self.processed_files = set()  # Track files that have been processed to avoid duplicates
        logger.debug(f"GenerationHooks initialized with config: {agent_config is not None}")

    async def on_start(self, context, agent) -> None:
        pass

    @optional_traceable(name="on_tool_end", run_type="chain")
    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Handle generation tool completion."""
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))

        logger.debug(f"Tool end: {tool_name}, result type: {type(result)}")

        # Intercept end_generation tool (for semantic models and metrics)
        if tool_name == "end_generation":
            await self._handle_end_generation(result)
        # Intercept write_file tool and check if it's SQL summary
        elif tool_name == "write_file":
            # Check if this is a SQL summary file by examining tool arguments
            if self._is_sql_summary_tool_call(context):
                await self._handle_sql_summary_result(result)

    async def on_tool_start(self, context, agent, tool) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_handoff(self, context, agent, source) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_end(self, context, agent, output) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    @optional_traceable(name="_handle_end_generation", run_type="chain")
    async def _handle_end_generation(self, result):
        """
        Handle end_generation tool result with user interaction.

        Args:
            result: Tool result from end_generation
        """
        try:
            # Extract filepath from result (dict or FuncToolResult object)
            file_path = ""

            if isinstance(result, dict):
                # Handle dict result
                result_dict = result.get("result", {})
                if isinstance(result_dict, dict):
                    file_path = result_dict.get("filepath", "")
            elif hasattr(result, "result") and hasattr(result, "success"):
                # Handle FuncToolResult object
                result_dict = result.result
                if isinstance(result_dict, dict):
                    file_path = result_dict.get("filepath", "")

            logger.debug(f"Extracted file_path: {file_path}")

            if not file_path:
                logger.warning(f"Could not extract file path from end_generation result: {result}")
                return

            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            if not yaml_content:
                logger.warning(f"Empty YAML content in {file_path}")
                return

            # Skip processing if this file has already been processed
            if file_path in self.processed_files:
                logger.info(f"File {file_path} already processed, skipping end_generation")
                return

            # Mark file as processed
            self.processed_files.add(file_path)

            # Stop live display BEFORE showing YAML content
            execution_controller.stop_live_display()
            await asyncio.sleep(0.1)

            # Display generated YAML for all file types
            self.console.print("\n" + "=" * 60)
            self.console.print(f"[bold green]Generated YAML: {os.path.basename(file_path)}[/]")
            self.console.print(f"[dim]Path: {file_path}[/]")
            self.console.print("=" * 60)

            # Display YAML with syntax highlighting
            syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            await asyncio.sleep(0.2)

            # Get user confirmation to sync
            await self._get_sync_confirmation(yaml_content, file_path)

        except GenerationCancelledException:
            self.console.print("[yellow]Generation workflow cancelled[/]")
        except Exception as e:
            logger.error(f"Error handling end_generation: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    async def _clear_output_and_show_sync_prompt(self):
        """Show sync confirmation prompt."""
        import sys

        await asyncio.sleep(0.3)
        sys.stdout.flush()
        sys.stderr.flush()

        self.console.print("\n  [bold cyan]SYNC TO KNOWLEDGE BASE?[/]")
        self.console.print("")
        self.console.print("  [bold green]1.[/bold green] Yes - Save to Knowledge Base")
        self.console.print("  [bold yellow]2.[/bold yellow] No - Keep file only")
        self.console.print("")

    @optional_traceable(name="_handle_sql_summary_result", run_type="chain")
    async def _handle_sql_summary_result(self, result):
        """
        Handle sql_summary tool result.

        Args:
            result: Tool result from sql_summary
        """
        try:
            # Extract file path from result
            file_path = ""
            if isinstance(result, dict):
                result_msg = result.get("result", "")
                if "File written successfully" in str(result_msg) or "SQL history file written successfully" in str(
                    result_msg
                ):
                    parts = str(result_msg).split(": ")
                    if len(parts) > 1:
                        file_path = parts[-1].strip()
            elif hasattr(result, "result"):
                result_msg = result.result
                if "File written successfully" in str(result_msg) or "SQL history file written successfully" in str(
                    result_msg
                ):
                    parts = str(result_msg).split(": ")
                    if len(parts) > 1:
                        file_path = parts[-1].strip()

            logger.debug(f"Extracted file_path: {file_path}")

            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Could not extract or find file path from result: {result}")
                return

            # Skip processing if this file has already been processed
            if file_path in self.processed_files:
                logger.info(f"File {file_path} already processed, skipping write_file_reference_sql")
                return

            # Mark file as processed
            self.processed_files.add(file_path)

            # Read the file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    yaml_content = f.read()
            except Exception as read_error:
                logger.error(f"Failed to read file {file_path}: {read_error}")
                return

            if not yaml_content:
                logger.warning(f"Empty content in {file_path}")
                return

            # Stop live display BEFORE showing YAML content
            execution_controller.stop_live_display()
            await asyncio.sleep(0.1)

            # Display generated YAML with syntax highlighting
            self.console.print("\n" + "=" * 60)
            self.console.print("[bold green]Generated SQL History YAML[/]")
            self.console.print(f"[dim]File: {file_path}[/]")
            self.console.print("=" * 60)

            syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            await asyncio.sleep(0.2)

            # Get user confirmation to sync
            await self._get_sync_confirmation(yaml_content, file_path)

        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error handling write_file_reference_sql result: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    async def _get_sync_confirmation(self, yaml_content: str, file_path: str):
        """
        Get user confirmation to sync to Knowledge Base.

        Args:
            yaml_content: Generated YAML content
            file_path: Path where YAML was saved
        """
        try:
            # Stop the live display if active
            execution_controller.stop_live_display()

            # Use execution control to prevent output interference
            async with execution_controller.pause_execution():
                await self._clear_output_and_show_sync_prompt()

                self.console.print("[bold yellow]Please enter your choice:[/bold yellow] ", end="")

                def get_user_input():
                    return blocking_input_manager.get_blocking_input(lambda: input("[1/2] ").strip() or "1")

                choice = await execution_controller.request_user_input(get_user_input)

                if choice == "1":
                    # Sync to Knowledge Base
                    self.console.print("[bold green]✓ Syncing to Knowledge Base...[/]")
                    await self._sync_to_storage(yaml_content, file_path)
                elif choice == "2":
                    # Keep file only
                    self.console.print(f"[yellow]✓ YAML saved to file only: {file_path}[/]")
                else:
                    self.console.print("[red]✗ Invalid choice. Please enter 1 or 2.[/]")
                    self.console.print("[dim]Please try again...[/]\n")
                    await self._get_sync_confirmation(yaml_content, file_path)

            # Print completion separator to prevent action stream from overwriting
            self.console.print("\n" + "=" * 80)
            self.console.print("[bold green]✓ Generation workflow completed, generating report...[/]", justify="center")
            self.console.print("=" * 80 + "\n")

            # Add delay to ensure message is visible before any new output
            await asyncio.sleep(0.1)

        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]✗ Sync cancelled by user[/]")
            raise GenerationCancelledException("User interrupted")
        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error in sync confirmation: {e}", exc_info=True)
            raise e

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
            # Determine file type based on path and call appropriate sync method
            loop = asyncio.get_event_loop()

            if self._is_semantic_yaml(yaml_content):
                result = await loop.run_in_executor(None, self._sync_semantic_to_db, file_path)
                item_type = "semantic model and metrics"
            elif self._is_sql_history_yaml(yaml_content):
                result = await loop.run_in_executor(None, self._sync_sql_history_to_db, file_path)
                item_type = "SQL history"
            else:
                self.console.print("[yellow]Unknown YAML type, cannot determine sync method[/]")
                self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")
                return

            if result.get("success"):
                self.console.print(f"[bold green]✓ Successfully synced {item_type} to Knowledge Base[/]")
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

    def _is_sql_summary_tool_call(self, context) -> bool:
        """
        Check if write_file tool call is for SQL summary.

        Examines tool arguments to determine if this is a SQL summary file write.

        Args:
            context: ToolContext with tool_arguments field (JSON string)

        Returns:
            bool: True if this is a SQL summary write operation
        """
        try:
            import json

            if hasattr(context, "tool_arguments"):
                if context.tool_arguments:
                    tool_args = json.loads(context.tool_arguments)

                    # Check if file_type indicates SQL summary
                    if isinstance(tool_args, dict):
                        if tool_args.get("file_type") == "sql_summary":
                            logger.debug(f"Detected SQL summary write_file call with args: {tool_args}")
                            return True

            logger.debug("Not a SQL summary write_file call")
            return False

        except Exception as e:
            logger.debug(f"Error checking tool arguments: {e}")
            return False

    def _is_semantic_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content contains semantic model (data_source) or metrics."""
        import yaml

        try:
            docs = list(yaml.safe_load_all(yaml_content))
            has_data_source = any("data_source" in doc for doc in docs if doc)
            has_metric = any("metric" in doc for doc in docs if doc)
            return has_data_source or has_metric
        except Exception:
            return False

    def _is_sql_history_yaml(self, yaml_content: str) -> bool:
        """Check if YAML content is Reference SQL (contains reference_sql or has id+sql+summary fields)."""
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

    def _sync_semantic_to_db(self, file_path: str) -> dict:
        """
        Sync semantic model and metrics from YAML file to Knowledge Base.

        This function handles both data_source (semantic model) and metric definitions
        in the same file. It checks for existing entries and only stores new ones.

        Args:
            file_path: Path to the YAML file containing semantic model and/or metrics

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            import json
            from datetime import datetime

            import yaml

            from datus.configuration.agent_config import MetricMeta
            from datus.storage.metric.init_utils import exists_semantic_metrics, gen_metric_id, gen_semantic_model_id
            from datus.storage.metric.store import SemanticMetricsRAG

            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            # Extract data_source and metrics
            data_source = None
            metrics_list = []
            for doc in docs:
                if doc and "data_source" in doc:
                    data_source = doc["data_source"]
                elif doc and "metric" in doc:
                    metrics_list.append(doc["metric"])

            if not data_source and not metrics_list:
                return {"success": False, "error": "No data_source or metrics found in YAML file"}

            # Get storage
            storage = SemanticMetricsRAG(self.agent_config)

            # Get existing semantic models and metrics
            existing_semantic_models, existing_metrics = exists_semantic_metrics(storage, build_mode="incremental")

            # Get database config
            current_db_config = self.agent_config.current_db_config()

            # Get domain/layer info - use default MetricMeta if not configured
            if hasattr(self.agent_config, "metric_meta") and self.agent_config.metric_meta:
                # Use the first available metric_meta
                first_meta_name = next(iter(self.agent_config.metric_meta.keys()))
                current_metric_meta = self.agent_config.metric_meta[first_meta_name]
            else:
                # Use default values
                current_metric_meta = MetricMeta()

            domain = current_metric_meta.domain
            layer1 = current_metric_meta.layer1
            layer2 = current_metric_meta.layer2

            synced_count = 0
            skipped_count = 0
            message_parts = []

            # Process semantic model (data_source)
            if data_source:
                # Extract table name from sql_table or infer from data_source name
                table_name = data_source.get("name", "")
                if "sql_table" in data_source:
                    # Parse table name from sql_table (e.g., "schema.table" or "table")
                    sql_table = data_source["sql_table"]
                    table_name = sql_table.split(".")[-1] if "." in sql_table else sql_table

                semantic_model_id = gen_semantic_model_id(
                    current_db_config.catalog, current_db_config.database, current_db_config.schema, table_name
                )

                # Check if semantic model already exists
                if semantic_model_id not in existing_semantic_models:
                    # Build semantic model dict
                    semantic_model_dict = {
                        "id": semantic_model_id,
                        "catalog_name": current_db_config.catalog or "",
                        "database_name": current_db_config.database or "",
                        "schema_name": current_db_config.schema or "",
                        "table_name": table_name,
                        "domain": domain,
                        "layer1": layer1,
                        "layer2": layer2,
                        "semantic_file_path": file_path,
                        "semantic_model_name": data_source.get("name", ""),
                        "semantic_model_desc": data_source.get("description", ""),
                        "identifiers": json.dumps(data_source.get("identifiers", []), ensure_ascii=False),
                        "dimensions": json.dumps(data_source.get("dimensions", []), ensure_ascii=False),
                        "measures": json.dumps(data_source.get("measures", []), ensure_ascii=False),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    storage.semantic_model_storage.store([semantic_model_dict])
                    existing_semantic_models.add(semantic_model_id)
                    synced_count += 1
                    logger.info(f"Synced semantic model: {table_name}")
                    message_parts.append(f"semantic model '{table_name}'")
                else:
                    skipped_count += 1
                    logger.info(f"Semantic model {table_name} already exists, skipped")

            # Process metrics
            semantic_model_name = data_source.get("name", "") if data_source else ""
            for metric_doc in metrics_list:
                metric_name = metric_doc.get("name", "")
                metric_id = gen_metric_id(domain, layer1, layer2, semantic_model_name, metric_name)

                # Check if metric already exists
                if metric_id not in existing_metrics:
                    # Generate LLM-friendly text
                    llm_text = generate_metric_llm_text(metric_doc, data_source)

                    metric_dict = {
                        "id": metric_id,
                        "semantic_model_name": semantic_model_name,
                        "domain": domain,
                        "layer1": layer1,
                        "layer2": layer2,
                        "name": metric_name,
                        "llm_text": llm_text,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    storage.metric_storage.store([metric_dict])
                    existing_metrics.add(metric_id)
                    synced_count += 1
                    logger.info(f"Synced metric: {metric_name}")
                    message_parts.append(f"metric '{metric_name}'")
                else:
                    skipped_count += 1
                    logger.info(f"Metric {metric_name} already exists, skipped")

            # Build result message
            if synced_count > 0:
                message = f"Synced {synced_count} item(s): {', '.join(message_parts)}"
                if skipped_count > 0:
                    message += f" (skipped {skipped_count} existing item(s))"
                return {"success": True, "message": message}
            elif skipped_count > 0:
                return {"success": True, "message": f"All {skipped_count} item(s) already exist, nothing to sync"}
            else:
                return {"success": False, "error": "No items to sync"}

        except Exception as e:
            logger.error(f"Error syncing semantic model and metrics to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _sync_sql_history_to_db(self, file_path: str) -> dict:
        """
        Sync SQL history YAML file to Knowledge Base.

        Args:
            file_path: Path to the SQL history YAML file

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            import yaml

            from datus.storage.sql_history.init_utils import exists_sql_history, gen_sql_history_id

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
                return {"success": False, "error": "No reference_sql data found in YAML file"}

            # Generate ID if not present or if it's a placeholder
            sql_query = sql_history_data.get("sql", "")
            comment = sql_history_data.get("comment", "")
            item_id = sql_history_data.get("id", "")

            if not item_id or item_id == "auto_generated":
                item_id = gen_sql_history_id(sql_query, comment)
                sql_history_data["id"] = item_id

            # Get storage and check if item already exists
            storage = SqlHistoryRAG(self.agent_config)
            existing_ids = exists_sql_history(storage, build_mode="incremental")

            # Check for duplicate
            if item_id in existing_ids:
                logger.info(f"Reference SQL {item_id} already exists in Knowledge Base, skipping")
                return {
                    "success": True,
                    "message": f"Reference SQL '{sql_history_data.get('name', '')}' already exists, skipped",
                }

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

            # Store to Knowledge Base
            storage.store_batch([sql_history_dict])

            logger.info(f"Successfully synced SQL history {item_id} to Knowledge Base")
            return {"success": True, "message": f"Synced SQL history: {sql_history_dict['name']}"}

        except Exception as e:
            logger.error(f"Error syncing SQL history to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
