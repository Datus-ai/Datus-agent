# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

# -*- coding: utf-8 -*-
"""Generation hooks implementation for intercepting generation tool execution flow."""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional

import yaml
from agents.lifecycle import AgentHooks
from rich.console import Console
from rich.syntax import Syntax

from datus.cli.blocking_input_manager import blocking_input_manager
from datus.cli.execution_state import execution_controller
from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.utils.constants import DBType
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

        # Intercept semantic model generation completion
        if tool_name == "end_semantic_model_generation":
            await self._handle_end_semantic_model_generation(result)
        # Intercept metric generation completion
        elif tool_name == "end_metric_generation":
            await self._handle_end_metric_generation(result)
        # Intercept write_file tool and check if it's SQL summary
        elif tool_name == "write_file":
            # Check if this is a SQL summary file by examining tool arguments
            if self._is_sql_summary_tool_call(context):
                await self._handle_sql_summary_result(result)
            # Check if this is an external knowledge file
            elif self._is_ext_knowledge_tool_call(context):
                await self._handle_ext_knowledge_result(result)

    async def on_tool_start(self, context, agent, tool) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_handoff(self, context, agent, source) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    async def on_end(self, context, agent, output) -> None:
        # Wait if execution is paused
        await execution_controller.wait_for_resume()

    @optional_traceable(name="_handle_end_semantic_model_generation", run_type="chain")
    async def _handle_end_semantic_model_generation(self, result):
        """
        Handle end_semantic_model_generation tool result.

        Args:
            result: Tool result containing semantic_model_files list
        """
        try:
            file_paths = self._extract_filepaths_from_result(result)

            if not file_paths:
                logger.warning(f"Could not extract file paths from end_semantic_model_generation result: {result}")
                return

            logger.debug(f"Processing semantic model files: {file_paths}")

            # Process each semantic model file
            for file_path in file_paths:
                await self._process_single_file(file_path)

        except GenerationCancelledException:
            self.console.print("[yellow]Generation workflow cancelled[/]")
        except Exception as e:
            logger.error(f"Error handling end_semantic_model_generation: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    @optional_traceable(name="_handle_end_metric_generation", run_type="chain")
    async def _handle_end_metric_generation(self, result):
        """
        Handle end_metric_generation tool result.

        Args:
            result: Tool result containing metric_file, optional semantic_model_file, and metric_sqls
        """
        try:
            metric_file, semantic_model_file, metric_sqls = self._extract_metric_generation_result(result)

            if not metric_file:
                logger.warning(f"Could not extract metric_file from end_metric_generation result: {result}")
                return

            logger.debug(
                f"Processing metric generation: metric_file={metric_file}, "
                f"semantic_model_file={semantic_model_file}, metric_sqls={list(metric_sqls.keys())}"
            )

            if semantic_model_file:
                # Process both files together for proper association
                await self._process_metric_with_semantic_model(semantic_model_file, metric_file, metric_sqls)
            else:
                # Process metric file alone (semantic model already exists in KB)
                await self._process_single_file(metric_file, metric_sqls=metric_sqls)

        except GenerationCancelledException:
            self.console.print("[yellow]Generation workflow cancelled[/]")
        except Exception as e:
            logger.error(f"Error handling end_metric_generation: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    def _extract_filepaths_from_result(self, result) -> list:
        """
        Extract semantic_model_files list from tool result.

        Args:
            result: Tool result (dict or FuncToolResult object)

        Returns:
            List of file paths
        """
        result_dict = None
        if isinstance(result, dict):
            result_dict = result.get("result", {})
        elif hasattr(result, "result") and hasattr(result, "success"):
            result_dict = result.result

        if isinstance(result_dict, dict):
            filepaths = result_dict.get("semantic_model_files", [])
            if filepaths and isinstance(filepaths, list):
                return filepaths

        return []

    def _extract_metric_generation_result(self, result) -> tuple:
        """
        Extract metric_file, semantic_model_file, and metric_sqls from tool result.

        Args:
            result: Tool result (dict or FuncToolResult object)

        Returns:
            Tuple of (metric_file, semantic_model_file, metric_sqls)
        """
        # Debug: log raw result type and content
        logger.info(f"_extract_metric_generation_result raw result: type={type(result).__name__}, value={result}")

        result_dict = None
        if isinstance(result, dict):
            result_dict = result.get("result", {})
        elif hasattr(result, "result") and hasattr(result, "success"):
            result_dict = result.result

        if isinstance(result_dict, dict):
            metric_file = result_dict.get("metric_file", "")
            semantic_model_file = result_dict.get("semantic_model_file", "")
            metric_sqls = result_dict.get("metric_sqls", {})
            logger.info(f"Extracted from end_metric_generation: metric_sqls={metric_sqls}")
            return metric_file, semantic_model_file, metric_sqls

        logger.warning(f"Could not extract metric_generation_result from: {result}")
        return "", "", {}

    async def _process_single_file(self, file_path: str, metric_sqls: dict = None):
        """
        Process a single YAML file: display and get user confirmation.

        Args:
            file_path: Path to the YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
        """
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
            logger.info(f"File {file_path} already processed, skipping")
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
        await self._get_sync_confirmation(yaml_content, file_path, "semantic", metric_sqls=metric_sqls)

    async def _process_metric_with_semantic_model(
        self, semantic_model_file: str, metric_file: str, metric_sqls: dict = None
    ):
        """
        Process metric file along with its semantic model file.
        Display both files and sync them together so metrics can reference semantic model data.

        Args:
            semantic_model_file: Path to the semantic model YAML file
            metric_file: Path to the metric YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
        """
        # Check if files exist
        if not os.path.exists(semantic_model_file):
            logger.warning(f"Semantic model file {semantic_model_file} does not exist")
            # Still try to process metric file alone
            if os.path.exists(metric_file):
                await self._process_single_file(metric_file, metric_sqls=metric_sqls)
            return

        if not os.path.exists(metric_file):
            logger.warning(f"Metric file {metric_file} does not exist")
            # Still try to process semantic model file alone
            await self._process_single_file(semantic_model_file)
            return

        # Skip if both files have already been processed
        if semantic_model_file in self.processed_files and metric_file in self.processed_files:
            logger.info("Both files already processed, skipping")
            return

        # Mark both files as processed
        self.processed_files.add(semantic_model_file)
        self.processed_files.add(metric_file)

        # Read both files
        with open(semantic_model_file, "r", encoding="utf-8") as f:
            semantic_content = f.read()
        with open(metric_file, "r", encoding="utf-8") as f:
            metric_content = f.read()

        if not semantic_content or not metric_content:
            logger.warning("Empty content in semantic model or metric file")
            return

        # Stop live display BEFORE showing YAML content
        execution_controller.stop_live_display()
        await asyncio.sleep(0.1)

        # Display both files
        self.console.print("\n" + "=" * 60)
        self.console.print(f"[bold green]Generated Semantic Model: {os.path.basename(semantic_model_file)}[/]")
        self.console.print(f"[dim]Path: {semantic_model_file}[/]")
        self.console.print("=" * 60)

        syntax = Syntax(semantic_content, "yaml", theme="monokai", line_numbers=True)
        self.console.print(syntax)
        await asyncio.sleep(0.2)

        self.console.print("\n" + "=" * 60)
        self.console.print(f"[bold green]Generated Metric: {os.path.basename(metric_file)}[/]")
        self.console.print(f"[dim]Path: {metric_file}[/]")
        self.console.print("=" * 60)

        syntax = Syntax(metric_content, "yaml", theme="monokai", line_numbers=True)
        self.console.print(syntax)
        await asyncio.sleep(0.2)

        # Get user confirmation to sync both files together
        await self._get_sync_confirmation_for_pair(semantic_model_file, metric_file, metric_sqls)

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
                if "File written successfully" in str(result_msg) or "Reference SQL file written successfully" in str(
                    result_msg
                ):
                    parts = str(result_msg).split(": ")
                    if len(parts) > 1:
                        file_path = parts[-1].strip()
            elif hasattr(result, "result"):
                result_msg = result.result
                if "File written successfully" in str(result_msg) or "Reference SQL file written successfully" in str(
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
            self.console.print("[bold green]Generated Reference SQL YAML[/]")
            self.console.print(f"[dim]File: {file_path}[/]")
            self.console.print("=" * 60)

            syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            await asyncio.sleep(0.2)

            # Get user confirmation to sync (this is for SQL summary)
            await self._get_sync_confirmation(yaml_content, file_path, "sql_summary")

        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error handling write_file_reference_sql result: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    @optional_traceable(name="_handle_ext_knowledge_result", run_type="chain")
    async def _handle_ext_knowledge_result(self, result):
        """
        Handle ext_knowledge tool result.

        Args:
            result: Tool result from ext_knowledge
        """
        try:
            # Extract file path from result
            file_path = ""
            if isinstance(result, dict):
                result_msg = result.get("result", "")
                if "File written successfully" in str(
                    result_msg
                ) or "External knowledge file written successfully" in str(result_msg):
                    parts = str(result_msg).split(": ")
                    if len(parts) > 1:
                        file_path = parts[-1].strip()
            elif hasattr(result, "result"):
                result_msg = result.result
                if "File written successfully" in str(
                    result_msg
                ) or "External knowledge file written successfully" in str(result_msg):
                    parts = str(result_msg).split(": ")
                    if len(parts) > 1:
                        file_path = parts[-1].strip()

            logger.debug(f"Extracted file_path: {file_path}")

            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Could not extract or find file path from result: {result}")
                return

            # Skip processing if this file has already been processed
            if file_path in self.processed_files:
                logger.info(f"File {file_path} already processed, skipping write_file_ext_knowledge")
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
            self.console.print("[bold green]Generated External Knowledge YAML[/]")
            self.console.print(f"[dim]File: {file_path}[/]")
            self.console.print("=" * 60)

            syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            await asyncio.sleep(0.2)

            # Get user confirmation to sync (this is for external knowledge)
            await self._get_sync_confirmation(yaml_content, file_path, "ext_knowledge")

        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error handling write_file_ext_knowledge result: {e}", exc_info=True)
            self.console.print(f"[red]Error: {e}[/]")

    async def _get_sync_confirmation_for_pair(
        self, semantic_model_file: str, metric_file: str, metric_sqls: dict = None
    ):
        """
        Get user confirmation to sync semantic model and metric files together to Knowledge Base.

        Args:
            semantic_model_file: Path to semantic model YAML file
            metric_file: Path to metric YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
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
                    # Sync both files to Knowledge Base
                    self.console.print("[bold green]✓ Syncing to Knowledge Base...[/]")
                    await self._sync_semantic_and_metric(semantic_model_file, metric_file, metric_sqls)
                elif choice == "2":
                    # Keep files only
                    self.console.print("[yellow]✓ YAMLs saved to files only:[/]")
                    self.console.print(f"  - {semantic_model_file}")
                    self.console.print(f"  - {metric_file}")
                else:
                    self.console.print("[red]✗ Invalid choice. Please enter 1 or 2.[/]")
                    self.console.print("[dim]Please try again...[/]\n")
                    await self._get_sync_confirmation_for_pair(semantic_model_file, metric_file, metric_sqls)

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
            raise

    async def _get_sync_confirmation(self, yaml_content: str, file_path: str, yaml_type: str, metric_sqls: dict = None):
        """
        Get user confirmation to sync to Knowledge Base.

        Args:
            yaml_content: Generated YAML content
            file_path: Path where YAML was saved
            yaml_type: YAML type - "semantic" or "sql_summary"
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
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
                    await self._sync_to_storage(file_path, yaml_type, metric_sqls=metric_sqls)
                elif choice == "2":
                    # Keep file only
                    self.console.print(f"[yellow]✓ YAML saved to file only: {file_path}[/]")
                else:
                    self.console.print("[red]✗ Invalid choice. Please enter 1 or 2.[/]")
                    self.console.print("[dim]Please try again...[/]\n")
                    await self._get_sync_confirmation(yaml_content, file_path, yaml_type, metric_sqls=metric_sqls)

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
            raise

    @optional_traceable(name="_sync_to_storage", run_type="chain")
    async def _sync_to_storage(self, file_path: str, yaml_type: str, metric_sqls: dict = None):
        """
        Sync YAML file to RAG storage based on file type.

        Args:
            file_path: File path to sync
            yaml_type: YAML type - "semantic" or "sql_summary"
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
        """
        if not self.agent_config:
            self.console.print("[red]Agent configuration not available, cannot sync to RAG[/]")
            self.console.print(f"[yellow]YAML saved to file: {file_path}[/]")
            return

        try:
            # Sync based on yaml_type
            loop = asyncio.get_event_loop()

            if yaml_type == "semantic":
                result = await loop.run_in_executor(
                    None,
                    lambda: GenerationHooks._sync_semantic_to_db(file_path, self.agent_config, metric_sqls=metric_sqls),
                )
                item_type = "semantic model"
            elif yaml_type == "sql_summary":
                result = await loop.run_in_executor(
                    None, GenerationHooks._sync_reference_sql_to_db, file_path, self.agent_config
                )
                item_type = "reference SQL"
            elif yaml_type == "ext_knowledge":
                result = await loop.run_in_executor(
                    None, GenerationHooks._sync_ext_knowledge_to_db, file_path, self.agent_config, "incremental"
                )
                item_type = "external knowledge"
            else:
                self.console.print(
                    f"[red]Invalid yaml_type: {yaml_type}. Expected 'semantic', 'sql_summary', or 'ext_knowledge'[/]"
                )
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

    @optional_traceable(name="_sync_semantic_and_metric", run_type="chain")
    async def _sync_semantic_and_metric(self, semantic_model_file: str, metric_file: str, metric_sqls: dict = None):
        """
        Sync both semantic model and metric files to RAG storage.
        Creates a combined YAML for syncing so metrics can reference semantic model data.

        Args:
            semantic_model_file: Path to semantic model YAML file
            metric_file: Path to metric YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
        """
        if not self.agent_config:
            self.console.print("[red]Agent configuration not available, cannot sync to RAG[/]")
            self.console.print("[yellow]YAMLs saved to files:[/]")
            self.console.print(f"  - {semantic_model_file}")
            self.console.print(f"  - {metric_file}")
            return

        try:
            loop = asyncio.get_event_loop()

            # Load both YAML files
            with open(semantic_model_file, "r", encoding="utf-8") as f:
                semantic_docs = list(yaml.safe_load_all(f))
            with open(metric_file, "r", encoding="utf-8") as f:
                metric_docs = list(yaml.safe_load_all(f))

            # Create a temporary combined YAML content
            combined_docs = semantic_docs + metric_docs
            temp_file = semantic_model_file + ".combined.tmp"

            try:
                # Write combined YAML to temp file
                with open(temp_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump_all(combined_docs, f, allow_unicode=True, sort_keys=False)

                # Sync the combined file - only sync metrics, not semantic objects (avoid duplicates)
                result = await loop.run_in_executor(
                    None,
                    lambda: GenerationHooks._sync_semantic_to_db(
                        temp_file,
                        self.agent_config,
                        include_semantic_objects=False,  # Semantic model already synced separately
                        include_metrics=True,
                        metric_sqls=metric_sqls,
                        original_yaml_path=metric_file,  # Use original metric file path, not temp file
                    ),
                )

                if result.get("success"):
                    self.console.print(
                        "[bold green]✓ Successfully synced semantic model and metrics to Knowledge Base[/]"
                    )
                    message = result.get("message", "")
                    if message:
                        self.console.print(f"[dim]{message}[/]")
                    self.console.print("[dim]Files:[/]")
                    self.console.print(f"  - {semantic_model_file}")
                    self.console.print(f"  - {metric_file}")
                else:
                    error = result.get("error", "Unknown error")
                    self.console.print(f"[red]Sync failed: {error}[/]")
                    self.console.print("[yellow]YAMLs saved to files:[/]")
                    self.console.print(f"  - {semantic_model_file}")
                    self.console.print(f"  - {metric_file}")

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error syncing semantic and metric: {e}", exc_info=True)
            self.console.print(f"[red]Sync error: {e}[/]")
            self.console.print("[yellow]YAMLs saved to files:[/]")
            self.console.print(f"  - {semantic_model_file}")
            self.console.print(f"  - {metric_file}")

    def _is_sql_summary_tool_call(self, context) -> bool:
        """
        Check if write_file tool call is for SQL summary.
        """
        try:
            if hasattr(context, "tool_arguments"):
                if context.tool_arguments:
                    tool_args = json.loads(context.tool_arguments)
                    if isinstance(tool_args, dict):
                        if tool_args.get("file_type") == "sql_summary":
                            logger.debug(f"Detected SQL summary write_file call with args: {tool_args}")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Error checking tool arguments: {e}")
            return False

    def _is_ext_knowledge_tool_call(self, context) -> bool:
        """
        Check if write_file tool call is for external knowledge.

        Examines tool arguments to determine if this is an external knowledge file write.

        Args:
            context: ToolContext with tool_arguments field (JSON string)

        Returns:
            bool: True if this is an external knowledge write operation
        """
        try:
            import json

            if hasattr(context, "tool_arguments"):
                if context.tool_arguments:
                    tool_args = json.loads(context.tool_arguments)

                    # Check if file_type indicates external knowledge
                    if isinstance(tool_args, dict):
                        if tool_args.get("file_type") == "ext_knowledge":
                            logger.debug(f"Detected external knowledge write_file call with args: {tool_args}")
                            return True

            logger.debug("Not an external knowledge write_file call")
            return False

        except Exception as e:
            logger.debug(f"Error checking tool arguments: {e}")
            return False

    @staticmethod
    def _parse_subject_tree_from_tags(tags_list) -> Optional[list]:
        """
        Parse subject_path from metric tags.

        Looks for tag format: "subject_tree: path/component1/component2/..."

        Args:
            tags_list: List of tags from locked_metadata.tags

        Returns:
            List[str]: Subject path components or None if not found
        """
        if not tags_list or not isinstance(tags_list, list):
            return None

        for tag in tags_list:
            if isinstance(tag, str) and tag.startswith("subject_tree:"):
                # Extract the path after "subject_tree: "
                path = tag.split("subject_tree:", 1)[1].strip()
                parts = [part.strip() for part in path.split("/") if part.strip()]
                if parts:
                    return parts
                else:
                    logger.warning(f"Invalid subject_tree format: {tag}, expected 'subject_tree: path/component1/...'")

        return None

    @staticmethod
    def _sync_semantic_to_db(
        file_path: str,
        agent_config: AgentConfig,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        include_semantic_objects: bool = True,
        include_metrics: bool = True,
        metric_sqls: dict = None,
        original_yaml_path: Optional[str] = None,
    ) -> dict:
        """
        Sync semantic objects and/or metrics from YAML file to Knowledge Base.

        Args:
            file_path: Path to YAML file
            agent_config: Agent configuration
            include_semantic_objects: Whether to sync tables/columns/entities
            include_metrics: Whether to sync metrics
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
            original_yaml_path: Original YAML file path to store
                (if different from file_path, e.g., when using temp files)

        Now parses tables, columns, metrics, and entities as individual 'semantic_objects'.
        """
        # Use original_yaml_path if provided, otherwise use file_path
        yaml_path_to_store = original_yaml_path if original_yaml_path else file_path
        try:
            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            data_source = None
            metrics_list = []
            for doc in docs:
                if doc and "data_source" in doc:
                    data_source = doc["data_source"]
                elif doc and "metric" in doc:
                    metrics_list.append(doc["metric"])

            if not data_source and not metrics_list:
                return {"success": False, "error": "No data_source or metrics found in YAML file"}

            metric_rag = MetricRAG(agent_config)
            semantic_rag = SemanticModelRAG(agent_config)

            semantic_objects = []  # For tables, columns (goes to SemanticModelStorage)
            metric_objects = []  # For metrics (goes to MetricStorage)
            synced_items = []

            current_db_config = agent_config.current_db_config()
            table_name = ""

            # Get database hierarchy info
            # Prioritize explicitly passed parameters, then fallback to current db config
            catalog_name = catalog or getattr(current_db_config, "catalog", "")
            database_name = database or getattr(current_db_config, "database", "")
            schema_name = schema or getattr(current_db_config, "schema", "")

            # For StarRocks, use default_catalog if it's empty
            if agent_config.db_type == DBType.STARROCKS and not catalog_name:
                catalog_name = "default_catalog"

            # 1. Parse table context from data_source (always, for metric association)
            # Decoupled from include_semantic_objects to ensure metrics get proper table context
            table_fq_name = ""
            if data_source:
                table_name = data_source.get("name", "")
                sql_table = data_source.get("sql_table", "")

                # Try to parse hierarchy from sql_table if it's fully qualified
                if sql_table:
                    parts = [p.strip() for p in sql_table.split(".") if p.strip()]
                    if len(parts) > 0:
                        table_name = parts[-1]

                        # Replicate DBFuncTool._determine_field_order logic for parsing
                        dialect = agent_config.db_type
                        possible_fields = []
                        if DBType.support_catalog(dialect):
                            possible_fields.append("catalog")
                        if DBType.support_database(dialect) or dialect == DBType.SQLITE:
                            possible_fields.append("database")
                        if DBType.support_schema(dialect):
                            possible_fields.append("schema")

                        # Assign parts from right to left (excluding the table name itself)
                        idx = len(parts) - 2
                        for field in reversed(possible_fields):
                            if idx < 0:
                                break
                            if field == "schema":
                                schema_name = parts[idx]
                            elif field == "database":
                                database_name = parts[idx]
                            elif field == "catalog":
                                catalog_name = parts[idx]
                            idx -= 1

                # Clear schema_name if dialect doesn't support it (e.g. StarRocks, MySQL)
                if not DBType.support_schema(agent_config.db_type):
                    schema_name = ""

                # Build fully qualified name (excluding empty parts)
                fq_parts = [p for p in [catalog_name, database_name, schema_name, table_name] if p]
                table_fq_name = ".".join(fq_parts)

            # 2. Create and store semantic objects (table/columns) only when requested
            if data_source and include_semantic_objects:
                # --- A. Table Object ---
                table_obj = {
                    "id": f"table:{table_name}",
                    "kind": "table",
                    "name": table_name,
                    "fq_name": table_fq_name,
                    "table_name": table_name,
                    "description": data_source.get("description", ""),
                    "yaml_path": yaml_path_to_store,
                    "updated_at": datetime.now().replace(microsecond=0),
                    # Database hierarchy
                    "catalog_name": catalog_name,
                    "database_name": database_name,
                    "schema_name": schema_name,
                    "semantic_model_name": table_name,
                    # Required boolean fields
                    "is_dimension": False,
                    "is_measure": False,
                    "is_entity_key": False,
                    "is_deprecated": False,
                }
                semantic_objects.append(table_obj)
                synced_items.append(f"table:{table_name}")

                # --- B. Column Objects (Measures & Dimensions & Identifiers) ---

                # Helper to process columns
                def process_column(col_def, is_dim=False, is_meas=False, is_ent=False):
                    col_name = col_def.get("name")
                    if not col_name:
                        return

                    col_desc = col_def.get("description", "")
                    # Strip backticks from expr since YAML may contain quoted column names like `County Name`
                    raw_expr = col_def.get("expr", col_name)  # Default to column name if no expr
                    col_expr = raw_expr.strip("`") if raw_expr else raw_expr

                    # Extract time_granularity from type_params for TIME dimensions
                    time_granularity = ""
                    if is_dim and col_def.get("type") == "TIME":
                        type_params = col_def.get("type_params", {})
                        time_granularity = type_params.get("time_granularity", "")

                    col_obj = {
                        "id": f"column:{table_name}.{col_name}",
                        "kind": "column",
                        "name": col_name,
                        "fq_name": f"{table_fq_name}.{col_name}",
                        "table_name": table_name,
                        "description": col_desc,
                        "is_dimension": is_dim,
                        "is_measure": is_meas,
                        "is_entity_key": is_ent,
                        "is_deprecated": False,
                        "yaml_path": yaml_path_to_store,
                        "updated_at": datetime.now().replace(microsecond=0),
                        # Database hierarchy
                        "catalog_name": catalog_name,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "semantic_model_name": table_name,
                        "expr": col_expr,
                        "column_type": col_def.get(
                            "type", ""
                        ),  # CATEGORICAL/TIME for dims, PRIMARY/FOREIGN etc for idents
                        # Measure specific (empty for non-measures)
                        "agg": col_def.get("agg", "") if is_meas else "",
                        "create_metric": col_def.get("create_metric", False) if is_meas else False,
                        "agg_time_dimension": col_def.get("agg_time_dimension", "") if is_meas else "",
                        # Dimension specific (empty/false for non-dimensions)
                        "is_partition": col_def.get("is_partition", False) if is_dim else False,
                        "time_granularity": time_granularity,
                        # Identifier specific (empty for non-identifiers)
                        "entity": col_def.get("entity", "") if is_ent else "",
                    }
                    semantic_objects.append(col_obj)

                # Process Dimensions
                for dim in data_source.get("dimensions", []):
                    process_column(dim, is_dim=True)

                # Process Measures
                for meas in data_source.get("measures", []):
                    # Measures in MF are defined on columns but act as aggregations
                    # For semantic search, we treat them as 'fields' you can query
                    process_column(meas, is_meas=True)

                # Process Identifiers
                for ident in data_source.get("identifiers", []):
                    process_column(ident, is_ent=True)

            # 3. Process Metrics (Standard Metrics) - These go to MetricStorage
            if include_metrics:
                for metric in metrics_list:
                    m_name = metric.get("name")
                    if not m_name:
                        continue

                    m_desc = metric.get("description", "")
                    m_type = metric.get("type", "")

                    # Parse tags for subject_path (domain/layer1/layer2)
                    subject_path = []
                    locked_meta = metric.get("locked_metadata", {})
                    if locked_meta:
                        tags = locked_meta.get("tags", [])
                        parsed_path = GenerationHooks._parse_subject_tree_from_tags(tags)
                        if parsed_path:
                            subject_path = parsed_path

                    # If no subject_path found, use default path with semantic_model_name
                    if not subject_path:
                        subject_path = ["Metrics", table_name if table_name else "Unknown"]

                    # Extract type_params for measure_expr, base_measures
                    type_params = metric.get("type_params", {})
                    measure_expr = ""
                    base_measures = []

                    if m_type == "measure_proxy":
                        # Single measure reference
                        measure = type_params.get("measure")
                        if measure:
                            measure_expr = measure
                            base_measures = [measure]
                        # Or multiple measures
                        measures_list = type_params.get("measures", [])
                        for m in measures_list:
                            if isinstance(m, str):
                                base_measures.append(m)
                            elif isinstance(m, dict):
                                m_name_val = m.get("name", "")
                                if m_name_val:
                                    base_measures.append(m_name_val)
                    elif m_type == "ratio":
                        # Ratio has numerator and denominator
                        num = type_params.get("numerator", {})
                        denom = type_params.get("denominator", {})
                        if isinstance(num, str):
                            base_measures.append(num)
                        elif isinstance(num, dict):
                            num_name = num.get("name", "")
                            if num_name:
                                base_measures.append(num_name)
                        if isinstance(denom, str):
                            base_measures.append(denom)
                        elif isinstance(denom, dict):
                            denom_name = denom.get("name", "")
                            if denom_name:
                                base_measures.append(denom_name)
                    elif m_type in ["expr", "cumulative"]:
                        # Extract measures from measures list
                        measures_list = type_params.get("measures", [])
                        for m in measures_list:
                            if isinstance(m, str):
                                base_measures.append(m)
                            elif isinstance(m, dict):
                                m_name_val = m.get("name", "")
                                if m_name_val:
                                    base_measures.append(m_name_val)
                        # For expr type, also save the expression
                        if m_type == "expr":
                            expr_val = type_params.get("expr")
                            if expr_val:
                                measure_expr = str(expr_val)
                    elif m_type == "derived":
                        # Derived metrics reference other metrics
                        metrics_list_param = type_params.get("metrics", [])
                        for m in metrics_list_param:
                            if isinstance(m, str):
                                base_measures.append(m)
                            elif isinstance(m, dict):
                                m_name_val = m.get("name", "")
                                if m_name_val:
                                    base_measures.append(m_name_val)
                        # Save the derived expression
                        expr_val = type_params.get("expr")
                        if expr_val:
                            measure_expr = str(expr_val)

                    # Extract dimensions and entities from data_source if available
                    dimensions = []
                    entities = []
                    if data_source:
                        # Get dimension names
                        for dim in data_source.get("dimensions", []):
                            dim_name = dim.get("name")
                            if dim_name:
                                dimensions.append(dim_name)
                        # Get entity names
                        for ident in data_source.get("identifiers", []):
                            ident_name = ident.get("name")
                            if ident_name:
                                entities.append(ident_name)

                    # Build metric object for MetricStorage
                    metric_obj = {
                        "name": m_name,
                        "subject_path": subject_path,
                        "semantic_model_name": table_name,
                        "id": f"metric:{m_name}",
                        "description": m_desc,
                        "metric_type": m_type,
                        "measure_expr": measure_expr,
                        "base_measures": base_measures,
                        "dimensions": dimensions,
                        "entities": entities,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "updated_at": datetime.now().replace(microsecond=0),
                        # Database hierarchy
                        "catalog_name": catalog_name,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        # Generated SQL from dry_run
                        "sql": metric_sqls.get(m_name, "") if metric_sqls else "",
                    }
                    metric_objects.append(metric_obj)
                    synced_items.append(f"metric:{m_name}")

            # Store all objects using upsert (update if id exists, insert if not)
            all_objects = semantic_objects + metric_objects
            if all_objects:
                if semantic_objects:
                    semantic_rag.upsert_batch(semantic_objects)
                    semantic_rag.create_indices()

                if metric_objects:
                    metric_rag.upsert_batch(metric_objects)
                    metric_rag.create_indices()
                return {
                    "success": True,
                    "message": (
                        f"Synced {len(all_objects)} objects "
                        f"({len(semantic_objects)} semantic, {len(metric_objects)} metrics): "
                        f"{', '.join(synced_items[:5])}..."
                    ),
                }
            else:
                return {"success": False, "error": "No valid objects found to sync"}

        except Exception as e:
            logger.error(f"Error syncing semantic objects to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _sync_reference_sql_to_db(file_path: str, agent_config: AgentConfig, build_mode: str = "incremental") -> dict:
        """
        Sync reference SQL YAML file to Knowledge Base.
        """
        try:
            from datus.storage.reference_sql.init_utils import exists_reference_sql, gen_reference_sql_id

            # Load YAML file
            with open(file_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)

            if isinstance(doc, dict) and "sql" in doc:
                # Direct format without reference_sql wrapper
                reference_sql_data = doc
            else:
                return {"success": False, "error": "No reference_sql data found in YAML file"}

            # Generate ID if not present or if it's a placeholder
            sql_query = reference_sql_data.get("sql", "")
            comment = reference_sql_data.get("comment", "")
            item_id = reference_sql_data.get("id", "")

            if not item_id or item_id == "auto_generated":
                item_id = gen_reference_sql_id(sql_query)
                reference_sql_data["id"] = item_id

            # Get storage and check if item already exists
            storage = ReferenceSqlRAG(agent_config)
            existing_ids = exists_reference_sql(storage, build_mode=build_mode)

            # Check for duplicate
            if item_id in existing_ids:
                logger.info(f"Reference SQL {item_id} already exists in Knowledge Base, skipping")
                return {
                    "success": True,
                    "message": f"Reference SQL '{reference_sql_data.get('name', '')}' already exists, skipped",
                }

            # Parse subject_tree if available
            subject_path = []
            subject_tree_str = reference_sql_data.get("subject_tree", "")
            if subject_tree_str:
                # Parse subject_tree format: "path/component1/component2/..."
                parts = subject_tree_str.split("/")
                subject_path = [part.strip() for part in parts if part.strip()]

            # Ensure all required fields are present
            reference_sql_dict = {
                "id": item_id,
                "name": reference_sql_data.get("name", ""),
                "sql": sql_query,
                "comment": comment,
                "summary": reference_sql_data.get("summary", ""),
                "search_text": reference_sql_data.get("search_text", ""),
                "filepath": file_path,
                "subject_path": subject_path,
                "tags": reference_sql_data.get("tags", ""),
            }

            # Store to Knowledge Base
            storage.store_batch([reference_sql_dict])

            logger.info(f"Successfully synced reference SQL {item_id} to Knowledge Base")
            return {"success": True, "message": f"Synced reference SQL: {reference_sql_dict['name']}"}

        except Exception as e:
            logger.error(f"Error syncing reference SQL to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _sync_ext_knowledge_to_db(file_path: str, agent_config: AgentConfig, build_mode: str = "incremental") -> dict:
        """
        Sync external knowledge YAML file to Knowledge Base.

        Supports multi-document YAML files (documents separated by ---).
        Each document is treated as a separate knowledge entry.

        Args:
            file_path: Path to the external knowledge YAML file
            agent_config: Agent configuration
            build_mode: "overwrite" or "incremental" (default: "incremental")

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            import yaml

            from datus.storage.cache import get_storage_cache_instance
            from datus.storage.ext_knowledge.init_utils import exists_ext_knowledge, gen_ext_knowledge_id

            # Load YAML file - supports multiple documents
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            if not docs or all(doc is None for doc in docs):
                return {"success": False, "error": "Empty YAML file or all documents are empty"}

            # Get storage instance
            knowledge_store = get_storage_cache_instance(agent_config).ext_knowledge_storage()
            existing_ids = exists_ext_knowledge(knowledge_store, build_mode=build_mode)

            # Process each document
            synced_count = 0
            skipped_count = 0
            synced_names = []
            skipped_names = []

            for i, doc in enumerate(docs):
                if not doc:
                    logger.warning(f"Document {i+1} in {file_path} is empty, skipping")
                    continue

                # Parse subject_path
                subject_path_str = doc.get("subject_path", "")
                subject_path = [p.strip() for p in subject_path_str.split("/") if p.strip()]

                # Generate ID for duplicate check
                search_text = doc.get("search_text", "")
                item_id = gen_ext_knowledge_id(subject_path, search_text)
                name = doc.get("name", search_text)

                # Check for duplicate
                if item_id in existing_ids:
                    logger.info(f"External knowledge {item_id} already exists in Knowledge Base, skipping")
                    skipped_count += 1
                    skipped_names.append(name)
                    continue

                # Store to Knowledge Base
                knowledge_store.store_knowledge(
                    subject_path=subject_path,
                    name=name,
                    search_text=search_text,
                    explanation=doc.get("explanation", ""),
                )

                logger.info(f"Successfully synced external knowledge {item_id} to Knowledge Base")
                synced_count += 1
                synced_names.append(name)

            # Build result message
            messages = []
            if synced_count > 0:
                if synced_count == 1:
                    messages.append(f"Synced 1 knowledge entry: {synced_names[0]}")
                else:
                    messages.append(f"Synced {synced_count} knowledge entries: {', '.join(synced_names[:3])}")
                    if synced_count > 3:
                        messages[-1] += f" and {synced_count - 3} more"

            if skipped_count > 0:
                if skipped_count == 1:
                    messages.append(f"Skipped 1 existing entry: {skipped_names[0]}")
                else:
                    messages.append(f"Skipped {skipped_count} existing entries")

            total_processed = synced_count + skipped_count
            if total_processed == 0:
                return {"success": False, "error": "No valid knowledge entries found in YAML file"}

            return {
                "success": True,
                "message": "; ".join(messages),
                "synced_count": synced_count,
                "skipped_count": skipped_count,
            }

        except Exception as e:
            logger.error(f"Error syncing external knowledge to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
