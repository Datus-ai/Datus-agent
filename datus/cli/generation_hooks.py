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

from datus.cli.execution_state import InteractionBroker, InteractionCancelled
from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import MetricRAG
from datus.storage.reference_sql.store import ReferenceSqlRAG
from datus.storage.semantic_model.store import SemanticModelRAG
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger
from datus.utils.path_manager import get_path_manager
from datus.utils.traceable_utils import optional_traceable

logger = get_logger(__name__)


class GenerationCancelledException(Exception):
    """Exception raised when user cancels generation flow."""


class GenerationHooks(AgentHooks):
    """Hooks for handling generation tool results and user interaction."""

    def __init__(self, broker: InteractionBroker, agent_config: AgentConfig = None):
        """
        Initialize generation hooks.

        Args:
            broker: InteractionBroker for async user interactions
            agent_config: Agent configuration for storage access
        """
        self.broker = broker
        self.agent_config = agent_config
        self.processed_files = set()  # Track files that have been processed to avoid duplicates
        logger.debug(f"GenerationHooks initialized with config: {agent_config is not None}")

    async def on_start(self, context, agent) -> None:
        pass

    @optional_traceable(name="on_tool_end", run_type="chain")
    async def on_tool_end(self, context, agent, tool, result) -> None:
        """Handle generation tool completion."""
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
        pass

    async def on_handoff(self, context, agent, source) -> None:
        pass

    async def on_end(self, context, agent, output) -> None:
        pass

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
            logger.info("Generation workflow cancelled")
        except Exception as e:
            logger.error(f"Error handling end_semantic_model_generation: {e}", exc_info=True)

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

            # Convert relative paths to absolute paths
            if self.agent_config:
                base_dir = str(
                    get_path_manager(self.agent_config.home).semantic_model_path(self.agent_config.current_namespace)
                )
                if metric_file and not os.path.isabs(metric_file):
                    metric_file = os.path.join(base_dir, metric_file)
                if semantic_model_file and not os.path.isabs(semantic_model_file):
                    semantic_model_file = os.path.join(base_dir, semantic_model_file)

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
            logger.info("Generation workflow cancelled")
        except Exception as e:
            logger.error(f"Error handling end_metric_generation: {e}", exc_info=True)

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

        # Build display content (markdown format)
        display_content = f"## Generated YAML: {os.path.basename(file_path)}\n\n"
        display_content += f"*Path: {file_path}*\n\n"
        display_content += f"```yaml\n{yaml_content}\n```\n"

        # Get user confirmation to sync (content will be shown in request)
        await self._get_sync_confirmation(
            yaml_content, file_path, "semantic", metric_sqls=metric_sqls, display_content=display_content
        )

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

        # Build display content (markdown format) with both files
        display_content = f"## Generated Semantic Model: {os.path.basename(semantic_model_file)}\n\n"
        display_content += f"*Path: {semantic_model_file}*\n\n"
        display_content += f"```yaml\n{semantic_content}\n```\n\n"
        display_content += "---\n\n"
        display_content += f"## Generated Metric: {os.path.basename(metric_file)}\n\n"
        display_content += f"*Path: {metric_file}*\n\n"
        display_content += f"```yaml\n{metric_content}\n```\n"

        # Get user confirmation to sync both files together
        await self._get_sync_confirmation_for_pair(
            semantic_model_file, metric_file, metric_sqls, display_content=display_content
        )

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

            # Build display content (markdown format)
            display_content = "## Generated Reference SQL YAML\n\n"
            display_content += f"*File: {file_path}*\n\n"
            display_content += f"```yaml\n{yaml_content}\n```\n"

            # Get user confirmation to sync (this is for SQL summary)
            await self._get_sync_confirmation(yaml_content, file_path, "sql_summary", display_content=display_content)

        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error handling write_file_reference_sql result: {e}", exc_info=True)

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

            # Build display content (markdown format)
            display_content = "## Generated External Knowledge YAML\n\n"
            display_content += f"*File: {file_path}*\n\n"
            display_content += f"```yaml\n{yaml_content}\n```\n"

            # Get user confirmation to sync (this is for external knowledge)
            await self._get_sync_confirmation(yaml_content, file_path, "ext_knowledge", display_content=display_content)

        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error handling write_file_ext_knowledge result: {e}", exc_info=True)

    async def _get_sync_confirmation_for_pair(
        self,
        semantic_model_file: str,
        metric_file: str,
        metric_sqls: dict = None,
        display_content: str = "",
    ):
        """
        Get user confirmation to sync semantic model and metric files together to Knowledge Base.

        Args:
            semantic_model_file: Path to semantic model YAML file
            metric_file: Path to metric YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
            display_content: Pre-built markdown content to display (headers + YAML contents)
        """
        try:
            request_content = f"{display_content}\n### Sync to Knowledge Base?"

            choice, callback = await self.broker.request(
                content=request_content,
                choices={"y": "Yes - Save to Knowledge Base", "n": "No - Keep file only"},
                default_choice="y",
            )

            if choice == "y":
                # Sync both files to Knowledge Base
                sync_result = await self._sync_semantic_and_metric(semantic_model_file, metric_file, metric_sqls)
                callback_content = "---\n\n"
                callback_content += sync_result
                callback_content += "\n\n---\n**Generation workflow completed, generating report...**"
                await callback(callback_content)
            else:
                # Keep files only
                callback_content = "---\n\n"
                callback_content += f"YAMLs saved to files only:\n- `{semantic_model_file}`\n- `{metric_file}`"
                callback_content += "\n\n---\n**Generation workflow completed, generating report...**"
                await callback(callback_content)

        except InteractionCancelled:
            raise GenerationCancelledException("User interrupted")
        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error in sync confirmation: {e}", exc_info=True)
            raise

    async def _get_sync_confirmation(
        self,
        yaml_content: str,
        file_path: str,
        yaml_type: str,
        metric_sqls: dict = None,
        display_content: str = "",
    ):
        """
        Get user confirmation to sync to Knowledge Base.

        Args:
            yaml_content: Generated YAML content
            file_path: Path where YAML was saved
            yaml_type: YAML type - "semantic" or "sql_summary"
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)
            display_content: Pre-built markdown content to display (header + YAML)
        """
        try:
            # Build request content with YAML display
            if not display_content:
                display_content = f"## Generated YAML: {os.path.basename(file_path)}\n\n"
                display_content += f"*Path: {file_path}*\n\n"
                display_content += f"```yaml\n{yaml_content}\n```\n\n"

            request_content = f"{display_content}\n### Sync to Knowledge Base?"

            choice, callback = await self.broker.request(
                content=request_content,
                choices={"y": "Yes - Save to Knowledge Base", "n": "No - Keep file only"},
                default_choice="y",
            )

            if choice == "y":
                # Sync to Knowledge Base
                sync_result = await self._sync_to_storage(file_path, yaml_type, metric_sqls=metric_sqls)
                # Build callback content with result
                callback_content = "---\n\n"
                callback_content += sync_result
                callback_content += "\n\n---\n**Generation workflow completed, generating report...**"
                await callback(callback_content)
            else:
                # Keep file only
                callback_content = "---\n\n"
                callback_content += f"YAML saved to file only: `{file_path}`"
                callback_content += "\n\n---\n**Generation workflow completed, generating report...**"
                await callback(callback_content)

        except InteractionCancelled:
            raise GenerationCancelledException("User interrupted")
        except GenerationCancelledException:
            raise
        except Exception as e:
            logger.error(f"Error in sync confirmation: {e}", exc_info=True)
            raise

    async def _sync_to_storage(self, file_path: str, yaml_type: str, metric_sqls: dict = None) -> str:
        """
        Sync YAML file to RAG storage based on file type.

        Args:
            file_path: File path to sync
            yaml_type: YAML type - "semantic" or "sql_summary"
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)

        Returns:
            Markdown string describing the result
        """
        if not self.agent_config:
            return (
                f"**Error:** Agent configuration not available, cannot sync to RAG\n\nYAML saved to file: `{file_path}`"
            )

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
                return f"**Error:** Invalid yaml_type: {yaml_type}\n\nYAML saved to file: `{file_path}`"

            if result.get("success"):
                result_content = f"**Successfully synced {item_type} to Knowledge Base**\n\n"
                message = result.get("message", "")
                if message:
                    result_content += f"{message}\n\n"
                result_content += f"File: `{file_path}`"
                return result_content
            else:
                error = result.get("error", "Unknown error")
                return f"**Sync failed:** {error}\n\nYAML saved to file: `{file_path}`"

        except Exception as e:
            logger.error(f"Error syncing to storage: {e}")
            return f"**Sync error:** {e}\n\nYAML saved to file: `{file_path}`"

    async def _sync_semantic_and_metric(
        self, semantic_model_file: str, metric_file: str, metric_sqls: dict = None
    ) -> str:
        """
        Sync both semantic model and metric files to RAG storage.
        Creates a combined YAML for syncing so metrics can reference semantic model data.

        Args:
            semantic_model_file: Path to semantic model YAML file
            metric_file: Path to metric YAML file
            metric_sqls: Optional dict mapping metric names to generated SQL (from dry_run)

        Returns:
            Markdown string describing the result
        """
        files_info = f"- `{semantic_model_file}`\n- `{metric_file}`"

        if not self.agent_config:
            return (
                f"**Error:** Agent configuration not available, cannot sync to RAG\n\n"
                f"YAMLs saved to files:\n{files_info}"
            )

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
                    result_content = "**Successfully synced semantic model and metrics to Knowledge Base**\n\n"
                    message = result.get("message", "")
                    if message:
                        result_content += f"{message}\n\n"
                    result_content += f"Files:\n{files_info}"
                    return result_content
                else:
                    error = result.get("error", "Unknown error")
                    return f"**Sync failed:** {error}\n\nYAMLs saved to files:\n{files_info}"

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error syncing semantic and metric: {e}", exc_info=True)
            return f"**Sync error:** {e}\n\nYAMLs saved to files:\n{files_info}"

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
                    metric_table_name = table_name  # Track semantic model name for this metric
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
                    elif base_measures:
                        # Fallback: query dimensions/entities from Knowledge Base
                        # when data_source is not in the same YAML file (multi-table scenario)
                        try:
                            # Find semantic model containing the first base measure
                            measure_name = base_measures[0]
                            # Query semantic objects to find the measure's table
                            measure_objs = semantic_rag.storage._search_all(
                                where=f"kind = 'column' AND is_measure = true AND name = '{measure_name}'"
                            ).to_pylist()
                            if measure_objs:
                                measure_table = measure_objs[0].get("table_name", "")
                                if measure_table:
                                    metric_table_name = measure_table
                                    # Now query dimensions and entities for this table
                                    sm_result = semantic_rag.get_semantic_model(
                                        table_name=measure_table,
                                        select_fields=["dimensions", "identifiers"],
                                    )
                                    if sm_result:
                                        for dim in sm_result.get("dimensions", []):
                                            dim_name = dim.get("name")
                                            if dim_name:
                                                dimensions.append(dim_name)
                                        for ident in sm_result.get("identifiers", []):
                                            ident_name = ident.get("name")
                                            if ident_name:
                                                entities.append(ident_name)
                                        logger.debug(
                                            f"Retrieved dims/ents from KB for metric {m_name} "
                                            f"(table: {measure_table}): {len(dimensions)} dims, "
                                            f"{len(entities)} ents"
                                        )
                        except Exception as e:
                            logger.warning(f"Failed to query dimensions from KB for metric {m_name}: {e}")

                    # Build metric object for MetricStorage
                    metric_obj = {
                        "name": m_name,
                        "subject_path": subject_path,
                        "semantic_model_name": metric_table_name,
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
                        "yaml_path": yaml_path_to_store,
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

            # Store to Knowledge Base (use upsert to handle duplicates)
            storage.upsert_batch([reference_sql_dict])

            logger.info(f"Successfully synced reference SQL {item_id} to Knowledge Base")
            return {"success": True, "message": f"Synced reference SQL: {reference_sql_dict['name']}"}

        except Exception as e:
            logger.error(f"Error syncing reference SQL to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _sync_ext_knowledge_to_db(file_path: str, agent_config: AgentConfig, build_mode: str = "incremental") -> dict:
        """
        Sync external knowledge YAML file to Knowledge Base using upsert.

        Supports multi-document YAML files (documents separated by ---).
        Each document is treated as a separate knowledge entry.
        If a knowledge entry with the same subject_path + search_text exists, it will be updated.

        Args:
            file_path: Path to the external knowledge YAML file
            agent_config: Agent configuration
            build_mode: "overwrite" or "incremental" (default: "incremental")

        Returns:
            dict: Sync result with success, error, and message fields
        """
        try:
            from datus.storage.cache import get_storage_cache_instance

            # Load YAML file - supports multiple documents
            with open(file_path, "r", encoding="utf-8") as f:
                docs = list(yaml.safe_load_all(f))

            if not docs or all(doc is None for doc in docs):
                return {"success": False, "error": "Empty YAML file or all documents are empty"}

            # Get storage instance
            knowledge_store = get_storage_cache_instance(agent_config).ext_knowledge_storage()

            # Collect all valid entries for batch upsert
            knowledge_entries = []
            invalid_count = 0

            for i, doc in enumerate(docs):
                if not doc:
                    logger.warning(f"Document {i+1} in {file_path} is empty, skipping")
                    invalid_count += 1
                    continue

                # Parse subject_path - supports both list and string formats
                subject_path_raw = doc.get("subject_path", "")
                if isinstance(subject_path_raw, list):
                    subject_path = subject_path_raw
                else:
                    subject_path = [p.strip() for p in str(subject_path_raw).split("/") if p.strip()]

                search_text = doc.get("search_text", "")
                name = doc.get("name", search_text)
                explanation = doc.get("explanation", "")

                # Validate required fields
                if not subject_path or not search_text or not explanation:
                    logger.warning(
                        f"Document {i+1} missing required fields (subject_path, search_text, or explanation), skipping"
                    )
                    invalid_count += 1
                    continue

                knowledge_entries.append(
                    {
                        "subject_path": subject_path,
                        "name": name,
                        "search_text": search_text,
                        "explanation": explanation,
                    }
                )

            if not knowledge_entries:
                return {"success": False, "error": "No valid knowledge entries found in YAML file"}

            # Batch upsert all entries (update if exists, insert if not)
            upserted_ids = knowledge_store.batch_upsert_knowledge(knowledge_entries)

            # Build result message from actual upserted names returned by the store
            upserted_count = len(upserted_ids)

            if upserted_count == 1:
                message = f"Upserted 1 knowledge entry: {upserted_ids[0]}"
            else:
                display_names = upserted_ids[:3]
                message = f"Upserted {upserted_count} knowledge entries: {', '.join(display_names)}"
                if upserted_count > 3:
                    message += f" and {upserted_count - 3} more"

            if invalid_count > 0:
                message += f"; Skipped {invalid_count} invalid entries"

            logger.info(f"Successfully upserted {upserted_count} external knowledge entries to Knowledge Base")

            return {
                "success": True,
                "message": message,
                "upserted_count": upserted_count,
                "invalid_count": invalid_count,
            }

        except Exception as e:
            logger.error(f"Error syncing external knowledge to DB: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
