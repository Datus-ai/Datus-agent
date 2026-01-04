# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os

import pandas as pd

from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def init_success_story_semantic_model(
    args: argparse.Namespace,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize ONLY semantic model from success story CSV using ALL SQL queries.

    IMPORTANT: This function processes the ENTIRE success_story CSV in one go,
    NOT line-by-line. It uses execution_mode="workflow" (not plan mode).

    The gen_semantic_model node will receive all SQL queries from the CSV
    and generate semantic models for all tables found in those queries.

    Args:
        args: Command line arguments
        agent_config: Agent configuration
    """
    # Load and validate CSV file
    csv_path = args.success_story
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        error_msg = f"Success story CSV file not found: {csv_path}"
        logger.error(error_msg)
        return False, error_msg
    except pd.errors.EmptyDataError:
        error_msg = f"Success story CSV file is empty: {csv_path}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Failed to read success story CSV file '{csv_path}': {e}"
        logger.exception(error_msg)
        return False, error_msg

    # Validate required columns
    required_columns = ["sql", "question"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = (
            f"Success story CSV '{csv_path}' is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(error_msg)
        return False, error_msg

    # Collect all SQL queries and questions
    all_sqls = df["sql"].tolist()
    all_questions = df["question"].tolist()

    # Validate data alignment
    if len(all_sqls) != len(all_questions):
        error_msg = (
            f"Success story CSV '{csv_path}' has mismatched column lengths: "
            f"sql={len(all_sqls)}, question={len(all_questions)}"
        )
        logger.error(error_msg)
        return False, error_msg

    if len(all_sqls) == 0:
        error_msg = f"Success story CSV '{csv_path}' contains no data rows"
        logger.error(error_msg)
        return False, error_msg

    # Build comprehensive context from all rows
    context_message = "Generate semantic models for the following SQL queries:\n\n"
    for idx, (sql, question) in enumerate(zip(all_sqls, all_questions), 1):
        context_message += f"Query {idx}:\n"
        context_message += f"Question: {question}\n"
        context_message += f"SQL:\n{sql}\n\n"

    async def generate_semantic_models() -> tuple[bool, str]:
        """Execute gen_semantic_model node with all SQL context."""
        current_db_config = agent_config.current_db_config()

        # Create semantic model generation node (workflow mode, NOT plan mode)
        semantic_node = GenSemanticModelAgenticNode(
            agent_config=agent_config,
            execution_mode="workflow",  # CRITICAL: workflow mode only
        )

        semantic_input = SemanticNodeInput(
            user_message=context_message,
            catalog=current_db_config.catalog,
            database=current_db_config.database,
            db_schema=current_db_config.schema,
        )

        action_history_manager = ActionHistoryManager()
        semantic_node.input = semantic_input

        try:
            generated_files = []
            async for action in semantic_node.execute_stream(action_history_manager):
                if action.status == ActionStatus.SUCCESS and action.output:
                    if isinstance(action.output, dict):
                        # Check for semantic_models field (from SemanticNodeResult)
                        if "semantic_models" in action.output:
                            models = action.output["semantic_models"]
                            if isinstance(models, list):
                                generated_files.extend(models)
                            elif models:  # Single file as string
                                generated_files.append(models)

            if not generated_files:
                error_msg = f"Failed to generate any semantic models from {len(all_sqls)} SQL queries in '{csv_path}'"
                logger.error(error_msg)
                return False, error_msg

            logger.info(f"Generated {len(generated_files)} semantic model files: {generated_files}")
            return True, ""

        except Exception as e:
            error_msg = f"Error generating semantic models from '{csv_path}': {e}"
            logger.exception(error_msg)
            return False, error_msg

    successful, error_message = asyncio.run(generate_semantic_models())
    return successful, error_message


def init_semantic_yaml_semantic_model(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize ONLY semantic model (table/column/entity) from YAML, skip metrics.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    return process_semantic_yaml_file(yaml_file_path, agent_config, include_metrics=False)


def process_semantic_yaml_file(
    yaml_file_path: str,
    agent_config: AgentConfig,
    include_semantic_objects: bool = True,
    include_metrics: bool = True,
) -> tuple[bool, str]:
    """
    Process semantic YAML file by directly syncing to LanceDB using GenerationHooks.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
        include_semantic_objects: Whether to sync tables/columns/entities
        include_metrics: Whether to sync metrics
    Returns:
        - Whether the execution was successful
        - Failed reason

    """
    logger.info(
        f"Processing semantic YAML file: {yaml_file_path} "
        f"(semantic_objects={include_semantic_objects}, metrics={include_metrics})"
    )

    # Validate file exists
    if not os.path.exists(yaml_file_path):
        error_msg = f"Semantic YAML file not found: {yaml_file_path}"
        logger.error(error_msg)
        return False, error_msg

    # Use GenerationHooks static method to sync to DB
    try:
        result = GenerationHooks._sync_semantic_to_db(
            yaml_file_path,
            agent_config,
            include_semantic_objects=include_semantic_objects,
            include_metrics=include_metrics,
        )
    except Exception as e:
        error_msg = f"Failed to sync semantic YAML file '{yaml_file_path}' to LanceDB: {e}"
        logger.exception(error_msg)
        return False, error_msg

    if result.get("success"):
        logger.info(f"Successfully synced to LanceDB: {result.get('message')}")
        return True, ""
    else:
        error = result.get("error", "Unknown error")
        error_msg = f"Failed to sync '{yaml_file_path}' to LanceDB: {error}"
        logger.error(error_msg)
        return False, error
