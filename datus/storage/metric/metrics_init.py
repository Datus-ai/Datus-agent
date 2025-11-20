# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import argparse
import asyncio
import os
from typing import Dict, List, Optional

import pandas as pd

from datus.agent.node.semantic_agentic_node import SemanticAgenticNode
from datus.cli.generation_hooks import GenerationHooks
from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.schemas.semantic_agentic_node_models import SemanticNodeInput
from datus.utils.loggings import get_logger
from datus.utils.sql_utils import extract_table_names

logger = get_logger(__name__)


def init_success_story_metrics(
    args: argparse.Namespace,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
):
    """
    Initialize metrics from success story CSV file using SemanticAgenticNode in workflow mode.

    Args:
        args: Command line arguments
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
    """
    df = pd.read_csv(args.success_story)

    async def process_all() -> tuple[bool, List[str]]:
        errors: List[str] = []
        for idx, row in df.iterrows():
            row_idx = idx + 1
            logger.info(f"Processing row {row_idx}/{len(df)}")
            try:
                result = await process_line(row.to_dict(), agent_config, subject_tree)
                if not result.get("successful"):
                    errors.append(f"Error processing row {row_idx}: {result.get('error')}")
            except Exception as e:
                errors.append(f"Error processing row {row_idx}: {e}")
                logger.error(f"Error processing row {row_idx}: {e}")
        return (len(df) - len(errors)) > 0, errors

    # Run the async function
    successful, errors = asyncio.run(process_all())
    if errors:
        error_message = "\n    ".join(errors)
    else:
        error_message = ""
    return successful, error_message


async def process_line(
    row: dict,
    agent_config: AgentConfig,
    subject_tree: Optional[list] = None,
) -> Dict[str, any]:
    """
    Process a single line from the CSV using SemanticAgenticNode in workflow mode.

    Args:
        row: CSV row data containing question and sql
        agent_config: Agent configuration
        subject_tree: Optional predefined subject tree categories
    """
    logger.info(f"processing line: {row}")

    current_db_config = agent_config.current_db_config()

    # Extract table name from SQL query (as requested by user)
    table_names = extract_table_names(row["sql"], agent_config.db_type)
    table_name = table_names[0] if table_names else ""

    if not table_name:
        logger.error(f"No table name found in SQL query: {row['sql']}")
        return {
            "successful": False,
            "error": "No table name found in SQL query",
        }

    # Step 1: Generate semantic model using SemanticAgenticNode
    semantic_user_message = f"Generate semantic model for table: {table_name}\nQuestion context: {row['question']}"
    semantic_input = SemanticNodeInput(
        user_message=semantic_user_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    semantic_node = SemanticAgenticNode(
        node_name="gen_semantic_model",
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()
    semantic_model_file = None

    try:
        semantic_node.input = semantic_input
        async for action in semantic_node.execute_stream(action_history_manager):
            if action.status == ActionStatus.SUCCESS and action.output:
                output = action.output
                if isinstance(output, dict):
                    semantic_model_file = output.get("semantic_model")

        if not semantic_model_file:
            logger.error(f"Failed to generate semantic model for {row['question']}")
            return {
                "successful": False,
                "error": "Failed to generate semantic model",
            }

        logger.info(f"Generated semantic model: {semantic_model_file}")

    except Exception as e:
        logger.error(f"Error generating semantic model for {row['question']}: {e}")
        return {
            "successful": False,
            "error": f"Error generating semantic model for this question, reason: {str(e)}",
        }

    # Step 2: Generate metrics using SemanticAgenticNode
    metrics_user_message = (
        f"Generate metrics for the following SQL query:\n\nSQL:\n{row['sql']}\n\n"
        f"Question: {row['question']}\n\nTable: {table_name}"
        f"Use the following semantic model: {semantic_model_file}"
    )
    metrics_input = SemanticNodeInput(
        user_message=metrics_user_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    metrics_node = SemanticAgenticNode(
        node_name="gen_metrics",
        agent_config=agent_config,
        execution_mode="workflow",
        subject_tree=subject_tree,
    )

    action_history_manager = ActionHistoryManager()

    try:
        metrics_node.input = metrics_input
        async for action in metrics_node.execute_stream(action_history_manager):
            if action.status == ActionStatus.SUCCESS and action.output:
                logger.debug(f"Metrics generation action: {action.messages}")

        logger.info(f"Generated metrics for {row['question']}")
        return {
            "successful": True,
            "error": "",
        }
    except Exception as e:
        logger.error(f"Error generating metrics for {row['question']}: {e}")
        return {
            "successful": False,
            "error": f"Error generating metrics for this question, reason: {str(e)}",
        }


def init_semantic_yaml_metrics(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Initialize metrics from semantic YAML file by syncing directly to LanceDB.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    """
    if not os.path.exists(yaml_file_path):
        logger.error(f"Semantic YAML file {yaml_file_path} not found")
        return False, f"Semantic YAML file {yaml_file_path} not found"

    return process_semantic_yaml_file(yaml_file_path, agent_config)


def process_semantic_yaml_file(
    yaml_file_path: str,
    agent_config: AgentConfig,
) -> tuple[bool, str]:
    """
    Process semantic YAML file by directly syncing to LanceDB using GenerationHooks.

    Args:
        yaml_file_path: Path to semantic YAML file
        agent_config: Agent configuration
    Returns:
        - Whether the execution was successful
        - Failed reason

    """
    logger.info(f"Processing semantic YAML file: {yaml_file_path}")

    # Use GenerationHooks static method to sync to DB
    result = GenerationHooks._sync_semantic_to_db(yaml_file_path, agent_config)

    if result.get("success"):
        logger.info(f"Successfully synced semantic YAML to LanceDB: {result.get('message')}")
        return True, ""
    else:
        error = result.get("error", "Unknown error")
        logger.error(f"Failed to sync semantic YAML to LanceDB: {error}")
        return False, error
