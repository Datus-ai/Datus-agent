# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Auto-create missing semantic models before metrics generation."""

import asyncio
from typing import Callable, List, Optional, Set

from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def extract_tables_from_sql_list(
    sql_list: List[str],
    agent_config: AgentConfig,
) -> Set[str]:
    """
    Extract table names from a list of SQL statements.

    Args:
        sql_list: List of SQL statements
        agent_config: Agent configuration (for dialect)

    Returns:
        Set of table names (may include fully qualified names)
    """
    from datus.utils.sql_utils import extract_table_names

    all_tables = set()
    dialect = agent_config.db_type

    for sql in sql_list:
        if sql and sql.strip():
            try:
                tables = extract_table_names(sql, dialect=dialect, ignore_empty=True)
                all_tables.update(tables)
            except Exception as e:
                logger.warning(f"Failed to extract tables from SQL: {e}")
                continue

    return all_tables


def find_missing_semantic_models(
    tables: Set[str],
    agent_config: AgentConfig,
) -> List[str]:
    """
    Check which tables don't have semantic models in LanceDB.

    Args:
        tables: Set of table names to check
        agent_config: Agent configuration

    Returns:
        List of table names that are missing semantic models
    """
    from datus.storage.semantic_model.store import SemanticModelRAG

    if not tables:
        return []

    semantic_rag = SemanticModelRAG(agent_config)
    missing = []

    for table_fq_name in tables:
        # Parse table name (may be database.schema.table format)
        parts = table_fq_name.split(".")
        table_name = parts[-1]  # Last part is the table name

        # Search for existing semantic model
        try:
            result = semantic_rag.storage.search_objects(
                query_text=table_name,
                kinds=["table"],
                top_n=5,
            )

            # Exact match on table name (case insensitive)
            exists = any(obj.get("name", "").lower() == table_name.lower() for obj in result)

            if not exists:
                missing.append(table_fq_name)
        except Exception as e:
            logger.warning(f"Error checking semantic model for {table_name}: {e}")
            missing.append(table_fq_name)

    return missing


async def create_semantic_models_for_tables(
    tables: List[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
) -> tuple[bool, str]:
    """
    Create semantic models for the specified tables.

    Args:
        tables: List of table names to create semantic models for
        agent_config: Agent configuration
        emit: Optional progress callback

    Returns:
        (success, error_message)
    """
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

    if not tables:
        return True, ""

    # Build user message
    tables_str = ", ".join(tables)
    user_message = f"Generate semantic models for the following tables: {tables_str}"

    current_db_config = agent_config.current_db_config()
    semantic_input = SemanticNodeInput(
        user_message=user_message,
        catalog=current_db_config.catalog,
        database=current_db_config.database,
        db_schema=current_db_config.schema,
    )

    # Use workflow mode to auto-save to DB
    semantic_node = GenSemanticModelAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
    )
    semantic_node.input = semantic_input

    action_history_manager = ActionHistoryManager()
    try:
        async for action in semantic_node.execute_stream(action_history_manager):
            if emit:
                # Emit progress event
                emit(action)
            if action.status == ActionStatus.FAILED:
                return False, action.messages or "Semantic model generation failed"
        return True, ""
    except Exception as e:
        logger.error(f"Error creating semantic models: {e}", exc_info=True)
        return False, str(e)


def create_semantic_models_for_tables_sync(
    tables: List[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
) -> tuple[bool, str]:
    """
    Synchronous wrapper for create_semantic_models_for_tables.

    Args:
        tables: List of table names to create semantic models for
        agent_config: Agent configuration
        emit: Optional progress callback

    Returns:
        (success, error_message)
    """
    return asyncio.run(create_semantic_models_for_tables(tables, agent_config, emit))


async def ensure_semantic_models_exist(
    tables: Set[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
) -> tuple[bool, str, List[str]]:
    """
    Check and create missing semantic models.

    Args:
        tables: Set of table names to check
        agent_config: Agent configuration
        emit: Optional progress callback

    Returns:
        (success, error_message, created_tables)
    """
    missing_tables = find_missing_semantic_models(tables, agent_config)

    if not missing_tables:
        logger.info("All required semantic models already exist")
        return True, "", []

    logger.info(f"Found {len(missing_tables)} tables without semantic models: {missing_tables}")

    success, error = await create_semantic_models_for_tables(missing_tables, agent_config, emit)

    if success:
        logger.info(f"Successfully created semantic models for: {missing_tables}")
    else:
        logger.error(f"Failed to create semantic models: {error}")

    return success, error, missing_tables


def ensure_semantic_models_exist_sync(
    tables: Set[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
) -> tuple[bool, str, List[str]]:
    """
    Synchronous wrapper for ensure_semantic_models_exist.

    Args:
        tables: Set of table names to check
        agent_config: Agent configuration
        emit: Optional progress callback

    Returns:
        (success, error_message, created_tables)
    """
    return asyncio.run(ensure_semantic_models_exist(tables, agent_config, emit))
