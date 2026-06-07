# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Auto-create missing semantic models before metrics generation."""

import asyncio
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set

from datus.configuration.agent_config import AgentConfig
from datus.schemas.action_history import ActionHistoryManager, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

MAX_SQL_EVIDENCE_PER_TABLE = 8


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


def _table_lookup_keys(table: str) -> Set[str]:
    """Return stable lookup keys for a possibly-qualified table name."""
    if not table:
        return set()

    cleaned = str(table).strip().strip('"`[]')
    if not cleaned:
        return set()

    parts = [part.strip().strip('"`[]') for part in cleaned.split(".") if part.strip()]
    keys = {".".join(parts).lower()} if parts else {cleaned.lower()}
    if parts:
        keys.add(parts[-1].lower())
    return keys


def _format_sql_evidence(index: int, sql: str, question: Optional[str] = None) -> str:
    evidence = f"Query {index}:"
    if question:
        evidence += f"\nQuestion: {question}"
    evidence += f"\nSQL:\n{sql}"
    return evidence


def extract_table_sql_evidence(
    records: Sequence[dict],
    agent_config: AgentConfig,
    *,
    max_records_per_table: int = MAX_SQL_EVIDENCE_PER_TABLE,
) -> Dict[str, List[str]]:
    """
    Build table-scoped SQL evidence from success-story records.

    Keys include both fully-qualified table names and unqualified table names
    so callers can look up evidence regardless of how SQL parsing returned
    the target table.
    """
    from datus.utils.sql_utils import extract_table_names

    evidence_by_key: dict[str, list[str]] = defaultdict(list)
    dialect = agent_config.db_type

    for idx, record in enumerate(records, 1):
        sql = str(record.get("sql") or "").strip()
        if not sql:
            continue

        try:
            tables = extract_table_names(sql, dialect=dialect, ignore_empty=True)
        except Exception as e:
            logger.warning(f"Failed to extract table-scoped SQL evidence: {e}")
            continue

        evidence = _format_sql_evidence(idx, sql, record.get("question"))
        for table in tables:
            for key in _table_lookup_keys(table):
                items = evidence_by_key[key]
                if evidence not in items and len(items) < max_records_per_table:
                    items.append(evidence)

    return dict(evidence_by_key)


def _lookup_sql_evidence(table: str, sql_evidence_by_table: Optional[dict[str, list[str]]]) -> List[str]:
    if not sql_evidence_by_table:
        return []
    for key in _table_lookup_keys(table):
        evidence = sql_evidence_by_table.get(key)
        if evidence:
            return evidence
    return []


def _agent_config_dialect(agent_config: AgentConfig) -> str:
    raw = getattr(agent_config, "db_type", "")
    raw = getattr(raw, "value", raw)
    if isinstance(raw, str) and raw:
        return raw

    try:
        db_config = agent_config.current_db_config()
    except Exception:
        return ""
    raw = getattr(db_config, "type", "")
    raw = getattr(raw, "value", raw)
    return raw if isinstance(raw, str) else ""


def _resolved_table_target(table: str, agent_config: AgentConfig, current_db_config: object) -> dict[str, str]:
    from datus.utils.sql_utils import parse_table_name_parts

    dialect = _agent_config_dialect(agent_config)
    parsed = parse_table_name_parts(table, dialect=dialect or "snowflake")
    table_name = parsed.get("table_name") or str(table).split(".")[-1]

    return {
        "catalog_name": parsed.get("catalog_name") or getattr(current_db_config, "catalog", "") or "",
        "database_name": parsed.get("database_name") or getattr(current_db_config, "database", "") or "",
        "schema_name": parsed.get("schema_name") or getattr(current_db_config, "schema", "") or "",
        "table_name": table_name,
    }


def _format_table_target_for_prompt(table: str, agent_config: AgentConfig, current_db_config: object) -> str:
    target = _resolved_table_target(table, agent_config, current_db_config)
    lines = [
        f"- table_name: {target['table_name']}",
        f"- database: {target['database_name'] or '[default]'}",
        f"- schema_name: {target['schema_name'] or '[default]'}",
    ]
    if target["catalog_name"]:
        lines.insert(1, f"- catalog: {target['catalog_name']}")
    tool_args = [f'table_name="{target["table_name"]}"']
    if target["catalog_name"]:
        tool_args.append(f'catalog="{target["catalog_name"]}"')
    if target["database_name"]:
        tool_args.append(f'database="{target["database_name"]}"')
    if target["schema_name"]:
        tool_args.append(f'schema_name="{target["schema_name"]}"')
    lines.append(f"- database tool arguments: {', '.join(tool_args)}")
    return "\n".join(lines)


def find_missing_semantic_models(
    tables: Set[str],
    agent_config: AgentConfig,
) -> List[str]:
    """
    Check which tables don't have semantic models in vector store.

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


async def create_semantic_model_for_table(
    table: str,
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
    related_tables: Optional[List[str]] = None,
    sql_evidence: Optional[Sequence[str]] = None,
) -> tuple[bool, str]:
    """
    Create a semantic model for a single table.

    Args:
        table: Table to generate the semantic model for.
        agent_config: Agent configuration.
        emit: Optional progress callback.
        related_tables: Other tables being processed in the same batch.
            Passed as context so the LLM can infer join relationships.
        sql_evidence: Success-story SQL queries that reference this table.
            Passed as primary modeling evidence so joins and derived time
            columns are not lost during per-table auto-creation.

    Returns:
        (success, error_message)
    """
    from datus.agent.node.gen_semantic_model_agentic_node import GenSemanticModelAgenticNode
    from datus.schemas.semantic_agentic_node_models import SemanticNodeInput

    try:
        current_db_config = agent_config.current_db_config()
        target = _resolved_table_target(table, agent_config, current_db_config)
        user_message = (
            "Generate a semantic model for the following table.\n\n"
            "Target table coordinate:\n"
            f"{_format_table_target_for_prompt(table, agent_config, current_db_config)}\n\n"
            "When calling database tools, pass the namespace fields separately exactly as shown above; "
            "do not collapse a schema name into the database argument."
        )
        if related_tables:
            others = [t for t in related_tables if t != table]
            if others:
                related_context = "\n\n".join(
                    _format_table_target_for_prompt(related_table, agent_config, current_db_config)
                    for related_table in others
                )
                user_message += f"\n\nRelated tables (for join context):\n{related_context}"
        if sql_evidence:
            user_message += (
                "\n\nSuccess-story SQL evidence for this table. Use these queries as "
                "primary modeling evidence, preserving joins that derive business "
                "dimensions or real time columns:\n\n" + "\n\n".join(sql_evidence)
            )
    except Exception as e:
        error = f"Error preparing semantic model input for table {table}: {e}"
        logger.error(error, exc_info=True)
        return False, error

    semantic_input = SemanticNodeInput(
        user_message=user_message,
        catalog=target["catalog_name"],
        database=target["database_name"],
        db_schema=target["schema_name"],
    )

    semantic_node = GenSemanticModelAgenticNode(
        agent_config=agent_config,
        execution_mode="workflow",
    )
    semantic_node.input = semantic_input

    action_history_manager = ActionHistoryManager()
    try:
        terminal_error = None
        async for action in semantic_node.execute_stream(action_history_manager):
            if emit:
                emit(action)
            action_type = getattr(action, "action_type", "")
            if action.status == ActionStatus.FAILED and action_type == "error":
                terminal_error = action.messages or "Semantic model generation failed"
                logger.error(terminal_error)
                continue
        if terminal_error:
            return False, terminal_error
        return True, ""
    except Exception as e:
        logger.error(f"Error creating semantic model for table {table}: {e}", exc_info=True)
        return False, str(e)


async def create_semantic_models_for_tables(
    tables: List[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
    sql_evidence_by_table: Optional[dict[str, list[str]]] = None,
) -> tuple[List[str], List[tuple[str, str]]]:
    """
    Create semantic models for the specified tables, processing each table
    independently so that one failure does not block others.

    Args:
        tables: List of table names to create semantic models for
        agent_config: Agent configuration
        emit: Optional progress callback
        sql_evidence_by_table: Optional mapping of table lookup key to
            success-story SQL evidence.

    Returns:
        (succeeded_tables, failed_tables) where failed_tables is a list of
        (table_name, error_message) tuples.
    """
    if not tables:
        return [], []

    succeeded: List[str] = []
    failed: List[tuple[str, str]] = []

    for table in tables:
        logger.info(f"Creating semantic model for table: {table}")
        sql_evidence = _lookup_sql_evidence(table, sql_evidence_by_table)
        if sql_evidence:
            success, error = await create_semantic_model_for_table(
                table,
                agent_config,
                emit,
                related_tables=tables,
                sql_evidence=sql_evidence,
            )
        else:
            success, error = await create_semantic_model_for_table(table, agent_config, emit, related_tables=tables)
        if success:
            succeeded.append(table)
            logger.info(f"Successfully created semantic model for table: {table}")
        else:
            failed.append((table, error))
            logger.warning(
                f"Failed to create semantic model for table {table}: {error}, continuing with remaining tables"
            )

    return succeeded, failed


def create_semantic_models_for_tables_sync(
    tables: List[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
    sql_evidence_by_table: Optional[dict[str, list[str]]] = None,
) -> tuple[List[str], List[tuple[str, str]]]:
    """
    Synchronous wrapper for create_semantic_models_for_tables.

    Returns:
        (succeeded_tables, failed_tables)
    """
    if sql_evidence_by_table is None:
        return asyncio.run(create_semantic_models_for_tables(tables, agent_config, emit))
    return asyncio.run(create_semantic_models_for_tables(tables, agent_config, emit, sql_evidence_by_table))


async def ensure_semantic_models_exist(
    tables: Set[str],
    agent_config: AgentConfig,
    emit: Optional[Callable] = None,
    sql_evidence_by_table: Optional[dict[str, list[str]]] = None,
) -> tuple[bool, str, List[str]]:
    """
    Check and create missing semantic models. Processes each table independently
    so that failures on individual tables do not block the rest.

    Args:
        tables: Set of table names to check
        agent_config: Agent configuration
        emit: Optional progress callback
        sql_evidence_by_table: Optional mapping of table lookup key to
            success-story SQL evidence.

    Returns:
        (success, error_message, created_tables) — success is True when at
        least one table was created or none were missing; error_message
        summarises any per-table failures.
    """
    missing_tables = find_missing_semantic_models(tables, agent_config)

    if not missing_tables:
        logger.info("All required semantic models already exist")
        return True, "", []

    logger.info(f"Found {len(missing_tables)} tables without semantic models: {missing_tables}")

    if sql_evidence_by_table is not None:
        succeeded, failed = await create_semantic_models_for_tables(
            missing_tables,
            agent_config,
            emit,
            sql_evidence_by_table=sql_evidence_by_table,
        )
    else:
        succeeded, failed = await create_semantic_models_for_tables(missing_tables, agent_config, emit)

    if succeeded:
        logger.info(f"Successfully created semantic models for: {succeeded}")
    if failed:
        failed_summary = "; ".join(f"{t}: {e}" for t, e in failed)
        logger.warning(f"Failed to create semantic models for some tables: {failed_summary}")

    if not succeeded and failed:
        error_msg = "; ".join(f"{t}: {e}" for t, e in failed)
        return False, error_msg, []

    error_msg = ""
    if failed:
        error_msg = "Partial failures: " + "; ".join(f"{t}: {e}" for t, e in failed)

    return True, error_msg, succeeded
