# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Shared utility functions for agentic nodes."""

from typing import Any, Dict, Optional


def resolve_database_name_for_prompt(connector: Any, connection_name: str = "") -> str:
    """Resolve the real database name from a connector for use in LLM prompts.

    In Datus, the "connection name" (agent.yml key, e.g. "dacomp_lever") may differ
    from the actual database name reported by the connector (e.g. "lever_start" for
    a DuckDB file named lever_start.duckdb). The LLM needs the real name to write
    correct SQL, not the connection routing key.

    Priority:
    1. If connection_name is provided (user explicitly specified a database), use it.
       In multi-connector mode, the connector may point to the default database, not
       the user-requested one, so connector.database_name would be wrong.
    2. If connection_name is empty, try the connector's real database name.

    Args:
        connector: Database connector instance (BaseSqlConnector or similar).
        connection_name: Connection name from user_input.database or agent_config.current_datasource.

    Returns:
        The most appropriate database name for the LLM prompt.
    """
    if connection_name:
        return connection_name
    if connector and hasattr(connector, "database_name") and connector.database_name:
        return connector.database_name
    return connection_name


def build_database_context(
    db_type: str,
    catalog: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
) -> str:
    """Build a formatted database context string for LLM prompts.

    Args:
        db_type: Database dialect type (e.g. "sqlite", "postgresql").
        catalog: Optional catalog name.
        database: Optional database name.
        schema: Optional schema name.

    Returns:
        A markdown-formatted context string.
    """
    context_parts = [f"**Dialect**: {db_type}"]
    if catalog:
        context_parts.append(f"**Catalog**: {catalog}")
    if database:
        context_parts.append(f"**Database**: {database}")
    if schema:
        context_parts.append(f"**Schema**: {schema}")
    return f"Database Context: \n\n{', '.join(context_parts)}"


def build_datasource_prompt_context(agent_config: Any) -> Dict[str, Any]:
    """Build datasource context variables for system prompt templates.

    Returns a dict with keys consumed by Jinja2 templates:
    - datasource: current datasource config key
    - current_datasource_dialect: dialect type of current datasource (e.g. "snowflake")
    - available_datasources: {name: type} dict of all configured datasources
    """
    if not agent_config:
        return {}

    current_ds = getattr(agent_config, "current_datasource", None)
    services = getattr(agent_config, "services", None)
    if not services:
        return {"datasource": current_ds}

    all_datasources = {ds_name: ds_config.type for ds_name, ds_config in services.datasources.items()}
    current_dialect = all_datasources.get(current_ds) if current_ds else None

    return {
        "datasource": current_ds,
        "current_datasource_dialect": current_dialect,
        "available_datasources": all_datasources,
    }
