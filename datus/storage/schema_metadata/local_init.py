# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import asyncio
from typing import Any, Dict, List, Optional, Set

from datus_db_core import BaseSqlConnector, connector_registry

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.schemas.base import TABLE_TYPE
from datus.schemas.batch_events import BatchEventEmitter, BatchEventHelper
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from .init_utils import exists_table_value

logger = get_logger(__name__)

BIZ_NAME = "metadata_init"


async def init_local_schema_async(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_manager: DBManager,
    build_mode: str = "overwrite",
    table_type: TABLE_TYPE = "full",
    init_catalog_name: str = "",
    init_database_name: str = "",
    pool_size: int = 4,
    emit: Optional[BatchEventEmitter] = None,
):
    """Async version: Initialize local schema from the configured database.

    Wraps the synchronous init_local_schema in asyncio.to_thread so that
    blocking DB I/O does not stall the event loop.  The *emit* callback is
    invoked synchronously from the worker thread (it only appends to a
    queue / sends SSE, so this is safe).

    Args:
        table_lineage_store: Schema storage instance
        agent_config: Agent configuration
        db_manager: Database manager
        build_mode: "overwrite" or "incremental"
        table_type: Which object types to initialise
        init_catalog_name: Optional catalog filter
        init_database_name: Optional database filter
        pool_size: Reserved for future multi-threading support
        emit: Optional callback to stream BatchEvent progress events
    """
    await asyncio.to_thread(
        init_local_schema,
        table_lineage_store,
        agent_config,
        db_manager,
        build_mode,
        table_type,
        init_catalog_name,
        init_database_name,
        pool_size,
        emit,
    )


def init_local_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_manager: DBManager,
    build_mode: str = "overwrite",
    table_type: TABLE_TYPE = "full",
    init_catalog_name: str = "",
    init_database_name: str = "",
    pool_size: int = 4,  # TODO: support multi-threading
    emit: Optional[BatchEventEmitter] = None,
):
    """Initialize local schema from the configured database.

    Args:
        table_lineage_store: Schema storage instance
        agent_config: Agent configuration
        db_manager: Database manager
        build_mode: "overwrite" or "incremental"
        table_type: Which object types to initialise
        init_catalog_name: Optional catalog filter
        init_database_name: Optional database filter
        pool_size: Reserved for future multi-threading support
        emit: Optional callback to stream BatchEvent progress events
    """
    event_helper = BatchEventHelper(BIZ_NAME, emit)

    logger.info(f"Initializing local schema for namespace: {agent_config.current_database}")
    event_helper.task_started(namespace=agent_config.current_database, build_mode=build_mode, table_type=table_type)

    db_configs = agent_config.namespaces[agent_config.current_database]
    if len(db_configs) == 1:
        db_configs = list(db_configs.values())[0]

    if isinstance(db_configs, DbConfig):
        # Single database configuration (like StarRocks, MySQL, PostgreSQL, etc.)
        logger.info(f"Processing single database configuration: {db_configs.type}")
        if db_configs.type == DBType.SQLITE:
            init_sqlite_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                table_type=table_type,
                build_mode=build_mode,
                event_helper=event_helper,
            )
        elif db_configs.type == DBType.DUCKDB:
            init_duckdb_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                schema_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                event_helper=event_helper,
            )
        else:
            init_other_three_level_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                catalog_name=init_catalog_name,
                database_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
                event_helper=event_helper,
            )

    else:
        # Multiple database configuration (like multiple SQLite files)
        logger.info("Processing multiple database configuration")
        if not db_configs:
            logger.warning("No database configuration found")
            return

        for database_name, db_config in db_configs.items():
            logger.info(f"Processing database: {database_name}")
            if init_database_name and init_database_name != database_name:
                logger.info(f"Skip database: {database_name} because it is not the same as {init_database_name}")
                continue
            # only sqlite and duckdb support multiple databases
            if db_config.type == DBType.SQLITE:
                init_sqlite_schema(
                    table_lineage_store,
                    agent_config,
                    db_config,
                    db_manager,
                    table_type=table_type,
                    build_mode=build_mode,
                    event_helper=event_helper,
                )
            elif db_config.type == DBType.DUCKDB:
                init_duckdb_schema(
                    table_lineage_store,
                    agent_config,
                    db_config,
                    db_manager,
                    database_name=database_name,
                    schema_name=init_database_name,
                    table_type=table_type,
                    build_mode=build_mode,
                    event_helper=event_helper,
                )
            else:
                logger.warning(f"Unsupported database type {db_config.type} for multi-database configuration")
    # Create indices after initialization
    table_lineage_store.after_init()
    event_helper.task_completed(total_items=0, completed_items=0)
    logger.info("Local schema initialization completed")


def init_sqlite_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    event_helper: Optional[BatchEventHelper] = None,
):
    database_name = getattr(db_config, "database", "")
    sql_connector = db_manager.get_conn(agent_config.current_database, database_name)
    all_schema_tables, all_value_tables = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        table_type=table_type,
        build_mode=build_mode,
    )
    logger.info(
        f"Exists data from vector store {database_name}, "
        f"tables={len(all_schema_tables)}, values={len(all_value_tables)}"
    )
    if table_type == "table" or table_type == "full":
        tables = sql_connector.get_tables_with_ddl(schema_name=database_name)
        for table in tables:
            table["database_name"] = database_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            tables,
            "table",
            sql_connector,
            event_helper=event_helper,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl()
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            views,
            "view",
            sql_connector,
            event_helper=event_helper,
        )


def init_duckdb_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",  # init database_name
    schema_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    event_helper: Optional[BatchEventHelper] = None,
):
    # means schema_name here
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = schema_name or getattr(db_config, "schema", "")

    all_schema_tables, all_value_tables = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        table_type=table_type,
        build_mode=build_mode,
    )

    logger.info(
        f"Exists data from vector store {database_name}, tables={len(all_schema_tables)},values={len(all_value_tables)}"
    )
    sql_connector = db_manager.get_conn(agent_config.current_database, database_name)
    if table_type == "table" or table_type == "full":
        # Get all tables with DDL
        tables = sql_connector.get_tables_with_ddl(schema_name=schema_name)
        for table in tables:
            if not table.get("database_name"):
                table["database_name"] = database_name

        logger.info(f"Found {len(tables)} tables")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            tables,
            "table",
            sql_connector,
            event_helper=event_helper,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(schema_name=database_name)
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            views,
            "view",
            sql_connector,
            event_helper=event_helper,
        )


def init_other_three_level_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    catalog_name: str = "",
    database_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
    event_helper: Optional[BatchEventHelper] = None,
):
    db_type = db_config.type
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = getattr(db_config, "schema", "")
    catalog_name = catalog_name or getattr(db_config, "catalog", "")

    sql_connector = db_manager.get_conn(agent_config.current_database)

    if not database_name and hasattr(sql_connector, "database_name"):
        database_name = getattr(sql_connector, "database_name", "")

    if hasattr(sql_connector, "default_catalog"):
        catalog_name = catalog_name or sql_connector.default_catalog()
    elif hasattr(sql_connector, "catalog_name"):
        catalog_name = catalog_name or getattr(sql_connector, "catalog_name", "")
    if not connector_registry.support_schema(db_type):
        schema_name = ""
    elif not schema_name and hasattr(sql_connector, "schema_name"):
        schema_name = getattr(sql_connector, "schema_name", "")

    all_schema_tables, all_value_tables = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=table_type,
        build_mode=build_mode,
    )

    logger.info(
        f"Exists data from vector store {catalog_name or '[no catalog]'}.{database_name}, "
        f"tables={len(all_schema_tables)}, values={len(all_value_tables)}"
    )
    if table_type == "table" or table_type == "full":
        # Get all tables with DDL
        if hasattr(sql_connector, "get_tables_with_ddl"):
            tables = sql_connector.get_tables_with_ddl(
                catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
            )
        else:
            # Fallback: get table names and generate basic schema
            table_names = sql_connector.get_tables(database_name=database_name, schema_name=schema_name)
            tables = []
            for table_name in table_names:
                tables.append(
                    {
                        "identifier": sql_connector.identifier(
                            catalog_name=catalog_name,
                            database_name=database_name,
                            schema_name=schema_name,
                            table_name=table_name,
                        ),
                        "catalog_name": catalog_name,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "definition": f"-- Table: {table_name} (DDL not available)",
                    }
                )
        for table in tables:
            if not table.get("catalog_name"):
                table["catalog_name"] = catalog_name
            if not table.get("database_name"):
                table["database_name"] = database_name
            if not connector_registry.support_schema(db_type):
                table["schema_name"] = ""
            elif not table.get("schema_name"):
                table["schema_name"] = schema_name
        logger.info(f"Found {len(tables)} tables from {database_name}")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            tables,
            "table",
            sql_connector,
            event_helper=event_helper,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        for view in views:
            if not view.get("catalog_name"):
                view["catalog_name"] = catalog_name
            if not view.get("database_name"):
                view["database_name"] = database_name
            if not connector_registry.support_schema(db_type):
                view["schema_name"] = ""
            elif not view.get("schema_name"):
                view["schema_name"] = schema_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            views,
            "view",
            sql_connector,
            event_helper=event_helper,
        )
    if (table_type == "mv" or table_type == "full") and hasattr(sql_connector, "get_materialized_views_with_ddl"):
        materialized_views = sql_connector.get_materialized_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        for mv in materialized_views:
            if not mv.get("catalog_name"):
                mv["catalog_name"] = catalog_name
            if not mv.get("database_name"):
                mv["database_name"] = database_name
            if not connector_registry.support_schema(db_type):
                mv["schema_name"] = ""
            elif not mv.get("schema_name"):
                mv["schema_name"] = schema_name
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            materialized_views,
            "mv",
            sql_connector,
            event_helper=event_helper,
        )


def store_tables(
    table_lineage_store: SchemaWithValueRAG,
    database_name: str,
    exists_tables: Dict[str, str],
    exists_values: Set[str],
    tables: List[Dict[str, str]],
    table_type: TABLE_TYPE,
    connector: BaseSqlConnector,
    event_helper: Optional[BatchEventHelper] = None,
):
    """
    Store tables to the table_lineage_store.
    params:
        exists_tables: {full_name: schema_text}
        return the new tables.
    """
    if not tables:
        logger.info(f"No schemas of {table_type} to store for {database_name}")
        return

    if event_helper:
        event_helper.group_started(group_id=database_name, total_items=len(tables), table_type=table_type)

    new_tables: List[Dict[str, Any]] = []
    new_values: List[Dict[str, Any]] = []
    completed_count = 0
    failed_count = 0
    for table in tables:
        if not table.get("database_name"):
            table["database_name"] = database_name
        if not table.get("identifier"):
            table["identifier"] = connector.identifier(
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
                table_name=table["table_name"],
            )

        identifier = table["identifier"]
        try:
            if identifier not in exists_tables:
                logger.debug(f"Add {table_type} {identifier}")
                new_tables.append(table)
                if identifier not in exists_values:
                    _fill_sample_rows(
                        new_values=new_values, identifier=identifier, table_data=table, connector=connector
                    )

            elif exists_tables[identifier] != table["definition"]:
                # update table and value
                logger.debug(f"Update {table_type} {identifier}")
                table_lineage_store.remove_data(
                    catalog_name=table["catalog_name"],
                    database_name=table["database_name"],
                    schema_name=table["schema_name"],
                    table_name=table["table_name"],
                    table_type=table_type,
                )
                new_tables.append(table)

                _fill_sample_rows(new_values=new_values, identifier=identifier, table_data=table, connector=connector)

            elif identifier not in exists_values:
                logger.debug(f"Just add sample rows for {identifier}")

                _fill_sample_rows(new_values=new_values, identifier=identifier, table_data=table, connector=connector)

            completed_count += 1
            if event_helper:
                event_helper.item_completed(
                    item_id=identifier,
                    group_id=database_name,
                    table_name=table.get("table_name", ""),
                    table_type=table_type,
                )
        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed to process {table_type} {identifier}: {e}")
            if event_helper:
                event_helper.item_failed(
                    item_id=identifier,
                    error=str(e),
                    group_id=database_name,
                    exception_type=type(e).__name__,
                )

    if new_tables or new_values:
        for item in new_values:
            item["table_type"] = table_type
        table_lineage_store.store_batch(new_tables, new_values)
        logger.info(f"Stored {len(new_tables)} {table_type}s and {len(new_values)} values for {database_name}")
    else:
        logger.info(f"No new {table_type}s or values to store for {database_name}")

    if event_helper:
        event_helper.group_completed(
            group_id=database_name,
            total_items=len(tables),
            completed_items=completed_count,
            failed_items=failed_count,
            table_type=table_type,
        )


def _fill_sample_rows(
    new_values: List[Dict[str, Any]], identifier: str, table_data: Dict[str, Any], connector: BaseSqlConnector
):
    try:
        sample_rows = connector.get_sample_rows(
            tables=[table_data["table_name"]],
            top_n=5,
            catalog_name=table_data["catalog_name"],
            database_name=table_data["database_name"],
            schema_name=table_data["schema_name"],
        )
        if sample_rows:
            for row in sample_rows:
                if not row.get("identifier"):
                    row["identifier"] = identifier
            new_values.extend(sample_rows)
    except Exception as e:
        logger.warning(f"Failed to select sample rows for {identifier}: {e}")
