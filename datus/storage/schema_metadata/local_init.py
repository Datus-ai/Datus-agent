from typing import Any, Dict, List, Set

from datus.configuration.agent_config import AgentConfig, DbConfig
from datus.schemas.base import TABLE_TYPE
from datus.storage.schema_metadata.store import SchemaWithValueRAG
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.db_manager import DBManager
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

from .init_utils import exists_table_value

logger = get_logger(__name__)


def init_local_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_manager: DBManager,
    build_mode: str = "overwrite",
    table_type: TABLE_TYPE = "full",
    init_database_name: str = "",
    pool_size: int = 4,  # TODO: support multi-threading
):
    """Initialize local schema from the configured database."""
    logger.info(f"Initializing local schema for namespace: {agent_config.current_namespace}")
    db_configs = agent_config.namespaces[agent_config.current_namespace]

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
            )
        elif db_configs.type in [DBType.MYSQL, DBType.POSTGRES, DBType.POSTGRESQL, DBType.SQLSERVER]:
            init_other_two_level_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                database_name="",
                schema_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
            )
        else:
            init_other_three_level_schema(
                table_lineage_store,
                agent_config,
                db_configs,
                db_manager,
                database_name=init_database_name,
                table_type=table_type,
                build_mode=build_mode,
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
                )
            else:
                init_duckdb_schema(
                    table_lineage_store,
                    agent_config,
                    db_config,
                    db_manager,
                    database_name=database_name,
                    schema_name=init_database_name,
                    table_type=table_type,
                    build_mode=build_mode,
                )
    # Create indices after initialization
    table_lineage_store.after_init()
    logger.info("Local schema initialization completed")


def init_sqlite_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
):
    database_name = getattr(db_config, "database", "")
    sql_connector = db_manager.get_conn(agent_config.current_namespace, database_name)
    all_schema_tables, all_value_tables = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        table_type=table_type,
        build_mode=build_mode,
    )
    logger.info(
        f"Exsits data from LanceDB {database_name}, tables={len(all_schema_tables)}, values={len(all_value_tables)}"
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
        f"Exsits data from LanceDB {database_name}, tables={len(all_schema_tables)}," f"values={len(all_value_tables)}"
    )
    sql_connector = db_manager.get_conn(agent_config.current_namespace, database_name)
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
        )


def init_other_two_level_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",
    schema_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
):
    # means schema_name here
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = schema_name or getattr(db_config, "schema", "")

    all_schema_tables, all_value_tables = exists_table_value(
        storage=table_lineage_store,
        database_name=database_name,
        schema_name=schema_name,
        table_type=table_type,
        build_mode=build_mode,
    )

    logger.info(
        f"Exsits data from LanceDB {schema_name}, tables={len(all_schema_tables)}," f"values={len(all_value_tables)}"
    )
    sql_connector = db_manager.get_conn(
        agent_config.current_namespace,
    )
    if table_type == "table" or table_type == "full":
        # Get all tables with DDL
        if hasattr(sql_connector, "get_tables_with_ddl"):
            tables = sql_connector.get_tables_with_ddl(database_name=database_name, schema_name=schema_name)
            for table in tables:
                if not table["database_name"]:
                    table["database_name"] = database_name
        else:
            # Fallback: get table names and generate basic schema
            table_names = sql_connector.get_tables(schema_name=schema_name)
            tables = []
            for table_name in table_names:
                tables.append(
                    {
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "table_name": table_name,
                        "definition": f"-- Table: {table_name} (DDL not available)",
                    }
                )
        logger.info(f"Found {len(tables)} tables from {database_name or schema_name}")
        store_tables(
            table_lineage_store,
            schema_name,
            all_schema_tables,
            all_value_tables,
            tables,
            "table",
            sql_connector,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(database_name=database_name, schema_name=schema_name)
        store_tables(
            table_lineage_store,
            schema_name,
            all_schema_tables,
            all_value_tables,
            views,
            "view",
            sql_connector,
        )

    if (table_type == "mv" or table_type == "full") and hasattr(sql_connector, "get_materialized_views_with_ddl"):
        materialized_views = sql_connector.get_materialized_views_with_ddl(database_name=database_name)
        store_tables(
            table_lineage_store,
            schema_name,
            all_schema_tables,
            all_value_tables,
            materialized_views,
            "mv",
            sql_connector,
        )


def init_other_three_level_schema(
    table_lineage_store: SchemaWithValueRAG,
    agent_config: AgentConfig,
    db_config: DbConfig,
    db_manager: DBManager,
    database_name: str = "",
    table_type: TABLE_TYPE = "table",
    build_mode: str = "overwrite",
):
    database_name = database_name or getattr(db_config, "database", "")
    schema_name = getattr(db_config, "schema", "")
    catalog_name = getattr(db_config, "catalog", "")
    sql_connector = db_manager.get_conn(agent_config.current_namespace, "")
    if hasattr(sql_connector, "default_catalog"):
        catalog_name = catalog_name or sql_connector.default_catalog()
    all_schema_tables, all_value_tables = exists_table_value(
        table_lineage_store,
        database_name=database_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=table_type,
        build_mode=build_mode,
    )

    logger.info(
        f"Exsits data from LanceDB {database_name}, tables={len(all_schema_tables)}," f"values={len(all_value_tables)}"
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
        logger.info(f"Found {len(tables)} tables from {database_name}")
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            tables,
            "table",
            sql_connector,
        )

    if (table_type == "view" or table_type == "full") and hasattr(sql_connector, "get_views_with_ddl"):
        views = sql_connector.get_views_with_ddl(
            catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            views,
            "view",
            sql_connector,
        )

    if (table_type == "mv" or table_type == "full") and hasattr(sql_connector, "get_materialized_views_with_ddl"):
        materialized_views = sql_connector.get_materialized_views_with_ddl(
            database_name=database_name, schema_name=schema_name
        )
        store_tables(
            table_lineage_store,
            database_name,
            all_schema_tables,
            all_value_tables,
            materialized_views,
            "mv",
            sql_connector,
        )


def store_tables(
    table_lineage_store: SchemaWithValueRAG,
    database_name: str,
    exists_tables: Dict[str, str],
    exists_values: Set[str],
    tables: List[Dict[str, str]],
    table_type: TABLE_TYPE,
    connector: BaseSqlConnector,
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
    new_tables: List[Dict[str, Any]] = []
    new_values: List[Dict[str, Any]] = []
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
        if identifier not in exists_tables:
            logger.debug(f"Add {table_type} {identifier}")
            new_tables.append(table)
            if identifier in exists_values:
                continue
            sample_rows = connector.get_sample_rows(
                tables=[table["table_name"]],
                top_n=5,
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
            )
            if sample_rows:
                new_values.extend(sample_rows)

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
            sample_rows = connector.get_sample_rows(
                tables=[table["table_name"]],
                top_n=5,
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
            )
            if sample_rows:
                new_values.extend(sample_rows)
        elif identifier not in exists_values:
            logger.debug(f"Just add sample rows for {identifier}")
            sample_rows = connector.get_sample_rows(
                tables=[table["table_name"]],
                top_n=5,
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                schema_name=table["schema_name"],
            )
            if sample_rows:
                new_values.extend(sample_rows)

    if new_tables or new_values:
        for item in new_values:
            item["table_type"] = table_type
        table_lineage_store.store_batch(new_tables, new_values)
        logger.info(f"Stored {len(new_tables)} {table_type}s and {len(new_values)} values for {database_name}")
    else:
        logger.info(f"No new {table_type}s or values to store for {database_name}")
