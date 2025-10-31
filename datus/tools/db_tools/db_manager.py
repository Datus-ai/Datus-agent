# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import re
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union
from urllib.parse import unquote

from sqlalchemy.engine.url import URL, make_url

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.tools.db_tools.mysql_connector import MySQLConnector
from datus.tools.db_tools.snowflake_connector import SnowflakeConnector
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.tools.db_tools.sqlite_connector import SQLiteConnector
from datus.tools.db_tools.starrocks_connector import StarRocksConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def _normalize_dialect_name(db_type: Union[str, DBType, None]) -> str:
    """
    Normalize dialect names and collapse aliases so downstream checks work reliably.
    """
    if isinstance(db_type, DBType):
        value = db_type.value
    else:
        value = str(db_type or "").strip().lower()
    alias_map = {
        DBType.POSTGRES.value: DBType.POSTGRESQL.value,
        DBType.SQLSERVER.value: DBType.MSSQL.value,
    }
    return alias_map.get(value, value)


def _clean_str(value: Optional[Union[str, int]]) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item:
                return str(item).strip()
        return ""
    return str(value).strip()


def _resolve_connection_context(db_config: DbConfig, uri: str) -> Tuple[str, str, str, str]:
    """
    Infer catalog, database, and schema information from a SQLAlchemy URL.
    Returns (dialect, catalog_name, database_name, schema_name).
    """
    normalized_type = _normalize_dialect_name(db_config.type)
    try:
        url = make_url(uri)
    except Exception as exc:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Invalid database uri `{uri}`: {exc}",
        ) from exc

    backend_normalized = _normalize_dialect_name(url.get_backend_name())
    dialect = backend_normalized or normalized_type
    if not dialect:
        raise DatusException(
            code=ErrorCode.COMMON_CONFIG_ERROR,
            message=f"Unable to determine database type from uri `{uri}`",
        )

    query_params: Dict[str, str] = {k: _clean_str(v) for k, v in url.query.items()}
    catalog = ""
    database = _clean_str(url.database)
    schema = ""

    if dialect == DBType.POSTGRESQL.value:
        database = database or _clean_str(db_config.database)
        schema = (
            query_params.get("currentSchema")
            or query_params.get("schema")
            or _extract_schema_from_pg_options(query_params.get("options", ""))
            or _clean_str(db_config.schema)
            or "public"
        )
        catalog = ""
        dialect = DBType.POSTGRESQL.value
    elif dialect == DBType.CLICKHOUSE.value:
        database = database or _clean_str(db_config.database) or "default"
        schema = _clean_str(db_config.schema)
        catalog = _clean_str(db_config.catalog)
    elif dialect == DBType.BIGQUERY.value:
        catalog = _clean_str(url.host) or _clean_str(db_config.catalog)
        dataset = database or _clean_str(db_config.database) or _clean_str(db_config.schema)
        database = dataset
        schema = query_params.get("schema") or dataset
    elif dialect == DBType.MSSQL.value:
        database = database or _clean_str(db_config.database)
        schema = query_params.get("schema") or _clean_str(db_config.schema) or "dbo"
        catalog = ""
        dialect = DBType.MSSQL.value
    elif dialect == DBType.ORACLE.value:
        service = query_params.get("service_name") or query_params.get("sid")
        database = service or database or _clean_str(db_config.database)
        schema = query_params.get("schema") or _clean_str(db_config.schema) or _clean_str(url.username)
        catalog = ""
        dialect = DBType.ORACLE.value
    else:
        catalog = _clean_str(db_config.catalog)
        database = database or _clean_str(db_config.database)
        schema = _clean_str(db_config.schema)

    return dialect or "", catalog, database, schema


def _extract_schema_from_pg_options(options: str) -> str:
    if not options:
        return ""
    decoded = unquote(options)
    match = re.search(r"search_path\s*=\s*([^ ,]+)", decoded, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1)
    if "," in value:
        value = value.split(",", 1)[0]
    return value.strip()


def gen_uri(db_config: DbConfig) -> str:
    if db_config.uri:
        return db_config.uri

    normalized_type = _normalize_dialect_name(db_config.type)

    if normalized_type == DBType.POSTGRESQL.value:
        return str(
            URL.create(
                drivername="postgresql+psycopg",
                username=_value_or_none(db_config.username),
                password=_value_or_none(db_config.password),
                host=_value_or_none(db_config.host),
                port=_port_or_none(db_config.port),
                database=_value_or_none(db_config.database),
            )
        )
    if normalized_type == DBType.CLICKHOUSE.value:
        return str(
            URL.create(
                drivername="clickhouse",
                username=_value_or_none(db_config.username),
                password=_value_or_none(db_config.password),
                host=_value_or_none(db_config.host),
                port=_port_or_none(db_config.port),
                database=_value_or_none(db_config.database),
            )
        )
    if normalized_type == DBType.BIGQUERY.value:
        project = _clean_str(db_config.catalog) or _clean_str(db_config.host)
        dataset = _clean_str(db_config.database) or _clean_str(db_config.schema)
        if not project or not dataset:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR,
                message="BigQuery configuration requires `catalog` (project) and `database` (dataset)",
            )
        return str(URL.create(drivername="bigquery", host=project, database=dataset))
    if normalized_type == DBType.MSSQL.value:
        query: Dict[str, str] = {"driver": "ODBC Driver 17 for SQL Server"}
        if db_config.schema:
            query["schema"] = _clean_str(db_config.schema)
        return str(
            URL.create(
                drivername="mssql+pyodbc",
                username=_value_or_none(db_config.username),
                password=_value_or_none(db_config.password),
                host=_value_or_none(db_config.host),
                port=_port_or_none(db_config.port),
                database=_value_or_none(db_config.database),
                query=query,
            )
        )

    if normalized_type == DBType.ORACLE.value:
        query = {}
        service = _clean_str(db_config.database)
        sid = _clean_str(db_config.schema)
        if service:
            query["service_name"] = service
        elif sid:
            query["sid"] = sid
        return str(
            URL.create(
                drivername="oracle+cx_oracle",
                username=_value_or_none(db_config.username),
                password=_value_or_none(db_config.password),
                host=_value_or_none(db_config.host),
                port=_port_or_none(db_config.port),
                query=query or None,
            )
        )

    return str(
        URL.create(
            drivername=normalized_type,
            username=_value_or_none(db_config.username),
            password=_value_or_none(db_config.password),
            host=_value_or_none(db_config.host),
            port=_port_or_none(db_config.port),
            database=_value_or_none(db_config.database),
        )
    )


def _value_or_none(value: Optional[Union[str, int]]) -> Optional[str]:
    cleaned = _clean_str(value)
    return cleaned or None


def _port_or_none(port_value: Optional[Union[str, int]]) -> Optional[int]:
    cleaned = _clean_str(port_value)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def get_connection(
    connections: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]], logic_name: str = ""
) -> BaseSqlConnector:
    if isinstance(connections, BaseSqlConnector):
        return connections
    if len(connections) == 1:
        return next(iter(connections.values()))

    if not logic_name:
        return list(connections.values())[0]
    if logic_name not in connections:
        raise DatusException(
            code=ErrorCode.DB_CONNECTION_FAILED,
            message_args={
                "error_message": f"Database {logic_name} not found in current namespace",
            },
        )
    return connections[logic_name]


class DBManager:
    def __init__(self, db_configs: Dict[str, Dict[str, DbConfig]]):
        self._conn_dict: Dict[str, Union[BaseSqlConnector, Dict[str, BaseSqlConnector]]] = defaultdict(dict)
        self._db_configs: Dict[str, Dict[str, DbConfig]] = db_configs

    def get_conn(self, namespace: str, logic_name: str = "") -> BaseSqlConnector:
        self._init_connections(namespace)
        connector_or_dict = self._conn_dict[namespace]
        return get_connection(connector_or_dict, logic_name)

    def get_connections(self, namespace: str = "") -> Union[BaseSqlConnector, Dict[str, BaseSqlConnector]]:
        self._init_connections(namespace)
        return self._conn_dict[namespace]

    def current_db_configs(self, namespace: str) -> Dict[str, DbConfig]:
        return self._db_configs[namespace]

    def _init_connections(self, namespace):
        if namespace in self._conn_dict:
            return
        if namespace not in self._db_configs:
            raise DatusException(
                code=ErrorCode.COMMON_CONFIG_ERROR, message=f"Namespace {namespace} not found in config"
            )
        configs = self._db_configs[namespace]
        if len(configs) == 1:
            db_config = list(configs.values())[0]
            self._init_conn(namespace, db_config)
            return
        # Multiple database configuration
        for database_name, db_config in configs.items():
            self._init_conn(namespace, db_config, database_name=database_name)

        if namespace not in self._conn_dict:
            raise DatusException(
                ErrorCode.COMMON_CONFIG_ERROR,
                message=(
                    f"Database initialization under namespace {namespace} failed with the current configuration:"
                    f" {configs}"
                ),
            )

    def first_conn(self, namespace: str) -> BaseSqlConnector:
        self._init_connections(namespace)
        dbs: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]] = self._conn_dict[namespace]
        if isinstance(dbs, dict):
            return list(dbs.values())[0]
        return dbs

    def first_conn_with_name(self, namespace: str) -> Tuple[str, BaseSqlConnector]:
        self._init_connections(namespace)
        dbs: Union[BaseSqlConnector, Dict[str, BaseSqlConnector]] = self._conn_dict[namespace]
        if isinstance(dbs, dict):
            name = list(dbs.keys())[0]
            conn = dbs[name]
            return name, conn

        config = list(self._db_configs[namespace].values())[0]
        return config.logic_name, dbs

    def get_db_uris(self, namespace: str) -> Dict[str, str]:
        dbs = self._db_configs.get(namespace, {})
        return {name: db.uri for name, db in dbs.items()}

    def _init_conn(self, namespace: str, db_config: DbConfig, database_name: Optional[str] = None) -> BaseSqlConnector:
        if db_config.type == DBType.SQLITE:
            conn: BaseSqlConnector = SQLiteConnector(db_config.uri, database_name=db_config.database)
        elif db_config.type == DBType.DUCKDB:
            conn = DuckdbConnector(db_config.uri, database_name=db_config.database)
        elif db_config.type == DBType.SNOWFLAKE:
            conn = SnowflakeConnector(
                account=db_config.account,
                user=db_config.username,
                password=db_config.password,
                warehouse=db_config.warehouse,
                database=db_config.database,
                schema=db_config.schema,
            )
        elif db_config.type == DBType.MYSQL:
            conn = MySQLConnector(
                host=db_config.host,
                port=int(db_config.port) if db_config.port else 0,
                user=db_config.username,
                password=db_config.password,
                database=db_config.database,
            )
        elif db_config.type == DBType.STARROCKS:
            conn = StarRocksConnector(
                host=db_config.host,
                port=int(db_config.port) if db_config.port else 0,
                user=db_config.username,
                password=db_config.password,
                catalog=db_config.catalog or "default_catalog",
                database=db_config.database,
            )
        else:
            connection_uri = db_config.uri
            if not connection_uri:
                connection_uri = gen_uri(db_config)
                dialect = _normalize_dialect_name(db_config.type)
                catalog_name = db_config.catalog or ""
                inferred_database = db_config.database or ""
                inferred_schema = db_config.schema or ""
            else:
                dialect, catalog_name, inferred_database, inferred_schema = _resolve_connection_context(
                    db_config, connection_uri
                )
            if not dialect:
                dialect = _normalize_dialect_name(db_config.type)
            conn = SQLAlchemyConnector(connection_uri, dialect=dialect)
            if catalog_name:
                conn.catalog_name = catalog_name
            if inferred_database:
                conn.database_name = inferred_database
            if inferred_schema:
                conn.schema_name = inferred_schema

        if database_name:
            self._conn_dict[namespace][database_name] = conn
        else:
            self._conn_dict[namespace] = conn
        return conn

    def close(self):
        """Close all database connections."""
        for name, conn in list(self._conn_dict.items()):
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {name}: {str(e)}")
                finally:
                    self._conn_dict[name] = None

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()


def db_config_name(namespace: str, db_type: str, name: str = "") -> str:
    if db_type == DBType.SQLITE or db_type == DBType.DUCKDB:
        return f"{namespace}::{name}"
    # fix local snowflake
    return f"{namespace}::{namespace}"


_INSTANCE = None


def db_manager_instance(
    db_configs: Optional[Dict[str, Dict[str, DbConfig]]] = None,
) -> DBManager:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _db_manager(db_configs)
    return _INSTANCE


def _db_manager(
    db_configs: Optional[Dict[str, Dict[str, DbConfig]]] = None,
) -> DBManager:
    if db_configs is None:
        return DBManager({})
    manager = DBManager(db_configs)
    return manager
