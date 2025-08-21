from collections import defaultdict
from typing import Dict, Optional, Tuple, Union
from urllib.parse import quote_plus

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


def gen_uri(db_config: DbConfig) -> str:
    if db_config.uri:
        return db_config.uri

    elif db_config.type == DBType.SNOWFLAKE:
        return (
            f"snowflake://{quote_plus(db_config.username)}:{quote_plus(str(db_config.password))}"
            f"@{db_config.account}/?warehouse={db_config.warehouse}"
        )
    elif db_config.type == DBType.STARROCKS:
        catalog = getattr(db_config, "catalog", "default_catalog") or "default_catalog"
        return (
            f"starrocks://{quote_plus(db_config.username)}:{quote_plus(str(db_config.password))}"
            f"@{db_config.host}:{db_config.port}/{catalog}.{db_config.database}"
        )
    else:
        db_name = "" if not db_config.database else f"/{db_config.database}"
        return (
            f"{db_config.type}://{quote_plus(db_config.username)}:{quote_plus(str(db_config.password))}"
            f"@{db_config.host}:{db_config.port}{db_name}"
        )


class DBManager:
    def __init__(self, db_configs: Dict[str, Dict[str, DbConfig]]):
        self._conn_dict: Dict[str, Union[BaseSqlConnector, Dict[str, BaseSqlConnector]]] = defaultdict(dict)
        self._db_configs: Dict[str, Dict[str, DbConfig]] = db_configs

    def get_conn(self, namespace: str, db_name: str = "") -> BaseSqlConnector:
        self._init_connections(namespace)
        connector_or_dict = self._conn_dict[namespace]
        if isinstance(connector_or_dict, Dict):
            if not db_name:
                return list(connector_or_dict.values())[0]
            if db_name not in connector_or_dict:
                raise DatusException(
                    code=ErrorCode.DB_CONNECTION_FAILED,
                    message_args={
                        "error_message": f"Database {db_name} not found in namespace {namespace}",
                    },
                )
            return connector_or_dict[db_name]
        else:
            return connector_or_dict

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
        return config.database, dbs

    def get_db_uris(self, namespace: str) -> Dict[str, str]:
        dbs = self._db_configs.get(namespace, {})
        return {name: db.uri for name, db in dbs.items()}

    def _init_conn(self, namespace: str, db_config: DbConfig, database_name: Optional[str] = None) -> BaseSqlConnector:
        if db_config.type == DBType.SQLITE:
            conn: BaseSqlConnector = SQLiteConnector(db_config.uri)
        elif db_config.type == DBType.DUCKDB:
            conn = DuckdbConnector(db_config.uri)
        elif db_config.type == DBType.SNOWFLAKE:
            conn = SnowflakeConnector(
                account=db_config.account,
                user=db_config.username,
                password=db_config.password,
                warehouse=db_config.warehouse,
                database=db_config.database,
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
            conn = SQLAlchemyConnector(db_config.uri)
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
                    # try commit or rollback
                    # if hasattr(conn, 'engine') and conn.engine:
                    #     with conn.engine.connect() as connection:
                    #         try:
                    #             connection.commit()
                    #         except:
                    #             connection.rollback()
                    # then close connection
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
