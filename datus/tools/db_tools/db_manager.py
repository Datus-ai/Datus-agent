from typing import Any, Dict, Optional
from urllib.parse import quote_plus

from datus.configuration.agent_config import DbConfig
from datus.tools.db_tools.base import BaseSqlConnector
from datus.tools.db_tools.duckdb_connector import DuckdbConnector
from datus.tools.db_tools.mysql_connector import MySQLConnector
from datus.tools.db_tools.snowflake_connector import SnowflakeConnector
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.tools.db_tools.sqlite_connector import SQLiteConnector
from datus.tools.db_tools.starrocks_connector import StarRocksConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


def gen_uri(db_config: DbConfig) -> str:
    if db_config.uri:
        return db_config.uri

    elif db_config.type == "snowflake":
        return (
            f"snowflake://{quote_plus(db_config.username)}:{quote_plus(str(db_config.password))}"
            f"@{db_config.account}/?warehouse={db_config.warehouse}"
        )
    elif db_config.type == "starrocks":
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
    def __init__(self, db_configs: Dict[str, DbConfig]):
        self._conn_dict: Dict[str, Optional[BaseSqlConnector]] = {}
        self._db_configs: Dict[str, Dict[str, Any]] = {}
        for name, db_config in db_configs.items():
            if not db_config.uri:
                uri = gen_uri(db_config)
            else:
                uri = db_config.uri

            self._db_configs[name] = {
                "type": db_config.type,
                "uri": uri,
                "host": getattr(db_config, "host", ""),
                "port": getattr(db_config, "port", 0),
                "account": getattr(db_config, "account", ""),
                "username": getattr(db_config, "username", ""),
                "password": str(getattr(db_config, "password", "")),
                "warehouse": getattr(db_config, "warehouse", ""),
                "database": getattr(db_config, "database", ""),
                "catalog": getattr(db_config, "catalog", ""),
            }

    def get_conn(self, name: str, db_type: str, db_name: str = "") -> BaseSqlConnector:
        current_name = db_config_name(name, db_type, db_name)
        if current_name not in self._db_configs:
            raise DatusException(
                code=ErrorCode.TOOL_DB_FAILED,
                message=f"Database config not found, namespace: {name}, db_type: {db_type}, name: {db_name}",
            )

        return self._get_conn(current_name, self._db_configs[current_name])

    def _get_conn(self, current_name: str, config: Dict[str, str]) -> BaseSqlConnector:
        if current_name not in self._conn_dict:
            self._init_conn(current_name, config)
        conn = self._conn_dict[current_name]
        if conn is None:
            self._init_conn(current_name, config)
        conn = self._conn_dict[current_name]
        return conn

    def first_conn(self, namespace: str) -> BaseSqlConnector:
        for k, v in self._db_configs.items():
            if k == namespace or k.startswith(f"{namespace}::"):
                return self._get_conn(k, v)
        raise DatusException(
            code=ErrorCode.TOOL_DB_FAILED,
            message=f"Database config not found, namespace: {namespace}",
        )

    def get_db_uris(self, namespace: str) -> Dict[str, str]:
        result = {}
        for k, v in self._db_configs.items():
            if k == namespace:
                return v["uri"]
            elif k.startswith(f"{namespace}::"):
                a, b = k.split("::")
                if a == namespace:
                    result[b] = v["uri"]
        return result

    def _init_conn(self, name: str, db_config: Dict[str, str]) -> BaseSqlConnector:
        if db_config["type"] == "sqlite":
            conn = SQLiteConnector(db_config["uri"])
        elif db_config["type"] == "duckdb":
            conn = DuckdbConnector(db_config["uri"])
        elif db_config["type"] == "snowflake":
            conn = SnowflakeConnector(
                account=self._db_configs[name]["account"],
                user=self._db_configs[name]["username"],
                password=self._db_configs[name]["password"],
                warehouse=self._db_configs[name]["warehouse"],
                database=self._db_configs[name]["database"],
            )
        elif db_config["type"] == "mysql":
            conn = MySQLConnector(
                host=self._db_configs[name]["host"],
                port=self._db_configs[name]["port"],
                user=self._db_configs[name]["username"],
                password=self._db_configs[name]["password"],
                database=self._db_configs[name].get("database", ""),
            )
        elif db_config["type"] == "starrocks":
            conn = StarRocksConnector(
                host=self._db_configs[name]["host"],
                port=self._db_configs[name]["port"],
                user=self._db_configs[name]["username"],
                password=self._db_configs[name]["password"],
                catalog=self._db_configs[name]["catalog"] or "default_catalog",
                database=self._db_configs[name]["database"],
            )
        else:
            conn = SQLAlchemyConnector(db_config["uri"])

        self._conn_dict[name] = conn
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
    if db_type == "sqlite" or db_type == "duckdb":
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
    configs = {}

    if db_configs is None:
        return DBManager({})

    for name, db_config in db_configs.items():
        for db_name, db in db_config.items():
            full_name = db_config_name(name, db.type, db_name)
            if full_name in configs:
                raise ValueError(f"Database config for {full_name} already exists")
            configs[full_name] = db
    manager = DBManager(configs)
    return manager
