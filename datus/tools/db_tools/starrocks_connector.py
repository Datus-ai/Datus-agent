import atexit
import threading
import weakref
from typing import Any, Dict, List, Optional, override

from datus.tools.db_tools.mysql_connector import MySQLConnectorBase, list_to_in_str
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

# Global registry to track all StarRocks connections for cleanup
_starrocks_connections = weakref.WeakSet()
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _cleanup_starrocks_connections():
    """Global cleanup function for all StarRocks connections."""
    for conn in list(_starrocks_connections):
        try:
            conn._force_cleanup()
        except BaseException as e:
            logger.warning(f"StarRocks connection close error: {e}")


def _register_cleanup():
    """Register the global cleanup function to run at exit."""
    global _cleanup_registered
    with _cleanup_lock:
        if not _cleanup_registered:
            atexit.register(_cleanup_starrocks_connections)
            _cleanup_registered = True


# In order to follow the sqlalchemy connector spec, we use database as schema
class StarRocksConnector(MySQLConnectorBase):
    """
    Connector for StarRocks database
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        catalog: str = "default_catalog",
        database: str = "",
    ):
        super().__init__(host, port, user, password, database)
        self.dialect = DBType.STARROCKS
        self.catalog_name = catalog
        # Register this connection for cleanup and ensure global cleanup is registered
        _starrocks_connections.add(self)
        _register_cleanup()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with safe cleanup."""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor with safe cleanup."""
        try:
            self._force_cleanup()
        except BaseException as e:
            # Silently ignore all errors during destruction
            logger.warning(f"StarRocks connection close error: {e}")

    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get list of tables in the database."""
        # FIXME use full name?
        result = self._get_metadata(catalog_name=catalog_name, database_name=database_name)
        return [table["table_name"] for table in result]

    # @override
    # def catalog_valid(self) -> bool:
    #     return True

    def default_catalog(self) -> str:
        return "default_catalog"

    @override
    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get list of views in the database."""
        try:
            result = self._get_metadata(meta_table_name="VIEWS", catalog_name=catalog_name, database_name=database_name)
            return [view["table_name"] for view in result]
        except Exception as e:
            logger.warning(f"Failed to get views from StarRocks: {e}")
            return []

    def ignore_schemas(self) -> List[str]:
        return ["sys", "information_schema", "_statistics_"]

    @override
    def db_meta_table_type(self) -> List[str]:
        return ["TABLE", "BASE TABLE"]

    def get_materialized_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """
        Get all materialized views with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name)
        should be passed via kwargs and handled by subclasses as needed.
        parameters:
            catalog_name: The catalog name to filter the materialized views.
            database_name: The database name to filter the materialized views.
            schema_name: The schema name to filter the materialized views.
        """
        if database_name:
            query_sql = (
                "SELECT TABLE_SCHEMA,TABLE_NAME,MATERIALIZED_VIEW_DEFINITION "
                "FROM information_schema.materialized_views "
                f"WHERE TABLE_SCHEMA = '{database_name}'"
            )
        else:
            query_sql = (
                "SELECT TABLE_SCHEMA,TABLE_NAME,MATERIALIZED_VIEW_DEFINITION FROM information_schema.materialized_views"
                f"{list_to_in_str(' where TABLE_SCHEMA not in ', self.ignore_schemas())}"
            )
        result = self._execute_pandas(query_sql)
        view_list = []
        for i in range(len(result)):
            view_list.append(
                {
                    "identifier": self.identifier(
                        catalog_name=self.default_catalog(),
                        database_name=str(result["TABLE_SCHEMA"][i]),
                        table_name=str(result["TABLE_NAME"][i]),
                    ),
                    "catalog_name": self.default_catalog(),
                    "database_name": result["TABLE_SCHEMA"][i],
                    "schema_name": "",
                    "table_name": result["TABLE_NAME"][i],
                    "definition": result["MATERIALIZED_VIEW_DEFINITION"][i],
                    "table_type": "mv",
                }
            )
        return view_list

    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[Dict[str, str]]:
        """Get sample values from tables."""
        self.connect()
        catalog_name = self.reset_catalog_to_default(catalog_name)
        result = []
        if tables:
            for table_name in tables:
                if catalog_name:
                    if database_name:
                        full_table_name = f"`{catalog_name}`.`{database_name}`.`{table_name}`"
                    else:
                        full_table_name = f"`{table_name}`"
                else:
                    full_table_name = f"`{database_name}`.`{table_name}`" if database_name else f"`{table_name}`"

                sql = f"SELECT * FROM {full_table_name} LIMIT {top_n}"
                res = self._execute_pandas(sql)
                if not res.empty:
                    result.append(
                        {
                            "identifier": self.identifier(
                                catalog_name=catalog_name,
                                database_name=database_name,
                                table_name=table_name,
                            ),
                            "catalog_name": self.reset_catalog_to_default(catalog_name),
                            "database_name": database_name,
                            "schema_name": "",
                            "table_name": table_name,
                            "sample_rows": res.to_csv(index=False),
                        }
                    )
        else:
            for table in self._get_metadata(catalog_name=catalog_name, database_name=database_name):
                sql = (
                    f"SELECT * FROM `{table['catalog_name']}`.`{table['database_name']}`.`{table['table_name']}` "
                    "LIMIT {top_n}"
                )
                res = self._execute_pandas(sql)
                if not res.empty:
                    result.append(
                        {
                            "catalog_name": table["catalog_name"],
                            "database_name": table["database_name"],
                            "schema_name": "",
                            "table_name": table["table_name"],
                            "sample_rows": res.to_csv(index=False),
                        }
                    )
        return result

    @override
    def get_catalogs(self) -> List[str]:
        result = self._execute_pandas("SHOW CATALOGS")
        if result.empty:
            return []
        return result["Catalog"].tolist()

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {
            "db_type": DBType.STARROCKS,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "database": self.database,
        }

    def get_type(self) -> str:
        """Return the database type."""
        return DBType.STARROCKS

    def get_databases(self, catalog_name: str = "default_catalog", include_sys: bool = False) -> List[str]:
        """Get list of available databases."""
        return super().get_databases(catalog_name, include_sys=include_sys)

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            return super().test_connection()
        finally:
            if hasattr(self, "_conn") and self._conn:
                try:
                    self.close()
                except BaseException as e:
                    logger.warning(f"StarRocks connection close error: {e}")

    def close(self):
        """Close the database connection, handling PyMySQL struct.pack errors."""
        try:
            super().close()
        except Exception as e:
            # Handle specific PyMySQL struct.pack errors that occur during connection cleanup
            error_str = str(e)
            if (
                ("struct.error" in error_str and "required argument is not an integer" in error_str)
                or ("struct.pack" in error_str)
                or ("COMMAND.COM_QUIT" in error_str)
                or ("ROLLBACK" in error_str and "struct.pack" in error_str)
            ):
                logger.debug(f"Ignoring PyMySQL cleanup error for StarRocks connection: {e}")
                # Forcefully reset connection variables to ensure clean state
                if hasattr(self, "_conn"):
                    self._conn = None
                if hasattr(self, "engine"):
                    try:
                        # Try gentle dispose first
                        if self.engine:
                            self.engine.dispose()
                    except Exception:
                        # If that fails, just reset the variable
                        logger.warning(f"StarRocks connection close error: {e}")
                    finally:
                        self.engine = None
            else:
                # Re-raise non-PyMySQL errors
                logger.error(f"StarRocks connection close error: {e}")
                raise e

    def _force_cleanup(self):
        """Forcefully clean up the connection."""
        try:
            # Close connection while suppressing PyMySQL errors
            if hasattr(self, "_conn") and self._conn:
                try:
                    self._conn.close()
                except BaseException as e:
                    logger.warning(f"StarRocks connection close error: {e}")
                self._conn = None

            if hasattr(self, "engine") and self.engine:
                try:
                    self.engine.dispose()
                except BaseException as e:
                    logger.warning(f"StarRocks connection close error: {e}")
                self.engine = None
        except BaseException as e:
            logger.warning(f"StarRocks connection close error: {e}")
        finally:
            # Always try to remove from registry
            try:
                _starrocks_connections.discard(self)
            except BaseException as e:
                logger.warning(f"StarRocks connection close error: {e}")
