import atexit
import threading
import weakref
from typing import Any, Dict, List, Optional, override

from datus.tools.db_tools.mysql_connector import MySQLConnectorBase
from datus.utils.exceptions import DatusException, ErrorCode
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

    def get_tables(self, **kwargs) -> List[str]:
        """Get list of tables in the database."""
        # FIXME use full name?
        result = self._get_metadatas(**kwargs)
        return [table["table_name"] for table in result]

    @override
    def sqlalchemy_schema(self, **kwargs) -> Optional[str]:
        catalog_name = kwargs.get("catalog_name", "")
        database_name = kwargs.get("database_name", "")
        if catalog_name:
            if database_name:
                return f"{catalog_name}.{database_name}"
            else:
                return catalog_name
        elif database_name:
            return database_name
        else:
            return None

    # @override
    # def catalog_valid(self) -> bool:
    #     return True

    def default_catalog(self) -> str:
        return "default_catalog"

    @override
    def ignore_schemas(self) -> List[str]:
        return ["sys", "information_schema", "_statistics_"]

    @override
    def db_meta_table_type(self) -> str:
        return "TABLE"

    def get_materialized_views_with_ddl(self, **kwargs) -> List[Dict[str, str]]:
        """
        Get all materialized views with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name)
        should be passed via kwargs and handled by subclasses as needed.
        parameters:
            catalog_name: The catalog name to filter the materialized views.
            database_name: The database name to filter the materialized views.
            schema_name: The schema name to filter the materialized views.
        """
        database_name = kwargs.get("database_name", "")
        if database_name:
            query_sql = (
                "SELECT TABLE_SCHEMA,TABLE_NAME,MATERIALIZED_VIEW_DEFINITION "
                "FROM information_schema.materialized_views "
                f"WHERE TABLE_SCHEMA = '{database_name}'"
            )
        else:
            query_sql = (
                "SELECT TABLE_SCHEMA,TABLE_NAME,MATERIALIZED_VIEW_DEFINITION "
                f"FROM information_schema.materialized_views where  not in ({str(self.ignore_schemas())[1:-1]})"
            )
        result = self.execute_query(query_sql)
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
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Get sample values from tables."""
        self.connect()
        catalog_name = self.reset_catalog_to_default(kwargs.get("catalog_name", ""))
        database_name = kwargs.get("database_name", "")
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
                res = self.execute_query(sql)
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
            for table in self._get_metadatas(**kwargs):
                sql = (
                    f"SELECT * FROM `{table['catalog_name']}`.`{table['database_name']}`.`{table['table_name']}` "
                    "LIMIT {top_n}"
                )
                res = self.execute_query(sql)
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

    def get_schema(self, table_name: str = "", **kwargs) -> List[Dict[str, str]]:
        """Get schema information for a specific table."""
        if not table_name:
            return []

        try:
            # Use DESCRIBE or SHOW COLUMNS to get table schema
            if self.database:
                sql = f"DESCRIBE `{self.database}`.`{table_name}`"
            else:
                sql = f"DESCRIBE `{table_name}`"

            result = self.execute_query(sql)
            return result.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {
            "db_type": "starrocks",
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "database": self.database,
        }

    def get_type(self) -> str:
        """Return the database type."""
        return "starrocks"

    def get_databases(self) -> List[str]:
        """Get list of available databases."""
        try:
            result = self.execute_query("SHOW DATABASES")
            if result.empty:
                return []

            # Get the database names from the first column
            database_column = result.columns[0]
            return result[database_column].tolist()
        except Exception as e:
            logger.error(f"Error getting databases: {e}")
            return []

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            self.execute_query("SELECT 1")
            return True
        except DatusException as e:
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "test_connection",
                    "error_message": str(e),
                    "uri": self.connection_string,
                },
            ) from e
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
