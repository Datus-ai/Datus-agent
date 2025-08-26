from typing import Any, Dict, List, Optional, override

from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SQLiteConnector(SQLAlchemyConnector):
    """
    Connector for SQLite databases.
    """

    def __init__(self, db_path: str, **kwargs):
        super().__init__(
            connection_string=(db_path if db_path.startswith("sqlite:///") else f"sqlite:///{db_path}"),
            dialect=DBType.SQLITE,
        )
        self.db_path = db_path

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return f'"{table_name}"'

    @override
    def get_materialized_views(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        return []

    @override
    def sqlalchemy_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> Optional[str]:
        # sqlite has no schema
        return None

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """SQLite not support switch context"""

    def _get_schema_with_ddl(
        self, database_name: str = "", table_type: str = "table", filter_tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        tables = self.execute_arrow_iterator(
            f"SELECT name,sql FROM sqlite_master WHERE type='{table_type}';", max_rows=10000
        )
        schema_list = []

        for table in tables:
            table_name = table[0]
            if filter_tables and table_name not in filter_tables:
                continue
            definition = table[1]

            # Skip SQLite system tables
            if table_name.startswith("sqlite_"):
                continue

            # there has a magic code: use database_name to distinguish between different database files.
            schema_list.append(
                {
                    "identifier": self.identifier(
                        database_name=database_name,
                        table_name=table_name,
                    ),
                    "catalog_name": "",
                    "database_name": database_name,
                    "schema_name": "",
                    "table_name": table_name,
                    "definition": definition,
                    "table_type": table_type,
                }
            )

        return schema_list

    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - table_name: The name of the table
            - table_schema: The CREATE TABLE statement for the table
        """
        return self._get_schema_with_ddl(
            database_name=database_name,
            table_type="table",
            filter_tables=tables,
        )

    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - table_name: The name of the table
            - table_schema: The CREATE TABLE statement for the table
        """
        return self._get_schema_with_ddl(database_name=database_name, table_type="view")

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {"db_type": DBType.SQLITE, "db_path": self.db_path}

    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        """
        SQLite has only one database (the file itself), so return the database name.

        Args:
            catalog_name: Not used in SQLite
            include_sys: Not applicable for SQLite as there's only one database

        Returns:
            List containing the database name
        """
        # For SQLite, the database is the file itself
        return ["main"]

    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        """
        SQLite has a limited schema concept. Return 'main' as the default schema.

        Args:
            catalog_name: Not used in SQLite
            database_name: Not used in SQLite
            include_sys: Not applicable as SQLite has limited system schemas

        Returns:
            List containing 'main' as the schema name
        """
        return ["main"]

    def get_type(self) -> str:
        return DBType.SQLITE
