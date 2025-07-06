from typing import Any, Dict, List, Optional, override

from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.loggings import get_logger

logger = get_logger("sqlite_connector")


class SQLiteConnector(SQLAlchemyConnector):
    """
    Connector for SQLite databases.
    """

    def __init__(self, db_path: str, **kwargs):
        super().__init__(
            connection_string=(db_path if db_path.startswith("sqlite:///") else f"sqlite:///{db_path}"),
            dialect="sqlite",
        )
        self.db_path = db_path

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return f'"{table_name}"'

    @override
    def sqlalchemy_schema(self, **kwargs) -> Optional[str]:
        # sqlite has no schema
        return None

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

    def get_tables_with_ddl(self, tables: Optional[List[str]] = None, **kwargs) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - table_name: The name of the table
            - table_schema: The CREATE TABLE statement for the table
        """
        return self._get_schema_with_ddl(
            database_name=kwargs.get("database_name", "") or kwargs.get("schema_name", ""),
            table_type="table",
            filter_tables=tables,
        )

    def get_views_with_ddl(self, **kwargs) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - table_name: The name of the table
            - table_schema: The CREATE TABLE statement for the table
        """
        return self._get_schema_with_ddl(
            database_name=kwargs.get("database_name", "") or kwargs.get("schema_name", ""), table_type="view"
        )

    def get_schema(self, table_name: str = "", **kwargs) -> List[Dict[str, str]]:
        if table_name:
            sql = f"PRAGMA table_info('{table_name}')"
            result = self.execute_query(sql).to_dict(orient="records")
            return result
        else:
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {"db_type": "sqlite", "db_path": self.db_path}  # 假设类中有db_path属性存储数据库路径

    def get_type(self) -> str:
        return "sqlite"
