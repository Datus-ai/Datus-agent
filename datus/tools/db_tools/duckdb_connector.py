from typing import Any, Dict, List, Literal, Optional, override

from sqlalchemy.exc import SQLAlchemyError

from datus.schemas.base import TABLE_TYPE
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger("duckdb_connector")


class DuckdbConnector(SQLAlchemyConnector):
    """
    Connector for DuckDB databases.
    """

    def __init__(self, db_path: str, **kwargs):
        super().__init__(
            connection_string=(db_path if db_path.startswith("duckdb:///") else f"duckdb:///{db_path}"),
        )
        self.db_path = db_path

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "main", table_name: str = ""
    ) -> str:
        if database_name:
            if schema_name:
                return f'"{database_name}"."{schema_name}"."{table_name}"'
            return f'"{database_name}"."{table_name}"'
        return f'"{schema_name}"."{table_name}"'

    @override
    def sqlalchemy_schema(self, **kwargs) -> Optional[str]:
        database_name = kwargs.get("database_name")
        schema_name = kwargs.get("schema_name")
        if database_name:
            if schema_name:
                return f"{database_name}.{schema_name}"
            return None
        return schema_name

    def get_tables_with_ddl(self, tables: Optional[List[str]] = None, **kwargs) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - database_name: The CREATE TABLE statement for the table
        """
        filter_tables = self._reset_filter_tables(tables, **kwargs)
        return self._get_meta_with_ddl(
            database_name=kwargs.get("database_name", ""),
            schema_name=kwargs.get("schema_name", ""),
            name_field="table_name",
            table_type="table",
            meta_table="duckdb_tables",
            filter_tables=filter_tables,
        )

    def _get_meta_with_ddl(
        self,
        database_name: str = "",
        schema_name: str = "",
        name_field: str = "table_name",
        table_type: TABLE_TYPE = "table",
        meta_table: Literal["duckdb_tables", "duckdb_views"] = "duckdb_tables",
        filter_tables: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        query_sql = (
            f'select database_name, schema_name, {name_field}, "sql" from {meta_table}() '
            f"where database_name != 'system'"
        )
        if database_name:
            query_sql += f" and database_name = '{database_name}'"
        if schema_name:
            query_sql += f" and schema_name = '{schema_name}'"
        tables = self.execute_query(query_sql)
        result = []
        for i in range(len(tables)):
            table_name = str(tables[name_field][i])
            full_name = self.full_name(
                database_name=str(tables["database_name"][i]),
                schema_name=str(tables["schema_name"][i]),
                table_name=table_name,
            )
            if not database_name:
                full_name = ".".join(full_name.split(".")[1:])
            if filter_tables and full_name not in filter_tables:
                continue
            result.append(
                {
                    "identifier": self.identifier(
                        database_name=str(tables["database_name"][i]),
                        schema_name=str(tables["schema_name"][i]),
                        table_name=str(table_name),
                    ),
                    "catalog_name": "",
                    "database_name": tables["database_name"][i],
                    "schema_name": tables["schema_name"][i],
                    "table_name": table_name,
                    "definition": tables["sql"][i],
                    "table_type": table_type,
                }
            )
        return result

    def get_views_with_ddl(self, **kwargs) -> List[Dict[str, str]]:
        return self._get_meta_with_ddl(
            database_name=kwargs.get("database_name", ""),
            schema_name=kwargs.get("schema_name", ""),
            name_field="view_name",
            table_type="view",
            meta_table="duckdb_views",
        )

    @override
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """Get sample values from tables."""
        try:
            self.connect()
            samples = []
            if not tables:
                tables = self.get_tables(**kwargs)
            if tables:
                logger.debug(f"Getting sample data from tables {tables} LIMIT {top_n}")
                for table_name in tables:
                    database_name = kwargs.get("database_name", "")
                    schema_name = kwargs.get("schema_name", "")
                    if schema_name:
                        if database_name:
                            prefix = f'"{database_name}"."{schema_name}"'
                        else:
                            prefix = f'"{schema_name}"'
                    else:
                        prefix = "" if not schema_name else f'"{schema_name}"'
                    if prefix:
                        query = f"""SELECT * FROM {prefix}."{table_name}" LIMIT {top_n}"""
                    else:
                        query = f"""SELECT * FROM "{table_name}" LIMIT {top_n}"""
                    result = self.execute_query(query)
                    if len(result) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": database_name,
                                "table_name": table_name,
                                "schema_name": schema_name,
                                "sample_rows": result.to_csv(index=False),
                            }
                        )
            else:
                for table in self.get_tables_with_ddl(**kwargs):
                    query = (
                        f'SELECT * FROM "{table["database_name"]}"."{table["schema_name"]}"."{table["table_name"]}" '
                        f"LIMIT {top_n}"
                    )
                    result = self.execute_query(query)
                    if len(result) > 0:
                        samples.append(
                            {
                                "catalog_name": "",
                                "database_name": table["database_name"],
                                "table_name": table["table_name"],
                                "schema_name": table["schema_name"],
                                "sample_rows": result.to_csv(index=False),
                            }
                        )
            return samples
        except SQLAlchemyError as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED,
                message_args={
                    "operation": "get_sampe_rows",
                    "error_message": str(e),
                },
            ) from e

    def get_schema(self, table_name: str = "", **kwargs) -> List[Dict[str, str]]:
        schema_name = self.sqlalchemy_schema(**kwargs)
        full_name = table_name if not schema_name else f"{schema_name}.{table_name}"
        if table_name:
            sql = f"PRAGMA table_info('{full_name}')"
            try:
                result = self.execute_query(sql).to_dict(orient="records")
            except Exception as e:
                raise DatusException(
                    ErrorCode.TOOL_DB_FAILED,
                    message_args={
                        "operation": "get_schema",
                        "error_message": str(e),
                        "uri": self.connection_string,
                    },
                ) from e
            return result
        else:
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {"db_type": "duckdb", "db_path": self.db_path}

    def get_type(self) -> str:
        return "duckdb"
