from typing import Any, Dict, List, Literal, Optional, Set, override

from datus.configuration.agent_config import duckdb_database_name
from datus.schemas.base import TABLE_TYPE
from datus.tools.db_tools.base import list_to_in_str
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class DuckdbConnector(SQLAlchemyConnector):
    """
    Connector for DuckDB databases.
    """

    def __init__(self, db_path: str, **kwargs):
        # Force read-only mode for DuckDB to avoid lock conflicts
        connection_string = db_path if db_path.startswith("duckdb:///") else f"duckdb:///{db_path}"
        # connection_string += "?access_mode=read_only"
        super().__init__(connection_string=connection_string)
        self.db_path = db_path
        self.database_name = duckdb_database_name(self.connection_string)

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "main", table_name: str = ""
    ) -> str:
        if database_name:
            if schema_name:
                return f'"{database_name}"."{schema_name}"."{table_name}"'
            return f'"{database_name}"."{table_name}"'
        return f'"{schema_name}"."{table_name}"' if schema_name else table_name

    @override
    def get_materialized_views(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        return []

    @override
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        sql = "select schema_name from duckdb_schemas()"
        has_where = False
        if database_name:
            sql += f" WHERE database_name='{database_name}'"
            has_where = True

        if not include_sys:
            if not has_where:
                sql += list_to_in_str(" WHERE database_name not in", list(self._sys_schemas()))
            else:
                sql += list_to_in_str(" AND database_name not in", list(self._sys_schemas()))

        schema_names = self._execute_pandas(sql)
        return schema_names["schema_name"].to_list()

    @override
    def _sys_schemas(self) -> Set[str]:
        return {"system", "temp"}

    @override
    def sqlalchemy_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> Optional[str]:
        schema_name = schema_name or self.schema_name
        database_name = database_name or self.database_name
        if database_name:
            if schema_name:
                return f"{database_name}.{schema_name}"
            return None
        return schema_name

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        if schema_name:
            self._execute(f'use "{schema_name}"')

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries.

        Returns:
            A list of dictionaries, each containing:
            - database_name: The CREATE TABLE statement for the table
        """
        filter_tables = self._reset_filter_tables(
            tables, catalog_name=catalog_name, database_name=database_name, schema_name=schema_name
        )
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
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
        tables = self._execute_pandas(query_sql)
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

    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        return self._get_meta_with_ddl(
            database_name=database_name,
            schema_name=schema_name,
            name_field="view_name",
            table_type="view",
            meta_table="duckdb_views",
        )

    @override
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
        try:
            samples = []
            if not tables:
                tables = self.get_tables(database_name=database_name, schema_name=schema_name)
            if tables:
                logger.debug(f"Getting sample data from tables {tables} LIMIT {top_n}")
                for table_name in tables:
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
                    result = self._execute_pandas(query)
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
                for table in self.get_tables_with_ddl(database_name=database_name, schema_name=schema_name):
                    query = (
                        f'SELECT * FROM "{table["database_name"]}"."{table["schema_name"]}"."{table["table_name"]}" '
                        f"LIMIT {top_n}"
                    )
                    result = self._execute_pandas(query)
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
        except DatusException:
            raise
        except Exception as e:
            raise DatusException(
                ErrorCode.DB_EXECUTION_ERROR,
                message_args={
                    "error_message": str(e),
                },
            ) from e

    @override
    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        if not table_name:
            return []
        database_name = database_name or self.database_name
        schema_name = schema_name or self.schema_name or "main"
        full_name = self.full_name(database_name=database_name, schema_name=schema_name, table_name=table_name)
        if table_name:
            sql = f"PRAGMA table_info('{full_name}')"
            try:
                return self._execute_pandas(sql).to_dict(orient="records")
            except DatusException as e:
                if "error_message" in e.message_args:
                    message = e.message_args["error_message"]
                else:
                    message = e.message
                raise DatusException(ErrorCode.DB_QUERY_METADATA_FAILED, message=message)
            except Exception as e:
                raise DatusException(
                    ErrorCode.DB_QUERY_METADATA_FAILED,
                    message_args={"error_message": str(e), "sql": sql},
                ) from e
        else:
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert connector to serializable dictionary with only essential info."""
        return {"db_type": DBType.DUCKDB, "db_path": self.db_path}

    def get_type(self) -> str:
        return DBType.DUCKDB
