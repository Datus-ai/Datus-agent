from typing import Any, Dict, List, Literal, Optional, Set, Tuple, override
from urllib.parse import quote_plus

from datus.schemas.base import TABLE_TYPE
from datus.tools.db_tools.base import list_to_in_str
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.constants import DBType
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

CREATE_TYPE = Literal["TABLE", "VIEW", "MATERIALIZED VIEW"]
META_TABLE_NAMES = Literal["TABLES", "VIEWS", "MATERIALIZED_VIEWS"]


def db_table_type_to_inner(db_table_type: str) -> TABLE_TYPE:
    if db_table_type == "VIEW":
        return "view"
    elif db_table_type == "MATERIALIZED VIEW":
        return "mv"
    return "table"


def inner_table_type_to_db(table_type: TABLE_TYPE) -> Tuple[META_TABLE_NAMES, CREATE_TYPE]:
    if table_type == "table":
        return ("TABLES", "TABLE")
    elif table_type == "view":
        return ("VIEWS", "VIEW")
    elif table_type == "mv":
        return ("MATERIALIZED_VIEWS", "MATERIALIZED VIEW")
    else:
        raise DatusException(ErrorCode.COMMON_FIELD_INVALID, f"Invalid table type: {table_type}")


class MySQLConnectorBase(SQLAlchemyConnector):
    def __init__(self, host: str, port: int, user: str, password: str, database: str = ""):
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = str(password) if password else ""
        self.database = database
        # StarRocks uses MySQL protocol, use mysql+pymysql driver with specific pool settings
        # URL encode the password to handle special characters like @, :, etc.
        encoded_password = quote_plus(self.password) if self.password else ""
        # Use mysql+pymysql instead of starrocks:// to avoid pymysql struct.pack issues
        self.connection_string = (
            f"mysql+pymysql://{user}:{encoded_password}@{host}:{self.port}/{database}?charset=utf8mb4&autocommit=true"
        )
        super().__init__(self.connection_string, dialect=DBType.MYSQL)

    def _get_metadata(
        self,
        meta_table_name: META_TABLE_NAMES = "TABLES",
        inner_table_type: Optional[List[str]] = None,
        catalog_name: str = "",
        database_name: str = "",
    ) -> List[Dict[str, str]]:
        self.connect()
        database_name = database_name or self.database_name
        catalog = self.reset_catalog_to_def(catalog_name or self.catalog_name)
        where = f"TABLE_CATALOG = '{catalog}'"
        if database_name:
            where = f"{where} AND TABLE_SCHEMA = '{database_name}'"
        else:
            where = f"{where} {list_to_in_str('and TABLE_SCHEMA not in', list(self._sys_databases()))}"

        query_result = self._execute_pandas(
            (
                f"SELECT TABLE_CATALOG, TABLE_SCHEMA,TABLE_NAME FROM information_schema.{meta_table_name} WHERE {where}"
                f"{list_to_in_str(' and TABLE_TYPE in ', inner_table_type)}"
            )
        )
        result = []
        for i in range(len(query_result)):
            catalog = self.reset_catalog_to_default(str(query_result["TABLE_CATALOG"][i]))
            result.append(
                {
                    "catalog_name": catalog,
                    "database_name": query_result["TABLE_SCHEMA"][i],
                    "table_name": query_result["TABLE_NAME"][i],
                }
            )
        return result

    @override
    def _sys_databases(self) -> Set[str]:
        return {"sys", "information_schema", "performance_schema", "mysql"}

    @override
    def _sys_schemas(self) -> Set[str]:
        return {"sys", "information_schema", "performance_schema", "mysql"}

    def default_catalog(self) -> str:
        return ""

    def get_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, Any]]:
        if not table_name:
            return []

        catalog_name = self.reset_catalog_to_default(catalog_name or self.catalog_name)
        database_name = database_name or self.database_name
        table_name = self.full_name(catalog_name=catalog_name, database_name=database_name, table_name=table_name)
        # Use DESCRIBE or SHOW COLUMNS to get table schema
        sql = f"DESCRIBE {table_name}"
        query_result = self._execute_pandas(sql)
        result = []
        for i in range(len(query_result)):
            result.append(
                {
                    "cid": i,
                    "name": query_result["Field"][i],
                    "type": query_result["Type"][i],
                    "nullable": query_result["Null"][i] == "YES",
                    "default_value": query_result["Default"][i],
                    "pk": query_result["Key"][i] == "PRI",
                }
            )
        return result

    @override
    def sqlalchemy_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> Optional[str]:
        database_name = database_name or self.database_name
        if self.default_catalog():
            # catalog support
            catalog_name = catalog_name or self.catalog_name or self.default_catalog()
            if database_name:
                return f"{catalog_name}.{database_name}"
            return None
        else:
            return database_name if database_name else None

    @override
    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        if self.default_catalog():
            if catalog_name:
                self._execute(f"SET `{catalog_name}`")
            if database_name:
                self._execute(f"USE `{database_name}`")
        else:
            # not support catalog
            if database_name:
                self._execute(f"USE `{database_name}`")
        return

    def reset_catalog_to_default(self, catalog: str) -> str:
        """
        Reset the catalog to the default catalog if it is not set or is "def".
        """
        if not catalog or catalog == "def":
            return self.default_catalog()
        return catalog

    def reset_catalog_to_def(self, catalog: str) -> str:
        """
        Reset the catalog to "def" if it is not set or is the default catalog.
        """
        if not catalog or catalog == self.default_catalog():
            return "def"
        return catalog

    @override
    def get_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        return self._get_meta_with_ddl(
            tables=None, inner_table_type="view", catalog_name=catalog_name, database_name=database_name
        )

    @override
    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        catalog_name = self.reset_catalog_to_default(catalog_name)
        if catalog_name:
            if database_name:
                return f"`{catalog_name}`.`{database_name}`.`{table_name}`"
            else:
                return f"`{table_name}`"
        else:
            return f"`{database_name}`.`{table_name}`" if database_name else f"`{table_name}`"

    @override
    def _reset_filter_tables(
        self, tables: Optional[List[str]] = None, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        catalog_name = self.reset_catalog_to_default(catalog_name or self.catalog_name)
        database_name = database_name or self.database_name
        return super()._reset_filter_tables(tables, catalog_name=catalog_name, database_name=database_name)

    def db_meta_table_type(self) -> List[str]:
        return ["BASE TABLE"]

    @override
    def get_databases(self, catalog_name: str = "", include_sys: bool = False) -> List[str]:
        return super().get_schemas(catalog_name=catalog_name, include_sys=include_sys)

    @override
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        return []

    def _get_meta_with_ddl(
        self,
        tables: Optional[List[str]] = None,
        inner_table_type: TABLE_TYPE = "table",
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[Dict[str, str]]:
        """
        Get the database tables/views/materialized views as a list of dictionaries with DDL statements.

        Returns:
            A list of dictionaries, each containing:
            - schema_name: The schema name (usually same as database in StarRocks)
            - table_name: The table name
            - definition: The CREATE TABLE statement
            - table_type: The schema type (table, view or materialized_view)
        """
        result = []
        filter_tables = self._reset_filter_tables(tables, catalog_name=catalog_name, database_name=database_name)
        meta_table_name, create_type = inner_table_type_to_db(inner_table_type)
        for table in self._get_metadata(
            meta_table_name=meta_table_name,
            inner_table_type=None if inner_table_type != "table" else self.db_meta_table_type(),
            catalog_name=catalog_name,
            database_name=database_name,
        ):
            full_name = self.full_name(
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                table_name=table["table_name"],
            )
            if filter_tables and full_name not in filter_tables:
                continue
            try:
                create_statement = self._show_create(full_name=full_name, create_type=create_type)
            except Exception as e:
                logger.warning(f"Could not get DDL for table {full_name}: {e}")
                # Fallback to basic table info
                create_statement = f"-- DDL not available for table {table['table_name']}"
            table["identifier"] = self.identifier(
                catalog_name=table["catalog_name"],
                database_name=table["database_name"],
                table_name=table["table_name"],
            )
            table["definition"] = create_statement
            table["table_type"] = inner_table_type
            table["schema_name"] = ""
            result.append(table)
        return result

    @override
    def get_tables_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Get the database schema as a list of dictionaries with DDL statements.

        Returns:
            A list of dictionaries, each containing:
            - schema_name: The schema name (usually same as database in StarRocks)
            - table_name: The table name
            - definition: The CREATE TABLE statement
            - table_type: The schema type (table, view or materialized_view)
        """
        return self._get_meta_with_ddl(
            tables,
            inner_table_type="table",
            catalog_name=catalog_name,
            database_name=database_name,
        )

    def _show_create(self, full_name: str, create_type: CREATE_TYPE = "TABLE") -> str:
        sql = f"show create {create_type} {full_name}"
        ddl_result = self._execute_pandas(sql)
        if not ddl_result.empty and len(ddl_result.columns) >= 2:
            return str(ddl_result.iloc[0, 1])
        else:
            return f"-- DDL not available for table {full_name}"


class MySQLConnector(MySQLConnectorBase):
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        super().__init__(host, port, user, password, database)

    def sqlalchemy_schema(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> Optional[str]:
        return database_name

    @override
    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        return [
            table["table_name"]
            for table in self._get_metadata(
                meta_table_name="TABLES",
                inner_table_type=self.db_meta_table_type(),
                catalog_name=catalog_name,
                database_name=database_name,
            )
        ]

    @override
    def get_sample_rows(
        self,
        tables: Optional[List[str]] = None,
        top_n: int = 5,
        catalog_name: str = "",
        database_name: str = "",
        schema_name: str = "",
    ) -> List[Dict[str, str]]:
        self.connect()

        result = []
        if tables:
            for table in tables:
                full_name = self.full_name(database_name=database_name, table_name=table)
                sql = f"select * from {full_name} limit {top_n}"
                res = self._execute_pandas(sql)
                if not res.empty:
                    result.append(
                        {
                            "identifier": self.identifier(
                                database_name=database_name,
                                table_name=table,
                            ),
                            "catalog_name": "",
                            "database_name": schema_name if schema_name else "",
                            "schema_name": "",
                            "table_name": table,
                            "sample_rows": res.to_csv(index=False),
                        }
                    )
        else:
            for t in self._get_metadata(
                meta_table_name="TABLES",
                database_name=database_name,
            ):
                sql = f"select * from `{t['database_name']}`.`{t['table_name']}` limit {top_n}"
                res = self._execute_pandas(sql)
                if not res.empty:
                    result.append(
                        {
                            "catalog_name": "",
                            "database_name": t["database_name"],
                            "schema_name": "",
                            "table_name": t["table_name"],
                            "sample_rows": res.to_csv(index=False),
                        }
                    )
        return result
