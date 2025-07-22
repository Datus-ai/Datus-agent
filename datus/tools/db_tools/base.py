from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

from pyarrow import Table as ArrowTable

from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.utils.exceptions import DatusException, ErrorCode
from datus.utils.sql_utils import metadata_identifier


class BaseSqlConnector(ABC):
    def __init__(self, dialect: str, batch_size: int = 1024):
        self.batch_size = batch_size
        self.connection: Any = None
        self.dialect = dialect

    def close(self):
        """
        Close the database connection.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(self, input_params: Any, result_format: Literal["csv", "arrow", "list"] = "csv") -> ExecuteSQLResult:
        """Execute a SQL query against the SQLite database.

        Args:
            input_params: Dictionary containing the input parameters including:
                - sql_query: The SQL query to execute
            result_format: The format of the result to return

        Returns:
            A dictionary containing the query results
        """
        self.validate_input(input_params)
        if isinstance(input_params, dict):
            input_params = ExecuteSQLInput(**input_params)
        return self.do_execute(input_params, result_format)

    def validate_input(self, input_params: Any):
        """Validate the input parameters before execution.

        Args:
            input_params: Dictionary containing the input parameters to validate

        Raises:
            ValueError: If the input parameters are invalid
        """
        if not hasattr(input_params, "sql_query") and "sql_query" not in input_params:
            raise ValueError("'sql_query' parameter is required")
        if not isinstance(input_params["sql_query"], str):
            raise ValueError("'sql_query' must be a string")

    @abstractmethod
    def do_execute(
        self, input_params: ExecuteSQLInput, result_format: Literal["csv", "arrow", "list"] = "csv"
    ) -> ExecuteSQLResult:
        raise NotImplementedError

    def execute_arrow(self, query: str) -> ExecuteSQLResult:
        raise NotImplementedError

    @abstractmethod
    def execute_csv(self, query: str) -> ExecuteSQLResult:
        raise NotImplementedError

    def execute_arrow_iterator(self, query: str, max_rows: int = 100) -> Iterator[ArrowTable]:
        raise NotImplementedError

    def get_databases(self, catalog: str = "") -> List[str]:
        return []

    def get_schemas(self, catalog: str = "", database_name: str = "") -> List[str]:
        return []

    def execute_csv_iterator(
        self, query: str, max_rows: int = 100, with_header: bool = True
    ) -> Iterator[Tuple[str, ...]]:
        # if with_header is True, the first batch is the column names
        raise NotImplementedError

    # for internal streaming inter IPC or networking
    # def execute_to_arrow_stream(self, query: str, output_stream: Any,
    #                           compression: Optional[str] = 'lz4') -> None:
    #    pass
    @abstractmethod
    def test_connection(self):
        raise NotImplementedError

    def get_type(self) -> str:
        return self.dialect

    @abstractmethod
    def execute_queries(self, queries: List[str]) -> List[Any]:
        raise NotImplementedError

    def get_tables_with_ddl(self, tables: Optional[List[str]] = None, **kwargs) -> List[Dict[str, str]]:
        """
        Get all tables with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name) should be passed via kwargs and
        handled by subclasses as needed.
        parameters:
            tables: The table names to filter the tables.
            catalog_name: The catalog name to filter the tables.
            database_name: The database name to filter the tables.
            schema_name: The schema name to filter the tables.
        """
        raise NotImplementedError

    def _reset_filter_tables(self, tables: Optional[List[str]] = None, **kwargs) -> List[str]:
        filter_tables = []
        if tables:
            catalog_name = kwargs.get("catalog_name", "")
            database_name = kwargs.get("database_name", "")
            schema_name = kwargs.get("schema_name", "")
            for table_name in tables:
                filter_tables.append(
                    self.full_name(
                        table_name=table_name,
                        catalog_name=catalog_name,
                        database_name=database_name,
                        schema_name=schema_name,
                    )
                )
        return filter_tables

    def get_views_with_ddl(self, **kwargs) -> List[Dict[str, str]]:
        """
        Get all views with DDL from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name)
        should be passed via kwargs and handled by subclasses as needed.
        parameters:
            catalog_name: The catalog name to filter the views.
            database_name: The database name to filter the views.
            schema_name: The schema name to filter the views.
        """
        raise NotImplementedError

    def get_tables(self, **kwargs) -> List[str]:
        """
        Get all table names from the database.
        Parameters contains catalog_name, database_name and schema_name
        should be passed via kwargs and handled by subclasses as needed.
        """
        raise NotImplementedError

    def switch_context(self, **kwargs):
        """
        Switch context, including catalogs, databases and schemas.
        Parameters contains catalog_name, database_name and schema_name
        """
        try:
            self.do_switch_context(**kwargs)
        except DatusException as e:
            raise e
        except Exception as e:
            raise DatusException(
                ErrorCode.TOOL_DB_FAILED, message=f"Faield to switch context because {str(e)}, context_params:{kwargs}"
            ) from e

    @abstractmethod
    def do_switch_context(self, **kwargs):
        raise NotImplementedError

    def get_schema(self, **kwargs) -> List[Dict[str, str]]:
        """
        Get schema information for the specified table.
        Namespace parameters (such as catalog_name, database_name, schema_name, table_name)
        should be passed via kwargs and handled by subclasses as needed.
        """
        raise NotImplementedError

    def get_sample_rows(self, tables: Optional[List[str]] = None, top_n: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Get sample values for each table from the database.
        Namespace parameters (such as catalog_name, database_name, schema_name) should be passed via kwargs and
        handled by subclasses as needed.
        Args:
            tables: List of table names to sample from. If None, sample from all tables.
            top_n: Number of sample rows to retrieve per table.
            **kwargs: Additional namespace parameters for database selection.
        Returns:
            A list of dictionaries containing sample values for each table.
        """
        raise NotImplementedError

    def full_name(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        """
        Get the full name of the table. Special characters need to be taken into consideration during implementation.
        """
        raise NotImplementedError

    def identifier(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return metadata_identifier(
            dialect=self.dialect,
            catalog_name=catalog_name,
            database_name=database_name,
            schema_name=schema_name,
            table_name=table_name,
        )
