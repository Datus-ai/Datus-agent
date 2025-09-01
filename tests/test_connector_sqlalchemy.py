import os
import tempfile

import pyarrow as pa
import pytest
from sqlalchemy import text

from datus.schemas.node_models import ExecuteSQLInput, ExecuteSQLResult
from datus.tools.db_tools.sqlalchemy_connector import SQLAlchemyConnector
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TestSQLAlchemyTools:
    """Test suite for SQLAlchemy connector functionality"""

    @pytest.fixture
    def sqlalchemy_connector(self):
        """Create a SQLAlchemy connector instance"""
        db_path = os.path.join(os.path.dirname(__file__), "data/SSB.db")
        connector = SQLAlchemyConnector(f"sqlite:///{db_path}")
        connector.connect()

        # Create temporary test table
        connector.connection.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS test_insert_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """
            )
        )
        connector.connection.commit()

        yield connector

        # Cleanup - close all iterators first
        connector.close()

        # Create a new connection for cleanup
        cleanup_connector = SQLAlchemyConnector(f"sqlite:///{db_path}")
        cleanup_connector.connect()
        cleanup_connector.connection.execute(text("DROP TABLE IF EXISTS test_insert_table"))
        cleanup_connector.connection.commit()
        cleanup_connector.close()

    def test_do_execute_select(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test do_execute with SELECT query"""
        input_params = ExecuteSQLInput(sql_query="SELECT * FROM lineorder LIMIT 5")
        result = sqlalchemy_connector.execute(input_params)

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 5
        assert result.result_format == "csv"
        assert isinstance(result.sql_return, str)

    def test_do_execute_insert(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test do_execute with INSERT query"""
        input_params = ExecuteSQLInput(sql_query="INSERT INTO test_insert_table (name, value) VALUES ('test1', 1)")
        result = sqlalchemy_connector.execute(input_params)

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 1
        assert result.sql_return == "1"

    def test_execute_arrow_select(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_arrow with SELECT query"""
        result = sqlalchemy_connector.execute_arrow("SELECT * FROM lineorder LIMIT 5")

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 5
        assert result.result_format == "arrow"
        assert isinstance(result.sql_return, pa.Table)
        assert result.sql_return.num_rows == 5
        assert result.sql_return.num_columns > 0

    def test_execute_arrow_insert(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_arrow with INSERT query"""
        result = sqlalchemy_connector.execute_arrow("INSERT INTO test_insert_table (name, value) VALUES ('test2', 2)")

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 0
        assert result.result_format == "arrow"
        assert result.sql_return == 1  # rowcount for insert

    def test_execute_csv_select(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_csv with SELECT query"""
        result = sqlalchemy_connector.execute(ExecuteSQLInput(sql_query="SELECT * FROM lineorder LIMIT 5"))

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 5
        assert result.result_format == "csv"
        assert isinstance(result.sql_return, str)

    def test_execute_csv_insert(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_csv with INSERT query"""
        result = sqlalchemy_connector.execute(
            ExecuteSQLInput(sql_query="INSERT INTO test_insert_table (name, value) VALUES ('test3', 3)")
        )

        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is None
        assert result.row_count == 1
        assert result.sql_return == "1"

    def test_execute_csv_iterator(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_csv_iterator"""
        iterator = sqlalchemy_connector.execute_csv_iterator("SELECT * FROM lineorder LIMIT 10", max_rows=2)

        # First yield should be column names (could be tuple or key view)
        columns = next(iterator)
        assert len(columns) > 0

        # Next yields should be rows
        rows = list(iterator)
        assert len(rows) == 10  # 10 rows in total
        assert all(len(row) == len(columns) for row in rows)

    def test_execute_arrow_iterator(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test execute_arrow_iterator"""
        iterator = sqlalchemy_connector.execute_arrow_iterator("SELECT * FROM lineorder LIMIT 10", max_rows=2)

        # Should yield rows
        rows = list(iterator)
        assert len(rows) == 10  # 10 rows in total

    def test_stream_to_parquet(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test stream_to_parquet"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Stream data to parquet file
            sqlalchemy_connector.stream_to_parquet("SELECT * FROM lineorder LIMIT 10", temp_path)

            # Read back and verify
            table = pa.parquet.read_table(temp_path)
            assert table.num_rows == 10
            assert table.num_columns > 0

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_error_handling(self, sqlalchemy_connector: SQLAlchemyConnector):
        """Test error handling for invalid queries"""
        # Test invalid SQL
        result = sqlalchemy_connector.execute(ExecuteSQLInput(sql_query="SELECT * FROM nonexistent_table"))
        logger.debug(f"result: {result.error}")
        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is not None
        assert result.row_count is None or result.row_count == 0

        # Test invalid arrow query
        result = sqlalchemy_connector.execute_arrow("SELECT * FROM nonexistent_table")
        logger.debug(f"result: {result.error}")
        assert isinstance(result, ExecuteSQLResult)
        assert result.success is False
        assert result.error is not None
        assert result.row_count is None or result.row_count == 0

        # Test invalid csv query
        result = sqlalchemy_connector.execute(ExecuteSQLInput(sql_query="SELECT * FROM nonexistent_table"))
        logger.debug(f"result: {result.error}")
        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is not None
        assert result.row_count is None or result.row_count == 0

        result = sqlalchemy_connector.execute(
            ExecuteSQLInput(
                sql_query="""select sum(lo_revenue), d_year, p_brand
 from lineorder, dates, part, supplier
 where lo_orderdate = d_datekey
 and lo_partkey = p_partkey
 and lo_suppkey = s_suppkey
 and p_brand = 'MFGR#2339'
 and s_region = 'EUROPE'
 group by d_year, p_brand
 order by d_year, p_brand;"""
            ),
            result_format="csv",
        )
        logger.debug(f"result: {result.error}")
        assert isinstance(result, ExecuteSQLResult)
        assert result.success is True
        assert result.error is not None
        assert result.row_count is None or result.row_count == 0
