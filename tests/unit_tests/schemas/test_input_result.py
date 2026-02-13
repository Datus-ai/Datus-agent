import pytest
from pandas import DataFrame
from pydantic import ValidationError

from datus.schemas.node_models import ExecuteSQLResult, GenerateSQLResult, SqlTask, TableSchema, TableValue
from datus.schemas.schema_linking_node_models import SchemaLinkingInput, SchemaLinkingResult
from datus.utils.constants import DBType
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class TestSchemaLinkingInput:
    def test_initialization(self):
        with pytest.raises(ValidationError):
            SchemaLinkingInput(
                input_text="test query",
                matching_rate="123",
                database_type=DBType.SQLITE,
                database_name="test_db",
            )

        input = SchemaLinkingInput(
            input_text="test query",
            matching_rate="fast",
            database_type=DBType.SQLITE,
            database_name="test_db",
        )
        assert input.input_text == "test query"
        assert input.matching_rate == "fast"
        assert input.database_type == DBType.SQLITE
        assert input.database_name == "test_db"

    def test_from_sql_task(self):
        sql_task = SqlTask(task="test task", database_type=DBType.SQLITE, database_name="test_db")
        input = SchemaLinkingInput.from_sql_task(sql_task)
        assert input.input_text == "test task"
        assert input.database_type == DBType.SQLITE
        assert input.database_name == "test_db"

    def test_validation(self):
        with pytest.raises(ValueError):
            SchemaLinkingInput(input_text="test", max_num_tables="111", database_type=DBType.SQLITE, database_name="")


class TestSqlTask:
    def test_initialization(self):
        task = SqlTask(
            id="123",
            task="test task",
            database_type=DBType.SQLITE,
            database_name="test_db",
            output_dir="output",
        )
        assert task.id == "123"
        assert task.task == "test task"
        assert task.database_type == DBType.SQLITE
        assert task.database_name == "test_db"
        assert task.output_dir == "output"

        logger.debug(f"SqlTask: str({task})")

    def test_from_dict(self):
        data = {
            "id": "123",
            "task": "test task",
            "database_type": DBType.SQLITE,
            "database_name": "test_db",
            "output_dir": "output",
        }
        task = SqlTask.from_dict(data)
        assert task.id == "123"
        assert task.task == "test task"

    def test_from_to_str(self):
        task = SqlTask(
            id="12345",
            task="test task",
            database_type=DBType.SNOWFLAKE,
            database_name="test_db",
            output_dir="output",
        )
        logger.debug(f"SqlTask: str({str(task)})")
        logger.debug(f"SqlTask: str({task.to_str()})")
        logger.debug(f"SqlTask: str({task.to_dict()})")
        # str(id='12345' database_type='snowflake' task='test task' database_name='test_db' output_dir='output')
        task2 = SqlTask.from_str(task.to_str())
        assert task2.id == task.id
        assert task2.task == task.task
        assert task2.database_type == task.database_type
        assert task2.database_name == task.database_name
        assert task2.output_dir == task.output_dir
        logger.debug(f"SqlTask2: {task2}")

        task3 = SqlTask.from_dict(task.to_dict())
        assert task3.id == task.id
        assert task3.task == task.task
        assert task3.database_type == task.database_type
        assert task3.database_name == task.database_name
        assert task3.output_dir == task.output_dir

    def test_from_to_str_default(self):
        task = SqlTask(id="12345", task="test task", output_dir="output")
        task2 = SqlTask.from_str(task.to_str())
        assert task2.id == task.id
        assert task2.task == task.task
        assert task2.output_dir == task.output_dir
        logger.debug(f"SqlTask2: {task2}")

        task3 = SqlTask.from_dict(task.to_dict())
        assert task3.id == task.id
        assert task3.task == task.task
        assert task3.output_dir == task.output_dir

    def test_from_str(self):
        data = (
            '{"id":"sf001","database_type":"snowflake","task":"Assuming today is April 1, 2025,'
            " I would like to know the daily snowfall amounts greater than 6 inches for each U.S. postal"
            " code during the week ending after the first two full weeks of the previous year. Show the postal code,"
            ' date, and snowfall amount.",'
            '"database_name":"GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI","output_dir":"output"}'
        )
        failed_data = (
            '{"id":"sf001","database_type":"snowflake","task":"",'
            '"database_name":"GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI","output_dir":"output"}'
        )
        task = SqlTask.from_str(data)
        assert task.id == "sf001"
        assert task.database_type == DBType.SNOWFLAKE
        assert task.database_name == "GLOBAL_WEATHER__CLIMATE_DATA_FOR_BI"
        assert task.output_dir == "output"

        with pytest.raises(ValidationError):
            SqlTask.from_str(failed_data)

    def test_validation(self):
        with pytest.raises(ValueError):
            SqlTask(task=" ", database_type=DBType.SQLITE, database_name="")


class TestSchemaLinkingResult:
    def test_initialization(self):
        schema = TableSchema(
            table_name="test_table",
            database_name="test_db",
            schema_name="test_schema",
            definition="CREATE TABLE test_table (id INT)",
        )

        value = TableValue(
            table_name="test_table",
            database_name="test_db",
            schema_name="test_schema",
            table_values="[1, 2, 3]",
        )

        result = SchemaLinkingResult(
            success=True,
            table_schemas=[schema],
            schema_count=1,
            table_values=[value],
            value_count=1,
        )

        assert len(result.table_schemas) == 1
        assert len(result.table_values) == 1
        assert result.success is True
        assert result.schema_count == 1
        assert result.value_count == 1

    def test_validation(self):
        with pytest.raises(ValueError):
            SchemaLinkingResult(table_schemas=[], schema_count=1, table_values=[], value_count=0)

    def test_from_to_str(self):
        schema = TableSchema(
            table_name="test_table",
            database_name="test_db",
            schema_name="test_schema",
            definition="CREATE TABLE test_table (id INT)",
        )

        value = TableValue(
            table_name="test_table",
            database_name="test_db",
            schema_name="test_schema",
            table_values="[1, 2, 3]",
        )

        original = SchemaLinkingResult(
            table_schemas=[schema],
            schema_count=1,
            table_values=[value],
            value_count=1,
            success=True,  # Added required field
        )

        # Test serialization and deserialization
        serialized = original.to_str()
        logger.debug(f"SchemaLinkingResult: {serialized}")
        deserialized = SchemaLinkingResult.from_str(serialized)
        logger.debug(f"SchemaLinkingResult: {deserialized}")

        assert len(deserialized.table_schemas) == 1
        assert len(deserialized.table_values) == 1
        assert deserialized.schema_count == 1
        assert deserialized.value_count == 1
        assert deserialized.success is True  # Added assertion
        assert deserialized.table_schemas[0].table_name == "test_table"
        assert deserialized.table_values[0].table_name == "test_table"


class TestGenerateSQLResult:
    def test_initialization(self):
        result = GenerateSQLResult(
            success=True,
            sql_query="SELECT * FROM test",
            tables=["test_table"],
            explanation="test explanation",
        )
        assert result.sql_query == "SELECT * FROM test"
        assert result.tables == ["test_table"]
        assert result.explanation == "test explanation"

    def test_from_dict(self):
        data = {
            "sql_query": "SELECT * FROM test",
            "tables": ["test_table"],
            "explanation": "test explanation",
            "success": True,
        }
        result = GenerateSQLResult.from_dict(data)
        assert result.sql_query == "SELECT * FROM test"


class TestExecuteSQLResult:
    def test_compact_result(self):
        result = ExecuteSQLResult(
            success=True,
            sql_query="SELECT * FROM test",
            row_count=10,
            sql_return=DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}),
        )
        compact = result.compact_result()
        assert "Alice" in compact
        assert len(compact) > len(result.sql_return)


class TestReflectionInput:
    def test_validation(self):
        task = SqlTask(task="test")
        print(task)
        gen_result = GenerateSQLResult(
            success=True,
            sql_query="SELECT * FROM test",
            tables=["test_table"],
            explanation="test explanation",
        )
        print(gen_result)
        exec_result = ExecuteSQLResult(
            success=True, sql_query="SELECT * FROM test", row_count=1, sql_return="test result"
        )
        print(exec_result)
        # input = ReflectionInput(
        #     task_description=task,
        #     sql_generation_result=gen_result,
        #     sql_execution_result=exec_result
        # )

        # assert input.task_description.task == "test"

        # with pytest.raises(ValueError):
        #     ReflectionInput(
        #         task_description=task,
        #         sql_generation_result=GenerateSQLResult(success=False),
        #         sql_execution_result=exec_result
        #     )
