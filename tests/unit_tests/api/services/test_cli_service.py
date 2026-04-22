"""Tests for datus.api.services.cli_service — CLI command operations."""

import asyncio
import uuid

import pytest

from datus.api.models.cli_models import ExecuteContextInput, ExecuteSQLInput
from datus.api.services.chat_service import ChatService
from datus.api.services.chat_task_manager import ChatTaskManager
from datus.api.services.cli_service import CLIService


@pytest.fixture
def cli_svc(real_agent_config):
    """Create CLIService with real config for reuse."""
    chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
    return CLIService(agent_config=real_agent_config, chat_service=chat_svc)


class TestCLIServiceInit:
    """Tests for CLIService initialization."""

    def test_init_with_real_config(self, cli_svc, real_agent_config):
        """CLIService initializes with real agent config and connector."""
        from datus.tools.db_tools.db_manager import DBManager

        assert cli_svc.agent_config is real_agent_config
        assert isinstance(cli_svc.db_manager, DBManager)
        assert cli_svc.current_datasource == real_agent_config.current_datasource
        # Connector must expose execute() — the only contract CLIService relies on
        assert callable(getattr(cli_svc.current_db_connector, "execute", None))

    def test_init_without_config(self):
        """CLIService initializes without agent config."""
        svc = CLIService(agent_config=None, chat_service=None)
        assert svc.db_manager is None
        assert svc.current_datasource is None
        assert svc.current_db_connector is None

    def test_init_sets_cli_context(self, cli_svc):
        """CLIService initializes CLI context with california_schools database."""
        from datus.cli.cli_context import CliContext

        assert isinstance(cli_svc.cli_context, CliContext)
        assert cli_svc.cli_context.current_db_name == "california_schools"

    def test_init_sets_current_db_name(self, cli_svc):
        """Init resolves current_db_name to the default database in the datasource."""
        assert cli_svc.current_db_name == "california_schools"


class TestCLIServiceExecuteSQL:
    """Tests for execute_sql with real SQLite."""

    @pytest.mark.asyncio
    async def test_execute_sql_select_success(self, cli_svc):
        """execute_sql runs a SELECT query and returns data."""
        from datus.api.models.cli_models import ExecuteSQLData

        request = ExecuteSQLInput(sql_query="SELECT COUNT(*) as cnt FROM schools")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert isinstance(result.data, ExecuteSQLData)
        assert result.data.sql_query == "SELECT COUNT(*) as cnt FROM schools"
        assert result.data.execution_time > 0
        # Server-generated IDs are UUID4 strings (36 chars with dashes)
        assert isinstance(result.data.execute_task_id, str)
        assert len(result.data.execute_task_id) == 36
        assert result.data.execute_task_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_execute_sql_returns_row_count(self, cli_svc):
        """execute_sql reports row count matching the LIMIT clause."""
        request = ExecuteSQLInput(sql_query="SELECT * FROM schools LIMIT 5")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.row_count == 5

    @pytest.mark.asyncio
    async def test_execute_sql_csv_format(self, cli_svc):
        """execute_sql with csv format returns CSV string with header and rows."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode, School FROM schools LIMIT 3", result_format="csv")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        # CSV writer emits header line + 3 data rows; assertion holds unconditionally
        csv_text = result.data.sql_return
        assert isinstance(csv_text, str)
        assert "CDSCode" in csv_text
        assert "School" in csv_text
        # Header + 3 rows => 4 newline-terminated lines
        assert csv_text.count("\n") == 4

    @pytest.mark.asyncio
    async def test_execute_sql_json_format(self, cli_svc):
        """execute_sql with json format returns JSON string."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode FROM schools LIMIT 2", result_format="json")
        result = await cli_svc.execute_sql(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_sql_invalid_sql_returns_error(self, cli_svc):
        """execute_sql with invalid SQL returns error."""
        request = ExecuteSQLInput(sql_query="SELCT INVALID SYNTAX")
        result = await cli_svc.execute_sql(request)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_sql_without_connector_returns_error(self):
        """execute_sql returns error when no connector available."""
        svc = CLIService(agent_config=None, chat_service=None)
        request = ExecuteSQLInput(sql_query="SELECT 1")
        result = await svc.execute_sql(request)
        assert result.success is False
        assert "No database connection" in result.errorMessage

    @pytest.mark.asyncio
    async def test_execute_sql_with_columns(self, cli_svc):
        """execute_sql returns column names when available."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode, School FROM schools LIMIT 1")
        result = await cli_svc.execute_sql(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_sql_arrow_default_format(self, cli_svc):
        """execute_sql with arrow format returns row_count matching LIMIT."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode FROM schools LIMIT 3", result_format="arrow")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.row_count == 3
        assert result.data.columns == ["CDSCode"]

    @pytest.mark.asyncio
    async def test_execute_sql_has_executed_at(self, cli_svc):
        """execute_sql result includes an ISO-8601 executed_at timestamp."""
        from datetime import datetime

        request = ExecuteSQLInput(sql_query="SELECT 1 as val")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        # ExecuteSQLData uses datetime.now().isoformat() + "Z"
        assert isinstance(result.data.executed_at, str)
        assert result.data.executed_at.endswith("Z")
        # Must parse as ISO-8601 after stripping the trailing Z
        datetime.fromisoformat(result.data.executed_at[:-1])

    @pytest.mark.asyncio
    async def test_execute_sql_with_database_name(self, cli_svc):
        """execute_sql with database_name parameter succeeds against that DB."""
        request = ExecuteSQLInput(
            sql_query="SELECT COUNT(*) FROM schools",
            database_name="california_schools",
        )
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.sql_query == "SELECT COUNT(*) FROM schools"


class TestCLIServiceStopExecuteSQL:
    """Tests for stop_execute_sql."""

    @pytest.mark.asyncio
    async def test_stop_nonexistent_task_returns_error(self, cli_svc):
        """stop_execute_sql with unknown task_id returns error."""
        result = await cli_svc.stop_execute_sql("nonexistent-task-id")
        assert result.success is False
        assert result.data.stopped is False
        assert "No running SQL execution" in result.errorMessage

    @pytest.mark.asyncio
    async def test_stop_completed_task_returns_error(self, cli_svc):
        """stop_execute_sql on a completed task returns already-completed error."""
        request = ExecuteSQLInput(sql_query="SELECT 1 as val")
        exec_result = await cli_svc.execute_sql(request)
        assert exec_result.success is True
        task_id = exec_result.data.execute_task_id

        # Task is already completed and cleaned up
        stop_result = await cli_svc.stop_execute_sql(task_id)
        assert stop_result.success is False
        assert stop_result.data.stopped is False

    @pytest.mark.asyncio
    async def test_execute_sql_returns_execute_task_id(self, cli_svc):
        """execute_sql result contains a UUID4-formatted execute_task_id."""
        request = ExecuteSQLInput(sql_query="SELECT 1 as val")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        # Server-generated IDs are canonical UUID4 strings
        task_id = result.data.execute_task_id
        assert isinstance(task_id, str)
        assert len(task_id) == 36
        assert task_id.count("-") == 4
        uuid.UUID(task_id, version=4)

    @pytest.mark.asyncio
    async def test_stop_running_task(self):
        """stop_execute_sql cancels a running task."""
        svc = CLIService(agent_config=None, chat_service=None)

        # Manually inject a long-running task to simulate a slow SQL execution
        async def _slow_task():
            await asyncio.sleep(60)

        task = asyncio.create_task(_slow_task())
        task_id = "test-stop-task"
        svc._sql_tasks[task_id] = task

        stop_result = await svc.stop_execute_sql(task_id)
        assert stop_result.success is True
        assert stop_result.data.stopped is True
        assert stop_result.data.execute_task_id == task_id

        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_execute_sql_honors_caller_supplied_task_id(self, cli_svc):
        """Caller-supplied execute_task_id is returned unchanged."""
        caller_task_id = "caller-supplied-abc-123"
        request = ExecuteSQLInput(sql_query="SELECT 1 as val", execute_task_id=caller_task_id)
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.execute_task_id == caller_task_id

    @pytest.mark.asyncio
    async def test_stop_execute_sql_uses_caller_supplied_task_id(self):
        """stop_execute_sql can cancel a task registered with a caller-supplied ID."""
        svc = CLIService(agent_config=None, chat_service=None)

        async def _slow_task():
            await asyncio.sleep(60)

        caller_task_id = "caller-cancel-id"
        svc._sql_tasks[caller_task_id] = asyncio.create_task(_slow_task())

        stop_result = await svc.stop_execute_sql(caller_task_id)
        assert stop_result.success is True
        assert stop_result.data.execute_task_id == caller_task_id
        assert stop_result.data.stopped is True

    @pytest.mark.asyncio
    async def test_execute_sql_rejects_duplicate_task_id(self):
        """execute_sql rejects a caller-supplied task_id that is already in use."""
        svc = CLIService(agent_config=None, chat_service=None)

        async def _slow_task():
            await asyncio.sleep(60)

        in_use_id = "in-use-task-id"
        existing = asyncio.create_task(_slow_task())
        svc._sql_tasks[in_use_id] = existing

        try:
            result = await svc.execute_sql(ExecuteSQLInput(sql_query="SELECT 1", execute_task_id=in_use_id))
            assert result.success is False
            assert "already in use" in (result.errorMessage or "")
        finally:
            existing.cancel()
            await asyncio.sleep(0)


class TestCLIServiceExecuteContext:
    """Tests for execute_context — context commands with real DB."""

    def test_context_tables(self, cli_svc):
        """execute_context 'tables' returns table list with expected entries."""
        from datus.api.models.cli_models import ExecuteContextData, TableInfo

        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("tables", request)
        assert result.success is True
        assert isinstance(result.data, ExecuteContextData)
        tables = result.data.result.tables
        assert isinstance(tables, list)
        # california_schools SQLite DB has schools/satscores/frpm tables
        assert len(tables) >= 3
        assert all(isinstance(t, TableInfo) for t in tables)

    def test_context_tables_has_schools(self, cli_svc):
        """execute_context 'tables' includes schools table."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("tables", request)
        table_names = [t.table_name for t in result.data.result.tables]
        assert "schools" in table_names

    def test_context_catalogs(self, cli_svc):
        """execute_context 'catalogs' returns catalog info as a dict."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("catalogs", request)
        assert result.success is True
        assert isinstance(result.data.result.context_info, dict)

    def test_context_context(self, cli_svc):
        """execute_context 'context' returns connection context."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("context", request)
        assert result.success is True
        info = result.data.result.context_info
        assert "current_datasource" in info
        assert "database" in info

    def test_context_catalog(self, cli_svc):
        """execute_context 'catalog' returns catalog context."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("catalog", request)
        assert result.success is True

    def test_context_subject(self, cli_svc):
        """execute_context 'subject' returns metrics context."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("subject", request)
        assert result.success is True

    def test_context_tables_without_connector(self):
        """execute_context 'tables' without connector returns empty."""
        svc = CLIService(agent_config=None, chat_service=None)
        request = ExecuteContextInput(context_type="tables")
        result = svc.execute_context("tables", request)
        assert result.success is True
        assert result.data.result.total_count == 0


class TestCLIServiceExecuteContextMore:
    """Additional context command tests."""

    def test_context_sql(self, cli_svc):
        """execute_context 'sql' returns historical SQL context."""
        request = ExecuteContextInput(context_type="sql")
        result = cli_svc.execute_context("sql", request)
        assert result.success is True

    def test_context_unsupported_type(self, cli_svc):
        """execute_context with unsupported type returns error."""
        request = ExecuteContextInput(context_type="unknown")
        result = cli_svc.execute_context("unknown_context", request)
        assert result.success is False
        assert "not supported" in result.errorMessage

    def test_context_catalogs_without_connector(self):
        """execute_context 'catalogs' without connector returns error info."""
        svc = CLIService(agent_config=None, chat_service=None)
        request = ExecuteContextInput(context_type="catalogs")
        result = svc.execute_context("catalogs", request)
        assert result.success is True
        assert "error" in result.data.result.context_info

    def test_context_context_without_connector(self):
        """execute_context 'context' without connector returns disconnected."""
        svc = CLIService(agent_config=None, chat_service=None)
        request = ExecuteContextInput(context_type="context")
        result = svc.execute_context("context", request)
        assert result.success is True
        assert result.data.result.context_info["database"]["connection_status"] == "disconnected"


class TestCLIServiceExecuteInternalCommand:
    """Tests for execute_internal_command — CLI commands."""

    def test_help_command(self, cli_svc):
        """help command returns available commands."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="help")
        result = cli_svc.execute_internal_command("help", request)
        assert result.success is True
        assert "help" in result.data.result.command_output.lower()
        assert result.data.result.action_taken == "display_help"

    def test_databases_command(self, cli_svc):
        """databases command lists available databases."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="databases")
        result = cli_svc.execute_internal_command("databases", request)
        assert result.success is True
        assert result.data.result.action_taken == "list_databases"
        assert isinstance(result.data.result.data, dict)
        assert "databases" in result.data.result.data

    def test_tables_command(self, cli_svc):
        """tables command lists available tables."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="tables")
        result = cli_svc.execute_internal_command("tables", request)
        assert result.success is True
        assert result.data.result.action_taken == "list_tables"
        assert "schools" in result.data.result.command_output

    def test_exit_command(self, cli_svc):
        """exit command returns goodbye message."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="exit")
        result = cli_svc.execute_internal_command("exit", request)
        assert result.success is True
        assert result.data.result.action_taken == "exit_program"
        assert "goodbye" in result.data.result.command_output.lower()

    def test_quit_command(self, cli_svc):
        """quit command works same as exit."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="quit")
        result = cli_svc.execute_internal_command("quit", request)
        assert result.success is True
        assert result.data.result.action_taken == "exit_program"

    def test_unsupported_command(self, cli_svc):
        """Unsupported command returns error."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="nonexistent_cmd")
        result = cli_svc.execute_internal_command("nonexistent_cmd", request)
        assert result.success is False
        assert "not supported" in result.errorMessage

    def test_chat_info_no_active_session(self, cli_svc):
        """chat_info command without active session returns message."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="chat_info")
        result = cli_svc.execute_internal_command("chat_info", request)
        assert result.success is True
        assert result.data.result.action_taken == "show_chat_info"

    def test_sessions_command(self, cli_svc):
        """sessions command lists chat sessions."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="sessions")
        result = cli_svc.execute_internal_command("sessions", request)
        assert result.success is True
        assert "sessions" in result.data.result.action_taken

    def test_clear_command_without_session_id(self, cli_svc):
        """clear command without session ID returns usage message."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="clear", args="")
        result = cli_svc.execute_internal_command("clear", request)
        assert result.success is True
        assert "clear" in result.data.result.action_taken

    def test_clear_command_with_session_id(self, cli_svc):
        """clear command with session ID attempts to clear session."""
        from datus.api.models.cli_models import InternalCommandInput

        request = InternalCommandInput(command="clear", args="some-session-id")
        result = cli_svc.execute_internal_command("clear", request)
        assert result.success is True
        assert "clear" in result.data.result.action_taken

    def test_tables_command_without_connector(self):
        """tables command without connector returns no connection."""
        from datus.api.models.cli_models import InternalCommandInput

        svc = CLIService(agent_config=None, chat_service=None)
        request = InternalCommandInput(command="tables")
        result = svc.execute_internal_command("tables", request)
        assert result.success is True
        assert "no database connection" in result.data.result.command_output.lower()

    def test_databases_command_without_manager(self):
        """databases command without db_manager returns message."""
        from datus.api.models.cli_models import InternalCommandInput

        svc = CLIService(agent_config=None, chat_service=None)
        request = InternalCommandInput(command="databases")
        result = svc.execute_internal_command("databases", request)
        assert result.success is True
        assert "no database" in result.data.result.command_output.lower()


class TestCLIServiceInitializeConnection:
    """Tests for _initialize_connection paths."""

    def test_initialize_connection_updates_cli_context(self, cli_svc):
        """_initialize_connection updates CLI context with database info."""
        from datus.cli.cli_context import CliContext

        assert isinstance(cli_svc.cli_context, CliContext)
        # CLI context should have been updated during init with the default DB
        assert cli_svc.current_db_name == "california_schools"
        assert cli_svc.cli_context.current_db_name == "california_schools"
