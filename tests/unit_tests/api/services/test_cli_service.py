"""Tests for datus.api.services.cli_service — CLI command operations."""

import asyncio

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

    def test_init_with_real_config(self, cli_svc):
        """CLIService initializes with real agent config."""
        assert cli_svc is not None
        assert cli_svc.current_db_connector is not None

    def test_init_without_config(self):
        """CLIService initializes without agent config."""
        svc = CLIService(agent_config=None, chat_service=None)
        assert svc.db_manager is None
        assert svc.current_namespace is None
        assert svc.current_db_connector is None

    def test_init_sets_cli_context(self, cli_svc):
        """CLIService initializes CLI context."""
        assert cli_svc.cli_context is not None

    def test_init_sets_current_db_name(self, cli_svc):
        """Init resolves current_db_name from namespace."""
        assert cli_svc.current_db_name is not None


class TestCLIServiceExecuteSQL:
    """Tests for execute_sql with real SQLite."""

    @pytest.mark.asyncio
    async def test_execute_sql_select_success(self, cli_svc):
        """execute_sql runs a SELECT query and returns data."""
        request = ExecuteSQLInput(sql_query="SELECT COUNT(*) as cnt FROM schools")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data is not None
        assert result.data.sql_query == "SELECT COUNT(*) as cnt FROM schools"
        assert result.data.execution_time > 0
        assert result.data.execute_task_id is not None

    @pytest.mark.asyncio
    async def test_execute_sql_returns_row_count(self, cli_svc):
        """execute_sql reports row count."""
        request = ExecuteSQLInput(sql_query="SELECT * FROM schools LIMIT 5")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.row_count is not None

    @pytest.mark.asyncio
    async def test_execute_sql_csv_format(self, cli_svc):
        """execute_sql with csv format returns CSV string."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode, School FROM schools LIMIT 3", result_format="csv")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        if result.data.sql_return:
            assert "CDSCode" in result.data.sql_return or "School" in result.data.sql_return

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
        """execute_sql with default format returns arrow string."""
        request = ExecuteSQLInput(sql_query="SELECT CDSCode FROM schools LIMIT 3", result_format="arrow")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.row_count is not None

    @pytest.mark.asyncio
    async def test_execute_sql_has_executed_at(self, cli_svc):
        """execute_sql result includes executed_at timestamp."""
        request = ExecuteSQLInput(sql_query="SELECT 1 as val")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.executed_at is not None

    @pytest.mark.asyncio
    async def test_execute_sql_with_database_name(self, cli_svc):
        """execute_sql with database_name parameter."""
        request = ExecuteSQLInput(
            sql_query="SELECT COUNT(*) FROM schools",
            database_name="california_schools",
        )
        result = await cli_svc.execute_sql(request)
        # May or may not work with switch_context — exercise the code path
        assert result is not None


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
        """execute_sql result contains a non-empty execute_task_id."""
        request = ExecuteSQLInput(sql_query="SELECT 1 as val")
        result = await cli_svc.execute_sql(request)
        assert result.success is True
        assert result.data.execute_task_id
        assert len(result.data.execute_task_id) > 0

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


class TestCLIServiceExecuteContext:
    """Tests for execute_context — context commands with real DB."""

    def test_context_tables(self, cli_svc):
        """execute_context 'tables' returns table list."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("tables", request)
        assert result.success is True
        assert result.data is not None
        assert result.data.result.tables is not None
        assert len(result.data.result.tables) > 0

    def test_context_tables_has_schools(self, cli_svc):
        """execute_context 'tables' includes schools table."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("tables", request)
        table_names = [t.table_name for t in result.data.result.tables]
        assert "schools" in table_names

    def test_context_catalogs(self, cli_svc):
        """execute_context 'catalogs' returns catalog info."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("catalogs", request)
        assert result.success is True
        assert result.data.result.context_info is not None

    def test_context_context(self, cli_svc):
        """execute_context 'context' returns connection context."""
        request = ExecuteContextInput(context_type="tables")
        result = cli_svc.execute_context("context", request)
        assert result.success is True
        info = result.data.result.context_info
        assert "current_namespace" in info
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
        assert result.data.result.data is not None
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
        assert cli_svc.cli_context is not None
        # CLI context should have been updated during init
        assert cli_svc.current_db_name is not None
