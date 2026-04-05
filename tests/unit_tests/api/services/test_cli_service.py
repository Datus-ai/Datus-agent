"""Tests for datus.api.services.cli_service — CLI command operations."""

import pytest

from datus.api.models.cli_models import ExecuteSQLInput
from datus.api.services.chat_service import ChatService
from datus.api.services.chat_task_manager import ChatTaskManager
from datus.api.services.cli_service import CLIService


class TestCLIServiceInit:
    """Tests for CLIService initialization."""

    def test_init_with_real_config(self, real_agent_config):
        """CLIService initializes with real agent config."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        assert svc is not None
        assert svc.current_db_connector is not None

    def test_init_without_config(self):
        """CLIService initializes without agent config."""
        svc = CLIService(agent_config=None, chat_service=None)
        assert svc.db_manager is None
        assert svc.current_namespace is None
        assert svc.current_db_connector is None

    def test_init_sets_cli_context(self, real_agent_config):
        """CLIService initializes CLI context."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        assert svc.cli_context is not None

    def test_init_creates_context_search_tools(self, real_agent_config):
        """CLIService creates ContextSearchTools when config is provided."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        assert svc.context_search_tools is not None

    def test_init_creates_output_tool(self, real_agent_config):
        """CLIService creates OutputTool when config is provided."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        assert svc.output_tool is not None


class TestCLIServiceExecuteSQL:
    """Tests for execute_sql with real SQLite."""

    def test_execute_sql_select(self, real_agent_config):
        """execute_sql runs a SELECT query successfully."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        request = ExecuteSQLInput(sql_query="SELECT COUNT(*) as cnt FROM schools")
        result = svc.execute_sql(request)
        assert result.success is True

    def test_execute_sql_without_connector_returns_error(self):
        """execute_sql returns error when no connector available."""
        svc = CLIService(agent_config=None, chat_service=None)
        request = ExecuteSQLInput(sql_query="SELECT 1")
        result = svc.execute_sql(request)
        assert result.success is False


class TestCLIServiceEnsureAgent:
    """Tests for _ensure_agent lazy initialization."""

    def test_agent_not_ready_initially(self, real_agent_config):
        """Agent is not ready immediately after init."""
        chat_svc = ChatService(real_agent_config, ChatTaskManager(), "test-proj")
        svc = CLIService(agent_config=real_agent_config, chat_service=chat_svc)
        assert svc.agent is None
        assert svc.agent_ready is False

    def test_ensure_agent_without_config(self):
        """_ensure_agent returns False when no config."""
        svc = CLIService(agent_config=None, chat_service=None)
        result = svc._ensure_agent()
        assert result is False
