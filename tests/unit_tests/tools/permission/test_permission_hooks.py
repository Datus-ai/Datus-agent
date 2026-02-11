# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the permission hooks module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.cli.execution_state import InteractionBroker
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import (
    CompositeHooks,
    PermissionDeniedException,
    PermissionHooks,
)
from datus.tools.permission.permission_manager import PermissionManager


@pytest.fixture
def mock_broker():
    """Create a mock InteractionBroker."""
    return MagicMock(spec=InteractionBroker)


class TestPermissionDeniedException:
    """Tests for PermissionDeniedException."""

    def test_exception_creation(self):
        """Test creating exception with message."""
        exc = PermissionDeniedException("Test error")
        assert str(exc) == "Test error"

    def test_exception_with_category_and_name(self):
        """Test exception includes tool category and name."""
        exc = PermissionDeniedException("Tool denied", tool_category="db_tools", tool_name="execute_sql")
        assert exc.tool_category == "db_tools"
        assert exc.tool_name == "execute_sql"


class TestCompositeHooks:
    """Tests for CompositeHooks."""

    def test_composite_hooks_filters_none(self):
        """Test that None values are filtered from hooks list."""
        hook1 = MagicMock()
        hook2 = None
        hook3 = MagicMock()

        composite = CompositeHooks([hook1, hook2, hook3])
        assert len(composite.hooks_list) == 2
        assert hook1 in composite.hooks_list
        assert hook3 in composite.hooks_list

    @pytest.mark.asyncio
    async def test_on_tool_start_calls_all_hooks(self):
        """Test on_tool_start calls all hooks."""
        hook1 = MagicMock()
        hook1.on_tool_start = AsyncMock()
        hook2 = MagicMock()
        hook2.on_tool_start = AsyncMock()

        composite = CompositeHooks([hook1, hook2])

        context = MagicMock()
        agent = MagicMock()
        tool = MagicMock()

        await composite.on_tool_start(context, agent, tool)

        hook1.on_tool_start.assert_awaited_once_with(context, agent, tool)
        hook2.on_tool_start.assert_awaited_once_with(context, agent, tool)

    @pytest.mark.asyncio
    async def test_on_tool_end_calls_all_hooks(self):
        """Test on_tool_end calls all hooks."""
        hook1 = MagicMock()
        hook1.on_tool_end = AsyncMock()
        hook2 = MagicMock()
        hook2.on_tool_end = AsyncMock()

        composite = CompositeHooks([hook1, hook2])

        context = MagicMock()
        agent = MagicMock()
        tool = MagicMock()
        result = {"success": True}

        await composite.on_tool_end(context, agent, tool, result)

        hook1.on_tool_end.assert_awaited_once_with(context, agent, tool, result)
        hook2.on_tool_end.assert_awaited_once_with(context, agent, tool, result)


class TestPermissionHooks:
    """Tests for PermissionHooks."""

    def test_initialization(self, mock_broker):
        """Test PermissionHooks initialization."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
        )

        assert hooks.broker == mock_broker
        assert hooks.permission_manager == manager
        assert hooks.node_name == "chat"
        assert hooks.tool_registry == {}

    def test_register_tools(self, mock_broker):
        """Test registering tools with their category."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Create mock tools
        tool1 = MagicMock()
        tool1.name = "execute_sql"
        tool2 = MagicMock()
        tool2.name = "list_tables"

        hooks.register_tools("db_tools", [tool1, tool2])

        assert hooks.tool_registry["execute_sql"] == "db_tools"
        assert hooks.tool_registry["list_tables"] == "db_tools"

    def test_get_category_and_pattern_native_tool(self, mock_broker):
        """Test category detection for native tools."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Register a tool
        tool = MagicMock()
        tool.name = "execute_sql"
        hooks.register_tools("db_tools", [tool])

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("execute_sql", context)
        assert category == "db_tools"
        assert pattern == "execute_sql"

    def test_get_category_and_pattern_mcp_tool(self, mock_broker):
        """Test category detection for MCP tools."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("mcp__filesystem__read_file", context)
        assert category == "mcp.filesystem"
        assert pattern == "read_file"

    def test_get_category_and_pattern_skill(self, mock_broker):
        """Test category detection for load_skill."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = '{"skill_name": "sql-optimization"}'

        category, pattern = hooks._get_category_and_pattern("load_skill", context)
        assert category == "skills"
        assert pattern == "sql-optimization"

    def test_get_category_and_pattern_unknown_tool(self, mock_broker):
        """Test category detection for unknown tools."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("unknown_tool", context)
        assert category == "tools"
        assert pattern == "unknown_tool"

    def test_parse_tool_args_valid_json(self, mock_broker):
        """Test parsing valid JSON tool arguments."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = '{"key": "value", "number": 42}'

        result = hooks._parse_tool_args(context)
        assert result == {"key": "value", "number": 42}

    def test_parse_tool_args_invalid_json(self, mock_broker):
        """Test parsing invalid JSON returns empty dict."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = "not valid json"

        result = hooks._parse_tool_args(context)
        assert result == {}

    def test_parse_tool_args_dict_input(self, mock_broker):
        """Test parsing dict input returns as-is."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        context = MagicMock()
        context.tool_arguments = {"key": "value"}

        result = hooks._parse_tool_args(context)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_on_tool_start_allow(self, mock_broker):
        """Test on_tool_start allows tool when permission is ALLOW."""
        # Create config that allows db_tools
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[],
        )
        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Register the tool
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        hooks.register_tools("db_tools", [tool_mock])

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # Should not raise any exception
        await hooks.on_tool_start(context, agent, tool)

    @pytest.mark.asyncio
    async def test_on_tool_start_deny(self, mock_broker):
        """Test on_tool_start raises exception when permission is DENY."""
        # Create config that denies db_tools
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.DENY),
            ],
        )
        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Register the tool
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        hooks.register_tools("db_tools", [tool_mock])

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # Should raise PermissionDeniedException
        with pytest.raises(PermissionDeniedException) as exc_info:
            await hooks.on_tool_start(context, agent, tool)

        assert "execute_sql" in str(exc_info.value)
        assert exc_info.value.tool_category == "db_tools"

    @pytest.mark.asyncio
    async def test_on_tool_start_ask_with_session_approval(self, mock_broker):
        """Test on_tool_start uses session cache for ASK permission."""
        # Create config that requires ask for db_tools
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            ],
        )
        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Register the tool
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        hooks.register_tools("db_tools", [tool_mock])

        # Pre-approve in session
        manager.approve_for_session("db_tools", "execute_sql")

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # Should not raise because of session approval
        await hooks.on_tool_start(context, agent, tool)


class TestPermissionHooksIntegration:
    """Integration tests for permission hooks with ChatAgenticNode patterns."""

    def test_mcp_tool_name_parsing(self, mock_broker):
        """Test various MCP tool name formats."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")
        context = MagicMock()
        context.tool_arguments = "{}"

        # Standard MCP format
        cat, pat = hooks._get_category_and_pattern("mcp__sqlite__read_query", context)
        assert cat == "mcp.sqlite"
        assert pat == "read_query"

        # Multi-part tool name
        cat, pat = hooks._get_category_and_pattern("mcp__filesystem__read_text_file", context)
        assert cat == "mcp.filesystem"
        assert pat == "read_text_file"

        # Complex server name
        cat, pat = hooks._get_category_and_pattern("mcp__duckdb-mftutorial__query", context)
        assert cat == "mcp.duckdb-mftutorial"
        assert pat == "query"

    def test_skill_name_extraction(self, mock_broker):
        """Test skill name extraction from tool arguments."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Valid skill name
        context = MagicMock()
        context.tool_arguments = '{"skill_name": "admin-tools"}'
        cat, pat = hooks._get_category_and_pattern("load_skill", context)
        assert cat == "skills"
        assert pat == "admin-tools"

        # Missing skill name
        context = MagicMock()
        context.tool_arguments = "{}"
        cat, pat = hooks._get_category_and_pattern("load_skill", context)
        assert cat == "skills"
        assert pat == "*"  # Fallback to wildcard

    def test_register_multiple_tool_categories(self, mock_broker):
        """Test registering tools from multiple categories."""
        manager = PermissionManager()
        hooks = PermissionHooks(broker=mock_broker, permission_manager=manager, node_name="chat")

        # Register db tools
        db_tool1 = MagicMock()
        db_tool1.name = "execute_sql"
        db_tool2 = MagicMock()
        db_tool2.name = "list_tables"
        hooks.register_tools("db_tools", [db_tool1, db_tool2])

        # Register skill tools
        skill_tool = MagicMock()
        skill_tool.name = "load_skill"
        hooks.register_tools("skills", [skill_tool])

        # Register filesystem tools
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        hooks.register_tools("filesystem_tools", [fs_tool])

        # Verify all registered
        assert len(hooks.tool_registry) == 4
        assert hooks.tool_registry["execute_sql"] == "db_tools"
        assert hooks.tool_registry["list_tables"] == "db_tools"
        assert hooks.tool_registry["load_skill"] == "skills"
        assert hooks.tool_registry["read_file"] == "filesystem_tools"
