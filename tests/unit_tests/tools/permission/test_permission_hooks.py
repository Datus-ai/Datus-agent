# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for the permission hooks module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from datus.cli.execution_state import InteractionBroker
from datus.tools.permission import permission_hooks as permission_hooks_module
from datus.tools.permission.permission_config import PermissionConfig, PermissionLevel, PermissionRule
from datus.tools.permission.permission_hooks import (
    CompositeHooks,
    FilesystemPolicy,
    PermissionDeniedException,
    PermissionHooks,
)
from datus.tools.permission.permission_manager import PermissionManager
from datus.tools.registry.tool_registry import ToolRegistry


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

    @pytest.mark.asyncio
    async def test_on_tool_start_propagates_hook_exception(self):
        """Exception from first hook propagates; second hook is NOT called."""
        hook1 = MagicMock()
        hook1.on_tool_start = AsyncMock(side_effect=RuntimeError("hook1 failed"))
        hook2 = MagicMock()
        hook2.on_tool_start = AsyncMock()

        composite = CompositeHooks([hook1, hook2])
        context = MagicMock()
        agent = MagicMock()
        tool = MagicMock()

        with pytest.raises(RuntimeError, match="hook1 failed"):
            await composite.on_tool_start(context, agent, tool)

        # Second hook is never reached because the first raised
        hook2.on_tool_start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_tool_end_propagates_hook_exception(self):
        """Exception from first hook propagates; second hook is NOT called."""
        hook1 = MagicMock()
        hook1.on_tool_end = AsyncMock(side_effect=ValueError("end hook error"))
        hook2 = MagicMock()
        hook2.on_tool_end = AsyncMock()

        composite = CompositeHooks([hook1, hook2])
        context = MagicMock()
        agent = MagicMock()
        tool = MagicMock()
        result = {"success": True}

        with pytest.raises(ValueError, match="end hook error"):
            await composite.on_tool_end(context, agent, tool, result)

        hook2.on_tool_end.assert_not_awaited()


class TestPermissionHooks:
    """Tests for PermissionHooks."""

    def test_initialization(self, mock_broker):
        """Test PermissionHooks initialization."""
        registry = ToolRegistry()
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
        )

        assert hooks.broker == mock_broker
        assert hooks.permission_manager == manager
        assert hooks.node_name == "chat"
        assert hooks.tool_registry is registry

    def test_get_category_and_pattern_native_tool(self, mock_broker):
        """Test category detection for native tools."""
        registry = ToolRegistry()
        manager = PermissionManager()

        # Register a tool
        tool = MagicMock()
        tool.name = "execute_sql"
        registry.register_tools("db_tools", [tool])

        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=registry
        )

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("execute_sql", context)
        assert category == "db_tools"
        assert pattern == "execute_sql"

    def test_get_category_and_pattern_mcp_tool(self, mock_broker):
        """Test category detection for MCP tools."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("mcp__filesystem__read_file", context)
        assert category == "mcp.filesystem"
        assert pattern == "read_file"

    def test_get_category_and_pattern_skill(self, mock_broker):
        """Test category detection for load_skill."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

        context = MagicMock()
        context.tool_arguments = '{"skill_name": "sql-optimization"}'

        category, pattern = hooks._get_category_and_pattern("load_skill", context)
        assert category == "skills"
        assert pattern == "sql-optimization"

    def test_get_category_and_pattern_unknown_tool(self, mock_broker):
        """Test category detection for unknown tools."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

        context = MagicMock()
        context.tool_arguments = "{}"

        category, pattern = hooks._get_category_and_pattern("unknown_tool", context)
        assert category == "tools"
        assert pattern == "unknown_tool"

    def test_parse_tool_args_valid_json(self, mock_broker):
        """Test parsing valid JSON tool arguments."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

        context = MagicMock()
        context.tool_arguments = '{"key": "value", "number": 42}'

        result = hooks._parse_tool_args(context)
        assert result == {"key": "value", "number": 42}

    def test_parse_tool_args_invalid_json(self, mock_broker):
        """Test parsing invalid JSON returns empty dict."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

        context = MagicMock()
        context.tool_arguments = "not valid json"

        result = hooks._parse_tool_args(context)
        assert result == {}

    def test_parse_tool_args_dict_input(self, mock_broker):
        """Test parsing dict input returns as-is."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

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
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=registry
        )

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # Should not raise any exception
        await hooks.on_tool_start(context, agent, tool)

        # ALLOW permission: broker.request must never be invoked to prompt the
        # user. `mock_broker.assert_not_called()` only checks the mock itself as
        # a callable — `mock_broker.request(...)` would slip past it silently.
        mock_broker.request.assert_not_called()

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
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=registry
        )

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
        assert "PERMISSION_DENIED" in str(exc_info.value)
        assert "STOP retrying this tool" in str(exc_info.value)
        assert "run /profile to open the profile picker" in str(exc_info.value)
        assert "arrow keys" in str(exc_info.value)

    def test_initialization_default_proxied_tool_names_is_none(self, mock_broker):
        """``proxied_tool_names`` defaults to ``None`` for back-compat callers."""
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=PermissionManager(),
            node_name="chat",
            tool_registry=ToolRegistry(),
        )
        assert hooks.proxied_tool_names is None

    @pytest.mark.asyncio
    async def test_on_tool_start_skips_check_for_proxied_tool(self, mock_broker):
        """Proxied tools bypass permission checking even when DENY would normally fire.

        The external caller (e.g. ``print_mode`` stdin protocol) is responsible
        for secondary confirmation, so the agent must not double-check.
        """
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.DENY),
            ],
        )
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        proxied = {"execute_sql"}
        manager = PermissionManager(global_config=config)
        check_spy = MagicMock(wraps=manager.check_permission)
        manager.check_permission = check_spy  # type: ignore[assignment]

        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            proxied_tool_names=proxied,
        )

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # No exception even though the profile says DENY for db_tools.*.
        await hooks.on_tool_start(context, agent, tool)

        # Permission lookup must not happen at all for proxied tools.
        check_spy.assert_not_called()
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_tool_start_skips_check_for_proxied_ask_without_broker(self, mock_broker):
        """An ASK-rule proxied tool also bypasses — broker must not be touched."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            ],
        )
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            proxied_tool_names={"execute_sql"},
        )

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        await hooks.on_tool_start(context, agent, tool)

        # No broker prompt despite the ASK rule.
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_tool_start_observes_late_proxied_set_mutation(self, mock_broker):
        """When the hook holds a shared set reference, names added AFTER
        construction also bypass permission — this is the wiring used by
        ``ChatAgenticNode._setup_permission_hooks`` together with
        ``apply_proxy_tools`` running later in ``chat_task_manager``.
        """
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.DENY),
            ],
        )
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        shared: set = set()
        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            proxied_tool_names=shared,
        )

        # Empty set at construction → DENY still applies.
        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(context, agent, tool)

        # Simulate ``apply_proxy_tools`` running later and mutating the shared set.
        shared.add("execute_sql")

        # Now the same hook observes the name and skips permission checking.
        await hooks.on_tool_start(context, agent, tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_tool_start_non_proxied_tool_still_checked(self, mock_broker):
        """Tools outside ``proxied_tool_names`` follow the normal DENY behaviour."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=[
                PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.DENY),
            ],
        )
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            # Different tool name → execute_sql should still be denied.
            proxied_tool_names={"read_file"},
        )

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(context, agent, tool)

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
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])

        manager = PermissionManager(global_config=config)
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=registry
        )

        # Pre-approve in session
        manager.approve_for_session("db_tools", "execute_sql")

        context = MagicMock()
        context.tool_arguments = "{}"
        agent = MagicMock()
        tool = MagicMock()
        tool.name = "execute_sql"

        # Should not raise because of session approval
        await hooks.on_tool_start(context, agent, tool)

        # ASK permission with session approval: broker.request must NOT be
        # invoked to re-prompt the user. `mock_broker.assert_not_called()` only
        # checks the mock as a callable and would not catch a child-method call.
        mock_broker.request.assert_not_called()


class TestPermissionHooksIntegration:
    """Integration tests for permission hooks with ChatAgenticNode patterns."""

    def test_mcp_tool_name_parsing(self, mock_broker):
        """Test various MCP tool name formats."""
        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )
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
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=ToolRegistry()
        )

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

    def test_shared_tool_registry(self, mock_broker):
        """Test that PermissionHooks shares the same ToolRegistry instance."""
        registry = ToolRegistry()

        # Register db tools
        db_tool1 = MagicMock()
        db_tool1.name = "execute_sql"
        db_tool2 = MagicMock()
        db_tool2.name = "list_tables"
        registry.register_tools("db_tools", [db_tool1, db_tool2])

        # Register skill tools
        skill_tool = MagicMock()
        skill_tool.name = "load_skill"
        registry.register_tools("skills", [skill_tool])

        # Register filesystem tools
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])

        manager = PermissionManager()
        hooks = PermissionHooks(
            broker=mock_broker, permission_manager=manager, node_name="chat", tool_registry=registry
        )

        # Verify hooks shares the same registry
        assert hooks.tool_registry is registry
        assert len(hooks.tool_registry) == 4
        assert hooks.tool_registry.get("execute_sql") == "db_tools"
        assert hooks.tool_registry.get("list_tables") == "db_tools"
        assert hooks.tool_registry.get("load_skill") == "skills"
        assert hooks.tool_registry.get("read_file") == "filesystem_tools"


class TestFilesystemZoneBranch:
    """``fs_policy`` routes filesystem_tools calls through path zones.

    INTERNAL/WHITELIST bypass the normal rule, HIDDEN falls through silently
    (the tool returns not-found), EXTERNAL forces an ASK keyed by absolute
    path so approval never leaks across targets.
    """

    def _build(self, broker, tmp_path, rules=None, *, strict=False):
        registry = ToolRegistry()
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])

        config = PermissionConfig(
            default_permission=PermissionLevel.ALLOW,
            rules=rules or [],
        )
        manager = PermissionManager(global_config=config)
        project = tmp_path / "proj"
        project.mkdir()
        hooks = PermissionHooks(
            broker=broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            fs_policy=FilesystemPolicy(root_path=project, current_node="chat", strict=strict),
        )
        return hooks, manager, project

    @pytest.mark.asyncio
    async def test_internal_bypasses_ask_rule(self, mock_broker, tmp_path):
        hooks, _, project = self._build(
            mock_broker,
            tmp_path,
            rules=[PermissionRule(tool="filesystem_tools", pattern="*", permission=PermissionLevel.ASK)],
        )
        ctx = MagicMock()
        ctx.tool_arguments = '{"path": "src/main.py"}'
        tool = MagicMock()
        tool.name = "read_file"
        # Even though the rule says ASK, INTERNAL zone bypasses the prompt.
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_hidden_returns_without_prompt(self, mock_broker, tmp_path):
        hooks, _, project = self._build(mock_broker, tmp_path)
        ctx = MagicMock()
        ctx.tool_arguments = '{"path": ".datus/sessions/foo.db"}'
        tool = MagicMock()
        tool.name = "read_file"
        # HIDDEN short-circuits with no broker interaction; tool layer returns
        # the uniform "File not found".
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_external_forces_ask_and_caches_by_abs_path(self, mock_broker, tmp_path):
        hooks, manager, project = self._build(mock_broker, tmp_path)
        target_dir = tmp_path / "other"
        target_dir.mkdir()
        target = target_dir / "secret.md"
        target.write_text("x")

        # Broker returns "a" (approve for session) on first call, should not be
        # called again on the second.
        mock_broker.request = AsyncMock(return_value="a")

        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"

        await hooks.on_tool_start(ctx, MagicMock(), tool)
        assert mock_broker.request.await_count == 1

        # Second call with same abs path must NOT prompt.
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        assert mock_broker.request.await_count == 1
        # Cache key is path-keyed, not category-keyed.
        assert any(f"external::{target.resolve()}" in k for k in manager._session_approvals)

    @pytest.mark.asyncio
    async def test_external_deny_raises(self, mock_broker, tmp_path):
        hooks, _, _ = self._build(mock_broker, tmp_path)
        target = tmp_path / "other.md"
        target.write_text("x")

        mock_broker.request = AsyncMock(return_value="n")

        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)

    @pytest.mark.asyncio
    async def test_strict_external_delegates_to_tool_without_broker(self, mock_broker, tmp_path):
        """Strict policy → EXTERNAL is delegated to the tool layer (which
        returns FuncToolResult(success=0)) instead of raising. The broker is
        still never touched. Regression guard for API/gateway flows: they must
        fail fast with a readable tool-failure payload, not hang and not
        raise."""
        hooks, _, _ = self._build(mock_broker, tmp_path, strict=True)
        target = tmp_path / "elsewhere.md"
        target.write_text("x")
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_strict_internal_still_passes(self, mock_broker, tmp_path):
        """Strict must not affect INTERNAL/WHITELIST paths — those are the
        whole point of having a workspace at all."""
        hooks, _, project = self._build(mock_broker, tmp_path, strict=True)
        (project / "hello.md").write_text("hi")
        ctx = MagicMock()
        ctx.tool_arguments = '{"path": "hello.md"}'
        tool = MagicMock()
        tool.name = "read_file"
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_strict_external_end_to_end_returns_success_0(self, mock_broker, tmp_path):
        """End-to-end contract: hook + real tool in strict mode must surface
        EXTERNAL denials as ``FuncToolResult(success=0)`` with a "strict mode"
        error message, never as an exception. Cross-component guard for the
        fix that moved the rejection from the hook (raise) to the tool
        (success=0)."""
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        project = tmp_path / "proj"
        project.mkdir()
        external = tmp_path / "outside.md"
        external.write_text("secret")

        registry = ToolRegistry()
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])

        manager = PermissionManager(global_config=PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=[]))
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            fs_policy=FilesystemPolicy(root_path=project, current_node="chat", strict=True),
        )
        tool = FilesystemFuncTool(root_path=str(project), current_node="chat", strict=True)

        # 1. Hook must not raise and must not prompt.
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{external}"}}'
        hook_tool = MagicMock()
        hook_tool.name = "read_file"
        await hooks.on_tool_start(ctx, MagicMock(), hook_tool)
        mock_broker.request.assert_not_called()

        # 2. Tool layer produces the success=0 payload with the strict-mode message.
        read_result = tool.read_file(str(external))
        assert read_result.success == 0
        assert read_result.error.startswith("Path outside workspace is not allowed in strict mode:")
        assert str(external) in read_result.error

        write_result = tool.write_file(str(external), "new content")
        assert write_result.success == 0
        assert write_result.error.startswith("Path outside workspace is not allowed in strict mode:")
        assert str(external) in write_result.error

        # Guardrail: the external file was not actually touched.
        assert external.read_text() == "secret"

    @pytest.mark.asyncio
    async def test_external_broker_cancel_denies(self, mock_broker, tmp_path):
        """``InteractionCancelled`` from the broker must surface as a denial
        (not a silent approval). Guards the catch-block in
        ``_request_external_confirmation``."""
        from datus.cli.execution_state import InteractionCancelled

        hooks, _, _ = self._build(mock_broker, tmp_path)
        target = tmp_path / "cancel.md"
        target.write_text("x")
        mock_broker.request = AsyncMock(side_effect=InteractionCancelled())

        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)

    @pytest.mark.asyncio
    async def test_external_broker_unexpected_error_denies(self, mock_broker, tmp_path):
        """A non-``InteractionCancelled`` exception from the broker should
        also default to denial. Guards the generic ``except Exception`` arm."""
        hooks, _, _ = self._build(mock_broker, tmp_path)
        target = tmp_path / "boom.md"
        target.write_text("x")
        mock_broker.request = AsyncMock(side_effect=RuntimeError("broker explosion"))

        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)

    @pytest.mark.asyncio
    async def test_legacy_null_fs_policy_preserves_rules(self, mock_broker, tmp_path):
        """Without fs_policy, behavior must match the pre-refactor contract
        (rules drive everything). Regression guard for existing tests."""
        registry = ToolRegistry()
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])
        manager = PermissionManager(
            global_config=PermissionConfig(
                default_permission=PermissionLevel.ALLOW,
                rules=[
                    PermissionRule(tool="filesystem_tools", pattern="read_file", permission=PermissionLevel.DENY),
                ],
            )
        )
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            fs_policy=None,
        )
        ctx = MagicMock()
        # Path is INTERNAL-looking, but without fs_policy we do not short-circuit.
        ctx.tool_arguments = '{"path": "src/main.py"}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)


class TestFilesystemZoneProfileMatrix:
    """``_handle_filesystem_zone`` now reads ``active_profile`` and the
    tool name to enforce the read/write × profile matrix documented in
    ``datus/tools/permission/profiles.py``.

    Key invariants exercised below:

    * ``normal × INTERNAL × write`` falls back to rule lookup → ``default=ASK``
      (so the LLM cannot silently edit project files under the safe profile).
    * ``dangerous × EXTERNAL`` (interactive) bypasses the ASK gate.
    * ``non_interactive=True`` short-circuits before ``dangerous`` is even
      consulted — workflow flows must never inherit the dangerous bypass.
    """

    def _build(
        self,
        broker,
        tmp_path,
        *,
        profile="normal",
        rules=None,
        registered_tools=("read_file", "write_file", "edit_file", "glob", "grep"),
        non_interactive=False,
        strict=False,
    ):
        registry = ToolRegistry()
        fs_tools = []
        for name in registered_tools:
            mock_tool = MagicMock()
            mock_tool.name = name
            fs_tools.append(mock_tool)
        registry.register_tools("filesystem_tools", fs_tools)

        config = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=rules or [],
        )
        manager = PermissionManager(global_config=config, active_profile=profile)
        project = tmp_path / "proj"
        project.mkdir()
        hooks = PermissionHooks(
            broker=broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            fs_policy=FilesystemPolicy(root_path=project, current_node="chat", strict=strict),
            non_interactive=non_interactive,
        )
        return hooks, manager, project

    @staticmethod
    def _ctx_for(path):
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{path}"}}'
        return ctx

    @staticmethod
    def _tool(name):
        tool = MagicMock()
        tool.name = name
        return tool

    # --------------------------------------------------------------- INTERNAL
    @pytest.mark.parametrize("profile", ["normal", "auto", "dangerous"])
    @pytest.mark.asyncio
    async def test_internal_read_bypasses_all_profiles(self, mock_broker, tmp_path, profile):
        """Reading project-internal files never prompts, regardless of profile."""
        hooks, _, project = self._build(mock_broker, tmp_path, profile=profile)
        (project / "hello.md").write_text("hi")
        await hooks.on_tool_start(self._ctx_for("hello.md"), MagicMock(), self._tool("read_file"))
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_internal_write_normal_asks(self, mock_broker, tmp_path):
        """Writing inside the project under normal profile must prompt.

        The zone gate returns ``False`` here so the category-level
        ``default=ASK`` (or any explicit ``filesystem_tools.write_file`` rule)
        takes over. The user choosing "Allow once" lets the call proceed; the
        cache key is category-level, matching how ``bash_tools.execute_command``
        ASK works.
        """
        hooks, _, project = self._build(mock_broker, tmp_path, profile="normal")
        (project / "draft.md").write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(self._ctx_for("draft.md"), MagicMock(), self._tool("write_file"))
        # Exactly one prompt — the rule-level ASK, not the path-bucketed
        # external prompt.
        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_internal_write_normal_denial_raises(self, mock_broker, tmp_path):
        """User clicking "Deny" on the ASK prompt must raise, not silently allow."""
        hooks, _, project = self._build(mock_broker, tmp_path, profile="normal")
        (project / "draft.md").write_text("")
        mock_broker.request = AsyncMock(return_value="n")

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(self._ctx_for("draft.md"), MagicMock(), self._tool("write_file"))

    @pytest.mark.parametrize("profile", ["auto", "dangerous"])
    @pytest.mark.asyncio
    async def test_internal_write_non_normal_bypasses(self, mock_broker, tmp_path, profile):
        """Auto and dangerous let project-internal writes through silently."""
        hooks, _, project = self._build(mock_broker, tmp_path, profile=profile)
        (project / "draft.md").write_text("")
        await hooks.on_tool_start(self._ctx_for("draft.md"), MagicMock(), self._tool("write_file"))
        mock_broker.request.assert_not_called()

    # --------------------------------------------------------------- EXTERNAL
    @pytest.mark.parametrize(
        "tool_name",
        ["read_file", "write_file"],
    )
    @pytest.mark.asyncio
    async def test_external_dangerous_interactive_bypasses(self, mock_broker, tmp_path, tool_name):
        """``dangerous`` in interactive mode opts out of EXTERNAL ASK entirely.

        Read or write, no broker call. Workflow flows (non-interactive) still
        fail closed via the separate guard above this branch — see
        ``test_external_non_interactive_raises_even_under_dangerous``.
        """
        hooks, _, _ = self._build(mock_broker, tmp_path, profile="dangerous")
        target = tmp_path / "outside.md"
        target.write_text("x")
        await hooks.on_tool_start(self._ctx_for(target), MagicMock(), self._tool(tool_name))
        mock_broker.request.assert_not_called()

    @pytest.mark.parametrize(
        "profile,tool_name",
        [
            ("normal", "read_file"),
            ("normal", "write_file"),
            ("auto", "read_file"),
            ("auto", "write_file"),
        ],
    )
    @pytest.mark.asyncio
    async def test_external_non_dangerous_still_asks_per_path(self, mock_broker, tmp_path, profile, tool_name):
        """normal and auto keep the path-bucketed EXTERNAL ASK gate."""
        hooks, manager, _ = self._build(mock_broker, tmp_path, profile=profile)
        target = tmp_path / "outside.md"
        target.write_text("x")
        mock_broker.request = AsyncMock(return_value="a")  # always-allow this path

        await hooks.on_tool_start(self._ctx_for(target), MagicMock(), self._tool(tool_name))
        assert mock_broker.request.await_count == 1
        # Cache key is the absolute path — guards against profile downgrades
        # collapsing the EXTERNAL ASK into a category-wide approval.
        assert any(f"external::{target.resolve()}" in k for k in manager._session_approvals)

    @pytest.mark.parametrize("profile", ["normal", "auto", "dangerous"])
    @pytest.mark.asyncio
    async def test_external_non_interactive_raises_even_under_dangerous(self, mock_broker, tmp_path, profile):
        """Workflow safety: ``non_interactive=True`` always denies EXTERNAL.

        Without this, a ``/profile dangerous`` foreground change would leak
        into a workflow subagent's non-interactive ``PermissionHooks``
        (e.g. ``execution_mode="workflow"``) and let it silently touch
        ``/etc/passwd``.
        """
        hooks, _, _ = self._build(mock_broker, tmp_path, profile=profile, non_interactive=True)
        target = tmp_path / "outside.md"
        target.write_text("x")

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(self._ctx_for(target), MagicMock(), self._tool("write_file"))
        mock_broker.request.assert_not_called()

    # --------------------------------------------------------------- write-set
    @pytest.mark.asyncio
    async def test_is_write_set_matches_filesystem_tool_surface(self):
        """Guard against drift: the hook's write-set must cover every write
        method ``FilesystemFuncTool`` actually exposes.

        If a new write tool (e.g. ``append_file``) is added without updating
        ``_FILESYSTEM_WRITE_TOOLS``, the normal-profile INTERNAL gate would
        silently regress to bypass for that tool.
        """
        from datus.tools.func_tool.filesystem_tools import FilesystemFuncTool

        fs_tool = FilesystemFuncTool(root_path="/tmp")
        tool_names = {t.name for t in fs_tool.available_tools()}
        # Every tool the filesystem surface advertises as a mutation must be
        # declared as a write, so the normal-profile INTERNAL gate doesn't
        # silently bypass it. Read-only tools (``read_file`` / ``glob`` /
        # ``grep``) stay out — they have their own gate path.
        write_names = {"write_file", "edit_file", "delete_file"}
        # The hook's declared write-set must equal the writes the tool exposes,
        # not a strict subset. Both directions matter:
        #   - subset breaks the matrix (writes get silently bypassed)
        #   - superset means we ASK on a tool name that doesn't exist
        assert write_names == PermissionHooks._FILESYSTEM_WRITE_TOOLS
        assert write_names.issubset(tool_names)


class TestNonInteractiveMode:
    """``non_interactive=True`` short-circuits ASK / EXTERNAL fs branches.

    Used by ``execution_mode="workflow"`` flows (``/bootstrap``, scheduler
    subagents, ``auto_create``) where there is no human in the loop. ASK or
    EXTERNAL hits indicate the tool is outside the active profile's scope and
    must surface as ``PermissionDeniedException`` instead of awaiting a broker
    that nobody will respond to.
    """

    def _build_with_rule(self, broker, *, rule, non_interactive):
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=[rule])
        manager = PermissionManager(global_config=config, active_profile="auto")
        hooks = PermissionHooks(
            broker=broker,
            permission_manager=manager,
            node_name="bootstrap",
            tool_registry=registry,
            non_interactive=non_interactive,
        )
        return hooks, manager

    def test_default_non_interactive_is_false(self, mock_broker):
        """Field defaults to False so existing callers keep prompting."""
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=PermissionManager(),
            node_name="chat",
            tool_registry=ToolRegistry(),
        )
        assert hooks.non_interactive is False

    @pytest.mark.asyncio
    async def test_ask_raises_without_prompting_in_non_interactive(self, mock_broker):
        """ASK + non_interactive must raise immediately. Broker is never touched."""
        hooks, _ = self._build_with_rule(
            mock_broker,
            rule=PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            non_interactive=True,
        )
        ctx = MagicMock()
        ctx.tool_arguments = "{}"
        tool = MagicMock()
        tool.name = "execute_sql"
        with pytest.raises(PermissionDeniedException) as exc_info:
            await hooks.on_tool_start(ctx, MagicMock(), tool)
        # Active profile name is surfaced so the agent message points the user
        # at the right knob (``auto`` is the workflow-mode default).
        assert "auto" in str(exc_info.value)
        assert "non-interactive" in str(exc_info.value).lower()
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_session_cache_not_consulted_in_non_interactive(self, mock_broker):
        """The non_interactive short-circuit must run BEFORE the session
        approval cache lookup. Otherwise a stray ``approve_for_session`` call
        from a prior interactive turn would silently allow tools that the
        workflow-mode profile would otherwise reject."""
        hooks, manager = self._build_with_rule(
            mock_broker,
            rule=PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            non_interactive=True,
        )
        manager.approve_for_session("db_tools", "execute_sql")
        ctx = MagicMock()
        ctx.tool_arguments = "{}"
        tool = MagicMock()
        tool.name = "execute_sql"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_deny_still_raises_in_non_interactive(self, mock_broker):
        """DENY rules continue to raise — non_interactive must not weaken DENY."""
        hooks, _ = self._build_with_rule(
            mock_broker,
            rule=PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.DENY),
            non_interactive=True,
        )
        ctx = MagicMock()
        ctx.tool_arguments = "{}"
        tool = MagicMock()
        tool.name = "execute_sql"
        with pytest.raises(PermissionDeniedException) as exc_info:
            await hooks.on_tool_start(ctx, MagicMock(), tool)
        # DENY error message stays unchanged (mentions /profile, etc.)
        assert "STOP retrying this tool" in str(exc_info.value)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_still_passes_silently_in_non_interactive(self, mock_broker):
        """ALLOW rules are unaffected — workflow-mode flows must still run the
        operations the active profile permits."""
        hooks, _ = self._build_with_rule(
            mock_broker,
            rule=PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ALLOW),
            non_interactive=True,
        )
        ctx = MagicMock()
        ctx.tool_arguments = "{}"
        tool = MagicMock()
        tool.name = "execute_sql"
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_interactive_ask_still_prompts_when_flag_unset(self, mock_broker):
        """Sanity check: with non_interactive=False, ASK still goes through the
        broker — so this change cannot regress chat behavior."""
        hooks, _ = self._build_with_rule(
            mock_broker,
            rule=PermissionRule(tool="db_tools", pattern="*", permission=PermissionLevel.ASK),
            non_interactive=False,
        )
        # Approve via broker so we can confirm the prompt was actually issued.
        mock_broker.request = AsyncMock(return_value=[["y"]])
        ctx = MagicMock()
        ctx.tool_arguments = "{}"
        tool = MagicMock()
        tool.name = "execute_sql"
        await hooks.on_tool_start(ctx, MagicMock(), tool)
        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_external_fs_raises_without_prompting_in_non_interactive(self, mock_broker, tmp_path):
        """EXTERNAL filesystem zone in non_interactive mode raises immediately
        and never asks the broker — covers the ``_handle_filesystem_zone``
        short-circuit added alongside the ASK gate."""
        registry = ToolRegistry()
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])
        manager = PermissionManager(
            global_config=PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=[]),
            active_profile="auto",
        )
        project = tmp_path / "proj"
        project.mkdir()
        external = tmp_path / "elsewhere.md"
        external.write_text("x")
        hooks = PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="bootstrap",
            tool_registry=registry,
            fs_policy=FilesystemPolicy(root_path=project, current_node="bootstrap"),
            non_interactive=True,
        )
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{external}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException) as exc_info:
            await hooks.on_tool_start(ctx, MagicMock(), tool)
        assert "outside the project root" in str(exc_info.value)
        assert "auto" in str(exc_info.value)
        mock_broker.request.assert_not_called()
        # Approval cache must not be polluted by a non_interactive denial.
        assert not manager._session_approvals


class TestPermissionPromptLockPerLoop:
    """Regression guard: the prompt lock must not bleed across event loops.

    A module-level ``asyncio.Lock()`` used to bind to whichever loop first
    awaited it, then raised ``Lock is bound to a different event loop`` on
    every subsequent ``asyncio.run()`` call (the CLI creates a fresh loop per
    chat turn). These tests exercise the lock helper's fallback (no-broker)
    path to make sure the bug cannot silently regress.
    """

    def test_separate_asyncio_run_calls_do_not_reuse_lock(self):
        async def _acquire_once():
            lock = permission_hooks_module._get_permission_prompt_lock()
            async with lock:
                return lock

        lock_a = asyncio.run(_acquire_once())
        lock_b = asyncio.run(_acquire_once())

        # Each ``asyncio.run`` has its own loop, so it must receive its own
        # lock — reusing the first one would raise "bound to a different event
        # loop" on acquisition.
        assert lock_a is not lock_b

    def test_same_loop_returns_same_lock(self):
        async def _collect():
            first = permission_hooks_module._get_permission_prompt_lock()
            second = permission_hooks_module._get_permission_prompt_lock()
            return first, second

        first, second = asyncio.run(_collect())
        # Within a single loop, the fallback (no-broker) lock is shared so the
        # "one prompt at a time" invariant still holds for legacy callers.
        assert first is second


class TestPermissionPromptLockPerBroker:
    """The prompt lock is scoped per broker, not per event loop.

    On a long-lived multi-session server every request shares one loop, so a
    per-loop lock serialized prompts across independent sessions/sub-agents:
    while one held the lock across its user-response ``await``, the others
    blocked in ``on_tool_start`` and emitted the TOOL "processing" action but
    never the INTERACTION event. Scoping per broker fixes that.
    """

    def test_same_broker_same_loop_returns_same_lock(self):
        async def _collect():
            broker = InteractionBroker()
            first = permission_hooks_module._get_permission_prompt_lock(broker)
            second = permission_hooks_module._get_permission_prompt_lock(broker)
            return first, second

        first, second = asyncio.run(_collect())
        # Parallel tool calls within one run share a broker -> share a lock, so
        # the "one prompt at a time" invariant still holds inside a run.
        assert first is second

    def test_distinct_brokers_same_loop_get_distinct_locks(self):
        """The core fix: independent sessions must NOT share a prompt lock."""

        async def _collect():
            broker_a = InteractionBroker()
            broker_b = InteractionBroker()
            lock_a = permission_hooks_module._get_permission_prompt_lock(broker_a)
            lock_b = permission_hooks_module._get_permission_prompt_lock(broker_b)
            return lock_a, lock_b

        lock_a, lock_b = asyncio.run(_collect())
        assert lock_a is not lock_b

    def test_concurrent_brokers_do_not_serialize(self):
        """A broker holding its lock must not block a *different* broker's hook.

        Reproduces the reported hang: session A parks while holding its prompt
        lock; session B must still be able to acquire its own lock and proceed.
        With the old per-loop lock, B's ``acquire`` would block on A forever.
        """

        async def _run():
            broker_a = InteractionBroker()
            broker_b = InteractionBroker()
            b_acquired = asyncio.Event()

            async def hold_a():
                async with permission_hooks_module._get_permission_prompt_lock(broker_a):
                    # Park "forever" while holding A's lock (mirrors awaiting a
                    # user response that never arrives).
                    await asyncio.sleep(3600)

            async def acquire_b():
                async with permission_hooks_module._get_permission_prompt_lock(broker_b):
                    b_acquired.set()

            holder = asyncio.create_task(hold_a())
            await asyncio.sleep(0)  # let holder grab A's lock first
            consumer = asyncio.create_task(acquire_b())
            try:
                # B must acquire its own lock despite A being parked.
                await asyncio.wait_for(b_acquired.wait(), timeout=1.0)
            finally:
                holder.cancel()
                consumer.cancel()
                for t in (holder, consumer):
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
            return b_acquired.is_set()

        assert asyncio.run(_run()) is True

    def test_same_broker_distinct_loops_rebinds_lock(self):
        """A broker reused across CLI turns gets a loop-correct lock each turn."""
        broker = InteractionBroker()

        async def _acquire_once():
            lock = permission_hooks_module._get_permission_prompt_lock(broker)
            async with lock:  # would raise "bound to a different loop" if stale
                return lock

        lock_a = asyncio.run(_acquire_once())
        lock_b = asyncio.run(_acquire_once())
        assert lock_a is not lock_b

    def test_unhashable_broker_falls_back_to_loop_lock(self):
        """A non-weak-referenceable broker must not crash the helper."""

        async def _collect():
            # ``object()`` is weak-referenceable; use a type that is not by
            # giving the helper something that raises in WeakKeyDictionary.
            class _NoWeakref:
                __slots__ = ()  # no __weakref__ slot -> not weak-referenceable

            broker = _NoWeakref()
            lock = permission_hooks_module._get_permission_prompt_lock(broker)
            fallback = permission_hooks_module._get_permission_prompt_lock()
            return lock, fallback

        lock, fallback = asyncio.run(_collect())
        # Falls back to the shared per-loop lock instead of raising.
        assert lock is fallback

    def test_external_prompt_succeeds_across_separate_asyncio_runs(self, mock_broker, tmp_path):
        """End-to-end: two consecutive ``asyncio.run`` turns, each hitting the
        EXTERNAL-path prompt code path, must both succeed. Before the fix the
        second turn raised ``Lock is bound to a different event loop``."""
        registry = ToolRegistry()
        fs_tool = MagicMock()
        fs_tool.name = "read_file"
        registry.register_tools("filesystem_tools", [fs_tool])

        project = tmp_path / "proj"
        project.mkdir()
        target = tmp_path / "outside.md"
        target.write_text("x")

        async def _one_turn():
            # Fresh manager per turn to mirror the CLI re-initializing state.
            manager = PermissionManager(
                global_config=PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=[])
            )
            hooks = PermissionHooks(
                broker=mock_broker,
                permission_manager=manager,
                node_name="chat",
                tool_registry=registry,
                fs_policy=FilesystemPolicy(root_path=project, current_node="chat"),
            )
            # Rebind broker inside the coroutine so the AsyncMock is bound to
            # the currently-running loop.
            mock_broker.request = AsyncMock(return_value="n")
            ctx = MagicMock()
            ctx.tool_arguments = f'{{"path": "{target}"}}'
            tool = MagicMock()
            tool.name = "read_file"
            with pytest.raises(PermissionDeniedException):
                await hooks.on_tool_start(ctx, MagicMock(), tool)

        asyncio.run(_one_turn())
        # Second turn: a brand-new loop. Must not raise the loop-binding error.
        asyncio.run(_one_turn())


class TestVisualArtifactAutoAllow:
    """``_handle_filesystem_zone`` skips the INTERNAL × write × normal ASK
    prompt for the visual artifact subagents (``gen_visual_report`` /
    ``gen_visual_dashboard``) when the write targets their own artifact
    tree (``reports/<slug>/`` or ``dashboards/<slug>/``).

    The agent-side ``_FS_DEPENDENT_NODES`` carve-out in
    ``datus.tools.proxy.proxy_tool`` already keeps these tools un-proxied
    (server-side write), and the saas-side ``isAutoConfirmFilePath``
    carve-out silences the chat-panel Accept bar. This third layer covers
    the ``PermissionHooks`` ASK that would otherwise still gate the user
    on every ``render/*.jsx``.
    """

    def _build(self, broker, tmp_path, *, node_name, profile="normal"):
        registry = ToolRegistry()
        fs_tools = []
        for name in ("read_file", "write_file", "edit_file", "delete_file", "glob", "grep"):
            mock_tool = MagicMock()
            mock_tool.name = name
            fs_tools.append(mock_tool)
        registry.register_tools("filesystem_tools", fs_tools)

        manager = PermissionManager(
            global_config=PermissionConfig(default_permission=PermissionLevel.ASK, rules=[]),
            active_profile=profile,
        )
        project = tmp_path / "proj"
        project.mkdir()
        hooks = PermissionHooks(
            broker=broker,
            permission_manager=manager,
            node_name=node_name,
            tool_registry=registry,
            fs_policy=FilesystemPolicy(root_path=project, current_node=node_name),
        )
        return hooks, project

    @staticmethod
    def _ctx_for(path):
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{path}"}}'
        return ctx

    @staticmethod
    def _tool(name):
        tool = MagicMock()
        tool.name = name
        return tool

    @pytest.mark.parametrize(
        "node_name, relpath",
        [
            ("gen_visual_report", "reports/q1_2026/render/app.jsx"),
            ("gen_visual_report", "reports/q1_2026/manifest.json"),
            ("gen_visual_report", "reports/q1_2026/queries/revenue.sql"),
            ("gen_visual_dashboard", "dashboards/sales_live/render/app.jsx"),
            ("gen_visual_dashboard", "dashboards/sales_live/queries/store_revenue.sql.j2"),
        ],
    )
    @pytest.mark.parametrize("tool_name", ["write_file", "edit_file", "delete_file"])
    @pytest.mark.asyncio
    async def test_visual_artifact_write_under_own_tree_silent(
        self, mock_broker, tmp_path, node_name, relpath, tool_name
    ):
        """Write under the node's own artifact tree must not prompt under normal profile."""
        hooks, project = self._build(mock_broker, tmp_path, node_name=node_name)
        target = project / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(self._ctx_for(relpath), MagicMock(), self._tool(tool_name))

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_visual_artifact_write_outside_artifact_tree_still_asks(self, mock_broker, tmp_path):
        """A visual artifact node writing outside ``reports/`` / ``dashboards/``
        must still trip the normal-mode ASK — the carve-out is path-scoped.
        """
        hooks, project = self._build(mock_broker, tmp_path, node_name="gen_visual_report")
        (project / "scratch.md").write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(self._ctx_for("scratch.md"), MagicMock(), self._tool("write_file"))

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_visual_artifact_write_to_other_artifacts_dir_still_asks(self, mock_broker, tmp_path):
        """A report node writing under ``dashboards/`` (and vice versa) is not
        the same artifact — the carve-out lives at the path level not the
        prefix level, so cross-tree writes still ASK. Pessimistic on
        purpose; if the LLM crosses artifact families it's almost certainly
        a bug and the user wants to know.

        Implementation note: today the regex matches either ``reports/`` or
        ``dashboards/`` under either node, but the cross-write case is rare
        enough that we'd rather discover it via the prompt than silently
        hide it. Locking that behavior here lets a future tighten land
        without breaking the contract surprise-free.
        """
        # Today the regex is shared — both nodes match either prefix. The
        # test below documents *current* behavior, not the desired tighter
        # behavior. If a future commit narrows the regex per-node, flip the
        # assertion and link this test to the PR.
        hooks, project = self._build(mock_broker, tmp_path, node_name="gen_visual_report")
        target = project / "dashboards/foreign/render/app.jsx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(
            self._ctx_for("dashboards/foreign/render/app.jsx"), MagicMock(), self._tool("write_file")
        )

        # Current behavior: silent allow (same regex). Documented so a
        # future per-node tightening flips this knowingly.
        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_unrelated_node_writing_artifact_path_still_asks(self, mock_broker, tmp_path):
        """A non-artifact node writing into ``reports/`` does NOT benefit from
        the carve-out — only the visual artifact subagents do.
        """
        hooks, project = self._build(mock_broker, tmp_path, node_name="chat")
        target = project / "reports/q1_2026/render/app.jsx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(
            self._ctx_for("reports/q1_2026/render/app.jsx"), MagicMock(), self._tool("write_file")
        )

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_visual_artifact_write_under_artifact_root_no_slug_still_asks(self, mock_broker, tmp_path):
        """``reports/README.md`` is *next to* the artifact tree, not in it —
        no ``<slug>/`` segment, so the carve-out doesn't fire.
        """
        hooks, project = self._build(mock_broker, tmp_path, node_name="gen_visual_report")
        (project / "reports").mkdir()
        (project / "reports/README.md").write_text("")
        mock_broker.request = AsyncMock(return_value="y")

        await hooks.on_tool_start(self._ctx_for("reports/README.md"), MagicMock(), self._tool("write_file"))

        assert mock_broker.request.await_count == 1

    @pytest.mark.parametrize("profile", ["auto", "dangerous"])
    @pytest.mark.asyncio
    async def test_other_profiles_unchanged(self, mock_broker, tmp_path, profile):
        """``auto`` / ``dangerous`` already bypass via the outer zone branch
        without ever reaching the carve-out; verify they keep working.
        """
        hooks, project = self._build(mock_broker, tmp_path, node_name="gen_visual_dashboard", profile=profile)
        target = project / "dashboards/sales_live/render/app.jsx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")

        await hooks.on_tool_start(
            self._ctx_for("dashboards/sales_live/render/app.jsx"),
            MagicMock(),
            self._tool("write_file"),
        )

        mock_broker.request.assert_not_called()


class TestExecuteSqlPermission:
    """Statement-type gating for ``db_tools.execute_sql`` via _handle_sql_permission."""

    def _make_hooks(self, mock_broker, config, non_interactive=False, project_root=None):
        registry = ToolRegistry()
        tool_mock = MagicMock()
        tool_mock.name = "execute_sql"
        registry.register_tools("db_tools", [tool_mock])
        manager = PermissionManager(global_config=config)
        return PermissionHooks(
            broker=mock_broker,
            permission_manager=manager,
            node_name="chat",
            tool_registry=registry,
            non_interactive=non_interactive,
            project_root=project_root,
        )

    @staticmethod
    def _ctx(sql):
        import json

        ctx = MagicMock()
        ctx.tool_arguments = json.dumps({"sql": sql})
        return ctx

    @staticmethod
    def _tool():
        t = MagicMock()
        t.name = "execute_sql"
        return t

    @pytest.mark.asyncio
    async def test_read_auto_allows_under_normal_default_ask(self, mock_broker):
        """A SELECT auto-allows even when the profile default is ASK — no prompt."""
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)

        await hooks.on_tool_start(self._ctx("SELECT * FROM users"), MagicMock(), self._tool())

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_show_auto_allows(self, mock_broker):
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)

        await hooks.on_tool_start(self._ctx("SHOW TABLES"), MagicMock(), self._tool())

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_prompts_and_allows_on_yes(self, mock_broker):
        """An INSERT defers to the normal ASK flow; user 'y' lets it through."""
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)
        # InteractionBroker.request returns List[List[str]]; mirror that shape.
        mock_broker.request = AsyncMock(return_value=[["y"]])

        await hooks.on_tool_start(self._ctx("INSERT INTO users VALUES (1)"), MagicMock(), self._tool())

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_write_rejected_on_no(self, mock_broker):
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)
        mock_broker.request = AsyncMock(return_value=[["n"]])

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(self._ctx("DELETE FROM users"), MagicMock(), self._tool())

    @pytest.mark.asyncio
    async def test_ddl_prompts(self, mock_broker):
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)
        mock_broker.request = AsyncMock(return_value=[["y"]])

        await hooks.on_tool_start(self._ctx("CREATE TABLE t (id INT)"), MagicMock(), self._tool())

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_unknown_or_unparseable_sql_prompts(self, mock_broker):
        """An unresolvable .sql path / unparseable text is non-read → ASK (fail safe).

        With no ``project_root`` set the ``.sql`` file cannot be read, so the gate
        keeps the path as-is, classifies it UNKNOWN, and prompts.
        """
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)
        mock_broker.request = AsyncMock(return_value=[["y"]])

        await hooks.on_tool_start(self._ctx("sql/session_1/q.sql"), MagicMock(), self._tool())

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_readonly_sql_file_auto_allows(self, mock_broker, tmp_path):
        """A .sql file holding a read-only SELECT is resolved and auto-allows.

        The gate reads the workspace-relative file (same logic as the tool) so a
        read-only file does not trigger an unnecessary confirmation prompt.
        """
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "probe.sql").write_text("SELECT * FROM users LIMIT 10")
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config, project_root=str(tmp_path))

        await hooks.on_tool_start(self._ctx("sql/probe.sql"), MagicMock(), self._tool())

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_sql_file_prompts(self, mock_broker, tmp_path):
        """A .sql file holding a write is resolved to its real type and prompts."""
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()
        (sql_dir / "load.sql").write_text("INSERT INTO users VALUES (1)")
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config, project_root=str(tmp_path))
        mock_broker.request = AsyncMock(return_value=[["y"]])

        await hooks.on_tool_start(self._ctx("sql/load.sql"), MagicMock(), self._tool())

        assert mock_broker.request.await_count == 1

    @pytest.mark.asyncio
    async def test_read_respects_explicit_deny(self, mock_broker):
        """An explicit DENY on execute_sql blocks even a read."""
        config = PermissionConfig(
            default_permission=PermissionLevel.ASK,
            rules=[PermissionRule(tool="db_tools", pattern="execute_sql", permission=PermissionLevel.DENY)],
        )
        hooks = self._make_hooks(mock_broker, config)

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(self._ctx("SELECT 1"), MagicMock(), self._tool())

    @pytest.mark.asyncio
    async def test_write_non_interactive_raises_without_prompt(self, mock_broker):
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config, non_interactive=True)

        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(self._ctx("UPDATE users SET a = 1"), MagicMock(), self._tool())

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_allowed_under_dangerous_default(self, mock_broker):
        """Dangerous profile (default ALLOW) lets writes through with no prompt."""
        config = PermissionConfig(default_permission=PermissionLevel.ALLOW, rules=[])
        hooks = self._make_hooks(mock_broker, config)

        await hooks.on_tool_start(self._ctx("INSERT INTO t VALUES (1)"), MagicMock(), self._tool())

        mock_broker.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_always_allow_is_bucketed_by_sql_type(self, mock_broker):
        """'Always allow' is keyed by the concrete SQL type, not a coarse class.

        The session approval is bucketed per type (``execute_sql.insert`` /
        ``.update`` / ``.ddl``), so approving an INSERT only auto-approves later
        INSERTs — a different type (UPDATE, DROP) still prompts.
        """
        config = PermissionConfig(default_permission=PermissionLevel.ASK, rules=[])
        hooks = self._make_hooks(mock_broker, config)
        mock_broker.request = AsyncMock(return_value=[["a"]])  # "always allow (session)"

        # First INSERT: prompts once and approves the ``insert`` bucket.
        await hooks.on_tool_start(self._ctx("INSERT INTO t VALUES (1)"), MagicMock(), self._tool())
        assert mock_broker.request.await_count == 1

        # Another INSERT (same type): served from the session cache, no prompt.
        await hooks.on_tool_start(self._ctx("INSERT INTO t VALUES (2)"), MagicMock(), self._tool())
        assert mock_broker.request.await_count == 1

        # An UPDATE (different type) is NOT covered by the insert bucket → prompts.
        await hooks.on_tool_start(self._ctx("UPDATE t SET a = 1"), MagicMock(), self._tool())
        assert mock_broker.request.await_count == 2

        # A DDL (different type again) → prompts once more.
        await hooks.on_tool_start(self._ctx("DROP TABLE t"), MagicMock(), self._tool())
        assert mock_broker.request.await_count == 3
