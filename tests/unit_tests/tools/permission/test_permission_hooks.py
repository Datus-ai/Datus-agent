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
        callback = AsyncMock()
        mock_broker.request = AsyncMock(return_value=("a", callback))

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

        callback = AsyncMock()
        mock_broker.request = AsyncMock(return_value=("n", callback))

        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException):
            await hooks.on_tool_start(ctx, MagicMock(), tool)

    @pytest.mark.asyncio
    async def test_strict_external_denies_without_broker(self, mock_broker, tmp_path):
        """Strict policy → EXTERNAL is rejected with PermissionDenied and the
        broker is never touched. Regression guard for API/claw flows that
        have no interactive prompt — they must fail fast, not hang."""
        hooks, _, _ = self._build(mock_broker, tmp_path, strict=True)
        target = tmp_path / "elsewhere.md"
        target.write_text("x")
        ctx = MagicMock()
        ctx.tool_arguments = f'{{"path": "{target}"}}'
        tool = MagicMock()
        tool.name = "read_file"
        with pytest.raises(PermissionDeniedException) as exc_info:
            await hooks.on_tool_start(ctx, MagicMock(), tool)
        assert "strict mode" in str(exc_info.value).lower()
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


class TestPermissionPromptLockPerLoop:
    """Regression guard: the prompt lock must not bleed across event loops.

    A module-level ``asyncio.Lock()`` used to bind to whichever loop first
    awaited it, then raised ``Lock is bound to a different event loop`` on
    every subsequent ``asyncio.run()`` call (the CLI creates a fresh loop per
    chat turn). These tests exercise the per-loop lock helper to make sure
    the bug cannot silently regress.
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
        # Within a single loop, concurrent tool calls must share one lock so
        # the "one prompt at a time" invariant still holds.
        assert first is second

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
            mock_broker.request = AsyncMock(return_value=("n", AsyncMock()))
            ctx = MagicMock()
            ctx.tool_arguments = f'{{"path": "{target}"}}'
            tool = MagicMock()
            tool.name = "read_file"
            with pytest.raises(PermissionDeniedException):
                await hooks.on_tool_start(ctx, MagicMock(), tool)

        asyncio.run(_one_turn())
        # Second turn: a brand-new loop. Must not raise the loop-binding error.
        asyncio.run(_one_turn())
