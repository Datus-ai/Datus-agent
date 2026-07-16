# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/bash_mode.py.

Tests cover:
- The shim contract with PermissionHooks (``context.tool_arguments`` /
  ``tool.name`` are the ONLY attributes the hook reads — pinned here so a
  future hook change fails loudly instead of silently bypassing the gate)
- _gate_and_execute: no display actions, permission-denied short circuit,
  execution success/failure propagation into BashModeRun
- _make_input_collector: fails closed (never returns None) so the gate can't hang
- run_bash_mode_command: fail-closed guards (bash disabled, no permission
  config) and the end-to-end drive through a real broker/merge, incl. the
  ASK-prompt round-trip answered by the collector
- _resolve_permission_components / _resolve_bash_tool: node reuse vs CLI-owned fallback
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from datus.cli.bash_mode import (
    BashModeRun,
    _BashContextShim,
    _BashToolStub,
    _gate_and_execute,
    _make_input_collector,
    _resolve_bash_tool,
    _resolve_permission_components,
    run_bash_mode_command,
)
from datus.schemas.action_history import ActionRole, ActionStatus
from datus.tools.func_tool.base import FuncToolResult
from datus.tools.permission.permission_hooks import PermissionDeniedException, PermissionHooks

_UNSET = object()


def _make_cli(bash_enabled=True, permissions_config=_UNSET, node=None):
    agent_config = MagicMock()
    agent_config.bash_tool_enabled = bash_enabled
    agent_config.permissions_config = MagicMock() if permissions_config is _UNSET else permissions_config
    agent_config.project_root = None
    # Keep plugin transformers out of the gate path by default so tests exercise
    # the permission gate + execution without needing the plugin registry.
    agent_config.plugins_enabled = False
    chat_commands = SimpleNamespace(current_node=node)
    return SimpleNamespace(
        agent_config=agent_config,
        chat_commands=chat_commands,
        console=Console(file=io.StringIO(), no_color=True),
        tui_app=None,
    )


async def _collect(gen):
    return [action async for action in gen]


# ---------------------------------------------------------------------------
# Tests: shim contract with PermissionHooks
# ---------------------------------------------------------------------------


class TestShimContract:
    def test_context_shim_round_trips_through_hook_parser(self):
        """``_parse_tool_args`` must recover the command from the shim exactly
        as it would from a real SDK ``RunContextWrapper``."""
        hooks = object.__new__(PermissionHooks)
        args = hooks._parse_tool_args(_BashContextShim("git status | head"))
        assert args == {"command": "git status | head"}

    def test_context_shim_is_json_encoded(self):
        shim = _BashContextShim("echo hi")
        assert json.loads(shim.tool_arguments) == {"command": "echo hi"}

    def test_tool_stub_exposes_bash_name(self):
        """``on_tool_start`` resolves the tool via ``getattr(tool, "name")``."""
        assert _BashToolStub().name == "bash"


# ---------------------------------------------------------------------------
# Tests: _gate_and_execute (gate + run, no display actions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGateAndExecute:
    async def test_yields_no_display_actions(self):
        """The gate emits nothing to render — the styled bash block does that."""
        run = BashModeRun(command="echo hi")
        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(return_value=None)
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(success=1, result="hi\n")

        actions = await _collect(_gate_and_execute(_make_cli(), run, hooks, bash_tool))
        assert actions == []

    async def test_success_populates_run(self):
        run = BashModeRun(command="echo hi")
        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(return_value=None)
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(success=1, result="hi\n")

        await _collect(_gate_and_execute(_make_cli(), run, hooks, bash_tool))

        assert run.executed is True
        assert run.success is True
        assert run.output == "hi\n"
        bash_tool.bash.assert_called_once_with("echo hi")

    async def test_denied_never_executes(self):
        run = BashModeRun(command="rm -rf /")
        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(side_effect=PermissionDeniedException("PERMISSION_DENIED: blocked"))
        bash_tool = MagicMock()

        await _collect(_gate_and_execute(_make_cli(), run, hooks, bash_tool))

        assert run.executed is False
        assert "blocked" in run.error
        bash_tool.bash.assert_not_called()

    async def test_command_failure_propagates_error(self):
        run = BashModeRun(command="false")
        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(return_value=None)
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(
            success=0, error="Command exited with code 1", result="partial out"
        )

        await _collect(_gate_and_execute(_make_cli(), run, hooks, bash_tool))

        assert run.executed is True
        assert run.success is False
        assert run.error == "Command exited with code 1"
        assert run.output == "partial out"

    async def test_gate_passes_shim_context_and_tool(self):
        """The hook receives the shim pair — the contract TestShimContract pins."""
        run = BashModeRun(command="ls")
        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(return_value=None)
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(success=1, result="")

        await _collect(_gate_and_execute(_make_cli(), run, hooks, bash_tool))

        (ctx_arg, agent_arg, tool_arg), _ = hooks.on_tool_start.call_args
        assert json.loads(ctx_arg.tool_arguments) == {"command": "ls"}
        assert agent_arg is None
        assert tool_arg.name == "bash"


# ---------------------------------------------------------------------------
# Tests: _make_input_collector — must fail CLOSED, never hang
# ---------------------------------------------------------------------------


class TestInputCollector:
    """A ``None`` return would skip the interaction without a broker submit,
    leaving the gate's ``broker.request()`` awaiting forever. Every failure
    path must yield the ESC/cancel answer ``[[""]]`` (== deny) instead."""

    def _action(self):
        from datus.schemas.action_history import ActionHistory

        return ActionHistory(
            action_id="a1",
            role=ActionRole.INTERACTION,
            action_type="request_choice",
            status=ActionStatus.PROCESSING,
            input={},
        )

    def test_empty_events_return_cancel_answer(self):
        collect = _make_input_collector(_make_cli())
        assert collect(self._action(), MagicMock()) == [[""]]

    def test_wizard_failure_returns_cancel_answer(self):
        collect = _make_input_collector(_make_cli())
        with (
            patch(
                "datus.schemas.interaction_event.InteractionEvent.from_broker_input",
                return_value=[MagicMock()],
            ),
            patch("datus.cli.interaction_app.InteractionApp", side_effect=RuntimeError("no tty")),
        ):
            assert collect(self._action(), MagicMock()) == [[""]]

    def test_wizard_answers_pass_through(self):
        collect = _make_input_collector(_make_cli())
        wizard = MagicMock()
        wizard.run.return_value = SimpleNamespace(answers=[["y"]])
        with (
            patch(
                "datus.schemas.interaction_event.InteractionEvent.from_broker_input",
                return_value=[MagicMock()],
            ),
            patch("datus.cli.interaction_app.InteractionApp", return_value=wizard),
        ):
            assert collect(self._action(), MagicMock()) == [["y"]]


# ---------------------------------------------------------------------------
# Tests: run_bash_mode_command
# ---------------------------------------------------------------------------


class TestRunBashModeCommand:
    def test_fails_closed_when_bash_disabled(self):
        cli = _make_cli(bash_enabled=False)
        run = run_bash_mode_command(cli, "echo hi")
        assert run.executed is False
        assert "disabled" in run.error

    def test_fails_closed_without_permission_config(self):
        cli = _make_cli(permissions_config=None)
        run = run_bash_mode_command(cli, "echo hi")
        assert run.executed is False
        assert "permission" in run.error.lower()

    def test_end_to_end_drives_gate_and_populates_run(self):
        """With hooks/bash tool mocked, the harness must drive the gate through
        a real broker+merge and surface the result on the run (no ASK prompt →
        the collector is never invoked)."""
        cli = _make_cli()

        hooks = MagicMock()
        hooks.on_tool_start = AsyncMock(return_value=None)
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(success=1, result="out\n")

        with (
            patch("datus.cli.bash_mode._build_permission_hooks", return_value=hooks),
            patch("datus.cli.bash_mode._resolve_bash_tool", return_value=bash_tool),
        ):
            run = run_bash_mode_command(cli, "echo out")

        assert run.executed is True
        assert run.success is True
        assert run.output == "out\n"
        bash_tool.bash.assert_called_once_with("echo out")

    def test_end_to_end_ask_prompt_answered_by_collector(self):
        """An ASK request from the gate is answered via the merged broker.

        A fake hook requests confirmation through the broker, then approves
        only when the collector returns ``[["y"]]`` — proving the wizard
        answer round-trips without a streaming context.
        """
        from datus.schemas.interaction_event import InteractionEvent

        cli = _make_cli()
        bash_tool = MagicMock()
        bash_tool.bash.return_value = FuncToolResult(success=1, result="ok\n")

        class _AskHooks:
            def __init__(self):
                self.broker = None

            async def on_tool_start(self, ctx, agent, tool):
                answers = await self.broker.request(
                    [InteractionEvent(title="Permission", content="Allow?", choices={"y": "Yes", "n": "No"})]
                )
                if not answers or answers[0] != ["y"]:
                    from datus.tools.permission.permission_hooks import PermissionDeniedException

                    raise PermissionDeniedException("User rejected execution of bash command")

        ask_hooks = _AskHooks()

        def _build(cli_arg, broker):
            ask_hooks.broker = broker
            return ask_hooks

        with (
            patch("datus.cli.bash_mode._build_permission_hooks", side_effect=_build),
            patch("datus.cli.bash_mode._resolve_bash_tool", return_value=bash_tool),
            patch("datus.cli.bash_mode._make_input_collector", return_value=lambda a, c: [["y"]]),
        ):
            run = run_bash_mode_command(cli, "docker ps")

        assert run.executed is True
        assert run.success is True
        bash_tool.bash.assert_called_once_with("docker ps")

    def test_end_to_end_ask_prompt_rejected_blocks_execution(self):
        from datus.schemas.interaction_event import InteractionEvent

        cli = _make_cli()
        bash_tool = MagicMock()

        class _AskHooks:
            def __init__(self):
                self.broker = None

            async def on_tool_start(self, ctx, agent, tool):
                answers = await self.broker.request(
                    [InteractionEvent(title="Permission", content="Allow?", choices={"y": "Yes", "n": "No"})]
                )
                if not answers or answers[0] != ["y"]:
                    from datus.tools.permission.permission_hooks import PermissionDeniedException

                    raise PermissionDeniedException("User rejected execution of bash command")

        ask_hooks = _AskHooks()

        def _build(cli_arg, broker):
            ask_hooks.broker = broker
            return ask_hooks

        with (
            patch("datus.cli.bash_mode._build_permission_hooks", side_effect=_build),
            patch("datus.cli.bash_mode._resolve_bash_tool", return_value=bash_tool),
            patch("datus.cli.bash_mode._make_input_collector", return_value=lambda a, c: [[""]]),
        ):
            run = run_bash_mode_command(cli, "docker ps")

        assert run.executed is False
        assert "rejected" in run.error
        bash_tool.bash.assert_not_called()

    def test_pipeline_setup_failure_is_reported_not_raised(self):
        cli = _make_cli()
        with patch(
            "datus.cli.bash_mode._build_permission_hooks",
            side_effect=RuntimeError("no profile"),
        ):
            run = run_bash_mode_command(cli, "echo hi")
        assert run.executed is False
        assert "no profile" in run.error


# ---------------------------------------------------------------------------
# Tests: _resolve_bash_tool
# ---------------------------------------------------------------------------


class TestResolveBashTool:
    def test_reuses_node_bash_tool(self):
        node = MagicMock()
        node.bash_tool = MagicMock()
        cli = _make_cli(node=node)
        assert _resolve_bash_tool(cli) is node.bash_tool

    def test_node_without_tool_falls_back_to_cli_owned(self):
        node = MagicMock()
        node.bash_tool = None
        cli = _make_cli(node=node)
        fake_tool = MagicMock()
        with patch("datus.tools.func_tool.bash_tool.BashTool", return_value=fake_tool) as ctor:
            assert _resolve_bash_tool(cli) is fake_tool
        # Gating already ran upstream; the tool itself must not re-filter.
        assert ctor.call_args.kwargs["allowed_patterns"] == ["*"]
        assert ctor.call_args.kwargs["identity"] == "cli-bash-mode"

    def test_cli_owned_tool_is_cached(self):
        cli = _make_cli(node=None)
        fake_tool = MagicMock()
        with patch("datus.tools.func_tool.bash_tool.BashTool", return_value=fake_tool):
            first = _resolve_bash_tool(cli)
        with patch("datus.tools.func_tool.bash_tool.BashTool") as ctor2:
            second = _resolve_bash_tool(cli)
        assert first is fake_tool
        assert second is fake_tool
        ctor2.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _resolve_permission_components
# ---------------------------------------------------------------------------


class TestResolvePermissionComponents:
    def test_reuses_node_manager_and_registry(self):
        from datus.tools.registry.tool_registry import ToolRegistry

        node = MagicMock()
        node.permission_manager = MagicMock()
        node.get_node_name.return_value = "chat_node"
        node.tool_registry = ToolRegistry({"bash": "bash_tools", "execute_sql": "db_tools", "grep": "filesystem_tools"})
        cli = _make_cli(node=node)

        pm, node_name, registry = _resolve_permission_components(cli)

        assert pm is node.permission_manager
        assert node_name == "chat_node"
        # Registry already maps bash + execute_sql — reused as-is (session
        # approvals and category routing stay shared with the agent).
        assert registry is node.tool_registry

    def test_node_registry_missing_bash_gets_augmented(self):
        from datus.tools.registry.tool_registry import ToolRegistry

        node = MagicMock()
        node.permission_manager = MagicMock()
        node.get_node_name.return_value = "chat_node"
        node.tool_registry = ToolRegistry({"grep": "filesystem_tools"})
        cli = _make_cli(node=node)

        _, _, registry = _resolve_permission_components(cli)

        assert registry.get("bash") == "bash_tools"
        assert registry.get("execute_sql") == "db_tools"
        assert registry.get("grep") == "filesystem_tools"

    def test_builds_and_caches_cli_owned_manager_without_node(self):
        cli = _make_cli(node=None)
        cli.agent_config.active_profile_name = "normal"
        cli.agent_config.plugin_bash_rules = None
        cli.agent_config.project_bash_allow = None

        fake_pm = MagicMock()
        with patch("datus.tools.permission.permission_manager.PermissionManager", return_value=fake_pm) as ctor:
            pm, node_name, registry = _resolve_permission_components(cli)

        assert pm is fake_pm
        assert node_name == "chat"
        assert registry.get("bash") == "bash_tools"
        ctor.assert_called_once_with(
            global_config=cli.agent_config.permissions_config,
            active_profile="normal",
            plugin_bash_rules=None,
            project_bash_allows=None,
        )
        # Cached: a second resolution must not rebuild the manager.
        with patch("datus.tools.permission.permission_manager.PermissionManager") as ctor2:
            pm2, _, _ = _resolve_permission_components(cli)
        assert pm2 is fake_pm
        ctor2.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: run_sql_gate (permission + transformers + sql_policy for manual SQL)
# ---------------------------------------------------------------------------


def _sql_cli(profile="normal", node=None):
    """CLI double whose agent_config carries a REAL permission profile so the
    gate exercises the actual PermissionHooks / _handle_sql_permission path."""
    from datus.tools.permission.profiles import get_profile

    ac = MagicMock()
    ac.permissions_config = get_profile(profile)
    ac.active_profile_name = profile
    ac.plugin_bash_rules = None
    ac.project_bash_allow = None
    ac.project_root = None
    ac.plugins_enabled = False
    ac.current_datasource = "duckdb"
    connector = MagicMock()
    connector.dialect = "duckdb"
    return SimpleNamespace(
        agent_config=ac,
        chat_commands=SimpleNamespace(current_node=node),
        console=Console(file=io.StringIO(), no_color=True),
        tui_app=None,
        db_connector=connector,
    )


class TestRunSqlGate:
    def test_read_auto_allows_without_prompt(self):
        from datus.cli.bash_mode import run_sql_gate

        gate = run_sql_gate(_sql_cli(), "SELECT 1")
        assert gate.approved is True
        assert gate.error is None
        assert gate.sql == "SELECT 1"

    def test_no_permission_config_degrades_to_allow(self):
        from datus.cli.bash_mode import run_sql_gate

        cli = _sql_cli()
        cli.agent_config.permissions_config = None
        gate = run_sql_gate(cli, "SELECT 1")
        assert gate.approved is True

    def test_write_ask_approved_via_collector(self):
        """A write prompts; approving via the collector marks it approved."""
        from datus.cli.bash_mode import run_sql_gate

        cli = _sql_cli()
        with patch("datus.cli.bash_mode._make_input_collector", return_value=lambda a, c: [["y"]]):
            gate = run_sql_gate(cli, "DELETE FROM t WHERE id=1")
        assert gate.approved is True

    def test_write_ask_rejected_blocks(self):
        from datus.cli.bash_mode import run_sql_gate

        cli = _sql_cli()
        with patch("datus.cli.bash_mode._make_input_collector", return_value=lambda a, c: [[""]]):
            gate = run_sql_gate(cli, "DELETE FROM t WHERE id=1")
        assert gate.approved is False
        assert gate.error

    def test_plugin_transformer_rewrites_sql(self):
        """When a plugin transformer matches execute_sql, its rewrite becomes the
        effective SQL (the plugin-hook arg layer, replicated for manual calls)."""
        from datus.cli.bash_mode import run_sql_gate

        cli = _sql_cli()
        cli.agent_config.plugins_enabled = True
        cli.agent_config.active_plugin_names = lambda: {"acme"}

        def _rewrite(tool_name, args, ctx):
            return {"sql": args["sql"] + " /* tenant=acme */"}

        with (
            patch(
                "datus.plugins.registry.collect_plugin_tool_transformers",
                return_value={"db_tools.execute_sql": [_rewrite]},
            ),
        ):
            gate = run_sql_gate(cli, "SELECT 1")
        assert gate.approved is True
        assert gate.sql == "SELECT 1 /* tenant=acme */"

    def test_plugin_transformer_denial_blocks(self):
        from datus.cli.bash_mode import run_sql_gate

        cli = _sql_cli()
        cli.agent_config.plugins_enabled = True
        cli.agent_config.active_plugin_names = lambda: {"acme"}

        def _deny(tool_name, args, ctx):
            raise ValueError("blocked by policy")

        with patch(
            "datus.plugins.registry.collect_plugin_tool_transformers",
            return_value={"db_tools.execute_sql": [_deny]},
        ):
            gate = run_sql_gate(cli, "SELECT 1")
        assert gate.approved is False
        assert "blocked by policy" in gate.error

    def test_sql_policy_rewrite_applied_for_read(self):
        """``_enforce_sql_policy`` (via the node's db_func_tool) can rewrite a read."""
        from datus.cli.bash_mode import run_sql_gate

        node = MagicMock()
        node.permission_manager = None  # force CLI-owned manager
        node.db_func_tool._enforce_sql_policy.return_value = "SELECT 1 WHERE x=1"
        cli = _sql_cli(node=node)
        # node has no permission_manager → CLI-owned; keep node for db_func_tool.
        node.permission_manager = None
        gate = run_sql_gate(cli, "SELECT 1")
        assert gate.approved is True
        assert gate.sql == "SELECT 1 WHERE x=1"
        node.db_func_tool._enforce_sql_policy.assert_called_once()
