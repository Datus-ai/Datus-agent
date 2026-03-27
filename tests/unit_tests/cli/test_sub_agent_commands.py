# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/sub_agent_commands.py — SubAgentCommands.

All external dependencies (SubAgentManager, run_wizard) are mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from datus.cli.sub_agent_commands import SubAgentCommands
from datus.utils.constants import SYS_SUB_AGENTS

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_cli():
    cli = MagicMock()
    cli.console = MagicMock()
    cli.agent_config = MagicMock()
    cli.agent_config.current_namespace = "test_ns"
    cli.agent_config.agentic_nodes = {}
    cli.available_subagents = set()
    return cli


def _make_sub_agent_config(name="my_agent"):
    cfg = MagicMock()
    cfg.system_prompt = name
    cfg.scoped_context = None
    cfg.scoped_kb_path = None
    cfg.tools = "db_tools.*"
    cfg.mcp = ""
    cfg.rules = []
    cfg.has_scoped_context.return_value = False
    return cfg


@pytest.fixture
def cmds():
    cli = _make_cli()
    sc = SubAgentCommands(cli)
    mock_manager = MagicMock()
    sc._sub_agent_manager = mock_manager
    return sc


# ---------------------------------------------------------------------------
# Tests: cmd routing
# ---------------------------------------------------------------------------


class TestCmdRouting:
    def test_empty_args_shows_help(self, cmds):
        cmds._show_help = MagicMock()
        cmds.cmd("")
        cmds._show_help.assert_called_once()

    def test_list_routes_to_list(self, cmds):
        cmds._list_agents = MagicMock()
        cmds.cmd("list")
        cmds._list_agents.assert_called_once()

    def test_add_routes_to_add(self, cmds):
        cmds._cmd_add_agent = MagicMock()
        cmds.cmd("add")
        cmds._cmd_add_agent.assert_called_once()

    def test_remove_no_name_prints_error(self, cmds):
        cmds.cmd("remove")
        cmds.cli_instance.console.print.assert_called()

    def test_remove_with_name_routes(self, cmds):
        cmds._remove_agent = MagicMock()
        cmds.cmd("remove myagent")
        cmds._remove_agent.assert_called_once_with("myagent")

    def test_update_no_name_prints_error(self, cmds):
        cmds.cmd("update")
        cmds.cli_instance.console.print.assert_called()

    def test_update_with_name_routes(self, cmds):
        cmds._cmd_update_agent = MagicMock()
        cmds.cmd("update myagent")
        cmds._cmd_update_agent.assert_called_once_with("myagent")

    def test_unknown_command_shows_help(self, cmds):
        cmds._show_help = MagicMock()
        cmds.cmd("unknown_xyz")
        cmds._show_help.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _show_help
# ---------------------------------------------------------------------------


class TestShowHelp:
    def test_show_help_prints(self, cmds):
        cmds._show_help()
        assert cmds.cli_instance.console.print.call_count >= 2


# ---------------------------------------------------------------------------
# Tests: _remove_agent
# ---------------------------------------------------------------------------


class TestRemoveAgent:
    def test_remove_sys_agent_blocked(self, cmds):
        sys_agent = list(SYS_SUB_AGENTS)[0]
        cmds._remove_agent(sys_agent)
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("cannot be removed" in c for c in calls)

    def test_remove_not_found(self, cmds):
        cmds._sub_agent_manager.remove_agent.return_value = False
        cmds._remove_agent("nonexistent")
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("not found" in c for c in calls)

    def test_remove_success(self, cmds):
        cmds._sub_agent_manager.remove_agent.return_value = True
        cmds._refresh_agent_config = MagicMock()
        cmds._remove_agent("my_agent")
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("my_agent" in c for c in calls)
        cmds._refresh_agent_config.assert_called_once()

    def test_remove_exception_prints_error(self, cmds):
        cmds._sub_agent_manager.remove_agent.side_effect = RuntimeError("db error")
        cmds._remove_agent("my_agent")
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("db error" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: _cmd_update_agent
# ---------------------------------------------------------------------------


class TestCmdUpdateAgent:
    def test_update_sys_agent_blocked(self, cmds):
        sys_agent = list(SYS_SUB_AGENTS)[0]
        cmds._cmd_update_agent(sys_agent)
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("cannot be modified" in c for c in calls)

    def test_update_not_found(self, cmds):
        cmds._sub_agent_manager.get_agent.return_value = None
        cmds._cmd_update_agent("nonexistent")
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("not found" in c for c in calls)

    def test_update_found_delegates_to_do_update(self, cmds):
        existing = _make_sub_agent_config("my_agent")
        cmds._sub_agent_manager.get_agent.return_value = existing
        cmds._do_update_agent = MagicMock()
        cmds._cmd_update_agent("my_agent")
        cmds._do_update_agent.assert_called_once_with(existing, original_name="my_agent")


# ---------------------------------------------------------------------------
# Tests: _format_scoped_context (static)
# ---------------------------------------------------------------------------


class TestFormatScopedContext:
    def test_none_returns_empty(self):
        result = SubAgentCommands._format_scoped_context(None)
        assert result == ""

    def test_string_returns_as_is(self):
        result = SubAgentCommands._format_scoped_context("plain string")
        assert result == "plain string"

    def test_dict_returns_syntax(self):
        from rich.syntax import Syntax

        result = SubAgentCommands._format_scoped_context({"tables": ["t1"], "metrics": [], "sqls": []})
        assert isinstance(result, Syntax)

    def test_non_dict_non_str_returns_str(self):
        result = SubAgentCommands._format_scoped_context(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# Tests: _do_update_agent — wizard cancelled
# ---------------------------------------------------------------------------


class TestDoUpdateAgent:
    def test_wizard_cancelled_prints_message(self, cmds):
        with patch("datus.cli.sub_agent_commands.run_wizard", return_value=None):
            cmds._do_update_agent()
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("cancelled" in c for c in calls)

    def test_wizard_exception_prints_error(self, cmds):
        with patch("datus.cli.sub_agent_commands.run_wizard", side_effect=RuntimeError("wizard boom")):
            cmds._do_update_agent()
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("wizard boom" in c for c in calls)

    def test_reserved_name_blocked(self, cmds):
        sys_agent = list(SYS_SUB_AGENTS)[0]
        result_agent = _make_sub_agent_config(sys_agent)
        with patch("datus.cli.sub_agent_commands.run_wizard", return_value=result_agent):
            cmds._do_update_agent()
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("reserved" in c for c in calls)

    def test_save_success_prints_confirmation(self, cmds):
        result_agent = _make_sub_agent_config("my_new_agent")
        cmds._sub_agent_manager.save_agent.return_value = {
            "changed": True,
            "kb_action": None,
            "config_path": "/path/to/agent.yml",
            "prompt_path": None,
        }
        cmds._refresh_agent_config = MagicMock()
        with patch("datus.cli.sub_agent_commands.run_wizard", return_value=result_agent):
            cmds._do_update_agent()
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("my_new_agent" in c for c in calls)

    def test_no_changes_skips_save(self, cmds):
        result_agent = _make_sub_agent_config("my_agent")
        cmds._sub_agent_manager.save_agent.return_value = {"changed": False}
        with patch("datus.cli.sub_agent_commands.run_wizard", return_value=result_agent):
            cmds._do_update_agent()
        calls = [str(c) for c in cmds.cli_instance.console.print.call_args_list]
        assert any("No changes" in c for c in calls)


# ---------------------------------------------------------------------------
# Tests: _refresh_agent_config
# ---------------------------------------------------------------------------


class TestRefreshAgentConfig:
    def test_refresh_updates_available_subagents(self, cmds):
        cmds._sub_agent_manager.list_agents.return_value = {"my_sub": {"system_prompt": "my_sub"}}
        cmds.cli_instance.agent_config.agentic_nodes = {}
        cmds._refresh_agent_config()
        # Should not raise

    def test_refresh_handles_exception_gracefully(self, cmds):
        cmds._sub_agent_manager.list_agents.side_effect = RuntimeError("oops")
        # Should not raise (defensive)
        cmds._refresh_agent_config()
