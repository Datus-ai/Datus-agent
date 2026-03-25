# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/print_mode.py
"""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from datus.schemas.message_content import MessageContent, MessagePayload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    defaults = dict(
        print_mode="hello",
        resume=None,
        subagent=None,
        namespace="test_ns",
        db_type="sqlite",
        db_path=None,
        config=None,
        debug=False,
        no_color=False,
        database="",
        history_file=None,
        save_llm_trace=False,
        web=False,
        port=8501,
        host="localhost",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner._write_payload
# ---------------------------------------------------------------------------


class TestWritePayload:
    def test_writes_json_line(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            payload = MessagePayload(
                message_id="m1",
                role="assistant",
                content=[MessageContent(type="markdown", payload={"content": "hi"})],
            )
            runner._write_payload(payload)

        line = buf.getvalue().strip()
        data = json.loads(line)
        assert data["message_id"] == "m1"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "markdown"


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner._read_interaction_input
# ---------------------------------------------------------------------------


class TestReadInteractionInput:
    def test_parses_valid_payload(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        payload = MessagePayload(
            message_id="m1",
            role="user",
            content=[MessageContent(type="user-interaction", payload={"content": "y"})],
        )
        with patch("sys.stdin", io.StringIO(payload.model_dump_json() + "\n")):
            result = runner._read_interaction_input()
        assert result == "y"

    def test_empty_input(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        with patch("sys.stdin", io.StringIO("\n")):
            result = runner._read_interaction_input()
        assert result == ""

    def test_invalid_json_returns_raw(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        with patch("sys.stdin", io.StringIO("raw text\n")):
            result = runner._read_interaction_input()
        assert result == "raw text"


# ---------------------------------------------------------------------------
# Tests: PrintModeRunner.run (mocked end-to-end)
# ---------------------------------------------------------------------------


class TestPrintModeRun:
    @pytest.mark.asyncio
    async def test_stream_chat_writes_payloads(self):
        """Test that _stream_chat writes payloads to stdout."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages="thinking",
            action_type="llm_generation",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
        )

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            yield mock_action

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        assert output  # at least one JSON line
        data = json.loads(output)
        assert data["message_id"] == "a1"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "thinking"
        assert data["depth"] == 0
        assert data["parent_action_id"] is None

    @pytest.mark.asyncio
    async def test_stream_chat_subagent_hierarchy(self):
        """Test that _stream_chat propagates depth and parent_action_id from subagent actions."""
        from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

        mock_action = ActionHistory(
            action_id="sub_a1",
            role=ActionRole.ASSISTANT,
            messages="subagent thinking",
            action_type="llm_generation",
            input=None,
            output=None,
            status=ActionStatus.PROCESSING,
            depth=1,
            parent_action_id="call_parent_123",
        )

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            yield mock_action

        mock_node.execute_stream_with_interactions = fake_stream

        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            await runner._stream_chat(mock_node)

        output = buf.getvalue().strip()
        data = json.loads(output)
        assert data["message_id"] == "sub_a1"
        assert data["depth"] == 1
        assert data["parent_action_id"] == "call_parent_123"


# ---------------------------------------------------------------------------
# Tests: --resume sets session_id
# ---------------------------------------------------------------------------


class TestResumeSessionId:
    def test_resume_sets_session_id(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="session_abc"))

        assert runner.session_id == "session_abc"

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield  # make it an async generator

        mock_node.execute_stream_with_interactions = fake_stream

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = True

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node),
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
        ):
            runner.run()

        # session_id should be set on the node
        assert mock_node.session_id == "session_abc"

    def test_resume_nonexistent_session_exits(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter"),
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="no_such_session"))

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = False

        with (
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
            pytest.raises(SystemExit, match="not found"),
        ):
            runner.run()

    def test_resume_derives_subagent_from_session_id(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args(resume="gen_sql_session_uuid123"))

        assert runner.subagent_name is None  # not yet resolved

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream

        mock_session_mgr = MagicMock()
        mock_session_mgr.session_exists.return_value = True

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node) as mock_create_node,
            patch("datus.cli.print_mode.create_node_input", return_value=MagicMock()),
            patch("datus.models.session_manager.SessionManager", return_value=mock_session_mgr),
        ):
            runner.run()

        # subagent_name should be derived from session_id
        assert runner.subagent_name == "gen_sql"
        mock_create_node.assert_called_once_with("gen_sql", runner.agent_config, node_id_suffix="_print")


# ---------------------------------------------------------------------------
# Tests: run delegates to factory functions
# ---------------------------------------------------------------------------


class TestRunUsesFactory:
    def test_run_calls_factory(self):
        with (
            patch("datus.cli.print_mode.load_agent_config") as mock_cfg,
            patch("datus.cli.print_mode.AtReferenceCompleter") as mock_completer,
        ):
            mock_cfg.return_value = MagicMock(namespaces=[])
            mock_completer.return_value.parse_at_context.return_value = ([], [], [])
            from datus.cli.print_mode import PrintModeRunner

            runner = PrintModeRunner(_make_args())

        mock_node = MagicMock()
        mock_node.session_id = None

        async def fake_stream(actions):
            return
            yield

        mock_node.execute_stream_with_interactions = fake_stream
        mock_input = MagicMock()

        with (
            patch("datus.cli.print_mode.create_interactive_node", return_value=mock_node) as mock_create_node,
            patch("datus.cli.print_mode.create_node_input", return_value=mock_input) as mock_create_input,
        ):
            runner.run()

        mock_create_node.assert_called_once_with(None, runner.agent_config, node_id_suffix="_print")
        mock_create_input.assert_called_once()
        assert mock_node.input == mock_input
