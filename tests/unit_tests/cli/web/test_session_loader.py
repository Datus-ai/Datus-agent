# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/cli/web/session_loader.py.

Tests cover:
- SessionLoader._parse_final_output: parsing SQL/output from last action
- SessionLoader.get_session_messages: message aggregation from SQLite

NO MOCK EXCEPT LLM. Uses real SQLite database in tmp_path.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta

from datus.cli.web.session_loader import SessionLoader
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session_db(sessions_dir, session_id, messages):
    """Create a session SQLite database with given messages.

    Each message is a tuple of (message_data_json_string, created_at_timestamp).
    """
    os.makedirs(sessions_dir, exist_ok=True)
    db_path = os.path.join(sessions_dir, f"{session_id}.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_sessions ("
            "session_id TEXT PRIMARY KEY, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "message_data TEXT NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute("INSERT OR IGNORE INTO agent_sessions (session_id) VALUES (?)", (session_id,))
        for msg_data, ts in messages:
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, ?)",
                (session_id, msg_data, ts),
            )
        conn.commit()
    return db_path


class TestParseOutputFromAction:
    """Tests for SessionLoader._parse_final_output."""

    def test_parse_final_output_with_sql(self):
        """_parse_final_output extracts SQL from assistant action messages."""
        loader = SessionLoader()
        action = ActionHistory(
            action_id="test1",
            role=ActionRole.ASSISTANT,
            messages=json.dumps({"sql": "SELECT * FROM t", "output": "3 rows"}),
            action_type="chat_response",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": "", "timestamp": "2025-01-01"}

        result = loader._parse_final_output(action, group)

        assert result is not None
        assert result.role == ActionRole.ASSISTANT
        assert result.status == ActionStatus.SUCCESS
        assert group["sql"] == "SELECT * FROM t"
        assert group["content"] == "3 rows"

    def test_parse_final_output_non_json_returns_none(self):
        """_parse_final_output returns None for non-JSON messages."""
        loader = SessionLoader()
        action = ActionHistory(
            action_id="test2",
            role=ActionRole.ASSISTANT,
            messages="Just a plain text response",
            action_type="chat_response",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output(action, group)
        assert result is None

    def test_parse_final_output_tool_role_returns_none(self):
        """_parse_final_output returns None for non-assistant actions."""
        loader = SessionLoader()
        action = ActionHistory(
            action_id="test3",
            role=ActionRole.TOOL,
            messages="tool output",
            action_type="read_query",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output(action, group)
        assert result is None


class TestGetSessionMessages:
    """Tests for SessionLoader.get_session_messages with real SQLite."""

    def test_get_messages_user_and_assistant(self, real_agent_config):
        """get_session_messages returns user and assistant messages in order."""
        from datus.utils.path_manager import get_path_manager

        loader = SessionLoader()
        sessions_dir = str(get_path_manager().sessions_dir)
        session_id = "loader_test_01"
        base = datetime(2025, 6, 1, 10, 0, 0)

        msgs = [
            (json.dumps({"role": "user", "content": "Hello"}), base.isoformat()),
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps({"sql": "SELECT 1", "output": "OK"})}],
                    }
                ),
                (base + timedelta(seconds=1)).isoformat(),
            ),
        ]
        _create_session_db(sessions_dir, session_id, msgs)

        result = loader.get_session_messages(session_id)
        assert len(result) >= 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_get_messages_with_tool_calls(self, real_agent_config):
        """get_session_messages handles function_call and function_call_output messages."""
        from datus.utils.path_manager import get_path_manager

        loader = SessionLoader()
        sessions_dir = str(get_path_manager().sessions_dir)
        session_id = "loader_test_02"
        base = datetime(2025, 6, 1, 10, 0, 0)

        msgs = [
            (json.dumps({"role": "user", "content": "Find students"}), base.isoformat()),
            (
                json.dumps(
                    {
                        "type": "function_call",
                        "name": "read_query",
                        "call_id": "call_001",
                        "arguments": json.dumps({"query": "SELECT * FROM students"}),
                    }
                ),
                (base + timedelta(seconds=1)).isoformat(),
            ),
            (
                json.dumps({"type": "function_call_output", "call_id": "call_001", "output": "3 rows returned"}),
                (base + timedelta(seconds=2)).isoformat(),
            ),
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"sql": "SELECT * FROM students", "output": "Found 3 students"}),
                            }
                        ],
                    }
                ),
                (base + timedelta(seconds=3)).isoformat(),
            ),
        ]
        _create_session_db(sessions_dir, session_id, msgs)

        result = loader.get_session_messages(session_id)
        assert len(result) >= 2
        # First should be user
        assert result[0]["role"] == "user"
        # Second should be assistant with actions
        assert result[1]["role"] == "assistant"
        assert "actions" in result[1]
        assert len(result[1]["actions"]) >= 2  # At least tool call + result

    def test_get_messages_invalid_session_id(self):
        """get_session_messages returns empty list for invalid session ID format."""
        loader = SessionLoader()
        result = loader.get_session_messages("../../../etc/passwd")
        assert result == []

    def test_get_messages_nonexistent_session(self, real_agent_config):
        """get_session_messages returns empty list for nonexistent session."""
        loader = SessionLoader()
        result = loader.get_session_messages("nonexistent_session_xyz")
        assert result == []

    def test_get_messages_flush_assistant_on_user_message(self, real_agent_config):
        """Assistant group is flushed with actions and progress when user message arrives."""
        from datus.utils.path_manager import get_path_manager

        loader = SessionLoader()
        sessions_dir = str(get_path_manager().sessions_dir)
        session_id = "loader_test_flush_01"
        base = datetime(2025, 6, 1, 10, 0, 0)

        msgs = [
            # First user message
            (json.dumps({"role": "user", "content": "Find students"}), base.isoformat()),
            # Tool call (sets current_assistant_group and current_actions)
            (
                json.dumps(
                    {
                        "type": "function_call",
                        "name": "read_query",
                        "call_id": "call_flush_001",
                        "arguments": json.dumps({"query": "SELECT * FROM students"}),
                    }
                ),
                (base + timedelta(seconds=1)).isoformat(),
            ),
            # Tool output
            (
                json.dumps({"type": "function_call_output", "call_id": "call_flush_001", "output": "3 rows returned"}),
                (base + timedelta(seconds=2)).isoformat(),
            ),
            # Final assistant response with SQL
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"sql": "SELECT * FROM students", "output": "Found 3 students"}),
                            }
                        ],
                    }
                ),
                (base + timedelta(seconds=3)).isoformat(),
            ),
            # Second user message triggers flush of the assistant group (lines 121-129)
            (json.dumps({"role": "user", "content": "Show me more"}), (base + timedelta(seconds=4)).isoformat()),
            # Second assistant response
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"sql": "SELECT COUNT(*) FROM students", "output": "Count: 3"}),
                            }
                        ],
                    }
                ),
                (base + timedelta(seconds=5)).isoformat(),
            ),
        ]
        _create_session_db(sessions_dir, session_id, msgs)

        result = loader.get_session_messages(session_id)

        # Should have: user, assistant (flushed with actions), user, assistant
        assert len(result) >= 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Find students"

        # The flushed assistant group should have actions
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        # First assistant group should have actions from tool calls
        first_assistant = assistant_msgs[0]
        assert "actions" in first_assistant
        assert len(first_assistant["actions"]) >= 2  # tool call + output

        # Second user message should be present
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 2
        assert user_msgs[1]["content"] == "Show me more"
