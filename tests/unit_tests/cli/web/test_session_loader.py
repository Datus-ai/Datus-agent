# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SessionLoader.

Tests session message loading, validation, and security checks
(moved from test_regression_web.py).
"""

import json
import os
import sqlite3
import uuid

import pytest

from datus.cli.web.session_loader import SessionLoader
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


@pytest.fixture
def session_loader():
    return SessionLoader()


@pytest.mark.ci
class TestSessionLoader:
    """Unit tests for SessionLoader session message loading and validation."""

    def test_invalid_session_id_path_traversal(self, session_loader):
        """Path traversal session_id is rejected."""
        messages = session_loader.get_session_messages("../../etc/passwd")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_invalid_session_id_special_chars(self, session_loader):
        """Session IDs with special characters are rejected."""
        messages = session_loader.get_session_messages("session;DROP TABLE")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_nonexistent_session(self, session_loader):
        """Nonexistent session returns empty list without error."""
        messages = session_loader.get_session_messages("nonexistent_session_99999")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_load_session_roundtrip(self, tmp_path):
        """Messages written to session DB can be read back by SessionLoader."""
        session_id = f"test_roundtrip_{uuid.uuid4().hex[:8]}"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_loader = SessionLoader(session_dir=str(sessions_dir))
        db_path = sessions_dir / f"{session_id}.db"

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_sessions ("
                "session_id TEXT PRIMARY KEY, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "total_tokens INTEGER DEFAULT 0)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT NOT NULL, "
                "message_data TEXT NOT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute("INSERT INTO agent_sessions (session_id) VALUES (?)", (session_id,))

            user_msg = {"role": "user", "content": "How many customers are there?"}
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) VALUES (?, ?, datetime('now'))",
                (session_id, json.dumps(user_msg)),
            )

            assistant_msg = {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "sql": "SELECT COUNT(*) FROM customer",
                                "output": "There are 30000 customers.",
                            }
                        ),
                    }
                ],
            }
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) "
                "VALUES (?, ?, datetime('now', '+1 second'))",
                (session_id, json.dumps(assistant_msg)),
            )
            conn.commit()
            conn.close()

            messages = session_loader.get_session_messages(session_id)
            assert len(messages) >= 1, "Should have at least one message"

            user_messages = [m for m in messages if m["role"] == "user"]
            assert len(user_messages) == 1
            assert user_messages[0]["content"] == "How many customers are there?"

            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            assert len(assistant_messages) >= 1
            assert assistant_messages[0].get("sql") == "SELECT COUNT(*) FROM customer"
            assert assistant_messages[0]["content"] == "There are 30000 customers."

        finally:
            if db_path.exists():
                os.remove(str(db_path))


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

        result = loader._parse_final_output([action], group)

        assert result is not None
        assert result.role == ActionRole.ASSISTANT
        assert result.status == ActionStatus.SUCCESS
        assert group["sql"] == "SELECT * FROM t"
        assert group["content"] == "3 rows"

    def test_parse_final_output_non_json_sets_content(self):
        """_parse_final_output sets content to raw text for non-JSON messages."""
        loader = SessionLoader()
        action = ActionHistory(
            action_id="test2",
            role=ActionRole.ASSISTANT,
            messages="Just a plain text response",
            action_type="chat_response",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output([action], group)
        assert result is None
        assert group["content"] == "Just a plain text response"

    def test_parse_final_output_tool_role_only_returns_none(self):
        """_parse_final_output returns None when only non-assistant actions exist."""
        loader = SessionLoader()
        action = ActionHistory(
            action_id="test3",
            role=ActionRole.TOOL,
            messages="tool output",
            action_type="read_query",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output([action], group)
        assert result is None
        assert group["content"] == ""

    def test_parse_final_output_finds_last_assistant(self):
        """_parse_final_output searches backwards for the last assistant action."""
        loader = SessionLoader()
        assistant_action = ActionHistory(
            action_id="a1",
            role=ActionRole.ASSISTANT,
            messages=json.dumps({"sql": "SELECT 1", "output": "result"}),
            action_type="thinking",
            status=ActionStatus.SUCCESS,
        )
        tool_action = ActionHistory(
            action_id="a2",
            role=ActionRole.TOOL,
            messages="tool output",
            action_type="read_query",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        # Tool action is last, but assistant action should be found
        result = loader._parse_final_output([assistant_action, tool_action], group)
        assert result is not None
        assert group["sql"] == "SELECT 1"
        assert group["content"] == "result"

    def test_parse_final_output_empty_list(self):
        """_parse_final_output returns None for empty action list."""
        loader = SessionLoader()
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output([], group)
        assert result is None
        assert group["content"] == ""

    def test_parse_final_output_markdown_content(self):
        """_parse_final_output preserves markdown text as content for chat agents."""
        loader = SessionLoader()
        markdown = "## Analysis\n\nHere are the key findings:\n\n| Col | Val |\n|-----|-----|\n| A | 1 |"
        action = ActionHistory(
            action_id="md1",
            role=ActionRole.ASSISTANT,
            messages=markdown,
            action_type="thinking",
            status=ActionStatus.SUCCESS,
        )
        group = {"role": "assistant", "content": ""}

        result = loader._parse_final_output([action], group)
        assert result is None
        assert group["content"] == markdown
