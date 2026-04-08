# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for SessionSearchTool.
"""

import json
import sqlite3

import pytest

from datus.tools.func_tool.session_search_tool import SessionSearchTool


@pytest.fixture
def tool(tmp_path):
    return SessionSearchTool(sessions_dir=str(tmp_path))


def _create_session_db(db_path, session_id, messages):
    """Helper to create a session DB with messages."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        "CREATE TABLE agent_messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT NOT NULL, "
        "message_data TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO agent_sessions (session_id) VALUES (?)", (session_id,))
    for msg in messages:
        conn.execute(
            "INSERT INTO agent_messages (session_id, message_data) VALUES (?, ?)",
            (session_id, json.dumps(msg)),
        )
    conn.commit()
    conn.close()


class TestSessionSearchTool:
    """Tests for search_skill_usage."""

    def test_missing_sessions_dir(self):
        """Should return error when sessions directory doesn't exist."""
        tool = SessionSearchTool(sessions_dir="/nonexistent/path")
        result = tool.search_skill_usage("some-skill")
        assert result.success == 0

    def test_empty_sessions_dir(self, tool, tmp_path):
        """Should return empty matches when no .db files exist."""
        result = tool.search_skill_usage("some-skill")
        assert result.success == 1
        assert result.result["matches"] == []
        assert result.result["total_sessions_scanned"] == 0

    def test_no_matches(self, tool, tmp_path):
        """Should return empty when skill not found in any session."""
        _create_session_db(
            tmp_path / "session_1.db",
            "session_1",
            [{"type": "function_call", "name": "list_tables", "arguments": "{}"}],
        )
        result = tool.search_skill_usage("nonexistent-skill")
        assert result.success == 1
        assert result.result["total_matches"] == 0
        assert result.result["total_sessions_scanned"] == 1

    def test_finds_skill_usage(self, tool, tmp_path):
        """Should find sessions where load_skill was called with target skill."""
        _create_session_db(
            tmp_path / "session_with_skill.db",
            "test_session",
            [
                {"type": "function_call", "name": "load_skill", "arguments": '{"skill_name": "bitcoin-analysis"}'},
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{'success': 1, 'result': 'skill content...'}",
                },
                {
                    "type": "function_call",
                    "name": "read_query",
                    "arguments": '{"sql": "SELECT * FROM gold_vs_bitcoin"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "{'success': 0, 'error': 'Table not found'}",
                },
            ],
        )
        result = tool.search_skill_usage("bitcoin-analysis")
        assert result.success == 1
        assert result.result["total_matches"] == 1
        match = result.result["matches"][0]
        assert match["session_id"] == "test_session"
        assert match["tool_call_count"] == 2
        assert match["error_count"] == 1
        assert "load_skill" in match["tools_used"]
        assert "read_query" in match["tools_used"]

    def test_respects_max_sessions(self, tool, tmp_path):
        """Should cap results at max_sessions."""
        for i in range(5):
            _create_session_db(
                tmp_path / f"session_{i}.db",
                f"session_{i}",
                [{"type": "function_call", "name": "load_skill", "arguments": '{"skill_name": "my-skill"}'}],
            )
        result = tool.search_skill_usage("my-skill", max_sessions=2)
        assert result.success == 1
        assert result.result["total_matches"] == 2

    def test_available_tools(self, tool):
        """Should return search_skill_usage tool."""
        tools = tool.available_tools()
        assert len(tools) == 1
        assert tools[0].name == "search_skill_usage"

    def test_skips_corrupted_db(self, tool, tmp_path):
        """Should skip corrupted .db files gracefully."""
        # Write garbage to a .db file
        (tmp_path / "bad.db").write_text("not a sqlite database")
        # Also create a valid one
        _create_session_db(
            tmp_path / "good.db",
            "good_session",
            [{"type": "function_call", "name": "load_skill", "arguments": '{"skill_name": "test-skill"}'}],
        )
        result = tool.search_skill_usage("test-skill")
        assert result.success == 1
        assert result.result["total_matches"] == 1
