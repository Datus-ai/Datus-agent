"""
Regression Tests: Streamlit Web UI Components (R-13)

Tests web UI components without requiring a running Streamlit server:
- ConfigManager: Configuration loading and namespace discovery
- SessionLoader: Session message loading and validation
- ChatExecutor: SQL/response extraction and action formatting
"""

import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

from datus.cli.web.chat_executor import ChatExecutor
from datus.cli.web.config_manager import ConfigManager, create_cli_args, get_available_namespaces
from datus.cli.web.session_loader import SessionLoader
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.utils.loggings import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(scope="module", autouse=True)
def setup_env():
    load_dotenv()
    from datus.cli.web.config_manager import _load_config_cached

    _load_config_cached.cache_clear()


@pytest.fixture
def mock_streamlit():
    """Mock streamlit module for tests that call create_cli_args."""
    mock_st = MagicMock()
    mock_st.session_state = {}
    original_st = sys.modules.get("streamlit")
    sys.modules["streamlit"] = mock_st
    yield mock_st
    if original_st is not None:
        sys.modules["streamlit"] = original_st
    else:
        sys.modules.pop("streamlit", None)


@pytest.fixture
def session_loader():
    return SessionLoader()


@pytest.fixture
def chat_executor():
    return ChatExecutor()


# ============================================================
# ConfigManager Tests
# ============================================================
@pytest.mark.regression
class TestWebConfigManager:
    """Test ConfigManager configuration loading and namespace discovery."""

    def test_get_available_namespaces(self):
        """R13-01: get_available_namespaces returns namespace list from config."""
        config_path = str(PROJECT_ROOT / "tests" / "conf" / "agent.yml")
        namespaces = get_available_namespaces(config_path)
        assert isinstance(namespaces, list)
        assert len(namespaces) > 0, "Should have at least one namespace"
        assert "ssb_sqlite" in namespaces, "Should contain ssb_sqlite namespace"

    def test_get_available_namespaces_invalid_path(self):
        """R13-02: get_available_namespaces returns empty list for invalid config."""
        namespaces = get_available_namespaces("/nonexistent/path/config.yml")
        assert isinstance(namespaces, list)
        assert len(namespaces) == 0

    def test_create_cli_args(self, mock_streamlit):
        """R13-03: create_cli_args generates correct argument namespace."""
        config_path = str(PROJECT_ROOT / "tests" / "conf" / "agent.yml")
        args = create_cli_args(config_path=config_path, namespace="ssb_sqlite")

        assert args.namespace == "ssb_sqlite"
        assert args.non_interactive is True
        assert args.disable_detail_views is True
        assert hasattr(args, "config")
        assert hasattr(args, "storage_path")

    def test_setup_config_and_models(self, mock_streamlit):
        """R13-04: ConfigManager.setup_config returns DatusCLI with models available."""
        cm = ConfigManager()
        config_path = str(PROJECT_ROOT / "tests" / "conf" / "agent.yml")
        cli = cm.setup_config(config_path=config_path, namespace="ssb_sqlite")

        assert cli is not None, "setup_config should return a DatusCLI instance"
        assert cli.streamlit_mode is True, "streamlit_mode should be True"
        assert cm.cli is cli

        models = cm.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0, "Should have at least one model configured"


# ============================================================
# SessionLoader Tests
# ============================================================
@pytest.mark.regression
class TestWebSessionLoader:
    """Test SessionLoader session message loading and validation."""

    def test_invalid_session_id_path_traversal(self, session_loader):
        """R13-05: Path traversal session_id is rejected."""
        messages = session_loader.get_session_messages("../../etc/passwd")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_invalid_session_id_special_chars(self, session_loader):
        """R13-06: Session IDs with special characters are rejected."""
        messages = session_loader.get_session_messages("session;DROP TABLE")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_nonexistent_session(self, session_loader):
        """R13-07: Nonexistent session returns empty list without error."""
        messages = session_loader.get_session_messages("nonexistent_session_99999")
        assert isinstance(messages, list)
        assert len(messages) == 0

    def test_load_session_roundtrip(self, session_loader):
        """R13-08: Messages written to session DB can be read back by SessionLoader."""
        from datus.utils.path_manager import get_path_manager

        session_id = f"test_roundtrip_{uuid.uuid4().hex[:8]}"
        sessions_dir = get_path_manager().sessions_dir
        sessions_dir.mkdir(parents=True, exist_ok=True)
        db_path = sessions_dir / f"{session_id}.db"

        try:
            # Create test session DB with schema matching SessionManager
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

            # Insert user message
            user_msg = {"role": "user", "content": "How many customers are there?"}
            conn.execute(
                "INSERT INTO agent_messages (session_id, message_data, created_at) " "VALUES (?, ?, datetime('now'))",
                (session_id, json.dumps(user_msg)),
            )

            # Insert assistant message with final SQL output
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

            # Read back via SessionLoader
            messages = session_loader.get_session_messages(session_id)
            assert len(messages) >= 1, "Should have at least one message"

            # Verify user message
            user_messages = [m for m in messages if m["role"] == "user"]
            assert len(user_messages) == 1
            assert user_messages[0]["content"] == "How many customers are there?"

            # Verify assistant message with SQL
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            assert len(assistant_messages) >= 1
            assert assistant_messages[0].get("sql") == "SELECT COUNT(*) FROM customer"
            assert assistant_messages[0]["content"] == "There are 30000 customers."

        finally:
            if db_path.exists():
                os.remove(str(db_path))


# ============================================================
# ChatExecutor Tests
# ============================================================
@pytest.mark.regression
class TestWebChatExecutor:
    """Test ChatExecutor SQL extraction and action formatting."""

    def test_extract_sql_and_response_success(self, chat_executor):
        """R13-09: extract_sql_and_response extracts SQL and response from successful action."""
        action = ActionHistory(
            action_id="test-1",
            role=ActionRole.TOOL,
            messages="",
            action_type="execute_sql",
            input={"function_name": "execute_sql", "query": "SELECT COUNT(*) FROM customer"},
            output={"sql": "SELECT COUNT(*) FROM customer", "response": "There are 30000 customers."},
            status=ActionStatus.SUCCESS,
        )
        sql, response = chat_executor.extract_sql_and_response([action], cli=None)
        assert sql == "SELECT COUNT(*) FROM customer"
        assert response is not None
        assert len(response) > 0

    def test_extract_sql_and_response_empty(self, chat_executor):
        """R13-10: extract_sql_and_response returns (None, None) for empty actions."""
        sql, response = chat_executor.extract_sql_and_response([], cli=None)
        assert sql is None
        assert response is None

    def test_extract_sql_and_response_no_output(self, chat_executor):
        """R13-11: extract_sql_and_response handles action without output."""
        action = ActionHistory(
            action_id="test-2",
            role=ActionRole.TOOL,
            messages="",
            action_type="execute_sql",
            input={"function_name": "execute_sql"},
            output=None,
            status=ActionStatus.PROCESSING,
        )
        sql, response = chat_executor.extract_sql_and_response([action], cli=None)
        assert sql is None
        assert response is None

    def test_format_action_tool_success(self, chat_executor):
        """R13-12: format_action_for_stream formats successful tool call with checkmark."""
        action = ActionHistory(
            action_id="test-3",
            role=ActionRole.TOOL,
            messages="",
            action_type="read_query",
            input={"function_name": "read_query", "query": "SELECT 1"},
            output={"result": "1"},
            status=ActionStatus.SUCCESS,
        )
        result = chat_executor.format_action_for_stream(action)
        assert isinstance(result, str)
        assert "read_query" in result
        assert result.startswith("\u2713")

    def test_format_action_tool_processing(self, chat_executor):
        """R13-13: format_action_for_stream formats processing tool with spinner."""
        action = ActionHistory(
            action_id="test-4",
            role=ActionRole.TOOL,
            messages="",
            action_type="describe_table",
            input={"function_name": "describe_table", "table": "customer"},
            output=None,
            status=ActionStatus.PROCESSING,
        )
        result = chat_executor.format_action_for_stream(action)
        assert isinstance(result, str)
        assert "describe_table" in result
        assert "\u27f3" in result

    def test_format_action_thinking(self, chat_executor):
        """R13-14: format_action_for_stream formats assistant thinking message."""
        action = ActionHistory(
            action_id="test-5",
            role=ActionRole.ASSISTANT,
            messages="I need to look at the customer table first.",
            action_type="thinking",
            input=None,
            output=None,
            status=ActionStatus.SUCCESS,
        )
        result = chat_executor.format_action_for_stream(action)
        assert isinstance(result, str)
        assert "Thinking:" in result
        assert "customer table" in result

    def test_format_action_empty_message(self, chat_executor):
        """R13-15: format_action_for_stream returns empty string for empty message."""
        action = ActionHistory(
            action_id="test-6",
            role=ActionRole.ASSISTANT,
            messages="",
            action_type="thinking",
            input=None,
            output=None,
            status=ActionStatus.SUCCESS,
        )
        result = chat_executor.format_action_for_stream(action)
        assert result == ""

    def test_format_action_truncates_long_message(self, chat_executor):
        """R13-16: format_action_for_stream truncates messages longer than 100 chars."""
        long_message = "A" * 200
        action = ActionHistory(
            action_id="test-7",
            role=ActionRole.ASSISTANT,
            messages=long_message,
            action_type="thinking",
            input=None,
            output=None,
            status=ActionStatus.SUCCESS,
        )
        result = chat_executor.format_action_for_stream(action)
        assert isinstance(result, str)
        assert "..." in result
        assert len(result) < 200
