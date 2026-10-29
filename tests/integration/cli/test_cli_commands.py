import re
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

from datus.cli.repl import DatusCLI
from datus.schemas.node_models import TableSchema
from tests.conftest import TEST_DATA_DIR
from tests.integration.conftest import wait_for_agent


@pytest.fixture
def gen_sql_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = TEST_DATA_DIR / "GenerateSQLInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(autouse=True)
def disable_tui_for_prompt_session_tests(monkeypatch):
    """Keep these PromptSession-based CLI tests deterministic under a PTY."""
    monkeypatch.setenv("DATUS_TUI", "0")


@pytest.mark.acceptance
def test_databases_command(mock_args, capsys):
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = ["/databases", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        captured = capsys.readouterr()
        assert "Databases" in captured.out


@pytest.mark.acceptance
def test_tables_command(mock_args, capsys):
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = ["/tables", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        captured = capsys.readouterr()
        assert "Tables in Database" in captured.out
        assert cli._bg_loop.is_closed()
        assert not cli._bg_loop_thread.is_alive()


@pytest.mark.acceptance
def test_sandbox_command_status_and_session_toggle(mock_args, capsys):
    """/sandbox dispatches through the slash handler map: status shows the
    current state and on/off flip the shared SandboxSettings for the session."""
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = ["/sandbox", "/sandbox on", "/sandbox status", "/sandbox off", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        captured = capsys.readouterr()
        assert "Bash sandbox: off" in captured.out
        assert "Bash sandbox on (this session only)" in captured.out
        assert "Bash sandbox: on" in captured.out
        assert "Bash sandbox off (this session only)" in captured.out
        assert cli.agent_config.bash_sandbox.enabled is False


@pytest.mark.nightly
@pytest.mark.product_e2e
def test_chat_command(mock_args, capsys, gen_sql_input: List[Dict[str, Any]]):
    """
    Tests bare chat input for multi-turn conversation and context memory.
    """
    input_data = gen_sql_input[0]["input"]
    sql_task = input_data["sql_task"]
    table_schemas = []
    if "table_schemas" in input_data:
        schemas_list = input_data.get("table_schemas", [])
        table_schemas = [TableSchema.from_dict(item) for item in schemas_list]

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            sql_task["task"],
            "/chat_info",
            EOFError,
        ]
        with (
            patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt,
            patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
        ):
            at_data.return_value = table_schemas, [], [], None
            mock_internal_prompt.side_effect = ["n"]
            cli = DatusCLI(args=mock_args)

            wait_for_agent(cli)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    # Check chat info is present
    assert "Chat Session Info:" in stdout, "Should have chat session info"

    # Check that actions were performed (tool calls happened)
    action_match = re.search(r"Action Count:\s*(\d+)", stdout)
    assert action_match and int(action_match.group(1)) > 0, (
        f"Should have actions (tool calls). stdout contains: {stdout[-500:]}"
    )


@pytest.mark.acceptance
def test_chat_info(mock_args, capsys):
    """
    Tests the '/chat_info' command for the current session state.
    """

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            "/chat_info",
            EOFError,
        ]
        cli = DatusCLI(args=mock_args)
        cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    assert stdout.strip().endswith("No active session.")


@pytest.mark.acceptance
def test_save_command(mock_args, capsys, tmp_path):
    """Save a real query result through the REPL and production exporter."""
    import json

    from datus.schemas.node_models import SQLContext

    with patch("datus.cli.repl.PromptSession.prompt", side_effect=["/save", EOFError]):
        cli = DatusCLI(args=mock_args)
        query = "SELECT 42 AS answer"
        result = cli.db_connector.execute_query(query, result_format="list")
        assert result.success, result.error
        cli.cli_context.add_sql_context(
            SQLContext(sql_query=query, sql_return=json.dumps(result.sql_return), row_count=1)
        )
        with patch.object(cli, "prompt_input", side_effect=["json", str(tmp_path), "query_result"]):
            cli.run()

    saved = json.loads((tmp_path / "query_result.json").read_text())
    assert saved["gen_sql"] == query
    assert json.loads(saved["sql_result"]) == [{"answer": 42}]
    assert "query_result.json" in capsys.readouterr().out
