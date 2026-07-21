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
def test_save_command(mock_args, capsys):
    """
    Tests the '/save' command with successful file save.
    """
    from datus.schemas.node_models import SQLContext

    # Create mock SQL context
    mock_sql_context = SQLContext(
        sql_query="SELECT * FROM schools",
        sql_return="[{'id': 1, 'name': 'School A'}]",
        row_count=1,
    )

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = ["/save", EOFError]

        with (
            patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt,
            patch("datus.cli.cli_context.CliContext.get_last_sql_context") as mock_context,
            patch("datus.cli.agent_commands.OutputTool.execute") as mock_output,
        ):
            mock_internal_prompt.side_effect = [
                "json",  # file_type
                "/tmp",  # target_dir
                "test_output",  # file_name
            ]
            mock_context.return_value = mock_sql_context
            mock_output.return_value = type("MockResult", (), {"output": "/tmp/test_output.json"})()

            cli = DatusCLI(args=mock_args)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "Save Output" in stdout
    assert "/tmp/test_output.json" in stdout
