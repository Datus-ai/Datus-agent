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
def schema_linking_input() -> List[Dict[str, Any]]:
    """Load test data from YAML file"""
    yaml_path = TEST_DATA_DIR / "SchemaLinkingInput.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


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


def _assert_stdout_contains_exactly_one(stdout: str, expected_messages: tuple[str, ...], context: str) -> None:
    matched_messages = [message for message in expected_messages if message in stdout]
    assert len(matched_messages) == 1, (
        f"Expected exactly one {context} message from {expected_messages}, "
        f"matched {matched_messages}. stdout: {stdout[:500]}"
    )


# This is now a true integration test
@pytest.mark.acceptance
def test_schema_linking(mock_args, capsys, schema_linking_input: List[Dict[str, Any]]):
    """
    Tests the '!sl' command against the real execution logic.
    Asserts that the command runs and prints the result table structure.
    """
    input_data = schema_linking_input[0]["input"]
    with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
        mock_repl_prompt.side_effect = ["!sl", EOFError]

        with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt:
            # Mocks user input for: input_text, database_name, top_n
            mock_internal_prompt.side_effect = [
                input_data["input_text"],
                input_data["database_name"],
                "5",
            ]

            cli = DatusCLI(args=mock_args)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "Schema Linking" in stdout
    assert (
        "relevant tables and" in stdout and "Schema Linking Results" in stdout
    ) or "No relevant tables found." in stdout
    assert "Error during schema linking" not in stdout
    assert "Traceback" not in stdout


# This is now a true integration test
@pytest.mark.acceptance
def test_search_reference_sql(mock_args, capsys, schema_linking_input: List[Dict[str, Any]]):
    """
    Tests the '!sq' and '!search_sql' commands against the real execution logic.
    Asserts that the command runs and prints the result table structure.
    """
    input_data = schema_linking_input[0]["input"]
    with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
        mock_repl_prompt.side_effect = ["!sq", EOFError]

        with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt:
            mock_internal_prompt.side_effect = [
                input_data["input_text"],
                "",  # subject_path
                "5",
            ]

            cli = DatusCLI(args=mock_args)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "Search Reference SQL" in stdout
    _assert_stdout_contains_exactly_one(
        stdout,
        ("Reference SQL Search Results", "No reference SQL queries found."),
        "reference SQL result",
    )
    assert "Error searching reference sql:" not in stdout
    assert "Traceback" not in stdout


# This is now a true integration test
@pytest.mark.acceptance
def test_search_metrics(mock_args, capsys, schema_linking_input: List[Dict[str, Any]]):
    """
    Tests the '!search_metrics' command against the real execution logic.
    Asserts that the command runs and prints the result table structure.
    """
    input_data = schema_linking_input[0]["input"]
    with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
        mock_repl_prompt.side_effect = ["!sm", EOFError]
        with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt:
            mock_internal_prompt.side_effect = [
                input_data["input_text"],
                "",
                "5",
            ]
            cli = DatusCLI(args=mock_args)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "Search Metrics" in stdout
    assert ("Found" in stdout and "Metrics Search Results" in stdout) or "No metrics found." in stdout
    assert "Error searching metrics" not in stdout


@pytest.mark.acceptance
def test_bash_command_allowed(mock_args, capsys):
    with (
        patch("datus.cli.repl.PromptSession.prompt") as mock_prompt,
        patch("subprocess.run") as mock_run,
    ):
        mock_prompt.side_effect = ["!bash ls -l", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        mock_run.assert_called_once_with("ls -l", shell=True, capture_output=True, text=True, timeout=10)


@pytest.mark.acceptance
def test_bash_command_denied(mock_args, capsys):
    with (
        patch("datus.cli.repl.PromptSession.prompt") as mock_prompt,
        patch("subprocess.run") as mock_run,
    ):
        mock_prompt.side_effect = ["!bash rm -rf ./temp.temp", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "Command 'rm' not in whitelist" in captured.out


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


@pytest.mark.nightly
@pytest.mark.product_e2e
def test_chat_command_with_ext_knowledge(mock_args):
    """
    Tests bare chat input with ext_knowledge context.
    Verifies that the query with 'consider all knowledge' still completes through
    the real CLI chat path and produces a database-grounded answer.
    """
    # bird california_schools q2
    question = (
        "Please list the zip code of all the charter schools "
        "in Fresno County Office of Education. consider all knowledge"
    )

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            question,
            EOFError,
        ]
        with (
            patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt,
        ):
            mock_internal_prompt.side_effect = ["n"]
            cli = DatusCLI(args=mock_args)

            wait_for_agent(cli)
            cli.run()

    # Use internal state for assertions instead of capsys,
    # because Rich Live streaming display may not be fully captured by capsys.
    actions = cli.actions.get_actions()

    # Find the final chat_response action which contains execution_stats
    chat_response = [a for a in actions if a.action_type == "chat_response"]
    assert len(chat_response) == 1, (
        "Should have exactly one chat_response action. "
        f"Recent actions: {[(a.action_type, str(a.status), a.output) for a in actions[-5:]]}"
    )

    response_output = chat_response[0].output
    assert response_output.get("success") is True, "Chat response should be successful."

    # Check execution stats for substantive database tool usage. In the nightly
    # fixture there may be no indexed external knowledge store, so requiring
    # knowledge tools here makes the real-LLM test nondeterministic.
    exec_stats = response_output.get("execution_stats", {})
    tools_used = exec_stats.get("tools_used", [])
    database_grounding_tools = {"read_query", "execute_reference_template"}
    assert database_grounding_tools & set(tools_used), (
        f"Should ground the answer with a database-backed tool. Got: {tools_used}"
    )

    # Check that the response includes query-derived content.
    # The CLI now routes bare text to chat, and the final answer may summarize
    # the SQL instead of returning the raw statement first.
    response_text = response_output.get("response", "")
    response_upper = response_text.upper()
    assert "FRESNO" in response_upper
    assert any(label in response_upper for label in ("ZIP", "POSTAL")), response_text
    known_zip_tokens = {"93706", "93726", "93628", "93662"}
    matched_zip_tokens = known_zip_tokens & set(re.findall(r"\b\d{5}\b", response_text))
    assert matched_zip_tokens != set(), response_text

    # Check that a chat node was created and has an active session
    current_node = cli.chat_commands.current_node
    if current_node is None:
        raise AssertionError("Should have an active chat node.")
    tool_names = {tool.name for tool in (getattr(current_node, "tools", None) or [])}
    assert "search_knowledge" in tool_names, (
        f"Nightly ext_knowledge fixture did not register search_knowledge. Available tools: {sorted(tool_names)}"
    )
    session_info = cli.run_on_bg_loop(current_node.get_session_info())
    assert session_info.get("session_id", "").startswith("chat_session_")
    assert session_info.get("action_count") == exec_stats.get("total_actions")


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
    Tests the '!save' command with successful file save.
    """
    from datus.schemas.node_models import SQLContext

    # Create mock SQL context
    mock_sql_context = SQLContext(
        sql_query="SELECT * FROM schools",
        sql_return="[{'id': 1, 'name': 'School A'}]",
        row_count=1,
    )

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = ["!save", EOFError]

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


# ── Search edge case tests (merged from test_cli_search.py) ──


@pytest.mark.nightly
class TestCLISearch:
    """N12: CLI search command edge case tests."""

    def test_search_document_command(self, mock_args, capsys):
        """N12-04: !sd (search_document) command executes and returns results."""
        with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
            mock_repl_prompt.side_effect = ["!sd", EOFError]

            with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal:
                # !sd prompts: platform, version, keywords, top_n
                mock_internal.side_effect = [
                    "snowflake",  # platform name
                    "",  # version (optional)
                    "SELECT, WHERE",  # keywords
                    "5",  # top_n
                ]

                cli = DatusCLI(args=mock_args)
                wait_for_agent(cli)
                cli.run()

        captured = capsys.readouterr()
        stdout = captured.out

        # Command should execute
        assert "Search Document" in stdout, f"Should show 'Search Document' header, got: {stdout[:200]}"
        # Should not have unhandled exceptions
        assert "Traceback" not in stdout, "Should not have Python traceback in output"

    def test_schema_linking_no_results(self, mock_args, capsys):
        """N12-05: !sl with nonsense query handles gracefully."""
        with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
            mock_repl_prompt.side_effect = ["!sl", EOFError]

            with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal:
                mock_internal.side_effect = [
                    "xyznonexistent_random_query_12345_abcdef",  # nonsense query
                    "california_schools",  # database
                    "5",  # top_n
                ]

                cli = DatusCLI(args=mock_args)
                wait_for_agent(cli)
                cli.run()

        captured = capsys.readouterr()
        stdout = captured.out

        # Command should execute without crash
        assert "Schema Linking" in stdout, f"Should show 'Schema Linking' header, got: {stdout[:200]}"
        # Should not crash
        assert "Traceback" not in stdout, "Should not have Python traceback"
        assert "Error during schema linking" not in stdout, "Should not have error during schema linking"

    def test_search_reference_sql_with_subject_path(self, mock_args, capsys):
        """N12-06: !sq with subject_path filter works correctly."""
        with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
            mock_repl_prompt.side_effect = ["!sq", EOFError]

            with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal:
                mock_internal.side_effect = [
                    "schools with high test scores",  # query_text
                    "california_schools",  # subject_path
                    "5",  # top_n
                ]

                cli = DatusCLI(args=mock_args)
                wait_for_agent(cli)
                cli.run()

        captured = capsys.readouterr()
        stdout = captured.out

        # Command should execute
        assert "Search Reference SQL" in stdout, f"Should show search header, got: {stdout[:200]}"
        # Should have results or no-results message
        _assert_stdout_contains_exactly_one(
            stdout,
            ("Reference SQL Search Results", "No reference SQL"),
            "reference SQL result",
        )
        # Should not have errors
        assert "Error searching reference sql:" not in stdout, "Should not have error message"
        assert "Traceback" not in stdout

    def test_search_metrics_special_characters(self, mock_args, capsys):
        """N12-07: !sm handles special characters in query gracefully."""
        with patch("datus.cli.repl.PromptSession.prompt") as mock_repl_prompt:
            mock_repl_prompt.side_effect = ["!sm", EOFError]

            with patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal:
                mock_internal.side_effect = [
                    "revenue & profit (2024)",  # query with special chars
                    "",  # empty subject_path
                    "3",  # top_n
                ]

                cli = DatusCLI(args=mock_args)
                wait_for_agent(cli)
                cli.run()

        captured = capsys.readouterr()
        stdout = captured.out

        # Command should execute
        assert "Search Metrics" in stdout, f"Should show 'Search Metrics' header, got: {stdout[:200]}"
        # Should handle gracefully
        assert "Traceback" not in stdout, "Should not have Python traceback"
        # Should show results or appropriate message
        _assert_stdout_contains_exactly_one(
            stdout,
            ("Metrics Search Results", "No metrics found"),
            "metrics result",
        )
