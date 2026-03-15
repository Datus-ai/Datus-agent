from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import yaml

from datus.cli.repl import DatusCLI
from datus.schemas.node_models import TableSchema
from tests.conftest import TEST_DATA_DIR
from tests.integration.conftest import wait_for_agent

pytestmark = pytest.mark.nightly


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
    assert "relevant tables and" in stdout and "Schema Linking Results" in stdout
    assert "Error during schema linking" not in stdout


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
    assert "Reference SQL Search Results" in stdout
    assert "Error searching reference sql:" not in stdout


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
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt, patch("subprocess.run") as mock_run:
        mock_prompt.side_effect = ["!bash ls -l", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        mock_run.assert_called_once_with("ls -l", shell=True, capture_output=True, text=True, timeout=10)


@pytest.mark.acceptance
def test_bash_command_denied(mock_args, capsys):
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt, patch("subprocess.run") as mock_run:
        mock_prompt.side_effect = ["!bash rm -rf ./temp.temp", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "Command 'rm' not in whitelist" in captured.out


@pytest.mark.acceptance
def test_databases_command(mock_args, capsys):
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [".databases", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        captured = capsys.readouterr()
        assert "Databases" in captured.out


@pytest.mark.acceptance
def test_tables_command(mock_args, capsys):
    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [".tables", EOFError]
        cli = DatusCLI(args=mock_args)
        cli.run()
        captured = capsys.readouterr()
        assert "Tables in Database" in captured.out


@pytest.mark.nightly
@pytest.mark.acceptance
def test_chat_command(mock_args, capsys, gen_sql_input: List[Dict[str, Any]]):
    """
    Tests the '/<chat>' command for multi-turn conversation and context memory.
    """
    input_data = gen_sql_input[0]["input"]
    sql_task = input_data["sql_task"]
    table_schemas = []
    if "table_schemas" in input_data:
        schemas_list = input_data.get("table_schemas", [])
        table_schemas = [TableSchema.from_dict(item) for item in schemas_list]

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            f"/{sql_task['task']}",
            ".chat_info",
            EOFError,
        ]
        with (
            patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt,
            patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
        ):
            at_data.return_value = table_schemas, [], []
            mock_internal_prompt.side_effect = ["n"]
            cli = DatusCLI(args=mock_args)

            wait_for_agent(cli)
            cli.run()

    captured = capsys.readouterr()
    stdout = captured.out

    # Check for "Tool cal" responses
    assert stdout.count("Tool call") > 0, "Should have some tool_call."

    # Check for assistant messages (thinking/response steps shown as 💬)
    assert stdout.count("💬") > 0, "Should have assistant message steps."
    assert stdout.count("Generated SQL") == 1, "Should have `Generated SQL`"

    # Check chat info
    assert stdout.count("Chat Session Info:") > 0, "Should have latest chat session info"


@pytest.mark.nightly
def test_chat_command_with_ext_knowledge(mock_args):
    """
    Tests the '/<chat>' command with ext_knowledge context.
    Verifies that the query with 'consider all knowledge' triggers knowledge search
    and generates SQL correctly.
    """
    import asyncio

    # bird california_schools q2
    question = (
        "Please list the zip code of all the charter schools "
        "in Fresno County Office of Education. consider all knowledge"
    )

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            f"/{question}",
            ".chat_info",
            EOFError,
        ]
        with (patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal_prompt,):
            mock_internal_prompt.side_effect = ["n"]
            cli = DatusCLI(args=mock_args)

            wait_for_agent(cli)
            cli.run()

    # Use internal state for assertions instead of capsys,
    # because Rich Live streaming display may not be fully captured by capsys.
    actions = cli.actions.get_actions()
    assert len(actions) > 0, "Should have action history from chat execution."

    # Find the final chat_response action which contains execution_stats
    chat_response = [a for a in actions if a.action_type == "chat_response"]
    assert len(chat_response) == 1, "Should have exactly one chat_response action."

    response_output = chat_response[0].output
    assert response_output.get("success") is True, "Chat response should be successful."

    # Check execution stats for tool usage
    exec_stats = response_output.get("execution_stats", {})
    tools_used = exec_stats.get("tools_used", [])
    assert len(tools_used) > 0, "Should have used tools during execution."

    # Verify knowledge exploration occurred — agent may call knowledge tools directly
    # or delegate to an explore sub-agent via task(type="explore")
    knowledge_tools = {"list_subject_tree", "search_knowledge", "get_knowledge", "task"}
    has_knowledge_exploration = bool(knowledge_tools & set(tools_used))
    assert has_knowledge_exploration, (
        f"Should explore knowledge via list_subject_tree, search_knowledge, "
        f"get_knowledge, or task(explore). Got: {tools_used}"
    )

    # Check that SQL was generated in the response text
    # ChatNodeResult stores SQL within the response field (no separate sql field)
    response_text = response_output.get("response", "")
    assert "SELECT" in response_text.upper(), "Should have generated SQL in the response."

    # Check that a chat node was created and has an active session
    assert cli.chat_commands.current_node is not None, "Should have an active chat node."
    session_info = asyncio.run(cli.chat_commands.current_node.get_session_info())
    assert session_info.get("session_id"), "Should have a valid session ID."
    assert session_info.get("action_count", 0) > 0, "Session should have recorded actions."


@pytest.mark.acceptance
def test_chat_info(mock_args, capsys):
    """
    Tests the '.chat_info' command for last_
    """

    with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
        mock_prompt.side_effect = [
            ".chat_info",
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
        sql_query="SELECT * FROM schools", sql_return="[{'id': 1, 'name': 'School A'}]", row_count=1
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
    assert "saved to" in stdout or "test_output" in stdout


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
        assert (
            "Reference SQL Search Results" in stdout or "No reference SQL" in stdout
        ), f"Should show results or no-results message, got: {stdout[:300]}"
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
        assert (
            "Metrics Search Results" in stdout or "No metrics found" in stdout or "Found" in stdout
        ), f"Should show results or no-results message, got: {stdout[:300]}"
