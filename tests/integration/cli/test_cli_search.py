"""
CLI Search Command Tests (N12).

Tests the search commands (!sl, !sq, !sm, !sd) in the CLI REPL.
Covers edge cases, filtering, and error handling scenarios.
"""

from unittest.mock import patch

import pytest

from datus.cli.repl import DatusCLI
from tests.integration.cli.conftest import wait_for_agent


@pytest.mark.nightly
class TestCLISearch:
    """N12: CLI search command tests."""

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
