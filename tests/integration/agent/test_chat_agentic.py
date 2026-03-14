from unittest.mock import patch

import pytest

from datus.cli.repl import DatusCLI
from tests.integration.conftest import wait_for_agent


@pytest.mark.nightly
class TestChatAgentic:
    """N5: Chat agentic workflow tests."""

    def test_multi_turn_context(self, mock_args):
        """N5-02: Multi-turn dialogue preserves context."""
        question1 = "How many schools are in Fresno county?"
        question2 = "And how many in Alameda county?"

        with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
            mock_prompt.side_effect = [
                f"/{question1}",
                f"/{question2}",
                EOFError,
            ]
            with (
                patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal,
                patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
            ):
                at_data.return_value = [], [], []
                mock_internal.side_effect = ["n", "n"]
                cli = DatusCLI(args=mock_args)

                wait_for_agent(cli)
                cli.run()

        actions = cli.actions.get_actions()
        chat_responses = [a for a in actions if a.action_type == "chat_response"]
        assert len(chat_responses) >= 2, f"Should have at least 2 chat responses for 2 turns, got {len(chat_responses)}"

        # Both should be successful
        for i, resp in enumerate(chat_responses):
            assert resp.output.get("success") is True, f"Chat response {i+1} should be successful"

    def test_tool_call_combination(self, mock_args):
        """N5-03: Tool call combination -- multiple tools used during execution."""
        question = "What is the average SAT reading score for schools in Fresno county?"

        with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
            mock_prompt.side_effect = [f"/{question}", EOFError]
            with (
                patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal,
                patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
            ):
                at_data.return_value = [], [], []
                mock_internal.side_effect = ["n"]
                cli = DatusCLI(args=mock_args)

                wait_for_agent(cli)
                cli.run()

        actions = cli.actions.get_actions()
        chat_responses = [a for a in actions if a.action_type == "chat_response"]
        assert len(chat_responses) == 1, f"Should have exactly one chat_response, got {len(chat_responses)}"

        response = chat_responses[0]
        assert response.output.get("success") is True, "Chat response should be successful"

        tools_used = response.output.get("execution_stats", {}).get("tools_used", [])
        assert len(tools_used) >= 2, f"Should use multiple tools, got: {tools_used}"

    def test_streaming_response(self, mock_args):
        """N5-05: Streaming response generates action sequence."""
        question = "How many schools are there in Los Angeles county?"

        with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
            mock_prompt.side_effect = [f"/{question}", ".chat_info", EOFError]
            with (
                patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal,
                patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
            ):
                at_data.return_value = [], [], []
                mock_internal.side_effect = ["n"]
                cli = DatusCLI(args=mock_args)

                wait_for_agent(cli)
                cli.run()

        actions = cli.actions.get_actions()
        assert len(actions) > 0, "Should have action history"

        # Verify action sequence includes chat_response
        action_types = [a.action_type for a in actions]
        assert "chat_response" in action_types, f"Should have chat_response action, got types: {action_types}"

        # Verify session info
        assert cli.chat_commands.current_node is not None, "Should have an active chat node"
        import asyncio

        session_info = asyncio.run(cli.chat_commands.current_node.get_session_info())
        assert session_info.get("session_id"), "Should have a valid session ID"
        assert session_info.get("action_count", 0) > 0, "Session should have recorded actions"

    def test_single_turn_chat(self, mock_args):
        """N5-01: Basic single-turn chat generates SQL and returns result."""
        question = "List all schools in Fresno county"

        with patch("datus.cli.repl.PromptSession.prompt") as mock_prompt:
            mock_prompt.side_effect = [f"/{question}", EOFError]
            with (
                patch("datus.cli.repl.DatusCLI.prompt_input") as mock_internal,
                patch("datus.cli.repl.AtReferenceCompleter.parse_at_context") as at_data,
            ):
                at_data.return_value = [], [], []
                mock_internal.side_effect = ["n"]
                cli = DatusCLI(args=mock_args)

                wait_for_agent(cli)
                cli.run()

        actions = cli.actions.get_actions()
        chat_responses = [a for a in actions if a.action_type == "chat_response"]
        assert len(chat_responses) == 1, f"Should have exactly one chat_response, got {len(chat_responses)}"

        response = chat_responses[0]
        assert response.output.get("success") is True, "Chat response should be successful"

        # Chat node returns SQL in the response text (markdown), not as a top-level key.
        # Verify the response contains SQL-like content (SELECT statement).
        response_text = response.output.get("response", "")
        assert (
            "select" in response_text.lower()
        ), f"Response should contain generated SQL, got response: {response_text[:500]}"
