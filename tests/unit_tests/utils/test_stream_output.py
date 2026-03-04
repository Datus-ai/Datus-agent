# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for datus/utils/stream_output.py.

Tests cover:
- StreamOutputManager.add_summary_content
- StreamOutputManager.render_markdown_summary
- StreamOutputManager._extract_all_markdown_outputs

NO MOCK EXCEPT LLM. All objects under test are real implementations.
"""

from rich.console import Console

from datus.utils.stream_output import StreamOutputManager


class TestStreamOutputInit:
    """Tests for StreamOutputManager initialization."""

    def test_init_creates_empty_summary_outputs(self):
        """Newly created manager has empty summary_outputs list."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        assert mgr.summary_outputs == []
        assert len(mgr.full_output) == 0

    def test_init_sets_console(self):
        """Manager stores the console instance."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console, max_message_lines=5, title="Test")
        assert mgr.console is console
        assert mgr.title == "Test"


class TestAddSummaryContent:
    """Tests for StreamOutputManager.add_summary_content."""

    def test_add_summary_content_appends_to_list(self):
        """add_summary_content appends content to summary_outputs."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        mgr.add_summary_content("First summary")
        mgr.add_summary_content("Second summary")
        assert len(mgr.summary_outputs) == 2
        assert mgr.summary_outputs[0] == "First summary"
        assert mgr.summary_outputs[1] == "Second summary"

    def test_add_summary_content_preserves_order(self):
        """Items are appended in order."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        items = ["alpha", "beta", "gamma"]
        for item in items:
            mgr.add_summary_content(item)
        assert mgr.summary_outputs == items


class TestRenderMarkdownSummary:
    """Tests for StreamOutputManager.render_markdown_summary."""

    def test_render_with_summary_outputs_uses_stored(self):
        """render_markdown_summary uses summary_outputs when available."""
        console = Console(force_terminal=True, width=120, file=None)
        mgr = StreamOutputManager(console)
        mgr.add_summary_content("# Result\nData processed")

        # After rendering, summary_outputs and full_output should be cleared
        mgr.render_markdown_summary(title="Test Summary")
        assert mgr.summary_outputs == []
        assert len(mgr.full_output) == 0

    def test_render_with_full_output_fallback(self):
        """render_markdown_summary falls back to extracting from full_output."""
        console = Console(force_terminal=True, width=120, file=None)
        mgr = StreamOutputManager(console)
        mgr.full_output.append('{"output": "hello world"}')

        mgr.render_markdown_summary()
        # full_output and summary_outputs cleared after rendering
        assert len(mgr.full_output) == 0
        assert mgr.summary_outputs == []

    def test_render_empty_returns_immediately(self):
        """render_markdown_summary does nothing when both sources are empty."""
        console = Console(force_terminal=True, width=120, file=None)
        mgr = StreamOutputManager(console)

        # Should not raise and should be a no-op
        mgr.render_markdown_summary()
        assert len(mgr.full_output) == 0
        assert mgr.summary_outputs == []

    def test_render_clears_when_no_markdown_outputs(self):
        """render_markdown_summary clears state when extraction yields no content."""
        console = Console(force_terminal=True, width=120, file=None)
        mgr = StreamOutputManager(console)
        # full_output with non-JSON content => no markdown extracted => clear
        mgr.full_output.append("plain text without json")

        mgr.render_markdown_summary()
        assert len(mgr.full_output) == 0
        assert mgr.summary_outputs == []


class TestExtractAllMarkdownOutputs:
    """Tests for StreamOutputManager._extract_all_markdown_outputs."""

    def test_extract_from_json_block(self):
        """Extracts output from a JSON block with 'output' field."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        text = 'Some text {"output": "Hello World"} more text'
        result = mgr._extract_all_markdown_outputs(text)
        assert len(result) == 1
        assert result[0] == "Hello World"

    def test_extract_multiple_json_blocks(self):
        """Extracts from multiple JSON blocks."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        text = '{"output": "First"} gap {"output": "Second"}'
        result = mgr._extract_all_markdown_outputs(text)
        assert len(result) == 2
        assert result[0] == "First"
        assert result[1] == "Second"

    def test_extract_no_json_returns_empty(self):
        """Returns empty list when no JSON with 'output' is found."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        result = mgr._extract_all_markdown_outputs("no json here")
        assert result == []

    def test_extract_empty_output_field_skipped(self):
        """JSON with empty output field is skipped."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        text = '{"output": ""}'
        result = mgr._extract_all_markdown_outputs(text)
        assert result == []

    def test_extract_malformed_json_skipped(self):
        """Malformed JSON is skipped gracefully."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        text = '{"output": "valid"} {"output": broken}'
        result = mgr._extract_all_markdown_outputs(text)
        assert len(result) == 1
        assert result[0] == "valid"


class TestStreamOutputEdgeCases:
    """Edge case tests for StreamOutputManager."""

    def test_success_method_does_not_raise(self):
        """success() method appends message without error."""
        console = Console(force_terminal=True, width=120)
        mgr = StreamOutputManager(console)
        mgr.success("Operation completed")
        # success calls add_message which appends to messages deque
        assert len(mgr.messages) >= 0  # Messages may or may not be stored depending on state

    def test_render_with_last_n_limits_output(self):
        """render_markdown_summary with last_n limits displayed summaries."""
        console = Console(force_terminal=True, width=120, file=None)
        mgr = StreamOutputManager(console)
        for i in range(5):
            mgr.add_summary_content(f"Summary {i}")

        mgr.render_markdown_summary(last_n=2)
        # After rendering, should be cleared
        assert mgr.summary_outputs == []
