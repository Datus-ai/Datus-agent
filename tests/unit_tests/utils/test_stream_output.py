# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for datus.utils.stream_output — StreamOutputManager rendering."""

from rich.console import Console

from datus.utils.stream_output import StreamOutputManager


def _make_mgr(**kwargs) -> StreamOutputManager:
    """Create a StreamOutputManager with a quiet console for testing."""
    console = Console(quiet=True)
    return StreamOutputManager(console=console, **kwargs)


class TestStreamOutputManagerInit:
    """Tests for StreamOutputManager initialization."""

    def test_default_initialization(self):
        """StreamOutputManager initializes with empty state."""
        mgr = _make_mgr()
        assert list(mgr.full_output) == []
        assert mgr.summary_outputs == []

    def test_title_parameter(self):
        """StreamOutputManager accepts title parameter."""
        mgr = _make_mgr(title="Test Task")
        assert mgr.title == "Test Task"


class TestAddSummaryContent:
    """Tests for add_summary_content method."""

    def test_add_single_summary(self):
        """Single summary content is stored."""
        mgr = _make_mgr()
        mgr.add_summary_content("Test summary")
        assert len(mgr.summary_outputs) == 1
        assert mgr.summary_outputs[0] == "Test summary"

    def test_add_multiple_summaries(self):
        """Multiple summary contents are accumulated."""
        mgr = _make_mgr()
        mgr.add_summary_content("First")
        mgr.add_summary_content("Second")
        assert len(mgr.summary_outputs) == 2
        assert mgr.summary_outputs == ["First", "Second"]


class TestRenderMarkdownSummary:
    """Tests for render_markdown_summary method."""

    def test_empty_state_is_noop(self):
        """Rendering with no content does nothing."""
        mgr = _make_mgr()
        mgr.render_markdown_summary()  # should not raise
        assert mgr.summary_outputs == []
        assert list(mgr.full_output) == []

    def test_summary_outputs_cleared_after_render(self):
        """Summary outputs are cleared after rendering."""
        mgr = _make_mgr()
        mgr.add_summary_content("Some content")
        mgr.render_markdown_summary()
        assert mgr.summary_outputs == []
        assert list(mgr.full_output) == []

    def test_full_output_fallback_cleared_after_render(self):
        """Full output is cleared after rendering even without summary_outputs."""
        mgr = _make_mgr()
        mgr.full_output.append('{"output": "test result"}')
        mgr.render_markdown_summary()
        assert list(mgr.full_output) == []

    def test_empty_markdown_cleared(self):
        """Empty extracted markdown results in cleared state."""
        mgr = _make_mgr()
        mgr.full_output.append("no json here")
        mgr.render_markdown_summary()
        assert list(mgr.full_output) == []
        assert mgr.summary_outputs == []
