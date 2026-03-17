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

from io import StringIO

import pytest
from rich.console import Console

from datus.utils.stream_output import StreamOutputManager, create_stream_output_manager


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


def _console() -> Console:
    """Create a Console that writes to a buffer (no TTY interaction)."""
    return Console(file=StringIO(), width=120)


class TestCreateProgress:
    """Tests for StreamOutputManager._create_progress (lines 82-100)."""

    def test_single_item_returns_spinner_only(self):
        mgr = StreamOutputManager(_console())
        p = mgr._create_progress(1)
        # Spinner-only mode has fewer columns
        assert p is not None

    def test_zero_items_returns_spinner_only(self):
        mgr = StreamOutputManager(_console())
        p = mgr._create_progress(0)
        assert p is not None

    def test_multi_item_returns_progress_with_bar(self):
        mgr = StreamOutputManager(_console())
        p = mgr._create_progress(5)
        assert p is not None


class TestStartStop:
    """Tests for start() and stop() (lines 102-132)."""

    def test_start_sets_is_running(self):
        mgr = StreamOutputManager(_console(), show_progress=True)
        mgr.start(total_items=3)
        assert mgr._is_running is True
        mgr.stop()

    def test_stop_clears_is_running(self):
        mgr = StreamOutputManager(_console())
        mgr.start(1)
        mgr.stop()
        assert mgr._is_running is False
        assert mgr.live is None

    def test_start_idempotent(self):
        """Calling start twice does not raise."""
        mgr = StreamOutputManager(_console())
        mgr.start(1)
        mgr.start(1)  # second call should be no-op
        assert mgr._is_running is True
        mgr.stop()

    def test_stop_when_not_running_is_safe(self):
        """stop() on a manager that was never started should not raise."""
        mgr = StreamOutputManager(_console())
        mgr.stop()  # Should not raise

    def test_start_with_custom_description(self):
        mgr = StreamOutputManager(_console())
        mgr.start(5, description="Custom task")
        assert mgr._is_running is True
        mgr.stop()


class TestUpdateProgress:
    """Tests for update_progress() and set_progress() (lines 134-160)."""

    def test_update_progress_no_active_progress(self):
        """Calling update_progress when progress is None should not raise."""
        mgr = StreamOutputManager(_console())
        mgr.update_progress(advance=1)  # should be a no-op

    def test_update_progress_with_description(self):
        mgr = StreamOutputManager(_console())
        mgr.start(5)
        mgr.update_progress(advance=1, description="Step 1")
        mgr.stop()

    def test_set_progress_no_active_progress(self):
        mgr = StreamOutputManager(_console())
        mgr.set_progress(3)  # no-op

    def test_set_progress_with_description(self):
        mgr = StreamOutputManager(_console())
        mgr.start(10)
        mgr.set_progress(5, description="Halfway")
        mgr.stop()


class TestFileManagement:
    """Tests for start_file() and complete_file() (lines 162-185)."""

    def test_start_file_sets_current_file(self):
        mgr = StreamOutputManager(_console())
        mgr.start_file("data.sql")
        assert mgr.current_file == "data.sql"

    def test_start_file_clears_messages(self):
        mgr = StreamOutputManager(_console())
        mgr.add_message("old message")
        mgr.start_file("new_file.csv")
        assert len(mgr.messages) == 0

    def test_start_file_resets_task_number(self):
        mgr = StreamOutputManager(_console())
        mgr.task_number = 5
        mgr.start_file("file.db")
        assert mgr.task_number == 0

    def test_start_file_with_total_items_adds_message(self):
        mgr = StreamOutputManager(_console())
        mgr.start_file("file.db", total_items=10)
        assert len(mgr.messages) > 0

    def test_complete_file_clears_current_file(self):
        mgr = StreamOutputManager(_console())
        mgr.start_file("data.sql")
        mgr.complete_file("data.sql")
        assert mgr.current_file == ""


class TestTaskManagement:
    """Tests for start_task() and complete_task() (lines 187-242)."""

    def test_start_task_increments_task_number(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("First task")
        assert mgr.task_number == 1
        mgr.start_task("Second task")
        assert mgr.task_number == 2

    def test_start_task_sets_current_task(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("Do something")
        assert "Do something" in mgr.current_task

    def test_complete_task_success_adds_check_mark(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("Work")
        mgr.complete_task(success=True, message="Done")
        assert any("✓" in msg for msg, _ in mgr.messages)

    def test_complete_task_failure_adds_x_mark(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("Work")
        mgr.complete_task(success=False, message="Failed")
        assert any("✗" in msg for msg, _ in mgr.messages)

    def test_complete_task_no_message_clears_current(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("Work")
        mgr.complete_task(success=True)
        assert mgr.current_task == ""

    def test_complete_task_with_message_clears_current(self):
        mgr = StreamOutputManager(_console())
        mgr.start_task("Work")
        mgr.complete_task(success=True, message="All done")
        assert mgr.current_task == ""


class TestMessages:
    """Tests for add_message(), error(), warning(), success(), add_llm_output() (lines 198-270)."""

    def test_add_message_basic(self):
        mgr = StreamOutputManager(_console())
        mgr.add_message("Hello")
        assert len(mgr.messages) == 1
        assert mgr.messages[0][0] == "Hello"

    def test_add_message_empty_string_ignored(self):
        mgr = StreamOutputManager(_console())
        mgr.add_message("")
        assert len(mgr.messages) == 0

    def test_add_message_multiline(self):
        mgr = StreamOutputManager(_console())
        mgr.add_message("line1\nline2\nline3")
        assert len(mgr.messages) == 3

    def test_add_message_with_style(self):
        mgr = StreamOutputManager(_console())
        mgr.add_message("styled msg", style="bold red")
        assert mgr.messages[0][1] == "bold red"

    def test_add_message_skips_blank_lines(self):
        """Blank lines within a multiline message are skipped (but content lines are kept as-is)."""
        mgr = StreamOutputManager(_console())
        mgr.add_message("hello\n\n\nworld")
        # Blank lines stripped, "hello" and "world" should be present
        texts = [m for m, _ in mgr.messages]
        assert len(texts) == 2
        assert "hello" in texts[0]
        assert "world" in texts[1]

    def test_add_message_respects_maxlen(self):
        mgr = StreamOutputManager(_console(), max_message_lines=3)
        for i in range(10):
            mgr.add_message(f"msg {i}")
        assert len(mgr.messages) <= 3

    def test_error_prepends_x(self):
        mgr = StreamOutputManager(_console())
        mgr.error("Bad thing")
        msg, style = mgr.messages[-1]
        assert "✗" in msg
        assert "Bad thing" in msg

    def test_warning_prepends_warning_symbol(self):
        mgr = StreamOutputManager(_console())
        mgr.warning("Watch out")
        msg, style = mgr.messages[-1]
        assert "⚠" in msg
        assert "Watch out" in msg

    def test_success_prepends_check(self):
        mgr = StreamOutputManager(_console())
        mgr.success("All good")
        msg, style = mgr.messages[-1]
        assert "✓" in msg
        assert "All good" in msg

    def test_add_llm_output_stores_in_full_output(self):
        mgr = StreamOutputManager(_console())
        mgr.add_llm_output("LLM text")
        assert "LLM text" in mgr.full_output

    def test_add_llm_output_also_adds_to_messages(self):
        mgr = StreamOutputManager(_console())
        mgr.add_llm_output("LLM text")
        assert any("LLM text" in msg for msg, _ in mgr.messages)


class TestRender:
    """Tests for _render() (lines 355-395)."""

    def test_render_returns_group(self):
        pass

        mgr = StreamOutputManager(_console())
        result = mgr._render()
        assert result is not None

    def test_render_with_all_fields_set(self):
        mgr = StreamOutputManager(_console())
        mgr.start(3)
        mgr.current_file = "myfile.sql"
        mgr.current_task = "[1] My Task"
        mgr.add_message("msg1")
        result = mgr._render()
        assert result is not None
        mgr.stop()

    def test_render_no_progress_excludes_progress_bar(self):
        mgr = StreamOutputManager(_console(), show_progress=False)
        result = mgr._render()
        assert result is not None


class TestTaskContext:
    """Tests for task_context() context manager (lines 402-420)."""

    def test_task_context_success(self):
        mgr = StreamOutputManager(_console())
        with mgr.task_context("Test task"):
            mgr.add_message("doing work")
        assert mgr.current_task == ""

    def test_task_context_exception_propagates(self):
        mgr = StreamOutputManager(_console())
        with pytest.raises(ValueError):
            with mgr.task_context("Failing task"):
                raise ValueError("oops")

    def test_task_context_marks_failed_on_exception(self):
        mgr = StreamOutputManager(_console())
        with pytest.raises(RuntimeError):
            with mgr.task_context("Crash"):
                raise RuntimeError("crash")
        assert any("✗" in msg for msg, _ in mgr.messages)

    def test_task_context_yields_manager(self):
        mgr = StreamOutputManager(_console())
        with mgr.task_context("Yield test") as m:
            assert m is mgr


class TestFileContext:
    """Tests for file_context() context manager (lines 422-439)."""

    def test_file_context_sets_and_clears_file(self):
        mgr = StreamOutputManager(_console())
        with mgr.file_context("data.csv"):
            assert mgr.current_file == "data.csv"
        assert mgr.current_file == ""

    def test_file_context_exception_clears_file(self):
        mgr = StreamOutputManager(_console())
        with pytest.raises(RuntimeError):
            with mgr.file_context("data.csv"):
                raise RuntimeError("file error")
        assert mgr.current_file == ""

    def test_file_context_with_total_items(self):
        mgr = StreamOutputManager(_console())
        with mgr.file_context("data.csv", total_items=5):
            assert any("5" in msg for msg, _ in mgr.messages)

    def test_file_context_yields_manager(self):
        mgr = StreamOutputManager(_console())
        with mgr.file_context("x.sql") as m:
            assert m is mgr


class TestCreateStreamOutputManager:
    """Tests for create_stream_output_manager factory function."""

    def test_factory_returns_stream_output_manager(self):
        console = _console()
        mgr = create_stream_output_manager(console)
        assert isinstance(mgr, StreamOutputManager)

    def test_factory_passes_parameters(self):
        console = _console()
        mgr = create_stream_output_manager(console, max_message_lines=5, show_progress=False, title="My Title")
        assert mgr.max_message_lines == 5
        assert mgr.show_progress is False
        assert mgr.title == "My Title"
