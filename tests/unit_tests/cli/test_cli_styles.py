# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from io import StringIO

from rich.console import Console

from datus.cli.cli_styles import (
    CLR_CURRENT,
    CLR_CURSOR,
    CLR_ERROR,
    CLR_SUCCESS,
    CLR_WARNING,
    CODE_THEME,
    SYM_ARROW,
    SYM_BULLET,
    SYM_CHECK,
    SYM_CROSS,
    TABLE_HEADER_STYLE,
    print_empty_set,
    print_error,
    print_info,
    print_status,
    print_success,
    print_usage,
    print_warning,
)


def _capture(fn, *args, **kwargs) -> str:
    buf = StringIO()
    console = Console(file=buf, no_color=True, force_terminal=True, highlight=False)
    fn(console, *args, **kwargs)
    import re

    return re.sub(r"\x1b\[[^m]*m", "", buf.getvalue()).strip()


class TestConstants:
    def test_symbols(self):
        assert SYM_CHECK == "\u2713"
        assert SYM_CROSS == "\u2717"
        assert SYM_ARROW == "\u2192"
        assert SYM_BULLET == "\u2022"

    def test_colors_are_not_bold(self):
        for name, val in [
            ("CLR_ERROR", CLR_ERROR),
            ("CLR_SUCCESS", CLR_SUCCESS),
            ("CLR_WARNING", CLR_WARNING),
        ]:
            assert "bold" not in val, f"{name} should not contain bold"

    def test_table_header_style(self):
        assert TABLE_HEADER_STYLE == "green"

    def test_code_theme(self):
        assert CODE_THEME == "monokai"

    def test_prompt_toolkit_colors(self):
        assert CLR_CURSOR == "ansicyan"
        assert CLR_CURRENT == "ansigreen"


class TestPrintError:
    def test_with_prefix(self):
        out = _capture(print_error, "Database name is required")
        assert out == "Error: Database name is required"

    def test_without_prefix(self):
        out = _capture(print_error, "Something failed", prefix=False)
        assert out == "Something failed"
        assert "Error:" not in out


class TestPrintSuccess:
    def test_plain(self):
        out = _capture(print_success, "Database switched to: mydb")
        assert out == "Database switched to: mydb"
        assert SYM_CHECK not in out

    def test_with_symbol(self):
        out = _capture(print_success, "Session compacted!", symbol=True)
        assert out == f"{SYM_CHECK} Session compacted!"


class TestPrintWarning:
    def test_basic(self):
        out = _capture(print_warning, "Empty set.")
        assert out == "Empty set."


class TestPrintInfo:
    def test_basic(self):
        out = _capture(print_info, "Creating session...")
        assert "Creating session..." in out


class TestPrintStatus:
    def test_ok(self):
        out = _capture(print_status, "Server reachable", ok=True)
        assert out == f"{SYM_CHECK} Server reachable"

    def test_fail(self):
        out = _capture(print_status, "Server unreachable", ok=False)
        assert out == f"{SYM_CROSS} Server unreachable"


class TestPrintUsage:
    def test_basic(self):
        out = _capture(print_usage, "/subagent [add|list|remove]")
        assert out == "Usage: /subagent [add|list|remove]"


class TestPrintEmptySet:
    def test_default_message(self):
        out = _capture(print_empty_set)
        assert out == "Empty set."

    def test_custom_message(self):
        out = _capture(print_empty_set, "No MCP servers found.")
        assert out == "No MCP servers found."


class TestColorMarkup:
    """Verify helpers embed the correct Rich markup tags (with color enabled)."""

    def _capture_with_color(self, fn, *args, **kwargs) -> str:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, color_system="truecolor", no_color=False)
        fn(console, *args, **kwargs)
        return buf.getvalue()

    def test_error_uses_red(self):
        raw = self._capture_with_color(print_error, "fail")
        assert "\x1b[" in raw  # ANSI escape present

    def test_success_uses_green(self):
        raw = self._capture_with_color(print_success, "ok")
        assert "\x1b[" in raw

    def test_status_ok_uses_green(self):
        raw = self._capture_with_color(print_status, "up", ok=True)
        assert SYM_CHECK in raw

    def test_status_fail_uses_red(self):
        raw = self._capture_with_color(print_status, "down", ok=False)
        assert SYM_CROSS in raw


def _render_renderable(renderable) -> str:
    import re

    buf = StringIO()
    Console(file=buf, no_color=True, force_terminal=True, width=100, highlight=False).print(renderable)
    return re.sub(r"\x1b\[[^m]*m", "", buf.getvalue())


class TestRenderCompactSummaryPanel:
    def test_contains_summary_token_check_history(self):
        from datus.cli.cli_styles import render_compact_summary_panel

        out = _render_renderable(render_compact_summary_panel("Did **work** on X.", 42, "/tmp/h.jsonl"))
        assert "Context compacted" in out
        assert SYM_CHECK in out
        assert "work" in out  # markdown body rendered
        assert "42 tokens" in out
        assert "/tmp/h.jsonl" in out

    def test_returns_panel(self):
        from rich.panel import Panel

        from datus.cli.cli_styles import render_compact_summary_panel

        assert isinstance(render_compact_summary_panel("s"), Panel)

    def test_no_history_no_token_footer(self):
        from datus.cli.cli_styles import render_compact_summary_panel

        out = _render_renderable(render_compact_summary_panel("Just summary.", 0, ""))
        assert "Just summary." in out
        assert "History:" not in out
        assert "tokens" not in out

    def test_empty_summary_uses_placeholder(self):
        from datus.cli.cli_styles import render_compact_summary_panel

        out = _render_renderable(render_compact_summary_panel("", 0, ""))
        assert "empty summary" in out


class TestRenderCompactProgressLine:
    def test_returns_text_with_message(self):
        from rich.text import Text

        from datus.cli.cli_styles import render_compact_progress_line

        line = render_compact_progress_line()
        assert isinstance(line, Text)
        assert "Compacting context" in line.plain
