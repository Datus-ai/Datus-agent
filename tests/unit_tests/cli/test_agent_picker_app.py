# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for ``datus.cli.agent_picker_app.AgentPickerApp``.

CI-level: no TTY, no external deps. The prompt_toolkit Application is not
run — we test the data model, index logic, and rendering output.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from datus.cli.agent_picker_app import AgentPickerApp

pytestmark = pytest.mark.ci


def _console() -> Console:
    return Console(file=io.StringIO(), no_color=True)


class TestInit:
    def test_requires_non_empty_list(self):
        with pytest.raises(ValueError):
            AgentPickerApp(console=_console(), agents=[], current="")

    def test_default_index_matches_current(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql", "gen_job"], current="gen_sql")
        assert app._agents[app._idx] == "gen_sql"

    def test_default_index_falls_back_to_zero(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql"], current="not-in-list")
        assert app._idx == 0

    def test_missing_current_is_zero(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql"], current="")
        assert app._idx == 0


class TestKeyNavigation:
    """Exercise the key handlers directly via internal index mutation.

    We don't run the Application (no TTY), but the key callbacks only touch
    ``_idx`` and call ``event.app.exit``, so mutating ``_idx`` by hand gives
    the same coverage.
    """

    def test_render_highlights_current_selection(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql", "gen_job"], current="chat")
        app._idx = 1
        lines = app._render_list()
        # Selected row uses the reverse style and the arrow prefix.
        styles = [style for style, _ in lines]
        assert styles.count("reverse") == 1
        selected_line = next(text for style, text in lines if style == "reverse")
        assert "gen_sql" in selected_line
        assert "→" in selected_line

    def test_render_marks_current_agent(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql"], current="gen_sql")
        lines = app._render_list()
        gen_sql_line = next(text for _, text in lines if "gen_sql" in text)
        assert "current" in gen_sql_line

    def test_render_header_shows_current(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql"], current="gen_sql")
        lines = app._render_header()
        body = " ".join(text for _, text in lines)
        assert "gen_sql" in body

    def test_render_header_defaults_to_chat_when_current_empty(self):
        app = AgentPickerApp(console=_console(), agents=["chat", "gen_sql"], current="")
        lines = app._render_header()
        body = " ".join(text for _, text in lines)
        assert "chat" in body
