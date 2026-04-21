# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Tests for :class:`datus.cli.autocomplete.SlashCommandCompleter`.

These tests exercise the public behaviour against a real prompt-toolkit
``Document`` — no mocking needed because the completer has no outside
dependencies beyond :data:`SLASH_COMMANDS`.
"""

from __future__ import annotations

from prompt_toolkit.document import Document

from datus.cli.autocomplete import SlashCommandCompleter
from datus.cli.slash_registry import SLASH_COMMANDS, iter_visible


def _completions(text: str):
    completer = SlashCommandCompleter()
    document = Document(text, cursor_position=len(text))
    return list(completer.get_completions(document))


class TestTriggering:
    def test_bare_slash_yields_every_visible_command(self):
        results = _completions("/")
        visible_names = {spec.name for spec in iter_visible()}
        displayed = {c.display[0][1].lstrip("/") for c in results}
        assert visible_names.issubset(displayed)

    def test_no_slash_yields_nothing(self):
        assert _completions("hello world") == []

    def test_midline_slash_yields_nothing(self):
        """Slash in the middle of a chat message must not trigger the menu."""
        assert _completions("what about /help later") == []

    def test_space_after_slash_command_stops_completion(self):
        """Once args start, top-level completion yields nothing (sub-command
        completion is deferred to a future change)."""
        assert _completions("/mcp list") == []


class TestMatching:
    def test_prefix_match_filters_list(self):
        results = _completions("/sk")
        names = {c.text.rstrip() for c in results}
        assert "skill" in names
        assert "tables" not in names

    def test_alias_surfaces_in_results(self):
        results = _completions("/qu")
        tokens = {c.text.rstrip() for c in results}
        assert "quit" in tokens

    def test_case_insensitive_match(self):
        results = _completions("/HE")
        names = {c.text.rstrip() for c in results}
        assert "help" in names

    def test_display_meta_populated_from_summary(self):
        results = _completions("/help")
        matching = [c for c in results if c.text.strip() == "help"]
        assert matching, "Expected /help completion"
        meta_fragments = matching[0].display_meta
        meta_text = "".join(fragment[1] for fragment in meta_fragments)
        help_spec = next(spec for spec in SLASH_COMMANDS if spec.name == "help")
        assert meta_text == help_spec.summary

    def test_display_uses_slash_prefix(self):
        results = _completions("/he")
        display_strings = ["".join(frag[1] for frag in c.display) for c in results]
        assert any(s.startswith("/") for s in display_strings)

    def test_completion_appends_space_for_continued_typing(self):
        results = _completions("/tab")
        # Canonical name starting with "tab" is "table_schema"; completion
        # text must end with a space so the user can type args immediately.
        for c in results:
            assert c.text.endswith(" "), f"Completion text should end with space, got {c.text!r}"
