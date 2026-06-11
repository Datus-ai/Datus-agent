# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/skill_command_utils.py.

``render_skill_prompt`` fills a skill-shortcut prompt template's
``{user_context}`` placeholder with an optional free-text description, so the
three slash shortcuts (``/init``, ``/session-summarize``, ``/memory-organize``)
can forward extra hints verbatim into the chat message.
"""

import pytest

from datus.cli.skill_command_utils import render_skill_prompt

_TEMPLATE = "Do the thing.{user_context}"


class TestRenderSkillPromptBlankArgs:
    """Blank / whitespace / None collapse the placeholder to nothing."""

    @pytest.mark.parametrize("args", ["", "   ", "\t\n ", None])
    def test_blank_args_drop_the_block(self, args):
        out = render_skill_prompt(_TEMPLATE, args)
        assert out == "Do the thing."
        assert "{user_context}" not in out
        assert "Additional context" not in out


class TestRenderSkillPromptWithArgs:
    """Non-blank args append an explicit, clearly-led context block."""

    def test_appends_verbatim_description(self):
        out = render_skill_prompt(_TEMPLATE, "focus on the orders domain")
        assert out.startswith("Do the thing.")
        assert "Additional context from the user" in out
        assert "focus on the orders domain" in out
        assert "{user_context}" not in out

    def test_strips_surrounding_whitespace_only(self):
        # Leading/trailing whitespace is trimmed; inner text is preserved verbatim.
        out = render_skill_prompt(_TEMPLATE, "  keep  inner   spaces  ")
        assert "keep  inner   spaces" in out
        assert "  keep  inner   spaces  " not in out

    def test_block_is_separated_by_blank_line(self):
        out = render_skill_prompt(_TEMPLATE, "note")
        # The appended block starts on its own paragraph.
        assert "\n\n" in out

    def test_multiline_description_preserved(self):
        out = render_skill_prompt(_TEMPLATE, "line one\nline two")
        assert "line one\nline two" in out
