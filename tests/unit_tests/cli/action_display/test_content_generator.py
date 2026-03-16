# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus/cli/action_display/content_generator.py.

Most functionality is already tested via test_action_history_display.py;
this file adds targeted tests for the module's standalone functions.
"""

import pytest

from datus.cli.action_display.renderers import (
    ActionContentGenerator,
    BaseActionContentGenerator,
    _get_assistant_content,
    _truncate_middle,
)
from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus


@pytest.mark.ci
class TestTruncateMiddle:
    def test_short_unchanged(self):
        assert _truncate_middle("hello", 120) == "hello"

    def test_long_truncated(self):
        text = "A" * 200
        result = _truncate_middle(text, 120)
        assert len(result) <= 120
        assert " ... " in result


@pytest.mark.ci
class TestGetAssistantContent:
    def test_prefers_raw_output(self):
        import uuid

        action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.ASSISTANT,
            messages="fallback",
            action_type="test",
            status=ActionStatus.SUCCESS,
            output={"raw_output": "preferred"},
        )
        assert _get_assistant_content(action) == "preferred"

    def test_falls_back_to_messages(self):
        import uuid

        action = ActionHistory(
            action_id=str(uuid.uuid4()),
            role=ActionRole.ASSISTANT,
            messages="fallback msg",
            action_type="test",
            status=ActionStatus.SUCCESS,
        )
        assert _get_assistant_content(action) == "fallback msg"


@pytest.mark.ci
class TestContentGeneratorInit:
    def test_base_has_role_colors(self):
        gen = BaseActionContentGenerator()
        assert ActionRole.TOOL in gen.role_colors

    def test_truncation_flag(self):
        gen = ActionContentGenerator(enable_truncation=False)
        assert gen.enable_truncation is False
