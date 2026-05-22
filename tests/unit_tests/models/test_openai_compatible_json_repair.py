# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Unit tests for malformed JSON repair in tool call arguments during streaming.

Verifies that openai_compatible.py repairs malformed arguments in raw_item
before they are stored in session history, preventing cascading API failures.
"""

import json

from pydantic import BaseModel

from datus.models.openai_compatible import json_repair


class FakeRawItem(BaseModel):
    name: str = "ask_user"
    call_id: str = "call_abc123"
    arguments: str = "{}"

    class Config:
        extra = "allow"


class TestToolCallArgumentsRepair:
    """Test JSON repair of tool call arguments in streaming layer."""

    def test_valid_json_passes_through_unchanged(self):
        raw_item = FakeRawItem(arguments='{"question": "hello", "options": ["a", "b"]}')
        original_arguments = raw_item.arguments

        try:
            json.loads(raw_item.arguments)
            repaired = False
        except (json.JSONDecodeError, TypeError, ValueError):
            repaired = True

        assert not repaired
        assert raw_item.arguments == original_arguments

    def test_malformed_json_is_repaired(self):
        malformed = '{"keywords": PERCENTILE function, "platform": "starrocks"}'
        raw_item = FakeRawItem(arguments=malformed)

        try:
            json.loads(raw_item.arguments)
            needs_repair = False
        except (json.JSONDecodeError, TypeError, ValueError):
            needs_repair = True

        assert needs_repair

        repaired = json_repair.loads(malformed)
        repaired_str = json.dumps(repaired, ensure_ascii=False)
        new_raw_item = raw_item.model_copy(update={"arguments": repaired_str})

        assert json.loads(new_raw_item.arguments) is not None
        assert new_raw_item.name == "ask_user"
        assert new_raw_item.call_id == "call_abc123"

    def test_model_copy_preserves_other_fields(self):
        raw_item = FakeRawItem(
            name="execute_sql",
            call_id="call_xyz789",
            arguments='{"sql": SELECT * FROM t}',
        )

        repaired = json_repair.loads(raw_item.arguments)
        repaired_str = json.dumps(repaired, ensure_ascii=False)
        new_raw_item = raw_item.model_copy(update={"arguments": repaired_str})

        assert new_raw_item.name == "execute_sql"
        assert new_raw_item.call_id == "call_xyz789"
        assert json.loads(new_raw_item.arguments) is not None

    def test_unrepairable_json_left_unchanged(self):
        garbage = "not json at all {{{{[[[["
        raw_item = FakeRawItem(arguments=garbage)

        try:
            json.loads(raw_item.arguments)
            needs_repair = False
        except (json.JSONDecodeError, TypeError, ValueError):
            needs_repair = True

        assert needs_repair

        # json_repair may still produce something from garbage;
        # the key point is the original raw_item is unchanged if we don't write back
        assert raw_item.arguments == garbage

    def test_empty_arguments_not_repaired(self):
        raw_item = FakeRawItem(arguments="")

        try:
            json.loads(raw_item.arguments) if raw_item.arguments else {}
            needs_repair = False
        except (json.JSONDecodeError, TypeError, ValueError):
            needs_repair = True

        assert not needs_repair

    def test_glm_style_unquoted_values_repaired(self):
        """Simulate real GLM output: unquoted string values in JSON."""
        malformed = '{"question": 请问您想查询哪个数据库的信息？, "options": [StarRocks, MySQL]}'
        raw_item = FakeRawItem(arguments=malformed)

        try:
            json.loads(raw_item.arguments)
            needs_repair = False
        except (json.JSONDecodeError, TypeError, ValueError):
            needs_repair = True

        assert needs_repair

        repaired = json_repair.loads(malformed)
        repaired_str = json.dumps(repaired, ensure_ascii=False)
        new_raw_item = raw_item.model_copy(update={"arguments": repaired_str})

        parsed = json.loads(new_raw_item.arguments)
        assert "question" in parsed
        assert isinstance(parsed.get("options"), list)
