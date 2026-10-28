# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Hand-off of AI review verdicts from the permission gate to tool actions."""

import pytest

from datus.schemas.action_history import ActionHistory, ActionRole, ActionStatus
from datus.tools.permission import review_registry as rr


@pytest.fixture(autouse=True)
def _clean_registry():
    rr.clear()
    yield
    rr.clear()


def verdict(outcome="auto_allowed", **overrides):
    payload = {
        "outcome": outcome,
        "decision": "allow",
        "risk_level": "low",
        "user_authorization": "medium",
        "confidence": 0.9,
        "rationale": "scoped to the workspace",
    }
    payload.update(overrides)
    return payload


def tool_action(action_id, output=None, status=ActionStatus.SUCCESS, role=ActionRole.TOOL):
    return ActionHistory(
        action_id=action_id,
        role=role,
        messages="Tool call: bash",
        action_type="bash",
        input={"function_name": "bash"},
        output=output,
        status=status,
    )


class TestRecordAndTake:
    def test_round_trip(self):
        rr.record("c1", verdict())
        assert rr.take("c1")["rationale"] == "scoped to the workspace"

    def test_take_consumes(self):
        rr.record("c1", verdict())
        rr.take("c1")
        assert rr.take("c1") is None

    def test_unknown_call_id(self):
        assert rr.take("nope") is None

    @pytest.mark.parametrize("call_id,payload", [(None, {"a": 1}), ("c1", None), ("", {"a": 1}), ("c1", {})])
    def test_missing_inputs_are_noops(self, call_id, payload):
        rr.record(call_id, payload)
        assert rr.take(call_id or "c1") is None

    def test_rerecord_overwrites(self):
        """The gate refines outcome after the user answers a prompt."""
        rr.record("c1", verdict("auto_allowed"))
        rr.record("c1", verdict("user_approved"))
        assert rr.take("c1")["outcome"] == "user_approved"

    def test_bounded_eviction_drops_oldest(self):
        for i in range(rr._MAX_PENDING + 5):
            rr.record(f"c{i}", verdict())
        assert rr.take("c0") is None
        assert rr.take("c4") is None
        assert rr.take(f"c{rr._MAX_PENDING + 4}") == verdict()


class TestCallIdOf:
    @pytest.mark.parametrize(
        "action_id,expected",
        [
            ("complete_call_123", "call_123"),
            ("call_123", "call_123"),  # in-flight frame uses the bare id
            ("complete_complete_x", "complete_x"),  # strip only one prefix
            (None, None),
            ("", None),
        ],
    )
    def test_recovers_call_id(self, action_id, expected):
        assert rr.call_id_of(action_id) == expected


class TestStamp:
    def test_adds_key_to_dict_output(self):
        rr.record("c1", verdict())
        out = rr.stamp({"success": True}, "c1")
        assert out["success"] is True
        assert out[rr.PERMISSION_REVIEW_OUTPUT_KEY]["decision"] == "allow"

    def test_no_review_leaves_output_untouched(self):
        original = {"success": True}
        assert rr.stamp(original, "c1") is original
        assert rr.PERMISSION_REVIEW_OUTPUT_KEY not in original

    def test_non_dict_output_is_wrapped_not_dropped(self):
        rr.record("c1", verdict())
        out = rr.stamp("raw text", "c1")
        assert out["result"] == "raw text"
        assert out[rr.PERMISSION_REVIEW_OUTPUT_KEY]["decision"] == "allow"


class TestEnrichAction:
    def test_stamps_completed_tool_action(self):
        rr.record("c1", verdict())
        action = tool_action("complete_c1", output={"success": True})
        rr.enrich_action(action)
        assert action.output[rr.PERMISSION_REVIEW_OUTPUT_KEY]["risk_level"] == "low"

    def test_failed_tool_action_is_also_stamped(self):
        """A reviewed command that then failed still shows what authorised it."""
        rr.record("c1", verdict())
        action = tool_action("complete_c1", output={"success": False}, status=ActionStatus.FAILED)
        rr.enrich_action(action)
        assert rr.PERMISSION_REVIEW_OUTPUT_KEY in action.output

    def test_processing_frame_does_not_consume_the_entry(self):
        """The in-flight frame is emitted first and shares the call id.

        Consuming there would leave the completed action — the one that
        persists and gets replayed on ctrl+o — without the verdict.
        """
        rr.record("c1", verdict())
        rr.enrich_action(tool_action("c1", status=ActionStatus.PROCESSING))
        completed = tool_action("complete_c1", output={"success": True})
        rr.enrich_action(completed)
        assert rr.PERMISSION_REVIEW_OUTPUT_KEY in completed.output

    def test_non_tool_action_ignored(self):
        rr.record("c1", verdict())
        action = tool_action("complete_c1", output={"x": 1}, role=ActionRole.ASSISTANT)
        rr.enrich_action(action)
        assert rr.PERMISSION_REVIEW_OUTPUT_KEY not in action.output

    def test_unreviewed_action_keeps_none_output(self):
        """A miss must not rewrite ``output``; ``None`` stays ``None``."""
        action = tool_action("complete_c1", output=None)
        rr.enrich_action(action)
        assert action.output is None

    def test_already_stamped_action_is_left_alone(self):
        rr.record("c1", verdict("user_approved"))
        action = tool_action("complete_c1", output={rr.PERMISSION_REVIEW_OUTPUT_KEY: {"outcome": "original"}})
        rr.enrich_action(action)
        assert action.output[rr.PERMISSION_REVIEW_OUTPUT_KEY]["outcome"] == "original"


class TestActionHistoryManagerIntegration:
    def test_add_action_applies_the_registered_enricher(self):
        """Importing the permission package wires the enricher for every model."""
        import datus.tools.permission  # noqa: F401 - registration side effect
        from datus.schemas.action_history import ActionHistoryManager

        rr.record("c1", verdict())
        manager = ActionHistoryManager()
        action = tool_action("complete_c1", output={"success": True})
        manager.add_action(action)

        assert action.output[rr.PERMISSION_REVIEW_OUTPUT_KEY]["confidence"] == 0.9

    def test_a_failing_enricher_never_drops_the_action(self):
        from datus.schemas.action_history import ActionHistoryManager, register_action_enricher

        def boom(_action):
            raise RuntimeError("enricher exploded")

        register_action_enricher(boom)
        try:
            manager = ActionHistoryManager()
            manager.add_action(tool_action("complete_c9", output={"success": True}))
            assert len(manager.actions) == 1
        finally:
            from datus.schemas import action_history

            action_history._ACTION_ENRICHERS.remove(boom)
