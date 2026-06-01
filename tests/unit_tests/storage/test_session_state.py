# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/storage/session_state.py — CI tier."""

import json

import pytest

from datus.storage.session_state import ContextState, PlanModeState


class TestPlanModeStateRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "state" / "s1.json"
        state = PlanModeState(
            plan_mode_active=True,
            plan_file_path="./.datus/plans/abc12345.md",
            workflow_prompt_sent=True,
        )
        state.save(path)
        assert path.exists()

        loaded = PlanModeState.load(path)
        assert loaded.plan_mode_active is True
        assert loaded.plan_file_path == "./.datus/plans/abc12345.md"
        assert loaded.workflow_prompt_sent is True

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = PlanModeState.load(tmp_path / "absent.json")
        assert loaded.plan_mode_active is False
        assert loaded.plan_file_path is None
        assert loaded.workflow_prompt_sent is False

    def test_load_corrupted_json_falls_back_to_default(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        loaded = PlanModeState.load(path)
        assert loaded == PlanModeState()

    def test_save_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "state.json"
        PlanModeState(plan_mode_active=True).save(path)
        assert path.exists()
        # On-disk JSON uses the nested ``plan_mode`` layout so future readers
        # can tell apart a missing section from a falsy-valued one.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {
            "plan_mode": {
                "plan_mode_active": True,
                "plan_file_path": None,
                "workflow_prompt_sent": False,
            }
        }

    def test_default_values(self):
        state = PlanModeState()
        assert state.plan_mode_active is False
        assert state.plan_file_path is None
        assert state.workflow_prompt_sent is False

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Truthy strings and ints are NOT booleans — strict typing
            # falls back to the safe default so corrupted/legacy JSON
            # (e.g. ``"false"`` as a string) cannot mis-restore state.
            ({"plan_mode_active": "yes", "plan_file_path": None, "workflow_prompt_sent": 0}, (False, None, False)),
            ({"plan_mode_active": 0, "workflow_prompt_sent": 1}, (False, None, False)),
            # ``plan_file_path`` must be a string; anything else → None.
            ({"plan_mode_active": True, "plan_file_path": 42, "workflow_prompt_sent": True}, (True, None, True)),
            # Actual booleans are preserved.
            (
                {"plan_mode_active": True, "plan_file_path": "p.md", "workflow_prompt_sent": False},
                (True, "p.md", False),
            ),
            ({}, (False, None, False)),
        ],
    )
    def test_load_rejects_non_bool_and_non_str(self, tmp_path, raw, expected):
        path = tmp_path / "coerce.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = PlanModeState.load(path)
        assert (loaded.plan_mode_active, loaded.plan_file_path, loaded.workflow_prompt_sent) == expected


class TestContextStateRoundTrip:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "state" / "ctx.json"
        ContextState(last_call_input_tokens=52_499, context_length=1_000_000).save(path)
        assert path.exists()

        loaded = ContextState.load(path)
        assert loaded.last_call_input_tokens == 52_499
        assert loaded.context_length == 1_000_000

    def test_on_disk_layout_is_nested_under_context_state(self, tmp_path):
        path = tmp_path / "ctx.json"
        ContextState(last_call_input_tokens=10, context_length=200).save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"context_state": {"last_call_input_tokens": 10, "context_length": 200}}

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = ContextState.load(tmp_path / "absent.json")
        assert loaded.last_call_input_tokens == 0
        assert loaded.context_length == 0

    def test_load_corrupted_json_falls_back_to_default(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert ContextState.load(path) == ContextState()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Non-int (str / float / bool) coerces to the safe 0 default.
            ({"last_call_input_tokens": "500", "context_length": 1000}, (0, 1000)),
            ({"last_call_input_tokens": 12.5, "context_length": 1000}, (0, 1000)),
            ({"last_call_input_tokens": True, "context_length": 1000}, (0, 1000)),
            # Negative values are clamped to 0.
            ({"last_call_input_tokens": -5, "context_length": -1}, (0, 0)),
            # Valid ints preserved.
            ({"last_call_input_tokens": 800, "context_length": 128_000}, (800, 128_000)),
            ({}, (0, 0)),
        ],
    )
    def test_load_coerces_invalid_fields(self, tmp_path, raw, expected):
        path = tmp_path / "coerce.json"
        path.write_text(json.dumps({"context_state": raw}), encoding="utf-8")
        loaded = ContextState.load(path)
        assert (loaded.last_call_input_tokens, loaded.context_length) == expected


class TestSaveSectionErrors:
    def test_save_swallows_oserror(self, tmp_path):
        """A write failure (e.g. parent path is a file, not a dir) must be
        logged and swallowed — persistence never crashes the caller."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        # ``state`` would have to live *under* a regular file → mkdir raises
        # NotADirectoryError (an OSError), which save() must absorb.
        path = blocker / "state" / "s.json"
        ContextState(last_call_input_tokens=1, context_length=2).save(path)
        assert not path.exists()


class TestSectionsCoexist:
    """``plan_mode`` and ``context_state`` live in the same file and are
    written by different subsystems at different times — neither save may
    clobber the other's section."""

    def test_context_save_preserves_plan_mode(self, tmp_path):
        path = tmp_path / "state" / "s.json"
        PlanModeState(plan_mode_active=True, plan_file_path="p.md", workflow_prompt_sent=True).save(path)
        ContextState(last_call_input_tokens=42, context_length=1000).save(path)

        plan = PlanModeState.load(path)
        ctx = ContextState.load(path)
        assert plan.plan_mode_active is True
        assert plan.plan_file_path == "p.md"
        assert plan.workflow_prompt_sent is True
        assert ctx.last_call_input_tokens == 42
        assert ctx.context_length == 1000

    def test_plan_mode_save_preserves_context_state(self, tmp_path):
        path = tmp_path / "state" / "s.json"
        ContextState(last_call_input_tokens=42, context_length=1000).save(path)
        PlanModeState(plan_mode_active=True).save(path)

        ctx = ContextState.load(path)
        plan = PlanModeState.load(path)
        assert ctx.last_call_input_tokens == 42  # survived the plan-mode write
        assert ctx.context_length == 1000
        assert plan.plan_mode_active is True

    def test_both_sections_present_in_file(self, tmp_path):
        path = tmp_path / "s.json"
        PlanModeState(plan_mode_active=True).save(path)
        ContextState(last_call_input_tokens=7, context_length=99).save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data.keys()) == {"plan_mode", "context_state"}


class TestContextStateClear:
    """``ContextState.clear`` removes only the context-state mirror on session
    reset, so a status bar reading after ``/clear`` no longer sees the previous
    turn's occupancy."""

    def test_clear_removes_context_state_section(self, tmp_path):
        path = tmp_path / "state" / "s.json"
        ContextState(last_call_input_tokens=42, context_length=1000).save(path)
        ContextState.clear(path)
        # The section is gone — a subsequent load returns defaults.
        loaded = ContextState.load(path)
        assert (loaded.last_call_input_tokens, loaded.context_length) == (0, 0)
        assert "context_state" not in json.loads(path.read_text(encoding="utf-8"))

    def test_clear_preserves_sibling_plan_mode_section(self, tmp_path):
        path = tmp_path / "state" / "s.json"
        PlanModeState(plan_mode_active=True, plan_file_path="p.md").save(path)
        ContextState(last_call_input_tokens=7, context_length=99).save(path)
        ContextState.clear(path)
        plan = PlanModeState.load(path)
        assert plan.plan_mode_active is True
        assert plan.plan_file_path == "p.md"

    def test_clear_is_noop_when_file_absent(self, tmp_path):
        # Must not raise and must not create the file.
        path = tmp_path / "never.json"
        ContextState.clear(path)
        assert not path.exists()

    def test_clear_is_noop_when_section_absent(self, tmp_path):
        path = tmp_path / "state" / "s.json"
        PlanModeState(plan_mode_active=True).save(path)
        ContextState.clear(path)  # context_state never written
        assert PlanModeState.load(path).plan_mode_active is True


class TestLegacyCompactSectionIgnored:
    """Legacy state files written by older code carried a ``compact`` section
    alongside ``plan_mode``. The compact subsystem no longer persists state —
    idempotency is guaranteed by the in-message ``[DATUS_ARCHIVED]`` marker —
    so the loader must ignore the legacy key without crashing.
    """

    def test_loader_ignores_compact_section(self, tmp_path):
        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "plan_mode": {
                        "plan_mode_active": True,
                        "plan_file_path": "p.md",
                        "workflow_prompt_sent": False,
                    },
                    "compact": {"compacted_until": 14},
                }
            ),
            encoding="utf-8",
        )
        loaded = PlanModeState.load(path)
        # Plan-mode section restored verbatim; compact section silently dropped.
        assert loaded.plan_mode_active is True
        assert loaded.plan_file_path == "p.md"
        assert loaded.workflow_prompt_sent is False

    def test_save_does_not_emit_compact_key(self, tmp_path):
        """Even if a legacy file with a ``compact`` section is loaded and
        re-saved, the new write must NOT round-trip the dropped section —
        otherwise stale state would linger on disk forever.
        """
        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "plan_mode": {"plan_mode_active": False, "plan_file_path": None, "workflow_prompt_sent": False},
                    "compact": {"compacted_until": 7},
                }
            ),
            encoding="utf-8",
        )
        loaded = PlanModeState.load(path)
        loaded.plan_mode_active = True
        loaded.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "compact" not in data
        assert data == {"plan_mode": {"plan_mode_active": True, "plan_file_path": None, "workflow_prompt_sent": False}}
