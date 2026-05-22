# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for datus/storage/session_state.py — CI tier."""

import json

import pytest

from datus.storage.session_state import PlanModeState


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
