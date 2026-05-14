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
        # On-disk JSON contains the expected schema
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["plan_mode_active"] is True
        assert data["plan_file_path"] is None
        assert data["workflow_prompt_sent"] is False

    def test_default_values(self):
        state = PlanModeState()
        assert state.plan_mode_active is False
        assert state.plan_file_path is None
        assert state.workflow_prompt_sent is False

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ({"plan_mode_active": "yes", "plan_file_path": None, "workflow_prompt_sent": 0}, (True, None, False)),
            ({"plan_mode_active": 0, "workflow_prompt_sent": 1}, (False, None, True)),
            ({}, (False, None, False)),
        ],
    )
    def test_load_coerces_types(self, tmp_path, raw, expected):
        path = tmp_path / "coerce.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        loaded = PlanModeState.load(path)
        assert (loaded.plan_mode_active, loaded.plan_file_path, loaded.workflow_prompt_sent) == expected
