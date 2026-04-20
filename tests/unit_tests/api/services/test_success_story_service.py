# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.

"""Tests for datus.api.services.success_story_service."""

import csv
from unittest.mock import MagicMock

import pytest

from datus.api.models.success_story_models import SuccessStoryInput
from datus.api.services.success_story_service import (
    SubagentNotFoundError,
    SuccessStoryService,
)


def _make_config(tmp_path, agentic_nodes=None):
    cfg = MagicMock()
    cfg.agentic_nodes = agentic_nodes or {}
    cfg.path_manager = MagicMock()
    cfg.path_manager.benchmark_dir = tmp_path / "benchmark"
    return cfg


class TestSaveSuccessStory:
    def test_writes_new_csv_with_header(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        payload = SuccessStoryInput(
            session_id="sess-1",
            sql="SELECT 1",
            user_message="show one",
            subagent_id="gen_sql",
            session_link="http://x/session=sess-1",
        )
        data = svc.save(payload)

        csv_path = tmp_path / "benchmark" / "gen_sql" / "success_story.csv"
        assert data.csv_path == str(csv_path)
        assert data.subagent_name == "gen_sql"
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["session_id"] == "sess-1"
        assert rows[0]["sql"] == "SELECT 1"
        assert rows[0]["subagent_name"] == "gen_sql"
        assert rows[0]["session_link"] == "http://x/session=sess-1"

    def test_appends_without_duplicate_header(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        base = SuccessStoryInput(session_id="s1", sql="SELECT 1", user_message="q1", subagent_id="gen_sql")
        svc.save(base)
        svc.save(base.model_copy(update={"session_id": "s2", "sql": "SELECT 2"}))

        csv_path = tmp_path / "benchmark" / "gen_sql" / "success_story.csv"
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        # Header + 2 rows, no second header
        assert len(lines) == 3
        assert lines[0].startswith("session_link,")

    def test_defaults_subagent_id_to_default(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        data = svc.save(SuccessStoryInput(session_id="s", sql="SELECT 1", user_message="q"))
        assert data.subagent_name == "default"
        assert (tmp_path / "benchmark" / "default" / "success_story.csv").exists()

    def test_resolves_builtin_subagent(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        data = svc.save(SuccessStoryInput(session_id="s", sql="SELECT 1", user_message="q", subagent_id="gen_report"))
        assert data.subagent_name == "gen_report"

    def test_resolves_custom_node_by_key(self, tmp_path):
        nodes = {"my-agent": {"id": "uuid-xyz", "system_prompt": "x"}}
        svc = SuccessStoryService(_make_config(tmp_path, agentic_nodes=nodes))
        data = svc.save(SuccessStoryInput(session_id="s", sql="SELECT 1", user_message="q", subagent_id="my-agent"))
        assert data.subagent_name == "my-agent"

    def test_resolves_custom_node_by_uuid(self, tmp_path):
        nodes = {"my-agent": {"id": "uuid-xyz", "system_prompt": "x"}}
        svc = SuccessStoryService(_make_config(tmp_path, agentic_nodes=nodes))
        data = svc.save(SuccessStoryInput(session_id="s", sql="SELECT 1", user_message="q", subagent_id="uuid-xyz"))
        # UUID resolves back to the canonical human-readable key
        assert data.subagent_name == "my-agent"
        assert (tmp_path / "benchmark" / "my-agent" / "success_story.csv").exists()

    def test_unknown_subagent_id_raises(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        with pytest.raises(SubagentNotFoundError):
            svc.save(
                SuccessStoryInput(
                    session_id="s",
                    sql="SELECT 1",
                    user_message="q",
                    subagent_id="nope",
                )
            )

    def test_sanitizes_csv_injection_in_user_message(self, tmp_path):
        svc = SuccessStoryService(_make_config(tmp_path))
        svc.save(
            SuccessStoryInput(
                session_id="s",
                sql="=cmd|'/c calc'!A1",
                user_message="+HYPERLINK(...)",
                subagent_id="gen_sql",
            )
        )
        csv_path = tmp_path / "benchmark" / "gen_sql" / "success_story.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["sql"].startswith("'=")
        assert rows[0]["user_message"].startswith("'+")
