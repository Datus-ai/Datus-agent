# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.rag_scope."""

from unittest.mock import MagicMock

from datus_storage_base.conditions import build_where

from datus.storage.rag_scope import _build_sub_agent_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent_config(sub_agent_configs=None, db_type=""):
    """Create a mock AgentConfig with optional sub-agent configs."""
    config = MagicMock()
    config.db_type = db_type
    config.sub_agent_config = MagicMock(side_effect=lambda name: (sub_agent_configs or {}).get(name, {}))
    return config


def _mock_storage(has_subject_tree=False):
    """Create a mock storage, optionally with a subject_tree."""
    storage = MagicMock()
    if has_subject_tree:
        storage.subject_tree = MagicMock()
    else:
        storage.subject_tree = None
    return storage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSubAgentFilter:
    """Tests for _build_sub_agent_filter."""

    def test_no_sub_agent_returns_none(self):
        """No sub_agent_name -> no scoping."""
        config = _mock_agent_config()
        result = _build_sub_agent_filter(config, None, _mock_storage(), "tables")
        assert result is None

    def test_no_config_for_sub_agent_returns_none(self):
        """Sub-agent name with no matching config -> no scoping."""
        config = _mock_agent_config()
        result = _build_sub_agent_filter(config, "unknown_agent", _mock_storage(), "tables")
        assert result is None

    def test_no_scoped_context_returns_none(self):
        """Sub-agent config without scoped_context -> no scoping."""
        config = _mock_agent_config(sub_agent_configs={"team_a": {"system_prompt": "team_a"}})
        result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
        assert result is None

    def test_table_scope_builds_filter(self):
        """Sub-agent with tables scoped context -> table filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": "public.users"}}},
        )
        storage = _mock_storage()
        result = _build_sub_agent_filter(config, "team_a", storage, "tables")
        assert result is not None
        clause = build_where(result)
        assert "users" in clause

    def test_subject_scope_without_tree_raises(self):
        """Subject-based scope without subject_tree on storage -> raises DatusException."""
        import pytest

        from datus.utils.exceptions import DatusException

        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=False)
        with pytest.raises(DatusException, match="subject_tree"):
            _build_sub_agent_filter(config, "team_a", storage, "metrics")

    def test_subject_scope_with_tree_builds_filter(self):
        """Subject-based scope with subject_tree -> builds subject filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=True)
        storage.subject_tree.get_matched_children_id.return_value = [1, 2]
        result = _build_sub_agent_filter(config, "team_a", storage, "metrics")
        assert result is not None

    def test_empty_scope_value_returns_none(self):
        """Empty scope value -> no filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": ""}}},
        )
        result = _build_sub_agent_filter(config, "team_a", _mock_storage(), "tables")
        assert result is None
