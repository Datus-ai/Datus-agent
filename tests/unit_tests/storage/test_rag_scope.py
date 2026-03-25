# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""Unit tests for datus.storage.rag_scope."""

from unittest.mock import MagicMock

from datus_storage_base.conditions import build_where

from datus.storage.rag_scope import build_rag_scope
from datus.storage.registry import configure_storage_defaults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_agent_config(sub_agent_configs=None, db_type="", request_context=None):
    """Create a mock AgentConfig with optional sub-agent configs."""
    config = MagicMock()
    config.db_type = db_type
    config.request_context = request_context
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


class TestBuildRagScope:
    """Tests for build_rag_scope."""

    def setup_method(self):
        configure_storage_defaults()  # reset

    def teardown_method(self):
        configure_storage_defaults()  # reset

    def test_no_sub_agent_returns_none(self):
        """No sub_agent_name → no scoping."""
        config = _mock_agent_config()
        result = build_rag_scope(config, None, _mock_storage(), "tables")
        assert result is None

    def test_no_config_for_sub_agent_returns_none(self):
        """Sub-agent name with no matching config → no scoping."""
        config = _mock_agent_config()
        result = build_rag_scope(config, "unknown_agent", _mock_storage(), "tables")
        assert result is None

    def test_no_scoped_context_returns_none(self):
        """Sub-agent config without scoped_context → no scoping."""
        config = _mock_agent_config(sub_agent_configs={"team_a": {"system_prompt": "team_a"}})
        result = build_rag_scope(config, "team_a", _mock_storage(), "tables")
        assert result is None

    def test_table_scope_builds_filter(self):
        """Sub-agent with tables scoped context → table filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": "public.users"}}},
        )
        storage = _mock_storage()
        result = build_rag_scope(config, "team_a", storage, "tables")
        assert result is not None
        clause = build_where(result)
        assert "users" in clause

    def test_subject_scope_without_tree_raises(self):
        """Subject-based scope without subject_tree on storage → raises DatusException."""
        import pytest

        from datus.utils.exceptions import DatusException

        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=False)
        with pytest.raises(DatusException, match="subject_tree"):
            build_rag_scope(config, "team_a", storage, "metrics")

    def test_subject_scope_with_tree_builds_filter(self):
        """Subject-based scope with subject_tree → builds subject filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"metrics": "Finance.Revenue"}}},
        )
        storage = _mock_storage(has_subject_tree=True)
        storage.subject_tree.get_matched_children_id.return_value = [1, 2]
        result = build_rag_scope(config, "team_a", storage, "metrics")
        assert result is not None

    def test_empty_scope_value_returns_none(self):
        """Empty scope value → no filter."""
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": ""}}},
        )
        result = build_rag_scope(config, "team_a", _mock_storage(), "tables")
        assert result is None

    # -- request_context scope_fields filtering --

    def test_scope_fields_filter(self):
        """request_context with scope_fields → WHERE filter on scope_fields only."""
        configure_storage_defaults(scope_fields=["workspace_id"])
        config = _mock_agent_config(request_context={"workspace_id": "ws_abc", "creator_id": "u1"})
        result = build_rag_scope(config, None, _mock_storage(), "tables")
        assert result is not None
        clause = build_where(result)
        assert "workspace_id" in clause
        assert "ws_abc" in clause
        assert "creator_id" not in clause

    def test_scope_fields_combined_with_sub_agent(self):
        """Both scope_fields and sub-agent scope → AND combination."""
        configure_storage_defaults(scope_fields=["workspace_id"])
        config = _mock_agent_config(
            sub_agent_configs={"team_a": {"scoped_context": {"tables": "orders"}}},
            request_context={"workspace_id": "ws_123"},
        )
        result = build_rag_scope(config, "team_a", _mock_storage(), "tables")
        assert result is not None
        clause = build_where(result)
        assert "workspace_id" in clause
        assert "orders" in clause

    def test_no_scope_fields_no_sub_agent_returns_none(self):
        """No scope_fields configured, no sub-agent → None."""
        config = _mock_agent_config(request_context={"workspace_id": "ws_abc"})
        result = build_rag_scope(config, None, _mock_storage(), "tables")
        assert result is None

    def test_no_request_context_returns_none(self):
        """scope_fields configured but no request_context → None."""
        configure_storage_defaults(scope_fields=["workspace_id"])
        config = _mock_agent_config(request_context=None)
        result = build_rag_scope(config, None, _mock_storage(), "tables")
        assert result is None

    def test_multiple_scope_fields(self):
        """Multiple scope_fields → AND of all matching conditions."""
        configure_storage_defaults(scope_fields=["workspace_id", "tenant_id"])
        config = _mock_agent_config(request_context={"workspace_id": "ws_1", "tenant_id": "t_1"})
        result = build_rag_scope(config, None, _mock_storage(), "tables")
        assert result is not None
        clause = build_where(result)
        assert "workspace_id" in clause
        assert "tenant_id" in clause
